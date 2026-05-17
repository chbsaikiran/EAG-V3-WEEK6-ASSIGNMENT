"""
Perception module — the orchestrator that maintains goal state across iterations.

Pinned to Gemini (provider="gemini") because smaller TINY-tier models
hallucinate goal drops, stale artifact indices, and order inversions.

Contract (four obligations)
---------------------------
1. First call (prior_goals empty): decompose the user query into 1–5 goals.
2. Subsequent calls: preserve goal order; mark goals done based on history only.
3. First unfinished goal: set artifact_index (integer) if it needs raw artifact
   bytes. The integer refers to the "[ARTIFACT index=N]" shown in MEMORY HITS.
4. Never reorder, insert, or drop goals after the first call.

Anti-hallucination choices
--------------------------
- Goals have no id field in the LLM output. Identity is positional; the outer
  loop maps positions to the stable ids it already carries.
- Artifact references use an integer index into MEMORY HITS, not an art: string.
  The outer loop maps the integer back to the real art: handle.
"""
from __future__ import annotations

import json
import re

import httpx

from .schemas import Goal, MemoryItem, Observation

GATEWAY_URL = "http://localhost:8101/v1/chat"

# Gemini does not support JSON Schema array-type syntax ("type": ["integer","null"]).
# We use json_object mode instead — the system prompt enforces the structure,
# and perception.py parses + validates the output itself.

# ── system prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are the Perception module of a multi-step AI agent system.

═══ YOUR ROLE ═══════════════════════════════════════════════════════════════════

FIRST CALL (PRIOR GOALS is empty):
  Decompose the USER QUERY into 1–5 short, bounded, imperative goals.
  Each goal is one clear fetch, search, extraction, creation, or answer step.
  Goals should be ordered so each one builds on the previous.

SUBSEQUENT CALLS (PRIOR GOALS is populated):
  • Preserve ALL goals in their EXACT order. Do not add, remove, or reorder.
  • For each goal, examine RUN HISTORY. Mark done=true only when history
    contains concrete evidence the goal was accomplished (see criteria below).
  • Once done=true, keep it done forever — never revert a completed goal.
  • For the FIRST unfinished goal: if completing it requires reading the raw
    content of a previously fetched artifact, set artifact_index to the integer
    shown after "[ARTIFACT index=N]" in MEMORY HITS. Set null otherwise.
    NEVER invent an artifact_index not shown in MEMORY HITS.

═══ DONE CRITERIA ═══════════════════════════════════════════════════════════════

A goal is done ONLY when RUN HISTORY contains a matching completed step:
  • Fetch/search goal  → a [tool:fetch_url] or [tool:web_search] entry exists
                         WITHOUT an "ERROR:" prefix.
  • Extraction/answer  → an [assistant] entry contains a substantive response
                         (≥ 3 sentences or a numbered list) about that goal.
  • File creation      → a [tool:create_file] or [tool:update_file] entry shows
                         "ok: created …" or "ok: updated …".
  • Weather/data fetch → a [tool] entry shows the data was retrieved.

Do NOT mark done based on intent or plan — only completed history entries count.

═══ OUTPUT FORMAT ════════════════════════════════════════════════════════════════

Return ONLY this JSON object, nothing else:

{
  "goals": [
    {"text": "short imperative statement", "done": false, "artifact_index": null},
    {"text": "another goal",               "done": true,  "artifact_index": null},
    {"text": "goal needing artifact",      "done": false,  "artifact_index": 0}
  ]
}

Rules:
  • artifact_index MUST be an integer (from MEMORY HITS) or null — never a string.
  • Preserve goal count and order exactly on subsequent calls.
  • No extra keys, no markdown, no explanation — pure JSON only.
"""


# ── helpers ────────────────────────────────────────────────────────────────────

def _format_hits(hits: list[MemoryItem]) -> tuple[str, dict[int, str]]:
    """
    Format memory hits for the prompt. Assign integer indices to hits that
    carry artifact_id so Perception can reference them safely.

    Returns (formatted_text, {index: art_handle}).
    """
    lines: list[str] = []
    art_map: dict[int, str] = {}
    art_counter = 0

    for hit in hits:
        line = f"  [{hit.kind}] {hit.descriptor}"
        if hit.artifact_id:
            line += f"  [ARTIFACT index={art_counter}]"
            art_map[art_counter] = hit.artifact_id
            art_counter += 1
        lines.append(line)

    text = "\n".join(lines) if lines else "  (none)"
    return text, art_map


def _format_history(history: list[dict]) -> str:
    lines: list[str] = []
    for h in history[-14:]:
        role    = h.get("role", "?")
        name    = h.get("name", "")
        content = str(h.get("content", ""))[:500]
        label   = f"{role}:{name}" if name else role
        lines.append(f"  [{label}] {content}")
    return "\n".join(lines) if lines else "  (empty — this is the first iteration)"


def _format_prior_goals(prior_goals: list[Goal]) -> str:
    if not prior_goals:
        return "  (empty — decompose the user query into goals now)"
    lines: list[str] = []
    for i, g in enumerate(prior_goals):
        status = "done" if g.done else "open"
        line   = f"  [{status}] goal[{i}]: {g.text}"
        if g.attach_artifact_id:
            line += f"  (last attached: {g.attach_artifact_id})"
        lines.append(line)
    return "\n".join(lines)


def _parse_json(raw: str) -> dict:
    """Parse JSON from LLM output, stripping markdown fences if present."""
    text = raw.strip()
    # Strip ```json ... ``` fences
    text = re.sub(r"^```[a-z]*\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Last resort: find first {...} block
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            return json.loads(m.group())
        raise ValueError(f"Perception returned non-JSON: {raw[:300]!r}")


# ── main entry point ───────────────────────────────────────────────────────────

def observe(
    query: str,
    hits: list[MemoryItem],
    history: list[dict],
    prior_goals: list[Goal],
    run_id: str,
) -> Observation:
    """
    Run one Perception pass. Returns an Observation with an updated goal list.

    Goals in the returned Observation have id="" — the caller (agent.py) maps
    positions to the stable ids it carries in prior_goals.
    """
    hits_text, art_map = _format_hits(hits)
    hist_text           = _format_history(history)
    goals_text          = _format_prior_goals(prior_goals)

    prompt = f"""USER QUERY:
{query}

MEMORY HITS:
{hits_text}

RUN HISTORY:
{hist_text}

PRIOR GOALS:
{goals_text}

Emit the updated goals list as JSON now."""

    resp = httpx.post(
        GATEWAY_URL,
        json={
            "prompt":          prompt,
            "system":          SYSTEM_PROMPT,
            "provider":        "gemini",   # pinned — do not route
            "temperature":     0.1,
            "max_tokens":      1024,
            # json_object: ask Gemini to output JSON; we parse it ourselves.
            # json_schema is avoided because Gemini rejects array-type syntax
            # ("type": ["integer","null"]) in response_schema.
            "response_format": {"type": "json_object"},
        },
        timeout=90.0,
    )
    resp.raise_for_status()
    data = resp.json()

    parsed = _parse_json(data.get("text", "{}"))
    goals_raw: list[dict] = parsed.get("goals", [])

    # Map positions → stable goal ids; resolve artifact_index → art: handle
    goals: list[Goal] = []
    for i, g in enumerate(goals_raw):
        # Stable id: reuse from prior list by position; generate on first call
        if i < len(prior_goals):
            goal_id = prior_goals[i].id
        else:
            goal_id = f"g{i + 1}_{run_id}"

        # Resolve artifact_index safely
        art_id: str | None = None
        raw_idx = g.get("artifact_index")
        if raw_idx is not None:
            try:
                art_id = art_map.get(int(raw_idx))
            except (TypeError, ValueError):
                pass  # malformed index — ignore

        goals.append(Goal(
            id=goal_id,
            text=g.get("text", f"goal {i + 1}"),
            done=bool(g.get("done", False)),
            attach_artifact_id=art_id,
        ))

    return Observation(goals=goals)
