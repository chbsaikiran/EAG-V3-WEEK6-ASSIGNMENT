"""
Decision module — one LLM call, two possible outputs: answer or tool call.

Routes through the gateway with auto_route="decision" so the router pool
classifies the call and picks the optimal worker tier:
  • TINY  (< 1 000 tokens) → fast small model  (gpt-4.1-mini, Phi-4, etc.)
  • LARGE (1 000–8 000 t)  → Gemini 2.5 Flash or equivalent

The tools list uses the gateway's ToolDef format:
    {"name": str, "description": str, "input_schema": dict}

which is what providers.py expects — NOT the OpenAI {"type":"function",...} wrapper.

System-prompt obligations
-------------------------
1. Exactly ONE output: call a tool OR give a text answer (never both).
2. Artifact handles (art:...) are internal — never pass them to tools.
   Artifact content appears in ATTACHED ARTIFACTS when the goal needs it.
3. Substantive answers: ≥ 3 full sentences or a numbered list of ≥ 3 items.
"""
from __future__ import annotations

import json

import httpx

from .schemas import AgentToolCall, DecisionOutput, Goal, MemoryItem

GATEWAY_URL = "http://localhost:8101/v1/chat"

# ── system prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are the Decision module of an AI agent. Given a GOAL and supporting context,
take exactly ONE of two possible actions:

  ① CALL A TOOL — pick the single best tool and call it with correct arguments.
  ② GIVE AN ANSWER — write a direct, substantive plain-text answer.

═══ MANDATORY RULES ══════════════════════════════════════════════════════════════

1. EXACTLY ONE OUTPUT
   Either call one tool OR write an answer. Never do both. Never say "I will …"
   without acting. Never ask the user for clarification.

2. ARTIFACT HANDLES
   Strings that start with "art:" are internal artifact handles. They are NOT
   file paths, URLs, or content. NEVER pass them as arguments to any tool.
   When a goal requires the raw content of an artifact, that content already
   appears verbatim under "ATTACHED ARTIFACTS:" below — read it there directly.

3. SUBSTANTIVE ANSWERS
   When the goal asks for extraction, a list, a comparison, a recommendation,
   or any analysis, your answer MUST be substantive:
     • At least 3 full sentences, OR
     • A numbered list of at least 3 items.
   Do NOT give meta-answers such as "The page was fetched successfully."
   or "I found the information, here it is." — give the actual content.

4. TOOL ARGUMENTS
   Use the EXACT argument names from each tool's schema. Match expected types
   (string, integer, boolean). Do not invent argument names.

5. ONE TOOL PER TURN
   If multiple tools would help, pick the single most important one for this
   GOAL right now. The agent loop will call you again after each result.

6. FILE PATHS IN SANDBOX
   The MCP sandbox root already exists. Files must be created with paths that
   either sit at the sandbox root or inside an existing subdirectory.
   Subdirectories are NOT auto-created. If you are unsure what exists, call
   list_dir(".") first.
"""


# ── helpers ────────────────────────────────────────────────────────────────────

def _format_hits(hits: list[MemoryItem]) -> str:
    if not hits:
        return "  (none)"
    lines = []
    for h in hits:
        line = f"  [{h.kind}] {h.descriptor}"
        if h.artifact_id:
            line += f"  (artifact: {h.artifact_id})"
        lines.append(line)
    return "\n".join(lines)


def _format_history(history: list[dict]) -> str:
    if not history:
        return "  (empty)"
    lines = []
    for h in history[-10:]:
        role    = h.get("role", "?")
        name    = h.get("name", "")
        content = str(h.get("content", ""))[:600]
        label   = f"{role}:{name}" if name else role
        lines.append(f"  [{label}] {content}")
    return "\n".join(lines)


# ── main entry point ───────────────────────────────────────────────────────────

def next_step(
    goal: Goal,
    hits: list[MemoryItem],
    attached: list[tuple[str, bytes]],
    history: list[dict],
    mcp_tools: list[dict],
) -> DecisionOutput:
    """
    One Decision call. Returns either an answer (text) or a single tool call.

    Parameters
    ----------
    goal        : the current open goal
    hits        : memory hits from this iteration's memory.read()
    attached    : [(art_id, raw_bytes)] — artifact content to embed in prompt
    history     : full run history accumulated so far
    mcp_tools   : list of ToolDef dicts fetched from the MCP session
    """
    # Build artifact section
    artifact_section = ""
    if attached:
        parts = []
        for art_id, art_bytes in attached:
            try:
                text = art_bytes.decode("utf-8", errors="replace")
            except Exception:
                text = "<binary — cannot display>"
            parts.append(
                f"=== {art_id} ({len(art_bytes):,} bytes) ===\n{text}"
            )
        artifact_section = "\n\nATTACHED ARTIFACTS:\n" + "\n\n".join(parts)

    prompt = f"""GOAL: {goal.text}

MEMORY HITS:
{_format_hits(hits)}

RECENT HISTORY:
{_format_history(history)}{artifact_section}

Take the next action: call a tool or give a final answer for this GOAL."""

    base_body = {
        "prompt":      prompt,
        "system":      SYSTEM_PROMPT,
        "tools":       mcp_tools,
        "tool_choice": "auto",
        "temperature": 0.7,
        "max_tokens":  4096,
    }

    # Attempt 1: auto_route lets the router pick TINY vs LARGE.
    # When there are attached artifacts the prompt can be large; the router may
    # classify it HUGE (>8000 tokens) and return 503, or all LARGE-tier workers
    # may be busy. In both cases we fall back to Gemini directly — it has a
    # 1 M-token context and bypasses the HUGE rejection when called explicitly.
    resp = httpx.post(
        GATEWAY_URL,
        json={**base_body, "auto_route": "decision"},
        timeout=180.0,
    )

    if resp.status_code in (503, 502):
        # Fallback: pin to Gemini (no router, no HUGE check)
        print(f"[decision]      router/503 → fallback to gemini direct")
        resp = httpx.post(
            GATEWAY_URL,
            json={**base_body, "provider": "gemini"},
            timeout=180.0,
        )

    resp.raise_for_status()
    data = resp.json()

    tool_calls: list[dict] = data.get("tool_calls") or []
    if tool_calls:
        tc   = tool_calls[0]
        name = tc.get("name", "")
        args = tc.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        return DecisionOutput(tool_call=AgentToolCall(name=name, arguments=args))

    answer = (data.get("text") or "").strip()
    return DecisionOutput(answer=answer)
