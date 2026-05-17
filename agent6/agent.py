"""
Agent6 — Memory · Perception · Decision · Action cognitive loop.

Usage (CLI)
-----------
    python -m agent6 "Your query here"
    python -m agent6 --run-id myrun "Your query"

Usage (Python)
--------------
    import asyncio
    from agent6.agent import run
    result = asyncio.run(run("When is mom's birthday?"))
    print(result)

Loop flow (one iteration)
--------------------------
    memory.read        → hits
    perception.observe → updated Observation (goal list with done flags)
    [all goals done?]  → exit
    attach artifact    → load bytes if current_goal.attach_artifact_id is set
    decision.next_step → DecisionOutput (answer | tool_call)
    action.execute     → (descriptor, artifact_id | None)
    memory.record_outcome, append to history
    repeat

State that survives across iterations
--------------------------------------
    history        list[dict]   — every tool result and assistant answer
    prior_goals    list[Goal]   — the stable goal list (ids assigned by loop)
    memory.json                 — persists across runs (facts, preferences,
                                  tool outcomes including artifact handles)
    state/artifacts/            — artifact bytes (cleared only on explicit reset)
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import uuid
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from . import action as action_mod
from . import decision as decision_mod
from . import perception as perception_mod
from .artifact_store import store as artifact_store
from .memory import Memory
from .schemas import Goal

MAX_ITERATIONS = 25

# Path to the MCP server (sibling of this package's parent directory)
MCP_SERVER = Path(__file__).parent.parent / "mcp_server.py"

# Prefer the project .venv Python so the MCP subprocess inherits all installed
# packages (crawl4ai, ddgs, mcp, etc.).  When running via `uv run`, sys.executable
# points to the system Python which lacks those packages.
def _find_venv_python() -> str:
    base = Path(__file__).parent.parent
    for candidate in (
        base / ".venv" / "Scripts" / "python.exe",   # Windows
        base / ".venv" / "bin" / "python",            # Unix
        base / ".venv" / "bin" / "python3",
    ):
        if candidate.exists():
            return str(candidate)
    return sys.executable   # fallback: hope it has the packages

MCP_PYTHON = _find_venv_python()


# ── MCP helpers ────────────────────────────────────────────────────────────────

async def _get_mcp_tools(session: ClientSession) -> list[dict]:
    """
    Fetch the MCP tool list and return it in the gateway's ToolDef format:
        [{"name": str, "description": str, "input_schema": dict}, ...]

    This is what providers.py expects — NOT the OpenAI {"type":"function",...}
    wrapper.
    """
    result = await session.list_tools()
    tools: list[dict] = []
    for t in result.tools:
        tools.append({
            "name":         t.name,
            "description":  t.description or "",
            "input_schema": t.inputSchema or {"type": "object", "properties": {}},
        })
    return tools


# ── durable-memory detection ───────────────────────────────────────────────────

_FACT_RE = re.compile(
    r"\b(remember|birthday|anniversary|remind)\b"
    r"|\bmy\s+\w+\s+(is|are)\b"
    r"|\bis\s+(on\s+)?\d{1,2}\s+\w+\s+\d{4}\b",
    re.IGNORECASE,
)
_PREF_RE = re.compile(
    r"\bi\s+(prefer|like|love|always|never|hate|dislike)\b"
    r"|\bmy\s+preference\b",
    re.IGNORECASE,
)


def _is_memorable(query: str) -> bool:
    return bool(_FACT_RE.search(query) or _PREF_RE.search(query))


def _memory_kind(query: str) -> str:
    return "preference" if _PREF_RE.search(query) else "fact"


# ── logging helpers ────────────────────────────────────────────────────────────

def _args_preview(arguments: dict, max_len: int = 120) -> str:
    s = json.dumps(arguments, ensure_ascii=False)
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


# ── main loop ──────────────────────────────────────────────────────────────────

async def run(query: str, run_id: str | None = None) -> str:
    """
    Execute the agent loop for the given query.

    Returns a string starting with "FINAL: " containing the agent's answer.
    The loop is bounded by MAX_ITERATIONS as a safety ceiling.
    """
    run_id = run_id or str(uuid.uuid4())[:8]
    mem    = Memory()

    # ── store memorable facts / preferences from the query upfront ─────────────
    if _is_memorable(query):
        kind = _memory_kind(query)
        item = mem.remember(query, run_id=run_id, kind=kind)
        print(f"[memory.remember]  stored as {kind}: {item.descriptor[:80]}")
        print(f"                   keywords: {item.keywords[:8]}")

    history:     list[dict] = []
    prior_goals: list[Goal] = []
    final_answer: str | None = None

    # ── pre-create common sandbox subdirectories ───────────────────────────────
    # mcp_server.py's create_file requires the parent directory to already exist.
    # We seed a few common ones so the LLM can write without a mkdir round-trip.
    sandbox = MCP_SERVER.parent / "sandbox"
    for subdir in ("reminders", "notes", "research"):
        (sandbox / subdir).mkdir(parents=True, exist_ok=True)

    # ── MCP session ────────────────────────────────────────────────────────────
    params = StdioServerParameters(
        command=MCP_PYTHON,
        args=[str(MCP_SERVER)],
    )

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            mcp_tools = await _get_mcp_tools(session)

            for iteration in range(MAX_ITERATIONS):
                print(f"\n─── iter {iteration + 1} ───")

                # ── 1. memory.read ─────────────────────────────────────────────
                hits = mem.read(query, history, top_k=8)
                print(f"[memory.read]   {len(hits)} hits")
                for h in hits:
                    art_tag = f"  [artifact: {h.artifact_id}]" if h.artifact_id else ""
                    print(f"                {h.kind}: {h.descriptor[:80]}{art_tag}")

                # ── 2. perception ──────────────────────────────────────────────
                try:
                    obs = perception_mod.observe(
                        query, hits, history, prior_goals, run_id
                    )
                except Exception as exc:
                    print(f"[perception]    ERROR: {exc!s:.200}")
                    history.append({"role": "error", "content": f"perception: {exc}"})
                    continue

                # Map returned goals to stable ids by position
                new_goals: list[Goal] = []
                for i, g in enumerate(obs.goals):
                    goal_id = prior_goals[i].id if i < len(prior_goals) else f"g{i + 1}_{run_id}"
                    new_goals.append(Goal(
                        id=goal_id,
                        text=g.text,
                        done=g.done,
                        attach_artifact_id=g.attach_artifact_id,
                    ))
                prior_goals = new_goals

                for g in prior_goals:
                    status = "done" if g.done else "open"
                    print(f"[perception]    [{status}] {g.text}")
                    if g.attach_artifact_id:
                        print(f"                  attach={g.attach_artifact_id}")

                # ── check completion ───────────────────────────────────────────
                if prior_goals and all(g.done for g in prior_goals):
                    print(f"\n[done] all {len(prior_goals)} goals satisfied")
                    break

                # ── find first unfinished goal ─────────────────────────────────
                current_goal = next((g for g in prior_goals if not g.done), None)
                if current_goal is None:
                    break

                # ── 3. attach artifact if needed ───────────────────────────────
                attached: list[tuple[str, bytes]] = []
                if current_goal.attach_artifact_id:
                    try:
                        art_bytes = artifact_store.get(current_goal.attach_artifact_id)
                        attached  = [(current_goal.attach_artifact_id, art_bytes)]
                        print(
                            f"[attach]        {current_goal.attach_artifact_id}"
                            f" ({len(art_bytes):,} bytes)"
                        )
                    except KeyError as exc:
                        print(f"[attach]        WARNING: {exc}")

                # ── 4. decision ────────────────────────────────────────────────
                try:
                    dec = decision_mod.next_step(
                        current_goal, hits, attached, history, mcp_tools
                    )
                except Exception as exc:
                    print(f"[decision]      ERROR: {exc!s:.200}")
                    history.append({
                        "role":    "error",
                        "content": f"decision failed: {exc}",
                        "goal_id": current_goal.id,
                    })
                    continue

                if dec.answer:
                    ans_preview = dec.answer[:300].replace("\n", " ")
                    print(f"[decision]      ANSWER: {ans_preview}")
                    final_answer = dec.answer
                    history.append({
                        "role":    "assistant",
                        "content": dec.answer,
                        "goal_id": current_goal.id,
                    })

                elif dec.tool_call:
                    tc = dec.tool_call
                    print(f"[decision]      TOOL_CALL: {tc.name}({_args_preview(tc.arguments)})")

                    # ── 5. action ──────────────────────────────────────────────
                    try:
                        descriptor, artifact_id = await action_mod.execute(session, tc)
                    except Exception as exc:
                        descriptor  = f"ERROR: {exc}"
                        artifact_id = None
                    print(f"[action]        → {descriptor[:200]}")

                    # Record outcome in memory (makes artifact visible to future
                    # Perception iterations via memory.read hits)
                    mem.record_outcome(
                        tc,
                        descriptor,
                        artifact_id,
                        run_id=run_id,
                        goal_id=current_goal.id,
                    )

                    history.append({
                        "role":        "tool",
                        "name":        tc.name,
                        "content":     descriptor,
                        "artifact_id": artifact_id,
                        "goal_id":     current_goal.id,
                    })

                else:
                    # Decision returned neither answer nor tool call — safety exit
                    print("[decision]      WARNING: empty output — stopping loop")
                    break

    # ── build final return value ───────────────────────────────────────────────
    if final_answer:
        return f"FINAL: {final_answer}"

    # No explicit ANSWER emitted — surface the last assistant turn if any
    for h in reversed(history):
        if h.get("role") == "assistant" and h.get("content"):
            return f"FINAL: {h['content']}"

    return f"FINAL: Task complete ({len(history)} history steps, no explicit answer)."


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Agent6 — Memory·Perception·Decision·Action cognitive loop"
    )
    parser.add_argument("query",    nargs="?",  help="Query to process")
    parser.add_argument("--run-id", default=None, help="Run ID (default: random)")
    args = parser.parse_args()

    if not args.query:
        parser.print_help()
        sys.exit(1)

    result = asyncio.run(run(args.query, args.run_id))
    print(f"\n{result}")


if __name__ == "__main__":
    main()
