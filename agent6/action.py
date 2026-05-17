"""
Action module — pure MCP dispatch, no LLM call.

Behaviour
---------
1. art: guard  — refuse any tool argument that starts with "art:". These are
   internal artifact handles and are not valid file paths or URLs. Returns a
   clear error string that the history records and Perception can act on.

2. MCP dispatch — await session.call_tool(...), collapse content blocks to text.

3. Threshold check — if the payload exceeds ARTIFACT_THRESHOLD_BYTES (4 KB),
   persist it via ArtifactStore.put() and return a short descriptor of the form
       "[artifact art:<handle>, N bytes] preview: ..."
   together with the art: handle as the second element of the tuple.
   Payloads under the threshold are returned verbatim with None as the handle.
"""
from __future__ import annotations

from mcp import ClientSession

from .artifact_store import ARTIFACT_THRESHOLD_BYTES, store as artifact_store
from .schemas import AgentToolCall


async def execute(
    session: ClientSession,
    tool_call: AgentToolCall,
) -> tuple[str, str | None]:
    """
    Dispatch tool_call via MCP. Returns (descriptor, artifact_id | None).

    descriptor  — short string suitable for appending to history / printing
    artifact_id — art: handle if payload was stored, else None
    """
    # ── guard: reject art: handles passed as arguments ─────────────────────────
    for key, val in tool_call.arguments.items():
        if isinstance(val, str) and val.startswith("art:"):
            msg = (
                f"ERROR: argument '{key}' is an artifact handle ({val!r}). "
                "Artifact handles are internal references — they are not file "
                "paths or URLs. The artifact content is provided under "
                "'ATTACHED ARTIFACTS:' in the Decision prompt."
            )
            return msg, None

    # ── MCP dispatch ───────────────────────────────────────────────────────────
    result = await session.call_tool(
        tool_call.name,
        arguments=tool_call.arguments,
    )

    # Collapse all content blocks into a single string
    parts: list[str] = []
    for block in result.content:
        if hasattr(block, "text"):
            parts.append(block.text)
        elif hasattr(block, "data"):
            parts.append(str(block.data))
        else:
            parts.append(str(block))
    full_text = "\n".join(parts)

    # ── threshold check ────────────────────────────────────────────────────────
    payload = full_text.encode("utf-8")

    if len(payload) > ARTIFACT_THRESHOLD_BYTES:
        # Build a meaningful source label from the first argument value
        arg_vals = list(tool_call.arguments.values())
        source   = f"{tool_call.name}({str(arg_vals[0])[:80]})" if arg_vals else tool_call.name

        art_id = artifact_store.put(
            payload,
            source=source,
            content_type="text/plain",
            descriptor=f"{tool_call.name} result, {len(payload):,} bytes",
        )
        preview = full_text[:200].replace("\n", " ")
        descriptor = (
            f"[artifact {art_id}, {len(payload):,} bytes] preview: {preview}..."
        )
        return descriptor, art_id

    # Small payload — return as-is
    return full_text, None
