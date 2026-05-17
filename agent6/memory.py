"""
Memory module — read/write to state/memory.json.

All logic is pure Python (regex tokenisation, no LLM calls).

Public API
----------
mem.read(query, history, kinds=None, top_k=8)
    Keyword-overlap search across item.keywords + tokens of item.descriptor.
    Returns ranked top-k list.

mem.filter(kinds=..., goal_id=..., recent=N)
    Structured filter by kind, goal, recency.

mem.record_outcome(tool_call, result_text, artifact_id, ...)
    Write a tool_outcome item. Keywords come from tool name + argument tokens.

mem.remember(text, run_id, kind="fact", ...)
    Write a fact or preference item from user text.
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from .schemas import AgentToolCall, MemoryItem

# ── paths ──────────────────────────────────────────────────────────────────────

STATE_DIR = Path(__file__).parent.parent / "state"
MEMORY_FILE = STATE_DIR / "memory.json"


# ── helpers ────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> set[str]:
    """Lowercase word/digit tokens from arbitrary text."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _score(item: MemoryItem, query_tokens: set[str]) -> float:
    """Fraction of query tokens that overlap with item keyword+descriptor tokens."""
    candidate = set(item.keywords) | _tokenize(item.descriptor)
    if not candidate or not query_tokens:
        return 0.0
    return len(query_tokens & candidate) / len(query_tokens)


# ── Memory class ───────────────────────────────────────────────────────────────

class Memory:
    def __init__(self, path: Path = MEMORY_FILE) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("[]", encoding="utf-8")

    # ── persistence ────────────────────────────────────────────────────────────

    def _load(self) -> list[MemoryItem]:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return [MemoryItem(**r) for r in raw]

    def _save(self, items: list[MemoryItem]) -> None:
        self.path.write_text(
            json.dumps([i.model_dump(mode="json") for i in items], indent=2),
            encoding="utf-8",
        )

    # ── reads ──────────────────────────────────────────────────────────────────

    def read(
        self,
        query: str,
        history: list[dict],
        kinds: list[str] | None = None,
        top_k: int = 8,
    ) -> list[MemoryItem]:
        """
        Keyword-overlap search. Pure Python.

        Builds query tokens from the user query + last 4 history messages,
        scores each memory item by overlap fraction, returns top-k.
        Falls back to most-recent items when no overlap is found.
        """
        items = self._load()
        if kinds:
            items = [i for i in items if i.kind in kinds]

        # Enrich query with recent history content
        combined = query
        for h in history[-4:]:
            content = h.get("content", "")
            if isinstance(content, str):
                combined += " " + content

        query_tokens = _tokenize(combined)
        if not query_tokens:
            return items[:top_k]

        scored = sorted(
            [(item, _score(item, query_tokens)) for item in items],
            key=lambda x: x[1],
            reverse=True,
        )

        # Items with any overlap, capped at top_k
        hits = [item for item, s in scored if s > 0.0][:top_k]

        # Fallback: return most-recent items when nothing matched
        if not hits and items:
            hits = sorted(items, key=lambda i: i.created_at, reverse=True)[:top_k]

        return hits

    def filter(
        self,
        kinds: list[str] | None = None,
        goal_id: str | None = None,
        recent: int | None = None,
    ) -> list[MemoryItem]:
        """
        Structured filter by kind, goal_id, and recency (most-recent first).
        """
        items = self._load()
        if kinds:
            items = [i for i in items if i.kind in kinds]
        if goal_id:
            items = [i for i in items if i.goal_id == goal_id]
        items.sort(key=lambda i: i.created_at, reverse=True)
        if recent is not None:
            items = items[:recent]
        return items

    # ── writes ─────────────────────────────────────────────────────────────────

    def record_outcome(
        self,
        tool_call: AgentToolCall,
        result_text: str,
        artifact_id: str | None,
        run_id: str,
        goal_id: str | None = None,
        confidence: float = 1.0,
    ) -> MemoryItem:
        """
        Record an MCP tool-call outcome as a tool_outcome memory item.
        Keywords come from the tool name and argument value tokens.
        """
        kw: set[str] = _tokenize(tool_call.name)
        for v in tool_call.arguments.values():
            kw |= _tokenize(str(v))
        kw_list = list(kw)[:24]

        # Short descriptor: "tool_name(k=v, ...) → artifact or result preview"
        args_str = ", ".join(
            f"{k}={str(v)[:50]}" for k, v in tool_call.arguments.items()
        )
        desc = f"{tool_call.name}({args_str})"
        if artifact_id:
            desc += f" → {artifact_id}"
        else:
            preview = result_text[:100].replace("\n", " ")
            desc += f" → {preview}"

        item = MemoryItem(
            id=str(uuid.uuid4()),
            kind="tool_outcome",
            keywords=kw_list,
            descriptor=desc[:300],
            value={
                "tool": tool_call.name,
                "arguments": tool_call.arguments,
                "result_preview": result_text[:500],
                "artifact_id": artifact_id,
            },
            artifact_id=artifact_id,
            source=f"action/{tool_call.name}",
            run_id=run_id,
            goal_id=goal_id,
            confidence=confidence,
        )
        items = self._load()
        items.append(item)
        self._save(items)
        return item

    def remember(
        self,
        text: str,
        run_id: str,
        kind: Literal["fact", "preference"] = "fact",
        goal_id: str | None = None,
        confidence: float = 0.9,
    ) -> MemoryItem:
        """
        Store a user-supplied fact or preference in memory.
        Keywords are extracted from the raw text via tokenisation.
        """
        kw_list = list(_tokenize(text))[:24]
        item = MemoryItem(
            id=str(uuid.uuid4()),
            kind=kind,
            keywords=kw_list,
            descriptor=text[:200],
            value={"text": text},
            artifact_id=None,
            source="user",
            run_id=run_id,
            goal_id=goal_id,
            confidence=confidence,
        )
        items = self._load()
        items.append(item)
        self._save(items)
        return item
