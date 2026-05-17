"""Pydantic models for the agent6 cognitive loop.

Kept separate from the gateway's schemas.py to avoid import conflicts.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class MemoryItem(BaseModel):
    id: str
    kind: Literal["fact", "preference", "tool_outcome", "scratchpad"]
    keywords: list[str]
    descriptor: str          # one short human-readable line
    value: dict[str, Any]    # structured payload
    artifact_id: str | None = None   # handle into the artifact store
    source: str
    run_id: str
    goal_id: str | None = None
    confidence: float = 1.0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Artifact(BaseModel):
    id: str           # "art:<sha256-prefix-16chars>"
    content_type: str
    size_bytes: int
    source: str
    descriptor: str
    path: str         # absolute path to the stored .bin file


class Goal(BaseModel):
    id: str
    text: str                       # short imperative description
    done: bool = False
    attach_artifact_id: str | None = None   # art: handle to load for this goal


class Observation(BaseModel):
    goals: list[Goal]


class AgentToolCall(BaseModel):
    """Tool call selected by Decision. Separate from gateway's ToolCall."""
    name: str
    arguments: dict[str, Any]


class DecisionOutput(BaseModel):
    answer: str | None = None       # exactly one of these two is populated
    tool_call: AgentToolCall | None = None
