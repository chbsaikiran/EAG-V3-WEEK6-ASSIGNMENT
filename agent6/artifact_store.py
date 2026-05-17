"""
ArtifactStore — persist large tool-call payloads (> ARTIFACT_THRESHOLD_BYTES)
to disk so they can be re-attached to future Decision prompts without
re-fetching.

Layout
------
state/artifacts/<sha256[:16]>.bin   — raw UTF-8 bytes of the payload
state/artifacts/index.json          — {art_id: Artifact metadata dict}

Handles are strings of the form  "art:<sha256-prefix-16chars>".
The SHA256 makes puts idempotent: the same content always gets the same handle.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .schemas import Artifact

# ── thresholds & paths ─────────────────────────────────────────────────────────

ARTIFACT_THRESHOLD_BYTES = 4 * 1024   # 4 KB

STATE_DIR   = Path(__file__).parent.parent / "state"
ARTIFACT_DIR   = STATE_DIR / "artifacts"
ARTIFACT_INDEX = ARTIFACT_DIR / "index.json"


# ── store ──────────────────────────────────────────────────────────────────────

class ArtifactStore:
    def __init__(self) -> None:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        if not ARTIFACT_INDEX.exists():
            ARTIFACT_INDEX.write_text("{}", encoding="utf-8")

    # ── index I/O ──────────────────────────────────────────────────────────────

    def _load_index(self) -> dict[str, dict]:
        return json.loads(ARTIFACT_INDEX.read_text(encoding="utf-8"))

    def _save_index(self, index: dict[str, dict]) -> None:
        ARTIFACT_INDEX.write_text(json.dumps(index, indent=2), encoding="utf-8")

    # ── public API ─────────────────────────────────────────────────────────────

    def put(
        self,
        content: bytes,
        source: str,
        content_type: str = "text/plain",
        descriptor: str = "",
    ) -> str:
        """
        Persist bytes and return the art: handle.
        Idempotent — the same bytes always return the same handle.
        """
        sha    = hashlib.sha256(content).hexdigest()
        handle = f"art:{sha[:16]}"

        file_path = ARTIFACT_DIR / f"{sha[:16]}.bin"
        file_path.write_bytes(content)

        meta = Artifact(
            id=handle,
            content_type=content_type,
            size_bytes=len(content),
            source=source,
            descriptor=descriptor or f"{source} ({len(content):,} bytes)",
            path=str(file_path),
        )
        index = self._load_index()
        index[handle] = meta.model_dump()
        self._save_index(index)
        return handle

    def get(self, handle: str) -> bytes:
        """Load artifact bytes. Raises KeyError when handle is unknown."""
        index = self._load_index()
        if handle not in index:
            raise KeyError(f"Artifact {handle!r} not in store")
        return Path(index[handle]["path"]).read_bytes()

    def get_meta(self, handle: str) -> Artifact | None:
        """Return Artifact metadata or None if not found."""
        index = self._load_index()
        entry = index.get(handle)
        return Artifact(**entry) if entry else None


# Module-level singleton — import this everywhere instead of constructing a new one.
store = ArtifactStore()
