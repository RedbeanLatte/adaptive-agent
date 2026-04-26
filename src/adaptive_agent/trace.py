from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

_SAFE_SESSION_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_session_id(session_id: str) -> str:
    normalized = _SAFE_SESSION_ID_RE.sub("_", str(session_id or "default")).strip("._")
    return normalized or "default"


class TraceWriter:
    """Append-only compact trace writer. Never records raw chain-of-thought.

    Each event is one JSON object per line under
    ``<state_dir>/traces/<safe_session_id>.jsonl``.
    """

    def __init__(self, state_dir: Path | str | None, session_id: str) -> None:
        self.enabled = state_dir is not None
        if self.enabled:
            base = Path(state_dir) / "traces"
            base.mkdir(parents=True, exist_ok=True)
            safe_session_id = sanitize_session_id(session_id)
            self.path: Path | None = base / f"{safe_session_id}.jsonl"
        else:
            self.path = None

    def event(self, kind: str, **fields: Any) -> None:
        if not self.enabled or self.path is None:
            return
        payload = {"ts": round(time.time(), 3), "kind": kind, **fields}
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except OSError:
            # tracing must never crash the runtime
            return
