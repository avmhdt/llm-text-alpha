"""Response disk cache (FR-11): ``responses/<model>/<kk>/<key>.json``, written atomically.

Never deleted or overwritten by the pipeline: every entry is paid-for data (section 16).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CachedResponse:
    """What a provider returned for one request, enough to rebuild its ``ScoreResult``."""

    cache_key: str
    doc_id: str
    model_id: str
    provider: str
    raw_text: str
    stop_reason: str
    usage: dict[str, int]
    usd: float
    mode: str
    created_at: str
    schema_valid: bool
    payload: dict[str, Any] | None = None
    latency_ms: float | None = None
    batch_id: str | None = None
    batch_turnaround_s: float | None = None
    attempts: int = 1
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Sorted, indented JSON."""
        return json.dumps(asdict(self), sort_keys=True, indent=1)

    @classmethod
    def from_json(cls, text: str) -> CachedResponse:
        """Inverse of :meth:`to_json`."""
        return cls(**json.loads(text))


_SAFE = re.compile(r"[^A-Za-z0-9._-]")


class ResponseCache:
    """Content-addressed store of responses under ``root/responses``."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, key: str, model_id: str) -> Path:
        """Where the entry for ``key`` lives."""
        return self.root / "responses" / _SAFE.sub("_", model_id) / key[:2] / f"{key}.json"

    def get(self, key: str, model_id: str) -> CachedResponse | None:
        """The cached response, or None."""
        path = self.path(key, model_id)
        if not path.exists():
            return None
        return CachedResponse.from_json(path.read_text("utf-8"))

    def put(self, key: str, response: CachedResponse) -> Path:
        """Write ``response`` through a temporary file and an atomic rename."""
        path = self.path(key, response.model_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        temporary.write_text(response.to_json(), "utf-8")
        temporary.replace(path)
        return path
