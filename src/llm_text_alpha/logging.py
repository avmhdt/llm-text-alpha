"""structlog setup: JSON lines in ``logs/<command>-<run_id>.jsonl`` plus a console stream (NFR-08).

A redaction processor runs before any renderer, so secrets never reach a log line (NFR-05).
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Iterable, MutableMapping
from pathlib import Path
from typing import Any, TextIO

import structlog

REDACTED = "***"
_SECRET_KEY = re.compile(r"(api[_-]?key|token|secret|password|authorization)", re.IGNORECASE)


class Redactor:
    """Mask values of secret-looking keys, and any known secret inside any string value."""

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        self._secrets = sorted({s for s in secrets if s}, key=len, reverse=True)

    def scrub(self, value: Any) -> Any:
        """Return ``value`` with every known secret replaced by ``***``."""
        if isinstance(value, str):
            for secret in self._secrets:
                value = value.replace(secret, REDACTED)
            return value
        if isinstance(value, dict):
            return {key: self._field(str(key), item) for key, item in value.items()}
        if isinstance(value, list | tuple):
            return type(value)(self.scrub(item) for item in value)
        return value

    def _field(self, key: str, value: Any) -> Any:
        if _SECRET_KEY.search(key) and value not in (None, ""):
            return REDACTED
        return self.scrub(value)

    def __call__(
        self, logger: Any, method: str, event: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        return {key: self._field(key, value) for key, value in event.items()}


class _CurrentStderr:
    """Whatever ``sys.stderr`` is at write time, so a swapped or closed stream is never kept."""

    def write(self, line: str) -> None:
        sys.stderr.write(line)

    def flush(self) -> None:
        sys.stderr.flush()


class _Tee:
    """Write each rendered line to every stream."""

    def __init__(self, streams: list[TextIO | _CurrentStderr]) -> None:
        self._streams = streams

    def write(self, line: str) -> None:
        for stream in self._streams:
            stream.write(line)
            stream.flush()

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


_open_files: list[TextIO] = []


def configure_logging(
    level: str = "INFO",
    secrets: Iterable[str] = (),
    log_file: Path | None = None,
    console: TextIO | None = None,
) -> None:
    """Configure structlog for one command.

    Args:
        level: DEBUG, INFO or WARNING.
        secrets: values to scrub from every event.
        log_file: JSON-lines file to append to; its directory is created.
        console: stream for the same JSON lines; defaults to stderr.
    """
    for handle in _open_files:
        handle.close()
    _open_files.clear()
    streams: list[TextIO | _CurrentStderr] = [console or _CurrentStderr()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handle = log_file.open("a", encoding="utf-8")
        _open_files.append(handle)
        streams.append(handle)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            Redactor(secrets),
            structlog.processors.JSONRenderer(sort_keys=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        logger_factory=structlog.WriteLoggerFactory(file=_Tee(streams)),  # type: ignore[arg-type]
        cache_logger_on_first_use=False,
    )


def log_path(logs_dir: Path, command: str, run_id: str) -> Path:
    """``logs/<command>-<run_id>.jsonl`` with spaces in the command name turned into dashes."""
    return logs_dir / f"{command.replace(' ', '-')}-{run_id}.jsonl"
