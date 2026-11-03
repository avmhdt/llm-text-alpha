"""Atomic file writes: write a temporary file next to the target, then rename it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl


def write_parquet_atomic(frame: pl.DataFrame, path: Path) -> None:
    """Write ``frame`` to ``path`` as zstd Parquet through a temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.write_parquet(temporary, compression="zstd")
    temporary.replace(path)


def write_text_atomic(text: str, path: Path) -> None:
    """Write UTF-8 ``text`` to ``path`` with LF line endings through a temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    temporary.replace(path)


def write_json_atomic(payload: Any, path: Path) -> None:
    """Write ``payload`` as indented, key-sorted JSON through a temporary file."""
    write_text_atomic(json.dumps(payload, indent=2, sort_keys=True) + "\n", path)
