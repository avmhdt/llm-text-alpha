"""Shared by the scorers: the ``scores`` schema and which part of a document is read.

An Item 2.02 8-K's cover page is boilerplate: addresses, check boxes and the exhibit list.
When the filing has an EX-99 exhibit (the press release), the dictionary and FinBERT read
the exhibits only; otherwise they read the whole text. Section tags are dropped.
"""

from __future__ import annotations

import re

import polars as pl

_SECTION = re.compile(r"^\[\[([^\]]+)\]\]$", re.MULTILINE)


def sections(text: str) -> list[tuple[str, str]]:
    """``(tag, body)`` pairs in document order; untagged text is tagged ``""``."""
    marks = list(_SECTION.finditer(text))
    if not marks:
        return [("", text.strip())]
    found: list[tuple[str, str]] = []
    for i, mark in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        found.append((mark.group(1), text[mark.end() : end].strip()))
    return found


def scoring_text(text: str) -> str:
    """The EX-99 sections joined, or every section when there is no EX-99."""
    parts = sections(text)
    exhibits = [body for tag, body in parts if tag.upper().startswith("EX-99") and body]
    chosen = exhibits or [body for _, body in parts if body]
    return "\n\n".join(chosen)


SCORES_SCHEMA = pl.Schema(
    [
        ("doc_id", pl.String()),
        ("scorer", pl.String()),
        ("model_id", pl.String()),
        ("prompt_version", pl.String()),
        ("masked", pl.Boolean()),
        ("score", pl.Float64()),
        ("label", pl.String()),
        ("aux_json", pl.String()),
        ("latency_ms", pl.Float64()),
        ("cost_usd", pl.Float64()),
        ("scored_at", pl.Datetime("ns", "UTC")),
    ]
)
