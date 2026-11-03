"""Loughran-McDonald dictionary scores (FR-08).

Counts Positive, Negative, Uncertainty, Litigious and Constraining words. A positive word
with "no", "not" or "never" among the three words before it is not counted, following
Loughran and McDonald (2011). ``score = (pos - neg) / (pos + neg + 1)``, in (-1, 1).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from llm_text_alpha.scoring.common import SCORES_SCHEMA, scoring_text

CATEGORIES = ("positive", "negative", "uncertainty", "litigious", "constraining")
_COLUMNS = {
    "Positive": "positive",
    "Negative": "negative",
    "Uncertainty": "uncertainty",
    "Litigious": "litigious",
    "Constraining": "constraining",
}
NEGATORS = frozenset({"NO", "NOT", "NEVER"})
_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


class DictionaryError(ValueError):
    """The dictionary file does not have the Master Dictionary layout."""


@dataclass(frozen=True)
class LMDictionary:
    """Upper-case word sets per category, with an id that pins the file's content."""

    words: dict[str, frozenset[str]]
    model_id: str


def load_dictionary(path: Path) -> LMDictionary:
    """Read the Master Dictionary CSV. A non-zero year in a category column marks membership
    (negative years flag words later removed and are left out)."""
    if not path.exists():
        raise DictionaryError(f"{path} does not exist; place the Master Dictionary CSV there")
    frame = pl.read_csv(path, infer_schema_length=0)
    missing = [c for c in ("Word", *_COLUMNS) if c not in frame.columns]
    if missing:
        raise DictionaryError(f"{path} lacks columns {missing}")
    words = {
        name: frozenset(
            frame.filter(pl.col(column).cast(pl.Int64, strict=False) > 0)["Word"]
            .str.to_uppercase()
            .to_list()
        )
        for column, name in _COLUMNS.items()
    }
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:8]
    return LMDictionary(words=words, model_id=f"lm-{path.stem}@{digest}")


def tokens(text: str) -> list[str]:
    """Alphabetic tokens, upper-cased; apostrophes inside words are kept."""
    return [match.group(0).upper() for match in _WORD.finditer(text)]


def count(text: str, dictionary: LMDictionary, negation_window: int = 3) -> dict[str, int]:
    """Category counts and ``n_words``, with negated positives removed."""
    words = tokens(text)
    counts = dict.fromkeys(CATEGORIES, 0)
    negated = 0
    for i, word in enumerate(words):
        for category in CATEGORIES:
            if word in dictionary.words[category]:
                if category == "positive" and NEGATORS.intersection(
                    words[max(0, i - negation_window) : i]
                ):
                    negated += 1
                    continue
                counts[category] += 1
    return {**counts, "negated_positive": negated, "n_words": len(words)}


def tone(counts: dict[str, int]) -> float:
    """``(pos - neg) / (pos + neg + 1)``."""
    return (counts["positive"] - counts["negative"]) / (counts["positive"] + counts["negative"] + 1)


def label(score: float) -> str:
    """Sign of the score: positive, negative or neutral."""
    return "positive" if score > 0 else "negative" if score < 0 else "neutral"


def score_documents(
    documents: pl.DataFrame, dictionary: LMDictionary, negation_window: int = 3
) -> pl.DataFrame:
    """``scores`` rows (section 7b) for ``doc_id, text`` with the dictionary scorer."""
    rows: list[dict[str, object]] = []
    for doc_id, text in documents.select("doc_id", "text").iter_rows():
        started = time.perf_counter()
        counts = count(scoring_text(text), dictionary, negation_window)
        value = tone(counts)
        rows.append(
            {
                "doc_id": doc_id,
                "scorer": "lm",
                "model_id": dictionary.model_id,
                "prompt_version": None,
                "masked": False,
                "score": value,
                "label": label(value),
                "aux_json": json.dumps(counts, sort_keys=True),
                "latency_ms": (time.perf_counter() - started) * 1000.0,
                "cost_usd": 0.0,
                "scored_at": datetime.now(UTC),
            }
        )
    return pl.DataFrame(rows, schema=SCORES_SCHEMA)
