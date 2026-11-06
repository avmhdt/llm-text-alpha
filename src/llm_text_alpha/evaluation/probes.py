"""Memorization probes (FR-20): re-identification from masked text, and outcome recall.

Re-identification: the model reads a masked release and names the company and the quarter.
A company guess is right when its normalized name equals a normalized name variant of the
filer, one contains the other (at four or more letters), or the guessed ticker equals the
filer's. The quarter is right when the guessed calendar year and quarter equal those of the
reported quarter's end (``fqtr_end`` minus 7 days, which puts 52/53-week quarter ends such
as 2016-01-03 in the quarter they close).

Outcome recall: with no text, the model is told the company, its ticker and the filing date,
and says whether the stock beat the market over the next five sessions. The truth is the
sign of CAR[0,5]. 50% is chance.

Accuracy is reported before and after each model's training cutoff with Wilson 95%
intervals.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

import polars as pl

from llm_text_alpha.corpus.mask import name_variants
from llm_text_alpha.scoring.llm.base import ScoreRequest
from llm_text_alpha.scoring.llm.prompts import Prompt
from llm_text_alpha.scoring.llm.schema import RECALL_SCHEMA, REID_SCHEMA

MAX_OUTPUT_TOKENS = 256


def normalize(name: str, suffixes: Sequence[str]) -> str:
    """Lower case, letters and digits only, legal suffixes and a leading 'the' dropped."""
    words = re.findall(r"[a-z0-9]+", name.lower())
    while words and words[-1] in suffixes:
        words.pop()
    while words and words[0] == "the":
        words.pop(0)
    return " ".join(words)


def company_match(
    guess: str | None,
    guess_ticker: str | None,
    names: Sequence[str],
    ticker: str | None,
    suffixes: Sequence[str],
) -> bool:
    """Whether a guessed company (or ticker) is the filer."""
    if guess_ticker and ticker and guess_ticker.strip().upper() == ticker.strip().upper():
        return True
    g = normalize(guess or "", suffixes)
    if not g or g == "unknown":
        return False
    for variant in name_variants(names, suffixes):
        v = normalize(variant, suffixes)
        if not v:
            continue
        if g == v or (min(len(g), len(v)) >= 4 and (g in v or v in g)):
            return True
    return False


def true_quarter(fqtr_end: date | None) -> tuple[int, int] | None:
    """Calendar (year, quarter) of a fiscal quarter's end, shifted back 7 days."""
    if fqtr_end is None:
        return None
    from datetime import timedelta

    d = fqtr_end - timedelta(days=7)
    return d.year, (d.month - 1) // 3 + 1


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for ``k`` successes in ``n``."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def reid_requests(
    masked: pl.DataFrame, model_id: str, prompt: Prompt, params: Mapping[str, Any]
) -> list[ScoreRequest]:
    """One re-identification request per masked document (``doc_id``, ``text_masked``)."""
    return [
        ScoreRequest(
            doc_id=doc_id,
            model_id=model_id,
            system_prompt=prompt.text,
            user_text=text,
            output_schema=REID_SCHEMA,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            request_params=dict(params),
            answer="reid",
        )
        for doc_id, text in masked.select("doc_id", "text_masked").iter_rows()
    ]


def recall_text(company: str, ticker: str | None, filed: date) -> str:
    """The no-text question for one event."""
    tick = f" (ticker {ticker})" if ticker else ""
    return (
        f"Company: {company}{tick}.\n"
        f"Earnings release filed on Form 8-K, Item 2.02, on {filed.isoformat()}.\n"
        "Did the stock beat the market over the next five trading days?"
    )


def recall_requests(
    events: pl.DataFrame, model_id: str, prompt: Prompt, params: Mapping[str, Any]
) -> list[ScoreRequest]:
    """One outcome-recall request per event (``doc_id``, ``company_name``, ``ticker``, date)."""
    return [
        ScoreRequest(
            doc_id=r["doc_id"],
            model_id=model_id,
            system_prompt=prompt.text,
            user_text=recall_text(r["company_name"], r["ticker"], r["event_ts_utc"].date()),
            output_schema=RECALL_SCHEMA,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            request_params=dict(params),
            answer="recall",
        )
        for r in events.iter_rows(named=True)
    ]


def accuracy(frame: pl.DataFrame, correct: str, post_start: date | None) -> pl.DataFrame:
    """Accuracy of boolean column ``correct`` before and after ``post_start``, Wilson CIs."""
    rows = []
    window = (
        pl.when(pl.col("event_ts_utc").dt.date() >= post_start)
        .then(pl.lit("post"))
        .otherwise(pl.lit("pre"))
        if post_start is not None
        else pl.lit("all")
    )
    data = frame.with_columns(window.alias("window")).drop_nulls(correct)
    for name in ("pre", "post", "all"):
        part = data if name == "all" else data.filter(pl.col("window") == name)
        n = part.height
        k = int(part[correct].sum()) if n else 0
        lo, hi = wilson(k, n)
        rows.append(
            {"window": name, "n": n, "accuracy": k / n if n else None, "lo95": lo, "hi95": hi}
        )
    return pl.DataFrame(rows)
