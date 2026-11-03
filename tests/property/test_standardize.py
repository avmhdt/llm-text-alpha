"""DQ-10: perturbing any future score never changes a past ``score_z`` or quintile."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

from llm_text_alpha.corpus.calendar import xnys
from llm_text_alpha.signals.build import standardize

CAL = xnys()
SESSIONS = CAL.sessions(date(2016, 1, 4), date(2017, 12, 29))


def frame(days: list[int], scores: list[float]) -> pl.DataFrame:
    n = len(days)
    return pl.DataFrame(
        {
            "doc_id": [f"d{i}" for i in range(n)],
            "permno": list(range(n)),
            "scorer": ["lm"] * n,
            "model_id": ["m"] * n,
            "masked": [False] * n,
            "entry_date": [SESSIONS[d] for d in days],
            "score": scores,
        }
    )


@settings(max_examples=60, deadline=None)
@given(
    days=st.lists(st.integers(0, len(SESSIONS) - 1), min_size=30, max_size=120),
    seed=st.integers(0, 2**31),
    cut=st.integers(0, len(SESSIONS) - 1),
)
def test_trailing_only(days: list[int], seed: int, cut: int) -> None:
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=len(days)).tolist()
    base = standardize(frame(days, scores), CAL, window=252, min_events=5)
    perturbed_scores = [
        s + rng.normal() * 10 if d > cut else s for s, d in zip(scores, days, strict=True)
    ]
    perturbed = standardize(frame(days, perturbed_scores), CAL, window=252, min_events=5)
    past = pl.col("entry_date") <= SESSIONS[cut]
    a = base.filter(past).sort("doc_id")
    b = perturbed.filter(past).sort("doc_id")
    assert a["quintile"].to_list() == b["quintile"].to_list()
    assert a["score_z"].to_list() == b["score_z"].to_list()


def test_values_and_warm_up() -> None:
    days = [0] * 10 + [1]
    scores = [float(i) for i in range(10)] + [4.5]
    out = standardize(frame(days, scores), CAL, window=252, min_events=10).sort("doc_id")
    first = out.filter(pl.col("entry_date") == SESSIONS[0])
    assert first["score_z"].null_count() == 10 and (first["quintile"] == 0).all()
    last = out.filter(pl.col("doc_id") == "d10").row(0, named=True)
    history = np.arange(10.0)
    assert last["score_z"] == (4.5 - history.mean()) / history.std(ddof=1)
    assert last["quintile"] == 3


def test_window_drops_old_events_and_rejects_non_sessions() -> None:
    days = [0] * 5 + [300]
    out = standardize(frame(days, [1.0, 2, 3, 4, 5, 9]), CAL, window=252, min_events=5)
    assert out.filter(pl.col("doc_id") == "d5")["score_z"].to_list() == [None]
    bad = frame([0], [1.0]).with_columns(pl.lit(date(2016, 1, 9)).alias("entry_date"))
    try:
        standardize(bad, CAL)
    except ValueError as exc:
        assert "session" in str(exc)
    else:
        raise AssertionError("a Saturday entry date should be refused")
