from __future__ import annotations

import polars as pl
import pytest

from llm_text_alpha.evaluation import latency as lat


def test_quantiles_and_timing() -> None:
    q = lat.quantiles([1.0, 2.0, 3.0, 4.0, float("nan")])
    assert q["n"] == 4 and q["p50_ms"] == pytest.approx(2.5)
    assert lat.quantiles([])["p50_ms"] is None
    assert len(lat.time_calls(lambda: None, 5)) == 5


def test_cost_rows_from_calls_and_scores() -> None:
    signals = pl.DataFrame(
        {"scorer": ["llm"] * 5, "masked": [False] * 5, "quintile": [1, 2, 3, 4, 5]}
    )
    share = lat.traded_share(signals, "llm")
    assert share == pytest.approx(0.4)
    assert lat.traded_share(signals, "lm") is None
    calls = pl.DataFrame(
        {
            "mode": ["batch", "batch", "cache_hit"],
            "usd": [0.0002, 0.0004, 0.0],
            "latency_ms": [None, None, None],
            "batch_turnaround_s": [600.0, 640.0, None],
        },
        schema_overrides={"latency_ms": pl.Float64},
    )
    row = lat.cost_rows("llm", "m", calls, None, share, 2.0)
    assert row["usd_per_doc_batch"] == pytest.approx(0.0003)
    assert row["usd_per_doc_sync"] == pytest.approx(0.0006)
    assert row["usd_per_traded_signal"] == pytest.approx(0.0006 / 0.4)
    assert row["batch_turnaround_max_s"] == 640.0 and row["p50_ms"] is None
    scores = pl.DataFrame({"cost_usd": [0.0, 0.0], "latency_ms": [1.0, 3.0]})
    base = lat.cost_rows("lm", "lm-x", None, scores, 0.4, 1.0)
    assert base["usd_per_doc_sync"] == 0.0 and base["p50_ms"] == pytest.approx(2.0)


def test_budget_sums_components() -> None:
    fetch = {"p50_ms": 500.0, "p95_ms": 900.0}
    decision = {"p50_ms": 0.1, "p95_ms": 0.2}
    scoring = pl.DataFrame(
        [
            {"scorer": "lm", "p50_ms": 2.0, "p95_ms": 4.0},
            {"scorer": "x", "p50_ms": None, "p95_ms": None},
        ]
    )
    out = lat.budget(fetch, decision, scoring)
    assert out.height == 1 and out["total_p50_ms"][0] == pytest.approx(502.1)
    assert out["total_p95_ms"][0] == pytest.approx(904.2)
