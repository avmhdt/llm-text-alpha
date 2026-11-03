"""AC-W-07, unit part: the factor-model fit and abnormal returns."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import statsmodels.api as sm

from llm_text_alpha.corpus.calendar import xnys
from llm_text_alpha.evaluation.abnormal import (
    FACTORS,
    abnormal_returns,
    cumulative,
    drop_counts,
    fit_factor_model,
)
from tests.synthetic import make_market, random_events

CAL = xnys()


def test_betas_match_statsmodels() -> None:
    rng = np.random.default_rng(0)
    f = rng.normal(size=(220, 6))
    y = 0.001 + f @ rng.normal(size=6) + rng.normal(0, 0.01, 220)
    y[5] = np.nan
    beta, sigma, n = fit_factor_model(y, f)
    ref = sm.OLS(y[np.isfinite(y)], sm.add_constant(f[np.isfinite(y)])).fit()
    assert n == 219
    assert np.abs(beta - ref.params).max() < 1e-8
    assert sigma == pytest.approx(np.sqrt(ref.scale), rel=1e-10)


def test_too_few_observations() -> None:
    beta, sigma, n = fit_factor_model(np.ones(5), np.ones((5, 6)))
    assert n == 5 and np.isnan(beta).all() and np.isnan(sigma)


def test_models_and_windows() -> None:
    market = make_market(1, n_firms=20)
    events = random_events(2, market.sessions, 20, 30)
    for model in ("ff5umd", "market", "raw"):
        result = abnormal_returns(
            events,
            market.prices,
            market.factors,
            model,
            (-250, -30),
            (-10, 60),
            CAL,
            keep_residuals=True,
        )
        assert result.ar.height == 30 * 71
        assert set(result.ar["rel_day"].unique()) == set(range(-10, 61))
        assert (result.events["model_used"] == model).all()
        assert result.events["sigma"].is_not_null().all()
        assert result.residuals is not None and result.residuals.height > 0
    raw = abnormal_returns(
        events, market.prices, market.factors, "raw", (-250, -30), (-10, 60), CAL
    )
    day0 = raw.ar.filter(pl.col("rel_day") == 0).join(
        market.prices.select("permno", "date", "ret_open_close"), on=["permno", "date"]
    )
    assert (day0["ar"] - day0["ret_open_close"]).abs().max() < 1e-15
    assert raw.residuals is None


def test_fallback_and_drop_reasons() -> None:
    market = make_market(3, n_firms=5, n_days=400)
    sessions = market.sessions
    events = pl.DataFrame(
        {
            "doc_id": ["early", "ok", "late", "gap", "noopen", "unknown"],
            "permno": [0, 1, 2, 3, 4, 99],
            "entry_date": [
                sessions[60],
                sessions[300],
                sessions[350],
                sessions[300],
                sessions[300],
                sessions[300],
            ],
        }
    ).with_columns(pl.col("permno").cast(pl.Int64))
    prices = market.prices.filter(
        ~((pl.col("permno") == 3) & pl.col("date").is_between(sessions[301], sessions[345]))
    ).with_columns(
        pl.when((pl.col("permno") == 4) & (pl.col("date") == sessions[300]))
        .then(None)
        .otherwise(pl.col("open"))
        .alias("open")
    )
    result = abnormal_returns(events, prices, market.factors, "ff5umd", (-250, -30), (-10, 60), CAL)
    reasons = dict(zip(result.events["doc_id"], result.events["drop_reason"], strict=True))
    used = dict(zip(result.events["doc_id"], result.events["model_used"], strict=True))
    assert used["early"] == "market"
    assert reasons["ok"] is None
    assert reasons["late"] is None
    assert reasons["gap"] == "window_missing"
    assert reasons["noopen"] == "no_entry_open"
    assert reasons["unknown"] == "no_prices"
    counts = dict(drop_counts(result.events).iter_rows())
    assert counts["kept"] == 3
    late = result.ar.filter(pl.col("doc_id") == "late")
    assert late["ar"].null_count() > 0


def test_window_outside_calendar() -> None:
    market = make_market(4, n_firms=1, n_days=300)
    events = pl.DataFrame(
        {"doc_id": ["x"], "permno": [0], "entry_date": [CAL.first_session]}
    ).with_columns(pl.col("permno").cast(pl.Int64))
    result = abnormal_returns(
        events, market.prices, market.factors, "raw", (-250, -30), (-10, 60), CAL
    )
    assert result.events["drop_reason"].to_list() == ["window_outside_calendar"]
    assert result.ar.is_empty()


def test_cumulative() -> None:
    ar = pl.DataFrame(
        {
            "doc_id": ["a"] * 4,
            "permno": [1] * 4,
            "rel_day": [-1, 0, 1, 2],
            "ar": [9.0, 1.0, None, 2.0],
        }
    )
    car = cumulative(ar, [1, 2])
    assert car.row(0, named=True) == {"doc_id": "a", "permno": 1, "car_1": 1.0, "car_2": 3.0}
    assert FACTORS[0] == "mktrf"
