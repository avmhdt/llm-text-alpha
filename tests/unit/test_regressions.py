"""AC-W-08: regressions match linearmodels within NFR-03 tolerances."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest
from linearmodels.iv import IV2SLS
from linearmodels.panel import FamaMacBeth

from llm_text_alpha.evaluation.regressions import (
    cross_sectional_regression,
    fama_macbeth,
    newey_west_mean,
    ols_twoway,
)


def panel(seed: int, n: int = 4000) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    firm = rng.integers(0, 300, n)
    day = rng.integers(0, 400, n)
    x = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    y = (
        x @ [0.1, 0.5, -0.2]
        + rng.normal(size=n)
        + rng.normal(size=300)[firm]
        + rng.normal(size=400)[day]
    )
    return y, x, firm, day


def test_twoway_matches_linearmodels() -> None:
    y, x, firm, day = panel(3)
    beta, se, _ = ols_twoway(y, x, firm, day)
    frame = pd.DataFrame({"y": y, "c": x[:, 0], "a": x[:, 1], "b": x[:, 2], "f": firm, "d": day})
    ref = IV2SLS(frame.y, frame[["c", "a", "b"]], None, None).fit(
        cov_type="clustered", clusters=frame[["f", "d"]]
    )
    assert np.abs(beta - ref.params.to_numpy()).max() < 1e-8
    assert np.abs(se - ref.std_errors.to_numpy()).max() < 1e-6


def test_fama_macbeth_matches_linearmodels() -> None:
    y, x, _, day = panel(4)
    period = day // 10
    params, errors, n_periods = fama_macbeth(y, x, period, lags=3)
    index = pd.MultiIndex.from_arrays([np.arange(len(y)), period])
    ref = FamaMacBeth(
        pd.Series(y, index=index), pd.DataFrame(x, index=index, columns=["c", "a", "b"])
    ).fit(cov_type="kernel", kernel="bartlett", bandwidth=3)
    assert n_periods == 40
    assert np.abs(params - ref.params.to_numpy()).max() < 1e-8
    assert np.abs(errors - ref.std_errors.to_numpy()).max() < 1e-6


def test_newey_west_without_lags_is_the_plain_error() -> None:
    series = np.array([1.0, 2.0, 4.0, 3.0])
    mean, se = newey_west_mean(series, 0)
    assert mean == 2.5
    assert se == pytest.approx(series.std(ddof=1) / 2)


def test_frame_interface_and_dropped_rows() -> None:
    y, x, firm, day = panel(5, n=600)
    start = date(2016, 1, 4)
    frame = pl.DataFrame(
        {
            "car": y,
            "score_z": x[:, 1],
            "size_log": x[:, 2],
            "permno": firm,
            "entry_date": [start + timedelta(days=int(d)) for d in day],
        }
    ).with_columns(
        pl.when(pl.int_range(pl.len()) == 0).then(None).otherwise(pl.col("car")).alias("car")
    )
    twoway = cross_sectional_regression(frame, "car", ["score_z", "size_log"], "twoway")
    assert twoway.nobs == 599 and twoway.names == ["const", "score_z", "size_log"]
    assert twoway.to_frame()["t"].to_list() == pytest.approx(list(twoway.tstats))
    fm = cross_sectional_regression(frame, "car", ["score_z", "size_log"], "fm")
    assert fm.n_periods > 10 and fm.r2 is None
    assert abs(fm.params[1] - 0.5) < 0.2


def test_fama_macbeth_skips_thin_periods() -> None:
    x = np.column_stack([np.ones(6), [1.0, 2, 3, 1, 1, 1]])
    y = np.array([1.0, 2, 3, 1, 1, 1])
    period = np.array([0, 0, 0, 1, 1, 2])
    params, _, n_periods = fama_macbeth(y, x, np.array(period))
    assert n_periods == 1
    assert params == pytest.approx([0.0, 1.0])
