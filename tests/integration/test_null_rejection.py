"""AC-W-08: with random scores, the 5% two-way test rejects between 2% and 8% of the time."""

from __future__ import annotations

import numpy as np

from llm_text_alpha.evaluation.regressions import fama_macbeth, ols_twoway


def test_twoway_null_rejection_rate() -> None:
    rng = np.random.default_rng(99)
    rejections, sims = 0, 500
    n, firms, dates = 1500, 150, 120
    firm = rng.integers(0, firms, n)
    date = rng.integers(0, dates, n)
    for _ in range(sims):
        score = rng.normal(size=n)
        car = (
            rng.normal(size=n)
            + 0.5 * rng.normal(size=firms)[firm]
            + 0.5 * rng.normal(size=dates)[date]
        )
        x = np.column_stack([np.ones(n), score])
        beta, se, _ = ols_twoway(car, x, firm, date)
        rejections += abs(beta[1] / se[1]) > 1.96
    assert 0.02 <= rejections / sims <= 0.08


def test_fama_macbeth_null_rejection_rate() -> None:
    rng = np.random.default_rng(100)
    rejections, sims = 0, 500
    n, months = 2400, 120
    month = rng.integers(0, months, n)
    for _ in range(sims):
        x = np.column_stack([np.ones(n), rng.normal(size=n)])
        y = rng.normal(size=n) + rng.normal(size=months)[month]
        params, errors, _ = fama_macbeth(y, x, month, lags=3)
        rejections += abs(params[1] / errors[1]) > 1.96
    assert 0.02 <= rejections / sims <= 0.08
