"""Cross-sectional regressions of CARs on scores and controls (FR-17, NFR-03).

- ``twoway``: pooled OLS with standard errors clustered by firm and by entry date
  (Cameron, Gelbach and Miller 2011; Petersen 2009): ``V = V_firm + V_date - V_firm x date``.
- ``fm``: Fama-MacBeth by calendar month of entry, with Newey-West (Bartlett) errors on
  the monthly coefficients.

Both are validated against ``linearmodels`` in the tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray

FM_LAGS = 3


@dataclass(frozen=True)
class RegressionResult:
    """Coefficients with standard errors; ``names`` starts with ``const``."""

    names: list[str]
    params: NDArray[np.float64]
    std_errors: NDArray[np.float64]
    nobs: int
    n_periods: int
    r2: float | None
    se: str

    @property
    def tstats(self) -> NDArray[np.float64]:
        """Coefficient over standard error."""
        return self.params / self.std_errors

    def to_frame(self) -> pl.DataFrame:
        """One row per regressor."""
        return pl.DataFrame(
            {
                "term": self.names,
                "coef": self.params,
                "se": self.std_errors,
                "t": self.tstats,
            }
        )


def _cluster_meat(
    x: NDArray[np.float64], e: NDArray[np.float64], groups: NDArray[np.int64]
) -> NDArray[np.float64]:
    scores = x * e[:, None]
    _, index = np.unique(groups, return_inverse=True)
    summed = np.zeros((int(index.max()) + 1, x.shape[1]))
    np.add.at(summed, index, scores)
    return summed.T @ summed


def ols_twoway(
    y: NDArray[np.float64],
    x: NDArray[np.float64],
    firm: NDArray[np.int64],
    date: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """OLS coefficients, two-way clustered standard errors and R-squared.

    No small-sample scaling, matching ``linearmodels``' clustered covariance defaults.
    """
    bread = np.linalg.inv(x.T @ x)
    beta = bread @ x.T @ y
    e = y - x @ beta
    pair = np.unique(np.column_stack([firm, date]), axis=0, return_inverse=True)[1].ravel()
    meat = _cluster_meat(x, e, firm) + _cluster_meat(x, e, date) - _cluster_meat(x, e, pair)
    cov = bread @ meat @ bread
    centered = y - y.mean()
    r2 = float(1 - (e @ e) / (centered @ centered)) if centered @ centered > 0 else None
    return beta, np.sqrt(np.diag(cov)), r2 if r2 is not None else float("nan")


def newey_west_mean(series: NDArray[np.float64], lags: int) -> tuple[float, float]:
    """Mean of ``series`` and its Newey-West standard error with Bartlett weights.

    The long-run variance is divided by ``T - 1``, as ``linearmodels`` does for
    Fama-MacBeth; with ``T`` the errors would be smaller by ``sqrt((T - 1) / T)``.
    """
    t = len(series)
    mean = float(series.mean())
    if t < 2:
        return mean, float("nan")
    d = series - mean
    gamma0 = float(d @ d) / t
    total = gamma0
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        total += 2.0 * weight * float(d[lag:] @ d[:-lag]) / t
    return mean, float(np.sqrt(total / (t - 1)))


def fama_macbeth(
    y: NDArray[np.float64], x: NDArray[np.float64], period: NDArray[np.int64], lags: int = FM_LAGS
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Average of per-period OLS coefficients with Newey-West errors; periods with too few
    observations to fit are skipped."""
    coefs: list[NDArray[np.float64]] = []
    for value in np.unique(period):
        mask = period == value
        if mask.sum() <= x.shape[1]:
            continue
        xs, ys = x[mask], y[mask]
        if np.linalg.matrix_rank(xs) < x.shape[1]:
            continue
        coefs.append(np.linalg.lstsq(xs, ys, rcond=None)[0])
    if not coefs:
        nan = np.full(x.shape[1], np.nan)
        return nan, nan.copy(), 0
    table = np.array(coefs)
    params = np.empty(x.shape[1])
    errors = np.empty(x.shape[1])
    for j in range(x.shape[1]):
        params[j], errors[j] = newey_west_mean(table[:, j], lags)
    return params, errors, len(coefs)


def cross_sectional_regression(
    panel: pl.DataFrame,
    y: str,
    x: Sequence[str],
    se: Literal["twoway", "fm"],
    firm: str = "permno",
    date: str = "entry_date",
    lags: int = FM_LAGS,
) -> RegressionResult:
    """Regress ``y`` on a constant and ``x``; rows with any null are dropped."""
    used = panel.select(y, *x, firm, date).drop_nulls()
    yv = used[y].to_numpy().astype(np.float64)
    xv = np.column_stack([np.ones(used.height), used.select(x).to_numpy().astype(np.float64)])
    names = ["const", *x]
    if used.height <= xv.shape[1] or np.linalg.matrix_rank(xv) < xv.shape[1]:
        nan = np.full(xv.shape[1], np.nan)
        return RegressionResult(names, nan, nan.copy(), used.height, 0, None, se)
    if se == "twoway":
        params, errors, r2 = ols_twoway(
            yv,
            xv,
            used[firm].to_numpy().astype(np.int64),
            used[date].cast(pl.Int64).to_numpy(),
        )
        return RegressionResult(names, params, errors, used.height, used[date].n_unique(), r2, se)
    months = (
        used.select((pl.col(date).dt.year() * 12 + pl.col(date).dt.month()).alias("m"))["m"]
        .to_numpy()
        .astype(np.int64)
    )
    params, errors, n_periods = fama_macbeth(yv, xv, months, lags)
    return RegressionResult(names, params, errors, used.height, n_periods, None, se)
