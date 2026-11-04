"""Trial accounting and the Deflated Sharpe Ratio (FR-30, AC-H-08).

The DSR follows Bailey and Lopez de Prado (2014), as backtest-overfitting-lab implements it
(``overfitlab.deflated``), and the tests check this module against that package. Sharpe
ratios here are per period (daily) unless a name says ``annual``.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np
import polars as pl

EULER_GAMMA = 0.5772156649015329
_NORMAL = NormalDist()


def expected_max_sharpe(n_trials: float, var_trials: float) -> float:
    """Expected maximum of ``n_trials`` per-period Sharpe ratios with variance ``var_trials``."""
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if n_trials == 1:
        return 0.0
    upper = _NORMAL.inv_cdf
    z = (1 - EULER_GAMMA) * upper(1 - 1 / n_trials) + EULER_GAMMA * upper(
        1 - 1 / (n_trials * math.e)
    )
    return math.sqrt(var_trials) * z


@dataclass(frozen=True)
class Deflated:
    """DSR (a probability), the selected Sharpe and the threshold ``sr0``, per period."""

    dsr: float
    sr: float
    sr0: float
    z: float
    n_trials: int
    var_trials: float
    n_obs: int


def deflated_sharpe(returns: np.ndarray, n_trials: int, var_trials: float) -> Deflated:
    """DSR of a daily return series selected out of ``n_trials`` trials."""
    r = np.asarray(returns, dtype=np.float64)
    n = len(r)
    if n < 3:
        raise ValueError("need at least 3 returns")
    std = float(r.std(ddof=1))
    sr = float(r.mean()) / std
    centred = r - r.mean()
    m2 = float((centred**2).mean())
    skew = float((centred**3).mean()) / m2**1.5
    kurt = float((centred**4).mean()) / m2**2
    sr0 = expected_max_sharpe(n_trials, var_trials)
    variance = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    z = (sr - sr0) / math.sqrt(variance / (n - 1))
    return Deflated(_NORMAL.cdf(z), sr, sr0, z, n_trials, var_trials, n)


def read_log(path: Path) -> pl.DataFrame:
    """The trial log as strings, one row per trial."""
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return pl.DataFrame(rows, schema={k: pl.String for k in (rows[0] if rows else {"run_id": ""})})


def counts(log: pl.DataFrame) -> pl.DataFrame:
    """Trials by family and kind, with how many have a return series."""
    return (
        log.with_columns(pl.col("n_obs").cast(pl.Int64))
        .group_by("strategy_family", "x_run_kind")
        .agg(pl.len().alias("trials"), (pl.col("n_obs") > 0).sum().alias("with_returns"))
        .sort("strategy_family", "x_run_kind")
    )


def sharpe_variance(log: pl.DataFrame) -> tuple[int, float]:
    """Number of trials with a series, and the variance of their per-period Sharpe ratios."""
    series = log.filter(pl.col("n_obs").cast(pl.Int64) > 0)
    annual = series["sharpe_annual"].cast(pl.Float64).to_numpy()
    periods = series["x_periods_per_year"].cast(pl.Float64).to_numpy()
    daily = annual / np.sqrt(periods)
    var = float(daily.var(ddof=1)) if len(daily) > 1 else 0.0
    return len(daily), var
