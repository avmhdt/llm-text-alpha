"""Calendar-time long-short portfolios net of costs, and their factor alpha (FR-18).

Each event whose trailing quintile is the top one enters the long leg, and each in the
bottom quintile enters the short leg, at the open of its ``entry_date``. It is held ``hold``
trading days: day 0 earns the open-to-close return, days 1 to ``hold - 1`` the close-to-close
return, and it leaves at the close of day ``hold - 1``. Positions in a leg are equally
weighted each day, and the long-short return is the long leg's mean minus the short leg's,
one dollar each side. A missing daily return counts as zero (the position sits in cash that
day) and is counted.

Costs are one-way, in basis points by the company's universe rank at the month end before
entry (D-11), charged on day 0 and on the exit day. Returns are decimals; dates are XNYS
sessions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import polars as pl
import statsmodels.api as sm

from llm_text_alpha.corpus.calendar import TradingCalendar
from llm_text_alpha.evaluation.abnormal import FACTORS

ALPHA_FACTORS = (*FACTORS, "pead")
FACTORS_ONLY = tuple(FACTORS)
"""For the SUE strategy itself, whose returns the PEAD factor is built from."""
DEFAULT_COSTS = {"1-200": 5.0, "201-500": 10.0, "501-1000": 20.0}


def _bands(costs_bps: Mapping[str, float]) -> list[tuple[int, int, float]]:
    bands = []
    for key, bps in costs_bps.items():
        low, high = (int(v) for v in str(key).split("-"))
        bands.append((low, high, float(bps)))
    return sorted(bands)


def cost_bps(rank: pl.Expr, costs_bps: Mapping[str, float]) -> pl.Expr:
    """One-way cost in bps for a universe ``rank``; unknown or deeper ranks take the top band."""
    bands = _bands(costs_bps)
    expr = pl.lit(bands[-1][2])
    for low, high, bps in reversed(bands):
        expr = pl.when(rank.is_between(low, high)).then(pl.lit(bps)).otherwise(expr)
    return expr


def with_rank(
    events: pl.DataFrame, universe: pl.DataFrame, security_of: pl.DataFrame
) -> pl.DataFrame:
    """Add ``rank``: the company's universe rank at the last month end before entry."""
    keyed = events.join(security_of, on="permno", how="left").with_columns(
        (pl.col("entry_date") - timedelta(days=1)).alias("_asof")
    )
    ranks = universe.select("security_id", "month_end", "rank").sort("month_end")
    return (
        keyed.sort("_asof")
        .join_asof(
            ranks, left_on="_asof", right_on="month_end", by="security_id", strategy="backward"
        )
        .drop("_asof", "month_end", "security_id")
    )


def positions(signals: pl.DataFrame, long_q: int = 5, short_q: int = 1) -> pl.DataFrame:
    """One row per event that enters a leg: ``side`` is +1 (long) or -1 (short)."""
    chosen = signals.filter(pl.col("quintile").is_in([long_q, short_q]))
    return chosen.select(
        "doc_id",
        "permno",
        "entry_date",
        pl.when(pl.col("quintile") == long_q).then(1).otherwise(-1).cast(pl.Int8).alias("side"),
        *([pl.col("rank")] if "rank" in chosen.columns else [pl.lit(None, pl.Int32).alias("rank")]),
        *([pl.col("scale")] if "scale" in chosen.columns else [pl.lit(1.0).alias("scale")]),
        *([pl.col("impact")] if "impact" in chosen.columns else [pl.lit(0.0).alias("impact")]),
    )


def holdings(positions: pl.DataFrame, calendar: TradingCalendar, hold: int) -> pl.DataFrame:
    """Each position's ``hold`` sessions: ``day`` 0 is entry; sessions past the calendar drop."""
    days = calendar.days
    start = np.searchsorted(days, positions["entry_date"].cast(pl.Int64).to_numpy())
    frame = positions.with_columns(pl.Series("_s0", start, pl.Int64)).with_columns(
        pl.int_ranges(0, hold).alias("day")
    )
    frame = frame.explode("day").with_columns((pl.col("_s0") + pl.col("day")).alias("_s"))
    frame = frame.filter(pl.col("_s") < len(days))
    return frame.with_columns(
        pl.Series("date", days[frame["_s"].to_numpy()]).cast(pl.Date),
        (pl.col("day") == hold - 1).alias("exit"),
    ).drop("_s0", "_s")


@dataclass(frozen=True)
class PortfolioResult:
    """Daily series (``date``, ``long``, ``short``, ``ls_gross``, ``ls_net``, counts)."""

    daily: pl.DataFrame
    n_positions: int
    missing_returns: int


def daily_returns(
    signals: pl.DataFrame,
    prices: pl.DataFrame,
    calendar: TradingCalendar,
    hold: int,
    costs_bps: Mapping[str, float] = DEFAULT_COSTS,
    cost_mult: float = 1.0,
    end: date | None = None,
) -> PortfolioResult:
    """Daily long-short returns, gross and net of costs, through ``end`` (inclusive).

    Optional ``scale`` (share of the intended position actually held, for capacity) and
    ``impact`` (extra one-way cost, decimal) columns on ``signals`` are applied per event.
    """
    held = holdings(positions(signals), calendar, hold)
    if end is not None:
        held = held.filter(pl.col("date") <= end)
    priced = held.join(
        prices.select("permno", "date", "ret", "ret_open_close"), on=["permno", "date"], how="left"
    ).with_columns(
        pl.when(pl.col("day") == 0)
        .then(pl.col("ret_open_close"))
        .otherwise(pl.col("ret"))
        .alias("r")
    )
    missing = int(priced["r"].null_count())
    one_way = cost_bps(pl.col("rank"), costs_bps) * cost_mult / 10_000.0 + pl.col("impact")
    charge = (
        pl.when(pl.col("day") == 0).then(one_way).otherwise(0.0)
        + pl.when(pl.col("exit")).then(one_way).otherwise(0.0)
    ) * pl.col("scale")
    pnl = priced.with_columns(
        (pl.col("side") * pl.col("r").fill_null(0.0) * pl.col("scale")).alias("gross"),
    ).with_columns((pl.col("gross") - charge).alias("net"))
    legs = pnl.group_by("date", "side").agg(
        pl.col("gross").sort().mean(), pl.col("net").sort().mean(), pl.len().alias("n")
    )
    long = legs.filter(pl.col("side") == 1).select(
        "date",
        pl.col("gross").alias("long_gross"),
        pl.col("net").alias("long_net"),
        pl.col("n").alias("n_long"),
    )
    short = legs.filter(pl.col("side") == -1).select(
        "date",
        pl.col("gross").alias("short_gross"),
        pl.col("net").alias("short_net"),
        pl.col("n").alias("n_short"),
    )
    daily = (
        long.join(short, on="date", how="full", coalesce=True)
        .with_columns(pl.col("n_long", "n_short").fill_null(0))
        .with_columns(pl.col("long_gross", "long_net", "short_gross", "short_net").fill_null(0.0))
        .with_columns(
            (pl.col("long_gross") + pl.col("short_gross")).alias("ls_gross"),
            (pl.col("long_net") + pl.col("short_net")).alias("ls_net"),
        )
        .sort("date")
    )
    return PortfolioResult(daily, held.select("doc_id").n_unique(), missing)


@dataclass(frozen=True)
class AlphaResult:
    """Daily alpha and its Newey-West t; ``alpha_annual`` is 252 times the daily alpha."""

    alpha_daily: float
    t_alpha: float
    alpha_annual: float
    betas: dict[str, float]
    n_days: int
    sharpe_annual: float
    mean_daily: float
    r2: float


def factor_alpha(
    returns: pl.DataFrame,
    factors: pl.DataFrame,
    column: str = "ls_net",
    regressors: Sequence[str] = ALPHA_FACTORS,
    lags: int = 10,
) -> AlphaResult:
    """Regress a zero-investment daily return on factors with an intercept, HAC errors."""
    data = (
        returns.select("date", column)
        .join(factors.select("date", *regressors), on="date", how="inner")
        .drop_nulls()
    )
    if data.height <= len(regressors) + 2:
        raise ValueError(f"only {data.height} days to fit {len(regressors)} factors")
    y = data[column].to_numpy()
    x = sm.add_constant(data.select(regressors).to_numpy(), has_constant="add")
    fit = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    params = np.asarray(fit.params)
    tvalues = np.asarray(fit.tvalues)
    std = float(y.std(ddof=1))
    return AlphaResult(
        alpha_daily=float(params[0]),
        t_alpha=float(tvalues[0]),
        alpha_annual=float(params[0]) * 252.0,
        betas={name: float(b) for name, b in zip(regressors, params[1:], strict=True)},
        n_days=data.height,
        sharpe_annual=float(y.mean() / std * np.sqrt(252.0)) if std > 0 else float("nan"),
        mean_daily=float(y.mean()),
        r2=float(fit.rsquared),
    )


def active_days(daily: pl.DataFrame) -> pl.DataFrame:
    """Days with at least one position in each leg, so the portfolio is zero-investment."""
    return daily.filter((pl.col("n_long") > 0) & (pl.col("n_short") > 0))
