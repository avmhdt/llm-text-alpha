"""Capacity of the 8-K long-short strategy: net alpha against assets under management (FR-21).

``alpha_annual`` is net of D-11 costs and impact. ``alpha_annual_impact_only`` drops the D-11
costs and keeps impact, which shows how far size alone erodes the gross alpha.

At AUM ``A`` each leg holds ``A`` dollars, split equally across that leg's positions on an
event's entry day. A position is capped at ``participation`` times the stock's average
dollar volume over the ``adv_days`` sessions before entry; the part above the cap is not
traded (it sits in cash). Each trade pays a square-root impact cost on entry and on exit,
``impact_coef * sigma * sqrt(q / ADV)``, with ``sigma`` the daily return volatility over the
same ``adv_days`` sessions, on top of the D-11 costs. Dollars are USD; returns decimals.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date

import numpy as np
import polars as pl

from llm_text_alpha.corpus.calendar import TradingCalendar
from llm_text_alpha.evaluation.portfolio import (
    DEFAULT_COSTS,
    active_days,
    daily_returns,
    factor_alpha,
    holdings,
    positions,
)


def liquidity(prices: pl.DataFrame, adv_days: int) -> pl.DataFrame:
    """Per ``permno`` and session: mean dollar volume and return std over the prior sessions."""
    return (
        prices.select("permno", "date", "close", "volume", "ret")
        .sort("permno", "date")
        .with_columns((pl.col("close") * pl.col("volume")).alias("_dv"))
        .with_columns(
            pl.col("_dv")
            .rolling_mean(adv_days, min_samples=adv_days // 2)
            .shift(1)
            .over("permno")
            .alias("adv"),
            pl.col("ret")
            .rolling_std(adv_days, min_samples=adv_days // 2)
            .shift(1)
            .over("permno")
            .alias("sigma"),
        )
        .select("permno", pl.col("date").alias("entry_date"), "adv", "sigma")
    )


def sized(
    signals: pl.DataFrame,
    liquid: pl.DataFrame,
    calendar: TradingCalendar,
    hold: int,
    aum: float,
    participation: float,
    impact_coef: float,
) -> pl.DataFrame:
    """``signals`` with ``scale`` and ``impact`` for AUM ``aum`` (each leg's notional)."""
    pos = positions(signals)
    held = holdings(pos, calendar, hold)
    counts = held.group_by("date", "side").agg(pl.len().alias("n_leg"))
    entry = (
        pos.join(counts, left_on=["entry_date", "side"], right_on=["date", "side"], how="left")
        .join(liquid, on=["permno", "entry_date"], how="left")
        .with_columns((aum / pl.col("n_leg")).alias("q"))
        .with_columns(
            pl.min_horizontal(pl.col("q"), participation * pl.col("adv"))
            .fill_null(0.0)
            .alias("q_cap")
        )
        .with_columns(
            pl.when(pl.col("q") > 0)
            .then(pl.col("q_cap") / pl.col("q"))
            .otherwise(0.0)
            .alias("scale"),
            (
                impact_coef
                * pl.col("sigma").fill_null(0.0)
                * (pl.col("q_cap") / pl.col("adv")).clip(lower_bound=0.0).sqrt()
            )
            .fill_null(0.0)
            .fill_nan(0.0)
            .alias("impact"),
        )
    )
    return signals.join(
        entry.select("doc_id", "scale", "impact", "q", "q_cap"), on="doc_id", how="left"
    )


def capacity_curve(
    signals: pl.DataFrame,
    prices: pl.DataFrame,
    factors: pl.DataFrame,
    calendar: TradingCalendar,
    hold: int,
    aum_grid: Sequence[float],
    participation: float = 0.10,
    adv_days: int = 20,
    impact_coef: float = 0.1,
    costs_bps: Mapping[str, float] = DEFAULT_COSTS,
    lags: int = 10,
    end: date | None = None,
) -> pl.DataFrame:
    """Net alpha, its t and how binding the cap is, for each AUM in ``aum_grid``."""
    liquid = liquidity(prices, adv_days)
    rows = []
    for aum in aum_grid:
        frame = sized(signals, liquid, calendar, hold, float(aum), participation, impact_coef)
        result = daily_returns(frame, prices, calendar, hold, costs_bps, end=end)
        fit = factor_alpha(active_days(result.daily), factors, lags=lags)
        impact_only = daily_returns(frame, prices, calendar, hold, costs_bps, 0.0, end)
        fit_impact = factor_alpha(active_days(impact_only.daily), factors, lags=lags)
        traded = frame.filter(pl.col("q").is_not_null())
        rows.append(
            {
                "aum_usd": float(aum),
                "alpha_annual": fit.alpha_annual,
                "t_alpha": fit.t_alpha,
                "sharpe_annual": fit.sharpe_annual,
                "alpha_annual_impact_only": fit_impact.alpha_annual,
                "t_alpha_impact_only": fit_impact.t_alpha,
                "capped_share": float(np.mean(traded["scale"].to_numpy() < 1.0 - 1e-12)),
                "mean_scale": float(np.mean(traded["scale"].to_numpy())),
                "mean_impact_bps": float(np.mean(traded["impact"].to_numpy())) * 10_000.0,
            }
        )
    return pl.DataFrame(rows)


def breakpoints(curve: pl.DataFrame, column: str = "alpha_annual") -> dict[str, float | None]:
    """AUM at which ``column`` falls to half its smallest-AUM value, and to zero.

    Interpolated linearly in log AUM between grid points; ``None`` when the curve does not
    cross within the grid (or starts at or below zero).
    """
    aum = np.log(curve["aum_usd"].to_numpy())
    alpha = curve[column].to_numpy()
    base = float(alpha[0])

    def crossing(level: float) -> float | None:
        if base <= 0:
            return None
        for i in range(1, len(alpha)):
            if alpha[i] <= level:
                lo, hi = alpha[i - 1], alpha[i]
                share = 0.0 if hi == lo else (lo - level) / (lo - hi)
                return float(np.exp(aum[i - 1] + share * (aum[i] - aum[i - 1])))
        return None

    return {"alpha_at_min_aum": base, "aum_half": crossing(base / 2), "aum_zero": crossing(0.0)}
