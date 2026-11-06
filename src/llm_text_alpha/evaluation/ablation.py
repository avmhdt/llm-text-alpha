"""Ablation on the identical event set (FR-19, AC-W-03).

The event set is the intersection: events every scorer in ``scorers`` scored, each with a
trailing ``score_z`` from its own history. Every combination of scorers (single scorers,
pairs, the triple) gets a combined score, the mean of its members' ``score_z``, which is
standardized again against the trailing 252 sessions of the same event set, with trailing
quintiles. Single scorers go through the same step, so every row is ranked the same way
on the same events.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np
import polars as pl

from llm_text_alpha.corpus.calendar import TradingCalendar
from llm_text_alpha.signals.build import standardize

KEEP = (
    "doc_id",
    "permno",
    "event_ts_utc",
    "entry_date",
    "entry_ts_utc",
    "size_log",
    "mom_12_2",
    "sue",
    "bm",
    "rank",
)


def combos(scorers: Sequence[str]) -> list[tuple[str, ...]]:
    """Every non-empty combination, singles first."""
    return [c for k in range(1, len(scorers) + 1) for c in combinations(scorers, k)]


def intersection(signals: pl.DataFrame, scorers: Sequence[str]) -> pl.DataFrame:
    """One row per event with a ``z_<scorer>`` column for each scorer; events missing any drop."""
    wide: pl.DataFrame | None = None
    for scorer in scorers:
        part = signals.filter(
            (pl.col("scorer") == scorer) & ~pl.col("masked") & pl.col("score_z").is_not_null()
        )
        cols = part.select(*KEEP, pl.col("score_z").alias(f"z_{scorer}"))
        wide = (
            cols if wide is None else wide.join(cols.select("doc_id", f"z_{scorer}"), on="doc_id")
        )
    if wide is None:
        raise ValueError("no scorers")
    return wide.sort("entry_date", "permno", "doc_id")


def combined(events: pl.DataFrame, combo: Sequence[str], calendar: TradingCalendar) -> pl.DataFrame:
    """``events`` with the combination's trailing ``score_z`` and ``quintile``."""
    name = "+".join(combo)
    score = pl.mean_horizontal([pl.col(f"z_{s}") for s in combo])
    frame = events.select(
        *KEEP,
        pl.lit(name).alias("scorer"),
        pl.lit("ablation").alias("model_id"),
        pl.lit(False).alias("masked"),
        score.alias("score"),
        pl.lit(None, pl.Float64).alias("score_z"),
    )
    return standardize(frame, calendar)


def spread(frame: pl.DataFrame, car: str) -> dict[str, float]:
    """Top minus bottom quintile mean of ``car`` and its Welch t."""
    top = frame.filter(pl.col("quintile") == 5)[car].drop_nulls().to_numpy()
    bottom = frame.filter(pl.col("quintile") == 1)[car].drop_nulls().to_numpy()
    if len(top) < 2 or len(bottom) < 2:
        return {"spread": float("nan"), "t_spread": float("nan")}
    diff = float(top.mean() - bottom.mean())
    se = float(np.sqrt(top.var(ddof=1) / len(top) + bottom.var(ddof=1) / len(bottom)))
    return {"spread": diff, "t_spread": diff / se if se > 0 else float("nan")}


def information_coefficient(frame: pl.DataFrame, car: str) -> dict[str, float]:
    """Pooled Spearman IC of ``score_z`` against ``car``, and the mean monthly IC with its t."""
    # sorted so every reduction sums in the same order on every run (byte-identical repro)
    data = (
        frame.select("entry_date", "score_z", car).drop_nulls().sort("entry_date", "score_z", car)
    )
    pooled = (
        float(data.select(pl.corr("score_z", car, method="spearman")).item())
        if data.height > 2
        else float("nan")
    )
    monthly = (
        data.group_by(pl.col("entry_date").dt.truncate("1mo"), maintain_order=True)
        .agg(pl.corr("score_z", car, method="spearman").alias("ic"), pl.len().alias("n"))
        .filter(pl.col("n") >= 10)
        .drop_nulls("ic")
        .sort("entry_date")
    )
    ic = monthly["ic"].to_numpy()
    if len(ic) < 3:
        return {
            "ic": pooled,
            "ic_monthly": float("nan"),
            "t_ic_monthly": float("nan"),
            "months": len(ic),
        }
    mean = float(ic.mean())
    t = mean / (float(ic.std(ddof=1)) / np.sqrt(len(ic)))
    return {"ic": pooled, "ic_monthly": mean, "t_ic_monthly": t, "months": len(ic)}
