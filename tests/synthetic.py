"""A synthetic market for statistical tests: factor-model returns on real XNYS sessions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl

from llm_text_alpha.corpus.calendar import xnys
from llm_text_alpha.evaluation.abnormal import FACTORS

START = date(2016, 1, 4)


@dataclass(frozen=True)
class Market:
    prices: pl.DataFrame
    factors: pl.DataFrame
    sessions: list[date]
    betas: np.ndarray


def make_market(seed: int, n_firms: int = 200, n_days: int = 600, idio: float = 0.015) -> Market:
    """Daily factor returns, and firm returns = rf + alpha + beta' f + noise; open = prior close."""
    rng = np.random.default_rng(seed)
    sessions = xnys().sessions(START, date(2030, 1, 1))[:n_days]
    f = rng.normal(0.0, [0.01, 0.005, 0.005, 0.004, 0.004, 0.006], size=(n_days, len(FACTORS)))
    rf = np.full(n_days, 0.0001)
    betas = np.column_stack(
        [
            rng.normal(0.0002, 0.0002, n_firms),
            rng.normal(1.0, 0.3, n_firms),
            rng.normal(0.0, 0.5, (n_firms, len(FACTORS) - 1)),
        ]
    )
    rets = rf[None, :] + betas[:, :1] + betas[:, 1:] @ f.T + rng.normal(0, idio, (n_firms, n_days))
    prices = pl.DataFrame(
        {
            "permno": np.repeat(np.arange(n_firms), n_days),
            "date": np.tile(np.array(sessions, dtype="datetime64[D]"), n_firms),
            "ret": rets.ravel(),
        }
    ).with_columns(
        pl.col("date").cast(pl.Date),
        pl.lit(10.0).alias("open"),
        (pl.col("ret") * 0.6).alias("ret_open_close"),
    )
    factors = pl.DataFrame(
        {
            "date": np.array(sessions, dtype="datetime64[D]"),
            **{n: f[:, i] for i, n in enumerate(FACTORS)},
            "rf": rf,
        }
    ).with_columns(pl.col("date").cast(pl.Date))
    return Market(prices, factors, sessions, betas)


def plant(
    prices: pl.DataFrame, events: pl.DataFrame, sessions: list[date], total: float, days: int = 6
) -> pl.DataFrame:
    """Add ``total / days`` to each event's return on rel_days 0 to ``days - 1``.

    On day 0 the bump goes into the open-to-close return the event study uses.
    """
    index = {d: i for i, d in enumerate(sessions)}
    bumps: list[tuple[int, date]] = []
    for permno, entry in events.select("permno", "entry_date").iter_rows():
        i0 = index[entry]
        bumps += [(permno, sessions[i0 + k]) for k in range(days) if i0 + k < len(sessions)]
    bump = (
        pl.DataFrame(bumps, schema={"permno": pl.Int64, "date": pl.Date}, orient="row")
        .with_columns(pl.lit(total / days).alias("bump"))
        .group_by("permno", "date")
        .agg(pl.col("bump").sum())
    )
    return (
        prices.join(bump, on=["permno", "date"], how="left")
        .with_columns(pl.col("bump").fill_null(0.0))
        .with_columns(
            (pl.col("ret") + pl.col("bump")).alias("ret"),
            (pl.col("ret_open_close") + pl.col("bump")).alias("ret_open_close"),
        )
        .drop("bump")
    )


def random_events(
    seed: int, sessions: list[date], n_firms: int, n_events: int, first: int = 300
) -> pl.DataFrame:
    """Events on random firms and sessions after ``first`` (room for estimation), with scores."""
    rng = np.random.default_rng(seed)
    days = rng.integers(first, len(sessions) - 70, n_events)
    return pl.DataFrame(
        {
            "doc_id": [f"d{i}" for i in range(n_events)],
            "permno": rng.integers(0, n_firms, n_events),
            "entry_date": [sessions[d] for d in days],
            "score": rng.normal(size=n_events),
        }
    ).with_columns(pl.col("permno").cast(pl.Int64))
