"""Trailing standardization of gated scores, and the dated control table (DQ-08, DQ-10).

``score_z`` and ``quintile`` use only events of the same scorer, model and masking whose
entry is strictly earlier than the current event's and within the last 252 sessions. A
later score can never move an earlier ``score_z``.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from llm_text_alpha.corpus.calendar import TradingCalendar

TRAILING_SESSIONS = 252
MIN_TRAILING_EVENTS = 50
GROUP = ("scorer", "model_id", "masked")
MOM_SKIP_SESSIONS = 21
MOM_TOTAL_SESSIONS = 252


def _session_index(dates: pl.Series, calendar: TradingCalendar) -> np.ndarray:
    days = dates.cast(pl.Int64).to_numpy()
    index = np.searchsorted(calendar.days, days)
    if (calendar.days[np.minimum(index, len(calendar.days) - 1)] != days).any():
        raise ValueError("every entry_date must be a session")
    return index.astype(np.int64)


def _trailing_group(
    scores: np.ndarray, sessions: np.ndarray, window: int, min_events: int, n_quantiles: int
) -> tuple[np.ndarray, np.ndarray]:
    z = np.full(len(scores), np.nan)
    quantile = np.zeros(len(scores), dtype=np.int64)
    for session in np.unique(sessions):
        current = np.flatnonzero(sessions == session)
        past = np.flatnonzero((sessions < session) & (sessions >= session - window))
        if len(past) < min_events:
            continue
        history = np.sort(scores[past])
        mean = float(history.mean())
        std = float(history.std(ddof=1))
        if std > 0 and math.isfinite(std):
            z[current] = (scores[current] - mean) / std
        rank = np.searchsorted(history, scores[current], side="left") / len(history)
        quantile[current] = (
            np.minimum(np.floor(rank * n_quantiles).astype(np.int64), n_quantiles - 1) + 1
        )
    return z, quantile


def standardize(
    signals: pl.DataFrame,
    calendar: TradingCalendar,
    window: int = TRAILING_SESSIONS,
    min_events: int = MIN_TRAILING_EVENTS,
    n_quantiles: int = 5,
) -> pl.DataFrame:
    """Fill ``score_z`` and add ``quintile`` (1 to ``n_quantiles``; 0 before warm-up).

    The trailing set for an event is every event of its group entering in the ``window``
    sessions before its entry date; events entering the same day never see each other.
    Quintile breakpoints come from the same set. Fewer than ``min_events`` gives null
    ``score_z`` and quintile 0.
    """
    parts: list[pl.DataFrame] = []
    ordered = signals.with_columns(
        pl.Series("_session", _session_index(signals["entry_date"], calendar))
    ).sort("entry_date", "permno", "doc_id")
    for _, part in ordered.group_by(*GROUP, maintain_order=True):
        z, quantile = _trailing_group(
            part["score"].to_numpy().astype(np.float64),
            part["_session"].to_numpy(),
            window,
            min_events,
            n_quantiles,
        )
        parts.append(
            part.with_columns(
                pl.Series("score_z", z).fill_nan(None),
                pl.Series("quintile", quantile).cast(pl.Int8),
            )
        )
    if not parts:
        return signals.with_columns(pl.lit(None, pl.Int8).alias("quintile"))
    return pl.concat(parts).drop("_session").sort("entry_date", "permno", *GROUP)


def daily_controls(prices: pl.DataFrame, book: pl.DataFrame, sue: pl.DataFrame) -> pl.DataFrame:
    """Controls known at the close of each date, per ``permno`` (DQ-08).

    Args:
        prices: ``permno, company_id, date, ret, mcap``.
        book: ``company_id, available_at, book_equity``.
        sue: ``company_id, eps_available_at, sue``: a quarter's SUE once its EPS is public.

    ``size_log`` is the log market cap; ``mom_12_2`` compounds returns over sessions 252
    to 22 before the date, the daily form of months t-12 to t-2; ``bm`` is the latest
    public book equity over market cap; ``sue`` is the latest SUE whose EPS was public by
    the close. Each value is stamped with ``asof`` = its date, and the gate takes the row
    dated before entry.
    """
    ordered = prices.sort("permno", "date").with_columns(
        pl.col("ret").fill_null(0.0).log1p().alias("_log_ret")
    )
    momentum = (
        pl.col("_log_ret")
        .rolling_sum(window_size=MOM_TOTAL_SESSIONS - MOM_SKIP_SESSIONS, min_samples=200)
        .shift(MOM_SKIP_SESSIONS)
        .over("permno")
    )
    close_ts = (
        (
            pl.col("date").cast(pl.Datetime("ns")).dt.replace_time_zone("America/New_York")
            + pl.duration(hours=16)
        )
        .dt.convert_time_zone("UTC")
        .cast(pl.Datetime("ns", "UTC"))
    )
    base = ordered.with_columns(
        pl.when(pl.col("mcap") > 0).then(pl.col("mcap").log()).alias("size_log"),
        (momentum.exp() - 1.0).alias("mom_12_2"),
        close_ts.alias("_close_ts"),
    ).sort("_close_ts")
    with_book = base.join_asof(
        book.sort("available_at").select(
            "company_id", pl.col("available_at").cast(pl.Datetime("ns", "UTC")), "book_equity"
        ),
        left_on="_close_ts",
        right_on="available_at",
        by="company_id",
        strategy="backward",
    )
    with_sue = with_book.join_asof(
        sue.filter(pl.col("sue").is_not_null())
        .sort("eps_available_at")
        .select("company_id", pl.col("eps_available_at").cast(pl.Datetime("ns", "UTC")), "sue"),
        left_on="_close_ts",
        right_on="eps_available_at",
        by="company_id",
        strategy="backward",
    )
    return with_sue.select(
        "permno",
        pl.col("date").alias("asof"),
        "size_log",
        "mom_12_2",
        "sue",
        pl.when(pl.col("mcap") > 0).then(pl.col("book_equity") / pl.col("mcap")).alias("bm"),
    ).sort("permno", "asof")
