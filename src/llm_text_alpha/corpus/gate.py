"""The timestamp gate: the only place where text-derived data meets prices (FR-07, DQ-04).

A document published at ``event_ts_utc`` enters at the open of the first XNYS session whose
open is strictly later than ``event_ts_utc + buffer``. Controls are taken as of the latest
date strictly before the entry date. :func:`assert_no_lookahead` re-checks both on any
signals table and raises :class:`LookaheadError` on a single violation.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl

from llm_text_alpha.corpus.calendar import TradingCalendar, _date, _instant

DEFAULT_BUFFER = timedelta(minutes=5)
CONTROL_COLUMNS = ("size_log", "mom_12_2", "sue", "bm")
SCORE_KEY = ("doc_id", "scorer", "model_id", "masked")
SIGNAL_COLUMNS = (
    "doc_id",
    "permno",
    "scorer",
    "model_id",
    "masked",
    "event_ts_utc",
    "entry_date",
    "entry_ts_utc",
    "score",
    "score_z",
    *CONTROL_COLUMNS,
    "controls_asof",
    "n_docs",
)
_TS = pl.Datetime("ns", "UTC")


class LookaheadError(RuntimeError):
    """A signal would trade on information not yet public, or a control is dated too late."""


def _ns(value: datetime) -> int:
    if value.tzinfo is None:
        raise ValueError("event timestamps must be timezone-aware")
    delta = value.astimezone(UTC) - datetime(1970, 1, 1, tzinfo=UTC)
    return (delta.days * 86_400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1000


def _buffer_ns(buffer: timedelta) -> int:
    if buffer < timedelta(0):
        raise ValueError("the buffer cannot be negative")
    return int(buffer / timedelta(microseconds=1)) * 1000


def entry_for_event(
    event_ts_utc: datetime,
    calendar: TradingCalendar,
    buffer: timedelta = DEFAULT_BUFFER,
) -> tuple[date, datetime]:
    """Entry session and its open (UTC) for one event; ``event_ts_utc`` must be aware."""
    instant = np.array([_ns(event_ts_utc) + _buffer_ns(buffer)], dtype=np.int64)
    index = int(calendar.first_open_after(instant)[0])
    return _date(int(calendar.days[index])), _instant(int(calendar.opens_ns[index]))


def entries(
    event_ts: pl.Series, calendar: TradingCalendar, buffer: timedelta
) -> tuple[pl.Series, pl.Series]:
    """Vectorized :func:`entry_for_event`: ``entry_date`` and ``entry_ts_utc`` series."""
    if event_ts.null_count():
        raise ValueError("event_ts_utc has nulls")
    instants = event_ts.cast(_TS).dt.epoch("ns").to_numpy().astype(np.int64)
    index = calendar.first_open_after(instants + _buffer_ns(buffer))
    entry_date = pl.Series("entry_date", calendar.days[index].astype("datetime64[D]"))
    entry_ts = pl.Series("entry_ts_utc", calendar.opens_ns[index]).cast(pl.Datetime("ns"))
    return entry_date.cast(pl.Date), entry_ts.dt.replace_time_zone("UTC")


def apply_gate(
    scores: pl.DataFrame,
    documents: pl.DataFrame,
    calendar: TradingCalendar,
    controls: pl.DataFrame,
    buffer: timedelta = DEFAULT_BUFFER,
) -> pl.DataFrame:
    """Build ``signals`` from scores, document timestamps and dated controls.

    Args:
        scores: ``doc_id, scorer, model_id, masked, score``.
        documents: ``doc_id, permno, event_ts_utc``, plus optional boolean ``excluded``
            (amendments and duplicate texts, DQ-05). Unlinked documents (null ``permno``)
            and excluded ones never become signals.
        calendar: trading calendar.
        controls: ``permno, asof`` (date) and the control columns, each value known at the
            close of ``asof``. A signal takes the latest row with ``asof < entry_date``.
        buffer: minimum gap between publication and the entry open.

    Several documents of one firm entering on the same date become one event: the mean
    score, the earliest ``event_ts_utc`` and that document's ``doc_id`` (DQ-05).
    ``score_z`` is filled later by trailing standardization and is null here.
    """
    docs = documents.select("doc_id", "permno", "event_ts_utc", *_excluded(documents))
    joined = (
        scores.select(*SCORE_KEY, "score")
        .join(docs, on="doc_id", how="inner")
        .filter(pl.col("permno").is_not_null() & ~pl.col("excluded"))
        .drop("excluded")
    )
    entry_date, entry_ts = entries(joined["event_ts_utc"], calendar, buffer)
    gated = joined.with_columns(entry_date, entry_ts)
    events = (
        gated.sort("event_ts_utc", "doc_id")
        .group_by("permno", "scorer", "model_id", "masked", "entry_date", maintain_order=True)
        .agg(
            pl.col("doc_id").first(),
            pl.col("event_ts_utc").first(),
            pl.col("entry_ts_utc").first(),
            pl.col("score").sort().mean(),
            pl.len().alias("n_docs"),
        )
    )
    dated = controls.select(
        pl.col("permno"),
        pl.col("asof").alias("controls_asof"),
        *[
            pl.col(c) if c in controls.columns else pl.lit(None, pl.Float64).alias(c)
            for c in CONTROL_COLUMNS
        ],
    ).sort("controls_asof")
    lookup = events.with_columns(
        (pl.col("entry_date") - pl.duration(days=1)).alias("_control_key")
    ).sort("_control_key")
    signals = (
        lookup.join_asof(
            dated,
            left_on="_control_key",
            right_on="controls_asof",
            by="permno",
            strategy="backward",
        )
        .drop("_control_key")
        .with_columns(pl.lit(None, pl.Float64).alias("score_z"))
        .select(SIGNAL_COLUMNS)
        .sort("entry_date", "permno", "scorer", "model_id", "masked")
    )
    assert_no_lookahead(signals, buffer)
    return signals


def _excluded(documents: pl.DataFrame) -> list[pl.Expr]:
    if "excluded" in documents.columns:
        return [pl.col("excluded").fill_null(False)]
    return [pl.lit(False).alias("excluded")]


def assert_no_lookahead(signals: pl.DataFrame, buffer: timedelta = DEFAULT_BUFFER) -> None:
    """Raise :class:`LookaheadError` unless every row trades strictly after publication.

    Checks ``entry_ts_utc > event_ts_utc + buffer``, that ``entry_ts_utc`` falls on
    ``entry_date``, and that ``controls_asof`` (when present) is before ``entry_date``.
    """
    late = signals.filter(
        pl.col("entry_ts_utc").is_null()
        | pl.col("event_ts_utc").is_null()
        | (pl.col("entry_ts_utc") <= pl.col("event_ts_utc") + pl.duration(microseconds=_us(buffer)))
    )
    if late.height:
        raise LookaheadError(f"{late.height} signals enter at or before publication + buffer")
    shifted = signals.filter(
        pl.col("entry_ts_utc").dt.convert_time_zone("America/New_York").dt.date()
        != pl.col("entry_date")
    )
    if shifted.height:
        raise LookaheadError(f"{shifted.height} signals have entry_ts_utc off entry_date")
    if "controls_asof" in signals.columns:
        stale = signals.filter(pl.col("controls_asof") >= pl.col("entry_date"))
        if stale.height:
            raise LookaheadError(f"{stale.height} signals use controls dated on or after entry")


def _us(buffer: timedelta) -> int:
    return _buffer_ns(buffer) // 1000
