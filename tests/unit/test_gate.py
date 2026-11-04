from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest

from llm_text_alpha.corpus.calendar import CalendarRangeError, xnys
from llm_text_alpha.corpus.gate import (
    SIGNAL_COLUMNS,
    LookaheadError,
    apply_gate,
    assert_no_lookahead,
    entry_for_event,
)

CAL = xnys()
TS = pl.Datetime("ns", "UTC")


def docs(rows: list[tuple[str, int | None, datetime, bool]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "doc_id": pl.String,
            "permno": pl.Int64,
            "event_ts_utc": TS,
            "excluded": pl.Boolean,
        },
        orient="row",
    )


def scores(ids: list[str], values: list[float]) -> pl.DataFrame:
    n = len(ids)
    return pl.DataFrame(
        {
            "doc_id": ids,
            "scorer": ["lm"] * n,
            "model_id": ["lm-x"] * n,
            "masked": [False] * n,
            "score": values,
        }
    )


NO_CONTROLS = pl.DataFrame(schema={"permno": pl.Int64, "asof": pl.Date})


def test_calendar_basics() -> None:
    assert CAL.first_session == date(1990, 1, 2)
    assert CAL.last_session > date(2026, 12, 31)
    assert CAL.is_session(date(2024, 3, 15)) and not CAL.is_session(date(2024, 3, 16))
    assert CAL.open_at(date(2024, 7, 3)) == datetime(2024, 7, 3, 13, 30, tzinfo=UTC)
    assert CAL.sessions(date(2016, 3, 24), date(2016, 3, 28)) == [
        date(2016, 3, 24),
        date(2016, 3, 28),
    ]
    with pytest.raises(CalendarRangeError):
        CAL.open_at(date(2024, 3, 16))


def test_events_outside_the_calendar() -> None:
    with pytest.raises(CalendarRangeError, match="before"):
        entry_for_event(datetime(1980, 1, 1, tzinfo=UTC), CAL)
    with pytest.raises(CalendarRangeError, match="after"):
        entry_for_event(datetime(2040, 1, 1, tzinfo=UTC), CAL)


def test_bad_inputs() -> None:
    with pytest.raises(ValueError, match="aware"):
        entry_for_event(datetime(2024, 3, 14, 12), CAL)
    with pytest.raises(ValueError, match="negative"):
        entry_for_event(datetime(2024, 3, 14, 12, tzinfo=UTC), CAL, timedelta(minutes=-1))


def test_apply_gate_combines_filters_and_dates_controls() -> None:
    documents = docs(
        [
            ("a", 1, datetime(2024, 3, 14, 20, 30, tzinfo=UTC), False),  # after close
            ("b", 1, datetime(2024, 3, 15, 11, 0, tzinfo=UTC), False),  # pre-open next day
            ("c", 2, datetime(2024, 3, 14, 20, 30, tzinfo=UTC), True),  # amendment
            ("d", None, datetime(2024, 3, 14, 20, 30, tzinfo=UTC), False),  # unlinked
            ("e", 2, datetime(2024, 3, 15, 15, 0, tzinfo=UTC), False),  # intraday
        ]
    )
    controls = pl.DataFrame(
        {
            "permno": [1, 1, 2],
            "asof": [date(2024, 3, 14), date(2024, 3, 15), date(2024, 3, 18)],
            "size_log": [10.0, 11.0, 12.0],
            "sue": [0.5, 0.6, 0.7],
        }
    )
    signals = apply_gate(
        scores(list("abcde"), [0.2, 0.4, 9.0, 9.0, -0.1]), documents, CAL, controls
    )
    assert signals.columns == list(SIGNAL_COLUMNS)
    one = signals.row(0, named=True)
    assert one["doc_id"] == "a" and one["n_docs"] == 2
    assert one["score"] == pytest.approx(0.3)
    assert one["entry_date"] == date(2024, 3, 15)
    assert one["controls_asof"] == date(2024, 3, 14) and one["size_log"] == 10.0
    assert one["mom_12_2"] is None and one["score_z"] is None
    two = signals.row(1, named=True)
    assert two["doc_id"] == "e" and two["entry_date"] == date(2024, 3, 18)
    assert two["controls_asof"] is None
    assert signals.height == 2


def test_documents_without_an_excluded_column() -> None:
    documents = docs([("a", 1, datetime(2024, 3, 14, 12, tzinfo=UTC), False)]).drop("excluded")
    signals = apply_gate(scores(["a"], [0.1]), documents, CAL, NO_CONTROLS)
    assert signals["entry_date"].to_list() == [date(2024, 3, 14)]


def test_injected_violations_raise() -> None:
    documents = docs([("a", 1, datetime(2024, 3, 14, 12, tzinfo=UTC), False)])
    signals = apply_gate(scores(["a"], [0.1]), documents, CAL, NO_CONTROLS)
    assert_no_lookahead(signals)
    early = signals.with_columns(pl.col("event_ts_utc") + pl.duration(hours=2))
    with pytest.raises(LookaheadError, match="at or before"):
        assert_no_lookahead(early)
    tight = signals.with_columns(pl.col("event_ts_utc") + pl.duration(minutes=86))
    with pytest.raises(LookaheadError):
        assert_no_lookahead(tight, timedelta(minutes=5))
    shifted = signals.with_columns(pl.col("entry_date") - pl.duration(days=1))
    with pytest.raises(LookaheadError, match="off entry_date"):
        assert_no_lookahead(shifted)
    stale = signals.with_columns(pl.col("entry_date").alias("controls_asof"))
    with pytest.raises(LookaheadError, match="controls"):
        assert_no_lookahead(stale)
    missing = signals.with_columns(pl.lit(None, TS).alias("entry_ts_utc"))
    with pytest.raises(LookaheadError):
        assert_no_lookahead(missing)


def test_null_event_times_are_refused() -> None:
    documents = pl.DataFrame(
        {"doc_id": ["a"], "permno": [1], "event_ts_utc": [None]},
        schema_overrides={"event_ts_utc": TS},
    )
    with pytest.raises(ValueError, match="nulls"):
        apply_gate(scores(["a"], [0.1]), documents, CAL, NO_CONTROLS)
