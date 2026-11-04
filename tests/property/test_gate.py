"""AC-W-01, property part: every signal enters strictly after publication plus the buffer."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone

import numpy as np
import polars as pl
from hypothesis import example, given, settings
from hypothesis import strategies as st

from llm_text_alpha.corpus.calendar import xnys
from llm_text_alpha.corpus.gate import apply_gate, assert_no_lookahead, entries, entry_for_event

CALENDAR = xnys()
FIRST = datetime(1990, 1, 3, tzinfo=UTC)
LAST = datetime(2027, 10, 29, tzinfo=UTC)

offsets = st.integers(min_value=-12 * 60, max_value=14 * 60).map(
    lambda minutes: timezone(timedelta(minutes=minutes))
)
instants = st.datetimes(
    min_value=FIRST.replace(tzinfo=None), max_value=LAST.replace(tzinfo=None)
).map(lambda value: value.replace(microsecond=0, tzinfo=UTC))
buffers = st.integers(min_value=0, max_value=6 * 60 * 60).map(lambda s: timedelta(seconds=s))


def _check(event: datetime, buffer: timedelta) -> None:
    entry_date, entry_ts = entry_for_event(event, CALENDAR, buffer)
    assert entry_ts > event + buffer
    assert CALENDAR.is_session(entry_date)
    assert entry_ts == CALENDAR.open_at(entry_date)
    earlier = CALENDAR.sessions(date(1990, 1, 1), entry_date - timedelta(days=1))
    if earlier:
        assert CALENDAR.open_at(earlier[-1]) <= event + buffer


@settings(max_examples=10_000, deadline=None)
@given(instant=instants, zone=offsets, buffer=buffers)
@example(  # 09:31 ET: next day's open
    instant=datetime(2024, 3, 14, 13, 31, tzinfo=UTC), zone=UTC, buffer=timedelta(minutes=5)
)
@example(  # 08:00 ET: same day's open
    instant=datetime(2024, 3, 14, 12, 0, tzinfo=UTC), zone=UTC, buffer=timedelta(minutes=5)
)
@example(  # 09:25 ET exactly at open minus buffer: next day
    instant=datetime(2024, 3, 14, 13, 25, tzinfo=UTC), zone=UTC, buffer=timedelta(minutes=5)
)
@example(  # Good Friday 2016-03-25, 15:34 EDT: Monday's open
    instant=datetime(2016, 3, 25, 19, 34, 12, tzinfo=UTC), zone=UTC, buffer=timedelta(minutes=5)
)
@example(  # after the 2018-07-03 early close: 2018-07-05 open
    instant=datetime(2018, 7, 3, 18, 31, 40, tzinfo=UTC), zone=UTC, buffer=timedelta(minutes=5)
)
@example(  # Sunday of spring-forward 2026-03-08
    instant=datetime(2026, 3, 8, 7, 30, tzinfo=UTC), zone=UTC, buffer=timedelta(0)
)
@example(  # Sunday of fall-back 2026-11-01, read in a far-east zone
    instant=datetime(2026, 11, 1, 6, 30, tzinfo=UTC),
    zone=timezone(timedelta(hours=14)),
    buffer=timedelta(minutes=5),
)
def test_signal_precedes_trade(instant: datetime, zone: timezone, buffer: timedelta) -> None:
    _check(instant.astimezone(zone), buffer)


def test_named_edges() -> None:
    cases = {
        datetime(2024, 3, 14, 13, 31, tzinfo=UTC): date(2024, 3, 15),
        datetime(2024, 3, 14, 12, 0, tzinfo=UTC): date(2024, 3, 14),
        datetime(2024, 3, 14, 13, 25, tzinfo=UTC): date(2024, 3, 15),
        datetime(2024, 3, 14, 13, 24, 59, tzinfo=UTC): date(2024, 3, 14),
        datetime(2016, 3, 25, 19, 34, 12, tzinfo=UTC): date(2016, 3, 28),
        datetime(2018, 7, 3, 18, 31, 40, tzinfo=UTC): date(2018, 7, 5),
        datetime(2026, 11, 2, 21, 30, 23, tzinfo=UTC): date(2026, 11, 3),
        datetime(2026, 3, 9, 13, 0, tzinfo=UTC): date(2026, 3, 9),  # 09:00 EDT
        datetime(2026, 3, 6, 14, 0, tzinfo=UTC): date(2026, 3, 6),  # 09:00 EST Friday
    }
    for event, expected in cases.items():
        assert entry_for_event(event, CALENDAR)[0] == expected, event


def test_vectorized_matches_scalar_on_a_large_random_batch() -> None:
    rng = np.random.default_rng(20261030)
    low = int(FIRST.timestamp())
    high = int(LAST.timestamp())
    seconds = rng.integers(low, high, size=100_000)
    events = pl.Series("event_ts_utc", seconds * 1_000_000_000).cast(pl.Datetime("ns", "UTC"))
    buffer = timedelta(minutes=5)
    entry_date, entry_ts = entries(events, CALENDAR, buffer)
    frame = pl.DataFrame([events, entry_date, entry_ts])
    assert_no_lookahead(frame, buffer)
    for i in rng.choice(len(events), size=200, replace=False).tolist():
        event = datetime.fromtimestamp(int(seconds[i]), tz=UTC)
        assert entry_for_event(event, CALENDAR, buffer) == (entry_date[i], entry_ts[i])


@settings(max_examples=300, deadline=None)
@given(
    events=st.lists(instants, min_size=1, max_size=20),
    control_days=st.lists(
        st.dates(min_value=date(1990, 1, 1), max_value=date(2027, 10, 29)), min_size=1, max_size=40
    ),
)
def test_controls_dated_before_entry(events: list[datetime], control_days: list[date]) -> None:
    n = len(events)
    documents = pl.DataFrame(
        {
            "doc_id": [f"d{i}" for i in range(n)],
            "permno": [i % 3 for i in range(n)],
            "event_ts_utc": events,
        },
        schema_overrides={"event_ts_utc": pl.Datetime("ns", "UTC")},
    )
    scores = pl.DataFrame(
        {
            "doc_id": documents["doc_id"],
            "scorer": ["lm"] * n,
            "model_id": ["lm-test"] * n,
            "masked": [False] * n,
            "score": [float(i) for i in range(n)],
        }
    )
    controls = pl.DataFrame(
        {
            "permno": [i % 3 for i in range(len(control_days))],
            "asof": control_days,
            "size_log": [float(i) for i in range(len(control_days))],
        }
    )
    signals = apply_gate(scores, documents, CALENDAR, controls, timedelta(minutes=5))
    dated = signals.filter(pl.col("controls_asof").is_not_null())
    assert (dated["controls_asof"] < dated["entry_date"]).all()
    for row in signals.iter_rows(named=True):
        earlier = [
            d
            for d, p in zip(control_days, controls["permno"], strict=True)
            if p == row["permno"] and d < row["entry_date"]
        ]
        assert row["controls_asof"] == (max(earlier) if earlier else None)
