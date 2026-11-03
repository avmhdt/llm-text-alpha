from __future__ import annotations

from datetime import UTC, datetime

import pytest

from llm_text_alpha.corpus.timestamps import (
    TimestampError,
    parse_acceptance,
    parse_submissions_acceptance,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("20160126163106", datetime(2016, 1, 26, 21, 31, 6, tzinfo=UTC)),  # EST
        ("20180703143140", datetime(2018, 7, 3, 18, 31, 40, tzinfo=UTC)),  # EDT
        ("20240101000000", datetime(2024, 1, 1, 5, 0, 0, tzinfo=UTC)),  # midnight
        ("20231231235959", datetime(2024, 1, 1, 4, 59, 59, tzinfo=UTC)),  # crosses the date
        ("20260308010000", datetime(2026, 3, 8, 6, 0, 0, tzinfo=UTC)),  # before spring-forward
        ("20260308090000", datetime(2026, 3, 8, 13, 0, 0, tzinfo=UTC)),  # after spring-forward
        ("20261101090000", datetime(2026, 11, 1, 14, 0, 0, tzinfo=UTC)),  # after fall-back
        ("20261031163000", datetime(2026, 10, 31, 20, 30, 0, tzinfo=UTC)),  # day before fall-back
    ],
)
def test_acceptance_to_utc(raw: str, expected: datetime) -> None:
    assert parse_acceptance(raw) == expected


def test_repeated_fall_back_hour_takes_daylight_time() -> None:
    assert parse_acceptance("20261101013000") == datetime(2026, 11, 1, 5, 30, tzinfo=UTC)


@pytest.mark.parametrize("raw", ["", "2016012616310", "2016-01-26 16:31", "20161326163106"])
def test_malformed_acceptance(raw: str) -> None:
    with pytest.raises(TimestampError):
        parse_acceptance(raw)


def test_submissions_value_is_utc() -> None:
    parsed = parse_submissions_acceptance("2026-11-02T21:30:23.000Z")
    assert parsed == datetime(2026, 11, 2, 21, 30, 23, tzinfo=UTC)
    assert parsed == parse_acceptance("20261102163023")


@pytest.mark.parametrize("raw", ["2026-11-02T21:30:23", "not-a-timeZ"])
def test_submissions_value_must_be_utc_iso(raw: str) -> None:
    with pytest.raises(TimestampError):
        parse_submissions_acceptance(raw)
