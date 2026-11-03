from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from llm_text_alpha.data import fundamentals, prices
from llm_text_alpha.signals.build import daily_controls

TS = pl.Datetime("ns", "UTC")


def eps_series(values: list[float], start: date = date(2014, 3, 31)) -> pl.DataFrame:
    ends = []
    year, month = start.year, start.month
    for _ in values:
        ends.append(date(year, month, 28))
        month += 3
        if month > 12:
            month -= 12
            year += 1
    return pl.DataFrame(
        {
            "company_id": ["CIK:1"] * len(values),
            "cik": [1] * len(values),
            "fqtr_end": ends,
            "eps": values,
            "eps_available_at": [
                datetime(e.year, e.month, e.day, 21, tzinfo=UTC) + timedelta(days=35) for e in ends
            ],
            "eps_form": ["10-Q"] * len(values),
        },
        schema_overrides={"eps_available_at": TS},
    )


def test_seasonal_sue() -> None:
    values = [1.0, 1.1, 1.2, 1.3, 1.1, 1.3, 1.2, 1.6, 1.0, 1.5, 1.4, 1.5, 2.0, 1.6]
    out = fundamentals.seasonal_sue(eps_series(values))
    diffs = np.array(values[4:]) - np.array(values[:-4])
    assert out["diff4"].to_list()[4:] == pytest.approx(list(diffs))
    assert out["diff4"].to_list()[:4] == [None] * 4
    sue = out["sue"].to_list()
    assert all(v is None for v in sue[:10])
    expected = diffs[6] / np.std(diffs[:6], ddof=1)
    assert sue[10] == pytest.approx(expected)
    last = diffs[9] / np.std(diffs[1:9], ddof=1)
    assert sue[13] == pytest.approx(last)
    assert (out["sue_method"] == "srw").all()


def test_sue_needs_a_quarter_one_year_back() -> None:
    frame = eps_series([1.0, 2.0, 3.0, 4.0, 5.0]).with_columns(
        pl.when(pl.int_range(pl.len()) == 4)
        .then(pl.col("fqtr_end") + pl.duration(days=60))
        .otherwise(pl.col("fqtr_end"))
        .alias("fqtr_end")
    )
    assert fundamentals.seasonal_sue(frame)["diff4"].to_list()[4] is None


def test_link_announcements() -> None:
    earnings = eps_series([1.0, 1.1])
    docs = pl.DataFrame(
        {
            "doc_id": ["early", "release", "other", "late"],
            "cik": [1, 1, 2, 1],
            "event_ts_utc": [
                datetime(2014, 3, 20, tzinfo=UTC),
                datetime(2014, 4, 25, 20, tzinfo=UTC),
                datetime(2014, 4, 25, 20, tzinfo=UTC),
                datetime(2014, 9, 1, tzinfo=UTC),
            ],
        },
        schema_overrides={"event_ts_utc": TS},
    )
    linked = fundamentals.link_announcements(earnings, docs)
    first, second = linked.iter_rows(named=True)
    assert first["announce_doc_id"] == "release" and first["announce_source"] == "8k_item202"
    assert first["announce_ts_utc"] == datetime(2014, 4, 25, 20, tzinfo=UTC)
    assert second["announce_source"] == "eps_filing"
    assert second["announce_ts_utc"] == second["eps_available_at"]


def test_first_reported_from_the_store(efp_root: Path) -> None:
    from eqpit.store import PitStore

    store = PitStore.open(efp_root / "processed" / "pit")
    months = store.month_ends(date(2015, 6, 1), date(2016, 12, 31))
    first = fundamentals.first_reported(store, months, ["eps_diluted", "book_equity"])
    key = ["company_id", "item", "period_end", "period_type"]
    assert first.select(key).is_duplicated().sum() == 0
    eps = fundamentals.quarterly_eps(first)
    assert eps.height > 0 and set(eps.columns) >= {"fqtr_end", "eps", "eps_available_at"}
    book = fundamentals.book_equity(first)
    assert (book["book_equity"] > 0).all()
    universe = prices.universe_monthly(store, date(2016, 1, 1), date(2016, 3, 31), 10)
    assert universe.group_by("month_end").len()["len"].to_list() == [10, 10, 10]
    links = prices.link_table(store, months)
    assert links["valid_from"].le(links["valid_to"]).all()
    panel = prices.prices_daily(
        store, date(2016, 6, 30), 22, universe["security_id"].unique().to_list()
    )
    one = panel.row(5, named=True)
    assert one["ret_open_close"] == pytest.approx(one["close"] / one["open"] - 1)
    assert (1 + one["ret"]) == pytest.approx(
        (1 + one["ret_open_close"]) * (1 + one["ret_close_open"])
    )
    with pytest.raises(prices.ProviderUnavailableError):
        prices.universe_monthly(store, date(2016, 1, 2), date(2016, 1, 3), 10)


def test_daily_controls_are_known_by_the_close() -> None:
    days = [date(2016, 1, 4) + timedelta(days=i) for i in range(400)]
    panel = pl.DataFrame(
        {
            "permno": [7] * 400,
            "company_id": ["CIK:1"] * 400,
            "date": days,
            "ret": [0.001] * 400,
            "mcap": [1e9] * 400,
        }
    )
    book = pl.DataFrame(
        {
            "company_id": ["CIK:1"],
            "available_at": [datetime(2016, 1, 10, 21, 30, tzinfo=UTC)],
            "book_equity": [5e8],
        },
        schema_overrides={"available_at": TS},
    )
    sue = pl.DataFrame(
        {
            "company_id": ["CIK:1", "CIK:1"],
            "eps_available_at": [
                datetime(2016, 1, 10, 20, 0, tzinfo=UTC),
                datetime(2016, 2, 1, tzinfo=UTC),
            ],
            "sue": [1.5, None],
        },
        schema_overrides={"eps_available_at": TS},
    )
    controls = daily_controls(panel, book, sue).sort("asof")
    by_day = {row["asof"]: row for row in controls.iter_rows(named=True)}
    assert by_day[date(2016, 1, 9)]["sue"] is None
    assert by_day[date(2016, 1, 10)]["sue"] == 1.5
    assert by_day[date(2016, 1, 10)]["bm"] is None
    assert by_day[date(2016, 1, 11)]["bm"] == 0.5
    assert by_day[date(2016, 3, 1)]["sue"] == 1.5
    assert by_day[date(2016, 1, 4)]["mom_12_2"] is None
    assert by_day[days[-1]]["mom_12_2"] == pytest.approx(1.001**231 - 1)
    assert by_day[days[0]]["size_log"] == pytest.approx(np.log(1e9))
