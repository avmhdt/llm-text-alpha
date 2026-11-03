"""XNYS trading calendar: session dates with their open and close instants in UTC (DQ-04).

Holidays and early closes come from ``exchange_calendars``. Every session in the range used
here opens at 09:30 New York time, early-close days included.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from functools import lru_cache

import exchange_calendars
import numpy as np
from numpy.typing import NDArray

_EPOCH = date(1970, 1, 1)
CALENDAR_START = "1990-01-02"


class CalendarRangeError(ValueError):
    """A time falls outside the sessions the calendar knows."""


@dataclass(frozen=True)
class TradingCalendar:
    """Sessions as day numbers since 1970-01-01, with opens and closes in UTC nanoseconds."""

    name: str
    days: NDArray[np.int64]
    opens_ns: NDArray[np.int64]
    closes_ns: NDArray[np.int64]

    @property
    def first_session(self) -> date:
        """Earliest session."""
        return _date(int(self.days[0]))

    @property
    def last_session(self) -> date:
        """Latest session."""
        return _date(int(self.days[-1]))

    def sessions(self, start: date, end: date) -> list[date]:
        """Sessions in ``[start, end]``."""
        low = int(np.searchsorted(self.days, _day(start), side="left"))
        high = int(np.searchsorted(self.days, _day(end), side="right"))
        return [_date(int(day)) for day in self.days[low:high]]

    def is_session(self, value: date) -> bool:
        """True when the exchange trades on ``value``."""
        index = int(np.searchsorted(self.days, _day(value)))
        return index < len(self.days) and int(self.days[index]) == _day(value)

    def open_at(self, session: date) -> datetime:
        """Open of ``session`` as an aware UTC datetime."""
        index = int(np.searchsorted(self.days, _day(session)))
        if index >= len(self.days) or int(self.days[index]) != _day(session):
            raise CalendarRangeError(f"{session} is not an XNYS session")
        return _instant(int(self.opens_ns[index]))

    def first_open_after(self, instants_ns: NDArray[np.int64]) -> NDArray[np.int64]:
        """Index of the first session whose open is strictly later than each instant."""
        index = np.searchsorted(self.opens_ns, instants_ns, side="right").astype(np.int64)
        if len(index) and int(index.max()) >= len(self.days):
            raise CalendarRangeError("an event falls after the calendar's last open")
        if len(instants_ns) and int(instants_ns.min()) < int(self.opens_ns[0]):
            raise CalendarRangeError("an event falls before the calendar's first open")
        return index


def _day(value: date) -> int:
    return (value - _EPOCH).days


def _date(day: int) -> date:
    return date.fromordinal(_EPOCH.toordinal() + day)


def _instant(ns: int) -> datetime:
    seconds, rest = divmod(ns, 1_000_000_000)
    return datetime.fromtimestamp(seconds, tz=UTC).replace(microsecond=rest // 1000)


@lru_cache(maxsize=2)
def xnys(start: str = CALENDAR_START) -> TradingCalendar:
    """The NYSE calendar from ``start`` to ``exchange_calendars``' default end."""
    calendar = exchange_calendars.get_calendar("XNYS", start=start)
    days = calendar.sessions.to_numpy().astype("datetime64[D]").astype(np.int64)
    opens = calendar.opens.dt.tz_convert("UTC").dt.tz_localize(None)
    closes = calendar.closes.dt.tz_convert("UTC").dt.tz_localize(None)
    return TradingCalendar(
        name="XNYS",
        days=days,
        opens_ns=opens.to_numpy().astype("datetime64[ns]").astype(np.int64),
        closes_ns=closes.to_numpy().astype("datetime64[ns]").astype(np.int64),
    )
