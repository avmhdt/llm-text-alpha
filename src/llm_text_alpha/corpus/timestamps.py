"""Publication timestamps normalized to UTC (DQ-01 to DQ-03).

EDGAR's SGML header gives ``ACCEPTANCE-DATETIME`` as ``YYYYMMDDHHMMSS`` in New York local
time. The submissions JSON gives ``acceptanceDateTime`` with a ``Z`` suffix, which is true UTC
(checked in equity-factor-pipeline on 2026-11-01; re-checked here per DQ-02). The header wins
on any disagreement.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

NEW_YORK = ZoneInfo("America/New_York")
_ACCEPTANCE = re.compile(r"^\d{14}$")


class TimestampError(ValueError):
    """A timestamp field is missing or malformed."""


def parse_acceptance(raw: str) -> datetime:
    """Convert a header ``ACCEPTANCE-DATETIME`` (New York local time) to an aware UTC datetime.

    DST is handled by ``zoneinfo``. In the repeated hour of a November fall-back, the first
    (daylight-time) reading is taken; EDGAR does not accept filings at that hour, so this
    choice cannot move a real filing.
    """
    value = raw.strip()
    if not _ACCEPTANCE.match(value):
        raise TimestampError(f"acceptance datetime {raw!r} is not YYYYMMDDHHMMSS")
    try:
        local = datetime.strptime(value, "%Y%m%d%H%M%S").replace(tzinfo=NEW_YORK)
    except ValueError as exc:
        raise TimestampError(f"acceptance datetime {raw!r} is not a valid time") from exc
    return local.astimezone(UTC)


def parse_submissions_acceptance(raw: str) -> datetime:
    """Convert a submissions-JSON ``acceptanceDateTime`` such as ``2026-11-02T21:30:23.000Z``."""
    value = raw.strip()
    if not value.endswith("Z"):
        raise TimestampError(f"submissions acceptance {raw!r} has no Z suffix")
    try:
        parsed = datetime.fromisoformat(value[:-1])
    except ValueError as exc:
        raise TimestampError(f"submissions acceptance {raw!r} is not ISO 8601") from exc
    return parsed.replace(tzinfo=UTC)
