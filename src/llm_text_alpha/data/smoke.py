"""``lta data smoke``: the M1 data-access check behind AC-W-04.

Fetches the recorded filings' SGML headers and checks each acceptance time, cross-checks
them with the submissions JSON (DQ-02), loads one month of prices from the configured
provider, and downloads Ken French's daily factors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import polars as pl
import structlog

from llm_text_alpha.config import SmokeConfig
from llm_text_alpha.corpus.timestamps import parse_submissions_acceptance
from llm_text_alpha.data import factors as french
from llm_text_alpha.data.edgar import SecClient, header_url, parse_header, submissions_url
from llm_text_alpha.data.prices import (
    ProviderUnavailableError,
    SharadarPrices,
    resolve_provider,
)
from llm_text_alpha.data.ratelimit import ByteThrottle
from llm_text_alpha.io import write_json_atomic, write_parquet_atomic, write_text_atomic
from llm_text_alpha.settings import Settings

log = structlog.get_logger(__name__)

RATE_TOLERANCE = 1e-3
"""Relative slack for clock resolution: send times are read just after the limiter's slot."""


@dataclass(frozen=True)
class Check:
    """One line of the smoke report. ``ok=None`` is informational and never fails the run."""

    name: str
    ok: bool | None
    detail: str


@dataclass
class SmokeReport:
    """Everything the smoke run measured."""

    run_id: str
    checks: list[Check] = field(default_factory=list)
    filings: list[dict[str, Any]] = field(default_factory=list)
    provider: str = ""
    provider_label: str = ""

    @property
    def passed(self) -> bool:
        """True when no check failed."""
        return all(check.ok is not False for check in self.checks)

    def add(self, name: str, ok: bool | None, detail: str) -> None:
        """Record a check and log it."""
        self.checks.append(Check(name, ok, detail))
        log.info("smoke_check", check=name, ok=ok, detail=detail)


def _sessions(start: date, end: date) -> list[date]:
    from eqpit import calendar

    return calendar.sessions(start, end)


def check_edgar(
    report: SmokeReport,
    settings: Settings,
    config: SmokeConfig,
    transport: httpx.BaseTransport | None = None,
) -> None:
    """Fetch every recorded header, compare acceptance times, and audit the client."""
    raw_dir = settings.data_dir / "raw" / "edgar" / "headers"
    matched = 0
    agreed = 0
    found = 0
    with SecClient.from_settings(settings, transport=transport) as client:
        submissions: dict[int, dict[str, str]] = {}
        for filing in config.filings:
            text = client.get_text(header_url(filing.cik, filing.accession))
            write_text_atomic(text, raw_dir / str(filing.cik) / f"{filing.accession}.hdr.sgml")
            header = parse_header(text)
            same = (
                header.accession == filing.accession
                and header.form == filing.form
                and list(header.items) == filing.items
                and header.acceptance_raw == filing.acceptance_raw
                and header.acceptance_ts_utc == filing.acceptance_ts_utc
            )
            matched += same
            if filing.cik not in submissions:
                payload = client.get(submissions_url(filing.cik)).json()
                recent = payload["filings"]["recent"]
                submissions[filing.cik] = dict(
                    zip(recent["accessionNumber"], recent["acceptanceDateTime"], strict=True)
                )
            json_value = submissions[filing.cik].get(filing.accession)
            json_ts = parse_submissions_acceptance(json_value) if json_value else None
            found += json_ts is not None
            agreed += json_ts == header.acceptance_ts_utc
            report.filings.append(
                {
                    "accession": filing.accession,
                    "cik": filing.cik,
                    "form": header.form,
                    "items": list(header.items),
                    "acceptance_raw": header.acceptance_raw,
                    "acceptance_ts_utc": header.acceptance_ts_utc.isoformat(),
                    "matches_fixture": same,
                    "submissions_ts_utc": json_ts.isoformat() if json_ts else None,
                }
            )
        stats = client.stats
    n = len(config.filings)
    report.add("edgar_acceptance_matches_fixture", matched == n, f"{matched} of {n} filings")
    report.add(
        "edgar_submissions_json_agrees",
        agreed == found if found else None,
        f"{agreed} of {found} found in the recent submissions page agree (DQ-02)",
    )
    report.add(
        "edgar_user_agent_on_every_request",
        stats.user_agent_ok,
        f"{stats.requests} requests, statuses {dict(sorted(stats.status_counts.items()))}",
    )
    rps = stats.measured_rps()
    report.add(
        "edgar_rate_within_limit",
        rps is None or rps <= settings.sec_max_rps * (1 + RATE_TOLERANCE),
        f"measured {rps:.2f} req/s against SEC_MAX_RPS={settings.sec_max_rps:g}"
        if rps is not None
        else "fewer than two requests",
    )


def check_prices(
    report: SmokeReport,
    settings: Settings,
    config: SmokeConfig,
    sharadar_transport: httpx.BaseTransport | None = None,
) -> None:
    """Load one month of prices and, when a Sharadar key is set, prove it reads the bundle."""
    start, end = config.month_bounds()
    expected = len(_sessions(start, end))
    try:
        provider = resolve_provider(settings, free_tickers=config.free_tickers)
        prices = provider.daily_prices(start, end, config.max_securities)
    except ProviderUnavailableError as exc:
        report.add("prices_one_month", False, str(exc))
    else:
        report.provider = provider.name
        report.provider_label = provider.label
        dates = prices["date"].n_unique()
        ok = prices.height > 0 and dates == expected and prices["open"].null_count() == 0
        label = f", labelled {provider.label!r}" if provider.label else ""
        report.add(
            "prices_one_month",
            ok,
            f"provider {provider.name}{label}: {prices.height} rows, "
            f"{prices['security_id'].n_unique()} securities, {dates} of {expected} sessions",
        )
    key = settings.sharadar_api_key.get_secret_value()
    if key:
        status = SharadarPrices(key, transport=sharadar_transport).check_key()
        report.add("sharadar_key", status == 200, f"one-row SEP query answered HTTP {status}")
    else:
        report.add("sharadar_key", None, "SHARADAR_API_KEY not set; needed before M3")


def _span(table: pl.DataFrame) -> str:
    first, last = table.select(pl.col("date").min().alias("first"), pl.col("date").max().alias("last")).row(0)
    return f"{first} to {last}"


def check_factors(
    report: SmokeReport,
    settings: Settings,
    config: SmokeConfig,
    transport: httpx.BaseTransport | None = None,
    throttle: ByteThrottle | None = None,
) -> pl.DataFrame:
    """Download the French daily archives, build ``factors_daily`` and check the month."""
    snapshot = french.download_factors(
        settings.data_dir / "raw" / "factors", transport=transport, throttle=throttle
    )
    table = french.build_factors_daily(snapshot)
    write_parquet_atomic(table, settings.data_dir / "processed" / "panels" / "factors_daily.parquet")
    start, end = config.month_bounds()
    month = table.filter(pl.col("date").is_between(start, end))
    expected = _sessions(start, end)
    ok = month["date"].to_list() == expected and month.null_count().sum_horizontal().item() == 0
    report.add(
        "factors_one_month",
        ok,
        f"{month.height} of {len(expected)} sessions; table {_span(table)}",
    )
    return table


def run_smoke(
    settings: Settings,
    config: SmokeConfig,
    run_id: str,
    sec_transport: httpx.BaseTransport | None = None,
    http_transport: httpx.BaseTransport | None = None,
    throttle: ByteThrottle | None = None,
) -> SmokeReport:
    """Run every smoke check and write ``processed/smoke/smoke-<run_id>.json``."""
    report = SmokeReport(run_id=run_id)
    check_edgar(report, settings, config, transport=sec_transport)
    check_prices(report, settings, config, sharadar_transport=http_transport)
    check_factors(report, settings, config, transport=http_transport, throttle=throttle)
    write_json_atomic(
        {
            "run_id": run_id,
            "passed": report.passed,
            "provider": report.provider,
            "provider_label": report.provider_label,
            "checks": [asdict(check) for check in report.checks],
            "filings": report.filings,
        },
        smoke_report_path(settings.data_dir, run_id),
    )
    return report


def smoke_report_path(data_dir: Path, run_id: str) -> Path:
    """Where a smoke run's JSON report goes."""
    return data_dir / "processed" / "smoke" / f"smoke-{run_id}.json"
