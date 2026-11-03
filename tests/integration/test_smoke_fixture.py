"""``lta data smoke`` against recorded synthetic responses: no network (AC-W-04)."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from pathlib import Path

import httpx
import pytest

from llm_text_alpha.config import SmokeConfig, SmokeFiling, load_smoke
from llm_text_alpha.data import factors, prices
from llm_text_alpha.data.ratelimit import ByteThrottle
from llm_text_alpha.data.smoke import run_smoke, smoke_report_path
from llm_text_alpha.settings import Settings
from tests.conftest import french_csv, zipped

REPO = Path(__file__).parents[2]


def synthetic_header(filing: SmokeFiling, acceptance_raw: str | None = None) -> str:
    items = "\n".join(f"<ITEMS>{item}" for item in filing.items)
    return (
        f"<SEC-HEADER>{filing.accession}.hdr.sgml : 20200101\n"
        f"<ACCEPTANCE-DATETIME>{acceptance_raw or filing.acceptance_raw}\n"
        f"<ACCESSION-NUMBER>{filing.accession}\n<TYPE>{filing.form}\n{items}\n"
        f"<FILING-DATE>{filing.acceptance_raw[:8]}\n<FILER>\n<COMPANY-DATA>\n"
        f"<CONFORMED-NAME>Synthetic Filer {filing.cik}\n<CIK>{filing.cik:010d}\n"
        "</COMPANY-DATA>\n</FILER>\n</SEC-HEADER>\n"
    )


def sec_transport(
    config: SmokeConfig, bad_accession: str | None = None
) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    seen: list[httpx.Request] = []
    by_accession = {f.accession: f for f in config.filings}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        path = request.url.path
        if path.endswith(".hdr.sgml"):
            filing = by_accession[path.rsplit("/", 1)[-1].removesuffix(".hdr.sgml")]
            raw = "20000101000000" if filing.accession == bad_accession else None
            return httpx.Response(200, text=synthetic_header(filing, raw))
        cik = int(path.rsplit("CIK", 1)[-1].removesuffix(".json"))
        own = [f for f in config.filings if f.cik == cik]
        recent = {
            "accessionNumber": [f.accession for f in own],
            "acceptanceDateTime": [
                f.acceptance_ts_utc.strftime("%Y-%m-%dT%H:%M:%S.000Z") for f in own
            ],
        }
        return httpx.Response(200, json={"filings": {"recent": recent}})

    return httpx.MockTransport(handler), seen


def french_transport(month: str) -> httpx.MockTransport:
    from eqpit import calendar

    year, mon = (int(x) for x in month.split("-"))
    days = calendar.sessions(date(year, mon, 1), date(year, mon, 28 if mon == 2 else 30))
    stamps = [d.strftime("%Y%m%d") for d in days]
    ff5 = french_csv(
        ",Mkt-RF,SMB,HML,RMW,CMA,RF", [f"{s}, 0.1, 0.2, 0.3, 0.4, 0.5, 0.01" for s in stamps]
    )
    mom = french_csv(",Mom", [f"{s}, 0.2" for s in stamps])
    bodies = {factors.FF5_DAILY: zipped("a.csv", ff5), factors.MOM_DAILY: zipped("b.csv", mom)}

    def handler(request: httpx.Request) -> httpx.Response:
        if "sharadar" in request.url.host:
            return httpx.Response(200, text="ticker\nGME\n")
        return httpx.Response(200, content=bodies[request.url.path.rsplit("/", 1)[-1]])

    return httpx.MockTransport(handler)


@pytest.fixture
def smoke_config() -> SmokeConfig:
    recorded = load_smoke(REPO / "configs" / "smoke.yaml")
    return recorded.model_copy(update={"month": "2016-06", "max_securities": 10})


@pytest.fixture
def settings(
    make_settings: Callable[..., Settings], efp_root: Path, monkeypatch: pytest.MonkeyPatch
) -> Settings:
    monkeypatch.setattr(prices.EfpPrices, "accepted_sources", ("synthetic",))
    return make_settings(efp_data_dir=str(efp_root), sec_max_rps=10)


def test_recorded_config_has_ten_filings() -> None:
    config = load_smoke(REPO / "configs" / "smoke.yaml")
    assert len(config.filings) == 10
    assert len({f.accession for f in config.filings}) == 10


def test_smoke_passes_on_recorded_responses(settings: Settings, smoke_config: SmokeConfig) -> None:
    transport, seen = sec_transport(smoke_config)
    report = run_smoke(
        settings,
        smoke_config,
        "run-ok",
        sec_transport=transport,
        http_transport=french_transport("2016-06"),
        throttle=ByteThrottle(1e12),
    )
    by_name = {check.name: check for check in report.checks}
    assert report.passed, report.checks
    assert by_name["edgar_acceptance_matches_fixture"].detail == "10 of 10 filings"
    assert by_name["edgar_submissions_json_agrees"].ok is True
    assert by_name["sharadar_key"].ok is None
    assert by_name["prices_one_month"].detail.startswith("provider efp: ")
    assert all(r.headers["User-Agent"] == settings.sec_user_agent for r in seen)
    assert len(seen) == 10 + len({f.cik for f in smoke_config.filings})
    saved = list((settings.data_dir / "raw" / "edgar" / "headers").rglob("*.hdr.sgml"))
    assert len(saved) == 10
    payload = json.loads(smoke_report_path(settings.data_dir, "run-ok").read_text("utf-8"))
    assert payload["passed"] is True
    assert payload["provider"] == "efp"
    panel = settings.data_dir / "processed" / "panels" / "factors_daily.parquet"
    assert panel.exists()


def test_smoke_fails_on_a_changed_acceptance_time(
    settings: Settings, smoke_config: SmokeConfig
) -> None:
    transport, _ = sec_transport(smoke_config, bad_accession=smoke_config.filings[3].accession)
    report = run_smoke(
        settings,
        smoke_config,
        "run-bad",
        sec_transport=transport,
        http_transport=french_transport("2016-06"),
        throttle=ByteThrottle(1e12),
    )
    by_name = {check.name: check for check in report.checks}
    assert not report.passed
    assert by_name["edgar_acceptance_matches_fixture"].detail == "9 of 10 filings"
    assert by_name["edgar_submissions_json_agrees"].ok is False


def test_smoke_reports_a_missing_provider_and_checks_the_key(
    make_settings: Callable[..., Settings], smoke_config: SmokeConfig
) -> None:
    settings = make_settings(sharadar_api_key="k")
    transport, _ = sec_transport(smoke_config)
    report = run_smoke(
        settings,
        smoke_config,
        "run-key",
        sec_transport=transport,
        http_transport=french_transport("2016-06"),
        throttle=ByteThrottle(1e12),
    )
    by_name = {check.name: check for check in report.checks}
    assert by_name["prices_one_month"].ok is False
    assert "M3" in by_name["prices_one_month"].detail
    assert by_name["sharadar_key"].ok is True
    assert not report.passed
