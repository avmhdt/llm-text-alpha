"""Live EDGAR check of the 10 recorded filings (AC-W-04). Local only: ``pytest -m network``."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from llm_text_alpha.config import load_smoke
from llm_text_alpha.data.smoke import SmokeReport, check_edgar
from llm_text_alpha.settings import Settings, get_settings

REPO = Path(__file__).parents[2]


@pytest.mark.network
def test_ten_live_filings_match_the_recorded_acceptance_times(
    make_settings: Callable[..., Settings],
) -> None:
    owner = get_settings(REPO / ".env")
    if not owner.sec_user_agent:
        pytest.skip("SEC_USER_AGENT is not set in .env")
    settings = make_settings(sec_user_agent=owner.sec_user_agent, sec_max_rps=owner.sec_max_rps)
    report = SmokeReport(run_id="network-test")
    check_edgar(report, settings, load_smoke(REPO / "configs" / "smoke.yaml"))
    by_name = {check.name: check for check in report.checks}
    assert by_name["edgar_acceptance_matches_fixture"].ok, by_name
    assert by_name["edgar_user_agent_on_every_request"].ok
    assert by_name["edgar_rate_within_limit"].ok
    assert by_name["edgar_submissions_json_agrees"].ok is not False
