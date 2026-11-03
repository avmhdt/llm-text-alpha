from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pytest

from llm_text_alpha.data.edgar import (
    HeaderParseError,
    SecClient,
    SecConfigError,
    SecHttpError,
    accession_folder,
    header_url,
    parse_header,
    submissions_url,
    validate_user_agent,
)
from llm_text_alpha.settings import Settings
from tests.conftest import FIXTURES

UA = "lta-tests tests@example.com"


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def client_for(handler: object, **kwargs: object) -> tuple[SecClient, Clock]:
    clock = Clock()
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    client = SecClient(UA, transport=transport, clock=clock, sleep=clock.sleep, **kwargs)  # type: ignore[arg-type]
    return client, clock


def test_parse_synthetic_header() -> None:
    header = parse_header((FIXTURES / "edgar" / "synthetic_8k.hdr.sgml").read_text("utf-8"))
    assert header.accession == "0009999999-24-000001"
    assert header.form == "8-K"
    assert header.items == ("2.02", "9.01")
    assert header.acceptance_raw == "20240314161502"
    assert header.acceptance_ts_utc == datetime(2024, 3, 14, 20, 15, 2, tzinfo=UTC)
    assert header.filed_date == date(2024, 3, 14)
    assert header.ciks == (9999999, 9999998)
    assert header.cik == 9999999
    assert header.company_name == "Example Widgets Corp"
    assert header.former_names == ("EXAMPLE WIDGETS INC", "WIDGET HOLDINGS LLC")


@pytest.mark.parametrize(
    ("drop", "message"),
    [
        ("<ACCEPTANCE-DATETIME>", "ACCEPTANCE-DATETIME"),
        ("<ACCESSION-NUMBER>", "ACCESSION-NUMBER"),
        ("<CIK>", "CIK"),
    ],
)
def test_header_missing_field(drop: str, message: str) -> None:
    text = (FIXTURES / "edgar" / "synthetic_8k.hdr.sgml").read_text("utf-8")
    kept = "\n".join(line for line in text.splitlines() if not line.startswith(drop))
    with pytest.raises(HeaderParseError, match=message):
        parse_header(kept)


def test_header_bad_values() -> None:
    text = (FIXTURES / "edgar" / "synthetic_8k.hdr.sgml").read_text("utf-8")
    with pytest.raises(HeaderParseError, match="malformed accession"):
        parse_header(text.replace("<ACCESSION-NUMBER>0009999999-24-000001", "<ACCESSION-NUMBER>x"))
    with pytest.raises(HeaderParseError, match="YYYYMMDDHHMMSS"):
        parse_header(text.replace("20240314161502", "2024031416"))


def test_urls() -> None:
    assert accession_folder(320193, "0000320193-26-000024").endswith(
        "/Archives/edgar/data/320193/000032019326000024"
    )
    assert header_url(320193, "0000320193-26-000024").endswith(
        "/000032019326000024/0000320193-26-000024.hdr.sgml"
    )
    assert submissions_url(320193) == "https://data.sec.gov/submissions/CIK0000320193.json"


@pytest.mark.parametrize("bad", ["", "   ", "research agent"])
def test_user_agent_needs_an_email(bad: str) -> None:
    with pytest.raises(SecConfigError):
        validate_user_agent(bad)
    with pytest.raises(SecConfigError):
        SecClient(bad)


@pytest.mark.parametrize("rps", [0, 10.5, 50])
def test_rate_above_sec_limit_is_refused(rps: float) -> None:
    with pytest.raises(SecConfigError):
        SecClient(UA, max_rps=rps)


def test_from_settings() -> None:
    settings = Settings(_env_file=None, sec_user_agent=UA, sec_max_rps=5)  # type: ignore[call-arg]
    with SecClient.from_settings(settings) as client:
        assert client.max_rps == 5
        assert client.user_agent == UA


def test_every_request_carries_the_user_agent() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["User-Agent"])
        return httpx.Response(200, text="ok")

    client, _ = client_for(handler, max_rps=4)
    for n in range(9):
        assert client.get_text(f"https://www.sec.gov/x/{n}") == "ok"
    client.close()
    assert seen == [UA] * 9
    assert client.stats.user_agent_ok
    assert client.stats.requests == 9
    assert client.stats.measured_rps() == pytest.approx(4.0)
    assert client.stats.status_counts[200] == 9


def test_request_without_user_agent_is_stopped() -> None:
    client, _ = client_for(lambda request: httpx.Response(200))
    client._http.headers.pop("User-Agent")
    with pytest.raises(SecConfigError):
        client.get("https://www.sec.gov/x")
    assert not client.stats.user_agent_ok


@pytest.mark.parametrize("status", [403, 429, 503])
def test_backs_off_then_succeeds(status: int) -> None:
    replies = iter([httpx.Response(status), httpx.Response(status), httpx.Response(200, text="y")])
    client, clock = client_for(lambda request: next(replies), backoff_base=1.0)
    assert client.get_text("https://www.sec.gov/x") == "y"
    assert client.stats.retries == 2
    assert client.stats.status_counts[status] == 2
    backoffs = [s for s in clock.sleeps if s >= 1.0]
    assert len(backoffs) == 2
    assert 1.0 <= backoffs[0] <= 2.0
    assert 2.0 <= backoffs[1] <= 4.0


def test_retry_after_is_honoured() -> None:
    replies = iter([httpx.Response(429, headers={"Retry-After": "30"}), httpx.Response(200)])
    client, clock = client_for(lambda request: next(replies))
    client.get("https://www.sec.gov/x")
    assert max(clock.sleeps) == 30.0


def test_gives_up_after_max_attempts() -> None:
    client, _ = client_for(lambda request: httpx.Response(503), max_attempts=3)
    with pytest.raises(SecHttpError) as error:
        client.get("https://www.sec.gov/x")
    assert error.value.status == 503
    assert client.stats.requests == 3


def test_not_found_is_not_retried() -> None:
    client, _ = client_for(lambda request: httpx.Response(404))
    with pytest.raises(SecHttpError):
        client.get("https://www.sec.gov/x")
    assert client.stats.requests == 1


def test_transport_errors_are_retried() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("reset", request=request)
        return httpx.Response(200, text="z")

    client, _ = client_for(handler)
    assert client.get_text("https://www.sec.gov/x") == "z"
    assert client.stats.retries == 1


def test_transport_errors_reraise_after_attempts() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    client, _ = client_for(handler, max_attempts=2)
    with pytest.raises(httpx.ConnectError):
        client.get("https://www.sec.gov/x")


def test_download_is_atomic(tmp_path: Path) -> None:
    from llm_text_alpha.data.ratelimit import ByteThrottle

    clock = Clock()
    client, _ = client_for(lambda request: httpx.Response(200, content=b"abc"))
    target = tmp_path / "sub" / "file.bin"
    throttle = ByteThrottle(1000, clock=clock, sleep=clock.sleep)
    assert client.download("https://www.sec.gov/f", target, throttle) == 3
    assert target.read_bytes() == b"abc"
    assert not list(tmp_path.rglob("*.tmp"))
    assert throttle.total_bytes == 3
