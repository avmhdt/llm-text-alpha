"""SEC EDGAR access under the fair-access policy (FR-01, NFR-07).

Every request carries ``SEC_USER_AGENT``, passes one shared :class:`TokenBucket`, and backs
off with exponential delay and jitter on HTTP 403, 429 and 5xx.
"""

from __future__ import annotations

import random
import re
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import httpx
import structlog
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception_type,
    retry_if_result,
    stop_after_attempt,
)

from llm_text_alpha.corpus.timestamps import TimestampError, parse_acceptance
from llm_text_alpha.data.ratelimit import ByteThrottle, TokenBucket
from llm_text_alpha.settings import SEC_RPS_CEILING, Settings

log = structlog.get_logger(__name__)

ARCHIVES = "https://www.sec.gov/Archives/edgar"
SUBMISSIONS = "https://data.sec.gov/submissions"
RETRY_STATUS = frozenset({403, 429, 500, 502, 503, 504})
_EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
_ACCESSION = re.compile(r"^\d{10}-\d{2}-\d{6}$")


class SecConfigError(ValueError):
    """The client would break SEC's fair-access policy."""


class SecHttpError(RuntimeError):
    """A request still failed after the retries."""

    def __init__(self, url: str, status: int) -> None:
        super().__init__(f"HTTP {status} from {url}")
        self.url = url
        self.status = status


def validate_user_agent(user_agent: str) -> str:
    """Return ``user_agent`` stripped, or raise if it is empty or has no contact email."""
    candidate = user_agent.strip()
    if not candidate:
        raise SecConfigError("SEC_USER_AGENT is not set")
    if not _EMAIL.search(candidate):
        raise SecConfigError("SEC_USER_AGENT must contain a contact email address")
    return candidate


@dataclass
class ClientStats:
    """Counters for the logs and the smoke report (NFR-08)."""

    requests: int = 0
    retries: int = 0
    throttled_seconds: float = 0.0
    status_counts: Counter[int] = field(default_factory=Counter)
    request_times: list[float] = field(default_factory=list)
    user_agent_ok: bool = True

    def measured_rps(self) -> float | None:
        """Requests per second between the first and last request; None below two requests."""
        if len(self.request_times) < 2:
            return None
        span = self.request_times[-1] - self.request_times[0]
        return (len(self.request_times) - 1) / span if span > 0 else float("inf")


class SecClient:
    """Rate-limited EDGAR HTTP client.

    Args:
        user_agent: name plus contact email, sent on every request.
        max_rps: client-side cap, at most SEC's published 10 requests per second.
        transport: httpx transport override, used by tests.
        max_attempts: attempts per URL, including the first.
        backoff_base: first backoff in seconds; doubles per retry, plus up to 100% jitter.
        backoff_max: cap on a single backoff in seconds.
        clock: monotonic clock in seconds, shared with the limiter.
        sleep: sleep function in seconds, shared with the limiter.
    """

    def __init__(
        self,
        user_agent: str,
        max_rps: float = 8.0,
        transport: httpx.BaseTransport | None = None,
        max_attempts: int = 6,
        backoff_base: float = 1.0,
        backoff_max: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.user_agent = validate_user_agent(user_agent)
        if not 0 < max_rps <= SEC_RPS_CEILING:
            raise SecConfigError(f"SEC_MAX_RPS={max_rps} must be in (0, {SEC_RPS_CEILING}]")
        self.max_rps = max_rps
        self.stats = ClientStats()
        self._clock = clock
        self._sleep = sleep
        self._bucket = TokenBucket(max_rps, clock=clock, sleep=sleep)
        self._max_attempts = max_attempts
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max
        self._http = httpx.Client(
            headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"},
            transport=transport,
            timeout=httpx.Timeout(60.0),
            follow_redirects=True,
            event_hooks={"request": [self._check_request]},
        )

    @classmethod
    def from_settings(cls, settings: Settings, **kwargs: Any) -> Self:
        """Build a client from ``SEC_USER_AGENT`` and ``SEC_MAX_RPS``."""
        return cls(settings.sec_user_agent, max_rps=settings.sec_max_rps, **kwargs)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Release the connection pool and log the session counters."""
        self._http.close()
        log.info(
            "sec_client_closed",
            requests=self.stats.requests,
            retries=self.stats.retries,
            throttled_seconds=round(self.stats.throttled_seconds, 3),
            status_counts=dict(self.stats.status_counts),
        )

    def _check_request(self, request: httpx.Request) -> None:
        if request.headers.get("User-Agent") != self.user_agent:
            self.stats.user_agent_ok = False
            raise SecConfigError("an SEC request was about to go out without the User-Agent")

    def _wait(self, state: RetryCallState) -> float:
        delay = min(self._backoff_max, self._backoff_base * 2.0 ** (state.attempt_number - 1))
        delay += random.uniform(0, delay)
        outcome = state.outcome
        if outcome is not None and not outcome.failed:
            retry_after = outcome.result().headers.get("Retry-After", "")
            if retry_after.isdigit():
                delay = max(delay, float(retry_after))
        return delay

    def _before_sleep(self, state: RetryCallState) -> None:
        self.stats.retries += 1
        outcome = state.outcome
        status = None if outcome is None or outcome.failed else outcome.result().status_code
        delay = state.next_action.sleep if state.next_action is not None else None
        log.warning("sec_retry", attempt=state.attempt_number, status=status, delay=delay)

    def _attempt(self, url: str) -> httpx.Response:
        self.stats.throttled_seconds += self._bucket.acquire()
        self.stats.requests += 1
        self.stats.request_times.append(self._clock())
        response = self._http.get(url)
        self.stats.status_counts[response.status_code] += 1
        return response

    def get(self, url: str) -> httpx.Response:
        """GET ``url`` with the limiter and retries; raise :class:`SecHttpError` on failure."""
        retrying = Retrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=self._wait,
            retry=(
                retry_if_result(lambda r: r.status_code in RETRY_STATUS)
                | retry_if_exception_type(httpx.TransportError)
            ),
            sleep=self._sleep,
            before_sleep=self._before_sleep,
            retry_error_callback=lambda state: state.outcome.result() if state.outcome else None,
            reraise=True,
        )
        response: httpx.Response = retrying(self._attempt, url)
        if response.status_code != 200:
            raise SecHttpError(url, response.status_code)
        return response

    def get_text(self, url: str) -> str:
        """GET ``url`` and decode the body."""
        return self.get(url).text

    def download(self, url: str, dest: Path, throttle: ByteThrottle | None = None) -> int:
        """Fetch ``url`` to ``dest`` atomically under the byte cap; return the byte count."""
        content = self.get(url).content
        if throttle is not None:
            throttle.consumed(len(content))
        dest.parent.mkdir(parents=True, exist_ok=True)
        temporary = dest.with_name(dest.name + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(dest)
        return len(content)


def accession_folder(cik: int, accession: str) -> str:
    """Archive folder URL of a filing."""
    return f"{ARCHIVES}/data/{cik}/{accession.replace('-', '')}"


def header_url(cik: int, accession: str) -> str:
    """URL of a filing's SGML header file (``<accession>.hdr.sgml``)."""
    return f"{accession_folder(cik, accession)}/{accession}.hdr.sgml"


def submissions_url(cik: int) -> str:
    """URL of a company's submissions JSON."""
    return f"{SUBMISSIONS}/CIK{cik:010d}.json"


@dataclass(frozen=True)
class FilingHeader:
    """Fields read from an SGML header. ``acceptance_ts_utc`` is the event time (DQ-01)."""

    accession: str
    form: str
    acceptance_raw: str
    acceptance_ts_utc: datetime
    filed_date: date
    items: tuple[str, ...]
    ciks: tuple[int, ...]
    company_name: str
    former_names: tuple[str, ...]

    @property
    def cik(self) -> int:
        """The first filer's CIK."""
        return self.ciks[0]


class HeaderParseError(ValueError):
    """An SGML header lacks a required field."""


_TAG = re.compile(r"^<([A-Z0-9-]+)>(.*)$")


def parse_header(text: str) -> FilingHeader:
    """Parse an EDGAR ``.hdr.sgml`` (or the header block of a full submission).

    Items keep EDGAR's numbering, such as ``2.02``. The company name is the first filer's,
    which is the registrant for 8-K, 10-K and 10-Q filings; former names cover every filer.
    """
    values: dict[str, list[str]] = {}
    for line in text.splitlines():
        match = _TAG.match(line.strip())
        if match and match.group(2).strip():
            values.setdefault(match.group(1), []).append(match.group(2).strip())

    def first(tag: str) -> str:
        found = values.get(tag)
        if not found:
            raise HeaderParseError(f"header has no <{tag}>")
        return found[0]

    accession = first("ACCESSION-NUMBER")
    if not _ACCESSION.match(accession):
        raise HeaderParseError(f"malformed accession {accession!r}")
    acceptance_raw = first("ACCEPTANCE-DATETIME")
    try:
        acceptance = parse_acceptance(acceptance_raw)
    except TimestampError as exc:
        raise HeaderParseError(str(exc)) from exc
    filed = datetime.strptime(first("FILING-DATE"), "%Y%m%d").date()
    if "CIK" not in values:
        raise HeaderParseError("header has no <CIK>")
    return FilingHeader(
        accession=accession,
        form=first("TYPE"),
        acceptance_raw=acceptance_raw,
        acceptance_ts_utc=acceptance,
        filed_date=filed,
        items=tuple(values.get("ITEMS", [])),
        ciks=tuple(int(cik) for cik in values["CIK"]),
        company_name=first("CONFORMED-NAME"),
        former_names=tuple(values.get("FORMER-CONFORMED-NAME", [])),
    )
