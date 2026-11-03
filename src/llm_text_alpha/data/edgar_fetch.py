"""``lta data edgar-fetch``: SGML header, primary document and EX-99.x exhibits per filing.

One request to ``<accession>-index-headers.html`` returns the SGML header (acceptance time,
items, filer names) and the list of documents with their types. The primary document and
every EX-99.x exhibit in HTML or text form follow. A filing is complete once its
``_done.json`` manifest is written, so an interrupted run resumes where it stopped (NFR-09).
"""

from __future__ import annotations

import hashlib
import html
import re
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import polars as pl
import structlog

from llm_text_alpha.data.edgar import (
    FilingHeader,
    SecClient,
    SecHttpError,
    accession_folder,
    parse_header,
)
from llm_text_alpha.data.ratelimit import ByteThrottle
from llm_text_alpha.io import write_json_atomic

log = structlog.get_logger(__name__)

ITEM_202 = "2.02"
DONE = "_done.json"
TEXT_SUFFIXES = (".htm", ".html", ".txt")
_DOCUMENT = re.compile(
    r"<DOCUMENT>\s*<TYPE>([^\n<]+)\s*<SEQUENCE>(\d+)\s*<FILENAME>([^\n<]+)", re.MULTILINE
)


@dataclass(frozen=True)
class DocumentEntry:
    """One document of a filing, as listed in its index headers."""

    type: str
    sequence: int
    filename: str


def index_headers_url(cik: int, accession: str) -> str:
    """URL of ``<accession>-index-headers.html``."""
    return f"{accession_folder(cik, accession)}/{accession}-index-headers.html"


def parse_index_headers(page: str) -> tuple[FilingHeader, list[DocumentEntry]]:
    """The SGML header and the document list from an index-headers page."""
    text = html.unescape(page)
    end = text.find("</SEC-HEADER>")
    if end < 0:
        raise ValueError("index-headers page has no </SEC-HEADER>")
    header = parse_header(text[:end], require_acceptance=False)
    documents = [
        DocumentEntry(m.group(1).strip(), int(m.group(2)), m.group(3).strip())
        for m in _DOCUMENT.finditer(text[end:])
    ]
    return header, documents


def wanted_documents(
    header: FilingHeader, documents: Iterable[DocumentEntry]
) -> list[DocumentEntry]:
    """The primary document (type equal to the form) and EX-99.x exhibits, HTML or text."""
    chosen: list[DocumentEntry] = []
    for document in documents:
        primary = document.type == header.form
        exhibit = document.type.upper().startswith("EX-99")
        if (primary or exhibit) and document.filename.lower().endswith(TEXT_SUFFIXES):
            chosen.append(document)
    return chosen


def filing_dir(raw_root: Path, cik: int, accession: str) -> Path:
    """Where one filing's raw files live."""
    return raw_root / str(cik) / accession


def is_done(raw_root: Path, cik: int, accession: str) -> bool:
    """True when the filing's manifest exists."""
    return (filing_dir(raw_root, cik, accession) / DONE).exists()


def fetch_filing(
    client: SecClient,
    raw_root: Path,
    cik: int,
    accession: str,
    throttle: ByteThrottle | None = None,
) -> dict[str, Any]:
    """Fetch one filing's header and documents, then write its manifest; return it."""
    folder = filing_dir(raw_root, cik, accession)
    page = client.get_text(index_headers_url(cik, accession))
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "index-headers.html").write_text(page, "utf-8")
    header, documents = parse_index_headers(page)
    fetched: list[dict[str, Any]] = []
    missing: list[str] = []
    for document in wanted_documents(header, documents):
        url = f"{accession_folder(cik, accession)}/{document.filename}"
        try:
            n_bytes = client.download(url, folder / document.filename, throttle)
        except SecHttpError as exc:
            if exc.status != 404:
                raise
            missing.append(document.filename)
            continue
        digest = hashlib.sha256((folder / document.filename).read_bytes()).hexdigest()
        fetched.append(asdict(document) | {"n_bytes": n_bytes, "sha256": digest})
    manifest = {
        "accession": accession,
        "cik": cik,
        "form": header.form,
        "items": list(header.items),
        "acceptance_raw": header.acceptance_raw,
        "acceptance_ts_utc": (
            header.acceptance_ts_utc.isoformat() if header.acceptance_ts_utc else None
        ),
        "filed_date": header.filed_date.isoformat(),
        "ciks": list(header.ciks),
        "company_name": header.company_name,
        "former_names": list(header.former_names),
        "documents": fetched,
        "missing": missing,
        "fetched_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    write_json_atomic(manifest, folder / DONE)
    return manifest


def select_item202(index: pl.DataFrame, start: date, end: date) -> pl.DataFrame:
    """Original 8-Ks whose submissions items include 2.02, filed from ``start`` to ``end``.

    Three days are added after ``end``, since a filing accepted late in the evening carries
    the next business day's filing date; the corpus keeps events by acceptance time.
    Amendments are left out: DQ-05 keeps them out of signals. The header's own items are
    checked again when the corpus is built.
    """
    return index.filter(
        (pl.col("form") == "8-K")
        & pl.col("items").list.contains(ITEM_202)
        & pl.col("filed_date").is_between(start, end + timedelta(days=3))
    ).sort("filed_date", "cik", "accession")


@dataclass
class FetchSummary:
    """Counts for the log and the CLI."""

    selected: int = 0
    skipped_done: int = 0
    fetched: int = 0
    failed: int = 0
    seconds: float = 0.0


def fetch_all(
    client: SecClient,
    raw_root: Path,
    filings: pl.DataFrame,
    limit: int | None = None,
    resume: bool = True,
    throttle: ByteThrottle | None = None,
    progress_every: int = 100,
    workers: int = 1,
) -> FetchSummary:
    """Fetch every filing in ``filings`` (``cik``, ``accession``) not yet done.

    ``workers`` threads share the client's single rate limiter, so together they never
    exceed ``SEC_MAX_RPS`` (NFR-07); threads only hide each request's latency. A filing that
    fails after the client's retries is logged and left without a manifest, so the next
    ``--resume`` run tries it again.
    """
    summary = FetchSummary(selected=filings.height)
    started = time.monotonic()
    todo: list[tuple[int, str]] = []
    for cik, accession in filings.select("cik", "accession").iter_rows():
        if resume and is_done(raw_root, cik, accession):
            summary.skipped_done += 1
        else:
            todo.append((cik, accession))
    if limit is not None:
        todo = todo[:limit]
    lock = threading.Lock()

    def one(cik: int, accession: str) -> None:
        try:
            fetch_filing(client, raw_root, cik, accession, throttle)
        except (SecHttpError, ValueError, httpx.TransportError) as exc:
            with lock:
                summary.failed += 1
            log.warning("edgar_fetch_failed", cik=cik, accession=accession, error=str(exc))
            return
        with lock:
            summary.fetched += 1
            done = summary.fetched
        if done % progress_every == 0:
            log.info(
                "edgar_fetch_progress",
                fetched=done,
                skipped_done=summary.skipped_done,
                failed=summary.failed,
                selected=summary.selected,
                requests=client.stats.requests,
                elapsed_s=round(time.monotonic() - started, 1),
            )

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for future in [pool.submit(one, cik, accession) for cik, accession in todo]:
            future.result()
    summary.seconds = time.monotonic() - started
    return summary
