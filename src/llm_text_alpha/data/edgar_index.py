"""``lta data edgar-index``: quarterly master indexes and submissions JSON for universe CIKs.

The master index lists every filing with its CIK, form and filing date. The submissions
JSON adds 8-K item numbers and ``acceptanceDateTime`` (UTC), which select the Item 2.02
filings to fetch and give the second source for the DQ-02 check. Raw files are kept under
``raw/edgar/`` and reused on a rerun.
"""

from __future__ import annotations

import gzip
import json
import re
from collections.abc import Iterable, Sequence
from datetime import date
from pathlib import Path

import polars as pl
import structlog

from llm_text_alpha.data.edgar import ARCHIVES, SUBMISSIONS, SecClient, SecHttpError
from llm_text_alpha.data.ratelimit import ByteThrottle

log = structlog.get_logger(__name__)

DEFAULT_FORMS = ("8-K", "8-K/A", "10-K", "10-K/A", "10-Q", "10-Q/A")
_QUARTER = re.compile(r"^(\d{4})Q([1-4])$")
_FILENAME = re.compile(r"edgar/data/\d+/(\d{10}-\d{2}-\d{6})\.txt$")

MASTER_SCHEMA = pl.Schema(
    {
        "cik": pl.Int64,
        "company_name": pl.String,
        "form": pl.String,
        "filed_date": pl.Date,
        "accession": pl.String,
    }
)


def with_amendments(forms: Iterable[str]) -> list[str]:
    """Each form and its ``/A`` amendment, which the index keeps (DQ-05 drops them later)."""
    expanded: list[str] = []
    for form in forms:
        base = form.strip().removesuffix("/A")
        expanded += [base, f"{base}/A"]
    return sorted(set(expanded))


def quarters(start: str, end: str) -> list[tuple[int, int]]:
    """Quarters from ``start`` to ``end`` inclusive, written like ``2016Q1``."""
    first, last = (_QUARTER.match(value) for value in (start, end))
    if first is None or last is None:
        raise ValueError(f"quarters must look like 2016Q1, got {start!r} and {end!r}")
    a = int(first.group(1)) * 4 + int(first.group(2)) - 1
    b = int(last.group(1)) * 4 + int(last.group(2)) - 1
    if a > b:
        raise ValueError(f"{start} is after {end}")
    return [(n // 4, n % 4 + 1) for n in range(a, b + 1)]


def master_url(year: int, quarter: int) -> str:
    """URL of a quarter's gzipped master index."""
    return f"{ARCHIVES}/full-index/{year}/QTR{quarter}/master.gz"


def parse_master(text: str) -> pl.DataFrame:
    """Rows of a master index (``CIK|Company Name|Form Type|Date Filed|Filename``)."""
    rows: list[tuple[int, str, str, str, str]] = []
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) != 5 or not parts[0].isdigit():
            continue
        match = _FILENAME.search(parts[4])
        if match is None:
            continue
        rows.append((int(parts[0]), parts[1].strip(), parts[2].strip(), parts[3], match.group(1)))
    frame = pl.DataFrame(
        rows,
        schema={k: pl.String if k != "cik" else pl.Int64 for k in MASTER_SCHEMA},
        orient="row",
    )
    return frame.with_columns(pl.col("filed_date").str.strptime(pl.Date, "%Y-%m-%d"))


def download_masters(
    client: SecClient,
    raw_dir: Path,
    periods: Sequence[tuple[int, int]],
    throttle: ByteThrottle | None = None,
) -> list[Path]:
    """Fetch each quarter's ``master.gz`` unless it is already on disk."""
    paths: list[Path] = []
    for year, quarter in periods:
        target = raw_dir / f"{year}Q{quarter}.master.gz"
        if not target.exists():
            client.download(master_url(year, quarter), target, throttle)
            log.info("edgar_master_downloaded", quarter=f"{year}Q{quarter}")
        paths.append(target)
    return paths


def read_masters(paths: Iterable[Path], forms: Sequence[str], ciks: Iterable[int]) -> pl.DataFrame:
    """Master rows for ``forms`` filed by ``ciks``, one row per accession and filer."""
    wanted = pl.Series("cik", sorted(set(ciks)), dtype=pl.Int64)
    frames = [
        parse_master(gzip.decompress(path.read_bytes()).decode("latin-1")).filter(
            pl.col("form").is_in(list(forms)) & pl.col("cik").is_in(wanted)
        )
        for path in paths
    ]
    if not frames:
        return pl.DataFrame(schema=MASTER_SCHEMA)
    return pl.concat(frames).unique(["accession", "cik"], keep="first").sort("filed_date", "cik")


def submissions_page_url(name: str) -> str:
    """URL of an older submissions page such as ``CIK0000320193-submissions-001.json``."""
    return f"{SUBMISSIONS}/{name}"


def fetch_submissions(
    client: SecClient, raw_dir: Path, cik: int, start: date, end: date
) -> list[Path]:
    """Recent submissions JSON plus every older page that overlaps ``[start, end]``.

    Files already on disk are reused, so a rerun after an interruption costs nothing for
    the companies already done. A CIK EDGAR does not know returns no paths.
    """
    folder = raw_dir / f"{cik:010d}"
    main = folder / "recent.json"
    if not main.exists():
        try:
            client.download(f"{SUBMISSIONS}/CIK{cik:010d}.json", main)
        except SecHttpError as exc:
            if exc.status == 404:
                log.warning("edgar_submissions_missing", cik=cik)
                return []
            raise
    paths = [main]
    for page in json.loads(main.read_text("utf-8"))["filings"].get("files", []):
        if date.fromisoformat(page["filingTo"]) < start:
            continue
        if date.fromisoformat(page["filingFrom"]) > end:
            continue
        target = folder / page["name"]
        if not target.exists():
            client.download(submissions_page_url(page["name"]), target)
        paths.append(target)
    return paths


_SUBMISSION_SCHEMA = pl.Schema(
    [
        ("accession", pl.String()),
        ("cik", pl.Int64()),
        ("submissions_form", pl.String()),
        ("items", pl.List(pl.String())),
        ("submissions_acceptance_utc", pl.Datetime("ns", "UTC")),
        ("primary_document", pl.String()),
    ]
)


def parse_submissions(cik: int, paths: Sequence[Path]) -> pl.DataFrame:
    """Accession, form, items, UTC acceptance and primary document from submissions files."""
    frames: list[pl.DataFrame] = []
    for path in paths:
        payload = json.loads(path.read_text("utf-8"))
        block = payload["filings"]["recent"] if "filings" in payload else payload
        if not block.get("accessionNumber"):
            continue
        frames.append(
            pl.DataFrame(
                {
                    "accession": block["accessionNumber"],
                    "submissions_form": block["form"],
                    "items": block.get("items", [""] * len(block["form"])),
                    "acceptance": block["acceptanceDateTime"],
                    "primary_document": block.get("primaryDocument", [""] * len(block["form"])),
                }
            )
        )
    if not frames:
        return pl.DataFrame(schema=_SUBMISSION_SCHEMA)
    frame = pl.concat(frames)
    return frame.select(
        pl.col("accession"),
        pl.lit(cik, pl.Int64).alias("cik"),
        pl.col("submissions_form"),
        pl.when(pl.col("items") == "")
        .then(pl.lit([], pl.List(pl.String)))
        .otherwise(pl.col("items").str.split(","))
        .alias("items"),
        pl.col("acceptance")
        .str.strptime(pl.Datetime("ns"), "%Y-%m-%dT%H:%M:%S%.3fZ", strict=False)
        .dt.replace_time_zone("UTC")
        .alias("submissions_acceptance_utc"),
        pl.col("primary_document"),
    ).unique("accession", keep="first")


def build_filings_index(master: pl.DataFrame, submissions: pl.DataFrame) -> pl.DataFrame:
    """Master rows joined to submissions fields on accession and CIK.

    Acceptance times from the SGML header join later, for fetched filings (DQ-01).
    ``filed_date`` is kept for audit and never used for timing.
    """
    return (
        master.join(submissions, on=["accession", "cik"], how="left")
        .with_columns(pl.col("items").fill_null(pl.lit([], pl.List(pl.String))))
        .sort("filed_date", "cik", "accession")
    )
