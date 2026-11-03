"""Kenneth French daily factors: FF5 + momentum + risk-free rate, as decimals (FR-04).

The library publishes percent returns in CSV files inside ZIP archives. Raw archives are kept
under ``data/raw/factors/snapshot=<UTC time>/`` with a manifest; ``factors_daily`` is rebuilt
from them.
"""

from __future__ import annotations

import hashlib
import io
import re
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
import polars as pl
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from llm_text_alpha.data.ratelimit import ByteThrottle
from llm_text_alpha.io import write_json_atomic

FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
FF5_DAILY = "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
MOM_DAILY = "F-F_Momentum_Factor_daily_CSV.zip"
FACTOR_COLUMNS = ("mktrf", "smb", "hml", "rmw", "cma", "umd", "rf")
_FF5_NAMES = {"Mkt-RF": "mktrf", "SMB": "smb", "HML": "hml", "RMW": "rmw", "CMA": "cma", "RF": "rf"}
_MOM_NAMES = {"Mom": "umd"}
_ROW = re.compile(r"^\s*(\d{8})\s*,(.*)$")
_MISSING = (-99.99, -999.0)


class FactorFileError(ValueError):
    """A French file does not have the expected layout."""


@dataclass(frozen=True)
class DownloadRecord:
    """One archive in a snapshot manifest."""

    name: str
    url: str
    sha256: str
    n_bytes: int
    retrieved_at: str


@retry(
    retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
    wait=wait_random_exponential(multiplier=1, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _fetch(client: httpx.Client, url: str, throttle: ByteThrottle) -> bytes:
    chunks: list[bytes] = []
    with client.stream("GET", url) as response:
        response.raise_for_status()
        for chunk in response.iter_bytes(chunk_size=1 << 16):
            chunks.append(chunk)
            throttle.consumed(len(chunk))
    return b"".join(chunks)


def download_factors(
    raw_dir: Path,
    transport: httpx.BaseTransport | None = None,
    throttle: ByteThrottle | None = None,
    now: datetime | None = None,
) -> Path:
    """Download both daily archives into a new snapshot folder and return its path."""
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    snapshot = raw_dir / f"snapshot={stamp}"
    snapshot.mkdir(parents=True, exist_ok=True)
    throttle = throttle or ByteThrottle()
    records: list[DownloadRecord] = []
    with httpx.Client(transport=transport, timeout=httpx.Timeout(60.0)) as client:
        for name in (FF5_DAILY, MOM_DAILY):
            url = f"{FRENCH_BASE}/{name}"
            content = _fetch(client, url, throttle)
            (snapshot / name).write_bytes(content)
            records.append(
                DownloadRecord(
                    name=name,
                    url=url,
                    sha256=hashlib.sha256(content).hexdigest(),
                    n_bytes=len(content),
                    retrieved_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                )
            )
    manifest = {"source": FRENCH_BASE, "files": [asdict(record) for record in records]}
    write_json_atomic(manifest, snapshot / "manifest.json")
    return snapshot


def latest_snapshot(raw_dir: Path) -> Path | None:
    """Newest complete snapshot folder (one with a manifest), or None."""
    snapshots = sorted(p for p in raw_dir.glob("snapshot=*") if (p / "manifest.json").exists())
    return snapshots[-1] if snapshots else None


def _csv_text(archive: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(archive)) as handle:
        names = [n for n in handle.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise FactorFileError(f"expected one CSV in the archive, found {names}")
        return handle.read(names[0]).decode("latin-1")


def parse_french_daily(text: str, names: dict[str, str]) -> pl.DataFrame:
    """Parse the first daily block of a French CSV into ``date`` plus decimal columns.

    The block starts at the header line beginning with a comma and ends at the first line
    that is not ``YYYYMMDD,...``. Values are percent; -99.99 and -999 mark missing.
    """
    lines = text.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.startswith(","))
    except StopIteration as exc:
        raise FactorFileError("no header line starting with ','") from exc
    header = [cell.strip() for cell in lines[start].split(",")[1:]]
    missing = [name for name in names if name not in header]
    if missing:
        raise FactorFileError(f"columns {missing} not in header {header}")
    dates: list[str] = []
    rows: list[list[float | None]] = []
    for line in lines[start + 1 :]:
        match = _ROW.match(line)
        if not match:
            break
        cells = [cell.strip() for cell in match.group(2).split(",")]
        if len(cells) != len(header):
            raise FactorFileError(f"row {match.group(1)} has {len(cells)} values")
        values = [float(cell) for cell in cells]
        dates.append(match.group(1))
        rows.append([None if v in _MISSING else v / 100.0 for v in values])
    if not rows:
        raise FactorFileError("no daily rows after the header")
    data: dict[str, object] = {"date": dates}
    for source, target in names.items():
        index = header.index(source)
        data[target] = [row[index] for row in rows]
    frame = pl.DataFrame(data, schema_overrides={t: pl.Float64 for t in names.values()})
    return frame.with_columns(pl.col("date").str.strptime(pl.Date, "%Y%m%d")).sort("date")


def build_factors_daily(snapshot: Path) -> pl.DataFrame:
    """``factors_daily``: date, mktrf, smb, hml, rmw, cma, umd, rf (decimal daily returns).

    Rows are the FF5 dates; momentum joins on date and must cover every one of them.
    """
    ff5 = parse_french_daily(_csv_text((snapshot / FF5_DAILY).read_bytes()), _FF5_NAMES)
    mom = parse_french_daily(_csv_text((snapshot / MOM_DAILY).read_bytes()), _MOM_NAMES)
    joined = ff5.join(mom, on="date", how="left")
    gaps = joined.filter(pl.col("umd").is_null()).height
    if gaps:
        raise FactorFileError(f"momentum is missing on {gaps} FF5 dates")
    return joined.select("date", *FACTOR_COLUMNS)
