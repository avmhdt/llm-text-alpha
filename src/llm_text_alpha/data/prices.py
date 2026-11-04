"""Market-data adapter: one interface over the D-01 providers (FR-03).

``auto`` resolves to ``efp`` when ``EFP_DATA_DIR`` holds equity-factor-pipeline's
point-in-time store (itself built on Sharadar), else ``sharadar`` when a key is set, else
``free`` from the owner's Stooq snapshot, whose output is labelled survivorship-biased
(DQ-07). ``crsp`` needs WRDS, which the owner does not have.

M1 ships daily prices for a date range. M3 adds ``universe_monthly``, ``link_table`` and the
full ``prices_daily`` schema of section 7b.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, Protocol

import httpx
import polars as pl

from llm_text_alpha.settings import Settings

ProviderName = Literal["efp", "sharadar", "crsp", "free"]
PROTOTYPE_LABEL = "prototype, survivorship-biased"
SHARADAR_BASE_URL = "https://api.sharadar.com/v1.0"
SHARADAR_KEY_CHECK_TICKER = "GME"
"""Outside Sharadar's free tier: AAPL and MSFT answer with no key at all, so prove nothing."""

PRICE_COLUMNS = ("security_id", "date", "open", "close", "volume", "provider")
_PRICE_SCHEMA = {
    "security_id": pl.String,
    "date": pl.Date,
    "open": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "provider": pl.String,
}


class ProviderUnavailableError(RuntimeError):
    """The chosen provider cannot serve data in this environment."""


class PriceProvider(Protocol):
    """Daily prices from one source. ``label`` is printed on every table built from it."""

    name: ProviderName
    label: str

    def daily_prices(self, start: date, end: date, max_securities: int) -> pl.DataFrame:
        """Rows in ``PRICE_COLUMNS`` for sessions in ``[start, end]``; unadjusted USD prices."""
        ...


def _empty_prices() -> pl.DataFrame:
    return pl.DataFrame(schema=_PRICE_SCHEMA)


class EfpPrices:
    """equity-factor-pipeline's store, read only through ``eqpit.store.PitStore``.

    ``max_securities`` takes the largest names in eqpit's universe at the last month end
    before ``start``, so the selection uses nothing from inside the window.
    """

    name: ProviderName = "efp"
    label = ""
    accepted_sources: tuple[str, ...] = ("sharadar", "wrds")
    """eqpit stores built from licensed data; a synthetic or free build is refused."""

    def __init__(self, root: Path) -> None:
        self.pit_root = root / "processed" / "pit"

    def check(self) -> dict[str, object]:
        """Open the store and return its manifest summary, or raise ProviderUnavailableError."""
        manifest = self.pit_root / "manifest.json"
        if not manifest.exists():
            raise ProviderUnavailableError(f"no eqpit manifest at {manifest}")
        payload = json.loads(manifest.read_text("utf-8"))
        if payload.get("label"):
            raise ProviderUnavailableError(f"eqpit store is labelled {payload['label']!r}")
        if payload.get("provider") not in self.accepted_sources:
            raise ProviderUnavailableError(
                f"eqpit store was built from {payload.get('provider')!r}"
            )
        return {
            "provider": payload.get("provider"),
            "first_date": payload.get("first_date"),
            "last_date": payload.get("last_date"),
        }

    def universe_ciks(self, start: date, end: date, top_n: int) -> pl.DataFrame:
        """CIKs of companies in eqpit's top ``top_n`` at any month end in ``[start, end]``.

        Returns ``cik``, ``first_month`` and ``last_month``. Companies keyed by a Sharadar
        id rather than a CIK cannot be matched to EDGAR and are left out.
        """
        from eqpit.store import PitStore

        store = PitStore.open(self.pit_root)
        frames = [
            store.universe(month_end, top_n)
            .select("company_id")
            .with_columns(pl.lit(month_end).alias("month"))
            for month_end in store.month_ends(start, end)
        ]
        if not frames:
            return pl.DataFrame(
                schema={"cik": pl.Int64, "first_month": pl.Date, "last_month": pl.Date}
            )
        members = pl.concat(frames).filter(pl.col("company_id").str.starts_with("CIK:"))
        return (
            members.with_columns(
                pl.col("company_id").str.strip_prefix("CIK:").cast(pl.Int64).alias("cik")
            )
            .group_by("cik")
            .agg(
                pl.col("month").min().alias("first_month"),
                pl.col("month").max().alias("last_month"),
            )
            .sort("cik")
        )

    def daily_prices(self, start: date, end: date, max_securities: int) -> pl.DataFrame:
        from eqpit import calendar
        from eqpit.store import PitStore

        store = PitStore.open(self.pit_root)
        sessions = calendar.sessions(start, end)
        if not sessions:
            return _empty_prices()
        before = calendar.previous_session(sessions[0])
        formed = store.month_ends(date(before.year - 1, before.month, 1), before)[-1]
        ids = store.universe(formed, max_securities)["security_id"].to_list()
        frame = store.prices(sessions[-1], len(sessions), ids)
        return frame.select(
            pl.col("security_id"),
            pl.col("date"),
            pl.col("open"),
            pl.col("close"),
            pl.col("volume"),
            pl.lit("efp").alias("provider"),
        ).sort("security_id", "date")


class SharadarPrices:
    """Sharadar's direct API; the key travels in the ``x-api-key`` header, never the URL."""

    name: ProviderName = "sharadar"
    label = ""

    def __init__(self, api_key: str, transport: httpx.BaseTransport | None = None) -> None:
        if not api_key:
            raise ProviderUnavailableError("SHARADAR_API_KEY is not set")
        self._api_key = api_key
        self._transport = transport

    def check_key(self) -> int:
        """Query one SEP row for a ticker outside the free tier; return the HTTP status.

        200 means the key reads the bundle. Nothing about the key or the URL is logged.
        """
        with httpx.Client(
            base_url=SHARADAR_BASE_URL,
            headers={"x-api-key": self._api_key},
            transport=self._transport,
            timeout=httpx.Timeout(30.0),
        ) as client:
            response = client.get(
                "/data/stocks", params={"ticker": SHARADAR_KEY_CHECK_TICKER, "limit": 1}
            )
        return response.status_code

    def daily_prices(self, start: date, end: date, max_securities: int) -> pl.DataFrame:
        raise ProviderUnavailableError(
            "direct Sharadar price pulls arrive in M3; set EFP_DATA_DIR to read eqpit's store"
        )


@dataclass(frozen=True)
class StooqFile:
    """One ticker's daily file in the owner's Stooq snapshot."""

    ticker: str
    path: Path


def index_stooq(stooq_dir: Path) -> list[StooqFile]:
    """Walk ``data/daily/us/*stocks*`` once and list every ``<ticker>.us.txt`` file."""
    root = stooq_dir / "data" / "daily" / "us"
    files: list[StooqFile] = []
    for dirpath, _dirs, names in os.walk(root):
        if "stocks" not in Path(dirpath).relative_to(root).as_posix():
            continue
        for name in names:
            if name.endswith(".us.txt"):
                files.append(StooqFile(name.removesuffix(".us.txt").upper(), Path(dirpath, name)))
    return sorted(files, key=lambda f: f.ticker)


def read_stooq(path: Path) -> pl.DataFrame:
    """Read one Stooq daily file (``<TICKER>,<PER>,<DATE>,...``) into date/open/close/volume."""
    frame = pl.read_csv(path, schema_overrides={"<DATE>": pl.String, "<VOL>": pl.Float64})
    return frame.select(
        pl.col("<DATE>").str.strptime(pl.Date, "%Y%m%d").alias("date"),
        pl.col("<OPEN>").cast(pl.Float64).alias("open"),
        pl.col("<CLOSE>").cast(pl.Float64).alias("close"),
        pl.col("<VOL>").alias("volume"),
    )


class FreePrices:
    """Owner's Stooq snapshot: current tickers only, so every output is labelled prototype.

    Stooq prices appear split- and dividend-adjusted; M3 verifies that before any use.
    """

    name: ProviderName = "free"
    label = PROTOTYPE_LABEL

    def __init__(self, stooq_dir: Path, tickers: Sequence[str]) -> None:
        if not stooq_dir.is_dir():
            raise ProviderUnavailableError(f"STOOQ_DIR {stooq_dir} does not exist")
        self.stooq_dir = stooq_dir
        self.tickers = [t.upper() for t in tickers]

    def daily_prices(self, start: date, end: date, max_securities: int) -> pl.DataFrame:
        wanted = set(self.tickers[:max_securities])
        frames = [
            read_stooq(item.path)
            .filter(pl.col("date").is_between(start, end))
            .with_columns(
                pl.lit(item.ticker).alias("security_id"), pl.lit("free").alias("provider")
            )
            for item in index_stooq(self.stooq_dir)
            if item.ticker in wanted
        ]
        if not frames:
            return _empty_prices()
        return pl.concat(frames).select(PRICE_COLUMNS).sort("security_id", "date")


def resolve_provider(
    settings: Settings,
    requested: str | None = None,
    free_tickers: Sequence[str] = (),
) -> PriceProvider:
    """Pick the provider for ``requested`` (default ``PRICE_PROVIDER``) per D-01's order."""
    choice = requested or settings.price_provider
    if choice == "crsp":
        raise ProviderUnavailableError("crsp needs WRDS, and the owner has no WRDS access")
    if choice in ("auto", "efp") and settings.efp_root is not None:
        efp = EfpPrices(settings.efp_root)
        try:
            efp.check()
        except ProviderUnavailableError:
            if choice == "efp":
                raise
        else:
            return efp
    if choice == "efp":
        raise ProviderUnavailableError("PRICE_PROVIDER=efp needs EFP_DATA_DIR")
    key = settings.sharadar_api_key.get_secret_value()
    if choice == "sharadar" or (choice == "auto" and key):
        return SharadarPrices(key)
    if choice in ("auto", "free") and settings.stooq_dir:
        return FreePrices(Path(settings.stooq_dir), free_tickers)
    raise ProviderUnavailableError(
        f"no price provider available for {choice!r}: set EFP_DATA_DIR or SHARADAR_API_KEY"
    )
