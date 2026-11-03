from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from pathlib import Path

import httpx
import polars as pl
import pytest

from llm_text_alpha.data import prices
from llm_text_alpha.settings import Settings


@pytest.fixture
def synthetic_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let EfpPrices accept eqpit's synthetic store in tests."""
    monkeypatch.setattr(prices.EfpPrices, "accepted_sources", ("sharadar", "synthetic"))


def test_efp_reads_one_month_through_pitstore(efp_root: Path, synthetic_ok: None) -> None:
    provider = prices.EfpPrices(efp_root)
    assert provider.check()["provider"] == "synthetic"
    frame = provider.daily_prices(date(2016, 6, 1), date(2016, 6, 30), 10)
    assert frame.columns == list(prices.PRICE_COLUMNS)
    assert frame["security_id"].n_unique() == 10
    assert frame["date"].n_unique() == 22
    assert frame["date"].min() == date(2016, 6, 1)
    assert frame["date"].max() == date(2016, 6, 30)
    assert frame["provider"].unique().to_list() == ["efp"]


def test_efp_empty_window(efp_root: Path, synthetic_ok: None) -> None:
    frame = prices.EfpPrices(efp_root).daily_prices(date(2016, 7, 2), date(2016, 7, 4), 5)
    assert frame.is_empty()
    assert frame.columns == list(prices.PRICE_COLUMNS)


def test_efp_refuses_unlicensed_or_missing_store(efp_root: Path, tmp_path: Path) -> None:
    with pytest.raises(prices.ProviderUnavailableError, match="built from 'synthetic'"):
        prices.EfpPrices(efp_root).check()
    with pytest.raises(prices.ProviderUnavailableError, match="no eqpit manifest"):
        prices.EfpPrices(tmp_path).check()
    labelled = tmp_path / "labelled" / "processed" / "pit"
    labelled.mkdir(parents=True)
    (labelled / "manifest.json").write_text(json.dumps({"label": "dev only"}), "utf-8")
    with pytest.raises(prices.ProviderUnavailableError, match="labelled"):
        prices.EfpPrices(tmp_path / "labelled").check()


def test_resolve_prefers_efp(
    make_settings: Callable[..., Settings], efp_root: Path, synthetic_ok: None
) -> None:
    settings = make_settings(efp_data_dir=str(efp_root), sharadar_api_key="k")
    assert isinstance(prices.resolve_provider(settings), prices.EfpPrices)
    assert isinstance(prices.resolve_provider(settings, "sharadar"), prices.SharadarPrices)


def test_resolve_falls_back_in_order(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    broken = make_settings(efp_data_dir=str(tmp_path / "nothing"), sharadar_api_key="k")
    assert isinstance(prices.resolve_provider(broken), prices.SharadarPrices)
    with pytest.raises(prices.ProviderUnavailableError):
        prices.resolve_provider(broken, "efp")
    stooq = tmp_path / "stooq"
    stooq.mkdir()
    free = make_settings(stooq_dir=str(stooq))
    provider = prices.resolve_provider(free, free_tickers=["aapl"])
    assert isinstance(provider, prices.FreePrices)
    assert provider.label == prices.PROTOTYPE_LABEL


def test_resolve_errors(make_settings: Callable[..., Settings]) -> None:
    with pytest.raises(prices.ProviderUnavailableError, match="WRDS"):
        prices.resolve_provider(make_settings(), "crsp")
    with pytest.raises(prices.ProviderUnavailableError, match="needs EFP_DATA_DIR"):
        prices.resolve_provider(make_settings(), "efp")
    with pytest.raises(prices.ProviderUnavailableError, match="no price provider"):
        prices.resolve_provider(make_settings())
    with pytest.raises(prices.ProviderUnavailableError, match="not set"):
        prices.resolve_provider(make_settings(), "sharadar")


def test_sharadar_key_check_uses_the_header() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text="ticker,date\nGME,2026-10-30\n")

    provider = prices.SharadarPrices("secret-key", transport=httpx.MockTransport(handler))
    assert provider.check_key() == 200
    request = seen[0]
    assert request.headers["x-api-key"] == "secret-key"
    assert "secret-key" not in str(request.url)
    assert request.url.params["ticker"] == prices.SHARADAR_KEY_CHECK_TICKER
    assert request.url.path == "/v1.0/data/stocks"
    with pytest.raises(prices.ProviderUnavailableError, match="M3"):
        provider.daily_prices(date(2026, 9, 1), date(2026, 9, 30), 1)


def write_stooq(root: Path, folder: str, ticker: str, rows: list[str]) -> None:
    path = root / "data" / "daily" / "us" / folder / f"{ticker}.us.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>"
    path.write_text(header + "\n" + "\n".join(rows) + "\n", "utf-8")


def test_free_prices_from_a_stooq_tree(tmp_path: Path) -> None:
    write_stooq(
        tmp_path,
        "nasdaq stocks/1",
        "aapl",
        [
            "AAPL.US,D,20260831,000000,1,2,0.5,1.5,100.5,0",
            "AAPL.US,D,20260901,000000,2,3,1.5,2.5,200,0",
        ],
    )
    write_stooq(tmp_path, "nyse etfs", "spy", ["SPY.US,D,20260901,000000,5,6,4,5.5,10,0"])
    write_stooq(tmp_path, "nyse stocks", "ko", ["KO.US,D,20260901,000000,7,8,6,7.5,10,0"])
    index = prices.index_stooq(tmp_path)
    assert [item.ticker for item in index] == ["AAPL", "KO"]
    provider = prices.FreePrices(tmp_path, ["aapl", "ko", "msft"])
    frame = provider.daily_prices(date(2026, 9, 1), date(2026, 9, 30), 2)
    assert frame.columns == list(prices.PRICE_COLUMNS)
    assert frame["security_id"].to_list() == ["AAPL", "KO"]
    assert frame.schema["volume"] == pl.Float64
    assert provider.daily_prices(date(2020, 1, 1), date(2020, 1, 31), 2).is_empty()
    with pytest.raises(prices.ProviderUnavailableError):
        prices.FreePrices(tmp_path / "absent", [])
