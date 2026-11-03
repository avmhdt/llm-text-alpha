from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import polars as pl
import pytest

from llm_text_alpha.data import factors
from llm_text_alpha.data.ratelimit import ByteThrottle
from tests.conftest import french_csv, zipped

FF5_HEADER = ",Mkt-RF,SMB,HML,RMW,CMA,RF"
FF5_ROWS = [
    "20260901,   -0.67,    0.00,   -0.33,   -0.01,    0.16,    0.01",
    "20260902,    0.79,   -0.26,    0.26,   -0.07,   -0.20,    0.01",
    "20260903,  -99.99,    0.10,    0.20,    0.30,    0.40,    0.01",
]
MOM_ROWS = ["20260831,   0.10", "20260901,  -0.31", "20260902,   0.32", "20260903,   0.05"]


def write_snapshot(folder: Path, ff5_rows: list[str], mom_rows: list[str]) -> Path:
    folder.mkdir(parents=True)
    (folder / factors.FF5_DAILY).write_bytes(zipped("ff5.csv", french_csv(FF5_HEADER, ff5_rows)))
    (folder / factors.MOM_DAILY).write_bytes(zipped("mom.csv", french_csv(",Mom   ", mom_rows)))
    (folder / "manifest.json").write_text("{}", "utf-8")
    return folder


def test_parse_converts_percent_and_missing() -> None:
    frame = factors.parse_french_daily(
        french_csv(FF5_HEADER, FF5_ROWS), {"Mkt-RF": "mktrf", "RF": "rf"}
    )
    assert frame.columns == ["date", "mktrf", "rf"]
    assert frame["date"].to_list() == [date(2026, 9, 1), date(2026, 9, 2), date(2026, 9, 3)]
    assert frame["mktrf"].to_list()[:2] == pytest.approx([-0.0067, 0.0079])
    assert frame["mktrf"][2] is None
    assert frame["rf"].to_list() == pytest.approx([0.0001] * 3)


def test_parse_stops_at_the_first_non_row() -> None:
    text = french_csv(",Mom", ["20260901, 0.1"]) + "\n,Mom\n200001, 1.0\n"
    assert factors.parse_french_daily(text, {"Mom": "umd"}).height == 1


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("no header here\n20260901, 1\n", "no header"),
        (french_csv(",A,B", ["20260901, 1, 2"]), "not in header"),
        (french_csv(",Mom", ["20260901, 1, 2"]), "has 2 values"),
        (french_csv(",Mom", []), "no daily rows"),
    ],
)
def test_parse_errors(text: str, message: str) -> None:
    with pytest.raises(factors.FactorFileError, match=message):
        factors.parse_french_daily(text, {"Mom": "umd"})


def test_build_factors_daily(tmp_path: Path) -> None:
    snapshot = write_snapshot(tmp_path / "s", FF5_ROWS, MOM_ROWS)
    table = factors.build_factors_daily(snapshot)
    assert table.columns == ["date", *factors.FACTOR_COLUMNS]
    assert table.height == 3
    assert table["umd"].to_list() == pytest.approx([-0.0031, 0.0032, 0.0005])
    assert table.schema["hml"] == pl.Float64


def test_build_needs_momentum_on_every_date(tmp_path: Path) -> None:
    snapshot = write_snapshot(tmp_path / "s", FF5_ROWS, MOM_ROWS[:2])
    with pytest.raises(factors.FactorFileError, match="missing on 2"):
        factors.build_factors_daily(snapshot)


def test_archive_must_hold_one_csv(tmp_path: Path) -> None:
    snapshot = write_snapshot(tmp_path / "s", FF5_ROWS, MOM_ROWS)
    (snapshot / factors.FF5_DAILY).write_bytes(zipped("readme.txt", "x"))
    with pytest.raises(factors.FactorFileError, match="one CSV"):
        factors.build_factors_daily(snapshot)


def test_download_writes_snapshot_and_manifest(tmp_path: Path) -> None:
    bodies = {
        factors.FF5_DAILY: zipped("ff5.csv", french_csv(FF5_HEADER, FF5_ROWS)),
        factors.MOM_DAILY: zipped("mom.csv", french_csv(",Mom", MOM_ROWS)),
    }
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if len(calls) == 1:
            return httpx.Response(503)
        return httpx.Response(200, content=bodies[request.url.path.rsplit("/", 1)[-1]])

    throttle = ByteThrottle(1e12)
    snapshot = factors.download_factors(
        tmp_path / "raw",
        transport=httpx.MockTransport(handler),
        throttle=throttle,
        now=datetime(2026, 11, 3, 12, 0, tzinfo=UTC),
    )
    assert snapshot.name == "snapshot=20261103T120000Z"
    manifest = json.loads((snapshot / "manifest.json").read_text("utf-8"))
    assert [f["name"] for f in manifest["files"]] == [factors.FF5_DAILY, factors.MOM_DAILY]
    assert len(calls) == 3
    assert throttle.total_bytes == sum(len(b) for b in bodies.values())
    assert factors.latest_snapshot(tmp_path / "raw") == snapshot
    assert factors.build_factors_daily(snapshot).height == 3


def test_latest_snapshot_ignores_incomplete(tmp_path: Path) -> None:
    assert factors.latest_snapshot(tmp_path) is None
    (tmp_path / "snapshot=20990101T000000Z").mkdir()
    assert factors.latest_snapshot(tmp_path) is None
