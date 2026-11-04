from __future__ import annotations

import gzip
import json
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import polars as pl
import pytest

from llm_text_alpha.data import edgar_index
from llm_text_alpha.data.edgar import SecClient

UA = "lta-tests tests@example.com"
MASTER = """Description:           Master Index of EDGAR Dissemination Feed
Last Data Received:    March 31, 2016

CIK|Company Name|Form Type|Date Filed|Filename
--------------------------------------------------------------------------------
320193|APPLE INC|8-K|2016-01-26|edgar/data/320193/0001193125-16-438421.txt
320193|APPLE INC|4|2016-01-27|edgar/data/320193/0001181431-16-000001.txt
320193|APPLE INC|10-Q|2016-01-27|edgar/data/320193/0001193125-16-439878.txt
789019|MICROSOFT CORP|8-K/A|2016-02-01|edgar/data/789019/0001193125-16-000002.txt
999999|SOMEONE ELSE|8-K|2016-02-01|edgar/data/999999/0000999999-16-000001.txt
bad|line|here
"""


def test_quarters() -> None:
    assert edgar_index.quarters("2016Q3", "2017Q2") == [(2016, 3), (2016, 4), (2017, 1), (2017, 2)]
    with pytest.raises(ValueError, match="look like"):
        edgar_index.quarters("2016-1", "2016Q2")
    with pytest.raises(ValueError, match="after"):
        edgar_index.quarters("2017Q1", "2016Q4")


def test_with_amendments() -> None:
    assert edgar_index.with_amendments(["8-K", "10-Q/A"]) == ["10-Q", "10-Q/A", "8-K", "8-K/A"]


def test_parse_and_filter_masters(tmp_path: Path) -> None:
    frame = edgar_index.parse_master(MASTER)
    assert frame.height == 5
    assert frame.schema["filed_date"] == pl.Date
    path = tmp_path / "2016Q1.master.gz"
    path.write_bytes(gzip.compress(MASTER.encode("latin-1")))
    kept = edgar_index.read_masters(
        [path, path], edgar_index.with_amendments(["8-K", "10-Q"]), [320193, 789019]
    )
    assert kept["accession"].to_list() == [
        "0001193125-16-438421",
        "0001193125-16-439878",
        "0001193125-16-000002",
    ]
    assert edgar_index.read_masters([], ["8-K"], [1]).is_empty()


def mock_client(handler: object) -> SecClient:
    return SecClient(UA, max_rps=10, transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


def test_download_masters_skips_existing(tmp_path: Path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, content=gzip.compress(MASTER.encode()))

    (tmp_path / "2016Q2.master.gz").write_bytes(b"already")
    paths = edgar_index.download_masters(mock_client(handler), tmp_path, [(2016, 1), (2016, 2)])
    assert [p.name for p in paths] == ["2016Q1.master.gz", "2016Q2.master.gz"]
    assert calls == ["/Archives/edgar/full-index/2016/QTR1/master.gz"]


def recent(accessions: list[str], items: list[str]) -> dict[str, object]:
    n = len(accessions)
    return {
        "accessionNumber": accessions,
        "form": ["8-K"] * n,
        "items": items,
        "acceptanceDateTime": ["2016-01-26T21:31:06.000Z"] * n,
        "primaryDocument": ["d.htm"] * n,
    }


def test_fetch_and_parse_submissions(tmp_path: Path) -> None:
    pages = [
        {
            "name": "CIK0000320193-submissions-001.json",
            "filingFrom": "2010-01-01",
            "filingTo": "2016-06-30",
        },
        {
            "name": "CIK0000320193-submissions-002.json",
            "filingFrom": "1994-01-01",
            "filingTo": "2009-12-31",
        },
    ]
    bodies = {
        "/submissions/CIK0000320193.json": {
            "filings": {"recent": recent(["a-1"], ["2.02,9.01"]), "files": pages}
        },
        "/submissions/CIK0000320193-submissions-001.json": recent(["a-2", "a-1"], ["", "2.02"]),
    }
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path not in bodies:
            return httpx.Response(404)
        return httpx.Response(200, json=bodies[request.url.path])

    client = mock_client(handler)
    paths = edgar_index.fetch_submissions(
        client, tmp_path, 320193, date(2016, 1, 1), date(2026, 9, 30)
    )
    assert [p.name for p in paths] == ["recent.json", "CIK0000320193-submissions-001.json"]
    again = edgar_index.fetch_submissions(
        client, tmp_path, 320193, date(2016, 1, 1), date(2026, 9, 30)
    )
    assert again == paths and len(seen) == 2
    assert (
        edgar_index.fetch_submissions(client, tmp_path, 1, date(2016, 1, 1), date(2017, 1, 1)) == []
    )
    frame = edgar_index.parse_submissions(320193, paths).sort("accession")
    assert frame["accession"].to_list() == ["a-1", "a-2"]
    assert frame["items"].to_list() == [["2.02", "9.01"], []]
    assert frame["submissions_acceptance_utc"][0] == datetime(2016, 1, 26, 21, 31, 6, tzinfo=UTC)
    assert edgar_index.parse_submissions(1, []).is_empty()


def test_submissions_errors_other_than_404_raise(tmp_path: Path) -> None:
    from llm_text_alpha.data.edgar import SecHttpError

    client = SecClient(
        UA, max_rps=10, transport=httpx.MockTransport(lambda r: httpx.Response(400)), max_attempts=1
    )
    with pytest.raises(SecHttpError):
        edgar_index.fetch_submissions(client, tmp_path, 5, date(2016, 1, 1), date(2017, 1, 1))


def test_build_filings_index(tmp_path: Path) -> None:
    master = edgar_index.parse_master(MASTER).filter(pl.col("cik") == 320193)
    recent_path = tmp_path / "recent.json"
    recent_path.write_text(
        json.dumps({"filings": {"recent": recent(["0001193125-16-438421"], ["2.02,9.01"])}}),
        "utf-8",
    )
    submissions = edgar_index.parse_submissions(320193, [recent_path])
    index = edgar_index.build_filings_index(master, submissions)
    by_accession = {row["accession"]: row for row in index.iter_rows(named=True)}
    assert by_accession["0001193125-16-438421"]["items"] == ["2.02", "9.01"]
    assert by_accession["0001181431-16-000001"]["items"] == []
