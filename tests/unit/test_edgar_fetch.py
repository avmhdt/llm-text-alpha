from __future__ import annotations

import html
import json
from datetime import date
from pathlib import Path

import httpx
import polars as pl
import pytest

from llm_text_alpha.data import edgar_fetch
from llm_text_alpha.data.edgar import SecClient

UA = "lta-tests tests@example.com"


def index_headers_page(
    accession: str, cik: int, acceptance: str, items: list[str], documents: list[tuple[str, str]]
) -> str:
    header = (
        f"<SEC-HEADER>{accession}.hdr.sgml : 20240314\n<ACCEPTANCE-DATETIME>{acceptance}\n"
        f"<ACCESSION-NUMBER>{accession}\n<TYPE>8-K\n"
        + "".join(f"<ITEMS>{item}\n" for item in items)
        + f"<FILING-DATE>{acceptance[:8] or '20181001'}\n"
        + "<FILER>\n<COMPANY-DATA>\n<CONFORMED-NAME>Synthetic Co\n"
        f"<CIK>{cik:010d}\n</COMPANY-DATA>\n<FORMER-COMPANY>\n<FORMER-CONFORMED-NAME>OLD CO\n"
        "</FORMER-COMPANY>\n</FILER>\n</SEC-HEADER>\n"
    )
    docs = "".join(
        f"<DOCUMENT>\n<TYPE>{kind}\n<SEQUENCE>{n}\n<FILENAME>{name}\n<TEXT>\n</DOCUMENT>\n"
        for n, (kind, name) in enumerate(documents, start=1)
    )
    body = html.escape(f"<SEC-DOCUMENT>{accession}-index.html\n{header}{docs}</SEC-DOCUMENT>")
    return f"<HTML><HEAD><!--\n{header}-->\n</HEAD><BODY><PRE>{body}</PRE></BODY></HTML>"


DOCS = [
    ("8-K", "d8k.htm"),
    ("EX-99.1", "ex991.htm"),
    ("EX-99.2", "ex992.pdf"),
    ("EX-101.SCH", "x.xsd"),
    ("GRAPHIC", "g1.jpg"),
    ("EX-99.3", "ex993.txt"),
]


def test_parse_index_headers_reads_the_header_once() -> None:
    page = index_headers_page("0000000001-24-000001", 1, "20240314161502", ["2.02", "9.01"], DOCS)
    header, documents = edgar_fetch.parse_index_headers(page)
    assert header.items == ("2.02", "9.01")
    assert header.former_names == ("OLD CO",)
    assert [d.filename for d in documents] == [name for _, name in DOCS]
    wanted = edgar_fetch.wanted_documents(header, documents)
    assert [d.filename for d in wanted] == ["d8k.htm", "ex991.htm", "ex993.txt"]
    with pytest.raises(ValueError, match="SEC-HEADER"):
        edgar_fetch.parse_index_headers("<html>nothing</html>")


def handler_for(pages: dict[str, str], missing: set[str] = frozenset()):  # type: ignore[assignment]
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        name = request.url.path.rsplit("/", 1)[-1]
        if name in missing:
            return httpx.Response(404)
        if name.endswith("-index-headers.html"):
            return httpx.Response(200, text=pages[name.removesuffix("-index-headers.html")])
        return httpx.Response(200, text=f"<p>body of {name}</p>")

    return handler, seen


def test_fetch_filing_writes_documents_and_manifest(tmp_path: Path) -> None:
    accession = "0000000001-24-000001"
    page = index_headers_page(accession, 1, "20240314161502", ["2.02"], DOCS)
    handler, seen = handler_for({accession: page}, missing={"ex993.txt"})
    client = SecClient(UA, max_rps=10, transport=httpx.MockTransport(handler))
    manifest = edgar_fetch.fetch_filing(client, tmp_path, 1, accession)
    folder = edgar_fetch.filing_dir(tmp_path, 1, accession)
    assert manifest["acceptance_ts_utc"] == "2024-03-14T20:15:02+00:00"
    assert [d["filename"] for d in manifest["documents"]] == ["d8k.htm", "ex991.htm"]
    assert manifest["missing"] == ["ex993.txt"]
    assert json.loads((folder / edgar_fetch.DONE).read_text("utf-8")) == manifest
    assert (folder / "ex991.htm").read_text("utf-8") == "<p>body of ex991.htm</p>"
    assert edgar_fetch.is_done(tmp_path, 1, accession)
    assert seen[0].endswith(f"/data/1/000000000124000001/{accession}-index-headers.html")


def test_fetch_all_resumes_and_counts_failures(tmp_path: Path) -> None:
    good = [f"0000000001-24-00000{n}" for n in range(1, 5)]
    pages = {a: index_headers_page(a, 1, "20240314161502", ["2.02"], DOCS[:2]) for a in good}
    handler, _ = handler_for(pages, missing={"0000000001-24-000009-index-headers.html"})
    client = SecClient(UA, max_rps=10, transport=httpx.MockTransport(handler))
    filings = pl.DataFrame({"cik": [1] * 5, "accession": [*good, "0000000001-24-000009"]})
    first = edgar_fetch.fetch_all(client, tmp_path, filings, limit=2, workers=2)
    assert (first.fetched, first.failed) == (2, 0)
    second = edgar_fetch.fetch_all(client, tmp_path, filings, workers=3, progress_every=1)
    assert (second.skipped_done, second.fetched, second.failed) == (2, 2, 1)
    third = edgar_fetch.fetch_all(client, tmp_path, filings, resume=False, limit=1)
    assert third.fetched == 1


def test_select_item202() -> None:
    index = pl.DataFrame(
        {
            "cik": [1, 1, 1, 1, 1],
            "accession": ["a", "b", "c", "d", "e"],
            "form": ["8-K", "8-K/A", "8-K", "8-K", "10-Q"],
            "items": [["2.02", "9.01"], ["2.02"], ["7.01"], ["2.02"], []],
            "filed_date": [
                date(2016, 1, 5),
                date(2016, 1, 6),
                date(2016, 1, 7),
                date(2026, 10, 2),
                date(2016, 1, 8),
            ],
        }
    )
    selected = edgar_fetch.select_item202(index, date(2016, 1, 1), date(2026, 9, 30))
    assert selected["accession"].to_list() == ["a", "d"]


def test_empty_acceptance_is_kept_for_the_corpus_to_resolve(tmp_path: Path) -> None:
    accession = "0000000001-18-000001"
    page = index_headers_page(accession, 1, "", ["2.02"], DOCS[:2])
    handler, _ = handler_for({accession: page})
    client = SecClient(UA, max_rps=10, transport=httpx.MockTransport(handler))
    manifest = edgar_fetch.fetch_filing(client, tmp_path, 1, accession)
    assert manifest["acceptance_raw"] == "" and manifest["acceptance_ts_utc"] is None
