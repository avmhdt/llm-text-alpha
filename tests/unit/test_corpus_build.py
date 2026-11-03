from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from llm_text_alpha.corpus import build
from llm_text_alpha.data.edgar_fetch import DONE

LONG = "Revenue rose. " * 40


def write_filing(
    root: Path,
    cik: int,
    accession: str,
    accepted: str,
    documents: dict[str, tuple[str, str]],
    items: list[str] | None = None,
    form: str = "8-K",
) -> None:
    folder = root / str(cik) / accession
    folder.mkdir(parents=True)
    entries = []
    for n, (filename, (kind, body)) in enumerate(documents.items(), start=1):
        (folder / filename).write_text(body, "utf-8")
        entries.append({"type": kind, "sequence": n, "filename": filename})
    manifest = {
        "accession": accession,
        "cik": cik,
        "form": form,
        "items": items if items is not None else ["2.02", "9.01"],
        "acceptance_raw": "x",
        "acceptance_ts_utc": accepted,
        "company_name": f"Company {cik}",
        "former_names": ["OLD NAME"],
        "documents": entries,
    }
    (folder / DONE).write_text(json.dumps(manifest), "utf-8")


def test_doc_id_and_window() -> None:
    assert build.doc_id("edgar", "a", "8k_item202") == build.doc_id("edgar", "a", "8k_item202")
    assert len(build.doc_id("edgar", "a", "8k_item202")) == 24
    first, last = build.window_bounds(date(2016, 1, 1), date(2026, 9, 30))
    assert first == datetime(2016, 1, 1, 5, tzinfo=UTC)
    assert last == datetime(2026, 10, 1, 4, tzinfo=UTC)


def test_build_item202(tmp_path: Path) -> None:
    good = {"d.htm": ("8-K", "<p>Cover page</p>"), "e.htm": ("EX-99.1", f"<p>{LONG}</p>")}
    write_filing(tmp_path, 1000010, "0000000001-16-000001", "2016-03-01T21:00:00+00:00", good)
    write_filing(tmp_path, 1000002, "0000000001-16-000002", "2016-03-02T21:00:00+00:00", good)
    write_filing(
        tmp_path,
        1000002,
        "0000000001-16-000003",
        "2016-03-03T21:00:00+00:00",
        {"d.htm": ("8-K", "<p>tiny</p>")},
    )
    write_filing(
        tmp_path,
        1000002,
        "0000000001-16-000004",
        "2016-03-04T21:00:00+00:00",
        {"d.htm": ("8-K", "<script>x</script>")},
    )
    write_filing(
        tmp_path, 1000002, "0000000001-16-000005", "2016-03-05T21:00:00+00:00", good, items=["7.01"]
    )
    write_filing(tmp_path, 1000002, "0000000001-16-000006", "2016-03-06T21:00:00+00:00", {})
    write_filing(tmp_path, 1000002, "0000000001-15-000007", "2015-12-31T21:00:00+00:00", good)
    result = build.build_item202(tmp_path, date(2016, 1, 1), date(2026, 9, 30))
    docs = result.documents
    assert docs.height == 2
    assert result.outside_window == 1
    first = docs.row(0, named=True)
    assert first["text"].startswith("[[8-K]]\nCover page\n\n[[EX-99.1]]\nRevenue rose.")
    assert first["sections"] == ["8-K", "EX-99.1"]
    assert first["n_tokens_est"] == first["n_chars"] // 4
    assert first["event_ts_source"] == "acceptance"
    assert docs["is_duplicate"].to_list() == [False, True]
    assert docs["excluded"].to_list() == [False, True]
    reasons = dict(zip(result.failures["accession"], result.failures["reason"], strict=True))
    assert reasons == {
        "0000000001-16-000003": "short_text",
        "0000000001-16-000004": "empty_text",
        "0000000001-16-000005": "header_not_item202",
        "0000000001-16-000006": "no_documents",
    }


def test_empty_build(tmp_path: Path) -> None:
    result = build.build_item202(tmp_path, date(2016, 1, 1), date(2016, 12, 31))
    assert result.documents.is_empty() and result.failures.is_empty()
    assert "excluded" in result.documents.columns


def test_link_and_write(tmp_path: Path, efp_root: Path) -> None:
    good = {"e.htm": ("EX-99.1", f"<p>{LONG}</p>")}
    raw = tmp_path / "raw"
    write_filing(raw, 1000010, "0000000001-16-000001", "2016-03-01T21:00:00+00:00", good)
    write_filing(
        raw,
        1000021,
        "0000000001-16-000002",
        "2016-03-02T21:00:00+00:00",
        {"e.htm": ("EX-99.1", f"<p>{LONG} other</p>")},
    )
    write_filing(
        raw,
        4242,
        "0000000001-16-000003",
        "2016-03-03T21:00:00+00:00",
        {"e.htm": ("EX-99.1", f"<p>{LONG} third</p>")},
    )
    docs = build.build_item202(raw, date(2016, 1, 1), date(2016, 12, 31)).documents
    linked = build.link_documents(docs, efp_root / "processed" / "pit", top_n=5)
    rows = {row["cik"]: row for row in linked.iter_rows(named=True)}
    assert rows[1000010]["security_id"] == "SYN:10"
    assert rows[1000010]["in_universe"] and rows[1000010]["universe_rank"] == 1
    assert rows[1000010]["universe_month_end"] == date(2016, 2, 29)
    assert rows[1000010]["ticker"] == "S10"
    assert rows[1000010]["permno"] == 10
    assert rows[1000021]["security_id"] == "SYN:22"
    assert rows[4242]["security_id"] is None and not rows[4242]["in_universe"]
    paths = build.write_documents(linked, tmp_path / "documents")
    assert [p.parent.name for p in paths] == ["year=2016"]
    assert pl.read_parquet(paths[0]).height == 3
