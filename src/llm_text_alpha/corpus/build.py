"""``lta corpus build``: one ``documents`` row per Item 2.02 8-K (FR-06, DQ-01, DQ-05, DQ-13).

Text is the primary 8-K document followed by its EX-99.x exhibits in filing order, each
under a section tag such as ``[[EX-99.1]]``. ``event_ts_utc`` is the SGML header's
acceptance time. Identifiers are linked as of the event through eqpit's ``PitStore``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import structlog

from llm_text_alpha.corpus.parse import extract_text
from llm_text_alpha.corpus.timestamps import NEW_YORK
from llm_text_alpha.data.edgar_fetch import DONE, ITEM_202
from llm_text_alpha.ids import permno_of

log = structlog.get_logger(__name__)

DOC_TYPE_ITEM202 = "8k_item202"
SECTION = "[[{}]]"
TS = pl.Datetime("ns", "UTC")
DOCUMENT_SCHEMA = pl.Schema(
    [
        ("doc_id", pl.String()),
        ("source", pl.String()),
        ("native_id", pl.String()),
        ("cik", pl.Int64()),
        ("ciks", pl.List(pl.Int64())),
        ("doc_type", pl.String()),
        ("form", pl.String()),
        ("items", pl.List(pl.String())),
        ("event_ts_utc", TS),
        ("event_ts_source", pl.String()),
        ("acceptance_source", pl.String()),
        ("acceptance_raw", pl.String()),
        ("company_name", pl.String()),
        ("former_names", pl.List(pl.String())),
        ("sections", pl.List(pl.String())),
        ("text", pl.String()),
        ("n_chars", pl.Int64()),
        ("n_tokens_est", pl.Int64()),
        ("text_sha256", pl.String()),
    ]
)
FAILURE_SCHEMA = pl.Schema(
    [
        ("accession", pl.String()),
        ("cik", pl.Int64()),
        ("reason", pl.String()),
        ("n_chars", pl.Int64()),
    ]
)


def doc_id(source: str, native_id: str, section: str) -> str:
    """``sha256(source|native_id|section)``, first 24 hex characters."""
    return hashlib.sha256(f"{source}|{native_id}|{section}".encode()).hexdigest()[:24]


def window_bounds(start: date, end: date) -> tuple[datetime, datetime]:
    """UTC instants of 00:00 New York on ``start`` and on the day after ``end``."""
    first = datetime.combine(start, time(), NEW_YORK).astimezone(UTC)
    last = datetime.combine(end + timedelta(days=1), time(), NEW_YORK).astimezone(UTC)
    return first, last


def iter_manifests(raw_root: Path) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Every fetched filing's folder and manifest, in path order."""
    for path in sorted(raw_root.glob(f"*/*/{DONE}")):
        yield path.parent, json.loads(path.read_text("utf-8"))


def unique_filings(raw_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    """One folder per accession.

    A filing with two filers in the universe is fetched under each filer's CIK. The copy
    under the header's first filer is kept, else the lowest CIK; the document carries every
    filer's CIK, and linking picks the one in the universe at the event date.
    """
    chosen: dict[str, tuple[Path, dict[str, Any]]] = {}
    for folder, manifest in iter_manifests(raw_root):
        accession = manifest["accession"]
        first_filer = (manifest.get("ciks") or [manifest["cik"]])[0]
        held = chosen.get(accession)
        if held is None:
            chosen[accession] = (folder, manifest)
            continue
        held_first = int(held[1]["cik"]) == int(first_filer)
        mine_first = int(manifest["cik"]) == int(first_filer)
        if (mine_first and not held_first) or (
            mine_first == held_first and int(manifest["cik"]) < int(held[1]["cik"])
        ):
            chosen[accession] = (folder, manifest)
    return sorted(chosen.values(), key=lambda item: item[0])


@dataclass(frozen=True)
class Extracted:
    """Result of extracting one filing: a document row, or a failure reason."""

    row: dict[str, Any] | None
    failure: str | None
    n_chars: int = 0


def extract_filing(
    folder: Path,
    manifest: dict[str, Any],
    min_chars: int,
    accepted: datetime,
    acceptance_source: str = "header",
) -> Extracted:
    """Build the document row for one fetched filing, published at ``accepted`` (UTC)."""
    if manifest["form"] != "8-K" or ITEM_202 not in manifest["items"]:
        return Extracted(None, "header_not_item202")
    documents = sorted(manifest["documents"], key=lambda d: int(d["sequence"]))
    if not documents:
        return Extracted(None, "no_documents")
    parts: list[str] = []
    sections: list[str] = []
    for document in documents:
        body = extract_text((folder / document["filename"]).read_bytes(), document["filename"])
        if body:
            sections.append(document["type"])
            parts.append(f"{SECTION.format(document['type'])}\n{body}")
    text = "\n\n".join(parts)
    if not text:
        return Extracted(None, "empty_text")
    if len(text) < min_chars:
        return Extracted(None, "short_text", len(text))
    accession = manifest["accession"]
    return Extracted(
        {
            "doc_id": doc_id("edgar", accession, DOC_TYPE_ITEM202),
            "source": "edgar",
            "native_id": accession,
            "cik": int(manifest["cik"]),
            "ciks": [int(c) for c in manifest.get("ciks") or [manifest["cik"]]],
            "doc_type": DOC_TYPE_ITEM202,
            "form": manifest["form"],
            "items": list(manifest["items"]),
            "event_ts_utc": accepted,
            "event_ts_source": "acceptance",
            "acceptance_source": acceptance_source,
            "acceptance_raw": manifest["acceptance_raw"],
            "company_name": manifest["company_name"],
            "former_names": list(manifest["former_names"]),
            "sections": sections,
            "text": text,
            "n_chars": len(text),
            "n_tokens_est": len(text) // 4,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        },
        None,
        len(text),
    )


@dataclass
class BuildResult:
    """Documents, failures, and how many filings fell outside the event window."""

    documents: pl.DataFrame
    failures: pl.DataFrame
    outside_window: int


def build_item202(
    raw_root: Path,
    start: date,
    end: date,
    min_chars: int = 200,
    fallback: dict[str, datetime] | None = None,
) -> BuildResult:
    """Extract every fetched Item 2.02 filing whose acceptance falls in the event window.

    A header with an empty acceptance time takes the submissions JSON's value from
    ``fallback`` (accession to UTC time), with ``acceptance_source = submissions_json``;
    DQ-02 found the two sources agree. With neither, the filing fails as
    ``no_acceptance``.
    """
    first, last = window_bounds(start, end)
    rows: list[dict[str, Any]] = []
    failures: list[tuple[str, int, str, int]] = []
    outside = 0
    for n, (folder, manifest) in enumerate(unique_filings(raw_root), start=1):
        source = "header"
        if manifest["acceptance_ts_utc"]:
            accepted = datetime.fromisoformat(manifest["acceptance_ts_utc"])
        elif fallback and manifest["accession"] in fallback:
            accepted, source = fallback[manifest["accession"]], "submissions_json"
        else:
            failures.append((manifest["accession"], int(manifest["cik"]), "no_acceptance", 0))
            continue
        if not first <= accepted < last:
            outside += 1
            continue
        result = extract_filing(folder, manifest, min_chars, accepted, source)
        if result.row is None:
            failures.append(
                (manifest["accession"], int(manifest["cik"]), str(result.failure), result.n_chars)
            )
        else:
            rows.append(result.row)
        if n % 5000 == 0:
            log.info("corpus_build_progress", filings=n, documents=len(rows))
    documents = pl.DataFrame(rows, schema=DOCUMENT_SCHEMA) if rows else DOCUMENT_SCHEMA.to_frame()
    return BuildResult(
        documents=mark_duplicates(documents),
        failures=pl.DataFrame(failures, schema=FAILURE_SCHEMA, orient="row"),
        outside_window=outside,
    )


def mark_duplicates(documents: pl.DataFrame) -> pl.DataFrame:
    """Flag a document whose text repeats an earlier one (DQ-05); the earliest stays."""
    return (
        documents.sort("event_ts_utc", "doc_id")
        .with_columns(
            (pl.int_range(pl.len()).over("text_sha256") > 0).alias("is_duplicate"),
            pl.col("form").str.ends_with("/A").alias("is_amendment"),
        )
        .with_columns((pl.col("is_duplicate") | pl.col("is_amendment")).alias("excluded"))
    )


def link_documents(documents: pl.DataFrame, pit_root: Path, top_n: int) -> pl.DataFrame:
    """Add ``permno``, ``ticker``, ``security_id`` and universe membership as of each event.

    ``permno`` is the number in the security id, Sharadar's permaticker for ``SHR:<n>``
    (section 7b allows a provider id). Each filer CIK in ``ciks`` is a candidate. The one
    ranked highest in eqpit's universe at the last month end before the event wins, and its
    top security is used; with no candidate in the universe, the first filer with a common
    security valid on the event date is linked. ``cik`` becomes the chosen filer.
    ``in_universe`` is DQ-06: the chosen company ranked in the top ``top_n`` at that month
    end. Unlinked documents keep null identifiers and never become signals.
    """
    from eqpit.store import PitStore

    store = PitStore.open(pit_root)
    keyed = documents.with_columns(
        pl.col("event_ts_utc").dt.convert_time_zone("America/New_York").dt.date().alias("_day")
    )
    days = keyed["_day"].unique().sort().to_list()
    month_end_of: dict[date, date] = {}
    for day in days:
        before = date(day.year, day.month, 1) - timedelta(days=1)
        month_end_of[day] = store.month_ends(date(before.year, before.month, 1), before)[-1]
    ranks = pl.concat(
        [
            store.universe(m, top_n)
            .select("security_id", "company_id", "rank")
            .with_columns(pl.lit(m).alias("_month_end"))
            for m in sorted(set(month_end_of.values()))
        ]
    ).filter(pl.col("company_id").str.starts_with("CIK:"))
    ranked = (
        ranks.with_columns(
            pl.col("company_id").str.strip_prefix("CIK:").cast(pl.Int64).alias("cik")
        )
        .sort("rank")
        .group_by("cik", "_month_end", maintain_order=True)
        .agg(pl.col("security_id").first().alias("_universe_security"), pl.col("rank").first())
    )
    securities = pl.concat(
        [
            store.securities(day)
            .filter(pl.col("cik").is_not_null())
            .select("security_id", "cik", "ticker", "share_type")
            .with_columns(pl.lit(day).alias("_day"))
            for day in days
        ]
    )
    common = (
        securities.sort(pl.col("share_type") != "common", "security_id")
        .group_by("cik", "_day", maintain_order=True)
        .agg(pl.col("security_id").first().alias("_fallback_security"))
    )
    month_ends = pl.DataFrame(
        {"_day": list(month_end_of), "_month_end": list(month_end_of.values())},
        schema={"_day": pl.Date, "_month_end": pl.Date},
    )
    candidates = (
        keyed.select("doc_id", "_day", "cik", "ciks")
        .with_columns(
            pl.when(pl.col("ciks").list.len() > 0)
            .then(pl.col("ciks"))
            .otherwise(pl.concat_list("cik"))
            .alias("_cand")
        )
        .with_columns(pl.int_ranges(pl.col("_cand").list.len()).alias("_order"))
        .select("doc_id", "_day", "_cand", "_order")
        .explode("_cand", "_order")
        .rename({"_cand": "cik"})
        .join(month_ends, on="_day", how="left")
        .join(ranked, on=["cik", "_month_end"], how="left")
        .join(common, on=["cik", "_day"], how="left")
    )
    chosen = (
        candidates.sort(
            pl.col("rank").is_null(),
            "rank",
            pl.col("_fallback_security").is_null(),
            "_order",
        )
        .group_by("doc_id", maintain_order=True)
        .first()
        .drop("_day", "_order")
    )
    linked = (
        keyed.drop("cik")
        .join(chosen, on="doc_id", how="left")
        .with_columns(
            pl.coalesce("_universe_security", "_fallback_security").alias("security_id"),
            pl.col("rank").is_not_null().alias("in_universe"),
            pl.col("rank").alias("universe_rank"),
        )
        .join(
            securities.select("security_id", "_day", "ticker"),
            on=["security_id", "_day"],
            how="left",
        )
        .with_columns(permno_of(pl.col("security_id")).alias("permno"))
        .rename({"_month_end": "universe_month_end"})
        .drop("_day", "_universe_security", "_fallback_security", "rank")
    )
    return linked.sort("event_ts_utc", "doc_id")


def write_documents(documents: pl.DataFrame, root: Path) -> list[Path]:
    """Write ``documents/source=<src>/year=YYYY/part-0.parquet``, replacing older parts."""
    paths: list[Path] = []
    with_year = documents.with_columns(pl.col("event_ts_utc").dt.year().alias("_year"))
    for (source, year), part in with_year.group_by("source", "_year", maintain_order=True):
        target = root / f"source={source}" / f"year={year}" / "part-0.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        part.drop("_year").write_parquet(temporary, compression="zstd")
        temporary.replace(target)
        paths.append(target)
    return paths


def header_table(raw_root: Path) -> pl.DataFrame:
    """Header fields of every fetched filing, for ``filings_index`` (section 7b)."""
    rows = [
        {
            "accession": manifest["accession"],
            "header_items": list(manifest["items"]),
            "acceptance_raw": manifest["acceptance_raw"],
            "acceptance_ts_utc": (
                datetime.fromisoformat(manifest["acceptance_ts_utc"])
                if manifest["acceptance_ts_utc"]
                else None
            ),
            "primary_doc_url": next(
                (
                    f"{_folder_url(manifest)}/{d['filename']}"
                    for d in manifest["documents"]
                    if d["type"] == manifest["form"]
                ),
                None,
            ),
            "exhibit_urls": [
                f"{_folder_url(manifest)}/{d['filename']}"
                for d in manifest["documents"]
                if d["type"].upper().startswith("EX-99")
            ],
        }
        for _, manifest in unique_filings(raw_root)
    ]
    schema = pl.Schema(
        [
            ("accession", pl.String()),
            ("header_items", pl.List(pl.String())),
            ("acceptance_raw", pl.String()),
            ("acceptance_ts_utc", TS),
            ("primary_doc_url", pl.String()),
            ("exhibit_urls", pl.List(pl.String())),
        ]
    )
    return pl.DataFrame(rows, schema=schema) if rows else schema.to_frame()


def _folder_url(manifest: dict[str, Any]) -> str:
    from llm_text_alpha.data.edgar import accession_folder

    return accession_folder(int(manifest["cik"]), manifest["accession"])


def dst_transitions(first_year: int, last_year: int) -> list[tuple[date, str]]:
    """US daylight-saving changes: second Sunday of March, first Sunday of November."""
    found: list[tuple[date, str]] = []
    for year in range(first_year, last_year + 1):
        march = date(year, 3, 8 + (6 - date(year, 3, 8).weekday()) % 7)
        november = date(year, 11, 1 + (6 - date(year, 11, 1).weekday()) % 7)
        found += [(march, "spring_forward"), (november, "fall_back")]
    return found


def acceptance_check(documents: pl.DataFrame, index: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """DQ-02: header acceptance against the submissions JSON, read as UTC.

    Returns ``by_year`` (agreement by year and EST/EDT), ``by_transition`` (filings within
    7 days of each DST change) and ``examples`` (the filings nearest each change on both
    sides).
    """
    joined = (
        documents.filter(pl.col("acceptance_source") == "header")
        .select("native_id", "event_ts_utc")
        .join(
            index.select(pl.col("accession").alias("native_id"), "submissions_acceptance_utc"),
            on="native_id",
            how="left",
        )
        .with_columns(
            pl.col("event_ts_utc").dt.convert_time_zone("America/New_York").alias("_local"),
        )
        .with_columns(
            pl.col("_local").dt.date().alias("local_date"),
            pl.when(
                pl.col("_local").dt.base_utc_offset() + pl.col("_local").dt.dst_offset()
                == pl.duration(hours=-4)
            )
            .then(pl.lit("EDT"))
            .otherwise(pl.lit("EST"))
            .alias("regime"),
            pl.col("submissions_acceptance_utc").is_null().alias("missing"),
            (pl.col("submissions_acceptance_utc") == pl.col("event_ts_utc")).alias("agree"),
        )
    )
    by_year = (
        joined.group_by(pl.col("local_date").dt.year().alias("year"), "regime")
        .agg(
            pl.len().alias("filings"),
            pl.col("agree").sum().alias("agree"),
            (~pl.col("agree") & ~pl.col("missing")).sum().alias("disagree"),
            pl.col("missing").sum().alias("missing_in_json"),
        )
        .sort("year", "regime")
    )
    years = joined["local_date"].dt.year().drop_nulls().to_list() or [2016]
    transitions = pl.DataFrame(
        dst_transitions(min(years), max(years)),
        schema={"transition": pl.Date, "kind": pl.String},
        orient="row",
    )
    near = joined.join(transitions, how="cross").filter(
        (pl.col("local_date") - pl.col("transition")).dt.total_days().abs() <= 7
    )
    by_transition = (
        near.group_by("transition", "kind")
        .agg(
            pl.len().alias("filings_within_7d"),
            pl.col("agree").sum().alias("agree"),
            (~pl.col("agree") & ~pl.col("missing")).sum().alias("disagree"),
        )
        .sort("transition")
    )
    before = near.filter(pl.col("local_date") < pl.col("transition")).sort("event_ts_utc")
    after = near.filter(pl.col("local_date") >= pl.col("transition")).sort("event_ts_utc")
    examples = (
        pl.concat(
            [
                before.group_by("transition", maintain_order=True).last(),
                after.group_by("transition", maintain_order=True).first(),
            ]
        )
        .select(
            "transition",
            "kind",
            pl.col("native_id").alias("accession"),
            "regime",
            pl.col("event_ts_utc").dt.strftime("%Y-%m-%dT%H:%M:%SZ").alias("header_utc"),
            pl.col("submissions_acceptance_utc")
            .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            .alias("submissions_utc"),
            "agree",
        )
        .sort("transition", "header_utc")
    )
    return {"by_year": by_year, "by_transition": by_transition, "examples": examples}


def corpus_summary(
    result: BuildResult, linked: pl.DataFrame, fetched: int, selected: int
) -> pl.DataFrame:
    """Counts behind FR-02's 98% extraction threshold and the linking coverage."""
    in_window = result.documents.height + result.failures.height
    failures = result.failures["reason"].value_counts().sort("reason")
    rows: list[tuple[str, float]] = [
        ("item202_filings_selected_from_index", float(selected)),
        ("filings_fetched", float(fetched)),
        ("filings_outside_event_window", float(result.outside_window)),
        ("filings_in_window", float(in_window)),
        ("documents_extracted", float(result.documents.height)),
        ("extraction_success_rate", result.documents.height / in_window if in_window else 0.0),
    ]
    rows += [(f"failed_{r}", float(n)) for r, n in failures.iter_rows()]
    rows += [
        ("duplicate_texts_excluded", float(result.documents["is_duplicate"].sum())),
        ("documents_linked_to_a_security", float(linked["security_id"].is_not_null().sum())),
        ("documents_with_permno", float(linked["permno"].is_not_null().sum())),
        ("documents_in_universe_at_prior_month_end", float(linked["in_universe"].sum())),
        ("median_chars", float(result.documents.select(pl.col("n_chars").median()).item() or 0)),
    ]
    return pl.DataFrame(rows, schema={"metric": pl.String, "value": pl.Float64}, orient="row")
