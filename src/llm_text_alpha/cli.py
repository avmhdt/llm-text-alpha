"""The ``lta`` command line (section 10).

Every command takes ``--config`` and ``--run-id``. Commands whose milestone has not landed
exit with code 2 and say which milestone brings them.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Annotated

import polars as pl
import structlog
import typer

from llm_text_alpha import verify as verify_mod
from llm_text_alpha.config import (
    DEFAULT_CONFIG,
    SMOKE_CONFIG,
    load_smoke,
    load_universe,
    load_yaml,
)
from llm_text_alpha.corpus import build as corpus
from llm_text_alpha.data import edgar_index
from llm_text_alpha.data import factors as french
from llm_text_alpha.data.edgar import SecClient
from llm_text_alpha.data.edgar_fetch import fetch_all, select_item202
from llm_text_alpha.data.prices import EfpPrices
from llm_text_alpha.data.ratelimit import ByteThrottle
from llm_text_alpha.data.smoke import run_smoke, smoke_report_path
from llm_text_alpha.io import write_parquet_atomic
from llm_text_alpha.logging import configure_logging, log_path
from llm_text_alpha.reporting.tables import write_table
from llm_text_alpha.settings import DataPlacementError, Settings, check_data_placement, get_settings
from llm_text_alpha.tracking.trial_log import DEFAULT_LOG, new_run_id

app = typer.Typer(help="LLM and NLP alpha with point-in-time discipline.", no_args_is_help=True)
data_app = typer.Typer(help="Download and ingest data (section 7e).", no_args_is_help=True)
app.add_typer(data_app, name="data")
corpus_app = typer.Typer(help="Build, mask and audit the timestamped corpus.", no_args_is_help=True)
app.add_typer(corpus_app, name="corpus")

LOGS_DIR = Path("logs")
ConfigOpt = Annotated[Path, typer.Option("--config", help="Experiment config YAML.")]
RunIdOpt = Annotated[str | None, typer.Option("--run-id", help="Run id; generated if omitted.")]


def _start(command: str, run_id: str | None) -> tuple[Settings, str]:
    """Load settings, check where data goes, and open this command's JSON log."""
    settings = get_settings()
    resolved = run_id or new_run_id()
    configure_logging(settings.log_level, settings.secrets(), log_path(LOGS_DIR, command, resolved))
    try:
        check_data_placement(settings)
    except DataPlacementError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    return settings, resolved


@data_app.command("smoke")
def data_smoke(config: ConfigOpt = SMOKE_CONFIG, run_id: RunIdOpt = None) -> None:
    """M1 smoke test: 10 EDGAR filings, one month of prices, Ken French factors (AC-W-04)."""
    settings, resolved = _start("data smoke", run_id)
    report = run_smoke(settings, load_smoke(config), resolved)
    for check in report.checks:
        mark = {True: "PASS", False: "FAIL", None: "INFO"}[check.ok]
        typer.echo(f"{mark}  {check.name}: {check.detail}")
    typer.echo(f"report: {smoke_report_path(settings.data_dir, resolved)}")
    if not report.passed:
        raise typer.Exit(1)


@data_app.command("factors")
def data_factors(config: ConfigOpt = DEFAULT_CONFIG, run_id: RunIdOpt = None) -> None:
    """Download Ken French daily FF5 and momentum; build factors_daily (FR-04)."""
    settings, _ = _start("data factors", run_id)
    snapshot = french.download_factors(settings.data_dir / "raw" / "factors")
    table = french.build_factors_daily(snapshot)
    target = settings.data_dir / "processed" / "panels" / "factors_daily.parquet"
    write_parquet_atomic(table, target)
    first, last = table.select(
        pl.col("date").min().alias("first"), pl.col("date").max().alias("last")
    ).row(0)
    typer.echo(f"{table.height} days, {first} to {last}: {target}")


@data_app.command("edgar-index")
def data_edgar_index(
    start: Annotated[str, typer.Option(help="First quarter, such as 2016Q1.")] = "2016Q1",
    end: Annotated[str, typer.Option(help="Last quarter, such as 2026Q3.")] = "2026Q3",
    forms: Annotated[str, typer.Option(help="Forms; amendments are added.")] = "8-K,10-K,10-Q",
    config: ConfigOpt = DEFAULT_CONFIG,
    run_id: RunIdOpt = None,
) -> None:
    """Quarterly master indexes and submissions JSON, filtered to universe CIKs (FR-01)."""
    settings, _ = _start("data edgar-index", run_id)
    universe = load_universe(config)
    if settings.efp_root is None:
        typer.echo("error: universe CIKs come from eqpit's store; set EFP_DATA_DIR", err=True)
        raise typer.Exit(1)
    first = universe.start - timedelta(days=31)
    ciks = EfpPrices(settings.efp_root).universe_ciks(first, universe.end, universe.top_n_by_mcap)
    interim = settings.data_dir / "interim" / "edgar"
    write_parquet_atomic(ciks, interim / "universe_ciks.parquet")
    raw = settings.data_dir / "raw" / "edgar"
    throttle = ByteThrottle()
    with SecClient.from_settings(settings) as client:
        masters = edgar_index.download_masters(
            client, raw / "full-index", edgar_index.quarters(start, end), throttle
        )
        master = edgar_index.read_masters(
            masters, edgar_index.with_amendments(forms.split(",")), ciks["cik"].to_list()
        )
        parts = []
        for n, cik in enumerate(master["cik"].unique().sort().to_list(), start=1):
            paths = edgar_index.fetch_submissions(
                client, raw / "submissions", cik, universe.start, universe.end
            )
            parts.append(edgar_index.parse_submissions(cik, paths))
            if n % 200 == 0:
                structlog.get_logger(__name__).info("edgar_submissions_progress", companies=n)
    index = edgar_index.build_filings_index(master, pl.concat(parts) if parts else master.clear())
    write_parquet_atomic(index, interim / "filings_index.parquet")
    item202 = select_item202(index, universe.start, universe.end).height
    typer.echo(
        f"{ciks.height} universe CIKs, {index.height} filings indexed, "
        f"{item202} Item 2.02 8-Ks to fetch: {interim / 'filings_index.parquet'}"
    )


@data_app.command("edgar-fetch")
def data_edgar_fetch(
    doc_types: Annotated[
        str, typer.Option(help="Document types; 8k_item202 in M2.")
    ] = "8k_item202",
    limit: Annotated[int | None, typer.Option(help="Stop after this many filings.")] = None,
    resume: Annotated[bool, typer.Option(help="Skip filings already fetched.")] = True,
    workers: Annotated[int, typer.Option(help="Threads sharing the one rate limiter.")] = 3,
    config: ConfigOpt = DEFAULT_CONFIG,
    run_id: RunIdOpt = None,
) -> None:
    """Fetch headers, primary documents and EX-99.x exhibits; resumable (FR-01, NFR-09)."""
    if doc_types != "8k_item202":
        typer.echo("error: only 8k_item202 is fetched in this milestone", err=True)
        raise typer.Exit(2)
    settings, _ = _start("data edgar-fetch", run_id)
    universe = load_universe(config)
    index_path = settings.data_dir / "interim" / "edgar" / "filings_index.parquet"
    if not index_path.exists():
        typer.echo("error: run `lta data edgar-index` first", err=True)
        raise typer.Exit(1)
    filings = select_item202(pl.read_parquet(index_path), universe.start, universe.end)
    with SecClient.from_settings(settings) as client:
        summary = fetch_all(
            client,
            settings.data_dir / "raw" / "edgar" / "filings",
            filings,
            limit=limit,
            resume=resume,
            throttle=ByteThrottle(),
            workers=workers,
        )
    typer.echo(
        f"selected {summary.selected}, fetched {summary.fetched}, already done "
        f"{summary.skipped_done}, failed {summary.failed}, {summary.seconds:.0f} s"
    )
    if summary.failed:
        raise typer.Exit(1)


@corpus_app.command("build")
def corpus_build(
    doc_types: Annotated[
        str, typer.Option(help="Document types; 8k_item202 in M2.")
    ] = "8k_item202",
    start: Annotated[str | None, typer.Option(help="First event date (default: config).")] = None,
    end: Annotated[str | None, typer.Option(help="Last event date (default: config).")] = None,
    config: ConfigOpt = DEFAULT_CONFIG,
    run_id: RunIdOpt = None,
) -> None:
    """Build documents with UTC event times, links and dedupe; write corpus tables (FR-06)."""
    if doc_types != "8k_item202":
        typer.echo("error: only 8k_item202 is built in this milestone", err=True)
        raise typer.Exit(2)
    settings, resolved = _start("corpus build", run_id)
    universe = load_universe(config)
    first = date.fromisoformat(start) if start else universe.start
    last = date.fromisoformat(end) if end else universe.end
    min_chars = int(load_yaml(config).get("corpus", {}).get("min_chars", 200))
    raw = settings.data_dir / "raw" / "edgar" / "filings"
    interim = settings.data_dir / "interim"
    index_path = interim / "edgar" / "filings_index.parquet"
    if settings.efp_root is None or not index_path.exists():
        typer.echo("error: needs EFP_DATA_DIR and `lta data edgar-index` output", err=True)
        raise typer.Exit(1)
    index = pl.read_parquet(index_path)
    headers = corpus.header_table(raw)
    index = index.drop([c for c in headers.columns if c != "accession" and c in index.columns])
    write_parquet_atomic(index.join(headers, on="accession", how="left"), index_path)
    known = index.filter(pl.col("submissions_acceptance_utc").is_not_null())
    fallback = dict(zip(known["accession"], known["submissions_acceptance_utc"], strict=True))
    result = corpus.build_item202(raw, first, last, min_chars, fallback)
    linked = corpus.link_documents(
        result.documents, settings.efp_root / "processed" / "pit", universe.top_n_by_mcap
    )
    corpus.write_documents(linked, interim / "corpus" / "documents")
    write_parquet_atomic(result.failures, interim / "corpus" / "extraction_failures.parquet")
    command = f"lta corpus build --doc-types {doc_types}"
    tables = Path("reports/tables")
    selected = select_item202(index, universe.start, universe.end).height
    summary = corpus.corpus_summary(result, linked, headers.height, selected)
    write_table(summary, tables, "corpus_item202", command, resolved)
    for name, frame in corpus.acceptance_check(result.documents, index).items():
        write_table(frame, tables, f"dq02_acceptance_{name}", command, resolved)
    rate = summary.filter(pl.col("metric") == "extraction_success_rate")["value"][0]
    typer.echo(f"{result.documents.height} documents; extraction success {rate:.2%}")


@app.command("verify")
def verify(
    ac_id: Annotated[str, typer.Argument(help="Criterion id such as ac-w-11, or 'all'.")],
    config: ConfigOpt = DEFAULT_CONFIG,
    run_id: RunIdOpt = None,
) -> None:
    """Run the check behind an acceptance criterion (section 12)."""
    load_yaml(config)
    repo = Path.cwd()

    def ac_w_04() -> verify_mod.Outcome:
        settings, resolved = _start("verify ac-w-04", run_id)
        report = run_smoke(settings, load_smoke(SMOKE_CONFIG), resolved)
        failed = [check.name for check in report.checks if check.ok is False]
        status = "pass" if report.passed else "fail"
        detail = "failed: " + ", ".join(failed) if failed else f"{len(report.checks)} checks ok"
        return verify_mod.Outcome("ac-w-04", status, detail)

    checks = {
        "ac-w-04": ac_w_04,
        "ac-w-01": lambda: verify_mod.check_gate_properties(repo),
        "ac-w-11": lambda: verify_mod.check_trial_log(repo / DEFAULT_LOG),
        "ac-w-12": lambda: verify_mod.check_hygiene(repo),
    }
    outcomes = verify_mod.run(ac_id, checks)
    for outcome in outcomes:
        typer.echo(f"{outcome.status.upper():8} {outcome.ac_id}: {outcome.detail}")
    if any(outcome.status == "fail" for outcome in outcomes):
        raise typer.Exit(1)


def _placeholder(group: typer.Typer, name: str, milestone: str, summary: str) -> None:
    def command(config: ConfigOpt = DEFAULT_CONFIG, run_id: RunIdOpt = None) -> None:
        typer.echo(f"'{name}' arrives in {milestone}.", err=True)
        raise typer.Exit(2)

    command.__doc__ = f"{summary} (arrives in {milestone})."
    group.command(name)(command)


_LATER: dict[str, list[tuple[str, str, str]]] = {
    "data": [
        ("prices", "M3", "Build prices_daily, universe_monthly and link_table"),
        ("earnings", "M3", "Build earnings with SUE"),
        ("lm-dict", "M3", "Load and validate the Loughran-McDonald dictionary"),
        ("gdelt", "M5", "Stream GDELT GKG records matched to the universe"),
    ],
    "corpus": [
        ("mask", "M6", "Mask entities and dates"),
        ("audit-mask", "M6", "Report residual masking hits"),
    ],
    "score": [
        ("dict", "M3", "Loughran-McDonald dictionary scores"),
        ("finbert", "M3", "FinBERT scores"),
        ("llm", "M4", "LLM scores through the provider abstraction"),
    ],
    "llm": [
        ("estimate", "M4", "Print docs, tokens and worst-case USD without spending"),
        ("spend", "M4", "Print ledger totals against LLM_BUDGET_USD"),
        ("batches", "M4", "Resume open batches"),
    ],
    "signals": [("build", "M2", "Run the timestamp gate")],
    "eval": [
        ("abnormal", "M3", "Abnormal returns"),
        ("event-study", "M3", "Event study with confidence bands"),
        ("regress", "M3", "Cross-sectional and Fama-MacBeth regressions"),
        ("portfolio", "M5", "Calendar-time long-short portfolio"),
        ("ablation", "M5", "Ablation on the intersection event set"),
        ("contamination", "M6", "Contamination protocol"),
        ("capacity", "M5", "Capacity analysis"),
        ("latency", "M6", "Latency budget"),
        ("trials", "M5", "Trial count and Deflated Sharpe Ratio"),
    ],
    "report": [("build", "M7", "Figures and tables")],
}

_GROUP_HELP = {
    "score": "Score documents with the dictionary, FinBERT and LLMs.",
    "llm": "LLM spend estimates, ledger and batches.",
    "signals": "Build signals through the timestamp gate.",
    "eval": "Evaluation: abnormal returns, event studies, regressions, portfolios.",
    "report": "Regenerate figures and tables.",
}

for _group_name, _commands in _LATER.items():
    _group = {"data": data_app, "corpus": corpus_app}.get(_group_name) or typer.Typer(
        help=_GROUP_HELP[_group_name], no_args_is_help=True
    )
    if _group_name not in ("data", "corpus"):
        app.add_typer(_group, name=_group_name)
    for _name, _milestone, _summary in _commands:
        _placeholder(_group, _name, _milestone, _summary)
_placeholder(app, "repro", "M7", "Rebuild signals, evaluation and reports from data and cache")
