"""The ``lta`` command line (section 10).

Every command takes ``--config`` and ``--run-id``. Commands whose milestone has not landed
exit with code 2 and say which milestone brings them.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import polars as pl
import structlog
import typer

from llm_text_alpha import pipeline
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
from llm_text_alpha.data.prices import EfpPrices, resolve_provider
from llm_text_alpha.data.ratelimit import ByteThrottle
from llm_text_alpha.data.smoke import run_smoke, smoke_report_path
from llm_text_alpha.io import write_parquet_atomic
from llm_text_alpha.logging import configure_logging, log_path
from llm_text_alpha.reporting.tables import write_table
from llm_text_alpha.scoring import dictionary, finbert
from llm_text_alpha.settings import DataPlacementError, Settings, check_data_placement, get_settings
from llm_text_alpha.tracking.trial_log import DEFAULT_LOG, new_run_id

if TYPE_CHECKING:
    from eqpit.store import PitStore

app = typer.Typer(help="LLM and NLP alpha with point-in-time discipline.", no_args_is_help=True)
data_app = typer.Typer(help="Download and ingest data (section 7e).", no_args_is_help=True)
app.add_typer(data_app, name="data")
corpus_app = typer.Typer(help="Build, mask and audit the timestamped corpus.", no_args_is_help=True)
app.add_typer(corpus_app, name="corpus")
signals_app = typer.Typer(help="Build signals through the timestamp gate.", no_args_is_help=True)
app.add_typer(signals_app, name="signals")
eval_app = typer.Typer(
    help="Evaluation: abnormal returns, event studies, regressions, portfolios.",
    no_args_is_help=True,
)
app.add_typer(eval_app, name="eval")
score_app = typer.Typer(
    help="Score documents with the dictionary, FinBERT and LLMs.", no_args_is_help=True
)
app.add_typer(score_app, name="score")

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


@score_app.command("dict")
def score_dict(
    doc_types: Annotated[str, typer.Option(help="Document types.")] = "8k_item202",
    masked: Annotated[bool, typer.Option(help="Score masked text (M6).")] = False,
    config: ConfigOpt = DEFAULT_CONFIG,
    run_id: RunIdOpt = None,
) -> None:
    """Loughran-McDonald scores for every scorable document (FR-08)."""
    if masked:
        typer.echo("error: masked text arrives in M6", err=True)
        raise typer.Exit(2)
    settings, _ = _start("score dict", run_id)
    cfg = load_yaml(config)["scorers"]["lm"]
    lm = dictionary.load_dictionary(settings.data_path(cfg["dictionary_file"]))
    paths = pipeline.Paths.of(settings)
    docs = pipeline.scorable(pipeline.load_documents(paths)).filter(pl.col("doc_type") == doc_types)
    scores = dictionary.score_documents(docs, lm, int(cfg.get("negation_window", 3)))
    write_parquet_atomic(scores, paths.scores("lm") / "part-00000.parquet")
    typer.echo(f"{scores.height} documents scored with {lm.model_id}")


@score_app.command("finbert")
def score_finbert(
    doc_types: Annotated[str, typer.Option(help="Document types.")] = "8k_item202",
    masked: Annotated[bool, typer.Option(help="Score masked text (M6).")] = False,
    device: Annotated[str, typer.Option(help="auto, cpu or cuda.")] = "auto",
    batch_size: Annotated[int, typer.Option(help="Chunks per forward pass.")] = 16,
    threads: Annotated[int, typer.Option(help="Torch threads; modest on a shared PC.")] = 8,
    config: ConfigOpt = DEFAULT_CONFIG,
    run_id: RunIdOpt = None,
) -> None:
    """FinBERT scores, resumable; the model revision is pinned on first run (FR-09)."""
    if masked:
        typer.echo("error: masked text arrives in M6", err=True)
        raise typer.Exit(2)
    settings, _ = _start("score finbert", run_id)
    cfg = load_yaml(config)["scorers"]["finbert"]
    paths = pipeline.Paths.of(settings)
    folder, revision = finbert.download_model(paths.models, cfg["model"], cfg.get("revision"))
    model = finbert.FinBert(
        folder,
        f"{cfg['model']}@{revision}",
        device=device,
        threads=threads,
        batch_size=batch_size,
        max_tokens=int(cfg["max_tokens"]),
        stride=int(cfg["stride"]),
        max_chunks=cfg.get("max_chunks"),
    )
    docs = pipeline.scorable(pipeline.load_documents(paths)).filter(pl.col("doc_type") == doc_types)
    scored = finbert.score_resumable(model, docs, paths.scores("finbert"))
    typer.echo(f"{scored} documents scored this run with {model.model_id}")


def _efp_store(settings: Settings) -> PitStore:
    from eqpit.store import PitStore

    if settings.efp_root is None:
        typer.echo("error: this step reads eqpit's store; set EFP_DATA_DIR", err=True)
        raise typer.Exit(1)
    return PitStore.open(settings.efp_root / "processed" / "pit")


@data_app.command("prices")
def data_prices(
    provider: Annotated[str, typer.Option(help="auto, efp, sharadar, crsp or free.")] = "auto",
    config: ConfigOpt = DEFAULT_CONFIG,
    run_id: RunIdOpt = None,
) -> None:
    """Build prices_daily, universe_monthly and link_table (FR-03)."""
    settings, _ = _start("data prices", run_id)
    chosen = resolve_provider(settings, provider)
    if not isinstance(chosen, EfpPrices):
        typer.echo(f"error: panels come from efp in this milestone, not {chosen.name}", err=True)
        raise typer.Exit(1)
    universe = load_universe(config)
    last = date.fromisoformat(str(chosen.check()["last_date"]))
    counts = pipeline.build_panels(
        _efp_store(settings),
        pipeline.Paths.of(settings),
        universe.start - timedelta(days=31),
        universe.end,
        universe.estimation_data_start or universe.start,
        last,
        universe.top_n_by_mcap,
    )
    typer.echo(", ".join(f"{k} {v}" for k, v in counts.items()))


@data_app.command("earnings")
def data_earnings(
    method: Annotated[str, typer.Option(help="srw (seasonal random walk).")] = "srw",
    config: ConfigOpt = DEFAULT_CONFIG,
    run_id: RunIdOpt = None,
) -> None:
    """Build earnings with SUE and announcement times, and book equity (FR-05)."""
    if method != "srw":
        typer.echo("error: only srw; I/B/E/S needs WRDS, which the owner does not have", err=True)
        raise typer.Exit(2)
    settings, _ = _start("data earnings", run_id)
    universe = load_universe(config)
    start = (universe.estimation_data_start or universe.start) - timedelta(days=3 * 366)
    earnings = pipeline.build_earnings(
        _efp_store(settings), pipeline.Paths.of(settings), start, universe.end
    )
    with_sue = earnings["sue"].is_not_null().sum()
    linked = (earnings["announce_source"] == "8k_item202").sum()
    typer.echo(
        f"{earnings.height} company-quarters, {with_sue} with SUE, {linked} linked to an 8-K"
    )


@data_app.command("lm-dict")
def data_lm_dict(
    path: Annotated[Path | None, typer.Option(help="Master Dictionary CSV to install.")] = None,
    config: ConfigOpt = DEFAULT_CONFIG,
    run_id: RunIdOpt = None,
) -> None:
    """Validate the Loughran-McDonald Master Dictionary and copy it to data/raw/lm/ (P-14)."""
    settings, _ = _start("data lm-dict", run_id)
    target = settings.data_path(load_yaml(config)["scorers"]["lm"]["dictionary_file"])
    if path is not None:
        manifest = dictionary.install_dictionary(path, target)
        typer.echo(
            f"installed {path.name}, sha256 {manifest['sha256']}, {manifest['n_bytes']} bytes"
        )
    lm = dictionary.load_dictionary(target)
    sizes = ", ".join(f"{k} {len(v)}" for k, v in lm.words.items())
    typer.echo(f"{lm.model_id}: {sizes}")


@signals_app.command("build")
def signals_build(
    scorers: Annotated[str, typer.Option(help="Comma-separated scorers.")] = "lm,finbert,sue",
    buffer_min: Annotated[int, typer.Option(help="Minutes between publication and entry.")] = 5,
    config: ConfigOpt = DEFAULT_CONFIG,
    run_id: RunIdOpt = None,
) -> None:
    """Run the timestamp gate and trailing standardization; fails on any violation (FR-07)."""
    settings, _ = _start("signals build", run_id)
    signals = pipeline.build_signals(pipeline.Paths.of(settings), scorers.split(","), buffer_min)
    counts = signals.group_by("scorer").len().sort("scorer")
    typer.echo(", ".join(f"{s} {n}" for s, n in counts.iter_rows()) + " signals")


@eval_app.command("abnormal")
def eval_abnormal(
    model: Annotated[str, typer.Option(help="ff5umd, market or raw.")] = "ff5umd",
    config: ConfigOpt = DEFAULT_CONFIG,
    run_id: RunIdOpt = None,
) -> None:
    """Abnormal returns for every gated event (FR-15); drop counts go to drops.csv."""
    from llm_text_alpha.evaluation.abnormal import drop_counts

    if model not in ("ff5umd", "market", "raw"):
        typer.echo("error: --model is ff5umd, market or raw", err=True)
        raise typer.Exit(2)
    settings, resolved = _start("eval abnormal", run_id)
    cfg = load_yaml(config)["evaluation"]
    paths = pipeline.Paths.of(settings)
    est = (int(cfg["estimation_window"][0]), int(cfg["estimation_window"][1]))
    window = (int(cfg["event_window"][0]), int(cfg["event_window"][1]))
    counts = pipeline.run_abnormal(paths, model, est, window)  # type: ignore[arg-type]
    events = pl.read_parquet(paths.abnormal / f"model={model}" / "events.parquet")
    write_table(
        drop_counts(events),
        Path("reports/tables"),
        "drops",
        f"lta eval abnormal --model {model}",
        resolved,
    )
    typer.echo(f"{counts['events']} events, {counts['kept']} kept")


def _report(command: str, run_id: str, config: Path) -> pipeline.Report:
    cfg = load_yaml(config).get("reporting", {})
    return pipeline.Report(
        tables=Path(cfg.get("tables_dir", "reports/tables")),
        figures=Path(cfg.get("figures_dir", "reports/figures")),
        trial_log=DEFAULT_LOG,
        command=command,
        run_id=run_id,
    )


@eval_app.command("event-study")
def eval_event_study(
    scorers: Annotated[str, typer.Option(help="Comma-separated scorers.")] = "lm,finbert,sue",
    model: Annotated[str, typer.Option(help="Abnormal-return model.")] = "ff5umd",
    config: ConfigOpt = DEFAULT_CONFIG,
    run_id: RunIdOpt = None,
) -> None:
    """CAAR by trailing quintile with t, BMP and bootstrap bands, one figure per scorer."""
    settings, resolved = _start("eval event-study", run_id)
    cfg = load_yaml(config)
    report = _report(f"lta eval event-study --scorers {scorers}", resolved, config)
    for scorer in scorers.split(","):
        summary = pipeline.run_event_study(
            pipeline.Paths.of(settings),
            scorer,
            model,
            report,
            int(cfg["seed"]),
            int(cfg["evaluation"]["bootstrap_reps"]),
            list(cfg["evaluation"]["car_horizons"]),
        )
        row = summary.filter((pl.col("group") == "Q5-Q1") & (pl.col("horizon") == "CAR[0,5]"))
        if row.height:
            typer.echo(f"{scorer}: Q5-Q1 CAR[0,5] {row['car'][0]:.4%} (t {row['t'][0]:.2f})")


@eval_app.command("regress")
def eval_regress(
    scorers: Annotated[str, typer.Option(help="Comma-separated scorers.")] = "lm,finbert",
    horizons: Annotated[str, typer.Option(help="k in CAR[0,k].")] = "1,5,20,60",
    se: Annotated[str, typer.Option(help="twoway, fm, or both.")] = "both",
    model: Annotated[str, typer.Option(help="Abnormal-return model.")] = "ff5umd",
    config: ConfigOpt = DEFAULT_CONFIG,
    run_id: RunIdOpt = None,
) -> None:
    """CAR[0,k] on score_z with size, momentum and SUE controls (FR-17)."""
    settings, resolved = _start("eval regress", run_id)
    report = _report(f"lta eval regress --scorers {scorers} --se {se}", resolved, config)
    methods = ["twoway", "fm"] if se == "both" else [se]
    table = pipeline.run_regressions(
        pipeline.Paths.of(settings),
        scorers.split(","),
        model,
        [int(k) for k in horizons.split(",")],
        report,
        methods,
    )
    typer.echo(f"{table.height} regressions written to {report.tables / 'regressions.md'}")


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
        "ac-w-02": lambda: verify_mod.check_figures(
            repo / "reports" / "figures", ["lm", "finbert"]
        ),
        "ac-w-07": lambda: verify_mod.run_pytest(
            "ac-w-07",
            repo,
            ["tests/unit/test_abnormal.py", "tests/integration/test_event_study_planted.py"],
        ),
        "ac-w-08": lambda: verify_mod.run_pytest(
            "ac-w-08",
            repo,
            ["tests/unit/test_regressions.py", "tests/integration/test_null_rejection.py"],
        ),
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
        ("gdelt", "M5", "Stream GDELT GKG records matched to the universe"),
    ],
    "corpus": [
        ("mask", "M6", "Mask entities and dates"),
        ("audit-mask", "M6", "Report residual masking hits"),
    ],
    "score": [
        ("llm", "M4", "LLM scores through the provider abstraction"),
    ],
    "llm": [
        ("estimate", "M4", "Print docs, tokens and worst-case USD without spending"),
        ("spend", "M4", "Print ledger totals against LLM_BUDGET_USD"),
        ("batches", "M4", "Resume open batches"),
    ],
    "eval": [
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
    "llm": "LLM spend estimates, ledger and batches.",
    "report": "Regenerate figures and tables.",
}

for _group_name, _commands in _LATER.items():
    _known = {"data": data_app, "corpus": corpus_app, "score": score_app, "eval": eval_app}
    _group = _known.get(_group_name) or typer.Typer(
        help=_GROUP_HELP[_group_name], no_args_is_help=True
    )
    if _group_name not in _known:
        app.add_typer(_group, name=_group_name)
    for _name, _milestone, _summary in _commands:
        _placeholder(_group, _name, _milestone, _summary)
_placeholder(app, "repro", "M7", "Rebuild signals, evaluation and reports from data and cache")
