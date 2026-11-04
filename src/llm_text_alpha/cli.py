"""The ``lta`` command line (section 10).

Every command takes ``--config`` and ``--run-id``. Commands whose milestone has not landed
exit with code 2 and say which milestone brings them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

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
from llm_text_alpha.tracking.trial_log import DEFAULT_LOG, new_run_id, record_run

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
llm_app = typer.Typer(help="LLM spend estimates, ledger and batches.", no_args_is_help=True)
app.add_typer(llm_app, name="llm")

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
    index = pl.read_parquet(index_path).unique(["accession", "cik"], keep="first")
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


@dataclass(frozen=True)
class _LLMJob:
    provider: Any
    requests: list[Any]
    prompt_version: str
    sample: str
    n_docs: int
    cache_root: Path


def _llm_job(
    settings: Settings,
    config: Path,
    provider: str,
    model: str,
    sample: str,
    masked: bool,
    max_docs: int | None,
    run_id: str,
    cache_root: Path | None = None,
) -> _LLMJob:
    from llm_text_alpha.scoring.llm import prompts as llm_prompts
    from llm_text_alpha.scoring.llm import run as llm_run
    from llm_text_alpha.scoring.llm.cache import ResponseCache
    from llm_text_alpha.scoring.llm.registry import ModelRegistry

    if masked:
        typer.echo("error: masked text arrives in M6", err=True)
        raise typer.Exit(2)
    cfg = load_yaml(config)
    llm_cfg = cfg["scorers"]["llm"]
    spec = llm_cfg["samples"].get(sample)
    if spec is None:
        typer.echo(f"error: no sample {sample!r} in {config}", err=True)
        raise typer.Exit(2)
    registry = ModelRegistry.load(Path(llm_cfg.get("models", "configs/models.yaml")))
    model_spec = registry.get(model)
    if model_spec.provider != provider:
        typer.echo(f"error: {model} is a {model_spec.provider} model, not {provider}", err=True)
        raise typer.Exit(2)
    docs = pipeline.scorable(pipeline.load_documents(pipeline.Paths.of(settings))).filter(
        pl.col("doc_type") == spec.get("doc_type", "8k_item202")
    )
    if spec.get("before"):
        cutoff = datetime.fromisoformat(str(spec["before"])).replace(tzinfo=UTC)
        docs = docs.filter(pl.col("event_ts_utc") < cutoff)
    if spec.get("after"):
        start = datetime.fromisoformat(str(spec["after"])).replace(tzinfo=UTC)
        docs = docs.filter(pl.col("event_ts_utc") >= start)
    chosen = llm_run.draw_sample(
        docs, int(spec["n"]), spec.get("stratify", "year"), int(spec["seed"])
    )
    if max_docs is not None:
        chosen = chosen.head(max_docs)
    prompt = llm_prompts.load_prompt(Path(llm_cfg["prompt"]))
    max_chars = int(int(llm_cfg["max_input_tokens"]) * float(llm_cfg["chars_per_token"]))
    requests = llm_prompts.build_requests(
        chosen, model, prompt, max_chars, int(llm_cfg["max_output_tokens"]), model_spec.request
    )
    root = cache_root or settings.llm_cache_path
    instance = llm_run.make_provider(
        provider,
        settings,
        registry,
        ResponseCache(root),
        run_id,
        Path(cfg["budget"]["price_table"]),
    )
    return _LLMJob(instance, requests, prompt.version, sample, chosen.height, root)


ProviderOpt = Annotated[str, typer.Option(help="anthropic, local or fake.")]
ModelOpt = Annotated[str, typer.Option(help="Model id from configs/models.yaml.")]
SampleOpt = Annotated[str, typer.Option(help="Sample name from the config.")]
ModeOpt = Annotated[str, typer.Option(help="batch or sync.")]


@llm_app.command("estimate")
def llm_estimate(
    provider: ProviderOpt = "anthropic",
    model: ModelOpt = "claude-haiku-5-5",
    sample: SampleOpt = "item202_main",
    masked: Annotated[bool, typer.Option(help="Masked text (M6).")] = False,
    mode: ModeOpt = "batch",
    max_docs: Annotated[int | None, typer.Option(help="Score only the first N.")] = None,
    config: ConfigOpt = DEFAULT_CONFIG,
    run_id: RunIdOpt = None,
) -> None:
    """Docs, tokens, worst-case USD and remaining budget; never spends or calls the API."""
    settings, resolved = _start("llm estimate", run_id)
    job = _llm_job(settings, config, provider, model, sample, masked, max_docs, resolved)
    estimate = job.provider.estimate_cost(job.requests, mode)
    lines = {
        "sample": sample,
        "model": model,
        "mode": mode,
        "documents": job.n_docs,
        "already_cached": estimate.n_cached,
        "worst_case_input_tokens": estimate.input_tokens,
        "max_output_tokens": estimate.max_output_tokens,
        "worst_case_usd": round(estimate.worst_case_usd, 4),
    }
    if provider == "anthropic":
        tracker = job.provider.tracker
        lines["committed_usd"] = round(tracker.committed_usd(), 4)
        lines["budget_usd"] = tracker.budget_usd
        lines["remaining_usd"] = round(tracker.budget_usd - tracker.committed_usd(), 4)
    for key, value in lines.items():
        typer.echo(f"{key}: {value}")


@llm_app.command("calibrate")
def llm_calibrate(
    model: ModelOpt = "claude-haiku-5-5",
    sample: SampleOpt = "item202_main",
    n: Annotated[int, typer.Option(help="Documents to count.")] = 200,
    config: ConfigOpt = DEFAULT_CONFIG,
    run_id: RunIdOpt = None,
) -> None:
    """Characters per token of the document text, from the free token-counting endpoint."""
    import dataclasses

    settings, resolved = _start("llm calibrate", run_id)
    job = _llm_job(settings, config, "anthropic", model, sample, False, n, resolved)
    base = job.provider.count_tokens(dataclasses.replace(job.requests[0], user_text="."))
    ratios = []
    for request in job.requests:
        tokens = job.provider.count_tokens(request) - base + 1
        ratios.append(len(request.user_text) / max(tokens, 1))
    series = pl.Series("chars_per_token", ratios)
    summary = {
        "documents": series.len(),
        "prompt_overhead_tokens": base,
        **{f"p{q}": round(float(series.quantile(q / 100) or 0.0), 3) for q in (1, 5, 25, 50, 75)},
    }
    record_run(resolved, {"command": "llm calibrate", "model": model, "n": n}, summary)
    for key, value in summary.items():
        typer.echo(f"{key}: {value}")


@llm_app.command("spend")
def llm_spend(
    by: Annotated[str, typer.Option(help="model or run.")] = "model",
    config: ConfigOpt = DEFAULT_CONFIG,
    run_id: RunIdOpt = None,
) -> None:
    """Settled spend from the ledger against LLM_BUDGET_USD."""
    from llm_text_alpha.scoring.llm import run as llm_run

    settings, _ = _start("llm spend", run_id)
    tracker = llm_run.tracker(settings, Path(load_yaml(config)["budget"]["price_table"]))
    for key, usd in tracker.totals(by):
        typer.echo(f"{key}: ${usd:.4f}")
    committed = tracker.committed_usd()
    typer.echo(f"committed ${committed:.4f} of ${tracker.budget_usd:.2f} (open holds included)")


@llm_app.command("batches")
def llm_batches(
    provider: ProviderOpt = "anthropic",
    model: ModelOpt = "claude-haiku-5-5",
    sample: SampleOpt = "item202_main",
    poll: Annotated[bool, typer.Option(help="Show open batches.")] = True,
    collect: Annotated[bool, typer.Option(help="Wait for and collect open batches.")] = False,
    config: ConfigOpt = DEFAULT_CONFIG,
    run_id: RunIdOpt = None,
) -> None:
    """Resume batches left open by a crash: poll, collect and settle, never resubmit."""
    settings, resolved = _start("llm batches", run_id)
    job = _llm_job(settings, config, provider, model, sample, False, None, resolved)
    batches_file = job.cache_root / "batches.json"
    open_ids: list[str] = []
    if batches_file.exists():
        entries = json.loads(batches_file.read_text("utf-8"))
        open_ids = [b for b, e in entries.items() if e.get("status") == "submitted"]
    if poll:
        typer.echo(f"{len(open_ids)} open batches: {', '.join(open_ids) or '-'}")
    if collect and open_ids:
        results = job.provider.resume(job.requests)
        typer.echo(f"collected {len(results)} results")


@score_app.command("llm")
def score_llm(
    provider: ProviderOpt = "fake",
    model: ModelOpt = "fake-v1",
    sample: SampleOpt = "item202_main",
    masked: Annotated[bool, typer.Option(help="Masked text (M6).")] = False,
    mode: ModeOpt = "batch",
    max_docs: Annotated[int | None, typer.Option(help="Score only the first N.")] = None,
    dry_run: Annotated[bool, typer.Option(help="Estimate only.")] = False,
    bypass_cache: Annotated[
        bool, typer.Option(help="D-06 check only: score into a fresh cache under this run id.")
    ] = False,
    config: ConfigOpt = DEFAULT_CONFIG,
    run_id: RunIdOpt = None,
) -> None:
    """Score a sample through the provider abstraction, cache and spend tracker (FR-10)."""
    from llm_text_alpha.scoring.llm import run as llm_run

    settings, resolved = _start("score llm", run_id)
    root = settings.llm_cache_path / "rescore" / resolved if bypass_cache else None
    job = _llm_job(settings, config, provider, model, sample, masked, max_docs, resolved, root)
    estimate = job.provider.estimate_cost(job.requests, mode)
    plan = {
        "sample": sample,
        "model": model,
        "provider": provider,
        "mode": mode,
        "documents": job.n_docs,
        "already_cached": estimate.n_cached,
        "worst_case_input_tokens": estimate.input_tokens,
        "worst_case_usd": estimate.worst_case_usd,
        "prompt_version": job.prompt_version,
        "bypass_cache": bypass_cache,
    }
    typer.echo(f"estimate: {json.dumps(plan, sort_keys=True)}")
    if dry_run:
        return
    config_record = {"command": "score llm", "config": str(config), **plan}
    record_run(resolved, config_record, {"estimate": plan, "status": "started"}, DEFAULT_LOG)
    started = datetime.now(UTC)
    results = job.provider.score(job.requests, mode)
    run = llm_run.tables(results, provider, job.prompt_version, masked, resolved)
    if bypass_cache:
        scores_dir = job.cache_root / "scores"
        calls_dir = job.cache_root / "llm_calls"
    else:
        model_dir = f"model={model}/masked={str(masked).lower()}/sample={sample}"
        scores_dir = pipeline.Paths.of(settings).scores("llm") / model_dir
        calls_dir = settings.llm_cache_path / "llm_calls"
    llm_run.write_tables(run, scores_dir, calls_dir, resolved)
    calls = run.calls
    metrics = {
        "estimate": plan,
        "started_at": started.isoformat(),
        "ended_at": datetime.now(UTC).isoformat(),
        "requests": calls.height,
        "cache_hits": int((calls["mode"] == "cache_hit").sum()),
        "schema_valid": int(calls["schema_valid"].sum()),
        "refusals": int((calls["stop_reason"] == "refusal").sum()),
        "retried": int((calls["attempts"] > 1).sum()),
        "usd": float(calls["usd"].sum()),
        "input_tokens": int(calls["input_tokens"].sum()),
        "cache_read_tokens": int(calls["cache_read_tokens"].sum()),
        "output_tokens": int(calls["output_tokens"].sum()),
    }
    metrics["status"] = "done"
    record_run(resolved, config_record, metrics, DEFAULT_LOG)
    typer.echo(
        f"{metrics['requests']} requests, {metrics['schema_valid']} valid, "
        f"{metrics['cache_hits']} cached, ${metrics['usd']:.4f}"
    )


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
    scorers: Annotated[str, typer.Option(help="Comma-separated scorers.")] = "lm,finbert,sue,llm",
    buffer_min: Annotated[int, typer.Option(help="Minutes between publication and entry.")] = 5,
    config: ConfigOpt = DEFAULT_CONFIG,
    run_id: RunIdOpt = None,
) -> None:
    """Run the timestamp gate and trailing standardization; fails on any violation (FR-07)."""
    settings, _ = _start("signals build", run_id)
    llm = load_yaml(config).get("signals", {}).get("llm_scorers", {})
    signals = pipeline.build_signals(
        pipeline.Paths.of(settings), scorers.split(","), buffer_min, llm
    )
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
    if "sue" in scorers.split(","):
        pead = pipeline.run_pead_check(pipeline.Paths.of(settings), model, report)
        spread = pead.row(-1, named=True)
        typer.echo(f"PEAD: Q5-Q1 SUE CAR[1,60] {spread['car_1_60']:.4%} (t {spread['t']:.2f})")
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
            t = row["t"][0]
            t_text = "n/a" if t is None else f"{t:.2f}"
            typer.echo(f"{scorer}: Q5-Q1 CAR[0,5] {row['car'][0]:.4%} (t {t_text})")


@eval_app.command("regress")
def eval_regress(
    scorers: Annotated[str, typer.Option(help="Comma-separated scorers.")] = "lm,finbert",
    horizons: Annotated[str, typer.Option(help="k in CAR[0,k].")] = "1,5,20,60",
    se: Annotated[str, typer.Option(help="twoway, fm, or both.")] = "both",
    model: Annotated[str, typer.Option(help="Abnormal-return model.")] = "ff5umd",
    window: Annotated[str, typer.Option(help="Event window from the config.")] = "all",
    config: ConfigOpt = DEFAULT_CONFIG,
    run_id: RunIdOpt = None,
) -> None:
    """CAR[0,k] on score_z with size, momentum and SUE controls (FR-17)."""
    settings, resolved = _start("eval regress", run_id)
    command = f"lta eval regress --scorers {scorers} --se {se} --window {window}"
    report = _report(command, resolved, config)
    methods = ["twoway", "fm"] if se == "both" else [se]
    chosen = (window, _windows(config)[window][0])
    suffix = "" if window == "all" else f"_{window}"
    paths = pipeline.Paths.of(settings)
    table = pipeline.run_regressions(
        paths,
        scorers.split(","),
        model,
        [int(k) for k in horizons.split(",")],
        report,
        methods,
        chosen,
        f"regressions{suffix}",
    )
    write_table(
        pipeline.sue_correlation(paths, scorers.split(","), chosen),
        report.tables,
        f"sue_correlation{suffix}",
        command,
        resolved,
    )
    typer.echo(f"{table.height} regressions written to {report.tables / f'regressions{suffix}.md'}")


def _windows(config: Path) -> dict[str, tuple[date | None, date | None]]:
    raw = load_yaml(config)["evaluation"]["portfolio"]["windows"]
    return {
        name: tuple(date.fromisoformat(str(v)) if v else None for v in bounds)  # type: ignore[misc]
        for name, bounds in raw.items()
    }


def _context(settings: Settings, config: Path) -> Any:
    from llm_text_alpha import evaluate

    ev = load_yaml(config)["evaluation"]
    return evaluate.context(
        pipeline.Paths.of(settings),
        {str(k): float(v) for k, v in ev["costs"]["bps_by_size_rank"].items()},
        int(ev["newey_west_lags"]["portfolio"]),
        str(ev["ar_model"]),
    )


@eval_app.command("portfolio")
def eval_portfolio(
    scorers: Annotated[str, typer.Option(help="Comma-separated scorers.")] = "llm,finbert,lm,sue",
    config: ConfigOpt = DEFAULT_CONFIG,
    run_id: RunIdOpt = None,
) -> None:
    """Calendar-time long-short net of costs; alpha on FF5 + UMD + PEAD, Newey-West t (FR-18)."""
    from llm_text_alpha import evaluate

    settings, resolved = _start("eval portfolio", run_id)
    ev = load_yaml(config)["evaluation"]
    report = _report(f"lta eval portfolio --scorers {scorers}", resolved, config)
    table = evaluate.run_portfolios(
        _context(settings, config),
        scorers.split(","),
        _windows(config),
        [int(h) for h in ev["portfolio"]["hold_days"]],
        [float(m) for m in ev["costs"]["sensitivity_mult"]],
        report,
    )
    head = ev["portfolio"]["headline"]
    if table.is_empty():
        typer.echo("no portfolio had 20 or more active days")
        return
    row = table.filter(
        (pl.col("scorer") == head["scorer"])
        & (pl.col("window") == head["window"])
        & (pl.col("hold_days") == int(head["hold_days"]))
        & (pl.col("cost_mult") == float(head["cost_mult"]))
    )
    typer.echo(f"{table.height} portfolios written to {report.tables / 'portfolio_alpha.md'}")
    if row.height:
        r = row.row(0, named=True)
        verdict = "met" if r["t_alpha_net"] > 3 else "not met"
        typer.echo(
            f"headline {head}: alpha {r['alpha_annual_net']:.2%}/yr, t {r['t_alpha_net']:.2f} "
            f"over {r['days']} days ({verdict})"
        )


@eval_app.command("ablation")
def eval_ablation(
    event_set: Annotated[str, typer.Option(help="Only 'intersection'.")] = "intersection",
    config: ConfigOpt = DEFAULT_CONFIG,
    run_id: RunIdOpt = None,
) -> None:
    """Dictionary vs FinBERT vs LLM and their combinations on identical events (FR-19)."""
    from llm_text_alpha import evaluate

    if event_set != "intersection":
        typer.echo("error: --event-set is intersection", err=True)
        raise typer.Exit(2)
    settings, resolved = _start("eval ablation", run_id)
    cfg = load_yaml(config)
    ab = cfg["evaluation"]["ablation"]
    scorers = [str(s) for s in ab["scorers"]]
    llm_chars = int(
        int(cfg["scorers"]["llm"]["max_input_tokens"])
        * float(cfg["scorers"]["llm"]["chars_per_token"])
    )
    fin = cfg["scorers"]["finbert"]
    reads = {
        "lm": "all EX-99 text",
        "finbert": f"first {fin.get('max_chunks')} chunks of {fin['max_tokens']} tokens",
        "llm": f"first {llm_chars:,} characters "
        f"(about {cfg['scorers']['llm']['max_input_tokens']} tokens)",
    }
    paths = pipeline.Paths.of(settings)
    report = _report("lta eval ablation --event-set intersection", resolved, config)
    table = evaluate.run_ablation(
        _context(settings, config),
        scorers,
        _windows(config),
        int(ab["hold_days"]),
        report,
        reads,
        evaluate.cost_per_document(paths, [s for s in scorers if s != "llm"])
        | {"llm": _llm_unit_cost(paths, cfg)},
        int(ab.get("min_events", 50)),
    )
    typer.echo(f"{table.height} rows written to {report.tables / 'ablation.md'}")


def _llm_unit_cost(paths: pipeline.Paths, cfg: dict[str, Any]) -> float:
    spec = cfg["signals"]["llm_scorers"]["llm"]
    scores = pipeline.load_llm_scores(paths, "llm", spec["model"], spec["samples"])
    return float(scores["cost_usd"].fill_null(0.0).to_numpy().mean())


@eval_app.command("capacity")
def eval_capacity(config: ConfigOpt = DEFAULT_CONFIG, run_id: RunIdOpt = None) -> None:
    """Net alpha against AUM under participation and square-root impact (FR-21)."""
    from llm_text_alpha import evaluate

    settings, resolved = _start("eval capacity", run_id)
    c = load_yaml(config)["evaluation"]["capacity"]
    report = _report("lta eval capacity", resolved, config)
    window = str(c["window"])
    _, points = evaluate.run_capacity(
        _context(settings, config),
        str(c["scorer"]),
        (window, _windows(config)[window]),
        int(c["hold_days"]),
        [float(a) for a in c["aum_usd"]],
        float(c["participation"]),
        int(c["adv_days"]),
        float(c["impact_coef"]),
        report,
    )
    typer.echo(f"capacity: {points}")


@eval_app.command("trials")
def eval_trials(config: ConfigOpt = DEFAULT_CONFIG, run_id: RunIdOpt = None) -> None:
    """Trial count by family and the Deflated Sharpe Ratio of the headline strategy (FR-30)."""
    from llm_text_alpha import evaluate

    _, resolved = _start("eval trials", run_id)
    head = load_yaml(config)["evaluation"]["portfolio"]["headline"]
    report = _report("lta eval trials", resolved, config)
    counts, deflated = evaluate.run_trials(report, head)
    typer.echo(f"{int(counts['trials'].sum())} trials logged")
    if deflated is not None:
        typer.echo(
            f"headline DSR {deflated.dsr:.3f} (daily Sharpe {deflated.sr:.4f} vs "
            f"threshold {deflated.sr0:.4f}, {deflated.n_trials} portfolio trials)"
        )


@app.command("verify")
def verify(
    ac_id: Annotated[str, typer.Argument(help="Criterion id such as ac-w-11, or 'all'.")],
    config: ConfigOpt = DEFAULT_CONFIG,
    run_id: RunIdOpt = None,
) -> None:
    """Run the check behind an acceptance criterion (section 12)."""
    cfg = load_yaml(config)
    repo = Path.cwd()

    def ac_w_04() -> verify_mod.Outcome:
        settings, resolved = _start("verify ac-w-04", run_id)
        report = run_smoke(settings, load_smoke(SMOKE_CONFIG), resolved)
        failed = [check.name for check in report.checks if check.ok is False]
        status = "pass" if report.passed else "fail"
        detail = "failed: " + ", ".join(failed) if failed else f"{len(report.checks)} checks ok"
        return verify_mod.Outcome("ac-w-04", status, detail)

    def ac_w_10() -> verify_mod.Outcome:
        settings, _ = _start("verify ac-w-10", run_id)
        return verify_mod.check_cost_latency(
            pipeline.Paths.of(settings).root / "processed" / "scores",
            settings.llm_cache_path / "llm_calls",
        )

    checks = {
        "ac-w-04": ac_w_04,
        "ac-w-05": lambda: verify_mod.run_pytest(
            "ac-w-05", repo, ["tests/integration/test_cache_rerun.py"]
        ),
        "ac-w-06": lambda: verify_mod.run_pytest(
            "ac-w-06",
            repo,
            [
                "tests/unit/test_spend.py",
                "tests/property/test_spend.py",
                "tests/unit/test_no_direct_client.py",
                "tests/unit/test_anthropic_provider.py::test_budget_cap_blocks_before_any_network_call",
            ],
        ),
        "ac-w-10": ac_w_10,
        "ac-w-03": lambda: verify_mod.check_ablation(
            repo / "reports" / "tables", [str(s) for s in cfg["evaluation"]["ablation"]["scorers"]]
        ),
        "ac-h-01": lambda: verify_mod.check_headline(
            repo / "reports" / "tables", dict(cfg["evaluation"]["portfolio"]["headline"])
        ),
        "ac-w-01": lambda: verify_mod.check_gate_properties(repo),
        "ac-w-02": lambda: verify_mod.check_figures(
            repo / "reports" / "figures", ["lm", "finbert", "llm"]
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
    "eval": [
        ("contamination", "M6", "Contamination protocol"),
        ("latency", "M6", "Latency budget"),
    ],
    "report": [("build", "M7", "Figures and tables")],
}

_GROUP_HELP = {
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
