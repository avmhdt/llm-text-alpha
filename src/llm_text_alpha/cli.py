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


@corpus_app.command("mask")
def corpus_mask(
    samples: Annotated[str, typer.Option(help="Comma-separated LLM samples to mask.")] = (
        "item202_main,item202_post_h55,item202_contam"
    ),
    processes: Annotated[int, typer.Option(help="spaCy worker processes (<= 8).")] = 4,
    config: ConfigOpt = DEFAULT_CONFIG,
    run_id: RunIdOpt = None,
) -> None:
    """Mask entities and dates in the LLM's input window of each sampled document (FR-14)."""
    from llm_text_alpha.corpus.mask import Entity, Masker, mask_many
    from llm_text_alpha.scoring.common import scoring_text
    from llm_text_alpha.scoring.llm.prompts import truncate

    if not 1 <= processes <= 8:
        typer.echo("error: --processes must be 1 to 8", err=True)
        raise typer.Exit(2)
    settings, _ = _start("corpus mask", run_id)
    cfg = load_yaml(config)
    mcfg = _mask_config(cfg)
    llm_cfg = cfg["scorers"]["llm"]
    max_chars = int(int(llm_cfg["max_input_tokens"]) * float(llm_cfg["chars_per_token"]))
    docs = (
        pl.concat(
            [
                _sample_docs(settings, cfg, s.strip()).select(
                    "doc_id", "text", "company_name", "former_names", "ticker", "cik"
                )
                for s in samples.split(",")
            ]
        )
        .unique("doc_id", keep="first")
        .sort("doc_id")
    )
    windows = [truncate(scoring_text(text), max_chars)[0] for text in docs["text"]]
    entities = [
        Entity(tuple([r["company_name"] or "", *(r["former_names"] or [])]), r["ticker"], r["cik"])
        for r in docs.iter_rows(named=True)
    ]
    from llm_text_alpha.scoring.dictionary import english_words

    english = english_words(settings.data_path(cfg["scorers"]["lm"]["dictionary_file"]))
    masker = Masker(mcfg, Masker.load_nlp(mcfg.ner_model), english)
    results = mask_many(masker, windows, entities, n_process=processes)
    kinds = list(mcfg.placeholders)
    table = pl.DataFrame(
        {
            "doc_id": docs["doc_id"],
            "mask_version": [mcfg.version] * docs.height,
            "text_masked": [r.text for r in results],
            **{f"n_{k}": [r.counts.get(k, 0) for r in results] for k in kinds},
            "residual_name_hits": [r.residual_name_hits for r in results],
            "residual_year_hits": [r.residual_year_hits for r in results],
            "residual_word_hits": [r.residual_word_hits for r in results],
            "passes": [r.passes for r in results],
            "raw_chars": [len(w) for w in windows],
        },
        schema_overrides={f"n_{k}": pl.Int32 for k in kinds}
        | {
            "residual_name_hits": pl.Int32,
            "residual_year_hits": pl.Int32,
            "residual_word_hits": pl.Int32,
        },
    )
    folder = _masked_folder(settings, mcfg.version)
    write_parquet_atomic(table, folder / "part-00000.parquet")
    (folder / "masking.yaml").write_text(Path(cfg["masking"]["config"]).read_text("utf-8"), "utf-8")
    leaked = int((table["residual_name_hits"] > 0).sum())
    typer.echo(
        f"{table.height} documents masked ({mcfg.version}); {leaked} with residual name hits"
    )


@corpus_app.command("audit-mask")
def corpus_audit_mask(
    review: Annotated[int, typer.Option(help="Random documents written out for review.")] = 50,
    config: ConfigOpt = DEFAULT_CONFIG,
    run_id: RunIdOpt = None,
) -> None:
    """Residual name, ticker and year hits (AC-W-09), and a local file for manual review."""
    settings, resolved = _start("corpus audit-mask", run_id)
    cfg = load_yaml(config)
    version = _mask_config(cfg).version
    folder = _masked_folder(settings, version)
    table = pl.read_parquet(folder / "*.parquet")
    n = table.height

    def mean(column: pl.Series) -> float:
        return float(column.cast(pl.Float64).to_numpy().mean())

    share_names = mean(table["residual_name_hits"] > 0)
    share_years = mean(table["residual_year_hits"] > 0)
    rows = [
        {"metric": "documents", "value": float(n)},
        {"metric": "share_with_residual_name_or_ticker", "value": share_names},
        {"metric": "share_with_residual_year", "value": share_years},
        {
            "metric": "share_with_lowercase_word_equal_to_short_name",
            "value": mean(table["residual_word_hits"] > 0)
            if "residual_word_hits" in table.columns
            else 0.0,
        },
        *[
            {"metric": f"mean_{c}", "value": mean(table[c])}
            for c in table.columns
            if c.startswith("n_")
        ],
        {"metric": "mean_passes", "value": mean(table["passes"])},
    ]
    command = f"lta corpus audit-mask ({version})"
    write_table(pl.DataFrame(rows), Path("reports/tables"), "mask_audit", command, resolved)
    picked = table.sample(n=min(review, n), seed=int(cfg["seed"]))
    lines = [f"# Masking review sample ({version})", ""]
    for r in picked.iter_rows(named=True):
        lines += [f"## {r['doc_id']}", "", r["text_masked"][:1500], ""]
    (folder / "review.md").write_text("\n".join(lines), "utf-8")
    verdict = "pass" if share_names <= 0.01 and share_years <= 0.01 else "fail"
    typer.echo(
        f"{n} documents: residual names {share_names:.2%}, residual years {share_years:.2%} "
        f"({verdict}); review sample at {folder / 'review.md'}"
    )


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


def _sample_docs(settings: Settings, cfg: dict[str, Any], sample: str) -> pl.DataFrame:
    """The documents of a configured sample, drawn reproducibly from the scorable corpus."""
    from llm_text_alpha.scoring.llm import run as llm_run

    spec = cfg["scorers"]["llm"]["samples"][sample]
    if spec.get("from_sample"):
        docs = _sample_docs(settings, cfg, str(spec["from_sample"]))
        return llm_run.draw_sample(
            docs,
            int(spec["n"]),
            spec.get("stratify", "year"),
            int(spec["seed"]),
            spec.get("equal_by"),
            {str(k): int(v) for k, v in spec["allocation"].items()},
        )
    docs = pipeline.scorable(pipeline.load_documents(pipeline.Paths.of(settings))).filter(
        pl.col("doc_type") == spec.get("doc_type", "8k_item202")
    )
    if spec.get("before"):
        cutoff = datetime.fromisoformat(str(spec["before"])).replace(tzinfo=UTC)
        docs = docs.filter(pl.col("event_ts_utc") < cutoff)
    if spec.get("after"):
        start = datetime.fromisoformat(str(spec["after"])).replace(tzinfo=UTC)
        docs = docs.filter(pl.col("event_ts_utc") >= start)
    if spec.get("cutoff_breaks"):
        docs = llm_run.cutoff_windows(docs, spec["cutoff_breaks"])
    return llm_run.draw_sample(
        docs, int(spec["n"]), spec.get("stratify", "year"), int(spec["seed"]), spec.get("equal_by")
    )


def _mask_config(cfg: dict[str, Any], ner: bool = True) -> Any:
    from llm_text_alpha.corpus.mask import MaskConfig

    return MaskConfig.load(Path(cfg["masking"]["config"]), ner=ner)


def _masked_folder(settings: Settings, version: str) -> Path:
    return pipeline.Paths.of(settings).root / "interim" / "masked" / f"mask_version={version}"


def _masked_texts(settings: Settings, cfg: dict[str, Any]) -> pl.DataFrame:
    version = _mask_config(cfg).version
    folder = _masked_folder(settings, version)
    if not any(folder.glob("*.parquet")):
        typer.echo(f"error: no masked documents for {version}; run lta corpus mask", err=True)
        raise typer.Exit(1)
    return pl.read_parquet(folder / "*.parquet", columns=["doc_id", "text_masked"])


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

    cfg = load_yaml(config)
    llm_cfg = cfg["scorers"]["llm"]
    if sample not in llm_cfg["samples"]:
        typer.echo(f"error: no sample {sample!r} in {config}", err=True)
        raise typer.Exit(2)
    registry = ModelRegistry.load(Path(llm_cfg.get("models", "configs/models.yaml")))
    model_spec = registry.get(model)
    if model_spec.provider != provider:
        typer.echo(f"error: {model} is a {model_spec.provider} model, not {provider}", err=True)
        raise typer.Exit(2)
    chosen = _sample_docs(settings, cfg, sample)
    if max_docs is not None:
        chosen = chosen.head(max_docs)
    max_chars = int(int(llm_cfg["max_input_tokens"]) * float(llm_cfg["chars_per_token"]))
    text_column = "text"
    if masked:
        masked_docs = _masked_texts(settings, cfg)
        chosen = chosen.drop("text").join(masked_docs, on="doc_id", how="left")
        missing = chosen["text_masked"].null_count()
        if missing:
            typer.echo(
                f"error: {missing} documents lack masked text; run lta corpus mask", err=True
            )
            raise typer.Exit(1)
        text_column = "text_masked"
        max_chars = 10**9  # the masked text is already the raw window, masked
    prompt = llm_prompts.load_prompt(Path(llm_cfg["prompt"]))
    max_chars = int(int(llm_cfg["max_input_tokens"]) * float(llm_cfg["chars_per_token"]))
    requests = llm_prompts.build_requests(
        chosen,
        model,
        prompt,
        max_chars,
        int(llm_cfg["max_output_tokens"]),
        model_spec.request,
        text_column,
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


@llm_app.command("models")
def llm_models(
    models: Annotated[str, typer.Option(help="Comma-separated Anthropic model ids.")] = (
        "claude-haiku-5-5,claude-haiku-4-5,claude-sonnet-4-6"
    ),
    config: ConfigOpt = DEFAULT_CONFIG,
    run_id: RunIdOpt = None,
) -> None:
    """Lifecycle and retirement date of each model from the Models API (D-04); no spend."""
    from llm_text_alpha.scoring.llm import run as llm_run
    from llm_text_alpha.scoring.llm.cache import ResponseCache
    from llm_text_alpha.scoring.llm.registry import ModelRegistry

    settings, resolved = _start("llm models", run_id)
    cfg = load_yaml(config)
    registry = ModelRegistry.load(Path(cfg["scorers"]["llm"].get("models", "configs/models.yaml")))
    provider: Any = llm_run.make_provider(
        "anthropic",
        settings,
        registry,
        ResponseCache(settings.llm_cache_path),
        resolved,
        Path(cfg["budget"]["price_table"]),
    )
    for model_id in models.split(","):
        info = provider.model_info(model_id)
        caps = info.get("capabilities") or {}
        structured = caps.get("structured_outputs", {}).get("supported")
        typer.echo(
            f"{model_id}: lifecycle {info.get('lifecycle')}, retires_at {info.get('retires_at')}, "
            f"deprecated_at {info.get('deprecated_at')}, structured outputs {structured}"
        )


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


def _probe_frames(settings: Settings, cfg: dict[str, Any]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The documents the probes ask about: masked texts, and events with CAR[0,5] and names."""
    from llm_text_alpha.evaluation.abnormal import cumulative

    pc = cfg["contamination"]["probes"]
    docs = _sample_docs(settings, cfg, str(pc["sample"]))
    if "cutoff_window" in docs.columns:
        per = int(pc["reidentify_n"]) // docs["cutoff_window"].n_unique()
        picked = pl.concat(
            [
                part.sample(n=min(per, part.height), seed=int(pc["seed"]))
                for _, part in docs.sort("doc_id").group_by("cutoff_window", maintain_order=True)
            ]
        )
    else:
        picked = docs.sort("doc_id").sample(n=int(pc["reidentify_n"]), seed=int(pc["seed"]))
    picked = picked.sort("doc_id")
    masked = picked.select("doc_id").join(_masked_texts(settings, cfg), on="doc_id", how="inner")
    paths = pipeline.Paths.of(settings)
    folder = paths.abnormal / f"model={cfg['evaluation']['ar_model']}"
    kept = pl.read_parquet(folder / "events.parquet").filter(pl.col("drop_reason").is_null())
    ar = pl.read_parquet(folder / "ar.parquet").join(kept.select("doc_id"), on="doc_id", how="semi")
    cars = cumulative(ar, [5]).drop("permno")
    events = picked.select(
        "doc_id", "event_ts_utc", "company_name", "former_names", "ticker", "cik"
    ).join(cars, on="doc_id", how="inner")
    return masked, events.head(int(pc["outcome_recall_n"]))


@score_app.command("probes")
def score_probes(
    provider: ProviderOpt = "anthropic",
    model: ModelOpt = "claude-haiku-5-5",
    mode: ModeOpt = "batch",
    dry_run: Annotated[bool, typer.Option(help="Estimate only.")] = False,
    config: ConfigOpt = DEFAULT_CONFIG,
    run_id: RunIdOpt = None,
) -> None:
    """Re-identification and outcome-recall probes for one model (FR-20)."""
    from llm_text_alpha.evaluation import probes
    from llm_text_alpha.scoring.llm import prompts as llm_prompts
    from llm_text_alpha.scoring.llm import run as llm_run
    from llm_text_alpha.scoring.llm.cache import ResponseCache
    from llm_text_alpha.scoring.llm.registry import ModelRegistry

    settings, resolved = _start("score probes", run_id)
    cfg = load_yaml(config)
    pc = cfg["contamination"]["probes"]
    registry = ModelRegistry.load(Path(cfg["scorers"]["llm"].get("models", "configs/models.yaml")))
    spec = registry.get(model)
    if spec.provider != provider:
        typer.echo(f"error: {model} is a {spec.provider} model, not {provider}", err=True)
        raise typer.Exit(2)
    masked, events = _probe_frames(settings, cfg)
    reid_prompt = llm_prompts.load_prompt(Path(pc["reid_prompt"]))
    recall_prompt = llm_prompts.load_prompt(Path(pc["recall_prompt"]))
    reid = probes.reid_requests(masked, model, reid_prompt, spec.request)
    recall = probes.recall_requests(events, model, recall_prompt, spec.request)
    instance: Any = llm_run.make_provider(
        provider,
        settings,
        registry,
        ResponseCache(settings.llm_cache_path),
        resolved,
        Path(cfg["budget"]["price_table"]),
    )
    plan = {
        "model": model,
        "provider": provider,
        "mode": mode,
        "reid_requests": len(reid),
        "recall_requests": len(recall),
        "reid_prompt": reid_prompt.version,
        "recall_prompt": recall_prompt.version,
        "worst_case_usd": instance.estimate_cost(reid, mode).worst_case_usd
        + instance.estimate_cost(recall, mode).worst_case_usd,
    }
    typer.echo(f"estimate: {json.dumps(plan, sort_keys=True)}")
    if dry_run:
        return
    record_run(
        resolved, {"command": "score probes", **plan}, {"estimate": plan, "status": "started"}
    )
    reid_results = instance.score(reid, mode)
    recall_results = instance.score(recall, mode)
    suffixes = _mask_config(cfg, ner=False).legal_suffixes
    earnings = pl.read_parquet(pipeline.Paths.of(settings).panels / "earnings.parquet").filter(
        pl.col("announce_doc_id").is_not_null()
    )
    quarter = {
        r["announce_doc_id"]: probes.true_quarter(r["fqtr_end"])
        for r in earnings.select("announce_doc_id", "fqtr_end").iter_rows(named=True)
    }
    info = {
        r["doc_id"]: r for r in _sample_docs(settings, cfg, str(pc["sample"])).iter_rows(named=True)
    }
    reid_rows = []
    for res in reid_results:
        doc = info[res.doc_id]
        names = [doc["company_name"] or "", *(doc["former_names"] or [])]
        ans = res.payload
        truth = quarter.get(res.doc_id)
        reid_rows.append(
            {
                "doc_id": res.doc_id,
                "event_ts_utc": doc["event_ts_utc"],
                "valid": res.schema_valid,
                "company_correct": probes.company_match(
                    ans.company if ans else None,
                    ans.ticker if ans else None,
                    names,
                    doc["ticker"],
                    suffixes,
                )
                if ans
                else None,
                "quarter_correct": (
                    (ans.period_year, ans.period_quarter) == truth if ans and truth else None
                ),
                "confidence": ans.confidence if ans else None,
                "guess_company": ans.company if ans else None,
                "guess_ticker": ans.ticker if ans else None,
                "guess_year": ans.period_year if ans else None,
                "guess_quarter": ans.period_quarter if ans else None,
                "true_company": doc["company_name"],
                "true_ticker": doc["ticker"],
                "true_year": truth[0] if truth else None,
                "true_quarter": truth[1] if truth else None,
                "usd": res.usd,
            }
        )
    recall_rows = []
    cars = {r["doc_id"]: r["car_5"] for r in events.iter_rows(named=True)}
    for res in recall_results:
        ans = res.payload
        up = cars[res.doc_id] > 0
        recall_rows.append(
            {
                "doc_id": res.doc_id,
                "event_ts_utc": info[res.doc_id]["event_ts_utc"],
                "valid": res.schema_valid,
                "direction_correct": ((ans.direction == "up") == up) if ans else None,
                "said_up": (ans.direction == "up") if ans else None,
                "truth_up": up,
                "confidence": ans.confidence if ans else None,
                "usd": res.usd,
            }
        )
    out = pipeline.Paths.of(settings).root / "processed" / "probes" / f"model={model}"
    write_parquet_atomic(pl.DataFrame(reid_rows), out / "reid.parquet")
    write_parquet_atomic(pl.DataFrame(recall_rows), out / "recall.parquet")
    spent = sum(r.usd for r in reid_results) + sum(r.usd for r in recall_results)
    metrics = {"estimate": plan, "status": "done", "usd": spent}
    record_run(resolved, {"command": "score probes", **plan}, metrics)
    typer.echo(
        f"{len(reid_rows)} re-identification and {len(recall_rows)} recall probes, ${spent:.4f}"
    )


def _prereg_sha(repo: Path) -> str:
    import subprocess

    out = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", "docs/preregistration.md"],
        capture_output=True,
        text=True,
        cwd=repo,
    )
    return out.stdout.strip()


@eval_app.command("contamination")
def eval_contamination(
    models: Annotated[str, typer.Option(help="Comma-separated model ids.")] = (
        "claude-haiku-5-5,claude-haiku-4-5,llama-3.1-8b-instruct-q4_k_m"
    ),
    probes_on: Annotated[
        bool, typer.Option("--probes/--no-probes", help="Add probe results.")
    ] = True,
    config: ConfigOpt = DEFAULT_CONFIG,
    run_id: RunIdOpt = None,
) -> None:
    """2x2 pre/post-cutoff x raw/masked tables, DiD against baselines and probes (FR-20)."""
    from llm_text_alpha.evaluation import contamination as ct
    from llm_text_alpha.evaluation import probes
    from llm_text_alpha.evaluation.abnormal import cumulative
    from llm_text_alpha.reporting.figures import contamination_figure
    from llm_text_alpha.scoring.llm.registry import ModelRegistry
    from llm_text_alpha.tracking import trial_log

    settings, resolved = _start("eval contamination", run_id)
    cfg = load_yaml(config)
    cc = cfg["contamination"]
    registry = ModelRegistry.load(Path(cfg["scorers"]["llm"].get("models", "configs/models.yaml")))
    paths = pipeline.Paths.of(settings)
    signals = pl.read_parquet(paths.signals)
    folder = paths.abnormal / f"model={cfg['evaluation']['ar_model']}"
    kept = pl.read_parquet(folder / "events.parquet").filter(pl.col("drop_reason").is_null())
    ar = pl.read_parquet(folder / "ar.parquet").join(kept.select("doc_id"), on="doc_id", how="semi")
    cars = cumulative(ar, [5]).drop("permno")
    report = _report(f"lta eval contamination --models {models}", resolved, config)
    sha = _prereg_sha(Path.cwd())
    cells_all, did_all, summary_all, probe_all = [], [], [], []
    for model_id in models.split(","):
        scorer = cc["models"][model_id]
        if signals.filter(pl.col("scorer") == scorer).is_empty():
            typer.echo(f"{model_id}: no signals for {scorer}; skipped")
            continue
        post = registry.get(model_id).post_cutoff_start
        assert post is not None
        cells = ct.two_by_two(signals, cars, scorer, post).with_columns(
            pl.lit(model_id).alias("model_id")
        )
        dids = [
            ct.did(signals, cars, scorer, b, post, int(cc["bootstrap_reps"]), int(cfg["seed"]))
            for b in cc["baselines"]
        ]
        summary = ct.summary_rows(model_id, scorer, post, cells, dids)
        if probes_on:
            pfolder = paths.root / "processed" / "probes" / f"model={model_id}"
            for kind, column in (
                ("reid", "company_correct"),
                ("reid", "quarter_correct"),
                ("recall", "direction_correct"),
            ):
                f = pfolder / f"{kind}.parquet"
                if f.exists():
                    acc = probes.accuracy(pl.read_parquet(f), column, post).with_columns(
                        pl.lit(model_id).alias("model_id"),
                        pl.lit(f"{kind}:{column}").alias("probe"),
                    )
                    probe_all.append(acc)
                    for r in acc.iter_rows(named=True):
                        summary[f"{kind}_{column}_{r['window']}"] = r["accuracy"]
        cells_all.append(cells)
        did_all += [d | {"model_id": model_id} for d in dids]
        summary_all.append(summary)
        contamination_figure(
            cells,
            f"Contamination: {model_id}",
            f"Q5 - Q1 CAR[0,5] with 95% intervals; post-cutoff from {post}; "
            f"paired events scored on raw and masked text",
            report.figures / f"contamination_{model_id}.png",
        )
        params = {
            "model_id": model_id,
            "scorer": scorer,
            "post_cutoff_start": post.isoformat(),
            "baselines": list(cc["baselines"]),
            "doc_type": "8k_item202",
            "preregistration_sha": sha,
        }
        rid = trial_log.new_run_id()
        trial_log.record_run(
            rid,
            params,
            {"cells": cells.to_dicts(), "did": dids, "summary": summary},
            report.trial_log,
        )
        trial_log.log_trial(
            rid,
            "contamination",
            params,
            None,
            None,
            0,
            None,
            None,
            report.trial_log,
            run_kind="analysis",
        )
    if not cells_all:
        typer.echo("no model had signals")
        raise typer.Exit(1)
    write_table(
        pl.concat(cells_all), report.tables, "contamination_cells", report.command, resolved
    )
    write_table(pl.DataFrame(did_all), report.tables, "contamination_did", report.command, resolved)
    write_table(pl.DataFrame(summary_all), report.tables, "contamination", report.command, resolved)
    if probe_all:
        write_table(
            pl.concat(probe_all), report.tables, "contamination_probes", report.command, resolved
        )
    typer.echo(f"{len(cells_all)} models written to {report.tables / 'contamination.md'}")


@eval_app.command("latency")
def eval_latency(
    fetch_sample: Annotated[int, typer.Option(help="Recent filings fetched live and timed.")] = 20,
    config: ConfigOpt = DEFAULT_CONFIG,
    run_id: RunIdOpt = None,
) -> None:
    """Cost per signal and the acceptance-to-order latency budget per scorer (FR-22)."""
    import shutil
    import tempfile
    import time

    import numpy as np

    from llm_text_alpha.corpus.calendar import xnys
    from llm_text_alpha.corpus.parse import extract_text
    from llm_text_alpha.data.edgar_fetch import fetch_filing
    from llm_text_alpha.evaluation import latency as lat
    from llm_text_alpha.tracking import trial_log

    settings, resolved = _start("eval latency", run_id)
    cfg = load_yaml(config)
    paths = pipeline.Paths.of(settings)
    signals = pl.read_parquet(paths.signals)
    calls_dir = settings.llm_cache_path / "llm_calls"
    calls = pl.read_parquet(calls_dir / "*.parquet") if any(calls_dir.glob("*.parquet")) else None
    rows = []
    for scorer in ("lm", "finbert"):
        scores = pipeline.load_scores(paths, scorer)
        rows.append(
            lat.cost_rows(
                scorer,
                str(scores["model_id"][0]),
                None,
                scores,
                lat.traded_share(signals, scorer),
                1.0,
            )
        )
    for alias, spec in cfg["signals"]["llm_scorers"].items():
        model_calls = None if calls is None else calls.filter(pl.col("model_id") == spec["model"])
        if model_calls is None or model_calls.is_empty():
            continue
        rows.append(
            lat.cost_rows(
                alias, spec["model"], model_calls, None, lat.traded_share(signals, alias), 2.0
            )
        )
    scoring = pl.DataFrame(rows)
    index = pl.read_parquet(paths.root / "interim" / "edgar" / "filings_index.parquet")
    recent = (
        index.filter(pl.col("items").list.contains("2.02") & (pl.col("form") == "8-K"))
        .sort("submissions_acceptance_utc", descending=True)
        .unique("accession", keep="first", maintain_order=True)
        .head(fetch_sample)
    )
    scratch = Path(tempfile.mkdtemp(dir=paths.root))
    fetch_ms: list[float] = []
    try:
        with SecClient.from_settings(settings) as client:
            for cik, accession in recent.select("cik", "accession").iter_rows():
                started = time.perf_counter()
                manifest = fetch_filing(client, scratch, int(cik), str(accession))
                for doc in manifest["documents"]:
                    files = list(scratch.rglob(doc["filename"]))
                    if files:
                        extract_text(files[0].read_bytes(), doc["filename"])
                fetch_ms.append((time.perf_counter() - started) * 1000.0)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    fetch = lat.quantiles(fetch_ms)
    calendar = xnys()
    instants = np.array([calendar.opens_ns[100] - 3_600_000_000_000], dtype=np.int64)
    decision = lat.quantiles(lat.time_calls(lambda: calendar.first_open_after(instants), 1000))
    plan = lat.budget(fetch, decision, scoring)
    docs = pipeline.load_documents(
        paths, ["doc_id", "event_ts_utc", "doc_type", "excluded"]
    ).filter((pl.col("doc_type") == "8k_item202") & ~pl.col("excluded"))
    local = docs["event_ts_utc"].dt.convert_time_zone("America/New_York")
    minutes = local.dt.hour().cast(pl.Int32) * 60 + local.dt.minute().cast(pl.Int32)
    in_session = float(((minutes >= 570) & (minutes < 960)).mean())  # type: ignore[arg-type]
    components = pl.DataFrame(
        [
            {
                "component": "EDGAR dissemination (acceptance to public)",
                "measured": False,
                **lat.quantiles([]),
            },
            {
                "component": f"EDGAR fetch and parse ({fetch['n']} live filings)",
                "measured": True,
                **fetch,
            },
            {"component": "gate and order decision", "measured": True, **decision},
        ]
    )
    report = _report("lta eval latency", resolved, config)
    write_table(scoring, report.tables, "cost_latency", report.command, resolved)
    write_table(plan, report.tables, "latency_budget", report.command, resolved)
    write_table(components, report.tables, "latency_components", report.command, resolved)
    params = {"fetch_sample": fetch_sample, "scorers": scoring["scorer"].to_list()}
    rid = trial_log.new_run_id()
    trial_log.record_run(
        rid,
        params,
        {
            "fetch": fetch,
            "decision": decision,
            "in_session_share": in_session,
            "scoring": scoring.to_dicts(),
        },
        report.trial_log,
    )
    trial_log.log_trial(
        rid, "latency", params, None, None, 0, None, None, report.trial_log, run_kind="analysis"
    )
    fetched = (
        f"fetch+parse p50 {fetch['p50_ms']:.0f} ms, p95 {fetch['p95_ms']:.0f} ms"
        if fetch["n"]
        else "no live fetch"
    )
    typer.echo(f"{fetched}; {in_session:.1%} of releases accepted during the regular session")


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
        "ac-w-09": lambda: verify_mod.check_masking(
            repo / "reports" / "tables", repo / "docs" / "contamination_protocol.md"
        ),
        "ac-h-07": lambda: verify_mod.check_preregistration(repo, repo / DEFAULT_LOG),
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
    "corpus": [],
    "eval": [],
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
