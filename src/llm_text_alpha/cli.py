"""The ``lta`` command line (section 10).

Every command takes ``--config`` and ``--run-id``. Commands whose milestone has not landed
exit with code 2 and say which milestone brings them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import polars as pl
import typer

from llm_text_alpha import verify as verify_mod
from llm_text_alpha.config import DEFAULT_CONFIG, SMOKE_CONFIG, load_smoke, load_yaml
from llm_text_alpha.data import factors as french
from llm_text_alpha.data.smoke import run_smoke, smoke_report_path
from llm_text_alpha.io import write_parquet_atomic
from llm_text_alpha.logging import configure_logging, log_path
from llm_text_alpha.settings import DataPlacementError, Settings, check_data_placement, get_settings
from llm_text_alpha.tracking.trial_log import DEFAULT_LOG, new_run_id

app = typer.Typer(help="LLM and NLP alpha with point-in-time discipline.", no_args_is_help=True)
data_app = typer.Typer(help="Download and ingest data (section 7e).", no_args_is_help=True)
app.add_typer(data_app, name="data")

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
        ("edgar-index", "M2", "Download quarterly EDGAR indexes"),
        ("edgar-fetch", "M2", "Fetch filing headers, primary documents and EX-99.x"),
        ("prices", "M3", "Build prices_daily, universe_monthly and link_table"),
        ("earnings", "M3", "Build earnings with SUE"),
        ("lm-dict", "M3", "Load and validate the Loughran-McDonald dictionary"),
        ("gdelt", "M5", "Stream GDELT GKG records matched to the universe"),
    ],
    "corpus": [
        ("build", "M2", "Build the timestamped corpus"),
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
    "corpus": "Build, mask and audit the timestamped corpus.",
    "score": "Score documents with the dictionary, FinBERT and LLMs.",
    "llm": "LLM spend estimates, ledger and batches.",
    "signals": "Build signals through the timestamp gate.",
    "eval": "Evaluation: abnormal returns, event studies, regressions, portfolios.",
    "report": "Regenerate figures and tables.",
}

for _group_name, _commands in _LATER.items():
    _group = (
        data_app
        if _group_name == "data"
        else typer.Typer(help=_GROUP_HELP[_group_name], no_args_is_help=True)
    )
    if _group is not data_app:
        app.add_typer(_group, name=_group_name)
    for _name, _milestone, _summary in _commands:
        _placeholder(_group, _name, _milestone, _summary)
_placeholder(app, "repro", "M7", "Rebuild signals, evaluation and reports from data and cache")
