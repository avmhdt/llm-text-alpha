"""``lta verify <ac-id>``: the check behind each acceptance criterion (section 12).

Criteria whose milestone has not landed report ``pending`` with that milestone.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

from llm_text_alpha.tracking import trial_log

MAX_TRACKED_BYTES = 1_000_000


@dataclass(frozen=True)
class Outcome:
    """Result of one criterion: ``status`` is pass, fail or pending."""

    ac_id: str
    status: str
    detail: str


PENDING = {
    "ac-w-03": "M5",
    "ac-w-09": "M6",
    "ac-w-13": "M7",
    "ac-h-01": "M5",
    "ac-h-02": "M6",
    "ac-h-03": "M6",
    "ac-h-04": "M7",
    "ac-h-05": "M5",
    "ac-h-06": "M5",
    "ac-h-07": "M6",
    "ac-h-08": "M5",
}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True, cwd=repo
    ).stdout


def check_hygiene(repo: Path) -> Outcome:
    """AC-W-12, the repository part: nothing under data/, no .env, no tracked file over 1 MB.

    CI, ruff, mypy, coverage and gitleaks are the rest of AC-W-12 and run in GitHub Actions.
    """
    tracked = [line for line in _git(repo, "ls-files", "-z").split("\0") if line]
    problems = [path for path in tracked if path == "data" or path.startswith("data/")]
    problems += [path for path in tracked if Path(path).name == ".env"]
    problems += [
        f"{path} ({size} bytes)"
        for path in tracked
        if (repo / path).is_file() and (size := (repo / path).stat().st_size) > MAX_TRACKED_BYTES
    ]
    if problems:
        return Outcome("ac-w-12", "fail", "tracked: " + ", ".join(problems))
    return Outcome("ac-w-12", "pass", f"{len(tracked)} tracked files, none under data/ or > 1 MB")


def check_trial_log(log_path: Path) -> Outcome:
    """AC-W-11: the committed log's header, and a round trip through the writer."""
    if not log_path.exists():
        return Outcome("ac-w-11", "fail", f"{log_path.as_posix()} does not exist")
    try:
        trial_log.ensure_log(log_path)
    except trial_log.TrialLogError as exc:
        return Outcome("ac-w-11", "fail", str(exc))
    with log_path.open(encoding="utf-8", newline="") as handle:
        rows = sum(1 for _ in csv.reader(handle)) - 1
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "trial_log.csv"
        run_id = trial_log.new_run_id()
        days = [date(2026, 1, 2), date(2026, 1, 5), date(2026, 1, 6)]
        rets = [0.001, -0.002, 0.003]
        path = trial_log.write_returns(run_id, days, rets, probe)
        trial_log.log_trial(
            run_id,
            "verify_probe",
            {"probe": True},
            days[0],
            days[-1],
            3,
            trial_log.annual_sharpe(rets),
            path,
            probe,
        )
        trial_log.log_trial(
            trial_log.new_run_id(),
            "verify_probe",
            {"probe": True},
            None,
            None,
            0,
            None,
            None,
            probe,
            run_kind="analysis",
        )
        with probe.open(encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle))
    if tuple(header[:10]) != trial_log.REQUIRED_COLUMNS:
        return Outcome("ac-w-11", "fail", f"writer header {header}")
    return Outcome("ac-w-11", "pass", f"header ok; {rows} rows in {log_path.as_posix()}")


def run_pytest(ac_id: str, repo: Path, targets: list[str], markers: str = "") -> Outcome:
    """Pass when ``pytest`` passes on ``targets``; the detail is pytest's summary line."""
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *targets]
    if markers:
        command += ["-m", markers]
    result = subprocess.run(command, capture_output=True, text=True, cwd=repo)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    detail = lines[-1] if lines else result.stderr.strip()[-200:]
    return Outcome(ac_id, "pass" if result.returncode == 0 else "fail", detail)


def check_gate_properties(repo: Path) -> Outcome:
    """AC-W-01: the hypothesis suite, then the check on every real signal (``-m data``)."""
    prop = run_pytest("ac-w-01", repo, ["tests/property/test_gate.py"])
    full = run_pytest(
        "ac-w-01", repo, ["tests/integration/test_signal_precedes_trade_full.py"], "data"
    )
    status = "pass" if prop.status == full.status == "pass" else "fail"
    return Outcome("ac-w-01", status, f"property: {prop.detail}; full data: {full.detail}")


def check_figures(figures: Path, scorers: list[str]) -> Outcome:
    """AC-W-02: one event-study figure per scorer."""
    missing = [s for s in scorers if not (figures / f"event_study_{s}_8k_item202.png").exists()]
    if missing:
        return Outcome("ac-w-02", "fail", f"no figure for {', '.join(missing)}")
    return Outcome("ac-w-02", "pass", f"figures for {', '.join(scorers)} in {figures.as_posix()}")


def check_cost_latency(scores_root: Path, calls_root: Path) -> Outcome:
    """AC-W-10: cost on every score; latency on sync, local, FinBERT and dictionary rows;
    batch turnaround on every batch call."""
    problems: list[str] = []
    counts: list[str] = []
    for folder in sorted(scores_root.glob("scorer=*")):
        files = list(folder.rglob("*.parquet"))
        if not files:
            continue
        frame = pl.read_parquet(files, columns=["scorer", "cost_usd", "latency_ms"])
        name = folder.name.removeprefix("scorer=")
        counts.append(f"{name} {frame.height}")
        if frame["cost_usd"].null_count():
            problems.append(f"{name}: {frame['cost_usd'].null_count()} rows without cost_usd")
        if name != "llm" and frame["latency_ms"].null_count():
            problems.append(f"{name}: {frame['latency_ms'].null_count()} rows without latency_ms")
    calls_files = list(calls_root.rglob("*.parquet")) if calls_root.exists() else []
    if calls_files:
        calls = pl.read_parquet(calls_files)
        counts.append(f"llm_calls {calls.height}")
        timed = calls.filter(pl.col("mode").is_in(["sync", "local"]))
        if timed["latency_ms"].null_count():
            problems.append(f"{timed['latency_ms'].null_count()} sync/local calls without latency")
        batch = calls.filter(pl.col("mode") == "batch")
        if batch["batch_turnaround_s"].null_count():
            problems.append(
                f"{batch['batch_turnaround_s'].null_count()} batch calls without turnaround"
            )
        if calls["usd"].null_count():
            problems.append(f"{calls['usd'].null_count()} calls without usd")
    if not counts:
        return Outcome("ac-w-10", "fail", f"no scores under {scores_root.as_posix()}")
    if problems:
        return Outcome("ac-w-10", "fail", "; ".join(problems))
    return Outcome("ac-w-10", "pass", "complete for " + ", ".join(counts))


def run(ac_id: str, checks: dict[str, Callable[[], Outcome]]) -> list[Outcome]:
    """Run one criterion, or ``all``."""
    wanted = sorted({*checks, *PENDING}) if ac_id == "all" else [ac_id.lower()]
    outcomes: list[Outcome] = []
    for key in wanted:
        if key in checks:
            outcomes.append(checks[key]())
        elif key in PENDING:
            outcomes.append(Outcome(key, "pending", f"arrives in {PENDING[key]}"))
        else:
            outcomes.append(Outcome(key, "fail", "unknown acceptance criterion"))
    return outcomes
