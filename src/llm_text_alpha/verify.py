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

from llm_text_alpha.tracking import trial_log

MAX_TRACKED_BYTES = 1_000_000


@dataclass(frozen=True)
class Outcome:
    """Result of one criterion: ``status`` is pass, fail or pending."""

    ac_id: str
    status: str
    detail: str


PENDING = {
    "ac-w-02": "M3",
    "ac-w-03": "M5",
    "ac-w-05": "M4",
    "ac-w-06": "M4",
    "ac-w-07": "M3",
    "ac-w-08": "M3",
    "ac-w-09": "M6",
    "ac-w-10": "M4",
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


def check_gate_properties(repo: Path) -> Outcome:
    """AC-W-01, property part: the hypothesis suite in ``tests/property/test_gate.py``.

    The full-data part (every real signal) joins in M3, once signals exist.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "tests/property/test_gate.py",
        ],
        capture_output=True,
        text=True,
        cwd=repo,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    detail = lines[-1] if lines else result.stderr.strip()[-200:]
    status = "pass" if result.returncode == 0 else "fail"
    return Outcome("ac-w-01", status, f"property part: {detail}; full-data part arrives in M3")


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
