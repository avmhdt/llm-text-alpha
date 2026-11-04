"""Experiment log in backtest-overfitting-lab's trial-log contract v1.0 (FR-23, AC-W-11).

Follows the contract's reference writer: UTF-8 without BOM, RFC 4180 quoting, LF endings,
append-only. This project adds ``x_run_kind`` (``portfolio`` or ``analysis``); event studies
and Fama-MacBeth runs have no return series and are logged with ``n_obs = 0``.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from llm_text_alpha.io import write_json_atomic, write_text_atomic

REQUIRED_COLUMNS = (
    "run_id",
    "timestamp",
    "git_sha",
    "strategy_family",
    "params_json",
    "start",
    "end",
    "n_obs",
    "sharpe_annual",
    "returns_path",
)
COLUMNS = (*REQUIRED_COLUMNS, "x_run_kind", "x_periods_per_year", "x_note")
DEFAULT_LOG = Path("results/trial_log.csv")
PERIODS_PER_YEAR = 252
RunKind = Literal["portfolio", "analysis"]

_FAMILY = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class TrialLogError(ValueError):
    """A row would break the trial-log contract."""


def canonical_json(params: Mapping[str, Any]) -> str:
    """The contract's canonical JSON: sorted keys, no spaces, ASCII, no NaN."""
    return json.dumps(
        dict(params), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def new_run_id() -> str:
    """A fresh lower-case UUID v4."""
    return str(uuid.uuid4())


def git_sha(log_path: Path) -> str:
    """HEAD commit, with ``-dirty`` if tracked code or config has uncommitted changes.

    The log itself and the generated outputs under ``results/`` and ``reports/`` do not
    count: an evaluation rewrites its own tables, which would otherwise mark every later
    run in the same session dirty.
    """

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, cwd=log_path.parent
        ).stdout

    try:
        sha = git("rev-parse", "HEAD").strip()
        root = Path(git("rev-parse", "--show-toplevel").strip())
        log = log_path.resolve().relative_to(root.resolve()).as_posix()
        status = git(
            "-C",
            str(root),
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--",
            ".",
            f":(exclude){log}",
            ":(exclude)results",
            ":(exclude)reports",
        )
    except (OSError, ValueError, subprocess.CalledProcessError):
        return "0" * 40 + "-dirty"
    return f"{sha}-dirty" if status.strip() else sha


def ensure_log(log_path: Path = DEFAULT_LOG) -> None:
    """Create the log with its header row, or check the header of an existing one."""
    if log_path.exists():
        with log_path.open(encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle), [])
        if tuple(header) != COLUMNS:
            raise TrialLogError(f"{log_path} header {header} differs from {list(COLUMNS)}")
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerow(COLUMNS)


def annual_sharpe(returns: Sequence[float], periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """``mean(ret) / std(ret, ddof=1) * sqrt(q)``; 0.0 for a flat series."""
    if len(returns) < 2:
        raise TrialLogError("a Sharpe ratio needs at least two returns")
    std = statistics.stdev(returns)
    return statistics.fmean(returns) / std * math.sqrt(periods_per_year) if std > 0 else 0.0


def returns_relpath(run_id: str) -> PurePosixPath:
    """``runs/<run_id>/returns.parquet``, relative to the log's directory."""
    return PurePosixPath("runs", run_id, "returns.parquet")


def write_returns(
    run_id: str,
    dates: Sequence[date],
    returns: Sequence[float],
    log_path: Path = DEFAULT_LOG,
) -> PurePosixPath:
    """Write a daily net return series for ``run_id``; dates must strictly increase."""
    if len(dates) != len(returns) or not dates:
        raise TrialLogError("dates and returns must be non-empty and the same length")
    if any(b <= a for a, b in pairwise(dates)):
        raise TrialLogError("dates must be strictly increasing")
    if not all(math.isfinite(r) for r in returns):
        raise TrialLogError("returns must be finite")
    relative = returns_relpath(run_id)
    target = log_path.parent / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"date": pa.array(dates, pa.date32()), "ret": pa.array(returns, pa.float64())})
    pq.write_table(table, target)
    return relative


def log_trial(
    run_id: str,
    strategy_family: str,
    params: Mapping[str, Any],
    start: date | None,
    end: date | None,
    n_obs: int,
    sharpe_annual: float | None,
    returns_path: Path | PurePosixPath | None,
    log_path: Path = DEFAULT_LOG,
    run_kind: RunKind = "portfolio",
    note: str = "",
) -> None:
    """Append one row. Failed runs and runs without a series pass ``n_obs=0``.

    Raises:
        TrialLogError: the row would break the contract; nothing is written.
    """
    try:
        parsed = uuid.UUID(run_id)
    except ValueError as exc:
        raise TrialLogError(f"run_id {run_id!r} is not a UUID") from exc
    if parsed.version != 4 or str(parsed) != run_id:
        raise TrialLogError(f"run_id {run_id!r} is not a lower-case canonical UUID v4")
    if not _FAMILY.match(strategy_family):
        raise TrialLogError(f"strategy_family {strategy_family!r} is not [a-z][a-z0-9_]*")
    if n_obs < 0:
        raise TrialLogError("n_obs must be >= 0")
    if start is not None and end is not None and start > end:
        raise TrialLogError("start is after end")
    if n_obs == 0:
        if sharpe_annual is not None or returns_path is not None:
            raise TrialLogError("a row with n_obs = 0 has no Sharpe ratio and no returns_path")
    else:
        if start is None or end is None:
            raise TrialLogError("a row with a series needs start and end")
        if sharpe_annual is None or not math.isfinite(sharpe_annual):
            raise TrialLogError("a row with a series needs a finite sharpe_annual")
        if returns_path is None or PurePosixPath(returns_path) != returns_relpath(run_id):
            raise TrialLogError(f"returns_path must be {returns_relpath(run_id)}")
    params_json = canonical_json(params)
    ensure_log(log_path)
    row = {
        "run_id": run_id,
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": git_sha(log_path),
        "strategy_family": strategy_family,
        "params_json": params_json,
        "start": start.isoformat() if start else "",
        "end": end.isoformat() if end else "",
        "n_obs": str(n_obs),
        "sharpe_annual": "" if sharpe_annual is None else f"{sharpe_annual:.10g}",
        "returns_path": "" if returns_path is None else PurePosixPath(returns_path).as_posix(),
        "x_run_kind": run_kind,
        "x_periods_per_year": str(PERIODS_PER_YEAR) if n_obs else "",
        "x_note": note,
    }
    with log_path.open("a", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n").writerow(row)


def record_run(
    run_id: str,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    log_path: Path = DEFAULT_LOG,
) -> Path:
    """Write ``runs/<run_id>/config.yaml`` and ``metrics.json`` next to the log."""
    run_dir = log_path.parent / "runs" / run_id
    write_text_atomic(yaml.safe_dump(dict(config), sort_keys=True), run_dir / "config.yaml")
    write_json_atomic(dict(metrics), run_dir / "metrics.json")
    return run_dir


def logged_params(log_path: Path, family: str) -> set[str]:
    """Canonical ``params_json`` of every row of ``family`` already in the log."""
    if not log_path.exists():
        return set()
    with log_path.open(encoding="utf-8", newline="") as handle:
        return {
            row["params_json"] for row in csv.DictReader(handle) if row["strategy_family"] == family
        }
