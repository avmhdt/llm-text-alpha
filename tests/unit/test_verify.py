from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import polars as pl

from llm_text_alpha import verify
from llm_text_alpha.tracking.trial_log import ensure_log

REPO = Path(__file__).parents[2]


def repo_with(tmp_path: Path, files: dict[str, bytes]) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        subprocess.run(["git", "add", "-f", name], cwd=tmp_path, check=True)
    return tmp_path


def test_hygiene_passes_on_a_clean_tree(tmp_path: Path) -> None:
    outcome = verify.check_hygiene(repo_with(tmp_path, {"src/a.py": b"x = 1\n"}))
    assert outcome.status == "pass"


def test_hygiene_catches_data_env_and_large_files(tmp_path: Path) -> None:
    repo = repo_with(
        tmp_path,
        {"data/raw/x.csv": b"1", ".env": b"K=1", "big.bin": b"0" * (verify.MAX_TRACKED_BYTES + 1)},
    )
    outcome = verify.check_hygiene(repo)
    assert outcome.status == "fail"
    assert "data/raw/x.csv" in outcome.detail
    assert ".env" in outcome.detail
    assert "big.bin" in outcome.detail


def test_trial_log_check(tmp_path: Path) -> None:
    log = tmp_path / "results" / "trial_log.csv"
    assert verify.check_trial_log(log).status == "fail"
    ensure_log(log)
    outcome = verify.check_trial_log(log)
    assert outcome.status == "pass"
    assert "0 rows" in outcome.detail
    log.write_text("run_id\n", "utf-8")
    assert verify.check_trial_log(log).status == "fail"


def test_staged_file_guard(tmp_path: Path) -> None:
    script = REPO / "scripts" / "check_staged.py"
    small = tmp_path / "ok.py"
    small.write_text("x = 1\n", "utf-8")
    big = tmp_path / "big.bin"
    big.write_bytes(b"0" * 1_000_001)
    ok = subprocess.run([sys.executable, str(script), str(small)], capture_output=True)
    assert ok.returncode == 0
    bad = subprocess.run(
        [sys.executable, str(script), "data/raw/x.parquet", ".env", str(big)],
        capture_output=True,
        text=True,
    )
    assert bad.returncode == 1
    assert bad.stderr.count("\n") == 3


def test_gate_property_check_reports_pytest(tmp_path: Path) -> None:
    target = tmp_path / "tests" / "property" / "test_gate.py"
    target.parent.mkdir(parents=True)
    target.write_text("def test_ok():\n    assert True\n", "utf-8")
    outcome = verify.run_pytest("ac-w-01", tmp_path, ["tests/property/test_gate.py"])
    assert outcome.status == "pass"
    assert "1 passed" in outcome.detail
    target.write_text("def test_bad():\n    assert False\n", "utf-8")
    assert verify.check_gate_properties(tmp_path).status == "fail"
    assert verify.run_pytest("x", tmp_path, ["tests/property/test_gate.py"]).status == "fail"


def test_figure_check(tmp_path: Path) -> None:
    assert verify.check_figures(tmp_path, ["lm"]).status == "fail"
    (tmp_path / "event_study_lm_8k_item202.png").write_bytes(b"png")
    assert verify.check_figures(tmp_path, ["lm"]).status == "pass"


def test_cost_latency_check(tmp_path: Path) -> None:
    from llm_text_alpha.scoring.llm.run import CALLS_SCHEMA

    scores = tmp_path / "scores"
    lm = scores / "scorer=lm"
    lm.mkdir(parents=True)
    pl.DataFrame({"scorer": ["lm"], "cost_usd": [0.0], "latency_ms": [1.0]}).write_parquet(
        lm / "p.parquet"
    )
    calls = tmp_path / "calls"
    calls.mkdir()
    row = {name: None for name in CALLS_SCHEMA.names()}
    good = pl.DataFrame(
        [row | {"mode": "batch", "batch_turnaround_s": 60.0, "usd": 0.001}], schema=CALLS_SCHEMA
    )
    good.write_parquet(calls / "a.parquet")
    assert verify.check_cost_latency(scores, calls).status == "pass"
    bad = pl.DataFrame([row | {"mode": "sync", "usd": 0.001}], schema=CALLS_SCHEMA)
    bad.write_parquet(calls / "b.parquet")
    outcome = verify.check_cost_latency(scores, calls)
    assert outcome.status == "fail" and "without latency" in outcome.detail
    assert verify.check_cost_latency(tmp_path / "none", calls).status == "fail"
