from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

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


def _write(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# provenance\n" + frame.write_csv(), "utf-8")


def test_ablation_check(tmp_path: Path) -> None:
    tables = tmp_path / "tables"
    assert verify.check_ablation(tables, ["lm", "llm"]).status == "fail"
    rows = {"combination": ["lm", "llm", "lm+llm"], "window": ["all"] * 3, "events": [10] * 3}
    _write(tables / "ablation.csv", pl.DataFrame(rows))
    assert verify.check_ablation(tables, ["lm", "llm"]).status == "pass"
    _write(tables / "ablation.csv", pl.DataFrame(rows | {"events": [10, 10, 9]}))
    assert "different" in verify.check_ablation(tables, ["lm", "llm"]).detail
    _write(tables / "ablation.csv", pl.DataFrame({k: v[:2] for k, v in rows.items()}))
    assert "missing" in verify.check_ablation(tables, ["lm", "llm"]).detail


def test_headline_check(tmp_path: Path) -> None:
    tables = tmp_path / "tables"
    head = {"scorer": "llm", "window": "post", "hold_days": 5, "cost_mult": 1.0}
    assert verify.check_headline(tables, head).status == "fail"
    row = {
        "scorer": ["llm"],
        "window": ["post"],
        "hold_days": [5],
        "cost_mult": [1.0],
        "alpha_annual_net": [0.2],
        "t_alpha_net": [3.5],
        "days": [60],
    }
    _write(tables / "portfolio_alpha.csv", pl.DataFrame(row))
    assert verify.check_headline(tables, head).status == "pass"
    _write(tables / "portfolio_alpha.csv", pl.DataFrame(row | {"t_alpha_net": [1.0]}))
    outcome = verify.check_headline(tables, head)
    assert outcome.status == "fail" and "not met" in outcome.detail
    assert verify.check_headline(tables, head | {"window": "x"}).detail.startswith("no row")


def test_writeup_check(tmp_path: Path) -> None:
    assert verify.check_writeup(tmp_path).status == "fail"
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("x", "utf-8")
    section = (
        "## Look-ahead bias and memorization\n"
        "Glasserman and Lin; Lopez-Lira and Tang; Sarkar and Vafa; He, Lv, Manela and Wu; "
        "Golchin and Surdeanu. Numbers: 75%, 54%, 28%. See [a](docs/a.md).\n## Next\n"
    )
    (tmp_path / "README.md").write_text("# Memo\n" + section, "utf-8")
    assert verify.check_writeup(tmp_path).status == "pass"
    (tmp_path / "README.md").write_text(
        "# Memo\n" + section.replace("Golchin", "X") + "[b](reports/b.md)", "utf-8"
    )
    outcome = verify.check_writeup(tmp_path)
    assert outcome.status == "fail" and "only 4" in outcome.detail and "broken" in outcome.detail
    (tmp_path / "README.md").write_text("# Memo\n## Other\n", "utf-8")
    assert "no '" in verify.check_writeup(tmp_path).detail


def test_repro_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess as sp

    outputs = iter([sp.CompletedProcess([], 0, "", ""), sp.CompletedProcess([], 0, "", "")])
    monkeypatch.setattr(verify.subprocess, "run", lambda *a, **k: next(outputs))
    assert verify.check_repro(tmp_path).status == "pass"
    outputs = iter(
        [sp.CompletedProcess([], 0, "", ""), sp.CompletedProcess([], 0, " 1 file changed", "")]
    )
    assert verify.check_repro(tmp_path).status == "fail"
    outputs = iter([sp.CompletedProcess([], 1, "", "boom")])
    assert "boom" in verify.check_repro(tmp_path).detail


def test_evidence_check(tmp_path: Path) -> None:
    assert verify.check_evidence("ac-h-06", tmp_path).status == "fail"
    for rel in verify.EVIDENCE["ac-h-02"]:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# p\nmodel_id\na\nb\n", "utf-8")
    outcome = verify.check_evidence("ac-h-02", tmp_path)
    assert outcome.status == "fail" and "need 3" in outcome.detail
    (tmp_path / "reports/tables/contamination.csv").write_text("# p\nmodel_id\na\nb\nc\n", "utf-8")
    assert verify.check_evidence("ac-h-02", tmp_path).status == "pass"
