from __future__ import annotations

import csv
import json
import math
import subprocess
from datetime import date
from pathlib import Path, PurePosixPath

import pytest
import yaml
from overfitlab.triallog import validate_trial_log

from llm_text_alpha.tracking import trial_log

DAYS = [date(2026, 1, 2), date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)]
RETS = [0.001, -0.002, 0.0015, 0.0005]


def rows(log: Path) -> list[dict[str, str]]:
    with log.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def log_series(log: Path, family: str = "ls_item202_lm") -> str:
    run_id = trial_log.new_run_id()
    path = trial_log.write_returns(run_id, DAYS, RETS, log)
    trial_log.log_trial(
        run_id,
        family,
        {"hold": 5, "scorer": "lm", "masked": False},
        DAYS[0],
        DAYS[-1],
        len(RETS),
        trial_log.annual_sharpe(RETS),
        path,
        log,
    )
    return run_id


def test_header_and_portfolio_row(tmp_path: Path) -> None:
    log = tmp_path / "results" / "trial_log.csv"
    run_id = log_series(log)
    text = log.read_bytes()
    assert not text.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in text
    assert text.decode().splitlines()[0] == ",".join(trial_log.COLUMNS)
    (row,) = rows(log)
    assert row["run_id"] == run_id
    assert row["params_json"] == '{"hold":5,"masked":false,"scorer":"lm"}'
    assert row["returns_path"] == f"runs/{run_id}/returns.parquet"
    assert row["n_obs"] == "4"
    assert row["x_run_kind"] == "portfolio"
    assert row["x_periods_per_year"] == "252"
    assert math.isclose(float(row["sharpe_annual"]), trial_log.annual_sharpe(RETS), rel_tol=1e-9)
    assert row["git_sha"] == "0" * 40 + "-dirty"
    assert (log.parent / row["returns_path"]).exists()


def test_analysis_and_failed_rows(tmp_path: Path) -> None:
    log = tmp_path / "trial_log.csv"
    trial_log.log_trial(
        trial_log.new_run_id(),
        "event_study",
        {"scorer": "lm"},
        date(2016, 1, 4),
        date(2026, 9, 30),
        0,
        None,
        None,
        log,
        run_kind="analysis",
    )
    trial_log.log_trial(
        trial_log.new_run_id(),
        "fm_regression",
        {},
        None,
        None,
        0,
        None,
        None,
        log,
        run_kind="analysis",
        note="failed: no events",
    )
    first, second = rows(log)
    assert first["sharpe_annual"] == "" and first["returns_path"] == ""
    assert first["x_periods_per_year"] == ""
    assert second["start"] == "" and second["x_note"] == "failed: no events"


def test_overfitlab_accepts_the_log(tmp_path: Path) -> None:
    log = tmp_path / "trial_log.csv"
    log_series(log)
    log_series(log, "ls_item202_finbert")
    trial_log.log_trial(
        trial_log.new_run_id(),
        "event_study",
        {},
        None,
        None,
        0,
        None,
        None,
        log,
        run_kind="analysis",
    )
    errors = [issue for issue in validate_trial_log(log) if issue.severity == "error"]
    assert not errors, errors


def test_committed_log_is_valid() -> None:
    log = Path(__file__).parents[2] / "results" / "trial_log.csv"
    trial_log.ensure_log(log)
    assert not [issue for issue in validate_trial_log(log) if issue.severity == "error"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"run_id": "not-a-uuid"}, "not a UUID"),
        ({"run_id": "6F9619FF-8B86-D011-B42D-00C04FC964FF"}, "canonical UUID v4"),
        ({"strategy_family": "Bad-Family"}, "strategy_family"),
        ({"n_obs": -1}, "n_obs"),
        ({"start": date(2026, 2, 1)}, "start is after end"),
        ({"sharpe_annual": None}, "finite sharpe"),
        ({"sharpe_annual": float("nan")}, "finite sharpe"),
        ({"returns_path": PurePosixPath("elsewhere.parquet")}, "returns_path must be"),
        ({"start": None}, "needs start and end"),
    ],
)
def test_contract_violations_write_nothing(
    tmp_path: Path, kwargs: dict[str, object], message: str
) -> None:
    log = tmp_path / "trial_log.csv"
    run_id = trial_log.new_run_id()
    args: dict[str, object] = {
        "run_id": run_id,
        "strategy_family": "ls_item202_lm",
        "params": {},
        "start": DAYS[0],
        "end": DAYS[-1],
        "n_obs": 4,
        "sharpe_annual": 1.0,
        "returns_path": trial_log.returns_relpath(run_id),
        "log_path": log,
    }
    args.update(kwargs)
    with pytest.raises(trial_log.TrialLogError, match=message):
        trial_log.log_trial(**args)  # type: ignore[arg-type]
    assert not log.exists()


def test_zero_obs_row_cannot_carry_a_series(tmp_path: Path) -> None:
    with pytest.raises(trial_log.TrialLogError, match="n_obs = 0"):
        trial_log.log_trial(
            trial_log.new_run_id(),
            "event_study",
            {},
            None,
            None,
            0,
            1.0,
            None,
            tmp_path / "log.csv",
        )


def test_non_canonical_params_are_refused() -> None:
    with pytest.raises(ValueError):
        trial_log.canonical_json({"x": float("nan")})


def test_header_mismatch_is_refused(tmp_path: Path) -> None:
    log = tmp_path / "trial_log.csv"
    log.write_text("run_id,timestamp\n", "utf-8")
    with pytest.raises(trial_log.TrialLogError, match="header"):
        trial_log.ensure_log(log)


def test_write_returns_checks_its_input(tmp_path: Path) -> None:
    log = tmp_path / "trial_log.csv"
    run_id = trial_log.new_run_id()
    with pytest.raises(trial_log.TrialLogError, match="same length"):
        trial_log.write_returns(run_id, DAYS, RETS[:2], log)
    with pytest.raises(trial_log.TrialLogError, match="strictly increasing"):
        trial_log.write_returns(run_id, [DAYS[1], DAYS[0]], RETS[:2], log)
    with pytest.raises(trial_log.TrialLogError, match="finite"):
        trial_log.write_returns(run_id, DAYS[:2], [0.1, float("inf")], log)


def test_annual_sharpe() -> None:
    assert trial_log.annual_sharpe([0.01, 0.01]) == 0.0
    with pytest.raises(trial_log.TrialLogError):
        trial_log.annual_sharpe([0.01])
    expected = (sum(RETS) / 4) / (math.sqrt(sum((r - sum(RETS) / 4) ** 2 for r in RETS) / 3))
    assert trial_log.annual_sharpe(RETS) == pytest.approx(expected * math.sqrt(252))


def test_record_run(tmp_path: Path) -> None:
    run_dir = trial_log.record_run(
        "abc", {"scorer": "lm"}, {"alpha": 0.1}, tmp_path / "results" / "trial_log.csv"
    )
    assert yaml.safe_load((run_dir / "config.yaml").read_text("utf-8")) == {"scorer": "lm"}
    assert json.loads((run_dir / "metrics.json").read_text("utf-8")) == {"alpha": 0.1}


def test_git_sha_in_a_repository(tmp_path: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / "code.py").write_text("x = 1\n", "utf-8")
    git("add", "code.py")
    git("commit", "-q", "-m", "init")
    log = tmp_path / "results" / "trial_log.csv"
    trial_log.ensure_log(log)
    clean = trial_log.git_sha(log)
    assert len(clean) == 40 and not clean.endswith("-dirty")
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "table.csv").write_text("a\n", "utf-8")
    git("add", "reports/table.csv")
    git("commit", "-q", "-m", "table")
    clean = trial_log.git_sha(log)
    (tmp_path / "reports" / "table.csv").write_text("b\n", "utf-8")
    assert trial_log.git_sha(log) == clean
    (tmp_path / "code.py").write_text("x = 2\n", "utf-8")
    assert trial_log.git_sha(log) == clean + "-dirty"
