from __future__ import annotations

import shutil
import subprocess
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from llm_text_alpha import cli
from llm_text_alpha.data.smoke import Check, SmokeReport
from llm_text_alpha.tracking.trial_log import ensure_log

REPO = Path(__file__).parents[2]
runner = CliRunner()


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A git repo with the committed configs and an empty trial log, data under tmp."""
    shutil.copytree(REPO / "configs", tmp_path / "configs")
    ensure_log(tmp_path / "results" / "trial_log.csv")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "configs", "results"], cwd=tmp_path, check=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SEC_USER_AGENT", "lta-tests tests@example.com")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    return tmp_path


def fake_smoke(passed: bool) -> object:
    def run(settings: object, config: object, run_id: str, **_: object) -> SmokeReport:
        report = SmokeReport(run_id=run_id)
        report.checks.append(Check("edgar_acceptance_matches_fixture", passed or None, "x"))
        report.checks.append(Check("prices_one_month", passed, "y"))
        return report

    return run


def test_help_lists_groups() -> None:
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    for group in ("data", "corpus", "score", "llm", "signals", "eval", "report", "verify"):
        assert group in result.output


@pytest.mark.parametrize(
    "args",
    [
        ["eval", "portfolio"],
        ["corpus", "mask"],
        ["eval", "capacity"],
        ["eval", "trials"],
        ["repro"],
    ],
)
def test_later_commands_say_when_they_arrive(args: list[str]) -> None:
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 2
    assert "arrives in M" in result.output


def test_data_smoke_passes(workdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "run_smoke", fake_smoke(True))
    result = runner.invoke(cli.app, ["data", "smoke", "--run-id", "r1"])
    assert result.exit_code == 0, result.output
    assert "PASS  prices_one_month: y" in result.output
    assert (workdir / "logs" / "data-smoke-r1.jsonl").exists()


def test_data_smoke_fails(workdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "run_smoke", fake_smoke(False))
    result = runner.invoke(cli.app, ["data", "smoke"])
    assert result.exit_code == 1
    assert "FAIL  prices_one_month" in result.output
    assert "INFO  edgar_acceptance_matches_fixture" in result.output


def test_data_commands_refuse_a_mnt_data_dir(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "check_data_placement", _raise_placement)
    result = runner.invoke(cli.app, ["data", "factors"])
    assert result.exit_code == 1
    assert "error:" in result.output


def _raise_placement(settings: object) -> Path:
    from llm_text_alpha.settings import DataPlacementError

    raise DataPlacementError("data directory /mnt/d/x is on a /mnt drive")


def test_data_factors(workdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import polars as pl

    from llm_text_alpha.data import factors

    frame = pl.DataFrame({"date": [date(2026, 9, 1)]}).select(
        pl.col("date"), *[pl.lit(0.0).alias(c) for c in factors.FACTOR_COLUMNS]
    )
    monkeypatch.setattr(factors, "download_factors", lambda raw: raw / "snapshot=x")
    monkeypatch.setattr(factors, "build_factors_daily", lambda snapshot: frame)
    result = runner.invoke(cli.app, ["data", "factors"])
    assert result.exit_code == 0, result.output
    assert "1 days, 2026-09-01 to 2026-09-01" in result.output
    assert (workdir / "data" / "processed" / "panels" / "factors_daily.parquet").exists()


def test_verify_repository_criteria(workdir: Path) -> None:
    result = runner.invoke(cli.app, ["verify", "ac-w-11"])
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output
    result = runner.invoke(cli.app, ["verify", "AC-W-12"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(cli.app, ["verify", "ac-w-03"])
    assert result.exit_code == 0
    assert "PENDING  ac-w-03: arrives in M5" in result.output
    result = runner.invoke(cli.app, ["verify", "ac-x-99"])
    assert result.exit_code == 1


def test_verify_all_runs_the_smoke(workdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "run_smoke", fake_smoke(True))
    monkeypatch.setattr(
        cli.verify_mod,
        "check_gate_properties",
        lambda repo: cli.verify_mod.Outcome("ac-w-01", "pass", "stub"),
    )
    monkeypatch.setattr(
        cli.verify_mod,
        "run_pytest",
        lambda ac_id, repo, targets, markers="": cli.verify_mod.Outcome(ac_id, "pass", "stub"),
    )
    monkeypatch.setattr(
        cli.verify_mod,
        "check_figures",
        lambda figures, scorers: cli.verify_mod.Outcome("ac-w-02", "pass", "stub"),
    )
    monkeypatch.setattr(
        cli.verify_mod,
        "check_cost_latency",
        lambda scores, calls: cli.verify_mod.Outcome("ac-w-10", "pass", "stub"),
    )
    result = runner.invoke(cli.app, ["verify", "all"])
    assert result.exit_code == 0, result.output
    assert "PASS     ac-w-04: 2 checks ok" in result.output
    monkeypatch.setattr(cli, "run_smoke", fake_smoke(False))
    result = runner.invoke(cli.app, ["verify", "ac-w-04"])
    assert result.exit_code == 1
    assert "failed: prices_one_month" in result.output


@pytest.mark.parametrize(
    "args",
    [["data", "edgar-fetch", "--doc-types", "10k_mdna"], ["corpus", "build", "--doc-types", "x"]],
)
def test_unsupported_doc_types(args: list[str]) -> None:
    assert runner.invoke(cli.app, args).exit_code == 2


@pytest.mark.parametrize(
    "args", [["data", "edgar-index"], ["data", "edgar-fetch"], ["corpus", "build"]]
)
def test_ingest_commands_need_their_inputs(workdir: Path, args: list[str]) -> None:
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 1
    assert "error:" in result.output


def test_corpus_build_end_to_end(
    workdir: Path, monkeypatch: pytest.MonkeyPatch, efp_root: Path
) -> None:
    from tests.unit.test_corpus_build import LONG, write_filing

    data = workdir / "data"
    raw = data / "raw" / "edgar" / "filings"
    for n, cik in enumerate((1000010, 1000021), start=1):
        write_filing(
            raw,
            cik,
            f"0000000001-16-00000{n}",
            f"2016-03-0{n}T21:00:00+00:00",
            {"e.htm": ("EX-99.1", f"<p>{LONG} {n}</p>")},
        )
    index = pl.DataFrame(
        {
            "cik": [1000010, 1000021],
            "company_name": ["a", "b"],
            "form": ["8-K", "8-K"],
            "filed_date": [date(2016, 3, 1), date(2016, 3, 2)],
            "accession": ["0000000001-16-000001", "0000000001-16-000002"],
            "items": [["2.02"], ["2.02"]],
            "submissions_acceptance_utc": [
                datetime(2016, 3, 1, 21, tzinfo=UTC),
                datetime(2016, 3, 2, 22, tzinfo=UTC),
            ],
        },
        schema_overrides={"submissions_acceptance_utc": pl.Datetime("ns", "UTC")},
    )
    (data / "interim" / "edgar").mkdir(parents=True)
    index.write_parquet(data / "interim" / "edgar" / "filings_index.parquet")
    monkeypatch.setenv("EFP_DATA_DIR", str(efp_root))
    result = runner.invoke(
        cli.app, ["corpus", "build", "--start", "2016-01-01", "--end", "2016-12-31"]
    )
    assert result.exit_code == 0, result.output
    assert "2 documents; extraction success 100.00%" in result.output
    by_year = (workdir / "reports" / "tables" / "dq02_acceptance_by_year.csv").read_text("utf-8")
    assert by_year.startswith("# command=lta corpus build")
    assert "2016,EST,2,1,1,0" in by_year
    docs = pl.read_parquet(data / "interim" / "corpus" / "documents" / "**" / "*.parquet")
    assert docs.height == 2
    enriched = pl.read_parquet(data / "interim" / "edgar" / "filings_index.parquet")
    assert enriched["acceptance_raw"].to_list() == ["x", "x"]
