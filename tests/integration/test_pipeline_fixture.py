"""The M3 pipeline end to end on synthetic data: corpus, panels, scorers, gate, evaluation.

No network: eqpit's synthetic store, synthetic filings, a mini LM dictionary and a tiny
random BERT stand in for the real inputs.
"""

from __future__ import annotations

import csv
import json
import shutil
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from llm_text_alpha import cli
from llm_text_alpha.corpus.calendar import xnys
from llm_text_alpha.data import prices
from llm_text_alpha.data.factors import FACTOR_COLUMNS
from llm_text_alpha.tracking.trial_log import ensure_log
from tests.unit.test_corpus_build import write_filing
from tests.unit.test_scorers import MINI, VOCAB

REPO = Path(__file__).parents[2]
runner = CliRunner()
WORDS = ["gain strong", "loss decline", "may required", "lawsuit loss gain"]


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, efp_root: Path) -> Path:
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    shutil.copytree(REPO / "configs", tmp_path / "configs")
    ensure_log(tmp_path / "results" / "trial_log.csv")
    data = tmp_path / "data"
    config = (tmp_path / "configs" / "default.yaml").read_text("utf-8")
    config = (
        config.replace("start: 2016-01-01", "start: 2016-03-01")
        .replace("end: 2026-09-30", "end: 2016-11-30")
        .replace("estimation_data_start: 2015-01-01", "estimation_data_start: 2015-01-02")
    )
    config = config.replace("top_n_by_mcap: 1000", "top_n_by_mcap: 25")
    config = config.replace("bootstrap_reps: 1000", "bootstrap_reps: 50")
    config = (
        config.replace("revision: 0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031", "revision: tiny")
        .replace("max_tokens: 512", "max_tokens: 32")
        .replace("stride: 64", "stride: 4")
    )
    (tmp_path / "configs" / "default.yaml").write_text(config, "utf-8")
    lm = data / "raw" / "lm" / "LM_MasterDictionary.csv"
    lm.parent.mkdir(parents=True)
    lm.write_text(MINI, "utf-8")
    model = data / "models" / "ProsusAI--finbert" / "tiny"
    model.mkdir(parents=True)
    (model / "vocab.txt").write_text("\n".join(VOCAB) + "\n", "utf-8")
    BertTokenizerFast(vocab_file=str(model / "vocab.txt")).save_pretrained(model)
    BertForSequenceClassification(
        BertConfig(
            vocab_size=len(VOCAB),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=64,
            num_labels=3,
            id2label={0: "positive", 1: "negative", 2: "neutral"},
            label2id={"positive": 0, "negative": 1, "neutral": 2},
        )
    ).save_pretrained(model)
    write_synthetic_filings(data / "raw" / "edgar" / "filings", efp_root)
    sessions = xnys().sessions(date(2015, 1, 2), date(2016, 12, 30))
    rng = np.random.default_rng(0)
    factors = pl.DataFrame(
        {"date": sessions, **{c: rng.normal(0, 0.01, len(sessions)) for c in FACTOR_COLUMNS}}
    ).with_columns(pl.col("rf").abs() / 100)
    (data / "processed" / "panels").mkdir(parents=True)
    factors.write_parquet(data / "processed" / "panels" / "factors_daily.parquet")
    monkeypatch.setattr(prices.EfpPrices, "accepted_sources", ("synthetic",))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SEC_USER_AGENT", "lta-tests tests@example.com")
    monkeypatch.setenv("DATA_DIR", str(data))
    monkeypatch.setenv("EFP_DATA_DIR", str(efp_root))
    return tmp_path


def write_synthetic_filings(raw: Path, efp_root: Path) -> None:
    """Four press releases per company per year for the synthetic universe."""
    from eqpit.store import PitStore

    store = PitStore.open(efp_root / "processed" / "pit")
    ciks = (
        store.universe(date(2016, 2, 29), 25)["company_id"].str.strip_prefix("CIK:").cast(pl.Int64)
    )
    rng = np.random.default_rng(1)
    n = 0
    for cik in ciks.to_list():
        for month in (3, 5, 8, 11):
            n += 1
            day = date(2016, month, 2) + timedelta(days=int(rng.integers(0, 20)))
            accepted = datetime(day.year, day.month, day.day, 21, 5, tzinfo=UTC)
            body = " ".join(rng.choice(WORDS, size=40)) + f" filing {n}"
            write_filing(
                raw,
                cik,
                f"0000000001-16-{n:06d}",
                accepted.isoformat(),
                {"d.htm": ("8-K", "<p>cover</p>"), "e.htm": ("EX-99.1", f"<p>{body}</p>")},
            )


def invoke(*args: str) -> str:
    result = runner.invoke(cli.app, list(args))
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        import traceback

        traceback.print_exception(result.exception)
    assert result.exit_code == 0, result.output
    return result.output


def test_pipeline_end_to_end(workdir: Path) -> None:
    index = pl.DataFrame(
        schema={
            "accession": pl.String,
            "cik": pl.Int64,
            "form": pl.String,
            "filed_date": pl.Date,
            "items": pl.List(pl.String),
            "submissions_acceptance_utc": pl.Datetime("ns", "UTC"),
        }
    )
    (workdir / "data" / "interim" / "edgar").mkdir(parents=True)
    index.write_parquet(workdir / "data" / "interim" / "edgar" / "filings_index.parquet")
    assert "100.00%" in invoke("corpus", "build")
    assert "securities" in invoke("data", "prices")
    assert "company-quarters" in invoke("data", "earnings")
    assert "lm-LM_MasterDictionary@" in invoke("data", "lm-dict")
    invoke("score", "dict")
    invoke("score", "finbert", "--threads", "1")
    llm = ("--provider", "fake", "--model", "fake-v1", "--sample", "item202_main")
    assert "worst_case_usd: 0.0" in invoke("llm", "estimate", *llm)
    first = invoke("score", "llm", *llm, "--run-id", "llm-first")
    assert "0 cached, $0.0000" in first
    again = invoke("score", "llm", *llm, "--run-id", "llm-again")
    n_docs = int(first.strip().splitlines()[-1].split(" ")[0])
    assert f"{n_docs} cached" in again
    llm_dir = workdir / "data" / "processed" / "scores" / "scorer=llm"
    llm_scores = pl.read_parquet(
        llm_dir / "model=fake-v1" / "masked=false" / "sample=item202_main" / "*.parquet"
    )
    assert llm_scores.height == n_docs and llm_scores["cost_usd"].is_not_null().all()
    calls = pl.read_parquet(workdir / "data" / "llm_cache" / "llm_calls" / "*.parquet")
    assert calls.filter(pl.col("run_id") == "llm-again")["mode"].unique().to_list() == ["cache_hit"]
    metrics = json.loads((workdir / "results" / "runs" / "llm-first" / "metrics.json").read_text())
    assert metrics["status"] == "done" and metrics["estimate"]["worst_case_usd"] == 0.0
    assert "open holds included" in invoke("llm", "spend")
    config = workdir / "configs" / "default.yaml"
    text = config.read_text("utf-8")
    text = (
        text.replace(
            "llm: {model: claude-haiku-5-5, samples: [item202_main, item202_post_h55], "
            "masked: [false, true]}",
            "llm: {model: fake-v1, samples: [item202_main], masked: [false, true]}",
        )
        .replace("min_events: 50}", "min_events: 5}")
        .replace(
            "models: {claude-haiku-5-5: llm, claude-haiku-4-5: llm_h45, "
            "llama-3.1-8b-instruct-q4_k_m: llm_llama}",
            "models: {fake-v1: llm}",
        )
        .replace(
            "    sample: item202_contam\n    reidentify_n: 300\n    outcome_recall_n: 300",
            "    sample: item202_main\n    reidentify_n: 20\n    outcome_recall_n: 20",
        )
        .replace("bootstrap_reps: 500", "bootstrap_reps: 20")
    )
    config.write_text(text, "utf-8")
    models = workdir / "configs" / "models.yaml"
    models.write_text(
        models.read_text("utf-8").replace(
            "provider: fake\n  training_data_cutoff: null",
            "provider: fake\n  training_data_cutoff: 2016-06",
        ),
        "utf-8",
    )
    assert "documents masked" in invoke(
        "corpus", "mask", "--samples", "item202_main", "--processes", "1"
    )
    assert "residual names" in invoke("corpus", "audit-mask", "--review", "5")
    masked_run = invoke("score", "llm", *llm, "--masked", "--run-id", "llm-masked")
    assert "requests" in masked_run
    out = invoke("signals", "build", "--scorers", "lm,finbert,sue,llm")
    assert "finbert" in out and "lm" in out and "llm" in out
    assert "kept" in invoke("eval", "abnormal")
    invoke("eval", "event-study", "--scorers", "lm,finbert")
    invoke("eval", "regress", "--scorers", "lm", "--horizons", "1,5")
    trial_rows_before_m5 = 2 + 2 * 6 * 2
    invoke("eval", "regress", "--scorers", "lm", "--horizons", "1,5")  # same specs: not new trials
    assert "portfolios written" in invoke("eval", "portfolio", "--scorers", "lm,llm,sue")
    assert "rows written" in invoke("eval", "ablation")
    assert "trials logged" in invoke("eval", "trials")
    assert "recall probes" in invoke("score", "probes", "--provider", "fake", "--model", "fake-v1")
    assert "models written" in invoke("eval", "contamination", "--models", "fake-v1")
    assert "releases accepted" in invoke("eval", "latency", "--fetch-sample", "0")
    assert (workdir / "reports" / "tables" / "contamination_cells.md").exists()
    assert (workdir / "reports" / "tables" / "mask_audit.md").exists()
    signals = pl.read_parquet(workdir / "data" / "processed" / "signals" / "signals.parquet")
    assert (signals["entry_ts_utc"] > signals["event_ts_utc"]).all()
    assert signals.filter(pl.col("quintile") > 0).height > 0
    tables = workdir / "reports" / "tables"
    assert (tables / "event_study_lm_8k_item202.md").exists()
    assert (workdir / "reports" / "figures" / "event_study_finbert_8k_item202.png").exists()
    assert (tables / "regressions.csv").read_text("utf-8").startswith("# command=lta eval regress")
    with (workdir / "results" / "trial_log.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    analysis = rows[:trial_rows_before_m5]
    assert {row["strategy_family"] for row in analysis} == {
        "event_study",
        "cs_regression",
        "fm_regression",
    }
    assert all(row["x_run_kind"] == "analysis" and row["n_obs"] == "0" for row in analysis)
    later = {row["strategy_family"] for row in rows[trial_rows_before_m5:]}
    assert later <= {"portfolio", "ablation", "contamination", "latency"}
    contamination = [r for r in rows if r["strategy_family"] == "contamination"]
    assert contamination and "preregistration_sha" in contamination[0]["params_json"]
    for row in rows:
        if row["strategy_family"] == "portfolio":
            assert int(row["n_obs"]) > 0
            assert (workdir / "results" / row["returns_path"]).exists()
    assert (tables / "trials.md").exists()
