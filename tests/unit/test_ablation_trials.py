from __future__ import annotations

import csv
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from llm_text_alpha.corpus.calendar import xnys
from llm_text_alpha.evaluation import ablation as abl
from llm_text_alpha.evaluation import trials as tr
from llm_text_alpha.pipeline import sue_orthogonal
from llm_text_alpha.tracking import trial_log

CAL = xnys()


def signals(n_days: int = 120, per_day: int = 3, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    days = CAL.sessions(date(2020, 1, 2), date(2021, 6, 30))[:n_days]
    rows = []
    for d, day in enumerate(days):
        for j in range(per_day):
            doc = f"d{d:04d}_{j}"
            base = rng.normal()
            for scorer in ("lm", "finbert", "llm"):
                if scorer == "llm" and j == 2:
                    continue  # the LLM scored a subsample
                rows.append(
                    {
                        "doc_id": doc,
                        "permno": j + 1,
                        "scorer": scorer,
                        "model_id": "m",
                        "masked": False,
                        "event_ts_utc": datetime(day.year, day.month, day.day, 12, tzinfo=UTC),
                        "entry_date": day,
                        "entry_ts_utc": datetime(day.year, day.month, day.day, 14, tzinfo=UTC),
                        "score_z": base + rng.normal(0, 0.5),
                        "size_log": 1.0,
                        "mom_12_2": 0.0,
                        "sue": base,
                        "bm": 0.5,
                        "rank": 100,
                    }
                )
    return pl.DataFrame(rows)


def test_combos_singles_first() -> None:
    assert abl.combos(["a", "b", "c"]) == [
        ("a",),
        ("b",),
        ("c",),
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
        ("a", "b", "c"),
    ]


def test_intersection_keeps_events_every_scorer_scored() -> None:
    sig = signals()
    events = abl.intersection(sig, ["lm", "finbert", "llm"])
    assert events.height == 120 * 2
    assert {"z_lm", "z_finbert", "z_llm"} <= set(events.columns)
    assert abl.intersection(sig, ["lm", "finbert"]).height == 120 * 3
    with pytest.raises(ValueError):
        abl.intersection(sig, [])


def test_combined_scores_are_trailing_quintiles() -> None:
    events = abl.intersection(signals(), ["lm", "finbert", "llm"])
    ranked = abl.combined(events, ("lm", "llm"), CAL)
    assert ranked["scorer"].unique().to_list() == ["lm+llm"]
    warm = ranked.filter(pl.col("quintile") == 0)
    assert warm.height >= 50  # the first 50 events have no trailing history
    assert set(ranked.filter(pl.col("quintile") > 0)["quintile"].unique().to_list()) <= {
        1,
        2,
        3,
        4,
        5,
    }
    first = ranked.sort("entry_date").head(1)
    assert first["score_z"].is_null().all()


def test_spread_and_ic() -> None:
    frame = pl.DataFrame(
        {
            "quintile": [5, 5, 5, 1, 1, 1, 3],
            "car": [0.03, 0.02, 0.04, -0.01, 0.0, -0.02, 0.0],
            "score_z": [2.0, 1.5, 2.5, -2.0, -1.0, -1.5, 0.0],
            "entry_date": [date(2020, 1, 2)] * 7,
        }
    )
    out = abl.spread(frame, "car")
    assert out["spread"] == pytest.approx(0.03 - (-0.01))
    assert out["t_spread"] > 0
    assert np.isnan(abl.spread(frame.head(1), "car")["spread"])
    ic = abl.information_coefficient(frame, "car")
    assert ic["ic"] > 0.8 and np.isnan(ic["ic_monthly"])


def test_sue_orthogonal_removes_sue_within_month() -> None:
    rng = np.random.default_rng(1)
    sue = rng.normal(size=300)
    panel = pl.DataFrame(
        {
            "entry_date": [date(2020, 1 + i // 100, 5) for i in range(300)],
            "sue": sue,
            "score_z": 0.8 * sue + rng.normal(size=300),
        }
    )
    out = sue_orthogonal(panel)
    for _, month in out.group_by(pl.col("entry_date").dt.month()):
        assert abs(float(np.corrcoef(month["score_orth"], month["sue"])[0, 1])) < 1e-9


def test_dsr_matches_overfitlab() -> None:
    deflated_mod = pytest.importorskip("overfitlab.deflated")
    rng = np.random.default_rng(5)
    returns = rng.standard_t(5, 1000) * 0.01 + 0.0006
    ours = tr.deflated_sharpe(returns, n_trials=40, var_trials=0.001)
    centred = returns - returns.mean()
    m2 = (centred**2).mean()
    theirs = deflated_mod.deflated_sharpe_ratio(
        ours.sr,
        1000,
        n_trials=40,
        var_trials=0.001,
        skew=float((centred**3).mean() / m2**1.5),
        kurtosis=float((centred**4).mean() / m2**2),
    )
    assert ours.dsr == pytest.approx(theirs.dsr, abs=1e-12)
    assert ours.sr0 == pytest.approx(theirs.sr0, abs=1e-12)
    assert tr.expected_max_sharpe(1, 0.5) == 0.0
    with pytest.raises(ValueError):
        tr.expected_max_sharpe(0.5, 0.5)
    with pytest.raises(ValueError):
        tr.deflated_sharpe(np.array([0.1, 0.2]), 2, 0.1)


def test_counts_and_sharpe_variance(tmp_path: Path) -> None:
    log = tmp_path / "results" / "trial_log.csv"
    trial_log.ensure_log(log)
    for i, sharpe in enumerate((0.5, 1.5)):
        run_id = trial_log.new_run_id()
        days = [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)]
        path = trial_log.write_returns(run_id, days, [0.01, -0.01, 0.002 * i], log)
        trial_log.log_trial(run_id, "portfolio", {"i": i}, days[0], days[-1], 3, sharpe, path, log)
    trial_log.log_trial(
        trial_log.new_run_id(),
        "ablation",
        {"x": 1},
        None,
        None,
        0,
        None,
        None,
        log,
        run_kind="analysis",
    )
    frame = tr.read_log(log)
    counts = tr.counts(frame)
    assert dict(zip(counts["strategy_family"], counts["trials"], strict=True)) == {
        "ablation": 1,
        "portfolio": 2,
    }
    n, var = tr.sharpe_variance(frame)
    assert n == 2
    assert var == pytest.approx(np.var([0.5 / np.sqrt(252), 1.5 / np.sqrt(252)], ddof=1))
    assert trial_log.logged_params(log, "ablation") == {'{"x":1}'}
    assert trial_log.logged_params(tmp_path / "none.csv", "ablation") == set()
    with log.open(encoding="utf-8") as handle:
        assert len(list(csv.reader(handle))) == 4
