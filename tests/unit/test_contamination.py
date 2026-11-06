from __future__ import annotations

from datetime import UTC, date, datetime

import numpy as np
import polars as pl
import pytest

from llm_text_alpha.evaluation import contamination as ct
from llm_text_alpha.evaluation import probes
from llm_text_alpha.scoring.llm.prompts import Prompt

SUFFIXES = ("inc", "corp", "co", "company", "the", "holdings")
POST = date(2025, 8, 1)


def test_company_match() -> None:
    names = ["APPLE INC", "APPLE COMPUTER INC"]
    assert probes.company_match("Apple Inc.", None, names, "AAPL", SUFFIXES)
    assert probes.company_match("apple", None, names, "AAPL", SUFFIXES)
    assert probes.company_match("Apple Computer", None, names, "AAPL", SUFFIXES)
    assert probes.company_match("something", "aapl", names, "AAPL", SUFFIXES)
    assert not probes.company_match("Microsoft", "MSFT", names, "AAPL", SUFFIXES)
    assert not probes.company_match("unknown", None, names, "AAPL", SUFFIXES)
    assert not probes.company_match(None, None, names, None, SUFFIXES)
    assert not probes.company_match("App", None, ["APPLIED MATERIALS INC"], None, SUFFIXES)


def test_true_quarter_handles_52_week_ends() -> None:
    assert probes.true_quarter(date(2016, 1, 3)) == (2015, 4)
    assert probes.true_quarter(date(2024, 6, 29)) == (2024, 2)
    assert probes.true_quarter(None) is None


def test_wilson_and_accuracy() -> None:
    lo, hi = probes.wilson(50, 100)
    assert lo < 0.5 < hi and hi - lo == pytest.approx(0.1923, abs=0.001)
    assert np.isnan(probes.wilson(0, 0)[0])
    frame = pl.DataFrame(
        {
            "event_ts_utc": [datetime(2024, 1, 1, tzinfo=UTC)] * 4
            + [datetime(2026, 1, 1, tzinfo=UTC)] * 2,
            "ok": [True, True, False, None, False, False],
        }
    )
    acc = probes.accuracy(frame, "ok", POST)
    rows = {r["window"]: r for r in acc.iter_rows(named=True)}
    assert rows["pre"]["n"] == 3 and rows["pre"]["accuracy"] == pytest.approx(2 / 3)
    assert rows["post"]["accuracy"] == 0.0 and rows["all"]["n"] == 5


def test_probe_requests_carry_their_answer_type() -> None:
    prompt = Prompt("p", "system")
    masked = pl.DataFrame({"doc_id": ["a"], "text_masked": ["[COMPANY] grew."]})
    reid = probes.reid_requests(masked, "m", prompt, {})
    assert reid[0].answer == "reid" and reid[0].user_text == "[COMPANY] grew."
    events = pl.DataFrame(
        {
            "doc_id": ["a"],
            "company_name": ["ACME INC"],
            "ticker": ["ACME"],
            "event_ts_utc": [datetime(2024, 5, 2, 21, tzinfo=UTC)],
        }
    )
    recall = probes.recall_requests(events, "m", prompt, {})
    assert recall[0].answer == "recall" and "2024-05-02" in recall[0].user_text
    assert "ACME" in recall[0].user_text
    assert reid[0].cache_key() != recall[0].cache_key()


def signals(
    n: int = 1600, effect_pre: float = 0.02, seed: int = 0
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """A scorer that predicts CAR before the cutoff (raw only) and not after; baselines flat."""
    rng = np.random.default_rng(seed)
    days = [date(2024, 1, 2) + (date(2026, 6, 1) - date(2024, 1, 2)) * (i / n) for i in range(n)]
    rows, cars = [], []
    for i, d in enumerate(days):
        doc = f"d{i:05d}"
        post = d >= POST
        q = int(rng.integers(1, 6))
        car = rng.normal(0, 0.01) + (0 if post else effect_pre * (q - 3) / 2)
        cars.append({"doc_id": doc, "car_5": car})
        ts = datetime(d.year, d.month, d.day, 21, tzinfo=UTC)
        for scorer, masked, quint in (
            ("llm", False, q),
            ("llm", True, int(rng.integers(1, 6))),
            ("lm", False, int(rng.integers(1, 6))),
            ("finbert", False, int(rng.integers(1, 6))),
        ):
            rows.append(
                {
                    "doc_id": doc,
                    "scorer": scorer,
                    "masked": masked,
                    "event_ts_utc": ts,
                    "quintile": quint,
                    "score_z": float(quint) + rng.normal(0, 0.1),
                }
            )
    return pl.DataFrame(rows), pl.DataFrame(cars)


def test_two_by_two_and_disappearing_share() -> None:
    sig, cars = signals()
    cells = ct.two_by_two(sig, cars, "llm", POST)
    assert cells.height == 4
    get = {(r["text"], r["window"]): r for r in cells.iter_rows(named=True)}
    assert get[("raw", "pre")]["spread"] > 0.015 and get[("raw", "pre")]["spread_lo95"] > 0
    assert abs(get[("raw", "post")]["spread"]) < 0.005
    assert abs(get[("masked", "pre")]["spread"]) < 0.005
    assert get[("raw", "pre")]["ic"] > 0.3 and get[("raw", "pre")]["ic_lo95"] > 0
    assert get[("raw", "post")]["underpowered"] == (get[("raw", "post")]["n"] < 500)
    shares = ct.disappearing(cells)
    assert shares["share_total"] == pytest.approx(1.0, abs=0.3)
    assert shares["share_cutoff"] == pytest.approx(1.0, abs=0.3)


def test_did_finds_planted_contamination() -> None:
    sig, cars = signals()
    out = ct.did(sig, cars, "llm", "lm", POST, reps=200)
    assert out["did"] > 0.015 and out["did_lo95"] > 0 and out["contamination_found"]
    clean_sig, clean_cars = signals(effect_pre=0.0, seed=2)
    clean = ct.did(clean_sig, clean_cars, "llm", "finbert", POST, reps=200)
    assert clean["did_lo95"] < 0 < clean["did_hi95"]
    summary = ct.summary_rows("m", "llm", POST, ct.two_by_two(sig, cars, "llm", POST), [out])
    assert summary["did_vs_lm"] == out["did"] and summary["post_cutoff_start"] == "2025-08-01"


def test_disappearing_without_a_base() -> None:
    cells = pl.DataFrame({"text": ["raw"], "window": ["pre"], "spread": [0.0]})
    assert ct.disappearing(cells)["share_total"] is None


def test_ic_did_survives_ties() -> None:
    sig, cars = signals()
    tied = sig.with_columns(
        pl.when(pl.col("scorer") == "llm")
        .then(pl.col("quintile").clip(1, 3))
        .otherwise(pl.col("quintile"))
        .alias("quintile")
    )
    out = ct.did(tied, cars, "llm", "lm", POST, reps=100)
    assert np.isnan(out["did"]) and np.isfinite(out["did_ic"])
    assert out["did_ic"] > 0 and out["contamination_found_ic"]
    assert ct._avg_rank(np.array([3.0, 1.0, 1.0, 2.0])).tolist() == [4.0, 1.5, 1.5, 3.0]
