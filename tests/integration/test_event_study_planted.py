"""AC-W-07: the event study recovers a planted CAR, and its 95% band covers the truth."""

from __future__ import annotations

import numpy as np
import polars as pl

from llm_text_alpha.corpus.calendar import xnys
from llm_text_alpha.evaluation.abnormal import abnormal_returns
from llm_text_alpha.evaluation.event_study import SPREAD, Z95, event_study, kp_correlation
from tests.synthetic import make_market, plant, random_events

CAL = xnys()
PLANTED = 0.01


def quintiles(events: pl.DataFrame) -> pl.DataFrame:
    return events.with_columns(
        (pl.col("score").rank("ordinal") * 5 / (pl.len() + 1))
        .floor()
        .cast(pl.Int8)
        .add(1)
        .alias("quintile")
    )


def test_planted_car_is_recovered_through_the_full_pipeline() -> None:
    market = make_market(11)
    events = quintiles(random_events(12, market.sessions, 200, 1500))
    top = events.filter(pl.col("quintile") == 5)
    prices = plant(market.prices, top, market.sessions, PLANTED)
    result = abnormal_returns(
        events, prices, market.factors, "ff5umd", (-250, -30), (-10, 60), CAL, keep_residuals=True
    )
    study = event_study(
        result.ar, events, sigma=result.events, residuals=result.residuals, reps=300, seed=1
    )
    summary = {(r["group"], r["horizon"]): r for r in study.summary.iter_rows(named=True)}
    q5 = summary[("Q5", "CAR[0,5]")]
    assert abs(q5["car"] - PLANTED) < 2 * q5["se"]
    spread = summary[(SPREAD, "CAR[0,5]")]
    assert abs(spread["car"] - PLANTED) < 2 * spread["se"]
    assert abs(summary[("Q1", "CAR[0,5]")]["car"]) < 2 * summary[("Q1", "CAR[0,5]")]["se"]
    assert q5["t_bmp"] > 3 and q5["t_kp"] is not None and q5["boot_p"] < 0.01
    assert study.kp_correlation is not None and abs(study.kp_correlation) < 0.1
    path = study.path.filter((pl.col("group") == "Q5") & (pl.col("rel_day") == 5)).row(
        0, named=True
    )
    for lo, hi in (("lo_t", "hi_t"), ("lo_bmp", "hi_bmp"), ("lo_boot", "hi_boot")):
        assert path[lo] < path["caar"] < path[hi]
    assert study.n_events == result.events.filter(pl.col("drop_reason").is_null()).height


def test_band_coverage_over_many_simulations() -> None:
    """95% band coverage of CAR[0,5] for a planted +1%, through ``event_study`` itself.

    1,000 draws (the spec asks for at least 200) keep the binomial error of the coverage
    rate near 0.7 points, so the 93-97% check is not left to chance.
    """
    rng = np.random.default_rng(7)
    covered, sims = 0, 1000
    for sim in range(sims):
        n = 200
        rel = np.arange(-2, 6)
        ar = rng.normal(0.0, 0.02, size=(n, len(rel)))
        ar[:, rel >= 0] += PLANTED / 6
        frame = pl.DataFrame(
            {
                "doc_id": np.repeat([f"s{sim}e{i}" for i in range(n)], len(rel)),
                "permno": np.repeat(np.arange(n), len(rel)),
                "rel_day": np.tile(rel, n),
                "ar": ar.ravel(),
            }
        )
        signals = pl.DataFrame(
            {
                "doc_id": [f"s{sim}e{i}" for i in range(n)],
                "permno": np.arange(n),
                "quintile": [5] * n,
            }
        )
        study = event_study(frame, signals, bands=("t",), horizons=(5,))
        row = study.summary.filter(pl.col("group") == "Q5").row(0, named=True)
        covered += abs(row["car"] - PLANTED) <= Z95 * row["se"]
    assert 0.93 <= covered / sims <= 0.97


def test_kp_correlation_needs_overlap() -> None:
    residuals = pl.DataFrame({"doc_id": ["a", "b"], "session": [1, 500], "std_resid": [0.1, 0.2]})
    assert kp_correlation(residuals) is None
    assert kp_correlation(residuals.head(1)) is None
