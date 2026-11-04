from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from llm_text_alpha.corpus.calendar import xnys
from llm_text_alpha.evaluation import capacity as cap
from llm_text_alpha.evaluation import portfolio as pf

CAL = xnys()
SESSIONS = CAL.sessions(date(2024, 1, 2), date(2024, 3, 29))


def prices(n_firms: int = 3, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for permno in range(1, n_firms + 1):
        for day in SESSIONS:
            rows.append(
                {
                    "permno": permno,
                    "date": day,
                    "ret": float(rng.normal(0, 0.01)),
                    "ret_open_close": float(rng.normal(0, 0.01)),
                    "close": 50.0,
                    "volume": 100_000.0 * permno,
                }
            )
    return pl.DataFrame(rows)


def signal(doc: str, permno: int, entry: date, quintile: int, rank: int = 100) -> dict[str, object]:
    return {
        "doc_id": doc,
        "permno": permno,
        "entry_date": entry,
        "quintile": quintile,
        "rank": rank,
    }


def test_cost_bands() -> None:
    ranks = pl.DataFrame({"rank": [1, 200, 201, 500, 501, 1000, None]})
    out = ranks.select(pf.cost_bps(pl.col("rank"), pf.DEFAULT_COSTS).alias("bps"))["bps"]
    assert out.to_list() == [5.0, 5.0, 10.0, 10.0, 20.0, 20.0, 20.0]


def test_returns_match_a_hand_calculation() -> None:
    px = prices()
    entry = SESSIONS[5]
    sig = pl.DataFrame(
        [signal("a", 1, entry, 5), signal("b", 2, entry, 5, rank=300), signal("c", 3, entry, 1)]
    )
    result = pf.daily_returns(sig, px, CAL, hold=3)
    daily = result.daily
    assert daily["date"].to_list() == SESSIONS[5:8]
    look = {(r["permno"], r["date"]): r for r in px.iter_rows(named=True)}

    def r(permno: int, k: int) -> float:
        row = look[(permno, SESSIONS[5 + k])]
        return float(row["ret_open_close"] if k == 0 else row["ret"])

    for k in range(3):
        long_gross = (r(1, k) + r(2, k)) / 2
        short_gross = -r(3, k)
        charge_a = 5e-4 if k in (0, 2) else 0.0
        charge_b = 10e-4 if k in (0, 2) else 0.0
        charge_c = 5e-4 if k in (0, 2) else 0.0
        row = daily.row(k, named=True)
        assert row["ls_gross"] == pytest.approx(long_gross + short_gross)
        expected_net = long_gross - (charge_a + charge_b) / 2 + short_gross - charge_c
        assert row["ls_net"] == pytest.approx(expected_net)
        assert row["n_long"] == 2 and row["n_short"] == 1
    assert result.n_positions == 3 and result.missing_returns == 0
    doubled = pf.daily_returns(sig, px, CAL, hold=3, cost_mult=2.0).daily
    assert (daily["ls_net"] - doubled["ls_net"]).to_list()[1] == pytest.approx(0.0)
    assert (daily["ls_net"] - doubled["ls_net"]).to_list()[0] == pytest.approx(
        (5e-4 + 10e-4) / 2 + 5e-4
    )


def test_missing_returns_count_as_cash_and_end_cuts() -> None:
    px = prices().filter(~((pl.col("permno") == 1) & (pl.col("date") == SESSIONS[6])))
    sig = pl.DataFrame([signal("a", 1, SESSIONS[5], 5), signal("c", 3, SESSIONS[5], 1)])
    result = pf.daily_returns(sig, px, CAL, hold=3, end=SESSIONS[6])
    assert result.missing_returns == 1
    assert result.daily.height == 2
    assert pf.active_days(result.daily).height == 2


def test_quintiles_outside_the_legs_are_ignored() -> None:
    sig = pl.DataFrame([signal("a", 1, SESSIONS[5], 3)])
    assert pf.positions(sig).is_empty()


def test_with_rank_uses_the_prior_month_end() -> None:
    universe = pl.DataFrame(
        {
            "security_id": ["S1", "S1"],
            "month_end": [date(2024, 1, 31), date(2024, 2, 29)],
            "rank": [10, 700],
        }
    )
    events = pl.DataFrame({"permno": [1, 1], "entry_date": [date(2024, 2, 29), date(2024, 3, 1)]})
    ranked = pf.with_rank(events, universe, pl.DataFrame({"permno": [1], "security_id": ["S1"]}))
    assert ranked.sort("entry_date")["rank"].to_list() == [10, 700]


def test_factor_alpha_recovers_a_planted_intercept() -> None:
    rng = np.random.default_rng(3)
    days = CAL.sessions(date(2018, 1, 2), date(2021, 12, 31))
    n = len(days)
    factors = pl.DataFrame({"date": days, **{f: rng.normal(0, 0.01, n) for f in pf.ALPHA_FACTORS}})
    y = 0.0008 + 0.5 * factors["mktrf"].to_numpy() + rng.normal(0, 0.002, n)
    fit = pf.factor_alpha(pl.DataFrame({"date": days, "ls_net": y}), factors)
    assert fit.alpha_daily == pytest.approx(0.0008, abs=0.0002)
    assert fit.betas["mktrf"] == pytest.approx(0.5, abs=0.02)
    assert fit.t_alpha > 5 and fit.n_days == n
    assert fit.alpha_annual == pytest.approx(fit.alpha_daily * 252)
    with pytest.raises(ValueError, match="days to fit"):
        pf.factor_alpha(pl.DataFrame({"date": days[:5], "ls_net": y[:5]}), factors)


def test_liquidity_uses_only_prior_sessions() -> None:
    px = prices(1)
    liquid = cap.liquidity(px, 4).sort("entry_date")
    assert liquid["adv"][0] is None
    expected = float(np.std(px["ret"].to_numpy()[:4], ddof=1))
    assert liquid["sigma"][4] == pytest.approx(expected)
    assert liquid["adv"][4] == pytest.approx(50.0 * 100_000.0)


def test_sizing_caps_large_positions_and_charges_impact() -> None:
    px = prices()
    entry = SESSIONS[30]
    sig = pl.DataFrame([signal("a", 1, entry, 5), signal("c", 3, entry, 1)])
    liquid = cap.liquidity(px, 20)
    small = cap.sized(sig, liquid, CAL, 5, 1_000.0, 0.10, 0.1)
    assert small["scale"].to_list() == [1.0, 1.0]
    big = cap.sized(sig, liquid, CAL, 5, 1e9, 0.10, 0.1)
    adv_a = 50.0 * 100_000.0
    assert big.filter(pl.col("doc_id") == "a")["q_cap"][0] == pytest.approx(0.1 * adv_a)
    assert big["scale"].to_list() == pytest.approx([5e5 / 1e9, 1.5e6 / 1e9])
    assert (big["impact"] > small["impact"]).all()


def test_breakpoints_interpolate_in_log_aum() -> None:
    curve = pl.DataFrame({"aum_usd": [1e6, 1e7, 1e8], "alpha_annual": [0.10, 0.06, -0.02]})
    points = cap.breakpoints(curve)
    assert points["alpha_at_min_aum"] == 0.10
    assert points["aum_half"] == pytest.approx(10 ** (7 + 0.01 / 0.08))
    assert points["aum_zero"] == pytest.approx(10 ** (7 + 0.06 / 0.08))
    flat = cap.breakpoints(pl.DataFrame({"aum_usd": [1e6, 1e7], "alpha_annual": [0.1, 0.09]}))
    assert flat["aum_half"] is None and flat["aum_zero"] is None
    negative = cap.breakpoints(pl.DataFrame({"aum_usd": [1e6, 1e7], "alpha_annual": [-0.1, -0.2]}))
    assert negative["aum_zero"] is None
