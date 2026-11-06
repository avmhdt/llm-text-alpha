"""Contamination protocol tables (FR-20, AC-H-02; pre-registered hypothesis H3).

For one LLM scorer the event set is every event it scored on both raw and masked text, so
the four cells (before/after the model's training cutoff, raw/masked) hold the same events.
Each cell reports the top-minus-bottom trailing-quintile mean of CAR[0,5] with a Welch 95%
interval and the Spearman IC of ``score_z`` with CAR[0,5] with a Fisher-z 95% interval.
Cells with fewer than 500 events are flagged.

The share of apparent alpha that disappears is ``(pre_raw - post_masked) / pre_raw``, with
the two margins ``(pre_raw - post_raw) / pre_raw`` (the cutoff) and
``(pre_raw - pre_masked) / pre_raw`` (masking). The difference-in-differences against a
baseline is ``(llm_pre - llm_post) - (base_pre - base_post)`` on raw text and on the same
events, with a 95% interval from resampling events within each window.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date
from typing import Any

import numpy as np
import polars as pl

from llm_text_alpha.evaluation.ablation import spread

UNDERPOWERED = 500


def cell_stats(frame: pl.DataFrame, car: str = "car_5") -> dict[str, Any]:
    """Spread with Welch CI and IC with Fisher CI for one cell."""
    ranked = frame.filter(pl.col("quintile") > 0)
    s = spread(ranked, car)
    se = (
        s["spread"] / s["t_spread"]
        if s["t_spread"] not in (0.0,) and math.isfinite(s["t_spread"])
        else float("nan")
    )
    data = ranked.select("score_z", car).drop_nulls()
    n = data.height
    ic = (
        float(data.select(pl.corr("score_z", car, method="spearman")).item())
        if n > 3
        else float("nan")
    )
    half = 1.96 / math.sqrt(n - 3) if n > 3 else float("nan")
    lo_ic, hi_ic = (
        (math.tanh(math.atanh(ic) - half), math.tanh(math.atanh(ic) + half))
        if n > 3 and abs(ic) < 1
        else (float("nan"), float("nan"))
    )
    return {
        "n": ranked.height,
        "spread": s["spread"],
        "spread_lo95": s["spread"] - 1.96 * se,
        "spread_hi95": s["spread"] + 1.96 * se,
        "t_spread": s["t_spread"],
        "ic": ic,
        "ic_lo95": lo_ic,
        "ic_hi95": hi_ic,
        "underpowered": ranked.height < UNDERPOWERED,
    }


def _window(frame: pl.DataFrame, post_start: date) -> pl.DataFrame:
    return frame.with_columns(
        pl.when(pl.col("event_ts_utc").dt.date() >= post_start)
        .then(pl.lit("post"))
        .otherwise(pl.lit("pre"))
        .alias("window")
    )


def paired_events(signals: pl.DataFrame, scorer: str) -> pl.Series:
    """Events the scorer scored on both raw and masked text."""
    part = signals.filter(pl.col("scorer") == scorer)
    raw = set(part.filter(~pl.col("masked"))["doc_id"].to_list())
    masked = set(part.filter(pl.col("masked"))["doc_id"].to_list())
    return pl.Series("doc_id", sorted(raw & masked))


def two_by_two(
    signals: pl.DataFrame, cars: pl.DataFrame, scorer: str, post_start: date
) -> pl.DataFrame:
    """The four cells for one LLM scorer on its paired events."""
    events = paired_events(signals, scorer)
    rows = []
    for masked in (False, True):
        part = (
            signals.filter((pl.col("scorer") == scorer) & (pl.col("masked") == masked))
            .filter(pl.col("doc_id").is_in(events.to_list()))
            .join(cars, on="doc_id", how="inner")
        )
        part = _window(part, post_start)
        for window in ("pre", "post"):
            stats = cell_stats(part.filter(pl.col("window") == window))
            rows.append(
                {"scorer": scorer, "text": "masked" if masked else "raw", "window": window, **stats}
            )
    return pl.DataFrame(rows)


def disappearing(cells: pl.DataFrame) -> dict[str, float | None]:
    """Shares of the pre-cutoff raw spread that disappear, in total and by margin."""

    def get(text: str, window: str) -> float:
        row = cells.filter((pl.col("text") == text) & (pl.col("window") == window))
        return float(row["spread"][0]) if row.height else float("nan")

    base = get("raw", "pre")
    if not math.isfinite(base) or base == 0:
        return {"share_total": None, "share_cutoff": None, "share_masking": None}
    return {
        "share_total": (base - get("masked", "post")) / base,
        "share_cutoff": (base - get("raw", "post")) / base,
        "share_masking": (base - get("masked", "pre")) / base,
    }


def _spread_fast(quintile: np.ndarray, car: np.ndarray) -> float:
    top = car[quintile == 5]
    bottom = car[quintile == 1]
    if len(top) == 0 or len(bottom) == 0:
        return float("nan")
    return float(top.mean() - bottom.mean())


def _avg_rank(x: np.ndarray) -> np.ndarray:
    """Ranks with ties given their average rank (1-based)."""
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    first = np.cumsum(counts) - counts
    return (first + (counts + 1) / 2.0)[inverse]


def _spearman_fast(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 4:
        return float("nan")
    rx, ry = _avg_rank(x), _avg_rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _interval(draws: list[float]) -> tuple[float, float]:
    clean = np.array([d for d in draws if math.isfinite(d)])
    if not len(clean):
        return float("nan"), float("nan")
    return float(np.quantile(clean, 0.025)), float(np.quantile(clean, 0.975))


def did(
    signals: pl.DataFrame,
    cars: pl.DataFrame,
    scorer: str,
    baseline: str,
    post_start: date,
    reps: int = 500,
    seed: int = 0,
) -> dict[str, Any]:
    """(LLM pre - post) minus (baseline pre - post), raw text, the same events, bootstrap CI.

    ``did`` uses the Q5 - Q1 CAR[0,5] spread (pre-registered). ``did_ic`` uses the Spearman
    IC of ``score_z`` with CAR[0,5] instead (secondary): it stays defined when ties leave a
    scorer's top or bottom trailing quintile empty.
    """
    events = paired_events(signals, scorer).to_list()

    def frame(name: str) -> pl.DataFrame:
        return (
            signals.filter(
                (pl.col("scorer") == name) & ~pl.col("masked") & (pl.col("quintile") > 0)
            )
            .filter(pl.col("doc_id").is_in(events))
            .select(
                "doc_id",
                "event_ts_utc",
                pl.col("quintile").alias(f"q_{name}"),
                pl.col("score_z").alias(f"z_{name}"),
            )
        )

    data = _window(
        frame(scorer)
        .join(frame(baseline).drop("event_ts_utc"), on="doc_id")
        .join(cars, on="doc_id")
        .drop_nulls([f"z_{scorer}", f"z_{baseline}", "car_5"]),
        post_start,
    )
    parts = {w: data.filter(pl.col("window") == w) for w in ("pre", "post")}
    arrays = {
        w: (
            p[f"q_{scorer}"].to_numpy(),
            p[f"q_{baseline}"].to_numpy(),
            p[f"z_{scorer}"].to_numpy(),
            p[f"z_{baseline}"].to_numpy(),
            p["car_5"].to_numpy(),
        )
        for w, p in parts.items()
    }

    def estimate(idx: dict[str, np.ndarray]) -> tuple[float, float]:
        spreads, ics = {}, {}
        for w, (ql, qb, zl, zb, car) in arrays.items():
            i = idx[w]
            spreads[w] = (_spread_fast(ql[i], car[i]), _spread_fast(qb[i], car[i]))
            ics[w] = (_spearman_fast(zl[i], car[i]), _spearman_fast(zb[i], car[i]))
        d_spread = (spreads["pre"][0] - spreads["post"][0]) - (
            spreads["pre"][1] - spreads["post"][1]
        )
        d_ic = (ics["pre"][0] - ics["post"][0]) - (ics["pre"][1] - ics["post"][1])
        return d_spread, d_ic

    full = {w: np.arange(len(a[4])) for w, a in arrays.items()}
    point, point_ic = estimate(full)
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    draws_ic: list[float] = []
    for _ in range(reps):
        idx = {w: rng.integers(0, len(a[4]), len(a[4])) for w, a in arrays.items() if len(a[4])}
        if len(idx) == 2:
            d, d_ic = estimate(idx)
            draws.append(d)
            draws_ic.append(d_ic)
    lo, hi = _interval(draws)
    lo_ic, hi_ic = _interval(draws_ic)
    return {
        "scorer": scorer,
        "baseline": baseline,
        "n_pre": len(arrays["pre"][4]),
        "n_post": len(arrays["post"][4]),
        "did": point,
        "did_lo95": lo,
        "did_hi95": hi,
        "contamination_found": bool(lo > 0),
        "did_ic": point_ic,
        "did_ic_lo95": lo_ic,
        "did_ic_hi95": hi_ic,
        "contamination_found_ic": bool(lo_ic > 0),
    }


def summary_rows(
    model_id: str,
    scorer: str,
    post_start: date,
    cells: pl.DataFrame,
    dids: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """One headline row per model for the README table."""
    shares = disappearing(cells)
    return {
        "model_id": model_id,
        "scorer": scorer,
        "post_cutoff_start": post_start.isoformat(),
        **shares,
        **{f"did_vs_{d['baseline']}": d["did"] for d in dids},
        **{f"did_vs_{d['baseline']}_lo95": d["did_lo95"] for d in dids},
        **{f"did_vs_{d['baseline']}_hi95": d["did_hi95"] for d in dids},
        **{f"did_ic_vs_{d['baseline']}": d["did_ic"] for d in dids},
        **{f"did_ic_vs_{d['baseline']}_lo95": d["did_ic_lo95"] for d in dids},
        **{f"did_ic_vs_{d['baseline']}_hi95": d["did_ic_hi95"] for d in dids},
    }
