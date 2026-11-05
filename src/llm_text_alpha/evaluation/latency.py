"""Cost per signal and the acceptance-to-order latency budget (FR-22, AC-H-03).

Measured components:
- EDGAR fetch and parse of one Item 2.02 filing (index page and documents through the
  rate-limited client, then text extraction), on a live sample of recent filings;
- scoring per document: dictionary and FinBERT from their production runs, Claude from its
  synchronous calls, the local model from its calls; batch turnaround for batch runs;
- the gate and signal decision for one event.

EDGAR's dissemination delay (acceptance to public availability) is not measured (D-13 skip)
and is carried as a stated input. Times are milliseconds; costs USD.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import polars as pl


def quantiles(values: Sequence[float] | np.ndarray) -> dict[str, Any]:
    """p50, p95 and p99 of ``values`` (``None`` when empty)."""
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None}
    return {
        "n": int(arr.size),
        "p50_ms": float(np.quantile(arr, 0.50)),
        "p95_ms": float(np.quantile(arr, 0.95)),
        "p99_ms": float(np.quantile(arr, 0.99)),
    }


def time_calls(fn: Callable[[], Any], n: int) -> list[float]:
    """Wall time in ms of ``n`` calls of ``fn``."""
    out = []
    for _ in range(n):
        started = time.perf_counter()
        fn()
        out.append((time.perf_counter() - started) * 1000.0)
    return out


def traded_share(signals: pl.DataFrame, scorer: str) -> float | None:
    """Share of a scorer's raw-text signals that enter a leg (quintile 1 or 5)."""
    part = signals.filter(
        (pl.col("scorer") == scorer) & ~pl.col("masked") & (pl.col("quintile") > 0)
    )
    if part.is_empty():
        return None
    return float(part["quintile"].is_in([1, 5]).mean())  # type: ignore[arg-type]


def cost_rows(
    scorer: str,
    model_id: str,
    calls: pl.DataFrame | None,
    scores: pl.DataFrame | None,
    traded: float | None,
    sync_multiplier: float,
) -> dict[str, Any]:
    """USD per document (batch and sync), USD per traded signal and latency for one scorer.

    ``calls`` are LLM calls (mode, usd, latency_ms, batch_turnaround_s); ``scores`` are
    dictionary or FinBERT scores with ``latency_ms``. ``sync_multiplier`` converts a batch
    cost to a synchronous one when no synchronous call was made (2 for a 50% batch discount).
    """
    row: dict[str, Any] = {"scorer": scorer, "model_id": model_id}
    if calls is not None and not calls.is_empty():
        fresh = calls.filter(pl.col("mode") != "cache_hit")
        batch = fresh.filter(pl.col("mode") == "batch")
        sync = fresh.filter(pl.col("mode").is_in(["sync", "local"]))
        usd_batch = float(batch["usd"].mean()) if batch.height else None  # type: ignore[arg-type]
        usd_sync = float(sync["usd"].mean()) if sync.height else None  # type: ignore[arg-type]
        if usd_sync is None and usd_batch is not None:
            usd_sync = usd_batch * sync_multiplier
        row |= {"usd_per_doc_batch": usd_batch, "usd_per_doc_sync": usd_sync}
        row |= quantiles(sync["latency_ms"].to_numpy())
        turn = batch["batch_turnaround_s"].drop_nulls()
        row["batch_turnaround_max_s"] = float(turn.max()) if turn.len() else None  # type: ignore[arg-type]
    elif scores is not None:
        cost = float(scores["cost_usd"].fill_null(0.0).mean() or 0.0)  # type: ignore[arg-type]
        row |= {"usd_per_doc_batch": cost, "usd_per_doc_sync": cost}
        row |= quantiles(scores["latency_ms"].to_numpy())
        row["batch_turnaround_max_s"] = None
    per_doc = row.get("usd_per_doc_sync")
    row["traded_share"] = traded
    row["usd_per_traded_signal"] = per_doc / traded if per_doc is not None and traded else None
    return row


def budget(
    fetch_ms: Mapping[str, Any], decision_ms: Mapping[str, Any], scoring: pl.DataFrame
) -> pl.DataFrame:
    """Acceptance-to-order latency per scorer at p50 and p95, dissemination excluded."""
    rows = []
    f50, f95 = fetch_ms.get("p50_ms") or 0.0, fetch_ms.get("p95_ms") or 0.0
    d50, d95 = decision_ms.get("p50_ms") or 0.0, decision_ms.get("p95_ms") or 0.0
    for r in scoring.iter_rows(named=True):
        if r.get("p50_ms") is None:
            continue
        rows.append(
            {
                "scorer": r["scorer"],
                "fetch_parse_p50_ms": f50,
                "score_p50_ms": r["p50_ms"],
                "decision_p50_ms": d50,
                "total_p50_ms": f50 + r["p50_ms"] + d50,
                "total_p95_ms": f95 + r["p95_ms"] + d95,
            }
        )
    return pl.DataFrame(rows)
