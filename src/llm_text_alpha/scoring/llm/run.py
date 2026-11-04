"""``lta score llm`` and ``lta llm estimate``: samples, providers and the tables they write.

``scores`` gets one row per document with a valid answer. ``llm_calls`` gets one row per
request, refusals and invalid answers included (section 7b).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from llm_text_alpha.io import write_parquet_atomic
from llm_text_alpha.scoring.common import SCORES_SCHEMA
from llm_text_alpha.scoring.llm.base import LLMProvider, ScoreResult
from llm_text_alpha.scoring.llm.cache import ResponseCache
from llm_text_alpha.scoring.llm.fake_provider import FakeProvider
from llm_text_alpha.scoring.llm.local_provider import LocalProvider
from llm_text_alpha.scoring.llm.registry import ModelRegistry
from llm_text_alpha.scoring.llm.spend import PriceTable, SpendTracker
from llm_text_alpha.settings import Settings

CALLS_SCHEMA = pl.Schema(
    [
        ("cache_key", pl.String()),
        ("doc_id", pl.String()),
        ("model_id", pl.String()),
        ("provider", pl.String()),
        ("mode", pl.String()),
        ("batch_id", pl.String()),
        ("input_tokens", pl.Int64()),
        ("cache_write_tokens", pl.Int64()),
        ("cache_read_tokens", pl.Int64()),
        ("output_tokens", pl.Int64()),
        ("usd", pl.Float64()),
        ("latency_ms", pl.Float64()),
        ("batch_turnaround_s", pl.Float64()),
        ("stop_reason", pl.String()),
        ("schema_valid", pl.Boolean()),
        ("attempts", pl.Int64()),
        ("run_id", pl.String()),
        ("masked", pl.Boolean()),
    ]
)


def draw_sample(
    documents: pl.DataFrame, n: int, stratify: str | Sequence[str], seed: int
) -> pl.DataFrame:
    """A stratified sample of ``n`` documents, proportional to each stratum's size.

    Strata come from ``stratify`` columns (``year`` is derived from ``event_ts_utc``).
    Documents are sorted by ``doc_id`` first, so the draw depends only on the seed and the
    corpus. Rounding is resolved by largest remainder, so the total is exactly ``n`` (or the
    corpus size, if smaller).
    """
    keys = [stratify] if isinstance(stratify, str) else list(stratify)
    frame = documents.sort("doc_id")
    if "year" in keys and "year" not in frame.columns:
        frame = frame.with_columns(pl.col("event_ts_utc").dt.year().alias("year"))
    if n >= frame.height:
        return frame
    sizes = frame.group_by(keys, maintain_order=True).len().sort(keys)
    exact = sizes["len"].to_numpy() * n / frame.height
    base = exact.astype(int)
    order = (exact - base).argsort()[::-1]
    for i in order[: n - int(base.sum())]:
        base[i] += 1
    parts = []
    for row, take in zip(sizes.iter_rows(named=True), base.tolist(), strict=True):
        stratum = frame.filter(pl.all_horizontal(pl.col(k) == row[k] for k in keys))
        parts.append(stratum.sample(n=take, seed=seed, shuffle=True))
    return pl.concat(parts).sort("doc_id")


def tracker(settings: Settings, price_table: Path, max_age_days: int = 45) -> SpendTracker:
    """The project's spend tracker: ``<llm cache>/spend.duckdb`` at ``LLM_BUDGET_USD``."""
    prices = PriceTable.load(price_table, max_age_days=max_age_days)
    return SpendTracker(settings.llm_cache_path / "spend.duckdb", settings.llm_budget_usd, prices)


def make_provider(
    name: str,
    settings: Settings,
    registry: ModelRegistry,
    cache: ResponseCache,
    run_id: str,
    price_table: Path,
) -> LLMProvider:
    """The provider for ``name``; only ``anthropic`` can spend, and only through a tracker."""
    if name == "fake":
        return FakeProvider(cache)
    if name == "local":
        return LocalProvider(settings.local_llm_base_url, settings.local_llm_model, cache, registry)
    if name == "anthropic":
        from llm_text_alpha.scoring.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            settings, cache, tracker(settings, price_table), registry, run_id=run_id
        )
    raise ValueError(f"unknown provider {name!r}")


@dataclass(frozen=True)
class RunTables:
    """What one scoring run produced."""

    scores: pl.DataFrame
    calls: pl.DataFrame


def tables(
    results: Sequence[ScoreResult],
    provider: str,
    prompt_version: str,
    masked: bool,
    run_id: str,
) -> RunTables:
    """``scores`` and ``llm_calls`` rows for ``results``.

    ``scores`` gets valid answers scored in this run; an answer served from the cache was
    written when it was first scored, so a re-run adds no duplicate rows. ``llm_calls`` gets
    every request, cache hits included.
    """
    now = datetime.now(UTC)
    score_rows: list[dict[str, Any]] = []
    call_rows: list[dict[str, Any]] = []
    for r in results:
        call_rows.append(
            {
                "cache_key": r.cache_key,
                "doc_id": r.doc_id,
                "model_id": r.model_id,
                "provider": provider,
                "mode": r.mode,
                "batch_id": r.batch_id,
                "input_tokens": r.usage.input_tokens,
                "cache_write_tokens": r.usage.cache_write_tokens,
                "cache_read_tokens": r.usage.cache_read_tokens,
                "output_tokens": r.usage.output_tokens,
                "usd": r.usd,
                "latency_ms": r.latency_ms,
                "batch_turnaround_s": r.batch_turnaround_s,
                "stop_reason": r.stop_reason,
                "schema_valid": r.schema_valid,
                "attempts": r.attempts,
                "run_id": run_id,
                "masked": masked,
            }
        )
        if r.payload is None or r.from_cache:
            continue
        aux = r.payload.model_dump(exclude={"score", "label"}) | {"stop_reason": r.stop_reason}
        score_rows.append(
            {
                "doc_id": r.doc_id,
                "scorer": "llm",
                "model_id": r.model_id,
                "prompt_version": prompt_version,
                "masked": masked,
                "score": r.payload.score,
                "label": r.payload.label,
                "aux_json": json.dumps(aux, sort_keys=True),
                "latency_ms": r.latency_ms,
                "cost_usd": r.usd,
                "scored_at": now,
            }
        )
    return RunTables(
        pl.DataFrame(score_rows, schema=SCORES_SCHEMA),
        pl.DataFrame(call_rows, schema=CALLS_SCHEMA),
    )


def write_tables(run: RunTables, scores_dir: Path, calls_dir: Path, run_id: str) -> None:
    """One Parquet part per run under each table's folder."""
    write_parquet_atomic(run.scores, scores_dir / f"part-{run_id}.parquet")
    write_parquet_atomic(run.calls, calls_dir / f"part-{run_id}.parquet")
