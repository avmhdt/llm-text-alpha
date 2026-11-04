"""Shared builders for the LLM tests: committed configs, a priced tracker, requests."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from llm_text_alpha.scoring.llm.anthropic_provider import AnthropicProvider
from llm_text_alpha.scoring.llm.base import ScoreRequest
from llm_text_alpha.scoring.llm.cache import ResponseCache
from llm_text_alpha.scoring.llm.prompts import build_requests, load_prompt
from llm_text_alpha.scoring.llm.registry import ModelRegistry
from llm_text_alpha.scoring.llm.spend import PriceTable, SpendTracker
from llm_text_alpha.settings import Settings
from tests.fake_anthropic import FakeAnthropic

REPO = Path(__file__).resolve().parents[1]
TODAY = date(2026, 11, 4)
"""The price table's retrieval date: tests stay valid however old the table gets."""
HAIKU = "claude-haiku-5-5"


def prices() -> PriceTable:
    return PriceTable.load(REPO / "configs" / "llm_prices.yaml", today=TODAY)


def registry() -> ModelRegistry:
    return ModelRegistry.load(REPO / "configs" / "models.yaml")


def requests(
    n: int, model: str = HAIKU, texts: Sequence[str] | None = None, start: int = 0
) -> list[ScoreRequest]:
    bodies = (
        list(texts)
        if texts is not None
        else [f"Quarter {i}: revenue rose {i % 17} percent and margins held." for i in range(n)]
    )
    docs = pl.DataFrame(
        {"doc_id": [f"doc{start + i:05d}" for i in range(len(bodies))], "text": bodies}
    )
    prompt = load_prompt(REPO / "configs" / "prompts" / "score_v1.md")
    return build_requests(docs, model, prompt, 9_000, 512, registry().get(model).request)


def settings(tmp_path: Path, budget: float = 25.0, key: str = "sk-test-not-a-key") -> Settings:
    values: dict[str, Any] = {
        "sec_user_agent": "lta-tests tests@example.com",
        "data_dir_override": str(tmp_path / "data"),
        "anthropic_api_key": key,
        "llm_budget_usd": budget,
    }
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


def tracker(root: Path, budget: float = 25.0) -> SpendTracker:
    return SpendTracker(root / "spend.duckdb", budget, prices())


def provider(
    tmp_path: Path,
    fake: FakeAnthropic,
    budget: float = 25.0,
    sleep: Any = None,
    **kwargs: Any,
) -> AnthropicProvider:
    root = tmp_path / "llm_cache"
    return AnthropicProvider(
        settings(tmp_path, budget),
        ResponseCache(root),
        tracker(root, budget),
        registry(),
        run_id="test-run",
        http_client=fake.client(),
        sleep=sleep or (lambda _seconds: None),
        poll_seconds=0.0,
        **kwargs,
    )
