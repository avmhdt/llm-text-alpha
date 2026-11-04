from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from llm_text_alpha.scoring.llm.base import TokenUsage
from llm_text_alpha.scoring.llm.spend import (
    BudgetExceededError,
    PriceTable,
    SpendTracker,
    StalePriceTableError,
)
from tests import llm_support as s


def test_haiku_5_5_rates_match_the_verified_table() -> None:
    table = s.prices()
    # 100K tokens: the top of the standard tier, so USD = per-MTok rate / 10.
    tokens_in = TokenUsage(input_tokens=100_000)
    tokens_out = TokenUsage(input_tokens=1, output_tokens=100_000)
    assert table.cost(s.HAIKU, tokens_in, batch=False) == pytest.approx(0.010)
    assert table.cost(s.HAIKU, tokens_out, batch=False) == pytest.approx(0.050, rel=1e-4)
    assert table.cost(s.HAIKU, tokens_in, batch=True) == pytest.approx(0.005)
    assert table.cost(s.HAIKU, tokens_out, batch=True) == pytest.approx(0.025, rel=1e-4)
    read = TokenUsage(cache_read_tokens=100_000)
    assert table.cost(s.HAIKU, read, batch=False) == pytest.approx(0.001)


def test_long_context_tier_applies_above_the_threshold() -> None:
    table = s.prices()
    long_prompt = TokenUsage(input_tokens=200_000)
    assert table.cost(s.HAIKU, long_prompt, batch=False) == pytest.approx(0.10)  # 0.2M x $0.50
    assert table.prices(s.HAIKU).rates(50_000).input == pytest.approx(0.10)


def test_worst_case_bounds_any_usage_of_that_size() -> None:
    table = s.prices()
    worst = table.worst_case(s.HAIKU, 3_000, 512, batch=True)
    for usage in (
        TokenUsage(input_tokens=3_000, output_tokens=512),
        TokenUsage(cache_write_tokens=3_000, output_tokens=512),
        TokenUsage(input_tokens=1_000, cache_read_tokens=2_000, output_tokens=512),
    ):
        assert table.cost(s.HAIKU, usage, batch=True) <= worst + 1e-12


def test_stale_table_and_unknown_model_are_refused() -> None:
    path = s.REPO / "configs" / "llm_prices.yaml"
    with pytest.raises(StalePriceTableError):
        PriceTable.load(path, today=s.TODAY + timedelta(days=46), max_age_days=45)
    with pytest.raises(KeyError, match="no price row"):
        s.prices().cost("claude-unknown", TokenUsage(), batch=False)


def test_reserve_settle_release_and_totals(tmp_path: Path) -> None:
    tracker = s.tracker(tmp_path, budget=1.0)
    first = tracker.reserve("run-a", s.HAIKU, 0.40)
    second = tracker.reserve("run-b", "claude-haiku-4-5", 0.30)
    assert tracker.committed_usd() == pytest.approx(0.70)
    assert len(tracker.open_reservations()) == 2
    tracker.settle(first, [TokenUsage(input_tokens=100_000)], batch=True)
    tracker.release(second)
    assert tracker.committed_usd() == pytest.approx(0.005)
    assert tracker.open_reservations() == []
    assert tracker.totals("model") == [(s.HAIKU, pytest.approx(0.005))]
    assert tracker.totals("run") == [("run-a", pytest.approx(0.005))]
    assert tracker.n_entries() == 4
    with pytest.raises(ValueError, match="already closed"):
        tracker.release(first)


def test_rejected_reservation_writes_nothing(tmp_path: Path) -> None:
    tracker = s.tracker(tmp_path, budget=0.50)
    tracker.reserve("run", s.HAIKU, 0.45)
    before = tracker.n_entries()
    with pytest.raises(BudgetExceededError, match=r"\$0.50 budget"):
        tracker.reserve("run", s.HAIKU, 0.10)
    assert tracker.n_entries() == before
    assert tracker.committed_usd() == pytest.approx(0.45)


def test_ledger_persists_across_instances(tmp_path: Path) -> None:
    first = s.tracker(tmp_path, budget=1.0)
    first.settle_usd(first.reserve("run", s.HAIKU, 0.9), 0.6)
    second = SpendTracker(tmp_path / "spend.duckdb", 1.0, s.prices())
    assert second.committed_usd() == pytest.approx(0.6)
    with pytest.raises(BudgetExceededError):
        second.reserve("run", s.HAIKU, 0.5)
