from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from llm_text_alpha.scoring.llm.anthropic_provider import (
    AnthropicProvider,
    ProviderConfigError,
    UnsupportedRequestError,
)
from llm_text_alpha.scoring.llm.cache import ResponseCache
from llm_text_alpha.scoring.llm.spend import BudgetExceededError
from tests import llm_support as s
from tests.fake_anthropic import FULL_CAPABILITIES, FakeAnthropic


def test_batch_scores_settles_and_records_turnaround(tmp_path: Path) -> None:
    fake = FakeAnthropic()
    provider = s.provider(tmp_path, fake)
    results = provider.score(s.requests(10), "batch")
    assert len(results) == 10
    assert all(r.schema_valid and r.payload is not None for r in results)
    assert {r.mode for r in results} == {"batch"}
    assert {r.batch_turnaround_s for r in results} == {1800.0}
    assert fake.count("POST", "/v1/messages/batches") == 1
    assert fake.count("GET", "/v1/models/") == 1
    assert fake.count("POST", "/v1/messages") == 1  # the batch create only
    tracker = provider.tracker
    assert tracker.open_reservations() == []
    spent = sum(r.usd for r in results)
    assert spent > 0
    assert tracker.committed_usd() == pytest.approx(spent)
    assert tracker.totals("model") == [(s.HAIKU, pytest.approx(spent))]
    batches = json.loads((tmp_path / "llm_cache" / "batches.json").read_text("utf-8"))
    assert [e["status"] for e in batches.values()] == ["collected"]


def test_requests_carry_cache_control_schema_and_no_sampling(tmp_path: Path) -> None:
    fake = FakeAnthropic()
    s.provider(tmp_path, fake).score(s.requests(3), "batch")
    assert len(fake.bodies) == 3
    for body in fake.bodies:
        assert not {"temperature", "top_p", "top_k"} & body.keys()
        assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert body["output_config"]["format"]["type"] == "json_schema"
        assert body["output_config"]["effort"] == "low"
        assert body["thinking"] == {"type": "disabled"}
        assert body["max_tokens"] == 512


def test_refusal_is_retried_once_synchronously(tmp_path: Path) -> None:
    fake = FakeAnthropic(refuse_first={"REFUSE"})
    provider = s.provider(tmp_path, fake)
    texts = ["Revenue rose.", "REFUSE this one.", "Margins fell."]
    results = provider.score(s.requests(3, texts=texts), "batch")
    refused = results[1]
    assert refused.attempts == 2 and refused.schema_valid
    assert [r.attempts for r in results] == [1, 2, 1]
    cached = provider.cache.get(refused.cache_key, s.HAIKU)
    assert cached is not None
    assert cached.extra["first_attempt"]["stop_reason"] == "refusal"
    assert fake.count("POST", "/v1/messages") == 2  # batch create + one sync retry


def test_sync_mode_records_latency(tmp_path: Path) -> None:
    fake = FakeAnthropic()
    results = s.provider(tmp_path, fake).score(s.requests(4), "sync")
    assert all(r.latency_ms is not None and r.latency_ms >= 0 for r in results)
    assert all(r.batch_id is None and r.mode == "sync" for r in results)
    assert fake.count("POST", "/v1/messages") == 4


def test_unsupported_setting_refused_before_sending(tmp_path: Path) -> None:
    caps: dict[str, Any] = json.loads(json.dumps(FULL_CAPABILITIES))
    caps["thinking"]["types"]["disabled"]["supported"] = False
    fake = FakeAnthropic(capabilities=caps)
    provider = s.provider(tmp_path, fake)
    with pytest.raises(UnsupportedRequestError, match="thinking type disabled"):
        provider.score(s.requests(5), "batch")
    assert fake.count("POST", "/v1/messages") == 0
    assert provider.tracker.open_reservations() == []
    assert provider.tracker.committed_usd() == 0.0


def test_no_structured_outputs_refused(tmp_path: Path) -> None:
    caps: dict[str, Any] = json.loads(json.dumps(FULL_CAPABILITIES))
    caps["structured_outputs"]["supported"] = False
    caps["effort"]["low"]["supported"] = False
    fake = FakeAnthropic(capabilities=caps)
    with pytest.raises(UnsupportedRequestError, match=r"structured outputs.*effort low"):
        s.provider(tmp_path, fake).score(s.requests(1), "sync")


def test_budget_cap_blocks_before_any_network_call(tmp_path: Path) -> None:
    """AC-W-06: a 100-document run against a USD 0.01 budget never reaches the API."""
    fake = FakeAnthropic()
    provider = s.provider(tmp_path, fake, budget=0.01)
    with pytest.raises(BudgetExceededError):
        provider.score(s.requests(100), "batch")
    assert fake.requests == []
    assert provider.tracker.n_entries() == 0


def test_estimate_is_offline_and_bounds_the_actual_cost(tmp_path: Path) -> None:
    fake = FakeAnthropic()
    provider = s.provider(tmp_path, fake)
    batch = s.requests(20)
    estimate = provider.estimate_cost(batch, "batch")
    assert fake.requests == []
    assert estimate.n_requests == 20 and estimate.n_cached == 0
    results = provider.score(batch, "batch")
    assert sum(r.usd for r in results) <= estimate.worst_case_usd
    assert provider.estimate_cost(batch, "batch").n_cached == 20


class Crash(RuntimeError):
    pass


def test_crash_while_waiting_leaves_batch_open_then_resume_collects(tmp_path: Path) -> None:
    fake = FakeAnthropic(polls_before_end=3)

    def crash(_seconds: float) -> None:
        raise Crash

    batch = s.requests(6)
    provider = s.provider(tmp_path, fake, sleep=crash)
    with pytest.raises(Crash):
        provider.score(batch, "batch")
    assert len(provider.tracker.open_reservations()) == 1
    resumed = s.provider(tmp_path, fake)
    results = resumed.resume(batch)
    assert len(results) == 6 and all(r.schema_valid for r in results)
    assert resumed.tracker.open_reservations() == []
    assert resumed.tracker.committed_usd() == pytest.approx(sum(r.usd for r in results))
    assert fake.count("POST", "/v1/messages/batches") == 1  # never resubmitted
    again = s.provider(tmp_path, fake).score(batch, "batch")
    assert {r.mode for r in again} == {"cache_hit"}


def test_multi_batch_reservation_settles_once(tmp_path: Path) -> None:
    fake = FakeAnthropic()
    provider = s.provider(tmp_path, fake, batch_chunk=4)
    results = provider.score(s.requests(10), "batch")
    assert fake.count("POST", "/v1/messages/batches") == 3
    assert len({r.batch_id for r in results}) == 3
    assert provider.tracker.committed_usd() == pytest.approx(sum(r.usd for r in results))


def test_needs_a_key_and_a_tracker(tmp_path: Path) -> None:
    root = tmp_path / "llm_cache"
    with pytest.raises(ProviderConfigError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider(
            s.settings(tmp_path, key=""), ResponseCache(root), s.tracker(root), s.registry()
        )
    with pytest.raises(ProviderConfigError, match="SpendTracker"):
        AnthropicProvider(
            s.settings(tmp_path),
            ResponseCache(root),
            object(),  # type: ignore[arg-type]
            s.registry(),
        )


def test_one_model_per_call_and_registered_models_only(tmp_path: Path) -> None:
    fake = FakeAnthropic()
    provider = s.provider(tmp_path, fake)
    mixed = s.requests(1) + s.requests(1, model="claude-haiku-4-5", start=5)
    with pytest.raises(ProviderConfigError, match="one model"):
        provider.score(mixed, "batch")
    with pytest.raises(ProviderConfigError, match="not an Anthropic model"):
        provider.check_model("fake-v1", {})


def test_identical_texts_are_sent_once(tmp_path: Path) -> None:
    fake = FakeAnthropic()
    same = s.requests(3, texts=["Same release text."] * 3)
    results = s.provider(tmp_path, fake).score(same, "batch")
    assert [r.doc_id for r in results] == ["doc00000", "doc00001", "doc00002"]
    assert len(fake.bodies) == 1
    assert len({r.cache_key for r in results}) == 1


def test_count_tokens_uses_the_free_endpoint(tmp_path: Path) -> None:
    fake = FakeAnthropic()
    provider = s.provider(tmp_path, fake)
    assert provider.count_tokens(s.requests(1)[0]) == 1234
    assert fake.count("POST", "/v1/messages/count_tokens") == 1
    assert provider.tracker.n_entries() == 0
