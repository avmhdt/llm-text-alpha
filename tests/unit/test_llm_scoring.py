from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import polars as pl
import pytest

from llm_text_alpha.scoring.llm import run as llm_run
from llm_text_alpha.scoring.llm.base import ScoreResult, TokenUsage
from llm_text_alpha.scoring.llm.cache import ResponseCache
from llm_text_alpha.scoring.llm.fake_provider import FakeProvider
from llm_text_alpha.scoring.llm.local_provider import LocalProvider
from llm_text_alpha.scoring.llm.prompts import load_prompt, parse_payload, truncate
from llm_text_alpha.scoring.llm.schema import LLM_SCORE_SCHEMA, LLMScore
from tests import llm_support as s
from tests.fake_anthropic import answer

VALID = {
    "score": 0.4,
    "label": "positive",
    "surprise": 1,
    "guidance": None,
    "materiality": 0.6,
    "confidence": 0.8,
}


def test_schema_accepts_valid_and_rejects_the_rest() -> None:
    assert LLMScore.model_validate(VALID).score == 0.4
    for bad in (
        VALID | {"score": 1.5},
        VALID | {"label": "bullish"},
        VALID | {"surprise": 2},
        VALID | {"extra": 1},
        {k: v for k, v in VALID.items() if k != "confidence"},
    ):
        assert parse_payload(json.dumps(bad)) == (None, False)
    assert parse_payload("not json") == (None, False)
    assert LLM_SCORE_SCHEMA["additionalProperties"] is False
    assert set(LLM_SCORE_SCHEMA["required"]) == set(VALID)


def test_truncate_keeps_the_start_and_cuts_at_a_line() -> None:
    text = "first line\n" * 50
    body, cut = truncate(text, 100)
    assert cut and len(body) <= 100 and text.startswith(body) and body.endswith("line")
    assert truncate("short", 100) == ("short", False)


def test_prompt_version_tracks_its_text() -> None:
    prompt = load_prompt(s.REPO / "configs" / "prompts" / "score_v1.md")
    assert prompt.version == f"score_v1@{prompt.sha8}" and len(prompt.sha8) == 8
    assert "JSON" in prompt.text


def test_registry_cutoffs_and_post_cutoff_windows() -> None:
    registry = s.registry()
    assert {"claude-haiku-5-5", "claude-haiku-4-5", "fake-v1"} <= set(registry.ids())
    assert registry.get("claude-haiku-4-5").post_cutoff_start == date(2025, 8, 1)
    assert registry.get("claude-haiku-5-5").request["thinking"] == {"type": "disabled"}
    with pytest.raises(KeyError):
        registry.get("gpt-4")


def corpus(n: int) -> pl.DataFrame:
    years = [2016 + i % 5 for i in range(n)]
    return pl.DataFrame(
        {
            "doc_id": [f"d{i:04d}" for i in range(n)],
            "event_ts_utc": [datetime(y, 3, 1, tzinfo=UTC) for y in years],
            "text": ["x"] * n,
        }
    )


def test_sample_is_exact_proportional_and_reproducible() -> None:
    docs = corpus(500)
    first = llm_run.draw_sample(docs, 101, "year", seed=7)
    assert first.height == 101
    assert first.equals(llm_run.draw_sample(docs.reverse(), 101, "year", seed=7))
    counts = first.group_by("year").len()["len"].to_list()
    assert max(counts) - min(counts) <= 1
    assert llm_run.draw_sample(docs, 1_000, "year", seed=7).height == 500
    assert not first.equals(llm_run.draw_sample(docs, 101, "year", seed=8))


def test_fake_provider_is_deterministic_and_cached(tmp_path: Path) -> None:
    batch = s.requests(5, model="fake-v1")
    first = FakeProvider(ResponseCache(tmp_path)).score(batch, "batch")
    second = FakeProvider(ResponseCache(tmp_path)).score(batch, "batch")
    assert [r.payload for r in first] == [r.payload for r in second]
    assert {r.mode for r in first} == {"local"} and {r.mode for r in second} == {"cache_hit"}
    assert FakeProvider(ResponseCache(tmp_path)).estimate_cost(batch).n_cached == 5


def local_server(bad_first: bool) -> tuple[httpx.MockTransport, list[dict[str, object]]]:
    seen: list[dict[str, object]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        text = (
            "{not json" if bad_first and len(seen) == 1 else answer(body["messages"][1]["content"])
        )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 800, "completion_tokens": 50},
            },
        )

    return httpx.MockTransport(handle), seen


def test_local_provider_greedy_schema_and_retry(tmp_path: Path) -> None:
    transport, seen = local_server(bad_first=True)
    model = "llama-3.1-8b-instruct-q4_k_m"
    provider = LocalProvider(
        "http://127.0.0.1:8080/v1/", model, ResponseCache(tmp_path), s.registry(), transport
    )
    batch = s.requests(2, model=model)
    results = provider.score(batch)
    assert [r.attempts for r in results] == [2, 1]
    assert all(r.schema_valid and r.usd == 0.0 and r.mode == "local" for r in results)
    assert seen[0]["temperature"] == 0
    assert seen[0]["response_format"]["type"] == "json_schema"  # type: ignore[index]
    assert results[0].usage == TokenUsage(800, 50)
    again = provider.score(batch)
    assert len(seen) == 3 and {r.mode for r in again} == {"cache_hit"}
    assert provider.estimate_cost(batch).worst_case_usd == 0.0


def test_tables_split_valid_scores_from_every_call() -> None:
    good = LLMScore.model_validate(VALID)
    results = [
        ScoreResult(
            "d1",
            "m",
            good,
            "end_turn",
            TokenUsage(10, 5),
            0.001,
            None,
            False,
            "k1",
            "batch",
            "b1",
            60.0,
            True,
            1,
        ),
        ScoreResult(
            "d2",
            "m",
            None,
            "refusal",
            TokenUsage(10, 0),
            0.0005,
            None,
            False,
            "k2",
            "batch",
            "b1",
            60.0,
            False,
            2,
        ),
    ]
    run = llm_run.tables(results, "anthropic", "score_v1@abcd1234", False, "r1")
    assert run.scores["doc_id"].to_list() == ["d1"]
    assert run.scores["cost_usd"].to_list() == [0.001]
    aux = json.loads(run.scores["aux_json"][0])
    assert aux["surprise"] == 1 and aux["stop_reason"] == "end_turn"
    assert run.calls.height == 2
    assert run.calls["stop_reason"].to_list() == ["end_turn", "refusal"]
    assert run.calls["batch_turnaround_s"].to_list() == [60.0, 60.0]


def test_make_provider_names(tmp_path: Path) -> None:
    settings = s.settings(tmp_path)
    cache = ResponseCache(tmp_path)
    prices = s.REPO / "configs" / "llm_prices.yaml"
    fake = llm_run.make_provider("fake", settings, s.registry(), cache, "r", prices)
    assert fake.name == "fake"
    local = llm_run.make_provider("local", settings, s.registry(), cache, "r", prices)
    assert local.name == "local"
    with pytest.raises(ValueError, match="unknown provider"):
        llm_run.make_provider("openai", settings, s.registry(), cache, "r", prices)
