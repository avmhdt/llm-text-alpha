"""Cache keys depend on what is sent, never on dict order (FR-11)."""

from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from llm_text_alpha.scoring.llm.base import ScoreRequest
from llm_text_alpha.scoring.llm.cache import CachedResponse, ResponseCache

keys = st.text(alphabet="abcdefgh", min_size=1, max_size=6)
values = st.one_of(st.integers(-5, 5), st.text(max_size=5), st.booleans(), st.none())
params = st.dictionaries(keys, values, max_size=5)


def request(**changes: object) -> ScoreRequest:
    base = ScoreRequest(
        doc_id="d1",
        model_id="claude-haiku-5-5",
        system_prompt="You are an analyst.",
        user_text="Revenue rose.",
        output_schema={"type": "object", "properties": {"score": {"type": "number"}}},
        max_output_tokens=512,
        request_params={"thinking": {"type": "disabled"}},
    )
    return dataclasses.replace(base, **changes)  # type: ignore[arg-type]


@given(extra=params)
def test_key_ignores_dict_order(extra: dict[str, object]) -> None:
    forward = request(request_params=dict(extra))
    backward = request(request_params=dict(reversed(list(extra.items()))))
    assert forward.cache_key() == backward.cache_key()


@given(text=st.text(min_size=1, max_size=40))
def test_key_changes_with_what_is_sent(text: str) -> None:
    base = request()
    variants = [
        request(model_id="claude-haiku-4-5"),
        request(system_prompt=base.system_prompt + text),
        request(user_text=base.user_text + text),
        request(output_schema={"type": "object", "x": text}),
        request(max_output_tokens=513),
        request(request_params={"thinking": {"type": "disabled"}, "k": text}),
    ]
    assert len({v.cache_key() for v in variants} | {base.cache_key()}) == len(variants) + 1


def test_key_ignores_doc_id() -> None:
    """Identical text scored for two filings is one request, paid once."""
    assert request(doc_id="a").cache_key() == request(doc_id="b").cache_key()


@settings(max_examples=25, deadline=None)
@given(raw=st.text(max_size=200), usd=st.floats(0, 1, allow_nan=False))
def test_cache_round_trip(raw: str, usd: float) -> None:
    with tempfile.TemporaryDirectory() as folder:
        cache = ResponseCache(Path(folder))
        key = request().cache_key()
        response = CachedResponse(
            cache_key=key,
            doc_id="d1",
            model_id="claude-haiku-5-5",
            provider="anthropic",
            raw_text=raw,
            stop_reason="end_turn",
            usage={"input_tokens": 3},
            usd=usd,
            mode="batch",
            created_at="2026-11-04T00:00:00+00:00",
            schema_valid=False,
        )
        cache.put(key, response)
        assert cache.get(key, "claude-haiku-5-5") == response
        assert cache.get(key, "claude-haiku-4-5") is None
