"""Local open-weight model over an OpenAI-compatible HTTP endpoint (llama.cpp or vLLM).

Greedy decoding (``temperature=0``) and a JSON-schema response format where the server
supports it. Calls go through the response cache; local inference costs nothing in API
terms, so there is no spend tracker. Plain ``httpx``, no OpenAI SDK (section 4).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime

import httpx

from llm_text_alpha.scoring.llm.base import (
    CostEstimate,
    Mode,
    ScoreRequest,
    ScoreResult,
    TokenUsage,
)
from llm_text_alpha.scoring.llm.cache import CachedResponse, ResponseCache
from llm_text_alpha.scoring.llm.prompts import parse_payload
from llm_text_alpha.scoring.llm.registry import ModelRegistry


class LocalProvider:
    """``LLMProvider`` for a model served at ``base_url`` (e.g. ``http://127.0.0.1:8080/v1``)."""

    name = "local"

    def __init__(
        self,
        base_url: str,
        model: str,
        cache: ResponseCache,
        registry: ModelRegistry,
        transport: httpx.BaseTransport | None = None,
        timeout_s: float = 600.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.cache = cache
        self.registry = registry
        self._http = httpx.Client(transport=transport, timeout=timeout_s)

    def estimate_cost(self, requests: Sequence[ScoreRequest], mode: Mode = "sync") -> CostEstimate:
        cached = sum(self.cache.get(r.cache_key(), r.model_id) is not None for r in requests)
        return CostEstimate(len(requests), cached, 0, 0, 0.0)

    def _body(self, request: ScoreRequest) -> dict[str, object]:
        params = dict(self.registry.get(request.model_id).request)
        params.update(request.request_params)
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_text},
            ],
            "max_tokens": request.max_output_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "score",
                    "schema": dict(request.output_schema),
                    "strict": True,
                },
            },
            **params,
        }

    def _call(self, request: ScoreRequest, attempts: int) -> CachedResponse:
        started = time.perf_counter()
        response = self._http.post(f"{self.base_url}/chat/completions", json=self._body(request))
        response.raise_for_status()
        latency = (time.perf_counter() - started) * 1000.0
        data = response.json()
        choice = data["choices"][0]
        raw = choice["message"].get("content") or ""
        finish = choice.get("finish_reason") or ""
        payload, valid = parse_payload(raw) if finish in ("stop", "") else (None, False)
        usage = data.get("usage", {})
        tokens = TokenUsage(
            int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
        )
        return CachedResponse(
            cache_key=request.cache_key(),
            doc_id=request.doc_id,
            model_id=request.model_id,
            provider=self.name,
            raw_text=raw,
            stop_reason=finish,
            usage=asdict(tokens),
            usd=0.0,
            mode="local",
            created_at=datetime.now(UTC).isoformat(),
            schema_valid=valid,
            payload=payload.model_dump() if payload else None,
            latency_ms=latency,
            attempts=attempts,
        )

    def score(self, requests: Sequence[ScoreRequest], mode: Mode = "sync") -> list[ScoreResult]:
        results: list[ScoreResult] = []
        for request in requests:
            key = request.cache_key()
            cached = self.cache.get(key, request.model_id)
            hit = cached is not None
            if cached is None:
                cached = self._call(request, 1)
                if not cached.schema_valid:
                    cached = self._call(request, 2)
                self.cache.put(key, cached)
            payload, valid = (
                parse_payload(cached.raw_text) if cached.schema_valid else (None, False)
            )
            results.append(
                ScoreResult(
                    doc_id=request.doc_id,
                    model_id=request.model_id,
                    payload=payload,
                    stop_reason=cached.stop_reason,
                    usage=TokenUsage(**cached.usage),
                    usd=0.0,
                    latency_ms=cached.latency_ms,
                    from_cache=hit,
                    cache_key=key,
                    mode="cache_hit" if hit else "local",
                    schema_valid=valid,
                    attempts=cached.attempts,
                )
            )
        return results
