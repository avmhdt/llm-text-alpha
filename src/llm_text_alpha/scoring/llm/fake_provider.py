"""Deterministic offline provider (``LLM_PROVIDER=fake``): CI and the fixture pipeline use it."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime

from llm_text_alpha.scoring.llm.base import (
    CostEstimate,
    Mode,
    ScoreRequest,
    ScoreResult,
    TokenUsage,
)
from llm_text_alpha.scoring.llm.cache import CachedResponse, ResponseCache
from llm_text_alpha.scoring.llm.prompts import parse_payload


class FakeProvider:
    """Scores derived from a hash of the request: same input, same output, no network."""

    name = "fake"

    def __init__(self, cache: ResponseCache, seed: int = 0) -> None:
        self.cache = cache
        self.seed = seed

    def estimate_cost(self, requests: Sequence[ScoreRequest], mode: Mode = "batch") -> CostEstimate:
        cached = sum(self.cache.get(r.cache_key(), r.model_id) is not None for r in requests)
        return CostEstimate(len(requests), cached, 0, 0, 0.0)

    def _respond(self, request: ScoreRequest) -> str:
        digest = hashlib.sha256(f"{self.seed}|{request.user_text}".encode()).digest()
        if request.answer == "reid":
            return json.dumps(
                {
                    "company": "unknown",
                    "ticker": None,
                    "period_year": None,
                    "period_quarter": None,
                    "confidence": 0.0,
                }
            )
        if request.answer == "recall":
            return json.dumps({"direction": "up" if digest[0] % 2 else "down", "confidence": 0.5})
        score = round(digest[0] / 127.5 - 1.0, 4)
        label = "positive" if score > 0.2 else "negative" if score < -0.2 else "neutral"
        return json.dumps(
            {
                "score": score,
                "label": label,
                "surprise": digest[1] % 3 - 1,
                "guidance": None if digest[2] % 4 == 0 else digest[2] % 3 - 1,
                "materiality": round(digest[3] / 255, 4),
                "confidence": round(digest[4] / 255, 4),
            }
        )

    def score(self, requests: Sequence[ScoreRequest], mode: Mode) -> list[ScoreResult]:
        results: list[ScoreResult] = []
        for request in requests:
            key = request.cache_key()
            cached = self.cache.get(key, request.model_id)
            hit = cached is not None
            if cached is None:
                raw = self._respond(request)
                payload, valid = parse_payload(raw, request.answer)
                cached = CachedResponse(
                    cache_key=key,
                    doc_id=request.doc_id,
                    model_id=request.model_id,
                    provider=self.name,
                    raw_text=raw,
                    stop_reason="end_turn",
                    usage={},
                    usd=0.0,
                    mode="local",
                    created_at=datetime.now(UTC).isoformat(),
                    schema_valid=valid,
                    payload=payload.model_dump() if payload else None,
                    latency_ms=0.0,
                )
                self.cache.put(key, cached)
            payload, valid = parse_payload(cached.raw_text, request.answer)
            results.append(
                ScoreResult(
                    doc_id=request.doc_id,
                    model_id=request.model_id,
                    payload=payload,
                    stop_reason=cached.stop_reason,
                    usage=TokenUsage(),
                    usd=0.0,
                    latency_ms=0.0 if not hit else cached.latency_ms,
                    from_cache=hit,
                    cache_key=key,
                    mode="cache_hit" if hit else "local",
                    schema_valid=valid,
                )
            )
        return results
