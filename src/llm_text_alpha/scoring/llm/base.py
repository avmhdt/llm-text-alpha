"""Provider-neutral request and result types, and the ``LLMProvider`` protocol (section 10)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Mode = Literal["batch", "sync"]


def canonical(value: Any) -> str:
    """Canonical JSON: sorted keys, no spaces, ASCII."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True)
class TokenUsage:
    """Token counts from a response's ``usage``."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0


@dataclass(frozen=True)
class ScoreRequest:
    """One document to score. ``request_params`` are the non-default request fields.

    The cache key depends on the model, the system prompt, the user text, the output
    schema and the request parameters, and on nothing else: identical inputs always map to
    the same cached response (FR-11).
    """

    doc_id: str
    model_id: str
    system_prompt: str
    user_text: str
    output_schema: Mapping[str, Any]
    max_output_tokens: int
    request_params: Mapping[str, Any] = field(default_factory=dict)
    answer: str = "score"
    """Which pydantic model validates the reply (``schema.ANSWERS``); not part of the key,
    because the schema already is."""

    def prompt_hash(self) -> str:
        """sha256 of the canonical JSON of prompt, text, schema and parameters."""
        payload = {
            "system": self.system_prompt,
            "user": self.user_text,
            "schema": self.output_schema,
            "max_output_tokens": self.max_output_tokens,
            "params": self.request_params,
        }
        return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()

    def cache_key(self) -> str:
        """``sha256(model_id + "\\n" + prompt_hash)``."""
        return hashlib.sha256(f"{self.model_id}\n{self.prompt_hash()}".encode()).hexdigest()


@dataclass(frozen=True)
class ScoreResult:
    """One scored document. ``payload`` is None for refusals and invalid output."""

    doc_id: str
    model_id: str
    payload: Any
    stop_reason: str
    usage: TokenUsage
    usd: float
    latency_ms: float | None
    from_cache: bool
    cache_key: str = ""
    mode: str = "sync"
    batch_id: str | None = None
    batch_turnaround_s: float | None = None
    schema_valid: bool = False
    attempts: int = 1
    cost_usd: float | None = None
    """What producing this answer cost when it was first scored (``usd`` is this run's spend)."""


@dataclass(frozen=True)
class CostEstimate:
    """Worst-case spend for the requests not already cached (FR-12)."""

    n_requests: int
    n_cached: int
    input_tokens: int
    max_output_tokens: int
    worst_case_usd: float


class LLMProvider(Protocol):
    """A source of LLM scores. Every call goes through the response cache."""

    name: str

    def estimate_cost(self, requests: Sequence[ScoreRequest], mode: Mode = "batch") -> CostEstimate:
        """Worst-case cost of the uncached requests; never spends or calls the network."""
        ...

    def score(self, requests: Sequence[ScoreRequest], mode: Mode) -> list[ScoreResult]:
        """Score every request, in the order given."""
        ...
