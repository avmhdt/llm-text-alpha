"""Claude through the Anthropic SDK: the only module that constructs an ``anthropic.Anthropic``.

Order on every call (FR-10 to FR-12, NFR-06, NFR-09):

1. Cached requests are answered from disk: no network call and no ledger entry.
2. The worst-case cost of the rest is reserved with the spend tracker. A reservation over
   the budget raises before anything is sent.
3. The model's capabilities are checked with the Models API against the registry's request
   settings and structured outputs.
4. Requests go out as Message Batches (or one by one in ``sync`` mode), with the system
   prompt marked for caching and the output constrained to the JSON schema. No sampling
   parameters are sent (D-06).
5. A refused, truncated or invalid answer is retried once, synchronously.
6. The reservation is settled with the actual cost from each response's ``usage``.

Open batches are saved to ``batches.json`` with their reservation, so after a crash they are
polled and collected, never resubmitted.
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import anthropic
import httpx2
import structlog

from llm_text_alpha.io import write_json_atomic
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
from llm_text_alpha.scoring.llm.spend import Reservation, SpendTracker
from llm_text_alpha.settings import Settings

log = structlog.get_logger(__name__)

WORST_CASE_CHARS_PER_TOKEN = 2.0
"""Characters per token for the worst-case estimate. FR-12's chars/3 undercounts on Haiku
5.5's tokenizer: 200 Item 2.02 releases counted with the token-counting endpoint gave a
median of 2.76 characters per token and a 1st percentile of 2.22 (docs/decisions.md), so
2.0 keeps the reservation an upper bound."""
SCHEMA_OVERHEAD_TOKENS = 600
"""Allowance for the structured-output instructions the API adds to each prompt."""
BATCH_CHUNK = 5_000


class ProviderConfigError(RuntimeError):
    """The provider cannot run: no key, no tracker, or an unregistered model."""


class UnsupportedRequestError(RuntimeError):
    """The Models API says the model does not accept a requested setting."""


@dataclass
class _Outcome:
    """One request's final response and every usage it cost, with batch flags."""

    response: CachedResponse
    costs: list[tuple[TokenUsage, bool]] = field(default_factory=list)


def _usage(raw: Any) -> TokenUsage:
    return TokenUsage(
        input_tokens=int(getattr(raw, "input_tokens", 0) or 0),
        output_tokens=int(getattr(raw, "output_tokens", 0) or 0),
        cache_write_tokens=int(getattr(raw, "cache_creation_input_tokens", 0) or 0),
        cache_read_tokens=int(getattr(raw, "cache_read_input_tokens", 0) or 0),
    )


def _text(message: Any) -> str:
    return "".join(block.text for block in message.content if block.type == "text")


class AnthropicProvider:
    """``LLMProvider`` over the Claude API; it needs a :class:`SpendTracker` to exist."""

    name = "anthropic"

    def __init__(
        self,
        settings: Settings,
        cache: ResponseCache,
        tracker: SpendTracker,
        registry: ModelRegistry,
        *,
        run_id: str = "",
        batches_file: Path | None = None,
        http_client: httpx2.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        poll_seconds: float = 60.0,
        batch_chunk: int = BATCH_CHUNK,
    ) -> None:
        if not isinstance(tracker, SpendTracker):
            raise ProviderConfigError("AnthropicProvider needs a SpendTracker")
        key = settings.anthropic_api_key.get_secret_value()
        if not key:
            raise ProviderConfigError("ANTHROPIC_API_KEY is not set")
        self.cache = cache
        self.tracker = tracker
        self.registry = registry
        self.run_id = run_id
        self.batches_file = batches_file or cache.root / "batches.json"
        self._sleep = sleep
        self._poll = poll_seconds
        self._chunk = batch_chunk
        self._checked: set[str] = set()
        self._client = anthropic.Anthropic(api_key=key, http_client=http_client, max_retries=4)

    # -- estimates ---------------------------------------------------------------------

    def _input_tokens(self, request: ScoreRequest) -> int:
        chars = len(request.system_prompt) + len(request.user_text)
        return int(chars / WORST_CASE_CHARS_PER_TOKEN) + SCHEMA_OVERHEAD_TOKENS

    def estimate_cost(self, requests: Sequence[ScoreRequest], mode: Mode = "batch") -> CostEstimate:
        """Worst case for uncached requests, and for the one retry each may need. Offline."""
        todo = [r for r in requests if self.cache.get(r.cache_key(), r.model_id) is None]
        tokens = sum(self._input_tokens(r) for r in todo)
        out = sum(r.max_output_tokens for r in todo)
        usd = sum(
            self.tracker.prices.worst_case(
                r.model_id, self._input_tokens(r), r.max_output_tokens, batch=mode == "batch"
            )
            for r in todo
        )
        return CostEstimate(len(requests), len(requests) - len(todo), tokens, out, usd)

    def _retry_allowance(self, requests: Sequence[ScoreRequest]) -> float:
        """Worst case of retrying 2% of requests synchronously (at least one)."""
        if not requests:
            return 0.0
        worst = max(
            self.tracker.prices.worst_case(
                r.model_id, self._input_tokens(r), r.max_output_tokens, False
            )
            for r in requests
        )
        return worst * max(1, int(len(requests) * 0.02) + 1)

    def count_tokens(self, request: ScoreRequest) -> int:
        """Prompt tokens for ``request`` from the free token-counting endpoint (no spend)."""
        params = self.params(request)
        params.pop("max_tokens")
        return int(self._client.messages.count_tokens(**params).input_tokens)

    # -- request shape -----------------------------------------------------------------

    def check_model(self, model_id: str, params: dict[str, Any]) -> None:
        """Refuse, before spending, any setting the Models API says the model rejects."""
        if model_id in self._checked:
            return
        if self.registry.get(model_id).provider != self.name:
            raise ProviderConfigError(f"{model_id} is not an Anthropic model in the registry")
        raw_caps: Any = self._client.models.retrieve(model_id).capabilities
        caps: dict[str, Any] = (
            raw_caps.model_dump() if hasattr(raw_caps, "model_dump") else dict(raw_caps or {})
        )
        problems: list[str] = []
        if not caps.get("structured_outputs", {}).get("supported", False):
            problems.append("structured outputs")
        thinking = params.get("thinking", {}).get("type")
        if thinking and not caps.get("thinking", {}).get("types", {}).get(thinking, {}).get(
            "supported", False
        ):
            problems.append(f"thinking type {thinking}")
        effort = params.get("output_config", {}).get("effort")
        if effort and not caps.get("effort", {}).get(effort, {}).get("supported", False):
            problems.append(f"effort {effort}")
        if problems:
            raise UnsupportedRequestError(f"{model_id} does not support: {', '.join(problems)}")
        self._checked.add(model_id)

    @staticmethod
    def params(request: ScoreRequest) -> dict[str, Any]:
        """Messages API parameters for one request."""
        extra = dict(request.request_params)
        output_config = dict(extra.pop("output_config", {}))
        output_config["format"] = {"type": "json_schema", "schema": dict(request.output_schema)}
        return {
            "model": request.model_id,
            "max_tokens": request.max_output_tokens,
            "system": [
                {
                    "type": "text",
                    "text": request.system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": request.user_text}],
            "output_config": output_config,
            **extra,
        }

    # -- responses ---------------------------------------------------------------------

    def _response(
        self,
        request: ScoreRequest,
        message: Any,
        mode: str,
        latency_ms: float | None = None,
        batch_id: str | None = None,
        turnaround: float | None = None,
        attempts: int = 1,
    ) -> CachedResponse:
        raw = _text(message)
        payload, valid = parse_payload(raw) if message.stop_reason == "end_turn" else (None, False)
        usage = _usage(message.usage)
        return CachedResponse(
            cache_key=request.cache_key(),
            doc_id=request.doc_id,
            model_id=request.model_id,
            provider=self.name,
            raw_text=raw,
            stop_reason=str(message.stop_reason),
            usage=asdict(usage),
            usd=self.tracker.prices.cost(request.model_id, usage, batch=mode == "batch"),
            mode=mode,
            created_at=datetime.now(UTC).isoformat(),
            schema_valid=valid,
            payload=payload.model_dump() if payload else None,
            latency_ms=latency_ms,
            batch_id=batch_id,
            batch_turnaround_s=turnaround,
            attempts=attempts,
        )

    def _sync_once(self, request: ScoreRequest, attempts: int = 1) -> _Outcome:
        started = time.perf_counter()
        message = self._client.messages.create(**self.params(request))
        latency = (time.perf_counter() - started) * 1000.0
        response = self._response(request, message, "sync", latency_ms=latency, attempts=attempts)
        return _Outcome(response, [(_usage(message.usage), False)])

    def _with_retry(self, request: ScoreRequest, first: _Outcome) -> _Outcome:
        if first.response.schema_valid:
            return first
        log.warning("llm_retry", doc_id=request.doc_id, stop_reason=first.response.stop_reason)
        second = self._sync_once(request, attempts=2)
        second.costs = first.costs + second.costs
        # A document's cost includes its first attempt.
        second.response = dataclasses.replace(
            second.response, usd=second.response.usd + first.response.usd
        )
        second.response.extra["first_attempt"] = {
            "stop_reason": first.response.stop_reason,
            "raw_text": first.response.raw_text,
            "mode": first.response.mode,
        }
        return second

    def _sync(self, requests: Sequence[ScoreRequest], done: list[_Outcome]) -> None:
        """Score one by one, caching each answer as it arrives so a crash loses none."""
        for request in requests:
            outcome = self._with_retry(request, self._sync_once(request))
            self._store([outcome])
            done.append(outcome)

    # -- batches -----------------------------------------------------------------------

    def _load_batches(self) -> dict[str, Any]:
        if self.batches_file.exists():
            loaded: dict[str, Any] = json.loads(self.batches_file.read_text("utf-8"))
            return loaded
        return {}

    def _save_batch(self, batch_id: str, entry: dict[str, Any]) -> None:
        batches = self._load_batches()
        batches[batch_id] = entry
        write_json_atomic(batches, self.batches_file)

    def _submit(self, requests: Sequence[ScoreRequest], reservation: Reservation) -> list[str]:
        ids: list[str] = []
        for start in range(0, len(requests), self._chunk):
            chunk = requests[start : start + self._chunk]
            batch = self._client.messages.batches.create(
                requests=cast(
                    Any, [{"custom_id": r.cache_key(), "params": self.params(r)} for r in chunk]
                )
            )
            self._save_batch(
                batch.id,
                {
                    "status": "submitted",
                    "created_at": batch.created_at.isoformat(),
                    "run_id": self.run_id,
                    "reservation": asdict(reservation),
                    "custom_ids": [r.cache_key() for r in chunk],
                    "doc_ids": [r.doc_id for r in chunk],
                },
            )
            log.info("llm_batch_submitted", batch_id=batch.id, requests=len(chunk))
            ids.append(batch.id)
        return ids

    def _wait(self, batch_id: str) -> Any:
        while True:
            batch = self._client.messages.batches.retrieve(batch_id)
            if batch.processing_status == "ended":
                return batch
            log.info(
                "llm_batch_status",
                batch_id=batch_id,
                status=batch.processing_status,
                processing=batch.request_counts.processing,
                succeeded=batch.request_counts.succeeded,
            )
            self._sleep(self._poll)

    def _collect(self, batch_id: str, by_key: dict[str, ScoreRequest]) -> list[_Outcome]:
        batch = self._wait(batch_id)
        turnaround = (batch.ended_at - batch.created_at).total_seconds() if batch.ended_at else None
        outcomes: list[_Outcome] = []
        for item in self._client.messages.batches.results(batch_id):
            request = by_key.get(item.custom_id)
            if request is None:
                continue
            if item.result.type == "succeeded":
                message = item.result.message
                first = _Outcome(
                    self._response(
                        request, message, "batch", batch_id=batch_id, turnaround=turnaround
                    ),
                    [(_usage(message.usage), True)],
                )
            else:
                failed = CachedResponse(
                    cache_key=item.custom_id,
                    doc_id=request.doc_id,
                    model_id=request.model_id,
                    provider=self.name,
                    raw_text="",
                    stop_reason=f"batch_{item.result.type}",
                    usage={},
                    usd=0.0,
                    mode="batch",
                    created_at=datetime.now(UTC).isoformat(),
                    schema_valid=False,
                    batch_id=batch_id,
                    batch_turnaround_s=turnaround,
                )
                first = _Outcome(failed, [])
            outcomes.append(self._with_retry(request, first))
        self._store(outcomes)
        entry = self._load_batches().get(batch_id, {})
        entry.update(
            {
                "status": "collected",
                "ended_at": batch.ended_at.isoformat() if batch.ended_at else None,
                "usd": self._cost(outcomes),
            }
        )
        self._save_batch(batch_id, entry)
        return outcomes

    def resume(self, requests: Sequence[ScoreRequest]) -> list[ScoreResult]:
        """Poll and collect every submitted batch, then settle each reservation (NFR-09).

        A reservation can span several batches; it is settled once all of them are
        collected, with the cost each batch recorded in ``batches.json`` when collected.
        """
        by_key = {r.cache_key(): r for r in requests}
        results: list[ScoreResult] = []
        for batch_id, entry in self._load_batches().items():
            if entry.get("status") != "submitted":
                continue
            outcomes = self._collect(batch_id, by_key)
            results += [
                self._result(o.response, by_key[o.response.cache_key], False) for o in outcomes
            ]
        batches = self._load_batches()
        open_ids = {r.reservation_id for r in self.tracker.open_reservations()}
        for reservation_id in open_ids:
            entries = [
                e for e in batches.values() if e["reservation"]["reservation_id"] == reservation_id
            ]
            if entries and all(e.get("status") == "collected" for e in entries):
                reservation = Reservation(**entries[0]["reservation"])
                self.tracker.settle_usd(reservation, sum(e.get("usd", 0.0) for e in entries))
        return results

    # -- the protocol ------------------------------------------------------------------

    def _store(self, outcomes: Sequence[_Outcome]) -> None:
        for outcome in outcomes:
            self.cache.put(outcome.response.cache_key, outcome.response)

    def _cost(self, outcomes: Sequence[_Outcome]) -> float:
        return sum(
            self.tracker.prices.cost(o.response.model_id, u, b)
            for o in outcomes
            for u, b in o.costs
        )

    @staticmethod
    def _result(response: CachedResponse, request: ScoreRequest, hit: bool) -> ScoreResult:
        payload, valid = (
            parse_payload(response.raw_text) if response.schema_valid else (None, False)
        )
        return ScoreResult(
            doc_id=request.doc_id,
            model_id=request.model_id,
            payload=payload,
            stop_reason=response.stop_reason,
            usage=TokenUsage(**response.usage),
            usd=0.0 if hit else response.usd,
            latency_ms=response.latency_ms,
            from_cache=hit,
            cache_key=response.cache_key,
            mode="cache_hit" if hit else response.mode,
            batch_id=response.batch_id,
            batch_turnaround_s=response.batch_turnaround_s,
            schema_valid=valid,
            attempts=response.attempts,
            cost_usd=response.usd,
        )

    def score(self, requests: Sequence[ScoreRequest], mode: Mode) -> list[ScoreResult]:
        """Score ``requests``; uncached ones are paid for under one reservation."""
        cached = {r.cache_key(): self.cache.get(r.cache_key(), r.model_id) for r in requests}
        # Identical text in two filings is one request (and one custom_id in a batch).
        todo = list({r.cache_key(): r for r in requests if cached[r.cache_key()] is None}.values())
        fresh: dict[str, CachedResponse] = {}
        if todo:
            models = {r.model_id for r in todo}
            if len(models) != 1:
                raise ProviderConfigError("one call scores one model")
            model_id = models.pop()
            hold = self.estimate_cost(todo, mode).worst_case_usd + self._retry_allowance(todo)
            reservation = self.tracker.reserve(self.run_id, model_id, hold)
            outcomes: list[_Outcome] = []
            submitted = False
            try:
                self.check_model(model_id, self.params(todo[0]))
                if mode == "batch":
                    by_key = {r.cache_key(): r for r in todo}
                    batch_ids = self._submit(todo, reservation)
                    submitted = True
                    for batch_id in batch_ids:
                        outcomes += self._collect(batch_id, by_key)
                else:
                    self._sync(todo, outcomes)
            finally:
                spent = self._cost(outcomes)
                if mode == "sync" or len(outcomes) == len(todo):
                    self.tracker.settle_usd(reservation, spent)
                elif not submitted:
                    self.tracker.release(reservation)
            fresh = {o.response.cache_key: o.response for o in outcomes}
            log.info(
                "llm_scored", model=model_id, mode=mode, requests=len(todo), usd=round(spent, 6)
            )
        results: list[ScoreResult] = []
        for request in requests:
            key = request.cache_key()
            hit = cached[key] is not None
            response = cached[key] if hit else fresh.get(key)
            if response is None:
                raise RuntimeError(f"no response for {request.doc_id}")
            results.append(self._result(response, request, hit))
        return results
