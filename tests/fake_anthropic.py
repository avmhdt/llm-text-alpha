"""An in-process stand-in for the Claude API, served through an ``httpx2.MockTransport``.

It answers ``GET /v1/models/{id}``, ``POST /v1/messages``, and the Message Batches create,
retrieve and results (JSONL) endpoints, and records every request it sees, so tests can count
network calls without a key or a connection.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import httpx2

CREATED = "2026-11-04T00:00:00Z"
ENDED = "2026-11-04T00:30:00Z"

FULL_CAPABILITIES: dict[str, Any] = {
    "structured_outputs": {"supported": True},
    "thinking": {
        "supported": True,
        "types": {
            "adaptive": {"supported": True},
            "enabled": {"supported": True},
            "disabled": {"supported": True},
        },
    },
    "effort": {
        "supported": True,
        "low": {"supported": True},
        "medium": {"supported": True},
        "high": {"supported": True},
    },
}


def answer(user_text: str) -> str:
    """A valid score object derived from the text, so identical input gives identical output."""
    digest = hashlib.sha256(user_text.encode()).digest()
    score = round(digest[0] / 127.5 - 1.0, 4)
    label = "positive" if score > 0.2 else "negative" if score < -0.2 else "neutral"
    return json.dumps(
        {
            "score": score,
            "label": label,
            "surprise": digest[1] % 3 - 1,
            "guidance": None,
            "materiality": round(digest[2] / 255, 4),
            "confidence": round(digest[3] / 255, 4),
        }
    )


@dataclass
class FakeAnthropic:
    """Configurable fake: refusals on first sight of some texts, capabilities, batch polls."""

    capabilities: dict[str, Any] = field(default_factory=lambda: dict(FULL_CAPABILITIES))
    refuse_first: set[str] = field(default_factory=set)
    """Substrings: a message whose user text contains one is refused the first time."""
    polls_before_end: int = 1
    input_tokens: int = 900
    output_tokens: int = 60
    requests: list[tuple[str, str]] = field(default_factory=list)
    batches: dict[str, dict[str, Any]] = field(default_factory=dict)
    bodies: list[dict[str, Any]] = field(default_factory=list)
    _seen: dict[str, int] = field(default_factory=dict)

    def transport(self) -> httpx2.MockTransport:
        return httpx2.MockTransport(self.handle)

    def client(self) -> httpx2.Client:
        return httpx2.Client(transport=self.transport(), base_url="https://api.anthropic.com")

    def count(self, method: str, prefix: str) -> int:
        return sum(1 for m, p in self.requests if m == method and p.startswith(prefix))

    # -- responses ---------------------------------------------------------------------

    def _message(self, params: dict[str, Any]) -> dict[str, Any]:
        content = params["messages"][0]["content"]
        text = content if isinstance(content, str) else content[0]["text"]
        seen = self._seen.get(text, 0)
        self._seen[text] = seen + 1
        refused = seen == 0 and any(marker in text for marker in self.refuse_first)
        return {
            "id": f"msg_{hashlib.sha256(text.encode()).hexdigest()[:12]}",
            "type": "message",
            "role": "assistant",
            "model": params["model"],
            "content": [] if refused else [{"type": "text", "text": answer(text)}],
            "stop_reason": "refusal" if refused else "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": 0 if refused else self.output_tokens,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 700,
            },
        }

    def _batch(self, batch_id: str) -> dict[str, Any]:
        entry = self.batches[batch_id]
        ended = entry["polls"] >= self.polls_before_end
        n = len(entry["requests"])
        return {
            "id": batch_id,
            "type": "message_batch",
            "processing_status": "ended" if ended else "in_progress",
            "request_counts": {
                "processing": 0 if ended else n,
                "succeeded": n if ended else 0,
                "errored": 0,
                "canceled": 0,
                "expired": 0,
            },
            "created_at": CREATED,
            "expires_at": "2026-11-05T00:00:00Z",
            "ended_at": ENDED if ended else None,
            "archived_at": None,
            "cancel_initiated_at": None,
            "results_url": (
                f"https://api.anthropic.com/v1/messages/batches/{batch_id}/results"
                if ended
                else None
            ),
        }

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        path = request.url.path
        self.requests.append((request.method, path))
        if request.method == "GET" and path.startswith("/v1/models/"):
            model_id = path.rsplit("/", 1)[1]
            return httpx2.Response(
                200,
                json={
                    "id": model_id,
                    "type": "model",
                    "display_name": model_id,
                    "created_at": CREATED,
                    "lifecycle": "active",
                    "capabilities": self.capabilities,
                },
            )
        if request.method == "POST" and path == "/v1/messages/count_tokens":
            return httpx2.Response(200, json={"input_tokens": 1234})
        if request.method == "POST" and path == "/v1/messages":
            body = json.loads(request.content)
            self.bodies.append(body)
            return httpx2.Response(200, json=self._message(body))
        if request.method == "POST" and path == "/v1/messages/batches":
            body = json.loads(request.content)
            batch_id = f"msgbatch_{len(self.batches):04d}"
            self.batches[batch_id] = {"requests": body["requests"], "polls": 0}
            self.bodies += [r["params"] for r in body["requests"]]
            return httpx2.Response(200, json=self._batch(batch_id))
        if request.method == "GET" and path.endswith("/results"):
            batch_id = path.split("/")[-2]
            lines = [
                json.dumps(
                    {
                        "custom_id": r["custom_id"],
                        "result": {"type": "succeeded", "message": self._message(r["params"])},
                    }
                )
                for r in self.batches[batch_id]["requests"]
            ]
            return httpx2.Response(200, content="\n".join(lines).encode())
        if request.method == "GET" and path.startswith("/v1/messages/batches/"):
            batch_id = path.rsplit("/", 1)[1]
            self.batches[batch_id]["polls"] += 1
            return httpx2.Response(200, json=self._batch(batch_id))
        return httpx2.Response(404, json={"type": "error", "error": {"type": "not_found_error"}})
