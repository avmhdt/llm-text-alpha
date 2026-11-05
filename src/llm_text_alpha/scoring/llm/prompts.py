"""Versioned prompts and request building (DQ-11, section 16).

A prompt file is frozen once a paid run uses it: its version is ``<name>@<sha8>``, and the
hash enters every cache key. The user message is the document text only, truncated to the
model's input budget from the start of the press release; no ticker, date, price or outcome
is added.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from llm_text_alpha.scoring.common import scoring_text
from llm_text_alpha.scoring.llm.base import ScoreRequest
from llm_text_alpha.scoring.llm.schema import ANSWERS, LLM_SCORE_SCHEMA


@dataclass(frozen=True)
class Prompt:
    """A system prompt and its version."""

    name: str
    text: str

    @property
    def sha8(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:8]

    @property
    def version(self) -> str:
        """``<name>@<sha8>``, recorded in ``scores.prompt_version``."""
        return f"{self.name}@{self.sha8}"


def load_prompt(path: Path) -> Prompt:
    """Read a prompt file; its stem is the prompt name."""
    return Prompt(name=path.stem, text=path.read_text("utf-8").strip())


def truncate(text: str, max_chars: int) -> tuple[str, bool]:
    """The first ``max_chars`` characters, cut back to the last line break when one is near."""
    if len(text) <= max_chars:
        return text, False
    cut = text[:max_chars]
    newline = cut.rfind("\n")
    if newline > max_chars * 0.9:
        cut = cut[:newline]
    return cut, True


def build_requests(
    documents: pl.DataFrame,
    model_id: str,
    prompt: Prompt,
    max_input_chars: int,
    max_output_tokens: int,
    request_params: Mapping[str, Any],
    text_column: str = "text",
) -> list[ScoreRequest]:
    """One request per document: the EX-99 text (or whole filing), truncated."""
    requests: list[ScoreRequest] = []
    for doc_id, text in documents.select("doc_id", text_column).iter_rows():
        body, _ = truncate(scoring_text(text), max_input_chars)
        requests.append(
            ScoreRequest(
                doc_id=doc_id,
                model_id=model_id,
                system_prompt=prompt.text,
                user_text=body,
                output_schema=LLM_SCORE_SCHEMA,
                max_output_tokens=max_output_tokens,
                request_params=dict(request_params),
            )
        )
    return requests


def parse_payload(raw_text: str, answer: str = "score") -> tuple[Any, bool]:
    """Validate a reply's JSON text against ``ANSWERS[answer]``; return it and whether valid."""
    try:
        return ANSWERS[answer].model_validate(json.loads(raw_text)), True
    except (ValueError, TypeError):
        return None, False
