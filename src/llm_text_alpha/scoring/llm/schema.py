"""The LLM's structured output (FR-10): one JSON object per document.

The JSON schema sent to the model uses only types, enums and required fields; numeric
bounds are checked here with pydantic after parsing, so a response that is valid JSON but
out of range is recorded as ``schema_valid = False`` rather than silently accepted.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Label = Literal["negative", "neutral", "positive"]
Direction = Literal[-1, 0, 1]


class LLMScore(BaseModel):
    """Validated score fields. ``score`` is directional sentiment in [-1, 1]."""

    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=-1.0, le=1.0)
    label: Label
    surprise: Direction
    guidance: Direction | None
    materiality: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)


LLM_SCORE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {
            "type": "number",
            "description": "Directional tone for the stock, from -1 (very negative) to 1.",
        },
        "label": {"type": "string", "enum": ["negative", "neutral", "positive"]},
        "surprise": {
            "type": "integer",
            "enum": [-1, 0, 1],
            "description": "Results versus expectations stated in the text: -1 miss, 0, 1 beat.",
        },
        "guidance": {
            "anyOf": [{"type": "integer", "enum": [-1, 0, 1]}, {"type": "null"}],
            "description": "Outlook change: -1 lowered, 0 maintained, 1 raised, null if none.",
        },
        "materiality": {"type": "number", "description": "0 routine to 1 highly material."},
        "confidence": {"type": "number", "description": "0 to 1."},
    },
    "required": ["score", "label", "surprise", "guidance", "materiality", "confidence"],
    "additionalProperties": False,
}


class ReidAnswer(BaseModel):
    """Re-identification probe (FR-20): the model's guess of the company and period."""

    model_config = ConfigDict(extra="forbid")

    company: str = Field(max_length=200)
    ticker: str | None = Field(default=None, max_length=12)
    period_year: int | None = Field(default=None, ge=1980, le=2035)
    period_quarter: Literal[1, 2, 3, 4] | None = None
    confidence: float = Field(ge=0.0, le=1.0)


REID_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "company": {"type": "string", "description": "Best guess of the company's name."},
        "ticker": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "period_year": {
            "anyOf": [{"type": "integer"}, {"type": "null"}],
            "description": "Calendar year in which the reported quarter ended.",
        },
        "period_quarter": {
            "anyOf": [{"type": "integer", "enum": [1, 2, 3, 4]}, {"type": "null"}],
            "description": "Calendar quarter (1-4) in which the reported quarter ended.",
        },
        "confidence": {"type": "number", "description": "0 to 1."},
    },
    "required": ["company", "ticker", "period_year", "period_quarter", "confidence"],
    "additionalProperties": False,
}


class RecallAnswer(BaseModel):
    """Outcome-recall probe (FR-20): the remembered direction of the stock's move."""

    model_config = ConfigDict(extra="forbid")

    direction: Literal["up", "down"]
    confidence: float = Field(ge=0.0, le=1.0)


RECALL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "direction": {"type": "string", "enum": ["up", "down"]},
        "confidence": {"type": "number", "description": "0 to 1."},
    },
    "required": ["direction", "confidence"],
    "additionalProperties": False,
}

ANSWERS: dict[str, type[BaseModel]] = {
    "score": LLMScore,
    "reid": ReidAnswer,
    "recall": RecallAnswer,
}
"""Pydantic model that validates each kind of answer, by ``ScoreRequest.answer``."""
