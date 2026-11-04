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
