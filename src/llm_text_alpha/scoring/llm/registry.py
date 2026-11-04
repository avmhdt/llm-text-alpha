"""Model registry from ``configs/models.yaml``: provider, cutoffs and request settings (DQ-12)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml


def _month(value: Any) -> date | None:
    if value is None:
        return None
    year, month = (int(part) for part in str(value).split("-")[:2])
    return date(year, month, 1)


@dataclass(frozen=True)
class ModelSpec:
    """One registry entry."""

    model_id: str
    provider: str
    training_data_cutoff: date | None
    reliable_knowledge_cutoff: date | None
    source: str
    retrieved: str | None
    request: dict[str, Any] = field(default_factory=dict)

    @property
    def post_cutoff_start(self) -> date | None:
        """First day of the month after the training data cutoff (the conservative date)."""
        cutoff = self.training_data_cutoff
        if cutoff is None:
            return None
        return date(cutoff.year + cutoff.month // 12, cutoff.month % 12 + 1, 1)


class ModelRegistry:
    """Lookup by model id."""

    def __init__(self, specs: dict[str, ModelSpec]) -> None:
        self._specs = specs

    @classmethod
    def load(cls, path: Path) -> ModelRegistry:
        """Read ``configs/models.yaml``."""
        raw = yaml.safe_load(path.read_text("utf-8"))
        specs = {
            model_id: ModelSpec(
                model_id=model_id,
                provider=str(entry["provider"]),
                training_data_cutoff=_month(entry.get("training_data_cutoff")),
                reliable_knowledge_cutoff=_month(entry.get("reliable_knowledge_cutoff")),
                source=str(entry.get("source", "")),
                retrieved=str(entry["retrieved"]) if entry.get("retrieved") else None,
                request=dict(entry.get("request") or {}),
            )
            for model_id, entry in raw.items()
        }
        return cls(specs)

    def get(self, model_id: str) -> ModelSpec:
        """The entry for ``model_id``; an unregistered model cannot be scored."""
        if model_id not in self._specs:
            raise KeyError(f"{model_id} is not in the model registry")
        return self._specs[model_id]

    def ids(self) -> list[str]:
        """Registered model ids."""
        return sorted(self._specs)
