"""Experiment configs (section 10): YAML files under ``configs/``, validated where they are used."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_CONFIG = Path("configs/default.yaml")
SMOKE_CONFIG = Path("configs/smoke.yaml")


class ConfigError(ValueError):
    """A config file is missing or malformed."""


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping."""
    if not path.exists():
        raise ConfigError(f"config {path} does not exist")
    loaded = yaml.safe_load(path.read_text("utf-8"))
    if not isinstance(loaded, dict):
        raise ConfigError(f"config {path} is not a mapping")
    return loaded


class SmokeFiling(BaseModel):
    """A filing whose acceptance time was recorded from its SGML header (AC-W-04)."""

    model_config = ConfigDict(extra="forbid")

    cik: int
    accession: str
    form: str
    items: list[str]
    acceptance_raw: str
    acceptance_ts_utc: datetime

    @field_validator("acceptance_ts_utc")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("acceptance_ts_utc must carry a timezone")
        return value


class SmokeConfig(BaseModel):
    """The ``smoke`` section: 10 filings, one month of prices and factors."""

    model_config = ConfigDict(extra="forbid")

    month: str = Field(pattern=r"^\d{4}-\d{2}$")
    max_securities: int = Field(default=50, gt=0)
    free_tickers: list[str] = Field(default_factory=list)
    filings: list[SmokeFiling]

    def month_bounds(self) -> tuple[date, date]:
        """First and last calendar day of :attr:`month`."""
        year, month = (int(part) for part in self.month.split("-"))
        first = date(year, month, 1)
        following = date(year + month // 12, month % 12 + 1, 1)
        return first, date.fromordinal(following.toordinal() - 1)


def load_smoke(path: Path = SMOKE_CONFIG) -> SmokeConfig:
    """The ``smoke`` section of ``path``."""
    section = load_yaml(path).get("smoke")
    if section is None:
        raise ConfigError(f"config {path} has no 'smoke' section")
    return SmokeConfig.model_validate(section)
