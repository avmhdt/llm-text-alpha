"""Typed environment settings, read from the process environment and ``.env`` (section 3).

Secrets are ``SecretStr``: they never appear in ``repr``, logs or configs.
"""

from __future__ import annotations

import sys
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PriceProviderName = Literal["auto", "efp", "sharadar", "crsp", "free"]
LLMProviderName = Literal["anthropic", "local", "fake"]

SEC_RPS_CEILING = 10.0
"""SEC's published fair-access limit; SEC_MAX_RPS may not exceed it."""

DEFAULT_DATA_ROOT = "~/data/llm-text-alpha"


class DataPlacementError(ValueError):
    """The data directory points somewhere the project must not write (section 7d)."""


class Settings(BaseSettings):
    """Every variable in ``.env.example``."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", populate_by_name=True
    )

    sec_user_agent: str = ""
    sec_max_rps: float = Field(default=8.0, gt=0, le=SEC_RPS_CEILING)
    data_root: str = DEFAULT_DATA_ROOT
    data_dir_override: str = Field(default="", validation_alias="DATA_DIR")
    log_level: Literal["DEBUG", "INFO", "WARNING"] = "INFO"
    price_provider: PriceProviderName = "auto"

    sharadar_api_key: SecretStr = SecretStr("")
    wrds_username: str = ""
    efp_data_dir: str = ""

    anthropic_api_key: SecretStr = SecretStr("")
    llm_budget_usd: float = Field(default=25.0, ge=0)
    llm_provider: LLMProviderName = "fake"
    llm_cache_dir: str = "./data/llm_cache"

    local_llm_base_url: str = "http://127.0.0.1:8080/v1"
    local_llm_model: str = "llama-3.1-8b-instruct-q4_k_m"
    hf_token: SecretStr = SecretStr("")
    torch_device: Literal["auto", "cpu", "cuda"] = "auto"

    tiingo_api_key: SecretStr = SecretStr("")
    stooq_dir: str = ""

    @property
    def data_dir(self) -> Path:
        """Root for raw/, interim/, processed/ and llm_cache/: ``DATA_DIR``, else ``DATA_ROOT``.

        ``~`` is expanded. Every ``data/`` path in the spec is relative to this directory.
        """
        return Path(self.data_dir_override or self.data_root).expanduser()

    def data_path(self, path: str | Path) -> Path:
        """Resolve a config path: a relative ``data/...`` path lands under :attr:`data_dir`."""
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        parts = PurePosixPath(candidate.as_posix()).parts
        if parts and parts[0] == ".":
            parts = parts[1:]
        if parts and parts[0] == "data":
            return self.data_dir.joinpath(*parts[1:])
        return candidate

    @property
    def llm_cache_path(self) -> Path:
        """Response cache and spend ledger directory."""
        return self.data_path(self.llm_cache_dir)

    @property
    def efp_root(self) -> Path | None:
        """equity-factor-pipeline's data root (it holds ``processed/pit``), when set."""
        return Path(self.efp_data_dir).expanduser() if self.efp_data_dir else None

    def secrets(self) -> list[str]:
        """Non-empty secret values, so the log redactor can scrub them from any string."""
        values = (self.sharadar_api_key, self.anthropic_api_key, self.hf_token, self.tiingo_api_key)
        return [value.get_secret_value() for value in values if value.get_secret_value()]


def check_data_placement(settings: Settings, platform: str = sys.platform) -> Path:
    """Return the data directory, or raise if it breaks the placement rule of section 7d.

    Data lives on the WSL ext4 filesystem, never under ``/mnt/*``. On native Windows an
    unexpanded ``~`` would put it on the system drive, so an explicit ``DATA_DIR`` is needed.
    """
    raw = settings.data_dir_override or settings.data_root
    resolved = settings.data_dir
    if platform == "win32":
        if raw.startswith("~"):
            raise DataPlacementError(
                "DATA_ROOT starts with '~', which is the system drive on native Windows. "
                "Run data commands under WSL, or set DATA_DIR to a path off C:."
            )
        return resolved
    posix = resolved.as_posix()
    if posix == "/mnt" or posix.startswith("/mnt/"):
        raise DataPlacementError(
            f"data directory {posix} is on a /mnt drive; keep data on the WSL ext4 filesystem"
        )
    return resolved


def get_settings(env_file: Path | None = Path(".env")) -> Settings:
    """Load settings fresh, so tests and long runs see the current environment.

    ``env_file=None`` ignores ``.env``; tests use it so the owner's file never leaks in.
    """
    if env_file is not None and not env_file.exists():
        env_file = None
    return Settings(_env_file=env_file)  # type: ignore[call-arg]
