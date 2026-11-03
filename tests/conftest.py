"""Shared fixtures. Tests never read the owner's ``.env`` and never touch the network."""

from __future__ import annotations

import io
import shutil
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import structlog

from llm_text_alpha.settings import Settings

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop every settings variable from the process environment, and force the fake LLM."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "fake")


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    """Each test starts and ends with structlog's defaults."""
    structlog.reset_defaults()
    yield
    structlog.reset_defaults()


@pytest.fixture
def make_settings(tmp_path: Path) -> Callable[..., Settings]:
    """Settings with data under ``tmp_path/data`` and no ``.env``."""

    def build(**overrides: Any) -> Settings:
        values: dict[str, Any] = {
            "sec_user_agent": "lta-tests tests@example.com",
            "data_dir_override": str(tmp_path / "data"),
        }
        values.update(overrides)
        return Settings(_env_file=None, **values)  # type: ignore[call-arg]

    return build


def french_csv(header: str, rows: list[str], preamble: str = "Synthetic file.\n\n") -> str:
    """A French-library style CSV: preamble, header starting with ',', rows, copyright."""
    return preamble + header + "\n" + "\n".join(rows) + "\n\nCopyright synthetic\n"


def zipped(name: str, text: str) -> bytes:
    """``text`` as a one-file ZIP archive."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as handle:
        handle.writestr(name, text)
    return buffer.getvalue()


@pytest.fixture(scope="session")
def efp_root(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """An equity-factor-pipeline style data root holding a small synthetic eqpit store."""
    from eqpit.pit.build import build_pit
    from eqpit.providers.synthetic import SyntheticProvider, write_market
    from eqpit.synthetic import SyntheticConfig

    root = tmp_path_factory.mktemp("efp")
    write_market(SyntheticConfig(seed=7, n_securities=30, years=2, start_year=2015), root / "raw")
    build_pit(SyntheticProvider(root / "raw"), root / "processed" / "pit")
    yield root
    shutil.rmtree(root, ignore_errors=True)
