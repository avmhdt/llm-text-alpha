from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_text_alpha.settings import (
    DataPlacementError,
    Settings,
    check_data_placement,
    get_settings,
)


def test_defaults_without_env_file() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.sec_max_rps == 8.0
    assert settings.price_provider == "auto"
    assert settings.llm_budget_usd == 25.0
    assert settings.data_dir == Path("~/data/llm-text-alpha").expanduser()


def test_ci_forces_the_fake_provider() -> None:
    assert get_settings(env_file=None).llm_provider == "fake"


def test_data_dir_override_wins(make_settings: Callable[..., Settings], tmp_path: Path) -> None:
    settings = make_settings()
    assert settings.data_dir == tmp_path / "data"


def test_data_path_maps_data_prefix(make_settings: Callable[..., Settings], tmp_path: Path) -> None:
    settings = make_settings()
    assert settings.data_path("data/raw/lm/x.csv") == tmp_path / "data" / "raw" / "lm" / "x.csv"
    assert settings.data_path("./data/llm_cache") == tmp_path / "data" / "llm_cache"
    assert settings.llm_cache_path == tmp_path / "data" / "llm_cache"
    assert settings.data_path("configs/models.yaml") == Path("configs/models.yaml")
    absolute = tmp_path / "elsewhere"
    assert settings.data_path(absolute) == absolute


def test_sec_rate_ceiling() -> None:
    assert Settings(_env_file=None, sec_max_rps=10).sec_max_rps == 10  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        Settings(_env_file=None, sec_max_rps=10.5)  # type: ignore[call-arg]


def test_secrets_are_hidden_and_listed(make_settings: Callable[..., Settings]) -> None:
    settings = make_settings(sharadar_api_key="sk-sharadar-1", hf_token="hf-2")
    assert "sk-sharadar-1" not in repr(settings)
    assert sorted(settings.secrets()) == ["hf-2", "sk-sharadar-1"]


def test_env_file_is_read(tmp_path: Path) -> None:
    env = tmp_path / "custom.env"
    env.write_text('SEC_USER_AGENT="a b@c.de" # comment\nDATA_DIR= # empty\n', "utf-8")
    settings = get_settings(env)
    assert settings.sec_user_agent == "a b@c.de"
    assert settings.data_dir_override == ""
    assert get_settings(tmp_path / "missing.env").sec_user_agent == ""


def test_efp_root(make_settings: Callable[..., Settings], tmp_path: Path) -> None:
    assert make_settings().efp_root is None
    assert make_settings(efp_data_dir=str(tmp_path)).efp_root == tmp_path


@pytest.mark.parametrize("path", ["/mnt/d/data", "/mnt"])
def test_placement_rejects_mnt(make_settings: Callable[..., Settings], path: str) -> None:
    with pytest.raises(DataPlacementError):
        check_data_placement(make_settings(data_dir_override=path), platform="linux")


def test_placement_accepts_ext4(make_settings: Callable[..., Settings]) -> None:
    settings = make_settings(data_dir_override="/home/ze/data/llm-text-alpha")
    assert check_data_placement(settings, platform="linux").as_posix().endswith("llm-text-alpha")


def test_placement_rejects_tilde_on_windows(make_settings: Callable[..., Settings]) -> None:
    settings = make_settings(data_dir_override="")
    with pytest.raises(DataPlacementError):
        check_data_placement(settings, platform="win32")
    explicit = make_settings(data_dir_override="E:/lta-data")
    assert check_data_placement(explicit, platform="win32") == Path("E:/lta-data")
