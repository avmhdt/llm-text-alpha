"""Only the provider module may build an API client (NFR-06): every call passes the tracker."""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).parents[2] / "src" / "llm_text_alpha"
ALLOWED = SRC / "scoring" / "llm" / "anthropic_provider.py"


def test_anthropic_client_built_only_in_the_provider() -> None:
    pattern = re.compile(r"\bAnthropic\(|\bAsyncAnthropic\(")
    offenders = [
        path.relative_to(SRC).as_posix()
        for path in SRC.rglob("*.py")
        if path != ALLOWED and pattern.search(path.read_text("utf-8"))
    ]
    assert offenders == []
    assert pattern.search(ALLOWED.read_text("utf-8"))


def test_no_openai_sdk() -> None:
    pattern = re.compile(r"^\s*(import openai|from openai)", re.MULTILINE)
    assert not [p for p in SRC.rglob("*.py") if pattern.search(p.read_text("utf-8"))]
