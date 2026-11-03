from __future__ import annotations

import io
import json
from pathlib import Path

import structlog

from llm_text_alpha.logging import REDACTED, Redactor, configure_logging, log_path


def test_redactor_masks_known_secrets_and_secret_keys() -> None:
    redact = Redactor(["sk-live-123", ""])
    event = redact(
        None,
        "info",
        {
            "event": "calling with sk-live-123 inside",
            "api_key": "anything",
            "x-api-key": "other",
            "headers": {"Authorization": "Bearer t", "accept": "json"},
            "urls": ["https://h/?k=sk-live-123", 3],
            "token": "",
            "count": 4,
        },
    )
    assert event["event"] == f"calling with {REDACTED} inside"
    assert event["api_key"] == REDACTED
    assert event["x-api-key"] == REDACTED
    assert event["headers"] == {"Authorization": REDACTED, "accept": "json"}
    assert event["urls"] == [f"https://h/?k={REDACTED}", 3]
    assert event["token"] == ""
    assert event["count"] == 4


def test_json_lines_go_to_file_and_console(tmp_path: Path) -> None:
    console = io.StringIO()
    target = log_path(tmp_path / "logs", "data smoke", "run-1")
    assert target.name == "data-smoke-run-1.jsonl"
    configure_logging("INFO", ["hunter2"], target, console)
    logger = structlog.get_logger("t")
    logger.info("hello", secret_value="hunter2", n=1)
    logger.debug("hidden")
    configure_logging("INFO", [], None, io.StringIO())
    lines = target.read_text("utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["event"] == "hello"
    assert record["secret_value"] == REDACTED
    assert record["level"] == "info"
    assert "hunter2" not in console.getvalue()
    assert json.loads(console.getvalue().splitlines()[0])["n"] == 1
