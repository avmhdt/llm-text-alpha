"""Committed tables: CSV and Markdown with a header comment line (section 10, Output files).

The comment names the generating command, run id, git SHA and provider label, so every
number in the repo traces back to a run.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from llm_text_alpha.io import write_text_atomic
from llm_text_alpha.tracking.trial_log import git_sha

_REPRO = {"active": False}


def set_repro(active: bool) -> None:
    """In ``lta repro``, provenance lines name the command only, so reruns are byte-identical."""
    _REPRO["active"] = active


def provenance(command: str, run_id: str, label: str, repo: Path) -> str:
    """``command=... | run_id=... | git_sha=... | provider=...``.

    Under ``lta repro`` the line is ``command=... | source=lta repro | provider=...``: the
    commit that reproduced a table is the commit it is checked in at.
    """
    if _REPRO["active"]:
        return f"command={command} | source=lta repro | provider={label or 'licensed'}"
    sha = git_sha(repo / "results" / "trial_log.csv")
    return f"command={command} | run_id={run_id} | git_sha={sha} | provider={label or 'licensed'}"


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


def to_markdown(frame: pl.DataFrame) -> str:
    """A GitHub Markdown table."""
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = ["| " + " | ".join(_cell(v) for v in row) + " |" for row in frame.iter_rows()]
    return "\n".join([header, rule, *rows])


def write_table(
    frame: pl.DataFrame,
    tables_dir: Path,
    name: str,
    command: str,
    run_id: str,
    label: str = "",
    repo: Path = Path(),
) -> tuple[Path, Path]:
    """Write ``<name>.csv`` and ``<name>.md``; the first line of each is the provenance."""
    line = provenance(command, run_id, label, repo)
    csv_path = tables_dir / f"{name}.csv"
    md_path = tables_dir / f"{name}.md"
    write_text_atomic(f"# {line}\n{frame.write_csv()}", csv_path)
    write_text_atomic(f"<!-- {line} -->\n\n{to_markdown(frame)}\n", md_path)
    return csv_path, md_path
