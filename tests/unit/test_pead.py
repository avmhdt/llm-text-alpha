from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from llm_text_alpha import pipeline


def test_pead_check_reports_the_spread(tmp_path: Path) -> None:
    paths = pipeline.Paths(tmp_path / "data")
    folder = paths.abnormal / "model=ff5umd"
    folder.mkdir(parents=True)
    rng = np.random.default_rng(0)
    docs = [f"d{i}" for i in range(100)]
    quintile = [i % 5 + 1 for i in range(100)]
    rows = [
        (d, 1, day, 0.001 * q + rng.normal(0, 1e-4))
        for d, q in zip(docs, quintile, strict=True)
        for day in range(-2, 61)
    ]
    pl.DataFrame(rows, schema=["doc_id", "permno", "rel_day", "ar"], orient="row").write_parquet(
        folder / "ar.parquet"
    )
    pl.DataFrame(
        {"doc_id": docs, "drop_reason": [None] * 100}, schema_overrides={"drop_reason": pl.String}
    ).write_parquet(folder / "events.parquet")
    paths.signals.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "doc_id": docs,
            "scorer": ["sue"] * 100,
            "model_id": ["srw"] * 100,
            "masked": [False] * 100,
            "quintile": quintile,
        }
    ).write_parquet(paths.signals)
    report = pipeline.Report(
        tmp_path / "tables", tmp_path / "figures", tmp_path / "log.csv", "cmd", "r"
    )
    table = pipeline.run_pead_check(paths, "ff5umd", report)
    spread = table.row(-1, named=True)
    assert spread["group"] == "Q5-Q1" and spread["n"] == 40
    assert spread["car_1_60"] == pytest.approx(60 * 0.004, rel=1e-2)
    assert spread["t"] > 10
    assert (tmp_path / "tables" / "pead_sanity.md").exists()
