"""AC-W-01, full-data part: every real signal enters strictly after its publication time.

Local only (``pytest -m data``): it reads ``signals.parquet`` under the owner's DATA_DIR.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import polars as pl
import pytest

from llm_text_alpha.corpus.calendar import xnys
from llm_text_alpha.corpus.gate import assert_no_lookahead, entries
from llm_text_alpha.pipeline import Paths
from llm_text_alpha.settings import get_settings

REPO = Path(__file__).parents[2]


@pytest.mark.data
def test_signal_precedes_trade_full() -> None:
    settings = get_settings(REPO / ".env")
    path = Paths.of(settings).signals
    if not path.exists():
        pytest.skip(f"{path} not built yet")
    signals = pl.read_parquet(path)
    assert signals.height > 0
    assert_no_lookahead(signals, timedelta(minutes=5))
    entry_date, entry_ts = entries(signals["event_ts_utc"], xnys(), timedelta(minutes=5))
    assert (entry_date == signals["entry_date"]).all()
    assert (entry_ts == signals["entry_ts_utc"]).all()
