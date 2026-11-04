"""AC-W-05: a second run of the same requests makes no HTTP request and no ledger entry."""

from __future__ import annotations

from pathlib import Path

from tests import llm_support as s
from tests.fake_anthropic import FakeAnthropic


def test_rerun_is_served_from_the_cache(tmp_path: Path) -> None:
    batch = s.requests(25)
    first_api = FakeAnthropic(refuse_first={"Quarter 3:"})
    first = s.provider(tmp_path, first_api).score(batch, "batch")
    assert first_api.requests
    entries = s.tracker(tmp_path / "llm_cache").n_entries()
    spent = s.tracker(tmp_path / "llm_cache").committed_usd()

    second_api = FakeAnthropic()
    second_provider = s.provider(tmp_path, second_api)
    for mode in ("batch", "sync"):
        second = second_provider.score(batch, mode)  # type: ignore[arg-type]
        assert second_api.requests == []
        assert second_provider.tracker.n_entries() == entries
        assert second_provider.tracker.committed_usd() == spent
        assert {r.mode for r in second} == {"cache_hit"}
        assert all(r.usd == 0.0 for r in second)
        assert [r.payload for r in second] == [r.payload for r in first]
        assert [r.cost_usd for r in second] == [r.usd for r in first]
    assert sum(r.usd for r in first) == spent  # retries included in the document's cost
