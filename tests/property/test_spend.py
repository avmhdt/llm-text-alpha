"""The ledger never commits more than the budget, whatever the order of holds and settles."""

from __future__ import annotations

import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from llm_text_alpha.scoring.llm.spend import BudgetExceededError
from tests import llm_support as s

steps = st.lists(
    st.tuples(
        st.floats(min_value=0.0, max_value=0.6, allow_nan=False),
        st.sampled_from(["settle", "release", "open"]),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    ),
    min_size=1,
    max_size=12,
)


@settings(max_examples=40, deadline=None)
@given(budget=st.floats(min_value=0.0, max_value=2.0, allow_nan=False), plan=steps)
def test_committed_never_exceeds_budget(
    budget: float, plan: list[tuple[float, str, float]]
) -> None:
    with tempfile.TemporaryDirectory() as folder:
        tracker = s.tracker(Path(folder), budget=budget)
        for usd, action, fraction in plan:
            before = tracker.n_entries()
            try:
                hold = tracker.reserve("run", s.HAIKU, usd)
            except BudgetExceededError:
                assert tracker.n_entries() == before
                continue
            if action == "settle":
                tracker.settle_usd(hold, usd * fraction)  # actual cost <= the worst case
            elif action == "release":
                tracker.release(hold)
            assert tracker.committed_usd() <= budget + 1e-9
