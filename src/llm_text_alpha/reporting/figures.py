"""Static figures (matplotlib, Agg backend): committed PNGs of aggregates only."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from llm_text_alpha.evaluation.event_study import SPREAD

_QUINTILE_COLORS = ("#b2182b", "#ef8a62", "#999999", "#67a9cf", "#2166ac")


def event_study_figure(path: pl.DataFrame, title: str, caption: str, out: Path) -> Path:
    """CAAR by quintile (left) and Q5 - Q1 with its t and bootstrap 95% bands (right)."""
    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for i, color in enumerate(_QUINTILE_COLORS, start=1):
        group = path.filter(pl.col("group") == f"Q{i}").sort("rel_day")
        if group.is_empty():
            continue
        days = group["rel_day"].to_numpy()
        left.plot(
            days, group["caar"].to_numpy() * 100, color=color, label=f"Q{i} (N={group['n'][0]})"
        )
        left.fill_between(
            days,
            group["lo_t"].to_numpy() * 100,
            group["hi_t"].to_numpy() * 100,
            color=color,
            alpha=0.12,
            linewidth=0,
        )
    spread = path.filter(pl.col("group") == SPREAD).sort("rel_day")
    if not spread.is_empty():
        days = spread["rel_day"].to_numpy()
        right.plot(days, spread["caar"].to_numpy() * 100, color="black", label="Q5 - Q1")
        right.fill_between(
            days,
            spread["lo_t"].to_numpy() * 100,
            spread["hi_t"].to_numpy() * 100,
            color="#4d4d4d",
            alpha=0.2,
            linewidth=0,
            label="95% band (t)",
        )
        right.plot(
            days,
            spread["lo_boot"].to_numpy() * 100,
            color="#4d4d4d",
            linestyle=":",
            label="95% band (firm bootstrap)",
        )
        right.plot(days, spread["hi_boot"].to_numpy() * 100, color="#4d4d4d", linestyle=":")
    for axis, heading in ((left, "CAAR by trailing score quintile"), (right, "Top minus bottom")):
        axis.axhline(0, color="#cccccc", linewidth=0.8)
        axis.axvline(0, color="#cccccc", linewidth=0.8, linestyle="--")
        axis.set_xlabel("Sessions from entry (0 = entry day, open to close)")
        axis.set_ylabel("Cumulative abnormal return (%)")
        axis.set_title(heading, fontsize=10)
        axis.legend(fontsize=8, frameon=False)
    figure.suptitle(title, fontsize=11)
    figure.text(0.5, 0.005, caption, ha="center", fontsize=8, color="#444444")
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=120, metadata={"Software": None})
    plt.close(figure)
    return out
