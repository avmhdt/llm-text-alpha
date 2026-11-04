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


def cumulative_returns_figure(
    series: dict[str, pl.DataFrame], title: str, caption: str, out: Path
) -> Path:
    """Cumulative net long-short return (simple sum of daily returns, percent) per label."""
    figure, axis = plt.subplots(figsize=(10, 4.5))
    for label, frame in series.items():
        ordered = frame.sort("date")
        axis.plot(
            ordered["date"].to_numpy(), ordered["ls_net"].cum_sum().to_numpy() * 100, label=label
        )
    axis.axhline(0, color="#cccccc", linewidth=0.8)
    axis.set_ylabel("cumulative net return, % (sum of daily)")
    axis.set_title(title)
    axis.legend(fontsize=8)
    figure.text(0.01, 0.01, caption, fontsize=7, color="#555555")
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=120)
    plt.close(figure)
    return out


def capacity_figure(curve: pl.DataFrame, title: str, caption: str, out: Path) -> Path:
    """Net annual alpha (percent) against AUM per leg on a log axis."""
    figure, axis = plt.subplots(figsize=(8, 4.5))
    aum = curve["aum_usd"].to_numpy()
    axis.plot(
        aum,
        curve["alpha_annual"].to_numpy() * 100,
        marker="o",
        label="net of D-11 costs and impact",
    )
    if "alpha_annual_impact_only" in curve.columns:
        axis.plot(
            aum,
            curve["alpha_annual_impact_only"].to_numpy() * 100,
            marker="s",
            linestyle="--",
            label="impact only (gross of D-11 costs)",
        )
    axis.legend(fontsize=8)
    axis.set_xscale("log")
    axis.axhline(0, color="#cccccc", linewidth=0.8)
    axis.set_xlabel("AUM per leg, USD")
    axis.set_ylabel("net alpha, % per year")
    axis.set_title(title)
    figure.text(0.01, 0.01, caption, fontsize=7, color="#555555")
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=120)
    plt.close(figure)
    return out


def ablation_figure(table: pl.DataFrame, title: str, caption: str, out: Path) -> Path:
    """CAR[0,5] Q5 - Q1 spread (percent) with a 95% interval per scorer combination."""
    figure, axis = plt.subplots(figsize=(9, 4.5))
    labels = table["combination"].to_list()
    spread = table["car5_spread"].to_numpy() * 100
    t = table["t_car5_spread"].to_numpy()
    err = [abs(s / tv) * 1.96 if tv else 0.0 for s, tv in zip(spread, t, strict=True)]
    axis.bar(range(len(labels)), spread, yerr=err, color="#67a9cf", capsize=4)
    axis.set_xticks(range(len(labels)), labels, rotation=30, ha="right", fontsize=8)
    axis.axhline(0, color="#999999", linewidth=0.8)
    axis.set_ylabel("CAR[0,5] Q5 - Q1, %")
    axis.set_title(title)
    figure.text(0.01, 0.01, caption, fontsize=7, color="#555555")
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=120)
    plt.close(figure)
    return out
