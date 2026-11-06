"""M5 evaluation runs: portfolios, ablation, capacity and trial accounting (FR-18 to FR-21, FR-30).

Each run writes a committed table (aggregates only), a figure where section 14 asks for one,
and trial-log rows: ``portfolio`` rows carry their daily net return series; the rest are
``analysis`` rows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from llm_text_alpha.corpus.calendar import TradingCalendar, xnys
from llm_text_alpha.evaluation import ablation as abl
from llm_text_alpha.evaluation import capacity as cap
from llm_text_alpha.evaluation import portfolio as pf
from llm_text_alpha.evaluation import trials as tr
from llm_text_alpha.evaluation.abnormal import cumulative
from llm_text_alpha.evaluation.regressions import cross_sectional_regression
from llm_text_alpha.pipeline import DOC_TYPE, Paths, Report, load_prices
from llm_text_alpha.reporting import figures
from llm_text_alpha.reporting.tables import write_table
from llm_text_alpha.tracking import trial_log

Window = tuple[date | None, date | None]


@dataclass(frozen=True)
class Context:
    """Inputs every M5 run shares, read once."""

    paths: Paths
    calendar: TradingCalendar
    signals: pl.DataFrame
    prices: pl.DataFrame
    factors: pl.DataFrame
    returns_end: date
    costs_bps: Mapping[str, float]
    lags: int
    ar_model: str


def context(
    paths: Paths, costs_bps: Mapping[str, float], lags: int = 10, ar_model: str = "ff5umd"
) -> Context:
    """Signals with each event's universe rank, prices and factors."""
    prices = load_prices(
        paths, ["security_id", "permno", "date", "ret", "ret_open_close", "close", "volume"]
    )
    security_of = prices.select("permno", "security_id").unique("permno")
    universe = pl.read_parquet(paths.panels / "universe_monthly.parquet")
    signals = pf.with_rank(pl.read_parquet(paths.signals), universe, security_of)
    factors = pl.read_parquet(paths.panels / "factors_daily.parquet")
    end = min(factors["date"].max(), prices["date"].max())  # type: ignore[type-var]
    return Context(
        paths,
        xnys(),
        signals,
        prices.drop("security_id"),
        factors,
        end,  # type: ignore[arg-type]
        costs_bps,
        lags,
        ar_model,
    )


def raw(ctx: Context, scorer: str) -> pl.DataFrame:
    """One scorer's raw-text signals with a quintile."""
    return ctx.signals.filter(
        (pl.col("scorer") == scorer) & ~pl.col("masked") & (pl.col("quintile") > 0)
    )


def in_window(frame: pl.DataFrame, window: Window) -> pl.DataFrame:
    """Events whose publication time falls in ``window`` (dates inclusive, UTC)."""
    start, end = window
    out = frame
    if start is not None:
        out = out.filter(pl.col("event_ts_utc").dt.date() >= start)
    if end is not None:
        out = out.filter(pl.col("event_ts_utc").dt.date() <= end)
    return out


def pead_factors(ctx: Context, hold: int) -> pl.DataFrame:
    """``factors_daily`` plus ``pead``: the gross long-short of SUE quintiles, same holding rule.

    Days with no SUE position in a leg contribute 0 for that leg.
    """
    pead = pf.daily_returns(
        raw(ctx, "sue"), ctx.prices, ctx.calendar, hold, ctx.costs_bps, end=ctx.returns_end
    ).daily.select("date", pl.col("ls_gross").alias("pead"))
    return ctx.factors.join(pead, on="date", how="left").with_columns(pl.col("pead").fill_null(0.0))


def _log_portfolio(
    params: dict[str, Any], daily: pl.DataFrame, fit: pf.AlphaResult, report: Report
) -> str:
    run_id = trial_log.new_run_id()
    dates = daily["date"].to_list()
    path = trial_log.write_returns(run_id, dates, daily["ls_net"].to_list(), report.trial_log)
    trial_log.record_run(
        run_id,
        params,
        {
            "alpha_daily": fit.alpha_daily,
            "t_alpha": fit.t_alpha,
            "betas": fit.betas,
            "n_days": fit.n_days,
            "r2": fit.r2,
        },
        report.trial_log,
    )
    trial_log.log_trial(
        run_id,
        "portfolio",
        params,
        dates[0],
        dates[-1],
        len(dates),
        float(daily["ls_net"].mean() / daily["ls_net"].std() * np.sqrt(252.0)),  # type: ignore[operator]
        path,
        report.trial_log,
    )
    return run_id


def run_portfolios(
    ctx: Context,
    scorers: Sequence[str],
    windows: Mapping[str, Window],
    holds: Sequence[int],
    cost_mults: Sequence[float],
    report: Report,
) -> pl.DataFrame:
    """Every scorer x window x hold x cost multiple; table ``portfolio_alpha`` and a figure."""
    rows: list[dict[str, Any]] = []
    curves: dict[str, pl.DataFrame] = {}
    for hold in holds:
        factors = pead_factors(ctx, hold)
        for window_name, window in windows.items():
            for scorer in scorers:
                events = in_window(raw(ctx, scorer), window)
                if events.is_empty():
                    continue
                for mult in cost_mults:
                    result = pf.daily_returns(
                        events, ctx.prices, ctx.calendar, hold, ctx.costs_bps, mult, ctx.returns_end
                    )
                    daily = pf.active_days(result.daily)
                    if daily.height < 20:
                        continue
                    regressors = pf.ALPHA_FACTORS if scorer != "sue" else pf.FACTORS_ONLY
                    fit = pf.factor_alpha(daily, factors, regressors=regressors, lags=ctx.lags)
                    params = {
                        "scorer": scorer,
                        "model_id": events["model_id"][0],
                        "masked": False,
                        "doc_type": DOC_TYPE,
                        "window": window_name,
                        "window_start": str(window[0]) if window[0] else None,
                        "hold_days": hold,
                        "cost_mult": mult,
                        "costs_bps": dict(ctx.costs_bps),
                        "weighting": "equal",
                        "alpha_factors": list(regressors),
                        "nw_lags": ctx.lags,
                    }
                    _log_portfolio(params, daily, fit, report)
                    gross = pf.factor_alpha(
                        daily, factors, "ls_gross", regressors=regressors, lags=ctx.lags
                    )
                    rows.append(
                        {
                            "scorer": scorer,
                            "window": window_name,
                            "hold_days": hold,
                            "cost_mult": mult,
                            "events": events.height,
                            "positions": result.n_positions,
                            "days": fit.n_days,
                            "mean_daily_net": fit.mean_daily,
                            "sharpe_net": fit.sharpe_annual,
                            "alpha_annual_net": fit.alpha_annual,
                            "t_alpha_net": fit.t_alpha,
                            "alpha_annual_gross": gross.alpha_annual,
                            "t_alpha_gross": gross.t_alpha,
                            "beta_pead": fit.betas.get("pead"),
                            "beta_mktrf": fit.betas["mktrf"],
                            "missing_returns": result.missing_returns,
                        }
                    )
                    if hold == 5 and mult == 1.0 and window_name == "all":
                        curves[scorer] = daily
    table = pl.DataFrame(rows)
    write_table(
        table, report.tables, "portfolio_alpha", report.command, report.run_id, report.label
    )
    if curves:
        figures.cumulative_returns_figure(
            curves,
            "Long-short Q5 - Q1, net of D-11 costs, H = 5",
            "Equal weight, entry at the open after acceptance + 5 min; "
            f"days through {ctx.returns_end}",
            report.figures / "portfolio_cumulative.png",
        )
    return table


def _cars(ctx: Context, horizons: Sequence[int]) -> pl.DataFrame:
    folder = ctx.paths.abnormal / f"model={ctx.ar_model}"
    events = pl.read_parquet(folder / "events.parquet").filter(pl.col("drop_reason").is_null())
    ar = pl.read_parquet(folder / "ar.parquet").join(
        events.select("doc_id"), on="doc_id", how="semi"
    )
    return cumulative(ar, horizons).drop("permno")


def cost_per_document(paths: Paths, scorers: Sequence[str]) -> dict[str, float]:
    """Mean ``cost_usd`` per scored document, by scorer (USD)."""
    from llm_text_alpha.pipeline import load_scores

    out: dict[str, float] = {}
    for scorer in scorers:
        scores = load_scores(paths, scorer)
        out[scorer] = float(np.mean(scores["cost_usd"].fill_null(0.0).to_numpy()))
    return out


def run_ablation(
    ctx: Context,
    scorers: Sequence[str],
    windows: Mapping[str, Window],
    hold: int,
    report: Report,
    reads: Mapping[str, str],
    unit_costs: Mapping[str, float],
    min_events: int = 50,
) -> pl.DataFrame:
    """The FR-19 table on the intersection event set, one row per combination and window."""
    events = abl.intersection(ctx.signals, scorers).join(
        _cars(ctx, [5, 20]), on="doc_id", how="inner"
    )
    factors = pead_factors(ctx, hold)
    rows: list[dict[str, Any]] = []
    plotted: list[dict[str, Any]] = []
    for combo in abl.combos(scorers):
        name = "+".join(combo)
        ranked = abl.combined(events, combo, ctx.calendar).join(
            events.select("doc_id", "car_5", "car_20"), on="doc_id"
        )
        for window_name, window in windows.items():
            frame = in_window(ranked, window).filter(pl.col("quintile") > 0)
            if frame.height < min_events:
                continue
            spread5 = abl.spread(frame, "car_5")
            spread20 = abl.spread(frame, "car_20")
            ic = abl.information_coefficient(frame, "car_5")
            fit = cross_sectional_regression(
                frame, "car_5", ["score_z", "size_log", "mom_12_2", "sue"], "twoway"
            )
            coef = fit.to_frame().filter(pl.col("term") == "score_z").row(0, named=True)
            port = pf.daily_returns(
                frame, ctx.prices, ctx.calendar, hold, ctx.costs_bps, 1.0, ctx.returns_end
            )
            daily = pf.active_days(port.daily)
            alpha = pf.factor_alpha(daily, factors, lags=ctx.lags) if daily.height > 20 else None
            per_doc = sum(unit_costs.get(s, 0.0) for s in combo)
            traded = port.n_positions
            row = {
                "combination": name,
                "window": window_name,
                "events": frame.height,
                "reads": "; ".join(f"{s}: {reads.get(s, '?')}" for s in combo),
                "car5_spread": spread5["spread"],
                "t_car5_spread": spread5["t_spread"],
                "car20_spread": spread20["spread"],
                "t_car20_spread": spread20["t_spread"],
                "ic_car5": ic["ic"],
                "ic_monthly": ic["ic_monthly"],
                "t_ic_monthly": ic["t_ic_monthly"],
                "coef_controls": coef["coef"],
                "t_coef_controls": coef["t"],
                "alpha_annual_net": alpha.alpha_annual if alpha else None,
                "t_alpha_net": alpha.t_alpha if alpha else None,
                "usd_per_document": per_doc,
                "usd_per_traded_signal": per_doc * frame.height / traded if traded else None,
            }
            rows.append(row)
            if window_name == "all":
                plotted.append(row)
            params = {
                "combination": name,
                "window": window_name,
                "doc_type": DOC_TYPE,
                "event_set": "intersection:" + ",".join(scorers),
                "hold_days": hold,
            }
            run_id = trial_log.new_run_id()
            trial_log.record_run(run_id, params, row, report.trial_log)
            trial_log.log_trial(
                run_id,
                "ablation",
                params,
                None,
                None,
                0,
                None,
                None,
                report.trial_log,
                run_kind="analysis",
            )
    table = pl.DataFrame(rows)
    write_table(table, report.tables, "ablation", report.command, report.run_id, report.label)
    if plotted:
        figures.ablation_figure(
            pl.DataFrame(plotted),
            "Ablation on identical events: CAR[0,5] Q5 - Q1",
            f"Intersection of {', '.join(scorers)}: {events.height} events; "
            "95% intervals from Welch t",
            report.figures / "ablation.png",
        )
    return table


def run_capacity(
    ctx: Context,
    scorer: str,
    window: tuple[str, Window],
    hold: int,
    grid: Sequence[float],
    participation: float,
    adv_days: int,
    impact_coef: float,
    report: Report,
) -> tuple[pl.DataFrame, dict[str, float | None]]:
    """Net alpha against AUM for one strategy; table ``capacity`` and its figure."""
    events = in_window(raw(ctx, scorer), window[1])
    curve = cap.capacity_curve(
        events,
        ctx.prices,
        pead_factors(ctx, hold),
        ctx.calendar,
        hold,
        grid,
        participation,
        adv_days,
        impact_coef,
        ctx.costs_bps,
        ctx.lags,
        ctx.returns_end,
    ).with_columns(pl.lit(scorer).alias("scorer"), pl.lit(window[0]).alias("window"))
    points = cap.breakpoints(curve) | {
        f"impact_only_{k}": v for k, v in cap.breakpoints(curve, "alpha_annual_impact_only").items()
    }
    write_table(curve, report.tables, "capacity", report.command, report.run_id, report.label)
    figures.capacity_figure(
        curve,
        f"Capacity: {scorer}, H = {hold}, {window[0]} window",
        f"Participation {participation:.0%} of {adv_days}-day ADV; impact {impact_coef} x sigma x "
        f"sqrt(q/ADV); net half alpha at {points['aum_half'] or 'n/a'}, "
        f"zero at {points['aum_zero'] or 'n/a'}",
        report.figures / "capacity.png",
    )
    params = {
        "scorer": scorer,
        "window": window[0],
        "hold_days": hold,
        "participation": participation,
        "adv_days": adv_days,
        "impact_coef": impact_coef,
        "grid": list(grid),
    }
    run_id = trial_log.new_run_id()
    trial_log.record_run(run_id, params, {"curve": curve.to_dicts(), **points}, report.trial_log)
    trial_log.log_trial(
        run_id, "capacity", params, None, None, 0, None, None, report.trial_log, run_kind="analysis"
    )
    return curve, points


def run_trials(
    report: Report, headline: Mapping[str, Any]
) -> tuple[pl.DataFrame, tr.Deflated | None]:
    """Trial counts by family and the DSR of the headline portfolio (FR-30, AC-H-08)."""
    log = tr.read_log(report.trial_log)
    counts = tr.counts(log)
    n_trials, var = tr.sharpe_variance(log)
    candidates = log.filter(pl.col("strategy_family") == "portfolio")
    match = None
    for row in candidates.iter_rows(named=True):
        params = trial_log_params(row["params_json"])
        if all(params.get(k) == v for k, v in headline.items()):
            match = row
    deflated = None
    if match is not None and n_trials > 1:
        returns = pl.read_parquet(report.trial_log.parent / match["returns_path"])
        deflated = tr.deflated_sharpe(returns["ret"].to_numpy(), n_trials, var)
    summary = counts.with_columns(pl.lit(None, pl.String).alias("note"))
    extra = {
        "portfolio_trials_with_returns": n_trials,
        "var_daily_sharpe": var,
        "headline_run_id": match["run_id"] if match else None,
        "headline_sharpe_daily": deflated.sr if deflated else None,
        "headline_sharpe_annual": deflated.sr * np.sqrt(252.0) if deflated else None,
        "sr0_daily": deflated.sr0 if deflated else None,
        "dsr": deflated.dsr if deflated else None,
    }
    write_table(
        summary.drop("note"), report.tables, "trials", report.command, report.run_id, report.label
    )
    write_table(
        pl.DataFrame([extra]),
        report.tables,
        "trials_dsr",
        report.command,
        report.run_id,
        report.label,
    )
    return summary, deflated


def trial_log_params(params_json: str) -> dict[str, Any]:
    import json

    loaded: dict[str, Any] = json.loads(params_json)
    return loaded


def results_dir(report: Report) -> Path:
    return report.trial_log.parent
