"""Abnormal returns around each gated event (FR-15, DQ-09, DQ-14).

``rel_day`` counts sessions from the entry date (0). The entry-day return is open to close,
since the trade enters at the open; every other day is close to close. Expected returns:

- ``ff5umd``: ``rf + alpha + beta' f`` with FF5 and momentum, fitted by OLS on excess returns
  over sessions -250 to -30. Fewer than 120 valid returns falls back to ``market`` and is
  flagged.
- ``market``: ``rf + mktrf``.
- ``raw``: zero, so ``ar`` is the return.

Events whose entry session has no open price, or whose event window misses more than 20%
of its returns, are dropped with a reason code.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray

from llm_text_alpha.corpus.calendar import TradingCalendar

Model = Literal["ff5umd", "market", "raw"]
FACTORS = ("mktrf", "smb", "hml", "rmw", "cma", "umd")
MIN_ESTIMATION_OBS = 120
MAX_MISSING_SHARE = 0.20


@dataclass(frozen=True)
class AbnormalResult:
    """``ar``: one row per event and day. ``events``: per-event fit details and drop reasons.

    ``residuals`` (when requested) holds each kept event's estimation residuals divided by
    its ``sigma``, keyed by calendar session index, for the Kolari-Pynnonen correlation.
    """

    ar: pl.DataFrame
    events: pl.DataFrame
    residuals: pl.DataFrame | None = None


def _grid(
    calendar: TradingCalendar, frame: pl.DataFrame, column: str
) -> dict[int, NDArray[np.float64]]:
    """Per permno, ``column`` on the full session grid (NaN where missing)."""
    index = np.searchsorted(calendar.days, frame["date"].cast(pl.Int64).to_numpy())
    values = frame[column].cast(pl.Float64).fill_null(np.nan).to_numpy()
    grids: dict[int, NDArray[np.float64]] = {}
    permnos = frame["permno"].to_numpy()
    order = np.argsort(permnos, kind="stable")
    bounds = np.flatnonzero(np.diff(permnos[order])) + 1
    for chunk in np.split(order, bounds):
        if len(chunk) == 0:
            continue
        grid = np.full(len(calendar.days), np.nan)
        grid[index[chunk]] = values[chunk]
        grids[int(permnos[chunk[0]])] = grid
    return grids


def _factor_grid(calendar: TradingCalendar, factors: pl.DataFrame) -> NDArray[np.float64]:
    index = np.searchsorted(calendar.days, factors["date"].cast(pl.Int64).to_numpy())
    known = (
        calendar.days[np.minimum(index, len(calendar.days) - 1)]
        == factors["date"].cast(pl.Int64).to_numpy()
    )
    grid = np.full((len(calendar.days), len(FACTORS) + 1), np.nan)
    columns = factors.select(*FACTORS, "rf").to_numpy()
    grid[index[known]] = columns[known]
    return grid


def fit_factor_model(
    excess: NDArray[np.float64], factors: NDArray[np.float64]
) -> tuple[NDArray[np.float64], float, int]:
    """OLS of excess returns on a constant and factors; rows with any NaN are skipped.

    Returns the coefficients (constant first), the residual standard deviation (ddof = k)
    and the number of observations.
    """
    valid = np.isfinite(excess) & np.isfinite(factors).all(axis=1)
    y = excess[valid]
    x = np.column_stack([np.ones(int(valid.sum())), factors[valid]])
    if len(y) <= x.shape[1]:
        return np.full(x.shape[1], np.nan), float("nan"), len(y)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    sigma = float(np.sqrt(resid @ resid / (len(y) - x.shape[1])))
    return beta, sigma, len(y)


def abnormal_returns(
    signals: pl.DataFrame,
    prices: pl.DataFrame,
    factors: pl.DataFrame,
    model: Model,
    est_window: tuple[int, int],
    event_window: tuple[int, int],
    calendar: TradingCalendar,
    keep_residuals: bool = False,
) -> AbnormalResult:
    """Abnormal returns for each distinct event (``doc_id``, ``permno``, ``entry_date``).

    Args:
        signals: gated signals; scorers sharing an event share its returns.
        prices: ``permno, date, open, ret, ret_open_close``.
        factors: ``factors_daily``.
        model: expected-return model.
        est_window: estimation sessions relative to entry, inclusive, e.g. (-250, -30).
        event_window: event sessions relative to entry, inclusive, e.g. (-10, 60).
        calendar: trading calendar.
        keep_residuals: also return standardized estimation residuals.
    """
    events = signals.select("doc_id", "permno", "entry_date").unique().sort("entry_date", "doc_id")
    rets = _grid(calendar, prices, "ret")
    intraday = _grid(calendar, prices, "ret_open_close")
    opens = _grid(calendar, prices, "open")
    fgrid = _factor_grid(calendar, factors)
    entry_index = np.searchsorted(calendar.days, events["entry_date"].cast(pl.Int64).to_numpy())
    rel = np.arange(event_window[0], event_window[1] + 1)
    n_days = len(rel)
    rows_doc: list[str] = []
    rows_permno: list[int] = []
    rows_index: list[NDArray[np.int64]] = []
    rows_ret: list[NDArray[np.float64]] = []
    rows_exp: list[NDArray[np.float64]] = []
    info: list[dict[str, object]] = []
    resid_doc: list[str] = []
    resid_session: list[NDArray[np.int64]] = []
    resid_value: list[NDArray[np.float64]] = []
    for doc, permno, i0 in zip(events["doc_id"], events["permno"], entry_index, strict=True):
        detail: dict[str, object] = {
            "doc_id": doc,
            "permno": permno,
            "model_used": model,
            "n_estimation": 0,
            "sigma": None,
            "drop_reason": None,
        }
        info.append(detail)
        ret = rets.get(int(permno))
        days = i0 + rel
        if ret is None or days[0] < 0 or days[-1] >= len(calendar.days):
            detail["drop_reason"] = "no_prices" if ret is None else "window_outside_calendar"
            continue
        if not np.isfinite(opens[int(permno)][i0]):
            detail["drop_reason"] = "no_entry_open"
            continue
        window = ret[days].copy()
        window[rel == 0] = intraday[int(permno)][i0]
        if np.isnan(window).mean() > MAX_MISSING_SHARE:
            detail["drop_reason"] = "window_missing"
            continue
        f_window = fgrid[days]
        est = i0 + np.arange(est_window[0], est_window[1] + 1)
        est = est[est >= 0]
        f_est = fgrid[est]
        used: str = model
        beta: NDArray[np.float64] | None = None
        if model == "ff5umd":
            beta, _, n_est = fit_factor_model(ret[est] - f_est[:, -1], f_est[:, :-1])
            detail["n_estimation"] = n_est
            if n_est < MIN_ESTIMATION_OBS or not np.isfinite(beta).all():
                beta = None
                used = "market"
        if model == "raw":
            expected = np.zeros(n_days)
            expected_est = np.zeros(len(est))
        elif beta is None:
            expected = f_window[:, -1] + f_window[:, 0]
            expected_est = f_est[:, -1] + f_est[:, 0]
        else:
            expected = f_window[:, -1] + beta[0] + f_window[:, :-1] @ beta[1:]
            expected_est = f_est[:, -1] + beta[0] + f_est[:, :-1] @ beta[1:]
        resid_all = ret[est] - expected_est
        finite = np.isfinite(resid_all)
        resid = resid_all[finite]
        k = len(FACTORS) + 1 if beta is not None else 0
        detail["n_estimation"] = len(resid)
        if len(resid) > k + 1:
            sigma = float(np.sqrt(resid @ resid / (len(resid) - k)))
            detail["sigma"] = sigma
            if keep_residuals and sigma > 0:
                resid_doc.append(doc)
                resid_session.append(est[finite])
                resid_value.append(resid / sigma)
        detail["model_used"] = used
        rows_doc.append(doc)
        rows_permno.append(int(permno))
        rows_index.append(days)
        rows_ret.append(window)
        rows_exp.append(expected)
    if rows_doc:
        index = np.concatenate(rows_index)
        ret_all = np.concatenate(rows_ret)
        exp_all = np.concatenate(rows_exp)
        ar = pl.DataFrame(
            {
                "doc_id": np.repeat(rows_doc, n_days),
                "permno": np.repeat(rows_permno, n_days),
                "rel_day": np.tile(rel, len(rows_doc)).astype(np.int32),
                "date": calendar.days[index].astype("datetime64[D]"),
                "ret": ret_all,
                "exp_ret": exp_all,
                "ar": ret_all - exp_all,
            }
        ).with_columns(
            pl.col("date").cast(pl.Date),
            pl.col("ret").fill_nan(None),
            pl.col("exp_ret").fill_nan(None),
            pl.col("ar").fill_nan(None),
            pl.lit(model).alias("model"),
        )
    else:
        ar = pl.DataFrame(
            schema={
                "doc_id": pl.String,
                "permno": pl.Int64,
                "rel_day": pl.Int32,
                "date": pl.Date,
                "ret": pl.Float64,
                "exp_ret": pl.Float64,
                "ar": pl.Float64,
                "model": pl.String,
            }
        )
    details = pl.DataFrame(
        info,
        schema={
            "doc_id": pl.String,
            "permno": pl.Int64,
            "model_used": pl.String,
            "n_estimation": pl.Int64,
            "sigma": pl.Float64,
            "drop_reason": pl.String,
        },
    )
    residuals = None
    if keep_residuals:
        residuals = pl.DataFrame(
            {
                "doc_id": np.repeat(resid_doc, [len(v) for v in resid_value]).astype(str)
                if resid_doc
                else np.array([], dtype=str),
                "session": np.concatenate(resid_session)
                if resid_session
                else np.array([], np.int64),
                "std_resid": np.concatenate(resid_value) if resid_value else np.array([]),
            },
            schema={"doc_id": pl.String, "session": pl.Int64, "std_resid": pl.Float64},
        )
    return AbnormalResult(ar=ar, events=details, residuals=residuals)


def drop_counts(events: pl.DataFrame) -> pl.DataFrame:
    """DQ-14 reason counts, with ``kept`` for events that stayed."""
    return (
        events.with_columns(pl.col("drop_reason").fill_null("kept"))
        .group_by("drop_reason")
        .len("events")
        .sort("drop_reason")
    )


def cumulative(ar: pl.DataFrame, horizons: Sequence[int], start: int = 0) -> pl.DataFrame:
    """``car_<k>`` = sum of ``ar`` over rel_day ``start`` to ``k`` per event; nulls count as 0."""
    exprs = [
        pl.col("ar")
        .filter(pl.col("rel_day").is_between(start, k))
        .fill_null(0.0)
        .sum()
        .alias(f"car_{k}")
        for k in horizons
    ]
    return ar.group_by("doc_id", "permno").agg(*exprs).sort("doc_id")
