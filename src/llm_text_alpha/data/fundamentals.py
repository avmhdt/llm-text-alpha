"""Quarterly EPS, SUE and book equity, point in time (FR-05, DQ-08, DQ-16).

Values are read through eqpit's ``PitStore`` at every month end. A period's value is its
first report: the version in the earliest month-end snapshot that holds the period.
``PitStore`` returns only the latest version known at a date, so a restatement inside the
same month as the first report would replace it; at monthly resolution that is the closest
this API allows to first-reported values.

SUE is the seasonal random walk: ``(EPS_q - EPS_{q-4}) / std`` of the previous eight such
differences, with at least six.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from eqpit.store import PitStore

SUE_WINDOW = 8
SUE_MIN_OBS = 6
TS = pl.Datetime("ns", "UTC")


def first_reported(
    store: PitStore, month_ends: list[date], items: list[str], history_periods: int = 3
) -> pl.DataFrame:
    """Each (company, item, period end, period type) at the first month end that knows it."""
    frames = [
        store.fundamentals(m, items, history_periods=history_periods)
        .select(
            "company_id",
            "cik",
            "item",
            "period_end",
            "period_type",
            "value",
            "available_at",
            "accession",
            "form",
        )
        .with_columns(pl.lit(m).alias("first_seen"))
        for m in month_ends
    ]
    key = ["company_id", "item", "period_end", "period_type"]
    return (
        pl.concat(frames)
        .sort("first_seen", "available_at")
        .unique(subset=key, keep="first", maintain_order=True)
        .sort(*key)
    )


def quarterly_eps(first: pl.DataFrame) -> pl.DataFrame:
    """Three-month diluted EPS per company and fiscal quarter end."""
    return (
        first.filter((pl.col("item") == "eps_diluted") & (pl.col("period_type") == "3m"))
        .select(
            "company_id",
            "cik",
            pl.col("period_end").alias("fqtr_end"),
            pl.col("value").alias("eps"),
            pl.col("available_at").alias("eps_available_at"),
            pl.col("form").alias("eps_form"),
        )
        .sort("company_id", "fqtr_end")
    )


def seasonal_sue(
    eps: pl.DataFrame, window: int = SUE_WINDOW, min_obs: int = SUE_MIN_OBS
) -> pl.DataFrame:
    """Add ``diff4`` and ``sue`` per company.

    ``diff4`` needs the quarter ending 350 to 380 days earlier; otherwise it is null. The
    scale is the standard deviation (ddof 1) of the ``window`` previous differences, with at
    least ``min_obs`` present.
    """
    ordered = eps.sort("company_id", "fqtr_end")
    lagged = ordered.join(
        ordered.select(
            "company_id",
            pl.col("fqtr_end").alias("_lag_end"),
            pl.col("eps").alias("_lag_eps"),
        ),
        on="company_id",
        how="left",
    ).filter((pl.col("fqtr_end") - pl.col("_lag_end")).dt.total_days().is_between(350, 380))
    with_diff = (
        ordered.join(
            lagged.select("company_id", "fqtr_end", "_lag_eps").unique(
                ["company_id", "fqtr_end"], keep="first"
            ),
            on=["company_id", "fqtr_end"],
            how="left",
        )
        .with_columns((pl.col("eps") - pl.col("_lag_eps")).alias("diff4"))
        .drop("_lag_eps")
        .sort("company_id", "fqtr_end")
    )
    scale = (
        pl.col("diff4")
        .shift(1)
        .rolling_std(window_size=window, min_samples=min_obs)
        .over("company_id")
    )
    return with_diff.with_columns(scale.alias("sue_scale")).with_columns(
        pl.when(pl.col("sue_scale") > 0).then(pl.col("diff4") / pl.col("sue_scale")).alias("sue"),
        pl.lit("srw").alias("sue_method"),
    )


def link_announcements(earnings: pl.DataFrame, documents: pl.DataFrame) -> pl.DataFrame:
    """Set ``announce_ts_utc`` to the quarter's Item 2.02 8-K, else the EPS filing's time.

    The 8-K is the company's last Item 2.02 document accepted after the quarter end and no
    later than one day after the 10-Q or 10-K that carries the EPS. ``announce_source`` is
    ``8k_item202`` or ``eps_filing``.
    """
    docs = documents.filter(pl.col("cik").is_not_null()).select(
        "cik", pl.col("event_ts_utc").alias("_doc_ts"), pl.col("doc_id").alias("announce_doc_id")
    )
    candidates = earnings.join(docs, on="cik", how="inner").filter(
        (pl.col("_doc_ts").dt.date() > pl.col("fqtr_end"))
        & (pl.col("_doc_ts") <= pl.col("eps_available_at") + pl.duration(days=1))
        & (pl.col("_doc_ts").dt.date() <= pl.col("fqtr_end") + timedelta(days=150))
    )
    chosen = (
        candidates.sort("_doc_ts")
        .group_by("company_id", "fqtr_end", maintain_order=True)
        .agg(pl.col("_doc_ts").last(), pl.col("announce_doc_id").last())
    )
    return (
        earnings.join(chosen, on=["company_id", "fqtr_end"], how="left")
        .with_columns(
            pl.coalesce("_doc_ts", "eps_available_at").alias("announce_ts_utc"),
            pl.when(pl.col("_doc_ts").is_not_null())
            .then(pl.lit("8k_item202"))
            .otherwise(pl.lit("eps_filing"))
            .alias("announce_source"),
        )
        .drop("_doc_ts")
        .sort("company_id", "fqtr_end")
    )


def book_equity(first: pl.DataFrame) -> pl.DataFrame:
    """Book equity per company and period end, with the time it became public."""
    return (
        first.filter(pl.col("item") == "book_equity")
        .select(
            "company_id",
            "cik",
            "period_end",
            pl.col("value").alias("book_equity"),
            "available_at",
        )
        .sort("company_id", "available_at")
    )
