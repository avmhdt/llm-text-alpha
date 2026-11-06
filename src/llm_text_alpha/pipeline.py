"""Pipeline steps behind the CLI: each reads and writes Parquet under the data directory.

Every stage can run alone (section 5). Paths follow section 7b, relative to ``DATA_DIR``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
import structlog

from llm_text_alpha.ids import permno_of
from llm_text_alpha.io import write_parquet_atomic
from llm_text_alpha.settings import Settings

if TYPE_CHECKING:
    from eqpit.store import PitStore

    from llm_text_alpha.evaluation.abnormal import Model


log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class Paths:
    """Where each table lives under the data directory."""

    root: Path

    @classmethod
    def of(cls, settings: Settings) -> Paths:
        """Paths under ``settings.data_dir``."""
        return cls(settings.data_dir)

    @property
    def documents(self) -> Path:
        return self.root / "interim" / "corpus" / "documents"

    def scores(self, scorer: str) -> Path:
        """Folder of one scorer's score shards."""
        return self.root / "processed" / "scores" / f"scorer={scorer}"

    @property
    def panels(self) -> Path:
        return self.root / "processed" / "panels"

    @property
    def signals(self) -> Path:
        return self.root / "processed" / "signals" / "signals.parquet"

    @property
    def abnormal(self) -> Path:
        return self.root / "processed" / "abnormal"

    @property
    def models(self) -> Path:
        return self.root / "models"


def load_documents(paths: Paths, columns: list[str] | None = None) -> pl.DataFrame:
    """The corpus, optionally only some columns."""
    pattern = paths.documents / "**" / "*.parquet"
    if not any(paths.documents.rglob("*.parquet")):
        raise FileNotFoundError(f"no documents under {paths.documents}; run `lta corpus build`")
    return pl.read_parquet(pattern, columns=columns)


def scorable(documents: pl.DataFrame) -> pl.DataFrame:
    """Documents that can become signals: linked, in the universe, and not excluded."""
    return documents.filter(
        pl.col("permno").is_not_null() & pl.col("in_universe") & ~pl.col("excluded")
    )


def load_scores(paths: Paths, scorer: str) -> pl.DataFrame:
    """All shards of one scorer."""
    folder = paths.scores(scorer)
    if not any(folder.glob("*.parquet")):
        raise FileNotFoundError(f"no {scorer} scores under {folder}")
    return pl.read_parquet(folder / "*.parquet")


def load_llm_scores(
    paths: Paths, alias: str, model: str, samples: Sequence[str], masked: bool = False
) -> pl.DataFrame:
    """One LLM's scores across ``samples`` (one row per document), labelled ``alias``.

    A document in several samples has the same cached answer in each, so the first is kept.
    """
    root = paths.scores("llm") / f"model={model}" / f"masked={str(masked).lower()}"
    frames = []
    for sample in samples:
        folder = root / f"sample={sample}"
        if not any(folder.glob("*.parquet")):
            raise FileNotFoundError(f"no {model} scores for sample {sample} under {folder}")
        frames.append(pl.read_parquet(folder / "*.parquet"))
    return (
        pl.concat(frames)
        .unique("doc_id", keep="first", maintain_order=True)
        .with_columns(pl.lit(alias).alias("scorer"))
    )


def build_panels(
    store: PitStore,
    paths: Paths,
    universe_start: date,
    universe_end: date,
    price_start: date,
    price_end: date,
    top_n: int,
) -> dict[str, int]:
    """``universe_monthly``, ``link_table`` and ``prices_daily`` from eqpit (FR-03).

    Prices run from ``price_start`` (the factor estimation start) to ``price_end`` for
    every security in the universe at any month end from ``universe_start`` to
    ``universe_end``.
    """
    from llm_text_alpha.corpus.calendar import xnys
    from llm_text_alpha.data import prices as market

    month_ends = store.month_ends(universe_start, universe_end)
    universe = market.universe_monthly(store, universe_start, universe_end, top_n)
    links = market.link_table(store, month_ends)
    sessions = xnys().sessions(price_start, price_end)
    final, n_sessions = sessions[-1], len(sessions)
    ids = universe["security_id"].unique().sort().to_list()
    daily = market.prices_daily(store, final, n_sessions, ids).with_columns(
        permno_of(pl.col("security_id")).alias("permno")
    )
    write_parquet_atomic(universe, paths.panels / "universe_monthly.parquet")
    write_parquet_atomic(links, paths.panels / "link_table.parquet")
    for (year,), part in daily.with_columns(pl.col("date").dt.year().alias("_y")).group_by(
        "_y", maintain_order=True
    ):
        write_parquet_atomic(
            part.drop("_y"), paths.panels / "prices_daily" / f"year={year}" / "part-0.parquet"
        )
    return {
        "universe_rows": universe.height,
        "links": links.height,
        "price_rows": daily.height,
        "securities": len(ids),
    }


def load_prices(paths: Paths, columns: list[str] | None = None) -> pl.DataFrame:
    """``prices_daily`` across all years."""
    return pl.read_parquet(paths.panels / "prices_daily" / "**" / "*.parquet", columns=columns)


def build_earnings(store: PitStore, paths: Paths, start: date, end: date) -> pl.DataFrame:
    """``earnings`` with SUE and announcement times, plus the book-equity panel (FR-05)."""
    from llm_text_alpha.data import fundamentals

    months = store.month_ends(start, end)
    first = fundamentals.first_reported(store, months, ["eps_diluted", "book_equity"])
    eps = fundamentals.seasonal_sue(fundamentals.quarterly_eps(first))
    docs = load_documents(paths, ["doc_id", "cik", "event_ts_utc", "excluded"]).filter(
        ~pl.col("excluded")
    )
    earnings = fundamentals.link_announcements(eps, docs)
    write_parquet_atomic(earnings, paths.panels / "earnings.parquet")
    write_parquet_atomic(fundamentals.book_equity(first), paths.panels / "book_equity.parquet")
    return earnings


def company_of(paths: Paths) -> pl.DataFrame:
    """``security_id`` to ``company_id``, from the universe panel."""
    universe = pl.read_parquet(paths.panels / "universe_monthly.parquet")
    return universe.select("security_id", "company_id").unique("security_id", keep="last")


def sue_scores(earnings: pl.DataFrame) -> pl.DataFrame:
    """The announced quarter's SUE as a pseudo-score per Item 2.02 document (PEAD check)."""
    from llm_text_alpha.scoring.common import SCORES_SCHEMA

    rows = earnings.filter(pl.col("announce_doc_id").is_not_null() & pl.col("sue").is_not_null())
    return (
        rows.select(
            pl.col("announce_doc_id").alias("doc_id"),
            pl.lit("sue").alias("scorer"),
            pl.lit("srw").alias("model_id"),
            pl.lit(None, pl.String).alias("prompt_version"),
            pl.lit(False).alias("masked"),
            pl.col("sue").alias("score"),
            pl.when(pl.col("sue") > 0)
            .then(pl.lit("positive"))
            .otherwise(pl.lit("negative"))
            .alias("label"),
            pl.lit(None, pl.String).alias("aux_json"),
            pl.lit(None, pl.Float64).alias("latency_ms"),
            pl.lit(0.0).alias("cost_usd"),
            pl.lit(None, pl.Datetime("ns", "UTC")).alias("scored_at"),
        )
        .unique("doc_id", keep="first")
        .cast(SCORES_SCHEMA)
    )


def build_signals(
    paths: Paths,
    scorers: list[str],
    buffer_minutes: int,
    llm_scorers: Mapping[str, Mapping[str, Any]] | None = None,
) -> pl.DataFrame:
    """Gate every scorer's scores, add trailing ``score_z`` and quintiles, write ``signals``.

    The announced quarter's SUE joins as ``sue_announced`` (DQ-08's labelled exception:
    the EPS is in the same 8-K, though the value comes from the later 10-Q).
    """
    from datetime import timedelta

    from llm_text_alpha.corpus.calendar import xnys
    from llm_text_alpha.corpus.gate import apply_gate
    from llm_text_alpha.signals.build import daily_controls, standardize

    calendar = xnys()
    docs = scorable(
        load_documents(
            paths, ["doc_id", "permno", "event_ts_utc", "excluded", "in_universe", "doc_type"]
        )
    )
    earnings = pl.read_parquet(paths.panels / "earnings.parquet")
    llm = dict(llm_scorers or {})

    def frame_of(scorer: str) -> pl.DataFrame:
        if scorer == "sue":
            return sue_scores(earnings)
        if scorer in llm:
            spec = llm[scorer]
            arms = []
            for masked in spec.get("masked", [False]):
                try:
                    arms.append(
                        load_llm_scores(paths, scorer, spec["model"], spec["samples"], bool(masked))
                    )
                except FileNotFoundError as exc:
                    log.warning("llm_scores_missing", scorer=scorer, masked=masked, error=str(exc))
            if not arms:
                raise FileNotFoundError(f"no scores for {scorer} ({spec['model']})")
            return pl.concat(arms)
        return load_scores(paths, scorer)

    frames = [frame_of(s) for s in scorers]
    scores = pl.concat(
        [f.select("doc_id", "scorer", "model_id", "masked", "score") for f in frames]
    )
    prices = load_prices(paths, ["security_id", "permno", "date", "ret", "mcap"]).join(
        company_of(paths), on="security_id", how="left"
    )
    book = pl.read_parquet(paths.panels / "book_equity.parquet")
    controls = daily_controls(
        prices, book, earnings.select("company_id", "eps_available_at", "sue")
    )
    gated = apply_gate(scores, docs, calendar, controls, timedelta(minutes=buffer_minutes))
    signals = standardize(gated, calendar).join(
        earnings.filter(pl.col("announce_doc_id").is_not_null())
        .select(pl.col("announce_doc_id").alias("doc_id"), pl.col("sue").alias("sue_announced"))
        .unique("doc_id"),
        on="doc_id",
        how="left",
    )
    write_parquet_atomic(signals, paths.signals)
    return signals


def run_abnormal(
    paths: Paths, model: Model, est: tuple[int, int], window: tuple[int, int]
) -> dict[str, int]:
    """Abnormal returns for every distinct gated event; writes ``ar``, ``events``, ``residuals``."""
    from llm_text_alpha.corpus.calendar import xnys
    from llm_text_alpha.evaluation.abnormal import abnormal_returns

    signals = pl.read_parquet(paths.signals)
    prices = load_prices(paths, ["permno", "date", "open", "ret", "ret_open_close"])
    factors = pl.read_parquet(paths.panels / "factors_daily.parquet")
    result = abnormal_returns(
        signals,
        prices,
        factors,
        model,
        est,
        window,
        xnys(),
        keep_residuals=True,
    )
    folder = paths.abnormal / f"model={model}"
    write_parquet_atomic(result.ar, folder / "ar.parquet")
    write_parquet_atomic(result.events, folder / "events.parquet")
    if result.residuals is not None:
        write_parquet_atomic(result.residuals, folder / "residuals.parquet")
    return {"events": result.events.height, "kept": result.events["drop_reason"].null_count()}


DOC_TYPE = "8k_item202"
REGRESSION_SPECS: dict[str, list[str]] = {
    "score_only": ["score_z"],
    "controls_no_sue": ["score_z", "size_log", "mom_12_2"],
    "controls": ["score_z", "size_log", "mom_12_2", "sue"],
    "controls_bm": ["score_z", "size_log", "mom_12_2", "sue", "bm"],
    "controls_sue_announced": ["score_z", "size_log", "mom_12_2", "sue_announced"],
    "sue_orth": ["score_orth", "size_log", "mom_12_2"],
}
"""``sue_orth`` reports the SUE-orthogonalized score's coefficient in the ``score_z`` columns."""


@dataclass(frozen=True)
class Report:
    """Where committed outputs go and how they are labelled."""

    tables: Path
    figures: Path
    trial_log: Path
    command: str
    run_id: str
    label: str = ""


def _one_scorer(signals: pl.DataFrame, scorer: str) -> pl.DataFrame:
    chosen = signals.filter((pl.col("scorer") == scorer) & ~pl.col("masked"))
    models = chosen["model_id"].unique().to_list()
    if len(models) != 1:
        raise ValueError(f"expected one model for {scorer}, found {models}")
    return chosen


def run_event_study(
    paths: Paths,
    scorer: str,
    model: str,
    report: Report,
    seed: int,
    reps: int,
    horizons: list[int],
) -> pl.DataFrame:
    """Event study for one scorer: table, figure and one trial-log row (FR-16, FR-23)."""
    from llm_text_alpha.evaluation.event_study import event_study
    from llm_text_alpha.reporting.figures import event_study_figure
    from llm_text_alpha.reporting.tables import write_table
    from llm_text_alpha.tracking import trial_log

    folder = paths.abnormal / f"model={model}"
    signals = _one_scorer(pl.read_parquet(paths.signals), scorer)
    events = pl.read_parquet(folder / "events.parquet")
    result = event_study(
        pl.read_parquet(folder / "ar.parquet"),
        signals,
        sigma=events,
        residuals=pl.read_parquet(folder / "residuals.parquet"),
        seed=seed,
        reps=reps,
        horizons=horizons,
    )
    name = f"event_study_{scorer}_{DOC_TYPE}"
    write_table(result.summary, report.tables, name, report.command, report.run_id, report.label)
    model_id = signals["model_id"][0]
    caption = (
        f"{model_id}; {result.n_events} events, {DOC_TYPE}, AR model {model}; "
        f"quintiles from trailing 252-session breakpoints"
        + (f"; {report.label}" if report.label else "")
    )
    event_study_figure(
        result.path, f"Event study: {scorer}", caption, report.figures / f"{name}.png"
    )
    params = {
        "scorer": scorer,
        "model_id": model_id,
        "masked": False,
        "doc_type": DOC_TYPE,
        "ar_model": model,
        "window": [-10, 60],
        "bootstrap_reps": reps,
        "seed": seed,
    }
    run_id = trial_log.new_run_id()
    spread = result.summary.filter(pl.col("group") == "Q5-Q1")
    trial_log.record_run(
        run_id,
        params,
        {
            "n_events": result.n_events,
            "kp_correlation": result.kp_correlation,
            "spread": spread.to_dicts(),
        },
        report.trial_log,
    )
    trial_log.log_trial(
        run_id,
        "event_study",
        params,
        None,
        None,
        0,
        None,
        None,
        report.trial_log,
        run_kind="analysis",
    )
    return result.summary


def sue_orthogonal(panel: pl.DataFrame) -> pl.DataFrame:
    """``score_orth``: the residual of ``score_z`` on ``sue`` within each entry month (AC-H-05)."""
    month = pl.col("entry_date").dt.truncate("1mo")
    valid = pl.col("score_z").is_not_null() & pl.col("sue").is_not_null()
    z = pl.when(valid).then(pl.col("score_z"))
    s = pl.when(valid).then(pl.col("sue"))
    dz = z - z.mean().over(month)
    ds = s - s.mean().over(month)
    beta = (dz * ds).sum().over(month) / (ds**2).sum().over(month)
    return panel.with_columns((dz - beta * ds).alias("score_orth"))


def regression_panel(paths: Paths, scorer: str, model: str, horizons: list[int]) -> pl.DataFrame:
    """Signals of one scorer joined to their CAR[0,k] (the panel FR-17 regresses)."""
    from llm_text_alpha.evaluation.abnormal import cumulative

    folder = paths.abnormal / f"model={model}"
    events = pl.read_parquet(folder / "events.parquet").filter(pl.col("drop_reason").is_null())
    ar = pl.read_parquet(folder / "ar.parquet").join(
        events.select("doc_id"), on="doc_id", how="semi"
    )
    cars = cumulative(ar, horizons)
    return (
        _one_scorer(pl.read_parquet(paths.signals), scorer)
        .join(cars, on=["doc_id", "permno"], how="inner")
        .filter(pl.col("score_z").is_not_null())
    )


def run_regressions(
    paths: Paths,
    scorers: list[str],
    model: str,
    horizons: list[int],
    report: Report,
    methods: list[str],
    window: tuple[str, date | None] = ("all", None),
    table_name: str = "regressions",
) -> pl.DataFrame:
    """CAR[0,k] on score_z with controls, two-way clustered and Fama-MacBeth (FR-17).

    ``window`` keeps events published on or after its date. A specification already in the
    trial log with the same parameters is not logged again: rerunning it on the same data is
    not a new trial.
    """
    from llm_text_alpha.evaluation.regressions import cross_sectional_regression
    from llm_text_alpha.reporting.tables import write_table
    from llm_text_alpha.tracking import trial_log

    rows: list[dict[str, object]] = []
    seen = {
        family: trial_log.logged_params(report.trial_log, family)
        for family in ("cs_regression", "fm_regression")
    }
    for scorer in scorers:
        panel = sue_orthogonal(regression_panel(paths, scorer, model, horizons))
        if window[1] is not None:
            panel = panel.filter(pl.col("event_ts_utc").dt.date() >= window[1])
        for k in horizons:
            for spec, regressors in REGRESSION_SPECS.items():
                for method in methods:
                    fit = cross_sectional_regression(panel, f"car_{k}", regressors, method)  # type: ignore[arg-type]
                    term = regressors[0]
                    coef = fit.to_frame().filter(pl.col("term") == term).row(0, named=True)
                    params: dict[str, object] = {
                        "scorer": scorer,
                        "horizon": k,
                        "spec": spec,
                        "se": method,
                        "ar_model": model,
                        "regressors": regressors,
                        "doc_type": DOC_TYPE,
                    }
                    if window[1] is not None:
                        params["window"] = window[0]
                        params["window_start"] = str(window[1])
                    family = "cs_regression" if method == "twoway" else "fm_regression"
                    rows.append(
                        {
                            "scorer": scorer,
                            "window": window[0],
                            "horizon": f"CAR[0,{k}]",
                            "spec": spec,
                            "se": method,
                            "coef_score_z": coef["coef"],
                            "se_score_z": coef["se"],
                            "t_score_z": coef["t"],
                            "nobs": fit.nobs,
                            "periods": fit.n_periods,
                            "r2": fit.r2,
                        }
                    )
                    if trial_log.canonical_json(params) in seen[family]:
                        continue
                    run_id = trial_log.new_run_id()
                    trial_log.record_run(
                        run_id,
                        params,
                        {
                            "terms": fit.to_frame().to_dicts(),
                            "nobs": fit.nobs,
                            "n_periods": fit.n_periods,
                            "r2": fit.r2,
                        },
                        report.trial_log,
                    )
                    trial_log.log_trial(
                        run_id,
                        family,
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
    write_table(table, report.tables, table_name, report.command, report.run_id, report.label)
    return table


def sue_correlation(
    paths: Paths, scorers: list[str], window: tuple[str, date | None]
) -> pl.DataFrame:
    """Spearman correlation of each scorer's ``score_z`` with SUE and announced SUE (AC-H-05)."""
    signals = pl.read_parquet(paths.signals)
    rows: list[dict[str, object]] = []
    for scorer in scorers:
        part = _one_scorer(signals, scorer).filter(pl.col("score_z").is_not_null())
        if window[1] is not None:
            part = part.filter(pl.col("event_ts_utc").dt.date() >= window[1])
        for control in ("sue", "sue_announced"):
            pair = part.select("score_z", control).drop_nulls()
            rho = (
                pair.select(pl.corr("score_z", control, method="spearman")).item()
                if pair.height > 2
                else None
            )
            rows.append(
                {
                    "scorer": scorer,
                    "window": window[0],
                    "with": control,
                    "n": pair.height,
                    "spearman": rho,
                }
            )
    return pl.DataFrame(rows)


def run_pead_check(paths: Paths, model: str, report: Report) -> pl.DataFrame:
    """PEAD sanity check (section 11): CAR[+1,+60] by SUE quintile, reported not asserted.

    Bernard and Thomas (1989) find high-minus-low SUE drift is positive. ``t`` is the
    Welch statistic of the difference between the top and bottom quintile means.
    """
    from llm_text_alpha.reporting.tables import write_table

    folder = paths.abnormal / f"model={model}"
    events = pl.read_parquet(folder / "events.parquet").filter(pl.col("drop_reason").is_null())
    car = (
        pl.read_parquet(folder / "ar.parquet")
        .join(events.select("doc_id"), on="doc_id", how="semi")
        .filter(pl.col("rel_day").is_between(1, 60))
        .group_by("doc_id")
        .agg(pl.col("ar").fill_null(0.0).sort().sum().alias("car_1_60"))
    )
    sue = _one_scorer(pl.read_parquet(paths.signals), "sue").filter(pl.col("quintile") > 0)
    data = sue.join(car, on="doc_id", how="inner")
    table = (
        data.group_by("quintile")
        .agg(
            pl.len().alias("n"),
            pl.col("car_1_60").sort().mean().alias("car_1_60"),
            (pl.col("car_1_60").sort().std() / pl.len().sqrt()).alias("se"),
        )
        .sort("quintile")
        .with_columns(pl.format("Q{}", pl.col("quintile")).alias("group"))
        .select("group", "n", "car_1_60", "se")
    )
    top, bottom = table.row(-1, named=True), table.row(0, named=True)
    diff = top["car_1_60"] - bottom["car_1_60"]
    se = (top["se"] ** 2 + bottom["se"] ** 2) ** 0.5
    table = pl.concat(
        [
            table,
            pl.DataFrame(
                {"group": ["Q5-Q1"], "n": [top["n"] + bottom["n"]], "car_1_60": [diff], "se": [se]},
                schema=table.schema,
            ),
        ]
    ).with_columns((pl.col("car_1_60") / pl.col("se")).alias("t"))
    write_table(table, report.tables, "pead_sanity", report.command, report.run_id, report.label)
    return table
