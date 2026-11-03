"""Event study: CAAR by trailing score quintile, with 95% bands (FR-16).

CARs cumulate from the first day of the event window. Bands:

- ``t``: cross-sectional ``CAAR +/- 1.96 sd(CAR) / sqrt(N)``.
- ``bmp``: Boehmer, Musumeci and Poulsen (1991). Each event's CAR is standardized by its
  estimation-window residual deviation times ``sqrt(L)``; the test statistic is the mean
  over its cross-sectional error. The band converts that error back to return units with
  the mean scale ``sigma_i sqrt(L)``.
- ``bootstrap``: firms resampled with replacement (``reps`` draws), percentile interval.

The Kolari-Pynnonen (2010) adjustment scales the BMP statistic by
``sqrt((1 - r) / (1 + (N - 1) r))``, with ``r`` the mean pairwise correlation of
standardized estimation residuals over sampled event pairs whose windows overlap.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray

Z95 = 1.959963984540054
SPREAD = "Q5-Q1"


@dataclass(frozen=True)
class EventStudyResult:
    """``path``: CAAR and bands by group and day. ``summary``: CAR[0,k] tests by group."""

    path: pl.DataFrame
    summary: pl.DataFrame
    n_events: int
    kp_correlation: float | None


def _paths(ar: pl.DataFrame, days: NDArray[np.int64]) -> tuple[list[str], NDArray[np.float64]]:
    wide = (
        ar.select("doc_id", "rel_day", pl.col("ar").fill_null(0.0))
        .pivot(on="rel_day", index="doc_id", values="ar", sort_columns=False)
        .sort("doc_id")
    )
    columns = [str(d) for d in days]
    for name in columns:
        if name not in wide.columns:
            wide = wide.with_columns(pl.lit(0.0).alias(name))
    matrix = wide.select(columns).fill_null(0.0).to_numpy().astype(np.float64)
    return wide["doc_id"].to_list(), np.cumsum(matrix, axis=1)


def _bootstrap(
    car: NDArray[np.float64],
    firm: NDArray[np.int64],
    group: NDArray[np.int64],
    n_groups: int,
    reps: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Bootstrap CAAR paths: shape (reps, n_groups + 1, days); the last group is Q5 - Q1."""
    firms, firm_index = np.unique(firm, return_inverse=True)
    sums = np.zeros((n_groups, len(firms), car.shape[1]))
    counts = np.zeros((n_groups, len(firms)))
    for g in range(n_groups):
        mask = group == g + 1
        np.add.at(sums[g], firm_index[mask], car[mask])
        np.add.at(counts[g], firm_index[mask], 1.0)
    weights = rng.multinomial(len(firms), np.full(len(firms), 1.0 / len(firms)), size=reps)
    out = np.empty((reps, n_groups + 1, car.shape[1]))
    for g in range(n_groups):
        total = weights @ counts[g]
        out[:, g] = (weights @ sums[g]) / np.where(total > 0, total, np.nan)[:, None]
    out[:, n_groups] = out[:, n_groups - 1] - out[:, 0]
    return out


def kp_correlation(
    residuals: pl.DataFrame, max_pairs: int = 2000, min_overlap: int = 60, seed: int = 0
) -> float | None:
    """Mean correlation of standardized estimation residuals over random event pairs.

    ``residuals`` has ``doc_id, session, std_resid``. Pairs need ``min_overlap`` common
    sessions. None when no pair qualifies.
    """
    by_doc = {
        doc: (frame["session"].to_numpy(), frame["std_resid"].to_numpy())
        for (doc,), frame in residuals.group_by("doc_id")
    }
    docs = sorted(by_doc)
    if len(docs) < 2:
        return None
    rng = np.random.default_rng(seed)
    found: list[float] = []
    for _ in range(max_pairs * 5):
        if len(found) >= max_pairs:
            break
        a, b = rng.choice(len(docs), size=2, replace=False)
        sa, ra = by_doc[docs[a]]
        sb, rb = by_doc[docs[b]]
        common, ia, ib = np.intersect1d(sa, sb, return_indices=True)
        if len(common) < min_overlap:
            continue
        x, y = ra[ia], rb[ib]
        if x.std() > 0 and y.std() > 0:
            found.append(float(np.corrcoef(x, y)[0, 1]))
    return float(np.mean(found)) if found else None


def event_study(
    ar: pl.DataFrame,
    signals: pl.DataFrame,
    n_quantiles: int = 5,
    bands: Sequence[str] = ("t", "bmp", "bootstrap"),
    seed: int = 0,
    sigma: pl.DataFrame | None = None,
    residuals: pl.DataFrame | None = None,
    reps: int = 1000,
    horizons: Sequence[int] = (1, 5, 20, 60),
) -> EventStudyResult:
    """CAAR by quantile and top minus bottom, for one scorer's signals.

    Args:
        ar: ``doc_id, permno, rel_day, ar``.
        signals: ``doc_id, permno, quintile`` for one scorer; quintile 0 (warm-up) is left
            out.
        n_quantiles: number of score groups.
        bands: which bands to compute.
        seed: bootstrap seed.
        sigma: ``doc_id, sigma`` from the abnormal-return fit; needed for ``bmp``.
        residuals: ``doc_id, session, std_resid``; enables the Kolari-Pynnonen adjustment.
        reps: bootstrap draws.
        horizons: k for the CAR[0,k] summary.
    """
    groups = signals.filter(pl.col("quintile") > 0).select("doc_id", "permno", "quintile")
    data = ar.join(groups.select("doc_id"), on="doc_id", how="semi")
    days = np.sort(data["rel_day"].unique().to_numpy()).astype(np.int64)
    docs, car = _paths(data, days)
    meta = pl.DataFrame({"doc_id": docs}).join(groups, on="doc_id", how="left")
    if sigma is not None:
        meta = meta.join(sigma.select("doc_id", "sigma"), on="doc_id", how="left")
    group = meta["quintile"].to_numpy().astype(np.int64)
    firm = meta["permno"].to_numpy().astype(np.int64)
    labels = [f"Q{q}" for q in range(1, n_quantiles + 1)] + [SPREAD]
    zero = int(np.searchsorted(days, 0))
    base = car[:, zero - 1] if zero > 0 else np.zeros(len(docs))
    length = np.arange(1, len(days) + 1, dtype=np.float64)
    boot = (
        _bootstrap(car, firm, group, n_quantiles, reps, np.random.default_rng(seed))
        if "bootstrap" in bands and len(docs)
        else None
    )
    r_kp = kp_correlation(residuals, seed=seed) if residuals is not None else None
    path_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for g, label in enumerate(labels):
        if label == SPREAD:
            top, bottom = group == n_quantiles, group == 1
            mean = car[top].mean(axis=0) - car[bottom].mean(axis=0)
            se = np.sqrt(
                car[top].var(axis=0, ddof=1) / top.sum()
                + car[bottom].var(axis=0, ddof=1) / bottom.sum()
            )
            n = int(top.sum() + bottom.sum())
            members = top | bottom
        else:
            members = group == g + 1
            n = int(members.sum())
            mean = car[members].mean(axis=0) if n else np.full(len(days), np.nan)
            se = (
                car[members].std(axis=0, ddof=1) / np.sqrt(n)
                if n > 1
                else np.full(len(days), np.nan)
            )
        bmp_se = np.full(len(days), np.nan)
        t_bmp = np.full(len(days), np.nan)
        if "bmp" in bands and sigma is not None and label != SPREAD and n > 1:
            s = meta["sigma"].to_numpy().astype(np.float64)[members]
            scale = s[:, None] * np.sqrt(length)[None, :]
            csar = car[members] / scale
            ok = np.isfinite(csar).all(axis=1)
            csar, scale = csar[ok], scale[ok]
            if len(csar) > 1:
                t_bmp = csar.mean(axis=0) * np.sqrt(len(csar)) / csar.std(axis=0, ddof=1)
                bmp_se = csar.std(axis=0, ddof=1) / np.sqrt(len(csar)) * scale.mean(axis=0)
        lo_b = hi_b = np.full(len(days), np.nan)
        if boot is not None:
            lo_b, hi_b = np.nanpercentile(boot[:, g], [2.5, 97.5], axis=0)
        for j, day in enumerate(days):
            path_rows.append(
                {
                    "group": label,
                    "rel_day": int(day),
                    "n": n,
                    "caar": float(mean[j]),
                    "lo_t": float(mean[j] - Z95 * se[j]),
                    "hi_t": float(mean[j] + Z95 * se[j]),
                    "lo_bmp": float(mean[j] - Z95 * bmp_se[j]),
                    "hi_bmp": float(mean[j] + Z95 * bmp_se[j]),
                    "t_bmp": float(t_bmp[j]),
                    "lo_boot": float(lo_b[j]),
                    "hi_boot": float(hi_b[j]),
                }
            )
        for k in horizons:
            if k not in set(days.tolist()):
                continue
            j = int(np.searchsorted(days, k))
            window = car[:, j] - base
            if label == SPREAD:
                a, b = window[group == n_quantiles], window[group == 1]
                estimate = float(a.mean() - b.mean())
                stderr = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
                t_bmp_k = t_kp = None
            else:
                values = window[members]
                estimate = float(values.mean())
                stderr = float(values.std(ddof=1) / np.sqrt(len(values)))
                t_bmp_k = t_kp = None
                if sigma is not None:
                    s = meta["sigma"].to_numpy().astype(np.float64)[members]
                    std_car = values / (s * np.sqrt(k + 1))
                    std_car = std_car[np.isfinite(std_car)]
                    if len(std_car) > 1:
                        t_bmp_k = float(
                            std_car.mean() * np.sqrt(len(std_car)) / std_car.std(ddof=1)
                        )
                        if r_kp is not None:
                            m = len(std_car)
                            t_kp = t_bmp_k * float(np.sqrt((1 - r_kp) / (1 + (m - 1) * r_kp)))
            boot_p = None
            if boot is not None:
                draws = boot[:, g, j] - (boot[:, g, zero - 1] if zero > 0 else 0.0)
                draws = draws[np.isfinite(draws)]
                if len(draws):
                    boot_p = float(min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean())))
            summary_rows.append(
                {
                    "group": label,
                    "horizon": f"CAR[0,{k}]",
                    "n": n,
                    "car": estimate,
                    "se": stderr,
                    "t": estimate / stderr if stderr > 0 else None,
                    "t_bmp": t_bmp_k,
                    "t_kp": t_kp,
                    "boot_p": boot_p,
                }
            )
    return EventStudyResult(
        path=pl.DataFrame(path_rows),
        summary=pl.DataFrame(
            summary_rows,
            schema={
                "group": pl.String,
                "horizon": pl.String,
                "n": pl.Int64,
                "car": pl.Float64,
                "se": pl.Float64,
                "t": pl.Float64,
                "t_bmp": pl.Float64,
                "t_kp": pl.Float64,
                "boot_p": pl.Float64,
            },
        ),
        n_events=len(docs),
        kp_correlation=r_kp,
    )
