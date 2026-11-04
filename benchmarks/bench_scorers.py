"""Scorer throughput and latency on a fixed sample of real documents (NFR-01).

Writes ``reports/tables/bench_scorers.md``. Run locally under WSL, where the data lives:

    uv run python benchmarks/bench_scorers.py --n 500

The table states the reference hardware (``docs/hardware.md``) and that live recorders
share the machine. FinBERT runs with the settings of ``configs/default.yaml``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl

from llm_text_alpha import pipeline
from llm_text_alpha.config import load_yaml
from llm_text_alpha.reporting.tables import write_table
from llm_text_alpha.scoring import dictionary, finbert
from llm_text_alpha.settings import get_settings
from llm_text_alpha.tracking.trial_log import new_run_id


def measure(name: str, latencies_ms: list[float], wall_s: float, extra: str) -> dict[str, object]:
    values = np.array(latencies_ms)
    return {
        "scorer": name,
        "docs": len(values),
        "docs_per_s": len(values) / wall_s,
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "hours_per_100k_docs": 100_000 / (len(values) / wall_s) / 3600,
        "settings": extra,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()
    settings = get_settings()
    cfg = load_yaml(args.config)
    paths = pipeline.Paths.of(settings)
    docs = pipeline.scorable(
        pipeline.load_documents(paths, ["doc_id", "text", "permno", "in_universe", "excluded"])
    )
    sample = docs.sample(min(args.n, docs.height), seed=int(cfg["seed"]))
    rows = []
    lm = dictionary.load_dictionary(settings.data_path(cfg["scorers"]["lm"]["dictionary_file"]))
    started = time.perf_counter()
    scores = dictionary.score_documents(sample, lm)
    rows.append(
        measure("lm", scores["latency_ms"].to_list(), time.perf_counter() - started, lm.model_id)
    )
    fb_cfg = cfg["scorers"]["finbert"]
    folder, revision = finbert.download_model(paths.models, fb_cfg["model"], fb_cfg.get("revision"))
    model = finbert.FinBert(
        folder,
        f"{fb_cfg['model']}@{revision}",
        threads=args.threads,
        batch_size=int(fb_cfg["batch_size"]),
        max_tokens=int(fb_cfg["max_tokens"]),
        stride=int(fb_cfg["stride"]),
        max_chunks=fb_cfg.get("max_chunks"),
    )
    started = time.perf_counter()
    latencies = [row["latency_ms"] for row in finbert.score_rows(model, sample)]
    rows.append(
        measure(
            "finbert",
            latencies,
            time.perf_counter() - started,
            f"{model.model_id}, CPU, {args.threads} threads, max_chunks {fb_cfg.get('max_chunks')}",
        )
    )
    write_table(
        pl.DataFrame(rows),
        Path("reports/tables"),
        "bench_scorers",
        f"python benchmarks/bench_scorers.py --n {args.n} --threads {args.threads}",
        new_run_id(),
        "reference hardware in docs/hardware.md; live recorders share the machine",
    )


if __name__ == "__main__":
    main()
