"""FinBERT scores (FR-09): ProsusAI/finbert over 512-token chunks.

Text is split into windows of at most 510 word-piece tokens (512 with ``[CLS]`` and
``[SEP]``) that overlap by ``stride`` tokens. Chunk probabilities are averaged with the
chunk's token count as weight; ``score = P(positive) - P(negative)``. The model files are
fetched once at a pinned revision under the 10 MB/s cap and loaded from disk. Scoring is
resumable: finished documents sit in numbered Parquet shards and are skipped on a rerun.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import polars as pl
import structlog

from llm_text_alpha.data.ratelimit import ByteThrottle
from llm_text_alpha.scoring.common import SCORES_SCHEMA, scoring_text

log = structlog.get_logger(__name__)

REPO = "ProsusAI/finbert"
HUB = "https://huggingface.co"
MODEL_FILES = ("config.json", "vocab.txt", "tokenizer_config.json", "special_tokens_map.json")
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def resolve_revision(
    repo: str, revision: str | None = None, transport: httpx.BaseTransport | None = None
) -> tuple[str, list[str]]:
    """Commit sha of ``repo`` at ``revision`` (default: the current main) and its file names."""
    path = f"{HUB}/api/models/{repo}" + (f"/revision/{revision}" if revision else "")
    with httpx.Client(transport=transport, timeout=30.0) as client:
        response = client.get(path)
        response.raise_for_status()
        payload = response.json()
    return str(payload["sha"]), [str(s["rfilename"]) for s in payload.get("siblings", [])]


def _complete(folder: Path) -> bool:
    return all((folder / name).exists() for name in MODEL_FILES[:2]) and any(
        (folder / name).exists() for name in WEIGHT_FILES
    )


def download_model(
    dest_root: Path,
    repo: str = REPO,
    revision: str | None = None,
    throttle: ByteThrottle | None = None,
    transport: httpx.BaseTransport | None = None,
) -> tuple[Path, str]:
    """Fetch the tokenizer, config and one weight file at ``revision`` (default: current).

    Files go to ``dest_root/<repo>/<revision>/``. A folder that already holds the config,
    vocabulary and weights is used without any network call. Returns the folder and sha.
    """
    if revision is not None and _complete(dest_root / repo.replace("/", "--") / revision):
        return dest_root / repo.replace("/", "--") / revision, revision
    sha, files = resolve_revision(repo, revision, transport)
    folder = dest_root / repo.replace("/", "--") / sha
    weights = next((w for w in WEIGHT_FILES if w in files), WEIGHT_FILES[-1])
    throttle = throttle or ByteThrottle()
    with httpx.Client(transport=transport, timeout=120.0, follow_redirects=True) as client:
        for name in (*MODEL_FILES, weights):
            target = folder / name
            if target.exists() or name not in files:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(name + ".tmp")
            with client.stream("GET", f"{HUB}/{repo}/resolve/{sha}/{name}") as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for chunk in response.iter_bytes(1 << 16):
                        handle.write(chunk)
                        throttle.consumed(len(chunk))
            temporary.replace(target)
            log.info("finbert_file_downloaded", file=name, revision=sha)
    return folder, sha


@dataclass(frozen=True)
class ChunkScore:
    """Token-weighted class probabilities for one document."""

    probabilities: dict[str, float]
    n_chunks: int
    n_tokens: int


class FinBert:
    """Chunked FinBERT inference on CPU (or CUDA when asked and available).

    Args:
        model_dir: folder with config, tokenizer and weights.
        model_id: id written to ``scores.model_id``, e.g. ``ProsusAI/finbert@<sha>``.
        device: ``auto``, ``cpu`` or ``cuda``.
        threads: torch intra-op threads; kept modest on a shared machine.
        batch_size: chunks per forward pass.
        max_tokens: model input length including special tokens.
        stride: overlap between consecutive chunks, in tokens.
        max_chunks: chunks read per document from its start; None reads everything.
    """

    def __init__(
        self,
        model_dir: Path,
        model_id: str,
        device: str = "auto",
        threads: int = 8,
        batch_size: int = 16,
        max_tokens: int = 512,
        stride: int = 64,
        max_chunks: int | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(threads)
        self.torch = torch
        self.device = "cuda" if device in ("auto", "cuda") and torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()
        self.labels = [
            self.model.config.id2label[i].lower() for i in range(self.model.config.num_labels)
        ]
        self.model_id = model_id
        self.batch_size = batch_size
        self.body = max_tokens - 2
        self.step = self.body - stride
        if self.step <= 0:
            raise ValueError("stride must be smaller than max_tokens - 2")
        self.max_chunks = max_chunks

    def chunks(self, text: str) -> list[list[int]]:
        """Word-piece ids per chunk, each wrapped in ``[CLS]`` and ``[SEP]``."""
        ids: list[int] = self.tokenizer(text, add_special_tokens=False, truncation=False)[
            "input_ids"
        ]
        if not ids:
            return []
        starts = list(range(0, max(1, len(ids) - self.body + self.step), self.step)) or [0]
        if self.max_chunks is not None:
            starts = starts[: self.max_chunks]
        cls, sep = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id
        return [[cls, *ids[s : s + self.body], sep] for s in starts]

    def _forward(self, batch: list[list[int]]) -> np.ndarray:
        width = max(len(c) for c in batch)
        pad = self.tokenizer.pad_token_id or 0
        input_ids = self.torch.tensor([c + [pad] * (width - len(c)) for c in batch])
        mask = self.torch.tensor([[1] * len(c) + [0] * (width - len(c)) for c in batch])
        with self.torch.inference_mode():
            logits = self.model(
                input_ids=input_ids.to(self.device), attention_mask=mask.to(self.device)
            ).logits
        return self.torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float64)

    def score_text(self, text: str) -> ChunkScore:
        """Token-weighted probabilities over all chunks; empty text is all neutral."""
        pieces = self.chunks(text)
        if not pieces:
            neutral = {name: float(name == "neutral") for name in self.labels}
            return ChunkScore(neutral, 0, 0)
        probs: list[np.ndarray] = []
        for start in range(0, len(pieces), self.batch_size):
            probs.append(self._forward(pieces[start : start + self.batch_size]))
        matrix = np.vstack(probs)
        weights = np.array([len(c) - 2 for c in pieces], dtype=np.float64)
        mean = weights @ matrix / weights.sum()
        return ChunkScore(
            {name: float(mean[i]) for i, name in enumerate(self.labels)},
            len(pieces),
            int(weights.sum()),
        )


def _label(probabilities: dict[str, float]) -> str:
    return max(probabilities, key=lambda name: probabilities[name])


def score_rows(finbert: FinBert, documents: pl.DataFrame) -> Iterator[dict[str, Any]]:
    """One ``scores`` row per document, as each finishes."""
    for doc_id, text in documents.select("doc_id", "text").iter_rows():
        started = time.perf_counter()
        result = finbert.score_text(scoring_text(text))
        p = result.probabilities
        yield {
            "doc_id": doc_id,
            "scorer": "finbert",
            "model_id": finbert.model_id,
            "prompt_version": None,
            "masked": False,
            "score": p.get("positive", 0.0) - p.get("negative", 0.0),
            "label": _label(p),
            "aux_json": json.dumps(
                {"probabilities": p, "n_chunks": result.n_chunks, "n_tokens": result.n_tokens},
                sort_keys=True,
            ),
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "cost_usd": 0.0,
            "scored_at": datetime.now(UTC),
        }


def done_ids(shard_dir: Path) -> set[str]:
    """Documents already scored in ``shard_dir``."""
    if not any(shard_dir.glob("part-*.parquet")):
        return set()
    return set(pl.read_parquet(shard_dir / "part-*.parquet", columns=["doc_id"])["doc_id"])


def score_resumable(
    finbert: FinBert, documents: pl.DataFrame, shard_dir: Path, shard_size: int = 500
) -> int:
    """Score documents not yet in ``shard_dir``, writing a shard every ``shard_size``.

    Returns how many documents were scored in this call.
    """
    shard_dir.mkdir(parents=True, exist_ok=True)
    todo = documents.filter(~pl.col("doc_id").is_in(list(done_ids(shard_dir))))
    next_part = len(list(shard_dir.glob("part-*.parquet")))
    buffer: list[dict[str, Any]] = []
    scored = 0
    started = time.perf_counter()

    def flush() -> None:
        nonlocal next_part
        if not buffer:
            return
        target = shard_dir / f"part-{next_part:05d}.parquet"
        temporary = target.with_name(target.name + ".tmp")
        pl.DataFrame(buffer, schema=SCORES_SCHEMA).write_parquet(temporary)
        temporary.replace(target)
        next_part += 1
        buffer.clear()

    for row in score_rows(finbert, todo):
        buffer.append(row)
        scored += 1
        if len(buffer) >= shard_size:
            flush()
            log.info(
                "finbert_progress",
                scored=scored,
                remaining=todo.height - scored,
                docs_per_s=round(scored / (time.perf_counter() - started), 3),
            )
    flush()
    return scored
