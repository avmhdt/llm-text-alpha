from __future__ import annotations

import json
from pathlib import Path

import httpx
import polars as pl
import pytest

from llm_text_alpha.scoring import dictionary, finbert
from llm_text_alpha.scoring.common import scoring_text, sections

MINI = """Word,Seq_num,Negative,Positive,Uncertainty,Litigious,Strong_Modal,Weak_Modal,Constraining
GAIN,1,0,2009,0,0,0,0,0
STRONG,2,0,2009,0,0,0,0,0
LOSS,3,2009,0,0,0,0,0,0
DECLINE,4,2009,0,0,0,0,0,0
MAY,5,0,0,2009,0,0,1,0
LAWSUIT,6,2009,0,0,2009,0,0,0
REQUIRED,7,0,0,0,0,0,0,2009
REMOVED,8,0,-2012,0,0,0,0,0
"""


@pytest.fixture
def lm(tmp_path: Path) -> dictionary.LMDictionary:
    path = tmp_path / "mini_dictionary.csv"
    path.write_text(MINI, "utf-8")
    return dictionary.load_dictionary(path)


def test_dictionary_counts_and_negation(lm: dictionary.LMDictionary) -> None:
    text = (
        "Strong gain. Not a gain at all; never, ever strong. A loss and a decline. "
        "We may face a lawsuit, as required. Removed."
    )
    counts = dictionary.count(text, lm)
    assert counts == {
        "positive": 2,
        "negative": 3,
        "uncertainty": 1,
        "litigious": 1,
        "constraining": 1,
        "negated_positive": 2,
        "n_words": 23,
    }
    assert dictionary.tone(counts) == pytest.approx((2 - 3) / 6)
    assert dictionary.label(0.0) == "neutral" and dictionary.label(-0.1) == "negative"
    assert lm.model_id.startswith("lm-mini_dictionary@")


def test_dictionary_errors(tmp_path: Path) -> None:
    with pytest.raises(dictionary.DictionaryError, match="does not exist"):
        dictionary.load_dictionary(tmp_path / "none.csv")
    bad = tmp_path / "bad.csv"
    bad.write_text("Word,Positive\nA,1\n", "utf-8")
    with pytest.raises(dictionary.DictionaryError, match="lacks"):
        dictionary.load_dictionary(bad)


def test_score_documents_reads_the_press_release(lm: dictionary.LMDictionary) -> None:
    docs = pl.DataFrame(
        {
            "doc_id": ["a", "b"],
            "text": ["[[8-K]]\nloss loss loss\n\n[[EX-99.1]]\nstrong gain", "gain decline decline"],
        }
    )
    scores = dictionary.score_documents(docs, lm)
    assert scores["score"].to_list() == pytest.approx([2 / 3, -1 / 4])
    assert scores["label"].to_list() == ["positive", "negative"]
    assert json.loads(scores["aux_json"][0])["positive"] == 2
    assert (scores["cost_usd"] == 0).all() and scores["latency_ms"].is_not_null().all()


def test_sections() -> None:
    text = "[[8-K]]\ncover\n\n[[EX-99.1]]\nrelease\n\n[[EX-99.2]]\nslides"
    assert sections(text) == [("8-K", "cover"), ("EX-99.1", "release"), ("EX-99.2", "slides")]
    assert scoring_text(text) == "release\n\nslides"
    assert scoring_text("[[8-K]]\ncover only") == "cover only"
    assert scoring_text("untagged") == "untagged"


VOCAB = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    "revenue",
    "rose",
    "fell",
    "the",
    "company",
    ".",
]


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A random two-layer BERT with FinBERT's label order, built offline."""
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    folder = tmp_path_factory.mktemp("tiny_bert")
    (folder / "vocab.txt").write_text("\n".join(VOCAB) + "\n", "utf-8")
    BertTokenizerFast(vocab_file=str(folder / "vocab.txt")).save_pretrained(folder)
    config = BertConfig(
        vocab_size=len(VOCAB),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
        num_labels=3,
        id2label={0: "positive", 1: "negative", 2: "neutral"},
        label2id={"positive": 0, "negative": 1, "neutral": 2},
    )
    BertForSequenceClassification(config).save_pretrained(folder)
    return folder


def test_finbert_chunks_and_weights(tiny_model: Path) -> None:
    model = finbert.FinBert(tiny_model, "tiny@x", threads=1, batch_size=2, max_tokens=12, stride=3)
    ids = model.tokenizer("the company revenue rose " * 6, add_special_tokens=False)["input_ids"]
    pieces = model.chunks("the company revenue rose " * 6)
    assert len(ids) == 24 and [len(p) for p in pieces] == [12, 12, 12]
    assert pieces[1][1:4] == ids[7:10]
    assert all(
        p[0] == model.tokenizer.cls_token_id and p[-1] == model.tokenizer.sep_token_id
        for p in pieces
    )
    result = model.score_text("the company revenue rose " * 6)
    assert result.n_chunks == 3 and result.n_tokens == 30
    assert sum(result.probabilities.values()) == pytest.approx(1.0)
    assert model.score_text("").n_chunks == 0
    capped = finbert.FinBert(tiny_model, "tiny@x", threads=1, max_tokens=12, stride=3, max_chunks=2)
    assert len(capped.chunks("the company revenue rose " * 6)) == 2
    with pytest.raises(ValueError, match="stride"):
        finbert.FinBert(tiny_model, "x", max_tokens=12, stride=10)


def test_finbert_resumable_shards(tiny_model: Path, tmp_path: Path) -> None:
    model = finbert.FinBert(tiny_model, "tiny@x", threads=1, max_tokens=12, stride=3)
    docs = pl.DataFrame({"doc_id": [f"d{i}" for i in range(5)], "text": ["revenue rose ."] * 5})
    shards = tmp_path / "scores"
    assert finbert.done_ids(shards) == set()
    assert finbert.score_resumable(model, docs.head(3), shards, shard_size=2) == 3
    assert finbert.score_resumable(model, docs, shards, shard_size=2) == 2
    scored = pl.read_parquet(shards / "part-*.parquet")
    assert sorted(scored["doc_id"]) == [f"d{i}" for i in range(5)]
    assert len(list(shards.glob("part-*.parquet"))) == 3
    row = scored.row(0, named=True)
    assert row["scorer"] == "finbert" and row["model_id"] == "tiny@x"
    aux = json.loads(row["aux_json"])
    assert row["score"] == pytest.approx(
        aux["probabilities"]["positive"] - aux["probabilities"]["negative"]
    )


def test_download_model_pins_and_reuses(tmp_path: Path) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path.startswith("/api/models/"):
            files = [{"rfilename": n} for n in (*finbert.MODEL_FILES, "pytorch_model.bin")]
            return httpx.Response(200, json={"sha": "abc123", "siblings": files})
        return httpx.Response(200, content=b"x" * 10)

    transport = httpx.MockTransport(handler)
    folder, sha = finbert.download_model(tmp_path, transport=transport)
    assert sha == "abc123" and folder.name == "abc123"
    assert sorted(p.name for p in folder.iterdir()) == sorted(
        [*finbert.MODEL_FILES, "pytorch_model.bin"]
    )
    count = len(seen)
    finbert.download_model(tmp_path, revision="abc123", transport=transport)
    assert len(seen) == count


def test_install_writes_a_manifest(tmp_path: Path) -> None:
    source = tmp_path / "Loughran-McDonald_MasterDictionary_test.csv"
    source.write_text(MINI, "utf-8")
    target = tmp_path / "raw" / "lm" / "LM_MasterDictionary.csv"
    manifest = dictionary.install_dictionary(source, target)
    saved = json.loads((target.parent / "manifest.json").read_text("utf-8"))
    assert saved == manifest
    assert manifest["source_file"] == source.name
    assert manifest["n_bytes"] == source.stat().st_size
    assert manifest["words"] == {
        "positive": 2,
        "negative": 3,
        "uncertainty": 1,
        "litigious": 1,
        "constraining": 1,
    }
    assert target.read_bytes() == source.read_bytes()
