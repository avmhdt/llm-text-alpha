"""Entity and date masking for the contamination protocol (FR-14, AC-W-09).

Rules run first, then spaCy NER, and the whole pass repeats until the text stops changing (at
most ``max_passes``), so masking a masked text changes nothing.

1. Tickers after an exchange name (``NYSE: XYZ``), web and e-mail addresses (a domain names
   the company), phone numbers, the filer's own ticker (two or more letters,
   case-sensitive) and its CIK.
2. The filer's current and former EDGAR conformed names: the full name with any legal-form
   suffix, and the short name without it. Multi-word names match in any case; one-word
   short names only in Title or UPPER case.
3. Explicit dates (``March 5, 2024``, ``03/05/2024``, ``2024-03-05``, ``March 2024``,
   ``March 5``), fiscal periods (``Q3 2024``, ``fiscal 2025``, ``FY25``, ``third quarter``,
   ``first half``, ``nine months``) and years 1990 to 2030.
4. spaCy entities: ORG as company, PERSON, GPE/LOC/FAC/NORP as place, and DATE spans that
   contain a digit or a month name.

The audit counts replacements by type, residual name and ticker hits (exact, after masking),
and residual years.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December|"
    "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
_MONTH = rf"(?:{MONTHS})\.?"
_DATE_PATTERNS = [
    rf"\b{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b",
    rf"\b\d{{1,2}}\s+{_MONTH}\s+\d{{4}}\b",
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    r"\b\d{4}-\d{2}-\d{2}\b",
    rf"\b{_MONTH},?\s+\d{{4}}\b",
    rf"\b{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?\b",
]
_ORDINAL = r"(?:first|second|third|fourth|1st|2nd|3rd|4th)"
_PERIOD_PATTERNS = [
    r"\b(?:Q[1-4]|[1-4]Q)\s*(?:FY)?\s*'?\d{2,4}\b",
    r"\bFY\s*'?\d{2,4}\b",
    r"\bQ[1-4]\b",
    r"\b[1-4]Q\b",
    r"\b(?:fiscal|full)[- ]year\s+\d{4}\b",
    r"\bfiscal\s+\d{4}\b",
    rf"\b{_ORDINAL}[- ]quarter(?:\s+of)?(?:\s+fiscal)?(?:\s+\d{{4}})?\b",
    rf"\b{_ORDINAL}[- ]half(?:\s+of)?(?:\s+fiscal)?(?:\s+\d{{4}})?\b",
    r"\b(?:three|six|nine|twelve)[- ]months?\b",
]
_YEAR = r"(?<![\d,.$])\b(?:199\d|20[0-2]\d|2030)\b(?![\d,%]|\.\d)"
_URL = (
    r"(?:https?://|www\.)\S+|\b[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+\b"
    r"|\b(?:[\w\-]+\.)+(?:com|net|org|io|us|co|biz|info)\b(?:/\S*)?"
)
_PHONE = r"(?<!\d)(?:\+?1[\s.\-]?)?\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}(?!\d)"
_STREET = (
    r"(?:Blvd|Boulevard|Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Way|Parkway|Pkwy|Place|Pl|"
    r"Court|Ct|Plaza|Circle|Highway|Hwy|Square|Sq|Center|Centre|Terrace|Trail|Pike)"
)
_ADDRESS = rf"\b\d{{1,6}}\s+(?:[NSEW]\.?\s+)?(?:[A-Z][A-Za-z0-9.'\-]*\s+){{1,4}}{_STREET}\b\.?"
_CAPS_BEFORE_PLACE = r"(?:\b[A-Z][A-Z.'\-]+\s+){1,3}(?=\[PLACE\])"
_CITY_BEFORE_PLACE = r"\b(?:[A-Z][A-Za-z.'\-]+\s+){0,2}[A-Z][A-Za-z.'\-]+(?=,\s*\[PLACE\])"
_CAPS_DATELINE = (
    r"\b[A-Z]{3,}(?:\s+[A-Z]{3,}){0,2}"
    r"(?=,\s*(?:\[DATE\]|\[PLACE\]|[A-Z]\.[A-Z]\.|[A-Z][a-z]{1,4}\.\s))"
)
_PLACEHOLDER = re.compile(r"\[[A-Z]+\]")


def _flags(case_sensitive: bool) -> int:
    return 0 if case_sensitive else re.IGNORECASE


@dataclass(frozen=True)
class MaskConfig:
    """``configs/masking.yaml``; ``version`` is ``m1@<sha8 of the file>``."""

    version: str
    placeholders: dict[str, str]
    legal_suffixes: tuple[str, ...]
    min_short_name_chars: int
    exchanges: tuple[str, ...]
    ner_model: str | None
    ner_labels: dict[str, str]
    max_passes: int
    first_word_stoplist: frozenset[str] = frozenset()
    ner_keep: frozenset[str] = frozenset()

    @classmethod
    def load(cls, path: Path, ner: bool = True) -> MaskConfig:
        raw_text = path.read_text("utf-8")
        raw = yaml.safe_load(raw_text)
        sha8 = hashlib.sha256(raw_text.encode()).hexdigest()[:8]
        return cls(
            version=f"{raw['version']}@{sha8}",
            placeholders=dict(raw["placeholders"]),
            legal_suffixes=tuple(str(s).lower() for s in raw["legal_suffixes"]),
            min_short_name_chars=int(raw["min_short_name_chars"]),
            exchanges=tuple(raw["exchanges"]),
            ner_model=str(raw["ner"]["model"]) if ner else None,
            ner_labels=dict(raw["ner"]["labels"]),
            max_passes=int(raw.get("max_passes", 3)),
            first_word_stoplist=frozenset(
                str(s).lower() for s in raw.get("first_word_stoplist", [])
            ),
            ner_keep=frozenset(str(s).lower() for s in raw.get("ner_keep", [])),
        )


def clean_name(name: str) -> str:
    """An EDGAR conformed name without state tags (``/DE/``) and extra spaces."""
    name = re.sub(r"/[A-Z]{2,3}/?", " ", name)
    name = re.sub(r"\s*\\[A-Z]{2,3}\\?", " ", name)
    return re.sub(r"\s+", " ", name).strip(" ,.")


def name_variants(
    names: Iterable[str], suffixes: Sequence[str], first_word_stop: Iterable[str] | None = None
) -> list[str]:
    """Full and short forms of each name, longest first; empty and duplicate forms dropped.

    With ``first_word_stop``, the first word of a multi-word short name (four or more
    letters, not in the stoplist) is a variant too: CISCO for CISCO SYSTEMS INC.
    """
    stop = {s.lower() for s in first_word_stop} if first_word_stop is not None else None
    out: set[str] = set()
    suffix_set = set(suffixes)
    for raw in names:
        if not raw:
            continue
        full = clean_name(raw)
        words = re.findall(r"[A-Za-z0-9&'\-]+", full)
        if not words:
            continue
        out.add(" ".join(words))
        core = list(words)
        while core and core[-1].lower().strip(".") in suffix_set:
            core.pop()
        while core and core[0].lower() == "the":
            core.pop(0)
        if core:
            out.add(" ".join(core))
        if stop is not None and len(core) > 1:
            first = core[0]
            if len(first) >= 4 and first.isalpha() and first.lower() not in stop:
                out.add(first)
    return sorted(out, key=lambda v: (-len(v), v))


def _name_regex(variant: str, min_chars: int) -> re.Pattern[str] | None:
    words = variant.split(" ")
    body = r"[\s,.\-]*".join(re.escape(w) for w in words)
    tail = r"(?:[\s,]+(?:Inc|Corp|Corporation|Co|Company|Ltd|LLC|L\.?P|PLC|N\.?V|S\.?A)\b\.?)?"
    if len(words) == 1:
        if len(variant) < min_chars or not re.search(r"[A-Za-z]", variant):
            return None
        title, upper = variant[:1].upper() + variant[1:].lower(), variant.upper()
        alts = "|".join(sorted({re.escape(title), re.escape(upper), re.escape(variant)}))
        return re.compile(rf"(?<![\w\[])(?:The\s+)?(?:{alts}){tail}(?![\w\]])")
    return re.compile(rf"(?<![\w\[])(?:the\s+)?{body}{tail}(?![\w\]])", re.IGNORECASE)


@dataclass
class MaskResult:
    """Masked text and the audit counts for one document."""

    text: str
    counts: dict[str, int] = field(default_factory=dict)
    residual_name_hits: int = 0
    residual_year_hits: int = 0
    passes: int = 0


class Masker:
    """Masks one company's text; ``nlp`` is a loaded spaCy pipeline or ``None`` (rules only)."""

    def __init__(self, config: MaskConfig, nlp: Any | None = None) -> None:
        self.config = config
        self.nlp = nlp
        exchanges = "|".join(re.escape(e) for e in sorted(config.exchanges, key=len, reverse=True))
        self._exchange = re.compile(
            rf"(\b(?:{exchanges})\b[\w\s]{{0,30}}?:\s*)([A-Z][A-Z0-9]{{0,5}}(?:[.\-][A-Z])?)\b"
        )
        self._dates = [re.compile(p, re.IGNORECASE) for p in _DATE_PATTERNS]
        self._periods = [re.compile(p, re.IGNORECASE) for p in _PERIOD_PATTERNS]
        self._year = re.compile(_YEAR)
        self._url = re.compile(_URL, re.IGNORECASE)
        self._phone = re.compile(_PHONE)
        self._month = re.compile(rf"\b(?:{MONTHS})\b", re.IGNORECASE)
        self._address = re.compile(_ADDRESS)
        self._caps_before_place = re.compile(_CAPS_BEFORE_PLACE)
        self._city_before_place = re.compile(_CITY_BEFORE_PLACE)
        self._caps_dateline = re.compile(_CAPS_DATELINE)

    def variants(self, names: Sequence[str]) -> list[str]:
        """Name variants of the filer, the brand word included."""
        return name_variants(names, self.config.legal_suffixes, self.config.first_word_stoplist)

    @staticmethod
    def load_nlp(model: str) -> Any:
        import spacy

        return spacy.load(model, disable=["lemmatizer", "tagger", "attribute_ruler", "parser"])

    def _sub(self, pattern: re.Pattern[str], kind: str, text: str, counts: dict[str, int]) -> str:
        token = self.config.placeholders[kind]
        new, n = pattern.subn(token, text)
        counts[kind] = counts.get(kind, 0) + n
        return new

    def rules(
        self,
        text: str,
        names: Sequence[str],
        ticker: str | None,
        cik: int | None,
        counts: dict[str, int],
    ) -> str:
        """The rule layer alone."""
        ph = self.config.placeholders
        n_exchange = 0

        def exchange(m: re.Match[str]) -> str:
            nonlocal n_exchange
            n_exchange += 1
            return m.group(1) + ph["ticker"]

        text = self._exchange.sub(exchange, text)
        text = self._sub(self._url, "url", text, counts)
        text = self._sub(self._phone, "phone", text, counts)
        counts["ticker"] = counts.get("ticker", 0) + n_exchange
        for variant in self.variants(names):
            pattern = _name_regex(variant, self.config.min_short_name_chars)
            if pattern is not None:
                text = self._sub(pattern, "company", text, counts)
        if ticker and len(ticker) >= 2:
            text = self._sub(
                re.compile(rf"(?<![\w\[]){re.escape(ticker)}(?![\w\]])"), "ticker", text, counts
            )
        if cik:
            text = self._sub(re.compile(rf"\b0*{cik}\b"), "cik", text, counts)
        for pattern in self._dates:
            text = self._sub(pattern, "date", text, counts)
        for pattern in self._periods:
            text = self._sub(pattern, "period", text, counts)
        text = self._sub(self._year, "year", text, counts)
        text = self._sub(self._address, "place", text, counts)
        text = self._sub(self._caps_dateline, "place", text, counts)
        text = self._sub(self._city_before_place, "place", text, counts)
        merged, n = self._caps_before_place.subn("", text)
        counts["place"] = counts.get("place", 0) + n
        return merged

    def _keep_date(self, span: str) -> bool:
        return bool(re.search(r"\d", span) or self._month.search(span))

    def entities(self, text: str, counts: dict[str, int], doc: Any | None = None) -> str:
        """Replace spaCy entities; ``doc`` may be a pre-parsed ``text`` (from ``nlp.pipe``)."""
        if self.nlp is None:
            return text
        parsed = doc if doc is not None else self.nlp(text)
        pieces: list[str] = []
        last = 0
        for ent in parsed.ents:
            kind = self.config.ner_labels.get(ent.label_)
            span = ent.text
            if kind is None or "[" in span or "]" in span or not span.strip():
                continue
            if span.strip().lower() in self.config.ner_keep:
                continue
            if kind == "date" and not self._keep_date(span):
                continue
            pieces.append(text[last : ent.start_char])
            pieces.append(self.config.placeholders[kind])
            counts[kind] = counts.get(kind, 0) + 1
            last = ent.end_char
        pieces.append(text[last:])
        return "".join(pieces)

    def audit(self, text: str, names: Sequence[str], ticker: str | None) -> tuple[int, int]:
        """Residual exact name/ticker hits and residual years in masked ``text``."""
        hits = 0
        for variant in self.variants(names):
            if len(variant) < self.config.min_short_name_chars:
                continue
            flags = re.IGNORECASE if " " in variant else 0
            hits += len(re.findall(rf"(?<![\w\[]){re.escape(variant)}(?![\w\]])", text, flags))
        if ticker and len(ticker) >= 2:
            hits += len(re.findall(rf"(?<![\w\[]){re.escape(ticker)}(?![\w\]])", text))
        return hits, len(self._year.findall(text))

    def mask(
        self,
        text: str,
        names: Sequence[str],
        ticker: str | None = None,
        cik: int | None = None,
        parse: Callable[[str], Any] | None = None,
    ) -> MaskResult:
        """Mask until stable; ``parse`` overrides ``nlp`` for the first pass (batching)."""
        counts: dict[str, int] = {}
        current = text
        passes = 0
        for passes in range(1, self.config.max_passes + 1):
            ruled = self.rules(current, names, ticker, cik, counts)
            doc = parse(ruled) if (parse is not None and passes == 1) else None
            new = self.entities(ruled, counts, doc)
            if new == current:
                break
            current = new
        residual, years = self.audit(current, names, ticker)
        return MaskResult(current, counts, residual, years, passes)


def leftover_placeholders(text: str) -> int:
    """Placeholders in ``text`` (for the review table)."""
    return len(_PLACEHOLDER.findall(text))


@dataclass(frozen=True)
class Entity:
    """What the masker needs to know about a document's company."""

    names: tuple[str, ...]
    ticker: str | None
    cik: int | None


def mask_many(
    masker: Masker, texts: Sequence[str], entities: Sequence[Entity], n_process: int = 1
) -> list[MaskResult]:
    """Mask many documents, parsing each pass with ``nlp.pipe`` (``n_process`` workers).

    Same result as ``masker.mask`` on each document; passes repeat only for documents that
    still change.
    """
    counts: list[dict[str, int]] = [{} for _ in texts]
    current = list(texts)
    passes = [0] * len(texts)
    todo = list(range(len(texts)))
    for pass_no in range(1, masker.config.max_passes + 1):
        if not todo:
            break
        ruled = {
            i: masker.rules(
                current[i], entities[i].names, entities[i].ticker, entities[i].cik, counts[i]
            )
            for i in todo
        }
        if masker.nlp is not None:
            docs = masker.nlp.pipe((ruled[i] for i in todo), batch_size=32, n_process=n_process)
            new = {
                i: masker.entities(ruled[i], counts[i], doc)
                for i, doc in zip(todo, docs, strict=True)
            }
        else:
            new = ruled
        still = []
        for i in todo:
            passes[i] = pass_no
            if new[i] != current[i]:
                current[i] = new[i]
                still.append(i)
        todo = still
    results = []
    for i, text in enumerate(current):
        residual, years = masker.audit(text, entities[i].names, entities[i].ticker)
        results.append(MaskResult(text, counts[i], residual, years, passes[i]))
    return results
