from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from llm_text_alpha.corpus.mask import (
    Entity,
    MaskConfig,
    Masker,
    clean_name,
    leftover_placeholders,
    mask_many,
    name_variants,
)

CONFIG = Path(__file__).parents[2] / "configs" / "masking.yaml"
RULES = Masker(MaskConfig.load(CONFIG, ner=False))
SUFFIXES = RULES.config.legal_suffixes
DASH = chr(0x2013)


@pytest.fixture(scope="module")
def full() -> Masker:
    pytest.importorskip("spacy")
    cfg = MaskConfig.load(CONFIG)
    return Masker(cfg, Masker.load_nlp(str(cfg.ner_model)))


def test_version_carries_the_file_hash() -> None:
    assert RULES.config.version.startswith("m2@") and len(RULES.config.version) == 11


def test_name_variants() -> None:
    assert clean_name("BERKSHIRE HATHAWAY INC /DE/") == "BERKSHIRE HATHAWAY INC"
    variants = name_variants(["THE COCA COLA CO", "APPLE COMPUTER INC"], SUFFIXES)
    assert "COCA COLA" in variants and "APPLE COMPUTER" in variants
    assert variants == sorted(variants, key=lambda v: (-len(v), v))
    assert name_variants(["", "  "], SUFFIXES) == []


def test_rules_mask_names_tickers_dates_and_periods() -> None:
    text = (
        "APPLE INC (NASDAQ: AAPL) reported fiscal 2024 third quarter results for the quarter "
        "ended June 29, 2024. Apple's revenue rose versus Q3 2023 and FY23. The Coca-Cola "
        "Company is a customer. CIK 0000320193. See investor.apple.com or call (408) 996-1010. "
        "For the nine months ended 6/29/2024 and in March 2025, sales grew. AAPL closed up."
    )
    r = RULES.mask(text, ["APPLE INC", "THE COCA COLA CO"], "AAPL", 320193)
    for leaked in (
        "Apple",
        "APPLE",
        "AAPL",
        "Coca",
        "2024",
        "2023",
        "2025",
        "320193",
        "June",
        "apple.com",
    ):
        assert leaked not in r.text, leaked
    assert r.counts["company"] >= 3 and r.counts["ticker"] == 2 and r.counts["cik"] == 1
    assert r.counts["period"] >= 4 and r.counts["date"] >= 3
    assert r.residual_name_hits == 0 and r.residual_year_hits == 0
    assert "revenue rose" in r.text


def test_one_word_short_names_need_title_or_upper_case() -> None:
    r = RULES.mask("Target said the target price for TARGET is unchanged.", ["TARGET CORP"], None)
    assert r.text == "[COMPANY] said the target price for [COMPANY] is unchanged."


def test_numbers_that_are_not_years_survive() -> None:
    r = RULES.mask(
        "Sales were $2,024 million, 2024.5 units and 12.2024% margin.", ["X Y Z INC"], None
    )
    assert "2,024" in r.text and r.counts.get("year", 0) == 0


def test_single_letter_tickers_only_after_an_exchange() -> None:
    r = RULES.mask("AT&T (NYSE: T) said T is a letter.", ["AT&T INC"], "T")
    assert r.text.count("[TICKER]") == 1 and "T is a letter" in r.text


company_word = st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ", min_size=4, max_size=9)
year = st.integers(1990, 2030)
filler = st.sampled_from(["revenue rose", "margins fell", "the board met", "guidance held", ""])


@settings(max_examples=200, deadline=None)
@given(
    words=st.lists(company_word, min_size=1, max_size=3),
    suffix=st.sampled_from(["INC", "CORP", "CO", "HOLDINGS INC", "LTD", ""]),
    years=st.lists(year, min_size=1, max_size=4),
    pieces=st.lists(filler, min_size=1, max_size=4),
)
def test_mask_idempotent_and_complete(
    words: list[str], suffix: str, years: list[int], pieces: list[str]
) -> None:
    """Masking twice equals masking once; no generated company name or year survives."""
    name = " ".join(words + ([suffix] if suffix else []))
    title = " ".join(w.title() for w in words)
    parts = [f"{title} reported results in {years[0]}."]
    for i, piece in enumerate(pieces):
        parts.append(f"{piece} {name} on March {i + 1}, {years[i % len(years)]}.")
    text = " ".join(parts)
    once = RULES.mask(text, [name], None)
    twice = RULES.mask(once.text, [name], None)
    assert twice.text == once.text
    assert once.residual_name_hits == 0 and once.residual_year_hits == 0
    for y in years:
        assert str(y) not in once.text
    assert title not in once.text and name not in once.text


def test_ner_masks_people_places_and_other_companies(full: Masker) -> None:
    text = (
        "Acme Widgets Inc. reported results. Chief Executive Officer Jane Rodriguez said demand "
        "in Germany and Ohio was strong, and that Microsoft remains a key partner."
    )
    r = full.mask(text, ["ACME WIDGETS INC"], "ACMW")
    for leaked in ("Acme", "Jane Rodriguez", "Germany", "Ohio", "Microsoft"):
        assert leaked not in r.text, leaked
    assert full.mask(r.text, ["ACME WIDGETS INC"], "ACMW").text == r.text
    assert leftover_placeholders(r.text) >= 5


def test_mask_many_matches_mask(full: Masker) -> None:
    texts = [
        "Acme Widgets Inc. (NYSE: ACMW) grew in Texas in fiscal 2023.",
        "Beta Corp said John Smith joined on May 1, 2021.",
    ]
    ents = [Entity(("ACME WIDGETS INC",), "ACMW", 1), Entity(("BETA CORP",), "BETA", 2)]
    many = mask_many(full, texts, ents)
    single = [full.mask(t, e.names, e.ticker, e.cik) for t, e in zip(texts, ents, strict=True)]
    assert [m.text for m in many] == [s.text for s in single]
    rules_only = mask_many(RULES, texts, ents)
    assert "Texas" in rules_only[0].text and "[PERIOD]" in rules_only[0].text


@pytest.mark.parametrize(
    ("text", "names", "ticker", "leaks"),
    [
        (
            "Moody's Corporation (NYSE:MCO) and MOODY'S said.",
            ["MOODYS CORP /DE/"],
            "MCO",
            ["Moody", "MOODY"],
        ),
        ("McKESSON REPORTS. McKesson Corporation said.", ["MCKESSON CORP"], "MCK", ["McK", "MCK"]),
        ("3M Delivers Strong Results. 3M (NYSE: MMM) said.", ["3M CO"], "MMM", ["3M", "MMM"]),
        (
            "RBC Bearings (NYSE: ROLL, RBCP) said.",
            ["RBC BEARINGS INC"],
            "ROLL",
            ["RBCP", "ROLL", "Bearings"],
        ),
        ("CISCO REPORTS RESULTS; Cisco grew.", ["CISCO SYSTEMS, INC."], "CSCO", ["CISCO", "Cisco"]),
        (
            "ELGIN, Ill.--(BUSINESS WIRE)--Today. PHILADELPHIA--(BUSINESS WIRE)--Today.",
            ["X CORP"],
            None,
            ["ELGIN", "PHILADELPHIA"],
        ),
        ("Riverwoods, [COMPANY], [DATE] - results.", ["X CORP"], None, ["Riverwoods"]),
        ("Visit 26125 N. Riverwoods Blvd., Suite 500.", ["X CORP"], None, ["Riverwoods"]),
    ],
)
def test_review_leaks_are_masked(
    text: str, names: list[str], ticker: str | None, leaks: list[str]
) -> None:
    out = RULES.mask(text, names, ticker).text
    for leak in leaks:
        assert leak not in out, (leak, out)


def test_generic_first_words_and_financial_terms_survive() -> None:
    out = RULES.mask(
        f"General Electric said American demand rose; EBITDA {DASH} up.",
        ["GENERAL MILLS INC"],
        None,
    )
    assert "American" in out.text and f"EBITDA {DASH}" in out.text


def test_audit_is_stricter_than_masking() -> None:
    r = RULES.mask("Target said the target price rose.", ["TARGET CORP"], None)
    assert (
        r.text.startswith("[COMPANY]") and r.residual_name_hits == 0 and r.residual_word_hits == 1
    )


def test_lowercase_brands_are_masked_with_an_english_word_list() -> None:
    english = frozenset({"waste", "target", "management"})
    masker = Masker(MaskConfig.load(CONFIG, ner=False), english=english)
    r = masker.mask(
        "lululemon athletica said lululemon grew; waste fell.", ["LULULEMON ATHLETICA INC."], None
    )
    assert "lululemon" not in r.text and r.residual_name_hits == 0
    w = masker.mask("Waste Management said waste volumes rose.", ["WASTE MANAGEMENT INC"], None)
    assert w.text.startswith("[COMPANY]") and "waste volumes" in w.text
    assert w.residual_name_hits == 0 and w.residual_word_hits == 1
