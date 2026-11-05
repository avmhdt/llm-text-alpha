# Contamination protocol

FR-20, pre-registered hypothesis H3 ([preregistration.md](preregistration.md)), AC-H-02,
AC-W-09.

An LLM that read news about a company's later returns during training can look predictive
on old filings without being able to predict anything. The protocol looks for that in three
ways:

1. **Training cutoff.** Each model is evaluated before and after its training cutoff.
2. **Masking.** Each model is evaluated on raw text and on text with the company, people,
   places and dates masked.
3. **Probes.** Each model is asked directly what it remembers about each filing.

The differences are compared against the dictionary and FinBERT, which have no cutoff in
this sense. That separates contamination from changes in the market.

## Models and windows

The post-cutoff window starts on the first day of the month after the published training
data cutoff (DQ-12, `configs/models.yaml`).

| Model | Training data cutoff | Post-cutoff from | Scored events |
|---|---|---|---|
| claude-haiku-5-5 | Jun 2026 | 2026-07-01 | `item202_main` + `item202_post_h55` (20,554) |
| claude-haiku-4-5 | Jul 2025 (reliable knowledge Feb 2025) | 2025-08-01 | `item202_contam` (3,000) |
| llama-3.1-8b-instruct-q4_k_m | Dec 2023 ("The pretraining data has a cutoff of December 2023.") | 2024-01-01 | `item202_local` (1,200) |

## Samples

`item202_contam` puts 750 events in each of four windows, split by year within each window:

- before 2024-01;
- 2024-01 to 2025-07 (after Llama's cutoff);
- 2025-08 to 2026-06 (after Haiku 4.5's);
- 2026-07 onwards (after Haiku 5.5's).

A draw proportional to the corpus would put only about 310 events after Haiku 4.5's cutoff,
under the 500-event power flag.

`item202_local` is the local model's sample. It is the part of `item202_contam` that CPU
scoring of both raw and masked text fits into about a day (D-05): 600 events before
Llama's cutoff and 600 after. The 50-document benchmark measured 32 s per document (p95 38
s) on 8 threads of an i9-14900K, about 2,680 documents per day.

Every model scores the same truncated input: the first 7,050 characters of the EX-99 text,
with the same prompt (`score_v1`).

## Masking (FR-14)

`lta corpus mask` masks each document's LLM input window (`configs/masking.yaml`,
`src/llm_text_alpha/corpus/mask.py`). Rules come first, then spaCy `en_core_web_sm` 3.8.0.
The rules and NER repeat until the text stops changing, so masking is idempotent.

| What | How | Placeholder |
|---|---|---|
| Filer's current and former EDGAR names | full name with legal suffix, short name, and the first word of a multi-word name unless it is generic (CISCO) | `[COMPANY]` |
| Other organizations | spaCy ORG, except financial terms such as EPS and EBITDA | `[COMPANY]` |
| Tickers | after an exchange name, and the filer's own ticker | `[TICKER]` |
| CIK | digits with or without leading zeros | `[CIK]` |
| People | spaCy PERSON | `[PERSON]` |
| Places | spaCy GPE, LOC, FAC and NORP; street addresses; dateline cities in capitals | `[PLACE]` |
| Web and e-mail addresses, phone numbers | rules | `[URL]`, `[PHONE]` |
| Dates | explicit dates and month-year; spaCy DATE spans that contain a digit or a month name | `[DATE]` |
| Fiscal periods | Q3 2024, FY25, fiscal 2025, third quarter, first half, nine months | `[PERIOD]` |
| Years 1990 to 2030 | not inside amounts or decimals | `[YEAR]` |

Financial figures stay unmasked, so a model can still recognize a company from its numbers.
The re-identification probe measures how often it does.

AUDIT_PLACEHOLDER

## Manual review

REVIEW_PLACEHOLDER

## Probes

- **Re-identification.** 300 masked releases from `item202_contam`, 75 per window. The model
  names the company and the calendar quarter of the reported period
  (`configs/prompts/probe_reid_v1.md`).
  - A company guess counts as right when its normalized name equals or contains one of the
    filer's name variants (four letters or more), or when the guessed ticker is right.
  - The quarter counts as right when it equals the calendar quarter of `fqtr_end` minus 7
    days.
- **Outcome recall.** The same events, with no text. The model is given the company, ticker
  and filing date, and asked whether the stock beat the market over the next five sessions
  (`configs/prompts/probe_recall_v1.md`). The truth is the sign of CAR[0,5]; chance is 50%.

Accuracy is reported before and after each model's cutoff, with Wilson 95% intervals.

## Tables

`lta eval contamination` writes four tables:

- `reports/tables/contamination_cells.md`: per model, the four cells (pre/post × raw/masked)
  on events scored both ways. Each cell has the Q5 − Q1 CAR[0,5] spread with a Welch 95% CI,
  the Spearman IC with a Fisher 95% CI, and N. Cells under 500 events are flagged.
- `reports/tables/contamination.md`: the share of apparent alpha that disappears,
  `(pre_raw − post_masked) / pre_raw`, with the cutoff and masking margins. It also has the
  DiD, `(LLM_pre − LLM_post) − (baseline_pre − baseline_post)` on raw text, against LM and
  FinBERT on the same events, with a 95% event-bootstrap CI (500 draws). Contamination counts
  as found when that CI excludes zero with pre > post.
- `reports/tables/contamination_did.md` and `reports/tables/contamination_probes.md`.

Each run's `params_json` carries the pre-registration commit SHA (AC-H-07).

RESULTS_PLACEHOLDER

## Limits

- Masking removes names, not knowledge. Distinctive figures, products and segments remain,
  so masked results are a lower bound on how much memorization was removed, not proof of none
  (R-07).
- Haiku 5.5's post-cutoff window is three months: about 1,000 events, and daily returns only
  to 2026-09-30.
- Pre- and post-cutoff periods differ in market conditions too. The DiD against LM and
  FinBERT controls for what the two have in common, not for everything (R-08).
- Glasserman and Lin (2023) find that masking can also help a model by removing distracting
  names. A masked score above the raw score is possible and is reported as found.
