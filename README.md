# LLM and NLP alpha with point-in-time discipline

> Score filings, transcripts and news with language models, and prove the result is not leakage or memorization.

**Status:** in progress. M1 to M3 are built: corpus, timestamp gate, returns, controls and the dictionary and FinBERT baselines. See [docs/decisions.md](docs/decisions.md).

| | |
|---|---|
| Track | Quant Researcher |
| Estimated effort | 6–8 weeks |
| Stack | Python, PyTorch, Transformers, LLM APIs |
| Data cost | Free → Paid |
| Strongest for | Systematic equity, data-science-heavy funds |

## Requirements

- [x] A text corpus with exact publication timestamps (SEC acceptance time, call time, news publish time). SEC acceptance times for 71,313 Item 2.02 8-Ks; transcripts are out of scope (D-07) and news arrives in M5. (FR-01, FR-02, FR-06)
- [x] Trades at the next available bar after each timestamp; returns labelled as abnormal returns against a factor model. (FR-07, FR-15, AC-W-01, AC-W-07)
- [ ] Baselines first: the Loughran–McDonald dictionary and FinBERT. Then LLM scores or embeddings.
- [x] An event study of cumulative abnormal returns around each event, plus cross-sectional regressions controlling for size, momentum and earnings surprise (SUE). (FR-16, FR-17, AC-W-02, AC-W-08)
- [ ] Contamination controls: test only after each model’s training cutoff, and rerun on text with company names and dates masked.
- [ ] Inference cost and latency tracked per document.

## Data

Raw and licensed data live in `data/`, which is git-ignored. Only download scripts and small synthetic fixtures are committed.

| Dataset | Cost | What it gives you |
|---|---|---|
| [SEC EDGAR filings and full-text search](https://www.sec.gov/edgar/sec-api-documentation) | Free | 8-K, 10-K and 10-Q with acceptance timestamps. |
| [Earnings call transcripts (Capital IQ via WRDS)](https://wrds-www.wharton.upenn.edu) | Academic | Timestamped transcripts. Check licensing on any free alternative. |
| [GDELT](https://www.gdeltproject.org) | Free | Global news events with timestamps. Large and noisy. |
| [Tiingo News](https://www.tiingo.com) | Paid | Ticker-tagged financial news with an affordable tier. |
| [Loughran–McDonald (Notre Dame SRAF)](https://sraf.nd.edu) | Free | Finance-specific sentiment word lists and parsed 10-X files. |
| [FinBERT on Hugging Face](https://huggingface.co/ProsusAI/finbert) | Free | Baseline financial sentiment model. |

## Success criteria

### Works (correct and reproducible)

- [x] An automated test proves every signal timestamp precedes its trade timestamp. (AC-W-01: `tests/property/test_gate.py` and the full-data check)
- [ ] Event-study plots with confidence bands for each method.
- [ ] An ablation table comparing dictionary, FinBERT and LLM on the same events.

### Impresses hirers

- [ ] Incremental alpha over SUE and known factors with t > 3, net of costs.
- [ ] A contamination analysis: performance before vs after the model cutoff and with masked text, showing how much apparent alpha disappears.
- [ ] Cost per signal and a realistic latency budget for live use.
- [ ] A writeup that engages with the look-ahead-bias literature (for example Glasserman and Lin, 2023).

## Questions this project must answer

- How do you know the model didn’t already know what happened after the filing?
- Is your signal just earnings surprise in disguise?
- What is the capacity of an 8-K-driven strategy?

## Quick start

Data commands run under WSL, with data under `DATA_ROOT` on the ext4 filesystem. Copy `.env.example` to `.env` and fill it in first.

```bash
uv sync
uv run lta data smoke
uv run lta verify all
```
