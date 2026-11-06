# LLM and NLP alpha with point-in-time discipline

> Score filings, transcripts and news with language models, and prove the result is not leakage or memorization.

**Status:** complete (M1–M7). This README is the research memo. Every number in it comes from a table in [`reports/tables/`](reports/tables) that `lta repro --offline` rebuilds byte for byte. Choices made along the way are in [docs/decisions.md](docs/decisions.md).

| | |
|---|---|
| Track | Quant Researcher |
| Estimated effort | 6–8 weeks |
| Stack | Python, PyTorch, Transformers, LLM APIs |
| Data cost | Free → Paid |
| Strongest for | Systematic equity, data-science-heavy funds |

## Summary

On 44,464 earnings press releases (8-K Item 2.02) from large US companies, 2016–2026:

- **Language-model scores carry a little information, but none of it survives trading costs.**
  - Claude Haiku 5.5's score is related to returns beyond earnings surprise, size and momentum: t = 2.98 at 20 days.
  - A long-short portfolio on that score earns −17% a year net of costs (t −1.16).
  - The pre-registered test, which counts only filings after the model's training cutoff, gives t = −0.51. That is far from the t > 3 bar.
- **Contamination: none measurable.**
  - Three models with different training cutoffs show no drop in predictive power after their cutoffs, beyond what the dictionary and FinBERT show on the same events.
  - The models do remember filings, and recognize most companies even with names masked, but they do not remember returns.
- **What looks like signal is mostly earnings surprise.** The LLM score's correlation with the announced earnings surprise is 0.51, against 0.11 for the dictionary.

A null result, reported with the evidence, is the outcome.

## Requirements

- [x] A text corpus with exact publication timestamps (SEC acceptance time, call time, news publish time). SEC acceptance times for 71,313 Item 2.02 8-Ks; transcripts are out of scope (D-07), and GDELT news was not ingested (FR-25 is a Should; see `docs/decisions.md`). (FR-01, FR-02, FR-06)
- [x] Trades at the next available bar after each timestamp; returns labelled as abnormal returns against a factor model. (FR-07, FR-15, AC-W-01, AC-W-07)
- [x] Baselines first: the Loughran–McDonald dictionary and FinBERT. Then LLM scores or embeddings. LM and FinBERT on 44,464 Item 2.02 releases (M3), then Claude Haiku 5.5 on 21,030 (M4). (FR-08, FR-09, FR-10)
- [x] An event study of cumulative abnormal returns around each event, plus cross-sectional regressions controlling for size, momentum and earnings surprise (SUE). (FR-16, FR-17, AC-W-02, AC-W-08)
- [x] Contamination controls: test only after each model’s training cutoff, and rerun on text with company names and dates masked. Three models with different cutoffs (Haiku 5.5, Haiku 4.5, Llama 3.1 8B), each on raw and masked text; masking audit and review in `docs/contamination_protocol.md`. (FR-14, FR-20, AC-W-09)
- [x] Inference cost and latency tracked per document. Every score has `cost_usd`; synchronous, local, FinBERT and dictionary scores have `latency_ms`; batch calls have their turnaround. (FR-13, AC-W-10)

## Success criteria

### Works (correct and reproducible)

- [x] An automated test proves every signal timestamp precedes its trade timestamp. (AC-W-01: `tests/property/test_gate.py` and the full-data check)
- [x] Event-study plots with confidence bands for each method. LM, FinBERT, the Haiku 5.5 LLM and SUE, each with t and firm-bootstrap 95% bands (`reports/figures/event_study_*`). (FR-16, AC-W-02)
- [x] An ablation table comparing dictionary, FinBERT and LLM on the same events. 20,110 identical events, every single scorer and combination, with CAR spread, IC, controlled coefficient, net alpha and cost per signal (`reports/tables/ablation.md`). (FR-19, AC-W-03)

### Impresses hirers

- [ ] Incremental alpha over SUE and known factors with t > 3, net of costs. **Not met.** The pre-registered test (Haiku 5.5 after its cutoff, H = 5, net of 1x costs, FF5 + UMD + PEAD) gives t = −0.51 over 60 days. No scorer has positive net alpha on the full sample either (`reports/tables/portfolio_alpha.md`). (FR-18, AC-H-01)
- [x] A contamination analysis: performance before vs after the model cutoff and with masked text, showing how much apparent alpha disappears. Result: no model shows measurable apparent alpha to lose. Every DiD against LM and FinBERT contains zero. The models recognize most companies from masked text, and remember the reported quarter of pre-cutoff filings but not returns (`reports/tables/contamination*.md`). (AC-H-02)
- [x] Cost per signal and a realistic latency budget for live use. $0.0008 per traded signal for Haiku 5.5 and $0.0087 for Haiku 4.5. Acceptance to order is 0.4 s (dictionary) to 1.7 s (Claude, sync) p50, plus EDGAR dissemination, which does not bind with entry at the next open (`docs/latency_budget.md`). (FR-22, AC-H-03)
- [x] A writeup that engages with the look-ahead-bias literature (for example Glasserman and Lin, 2023). See [Look-ahead bias and memorization](#look-ahead-bias-and-memorization). (AC-H-04)

## Hypothesis and problem

Earnings press releases are public the moment the SEC accepts them. If a language model reads tone, surprise and guidance better than a word list, its score should predict abnormal returns after the release. That edge should survive earnings surprise (SUE), standard factors and trading costs.

Two traps make this easy to get wrong:
- **Timing.** A signal must never use a price from before the text was public.
- **Memory.** A model trained on later news may "know" how the stock moved, so its score on an old filing measures memory, not reading.

The project builds the corpus with exact acceptance times and enforces a gate between text and returns. It pre-registers the main test ([docs/preregistration.md](docs/preregistration.md)) and tests memory directly.

## Data

| Source | What it gives | Licence and use |
|---|---|---|
| SEC EDGAR (index pages, headers, documents) | 71,313 Item 2.02 8-Ks, 2016-01 to 2026-09, with acceptance timestamps | Public; fetched at 8 req/s with a User-Agent |
| Sharadar Full History, through equity-factor-pipeline's `eqpit` point-in-time store | Prices, market caps, universe, first-reported EPS | Personal-use licence; aggregates only are published |
| Kenneth French Data Library | FF5 and momentum, daily | Free with citation |
| Loughran–McDonald Master Dictionary (SRAF) | Word lists | Used under the owner's ruling (P-14); never committed |
| FinBERT (`ProsusAI/finbert` at a pinned revision) | Baseline sentiment model | Model card licence |
| Claude Haiku 5.5 and Haiku 4.5 (Anthropic API) | LLM scores | Commercial terms; responses cached, never committed |
| Llama 3.1 8B Instruct, GGUF Q4_K_M (llama.cpp on CPU) | Open-weight model with a published cutoff | Llama 3.1 Community License |

The scorable set is 44,464 releases from the top 1,000 US stocks by market cap at the month end before the event. Event time is the SEC acceptance timestamp converted to UTC. It was checked against the submissions JSON on every filing, across all 21 daylight-saving changes (DQ-02). Every table's header line names its provider label; for data it is `licensed`. Details, access dates and volumes are in [docs/data_sources.md](docs/data_sources.md).

## Method

Full detail is in [docs/methodology.md](docs/methodology.md).

- **Timestamp gate.** Entry is at the first regular-session open strictly after acceptance plus 5 minutes. Controls are as of the day before entry. The gate is the only code that joins text to returns, and a property test plus a full-data check show every signal precedes its trade (AC-W-01).
- **Abnormal returns.** Expected return is FF5 + momentum, estimated over sessions −250 to −30. Day 0 is open-to-close from the entry open.
- **Scorers.**
  - Loughran–McDonald net tone.
  - FinBERT on the first seven 512-token chunks.
  - Claude Haiku 5.5 with a fixed, versioned prompt (`score_v1@190b8b18`), structured JSON output, no sampling parameters and thinking off. It reads the first 7,050 characters of the EX-99 press release.
  - Scores are standardized against each scorer's trailing 252 sessions and split into trailing quintiles.
- **Tests.**
  - Event studies by quintile with t, BMP and firm-bootstrap bands, plus the Kolari–Pynnönen correction.
  - Two-way clustered and Fama–MacBeth regressions of CAR on the score with size, momentum and SUE, including a SUE-orthogonalized score.
  - A calendar-time long-short portfolio (quintile 5 minus quintile 1, held 5 or 20 days), net of costs of 5 to 20 bps one way by size, with alpha against FF5 + UMD + a PEAD factor (Newey–West, 10 lags).
  - An ablation on identical events, and a capacity curve.
- **Contamination protocol.** Three models with different cutoffs score raw and masked text. A 2×2 (before/after cutoff × raw/masked) is compared with a difference-in-differences against the baselines, plus re-identification and outcome-recall probes ([docs/contamination_protocol.md](docs/contamination_protocol.md)).

## Results

### Event studies and ablation

Q5 − Q1 cumulative abnormal return over [0, 5] sessions:

| Scorer | Spread | t |
|---|---|---|
| Dictionary | +0.05% | 0.51 |
| FinBERT | +0.02% | 0.19 |
| LLM | +0.21% | 1.37 |
| SUE | +0.36% | 3.64 |

Over [0, 60], the LLM's spread grows to +1.11% (t 3.05, bootstrap p 0.004). That window is before the model's cutoff, where memory is possible. See `reports/tables/event_study_*` and the figures in `reports/figures/`.

On the 20,110 events all three scored ([ablation](reports/tables/ablation.md)), no single scorer or combination has a significant CAR[0,5] spread; the largest t is 1.79, for LM + FinBERT. The LLM alone has the least negative net alpha (−12% a year, t −0.90), but no scorer or combination is positive net of costs.

### Is it earnings surprise in disguise?

Mostly, yes.
- The LLM score's Spearman correlation with the announced quarter's SUE is 0.51, against 0.22 for FinBERT and 0.11 for LM ([sue_correlation](reports/tables/sue_correlation.md)).
- With size, momentum and SUE controls, the LLM's coefficient on CAR[0,20] is 0.26% per standard deviation (t 2.98, two-way clustered), and the SUE-orthogonalized score gives t 3.50 ([regressions](reports/tables/regressions.md)). That is over the full sample, which includes years the model may remember.
- After Haiku 5.5's cutoff the same regression gives −0.69% (t −1.51; [regressions_post_h55](reports/tables/regressions_post_h55.md)).
- The portfolio's loading on the PEAD factor is 0.27 at H = 5.

### Portfolio alpha net of costs

[portfolio_alpha](reports/tables/portfolio_alpha.md), H = 5, costs at 1x:

| Scorer | Window | Net alpha a year | t | Gross alpha a year | t |
|---|---|---|---|---|---|
| LLM (Haiku 5.5) | full | −17.5% | −1.16 | +10.2% | 0.68 |
| LLM, **pre-registered** | after cutoff (60 days) | −34.6% | **−0.51** | −8.6% | −0.13 |
| FinBERT | full | −44.9% | −4.53 | −17.2% | −1.73 |
| LM | full | −32.3% | −3.63 | −4.2% | −0.47 |
| SUE (PEAD check) | full | −5.9% | −0.52 | +21.6% | 1.93 |

- Costs of 5 to 20 bps one way, paid on both entry and exit of 5-day holds, cost about 28% a year in every portfolio here.
- Even SUE, the strongest known signal here, does not cover them.
- The pre-registered criterion (t > 3) is **not met**.
- The trial log holds 433 trials, 32 of them portfolios. The headline portfolio's Deflated Sharpe Ratio is 0.18 ([trials](reports/tables/trials.md), [trials_dsr](reports/tables/trials_dsr.md)).

### Contamination

[contamination_cells](reports/tables/contamination_cells.md), [contamination_did](reports/tables/contamination_did.md), [contamination_probes](reports/tables/contamination_probes.md):

| Model (cutoff) | Raw before / after cutoff, CAR[0,5] spread | DiD vs LM / FinBERT | Names company from masked text | Names quarter, before / after | Recalls 5-day direction |
|---|---|---|---|---|---|
| Haiku 5.5 (Jun 2026) | +0.21% / −0.06% | −0.73% / −1.04% (CIs contain 0) | 75% | 15% / 3% | 45% |
| Haiku 4.5 (Jul 2025) | +0.10% / +0.22% | +0.23% / −1.46% (CIs contain 0) | 54% | 19% / 0.7% | 48% |
| Llama 3.1 8B (Dec 2023) | IC −0.026 / −0.048 (ties) | IC +0.139 / −0.009 (CIs contain 0) | 28% | 2% / 5% | 45% |

Every spread and DiD interval contains zero. Masked text does at least as well as raw. The models recognize companies and remember filings, but not returns.

### Cost, latency and capacity

- **Cost.** $0.00015 per document in batch for Haiku 5.5 and $0.0016 for Haiku 4.5, or $0.0008 and $0.0087 per traded signal ([cost_latency](reports/tables/cost_latency.md)).
- **Latency.** Acceptance to order is 0.38 s for the dictionary, 1.4 s for FinBERT, 1.7 s for Claude (sync) and 33 s for Llama on CPU (p50), plus EDGAR dissemination (about 40 s in 2014; not measured). 96% of releases arrive outside market hours and trade at the next open, so latency does not bind ([docs/latency_budget.md](docs/latency_budget.md)).
- **Capacity.** Net alpha is negative at every size, so there is nothing to scale. With market impact alone (10% of 20-day volume, square-root impact), the LLM's gross alpha falls to about zero between $100M and $300M per leg ([capacity](reports/tables/capacity.md), `reports/figures/capacity.png`).

## Look-ahead bias and memorization

Five strands of work bear on whether an LLM's "alpha" on old text is real.

- **Lopez-Lira and Tang (2023)** find that ChatGPT's scores of news headlines predict next-day returns, more so for small stocks and bad news. They test after the model's training cutoff to avoid look-ahead.
  - Here, on large-cap earnings releases traded at the next open after SEC acceptance, the LLM's next-day spread is +0.23% (t 1.97). After its cutoff the 5-day spread is −0.06%.
  - We **partly disagree**. The effect is far smaller in this setting, and it does not survive costs.
- **Glasserman and Lin (2023)** test GPT sentiment on anonymized headlines and find a "distraction effect": removing company names can improve predictions, because the model's general view of a well-known company crowds out the news.
  - We **agree in direction**. Masked text scores at least as well as raw in three of four Claude comparisons (Haiku 5.5 before its cutoff: +0.30% masked against +0.21% raw), though no difference is significant.
  - We **add a caveat**. Masking names does not hide the company: Haiku 5.5 names it from masked text 75% of the time, Haiku 4.5 54%, Llama 28%. Glasserman and Lin's anonymized arm should be read as "names hidden", not "identity hidden".
- **Sarkar and Vafa (2024)** show that pretrained language models carry information from after the date of the text they are asked about, which biases backtests.
  - Our probes find that memory: the Claude models name the reported quarter of 15–19% of masked pre-cutoff filings, against 1–3% after their cutoffs.
  - But asked how the stock moved over the next five days, every model is at or below chance (45–49%). The memory is of the filing, not of the return.
- **He, Lv, Manela and Wu (2025)** train chronologically consistent models, using only text available up to each date. They report that return predictability largely survives, so look-ahead bias in such tests is smaller than feared.
  - We **agree**. No model's DiD against the baselines excludes zero.
  - Our test is weaker, because the post-cutoff window for current models is short: 937 events over three months for Haiku 5.5.
- **Golchin and Surdeanu (2023)** detect training-data contamination by asking a model to reproduce or identify its inputs. Our re-identification probe is in that spirit, applied to masked SEC filings.
  - Its result (75% for the newest model) says entity masking alone is a weak anonymizer for well-known companies.

Earlier text-and-returns work frames the baselines. Tetlock (2007) shows media tone predicts returns. Loughran and McDonald (2011) build the finance word lists used here. Ke, Kelly and Xiu (2019) and Chen, Kelly and Xiu (2022) find text and LLM-embedding signals that predict returns at larger scale. Our dictionary and FinBERT baselines find no CAR[0,5] effect on large-cap earnings releases (t 0.51 and 0.19), consistent with information in these documents being priced quickly, mostly overnight before the next open.

## What failed

- **The pre-registered alpha test.** It was expected to be underpowered (60 days after Haiku 5.5's cutoff) and gave t = −0.51.
- **Every strategy net of costs.** Gross alphas are small, and 5-day holds pay costs four times a month.
- **FinBERT has negative gross alpha (t −1.73).** Its tone is priced overnight, before the open entry, and then partly reverses.
- **The first masking rules leaked.** Three review rounds were needed: names in other cases and spellings, brand words, dateline cities, lowercase brands. Even now, 2 of 50 reviewed releases still show a brand, and models re-identify most companies anyway.
- **Llama's quintile spread is undefined.** It scores 55% of releases at exactly 1.0. The trailing-quintile rule ranks ties low and was not changed after the fact; Llama is judged by IC.
- **The M3 first run put the estimation alpha into expected returns**, which produced a spurious drift. It was fixed and disclosed, and its trials are kept in the count.
- **GDELT news was not ingested.** The archive does not fit the 10 MB/s download cap. Transcripts were out of scope (no WRDS).

## Limitations

- **Universe.** Top 1,000 US stocks by market cap; small caps, where text effects are reported to be larger, are excluded. Delisting returns are not merged (DQ-14 drops windows more than 20% short).
- **Post-cutoff window.** For current Claude models it is short: about 1,000 events, with daily returns only to 2026-09-30.
- **Masking.** It removes names, not identity. Masked results are a lower bound on how much memorization was removed.
- **Bars.** Daily only; intraday reaction to releases during market hours (4% of them) is not used.
- **Remaining drift.** FF5 + momentum leaves a uniform −0.3% to −0.5% CAR[1,60] drift in every quintile; spreads are unaffected.
- **One prompt.** One prompt and one truncation length (7,050 characters) per model; other prompts were not searched, by design.

## How to reproduce

The PC used is described in [docs/hardware.md](docs/hardware.md): i9-14900K, 62 GB RAM, no GPU, Windows 11 with WSL2 Ubuntu 22.04. Data lives under `DATA_ROOT` on WSL's ext4 disk.

```bash
uv sync
```

```bash
uv run lta data smoke
```

```bash
uv run lta llm estimate --sample item202_main --model claude-haiku-5-5
```

```bash
uv run lta score llm --provider fake --sample item202_main --max-docs 50
```

```bash
uv run lta repro --offline
```

```bash
uv run lta verify all
```

`lta repro --offline` rebuilds:
- signals and abnormal returns;
- every event study, regression, portfolio, ablation, capacity, contamination, masking-audit, latency and trials table;
- the figures.

It works from the data and the cached LLM responses, calls no API (spend 0), and takes about 90 seconds once data is in place. Repeated runs leave `reports/tables` byte-identical (AC-W-13). Corpus, DQ-02 and scorer-benchmark tables come from `lta corpus build` and `benchmarks/bench_scorers.py`, which re-read raw EDGAR files or time the hardware; they are not part of the repro.

Building the data from scratch needs the keys in `.env.example` and runs these steps:
1. `lta data edgar-index`
2. `lta data edgar-fetch`
3. `lta corpus build`
4. `lta data prices`
5. `lta data earnings`
6. `lta data factors`
7. `lta data lm-dict`
8. `lta score dict`
9. `lta score finbert`
10. `lta score llm` per sample
11. `lta corpus mask`

The local model also needs `scripts/serve_local_llm.sh`.

## Trials and LLM spend

- The trial log ([results/trial_log.csv](results/trial_log.csv), trial-log contract v1.0) holds 433 trials: 32 portfolios with return series, 368 regressions, 14 ablation rows, 8 contamination runs, 7 event studies, 2 capacity runs and 2 latency runs. Every number above is one of them.
- LLM spend, from `lta llm spend`:

```text
claude-haiku-4-5: $10.0030
claude-haiku-5-5: $6.2175
committed $16.2205 of $25.00 (open holds included)
```

The budget was USD 25, enforced by the spend tracker before every paid request. The local model cost nothing in API terms.

## Questions this project answers

- **How do you know the model didn’t already know what happened after the filing?** Its predictive power does not fall after its training cutoff relative to the baselines on the same events. Asked directly, it cannot recall the post-filing return direction better than chance. It does remember filings (the quarter probe), so "didn't know anything" would be wrong; "didn't know the outcome" is what the evidence supports ([contamination protocol](docs/contamination_protocol.md)).
- **Is your signal just earnings surprise in disguise?** Largely. The score correlates 0.51 with announced SUE. It keeps a SUE-orthogonal component in the full sample (t 3.50 at 20 days), but that component disappears after the cutoff, and no portfolio built on it earns alpha net of costs.
- **What is the capacity of an 8-K-driven strategy?** On this evidence, zero net of costs. Gross of standard costs, market impact alone erodes the LLM's gross alpha to zero at roughly $100–300M per leg.

## Data folder

Raw and licensed data live under `DATA_ROOT` (default `~/data/llm-text-alpha`) and `data/`, both outside git. Only code, configs, aggregate tables, figures and the trial log are committed.
