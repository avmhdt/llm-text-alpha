# Pre-registration of the holdout evaluation

Decision D-14, requirement FR-29. This file is committed before any LLM score is evaluated
against returns. Its commit SHA goes into `params_json` of every later contamination run
(AC-H-07). Anything evaluated that this file does not name is exploratory and is labelled
that way in the memo.

Registered 2026-11-04, at the end of M4, by the build agent. The owner may edit it, but only
in a commit that comes before the first post-cutoff evaluation. Later changes go in an
"Amendments" section with their date and reason, and do not replace the original text.

## What has been seen before registration

- M3 results for the dictionary (LM), FinBERT and SUE scorers on the full 2016–2026 sample:
  event studies, regressions and the PEAD check (`reports/tables/`). The design below copies
  their specification rather than choosing one from those results.
- LLM scores on 200 pre-2020 releases (sample `item202_dev`), used only to check schema
  validity, refusals and the label distribution of prompt v1. No returns were joined to them.
- Nothing else. No LLM score has been joined to any return, abnormal return or portfolio.

## Fixed inputs

| Item | Value |
|---|---|
| Documents | 8-K Item 2.02 press releases (EX-99 exhibits, else the whole filing), linked, in the top-1,000 universe at the prior month end, not excluded |
| LLM | `claude-haiku-5-5`, thinking disabled, effort low, no sampling parameters (D-06); structured output with the schema in `src/llm_text_alpha/scoring/llm/schema.py` |
| Prompt | `configs/prompts/score_v1.md`, version `score_v1@190b8b18`; input truncated to the first 7,050 characters (3,000 tokens at 2.35 characters per token) |
| Score used | The `score` field, in [-1, 1]. The other fields are used only in exploratory analysis |
| Samples | `item202_main`: 20,000 events, stratified by year, seed 7. `item202_post_h55`: every event on or after 2026-07-01 (1,030), added because `item202_main` holds only 476 of them, under the spec's 500-event power flag |
| Signal | Through the timestamp gate: entry at the first XNYS open strictly after acceptance + 5 minutes; score standardized against the same scorer's trailing 252 sessions of events, with at least 50 earlier events; quintiles from that trailing distribution |
| Abnormal returns | FF5 + UMD, estimation window [-250, -30] sessions, expected return `rf + beta' f` |
| Costs | D-11: one-way 5 / 10 / 20 bps for size ranks 1–200 / 201–500 / 501–1,000, charged on entry and exit; 2x as sensitivity |
| Windows | Post-cutoff windows from `configs/models.yaml` (DQ-12): Haiku 5.5 events from 2026-07-01, Haiku 4.5 from 2025-08-01, Llama 3.1 8B from 2024-01-01. Events end 2026-09-30, the corpus end. Daily returns run to the last date in `factors_daily` (2026-09-30 at registration; refreshed if the Ken French library updates before M5, with the date recorded) |

## Hypotheses and tests

### H1, primary (AC-H-01)

The Haiku 5.5 raw-text score predicts returns after its training cutoff, beyond SUE and known
factors, net of costs.

- Events: Haiku 5.5 post-cutoff window (`item202_main` and `item202_post_h55` together),
  raw text.
- Portfolio: each event whose trailing quintile is 5 enters long, quintile 1 enters short, at
  the entry open; equal weight within each leg; held H = 5 trading days; daily net returns
  after D-11 costs at 1x.
- Test: daily net long-short returns regressed on FF5 + UMD + a PEAD factor (long high-SUE,
  short low-SUE events, same entry and holding rule). Newey–West standard errors, 10 lags.
- Pass: t(alpha) > 3. If t <= 3, the result is recorded as not met, with the estimate and its
  t. The specification is not changed to pass.
- Reported alongside, not tested: 2x costs, H = 20, and the same portfolio for LM, FinBERT
  and SUE on the same events.

Expected power: about 1,000 events over about 60 trading days. A t above 3 would need a very
large effect, so H1 is likely to be recorded as not met even if the signal is real. That is
known now and is part of the registration.

### H2, the score is not SUE in disguise (AC-H-05)

- Regression: CAR[0,+5] and CAR[0,+20] on `score_z` with `size_log`, `mom_12_2` and `sue`,
  two-way clustered by firm and date, on the Haiku 5.5 post-cutoff events; the same on the
  full sample, reported as possibly contaminated.
- The SUE-orthogonalized score (residual of `score_z` on `sue` within each month) is reported
  with its coefficient and t.
- Pass: t > 3 on `score_z` at CAR[0,+5] with SUE in the model, post-cutoff.

### H3, contamination (AC-H-02)

Apparent predictive power of an LLM score is larger before the model's cutoff than after it,
and larger on raw than on masked text, by more than the same contrast for LM and FinBERT.

- Measure: Q5 minus Q1 CAR[0,+5] spread and rank IC, each with a 95% CI, in a 2x2 table
  (pre/post cutoff x raw/masked) for each of Haiku 5.5, Haiku 4.5 and Llama 3.1 8B.
- Share of apparent alpha that disappears: (pre_raw - post_masked) / pre_raw, and each margin.
- Difference-in-differences: the LLM's (pre - post) spread minus the same for LM and for
  FinBERT on the same events.
- Contamination is reported as found when the DiD's 95% CI excludes zero with pre > post.
  Cells with fewer than 500 events are flagged as underpowered and do not count as evidence
  either way.
- Probes: re-identification accuracy (company and quarter from masked text) and outcome
  recall (return direction after a named event, no text), pre vs post cutoff.

## Trial accounting

Every evaluation run, including exploratory ones, is a row in `results/trial_log.csv`. The
Deflated Sharpe Ratio of the H1 portfolio uses the total trial count (AC-H-08).

## Amendments

None.
