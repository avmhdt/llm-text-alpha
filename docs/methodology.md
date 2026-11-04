# Methodology

## Timestamps and the timestamp gate

This section answers the interview question in section 17 of the spec: why acceptance time rather than filing date, and what happens to a filing accepted at 16:05 ET.

### Event time (DQ-01)

Every EDGAR document's event time is the **acceptance datetime** in its SGML header (`<ACCEPTANCE-DATETIME>`, `YYYYMMDDHHMMSS`). EDGAR writes it in New York local time. `corpus/timestamps.py` attaches `America/New_York` with `zoneinfo`, so the EST or EDT offset is the one in force on that date, and converts to UTC. All timestamps are stored as `Datetime("ns", "UTC")`.

The filing date is never used for timing. EDGAR gives a filing accepted after 17:30 ET the next business day's filing date, so the filing date can be later than the time the text became public. It can't be earlier, but it carries no time of day, so it can't place an event before or after the open.

The raw header string is kept in `acceptance_raw`, so each conversion can be audited. One edge case is handled by rule. In the hour repeated at the November fall-back, the daylight-time reading is taken. EDGAR doesn't accept filings at that hour, so the choice can't move a real filing.

### Second source (DQ-02)

The submissions JSON on `data.sec.gov` gives `acceptanceDateTime` with a `Z` suffix. Read as true UTC, it should equal the header time converted from New York time. `lta corpus build` compares the two on every fetched filing and writes three tables:

- [`reports/tables/dq02_acceptance_by_year.md`](../reports/tables/dq02_acceptance_by_year.md): agreement by year and by EST or EDT.
- [`reports/tables/dq02_acceptance_by_transition.md`](../reports/tables/dq02_acceptance_by_transition.md): filings within 7 days of each DST change from 2016 to 2026.
- [`reports/tables/dq02_acceptance_examples.md`](../reports/tables/dq02_acceptance_examples.md): the filing nearest each DST change on both sides, about two per change across 21 changes, which clears DQ-02's minimum of 20.

On any disagreement the header wins, since DQ-01 makes it the authoritative source. The M1 smoke test recorded the first ten comparisons in `configs/smoke.yaml`, and all ten agreed.

### Entry bar (DQ-04, FR-07)

`corpus/gate.py` maps each event to its entry bar. The entry bar is the first XNYS session whose open (09:30 ET) is **strictly later** than `event_ts_utc + buffer`, with a default buffer of 5 minutes. The trade enters at that open.

| Accepted (ET) | Entry |
|---|---|
| 08:00 | Same day's open, since 08:05 is before 09:30 |
| 09:25:00 | Next session's open, since 09:30 is not strictly later than 09:30 |
| 09:31 | Next session's open |
| 16:05 | Next session's open. The close isn't tradable for this strategy, which enters only at opens |
| Friday 15:34 before Good Friday, or any holiday eve | The next session after the holiday |
| After a 13:00 early close | The next session's open |

Holidays and early closes come from `exchange_calendars` (XNYS), wrapped in `corpus/calendar.py`. Every session from 1990 opens at 09:30 ET, early-close days included. Daily open prices are the only entry price (section 2), so an intraday event can't trade before the next open.

### Controls (DQ-08)

The gate takes each control from the latest row dated strictly before the entry date. That is a backward as-of join on `entry_date - 1 day`. The `controls_asof` column stays in `signals`, so the check can be repeated on any table.

### Proof (AC-W-01)

`assert_no_lookahead` raises `LookaheadError` if any row has `entry_ts_utc <= event_ts_utc + buffer`, an `entry_ts_utc` off its `entry_date`, or `controls_asof >= entry_date`. `apply_gate` runs it on every table it writes, and it is the only writer of `signals`.

`tests/property/test_gate.py` checks it from three sides:

- `test_signal_precedes_trade`: 10,000 hypothesis cases. Any second from 1990 to 2027, in any UTC offset from −12:00 to +14:00, with buffers from 0 to 6 hours. Named examples cover 09:31, 08:00 and exactly open minus buffer, Good Friday, an early close, and both DST Sundays. Each case also checks that the entry is the **earliest** qualifying open.
- A batch of 100,000 random events through the vectorized path, with 200 of them checked against the scalar path.
- `test_controls_dated_before_entry`: random controls, checking that every `controls_asof` is the latest date before entry.

`lta verify ac-w-01` runs the suite. The full-data part, which checks every real signal, joins in M3 once there are signals.

### Corpus (FR-02, FR-06, DQ-05, DQ-13)

- **Scope:** an 8-K enters the corpus when its header lists Item 2.02 and it was accepted from 2016-01-01 to 2026-09-30, counting days in New York time.
- **Text:** the primary 8-K document, then each EX-99.x exhibit in filing order, each under a tag such as `[[EX-99.1]]`. Hidden inline-XBRL headers are dropped and inline-XBRL tags are unwrapped. Each table row becomes one pipe-delimited line.
- **Extraction:** it succeeds when the text has at least 200 characters. Failures keep their reason. Counts are in [`reports/tables/corpus_item202.md`](../reports/tables/corpus_item202.md).
- **Duplicates and amendments:** an exact duplicate of an earlier text (same `text_sha256`) is marked `excluded`, and so is any amendment. The gate drops both. When one firm has several documents entering on one date, the gate merges them into a single event: the mean score and the earliest timestamp.
- **Identifiers:** each document is linked as of its event date through eqpit's `PitStore`. `in_universe` is DQ-06's eligibility: the company ranked in the top 1,000 at the month end before the event.

## Returns, controls and baselines (M3)

### Prices and universe (FR-03, DQ-06, DQ-07)

`lta data prices` reads equity-factor-pipeline's Sharadar-built store through `PitStore` (D-01: `PRICE_PROVIDER=auto` resolves to `efp`). It writes three tables:

- `universe_monthly`: the top 1,000 at each month end.
- `link_table`.
- `prices_daily`: every security in the universe at any month end from December 2015 to September 2026, with prices from 2015-01-02.

`open` and `close` are unadjusted. `ret` is the split- and distribution-aware total return, and the entry day earns `close / open - 1`. eqpit exposes daily delisting returns only inside its monthly labels, and Sharadar has no delisting-return field. So daily returns stop at the last trade, and a window more than 20% short is dropped (DQ-14).

### Controls (DQ-08)

Each control is the value known at the close of its date. The gate takes the row dated before entry.

- `size_log`: log market cap.
- `mom_12_2`: compounded return over sessions 252 to 22 before the date, the daily form of months t-12 to t-2.
- `bm`: the latest public book equity over market cap.
- `sue`: the latest SUE whose EPS filing was public by the close.

`sue_announced` is the SUE of the quarter the 8-K announces. Its EPS appears in the press release, but the value used here comes from the 10-Q or 10-K filed later. That makes it DQ-08's labelled exception, and it enters only as a robustness control.

### SUE (FR-05)

Quarterly diluted EPS comes from monthly `PitStore` snapshots, taking each quarter's value from the first snapshot that holds it. SUE is the seasonal random walk: the change from the same quarter a year earlier, over the standard deviation of the previous eight such changes, with at least six required.

A quarter's announcement is the company's last Item 2.02 8-K after the quarter end and no later than a day after its EPS filing. Without one, the EPS filing's time stands in.

### Abnormal returns (FR-15, DQ-09)

For each event, OLS of excess returns on a constant, FF5 and momentum over sessions -250 to -30 gives the betas, and `exp_ret = rf + beta' f`. The fitted intercept is not carried into the event window: a year of alpha before an event is mostly past performance, and adding it to 60 sessions of expected return pushes every CAR down for recent winners. Fewer than 120 valid returns falls back to the market model (`rf + mktrf`), and the event is flagged. `rel_day` 0 is the entry session.

### Event study (FR-16)

Quintiles come from trailing 252-session breakpoints of each scorer's own scores (DQ-10). CAARs cumulate from day -10. There are three bands:

- Cross-sectional t.
- Boehmer-Musumeci-Poulsen, converted to return units with the mean scale.
- A firm-clustered bootstrap with 1,000 draws.

The Kolari-Pynnonen adjustment uses the mean correlation of standardized estimation residuals. The pairs are sampled events whose estimation windows overlap by 60 sessions or more.

### Regressions (FR-17)

The dependent variable is `CAR[0,k]` for k = 1, 5, 20 and 60, regressed on `score_z` under five specifications:

1. Score alone.
2. Size and momentum.
3. Size, momentum and SUE.
4. All of those plus book-to-market.
5. Size and momentum with `sue_announced` in place of SUE.

Each runs as pooled OLS with standard errors clustered by firm and entry date, and as monthly Fama-MacBeth with Newey-West errors at lag 3. Both match `linearmodels` within NFR-03's tolerances (`tests/unit/test_regressions.py`).

### Baseline scorers (FR-08, FR-09)

When a filing has EX-99 exhibits, both baselines read those rather than the cover page.

- **Loughran-McDonald:** counts Positive, Negative, Uncertainty, Litigious and Constraining words. A positive word is not counted when "no", "not" or "never" appears among the three words before it. `score = (pos - neg) / (pos + neg + 1)`.
- **FinBERT:** `ProsusAI/finbert` at a pinned revision, over 512-token chunks that overlap by 64. It reads the first seven chunks, about 3,100 word pieces, close to the LLM's 3,000-token budget. The ablation therefore compares models on the same reading length. `score = P(positive) - P(negative)`.

## LLM scoring (M4)

### Request (FR-10)

Each request has three parts:

- **System prompt:** `configs/prompts/score_v1.md`, marked for prompt caching.
- **User turn:** the release's EX-99 text, cut back to the last line break before 7,050 characters. That is about 3,000 tokens at the 2.35 characters per token measured on 200 releases.
- **Output format:** the JSON schema of `LLMScore`.

The answer has six fields: `score` in [-1, 1], `label`, `surprise` (-1, 0 or 1), `guidance` (-1, 0, 1 or null), `materiality` and `confidence` in [0, 1]. Pydantic validates it. A refused, truncated or invalid answer is retried once, synchronously. Both attempts are kept in the cache file and both are paid for.

Haiku 5.5 runs with thinking disabled and effort low, and no sampling parameters (D-06). Determinism comes from the cache: an answer is generated once and reused.

### Cache and spend (FR-11, FR-12)

Every provider reads the response cache first. A hit costs nothing and writes no ledger entry.

For the requests that miss, the spend tracker reserves the worst case before sending anything:

- every input token at the higher of the input and cache-write rates, counting 2.0 characters per token;
- plus the full 512 output tokens;
- plus a synchronous retry for 2% of the requests.

If the reservation would take committed spend past USD 25, the run stops before its first request. When results arrive, the reservation is settled with the cost computed from each response's `usage`.

Batches are saved with their reservation, so a crashed run is collected, never resubmitted.

### Samples

| Sample | Events | Use |
|---|---|---|
| `item202_dev` | 200 releases before 2020 | Prompt checks; synchronous latency |
| `item202_main` | 20,000, stratified by year, seed 7 | Main run, raw and (M6) masked |
| `item202_post_h55` | Every event from 2026-07-01 (1,030) | Haiku 5.5's post-cutoff window, the primary test in [preregistration.md](preregistration.md) |
