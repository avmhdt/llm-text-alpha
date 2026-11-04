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
