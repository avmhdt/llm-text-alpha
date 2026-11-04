# Decisions

Owner decisions copied from `D:\trading-strategy-brainstorm\DECISIONS.md` on 2026-11-03, followed by choices made while building. `DECISIONS.md` overrides the spec, and the spec overrides `README.md`. API keys never go in this file.

## Owner decisions

### Plan and merging

- **Scope:** project 5 of 15, for researcher and developer job applications, built a few at a time with no fixed date. Third batch, started 2026-11-03, with `event-driven-backtester`, `stat-arb-pairs` and `regime-portfolio-construction`.
- **Merging:** agents push feature branches and open PRs. PRs merge with `gh pr merge --auto --squash` as `avmhdt` once CI is green. Branch protection and auto-merge are turned on once CI exists (M1).
- **Dependencies:** `equity-factor-pipeline` (done, M1 to M7) supplies the `eqpit` data layer through a git install. `backtest-overfitting-lab` owns trial-log contract v1.0 and the Deflated Sharpe code.

### Identity and access

- **GitHub:** `gh` active account is `avmhdt`, with admin on the repo. Actions is enabled (P-16).
- **SEC (P-01):** `SEC_USER_AGENT` is the owner's handle and contact email, set in `.env`.
- **WRDS (P-04):** no access. CRSP, Compustat and Capital IQ paths stay off, so `PRICE_PROVIDER=crsp` is refused and price history starts in 1998.
- **D-07 transcripts:** out of scope, since Capital IQ needs WRDS.

### Data

- **D-01 Sharadar:** Full History Bundle bought directly from Sharadar, not through Nasdaq Data Link.
  - Base URL `https://api.sharadar.com/v1.0`, key in `SHARADAR_API_KEY`, sent in the `x-api-key` header so it stays out of URLs and logs.
  - Tables `stocks` (SEP), `funds` (SFP), `fundamentals` (SF1), `tickers`, `actions`. Bulk zips with `years=full`.
  - `actions` has no delisting-return field.
- **Licence:** Personal Use License. Sharadar replied in writing on 2026-10-31 that public code, aggregate results, job applications, aggregate cross-checks and hosted coding tools are allowed. Publishing stays aggregate-only, with no security-level data.
- **D-05 local model:** Llama 3.1 8B Instruct; the owner's Hugging Face access was granted 2026-10-31. Llama 4 isn't suitable: its cutoff is later (August 2024) and it is too large for this CPU-only PC.
- **D-08 Tiingo:** not approved. GDELT is the only news source.

### Money

| Item | Amount | Status |
|---|---|---|
| Sharadar Full History Bundle | Existing subscription | Held |
| LLM API (D-02) | USD 25 | Approved |
| Everything else | USD 0 | Not approved |

- **D-03:** `claude-haiku-5-5` approved as the primary scoring model. Its prices match Anthropic's pricing table dated 2026-10-06.
- **D-06:** acknowledged. Claude requests omit temperature; determinism comes from the response cache.

### Machine

- **Hardware (P-09):** no NVIDIA GPU, so FinBERT and the local LLM run on CPU. Details in [hardware.md](hardware.md).
- **Toolchain (P-11):** WSL2 Ubuntu 22.04.5 with `uv` and Python 3.11 to 3.13; native Windows with Python 3.12.
- **Storage (P-10):** data lives under `DATA_ROOT` on the WSL ext4 filesystem, never under `/mnt/*`, on `C:` or in git. Free space is checked at each milestone.
- **Shared PC:** live recorders run here (a crypto capture whose 14-day count started 2026-11-03, a Deribit cron, Tardis downloads). Downloads and CPU stay modest, every download is capped at 10 MB/s, and WSL is never restarted, the PC never rebooted, and `.wslconfig` never edited.

### Accepted

Spec defaults stand for D-09 to D-13 and D-15 (MIT licence).

## Choices made while building

| Milestone | Topic | Choice | Why |
|---|---|---|---|
| M1 | EDGAR rate limit | The token bucket holds one token, so requests are spaced at least `1/SEC_MAX_RPS` apart and shared by all threads | A larger bucket lets a burst plus the refill exceed the cap inside a one-second window. Learned in equity-factor-pipeline |
| M1 | Measured rate | The smoke check allows 0.1% over `SEC_MAX_RPS` | Send times are read just after the limiter's slot, so clock reads alone can push the measured rate a hair over the spacing the limiter enforces |
| M1 | EDGAR retries | HTTP 403, 429, 500, 502, 503, 504 and transport errors retry up to 6 attempts, with exponential backoff from 1 s plus up to 100% jitter, capped at 60 s, and at least `Retry-After` | FR-01 and NFR-07. SEC answers fair-access breaches with 403 |
| M1 | Smoke filings | Ten Item 2.02 8-Ks, recorded from their SGML headers on 2026-11-03, in `configs/smoke.yaml` | They cover EST and EDT, before the open, during the session and after the close, a Good Friday (2016-03-25), the 2018-07-03 early close, the weeks around DST changes, and filer-agent accession prefixes. 2016 to 2026 |
| M1 | DQ-02, first evidence | All ten header times agree with the submissions JSON `acceptanceDateTime` read as UTC | Matches equity-factor-pipeline's finding. The full check on 20 or more filings across DST changes is M2's |
| M1 | M1 price source | `PRICE_PROVIDER=auto` resolves to `efp`: equity-factor-pipeline's Sharadar-built store, read only through `eqpit.store.PitStore` | D-01's order. The store sits at `D:\trading-strategy-brainstorm\equity-factor-pipeline\data`, so WSL reads it through `/mnt/d`. That is a read of another project's data; nothing of this project's is written there |
| M1 | Which eqpit stores count | `efp` accepts a store built from `sharadar` or `wrds` and refuses synthetic, free or labelled builds | A synthetic store would otherwise pass the smoke test as real prices |
| M1 | Smoke universe | The 50 largest names in eqpit's universe at the month end before the window | Picks securities with nothing from inside the window |
| M1 | Sharadar key check | One `stocks` row for `GME` | AAPL and MSFT answer with no key at all. Learned in equity-factor-pipeline |
| M1 | Direct Sharadar prices | Arrive in M3 | M1 only needs one month, which `efp` serves. M3 builds `prices_daily`, `universe_monthly` and `link_table` |
| M1 | Pinned git dependencies | `eqpit` at `2ee3583` and `overfitlab` (dev only) at `ab5bf0c`, both through `[tool.uv.sources]` | Neither is on PyPI. A pinned commit keeps the lock reproducible |
| M1 | Trial-log validation | CI runs `overfitlab trials validate results/trial_log.csv`, and a test checks the writer's output with `overfitlab`'s validator | The contract asks for it; installing `overfitlab` as a dev dependency removes the skip |
| M1 | Trial-log columns | The ten required columns, then `x_run_kind`, `x_periods_per_year` and `x_note`. Rows with a series carry `x_periods_per_year = 252` | Section 7b names `x_run_kind`. The other two follow the contract's reference writer |
| M1 | Committed trial log | Header only at M1 | AC-W-11 checks the header and the writer; the first evaluation run is in M3 |
| M1 | Dependencies | Heavy packages (torch, transformers, spaCy, anthropic, statsmodels, linearmodels, matplotlib, selectolax, lxml) join in the milestone that first uses them. The `[gpu]` extra arrives with torch in M3 | Keeps M1's CI small. Section 4's versions still apply when they land |
| M1 | `[dev]` | A uv dependency group named `dev`, not an extra | uv installs it by default with `uv sync` |
| M1 | Lockfile in CI | `uv sync --locked` rather than `--frozen` | It also fails when `uv.lock` is stale |
| M1 | `LLM_PROVIDER` default | Unset means `fake` in code. `.env.example` keeps `anthropic`, as section 3 lists | Nothing paid can run by accident; CI also forces `fake` |
| M1 | Data placement | Data commands refuse a data directory under `/mnt/*` on Linux, and on Windows refuse a `DATA_ROOT` starting with `~`, which would land on `C:` | Section 7d and the owner's rule |
| M1 | Two virtual environments | Windows uses `.venv`; WSL uses `~/.venvs/llm-text-alpha` through `UV_PROJECT_ENVIRONMENT` | Both run from the same checkout on `D:`; one shared `.venv` would break both |
| M1 | Python | `.python-version` pins 3.12 | Section 4 and the CI matrix |
| M1 | `lta data smoke` config | Defaults to `configs/smoke.yaml`, which holds the `smoke` section. Other commands default to `configs/default.yaml` | Section 6 describes `smoke.yaml` as the 10-filing, one-month config |
| M1 | Logs | `logs/<command>-<run_id>.jsonl` in the repo, git-ignored | Section 6 places `logs/` at the top of the repo. Logs are small, and a redactor scrubs secrets first |
| M1 | Trading calendar | `eqpit.calendar` (XNYS from `exchange_calendars`) until M2's `corpus/calendar.py` | The same calendar equity-factor-pipeline validated |
| M1 | Config files for later milestones | `models.yaml`, `llm_prices.yaml`, `masking.yaml` and the prompts are written in M4 and M6 | Their values must be verified from official sources with a retrieval date when they are used |
| M1 | French factors | Daily `F-F_Research_Data_5_Factors_2x3_daily` and `F-F_Momentum_Factor_daily`; -99.99 and -999 read as missing; momentum must cover every FF5 date | FR-04. The 2026-11-03 download runs from 1963-07-01 to 2026-09-30 |
| M1 | Secret scanning | gitleaks in pre-commit and as a CI job | NFR-05 |
| M2 | Universe CIKs for the ingest | Every company eqpit ranks in the top 1,000 at any month end from 2015-12-01 to 2026-09-30: 2,113 CIKs. Companies keyed by a Sharadar id, with no CIK, cannot be matched to EDGAR | DQ-06 eligibility is checked per event later; the ingest takes the union |
| M2 | Item 2.02 selection | From the submissions JSON `items` field, then confirmed from each filing's own header when the corpus is built. Original 8-Ks only | Amendments are excluded from signals (DQ-05), so fetching them would cost requests for nothing |
| M2 | Filing window | Filing dates from 2016-01-01 to 3 days after 2026-09-30; the corpus keeps events by acceptance time in New York days | A filing accepted late on 2026-09-30 carries the next business day's filing date |
| M2 | One request per header | `<accession>-index-headers.html` gives the SGML header and the document list with types, so the primary document and EX-99.x exhibits need no index page | About 3.3 requests per filing instead of 4 or 5 |
| M2 | Fetch threads | Three threads share the one 8 req/s limiter | One thread reached about 4 req/s, since each request waits about 250 ms. Threads hide that latency; the limiter still caps the total (NFR-07) |
| M2 | Byte cap on SEC downloads | Each download streams in 64 KB chunks through the 10 MB/s throttle | The first version throttled after a whole file, so one file could burst above the cap |
| M2 | Limiter clock | `time.perf_counter` | Windows' `time.monotonic` ticks at about 15 ms, which put a measured rate at 10.04 against a cap of 10 |
| M2 | Text of an Item 2.02 8-K | Primary document, then EX-99.x exhibits in filing order, each under `[[TYPE]]` | DQ-13. The tags let scorers pick the press release |
| M2 | Extraction rule | Text of at least 200 characters counts as extracted. Failures keep a reason: `header_not_item202`, `no_documents`, `empty_text`, `short_text` | FR-02 |
| M2 | SGML lines inside exhibits | `<TYPE>`, `<SEQUENCE>`, `<FILENAME>` and similar lines are removed before parsing | Some exhibit files start with them, and they would enter the text |
| M2 | Duplicates | A text whose SHA-256 matches an earlier document's is `excluded`; so is any amendment | DQ-05 |
| M2 | Identifier link | Through `PitStore` as of the event: the company's top-ranked security at the prior month end, else its first common security valid that day. `permno` is the number in eqpit's security id (Sharadar's permaticker for `SHR:<n>`) | DQ-15. Section 7b allows a provider id where there is no CRSP PERMNO. eqpit's `permaticker` column is empty in its synthetic store, so the id is the one source both stores share |
| M2 | `in_universe` | True when the company is in the top 1,000 at the month end before the event | DQ-06; the gate drops the rest |
| M2 | Entry rule edge | An event exactly `buffer` before an open enters the next session | DQ-04 says strictly later |
| M2 | Controls in the gate | A backward as-of join on `entry_date - 1 day`; `controls_asof` stays in `signals` | Lets `assert_no_lookahead` re-check controls on any saved table |
| M2 | Same-day events | Merged per firm, scorer and entry date: mean score (values sorted before summing), earliest timestamp and its `doc_id`, `n_docs` | DQ-05. Sorting before a grouped sum keeps the result bit-stable, as equity-factor-pipeline found |
| M2 | Gate property test | 10,000 hypothesis cases plus 100,000 vectorized events | AC-W-01 asks for 10,000 or more |
| M2 | DQ-02 tables | Generated by `lta corpus build`: by year and EST/EDT, by DST change, and the nearest filing on each side of every change | DQ-02 asks for 20 or more filings across DST changes; the tables hold every fetched filing |
| M2 | Empty header acceptance time | A few EDGAR headers have an empty `<ACCEPTANCE-DATETIME>` (2 of the first 18,000 filings, such as Cal-Maine 0000016160-18-000093). Those filings take the submissions JSON `acceptanceDateTime` (UTC), marked `acceptance_source = submissions_json`, and stay out of the DQ-02 agreement tables. A filing with neither fails as `no_acceptance` | DQ-01 makes the header authoritative, but here it is blank. DQ-02 found the two sources agree on every filing that has both |
| M2 | SEC 503 bursts | A filing still failing after six attempts is left without a manifest, and `lta data edgar-fetch --resume` retries it | Five filings hit HTTP 503 around 20:11 UTC on 2026-11-03 |
| M2 | Filings with two filers in the universe | 147 accessions were fetched under two CIKs, such as Baker Hughes Co (1701605) and the older Baker Hughes entity (808362). The corpus keeps one copy per accession (the first filer's, else the lowest CIK) with every filer CIK in `ciks`. Linking picks the filer ranked highest in the universe at the prior month end, else the first filer with a common security that day | `doc_id` is keyed by accession and must be unique. The right company depends on the event date: 808362 was Baker Hughes Inc before the 2017 merger |
| M2 | Index duplicates | `lta corpus build` drops duplicate (accession, CIK) rows before joining header fields, and header fields come from one copy per accession | The join had added 58 duplicate rows across repeated builds (294,556 against 294,498) |
