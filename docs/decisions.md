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
