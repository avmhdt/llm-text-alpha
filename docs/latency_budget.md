# Latency budget and cost per signal

FR-22 and AC-H-03. The tables come from `lta eval latency` (`reports/tables/cost_latency.md`,
`latency_budget.md`, `latency_components.md`). All measurements were taken on the reference PC
described in [hardware.md](hardware.md), while the owner's market-data recorders were running.

## From acceptance to an order

| Component | p50 | p95 | Source |
|---|---|---|---|
| EDGAR dissemination: from acceptance to the filing on sec.gov | ~36–40 s in 2014; seconds after the SEC's fix | — | Not measured (D-13 skipped the live poller). Rogers, Skinner and Zechman (2017), "Run EDGAR Run: SEC Dissemination in a High-Frequency World", *Journal of Accounting Research* 55(2): Form 4 filings reached the public EDGAR site a mean of about 40 s (median about 36 s) after acceptance in 2014, and paying feed subscribers often got them first. The SEC then changed dissemination so that filings appear on its site first. |
| Fetch and parse one Item 2.02 filing (index page plus documents, through the 8 req/s limiter) | 374 ms | 427 ms | 20 live filings, 2026-11-06 |
| Score: Loughran–McDonald | 1.5 ms | 4.4 ms | 44,464 production documents |
| Score: FinBERT (CPU, 8 threads, 7 chunks) | 1.08 s | 2.33 s | 44,464 production documents |
| Score: Claude Haiku 5.5, synchronous | 1.30 s | 1.73 s | 200 synchronous calls |
| Score: Llama 3.1 8B (CPU, 8 threads) | 32.6 s | 46.7 s | 2,426 local calls |
| Gate and order decision | 1.4 µs | 1.5 µs | 1,000 timed calls |

Acceptance to order, before dissemination, as p50 / p95:

| Scorer | Total |
|---|---|
| Dictionary | 0.38 s / 0.43 s |
| FinBERT | 1.4 s / 2.8 s |
| Claude Haiku 5.5 (sync) | 1.7 s / 2.2 s |
| Llama 3.1 8B (CPU) | 33 s / 47 s |

Batch scoring is not live: its batches took up to 11 minutes (Haiku 5.5, 659 s) and 6 minutes
(Haiku 4.5, 354 s).

## What the budget means for this strategy

The strategy trades at the first regular-session open strictly after acceptance plus 5 minutes
(DQ-04). Only 4.0% of Item 2.02 releases are accepted during the regular session; the rest
arrive before the open or after the close. For those, any scorer here, Llama on a CPU
included, finishes hours before the order can trade. For the 4% that arrive intraday, the
5-minute buffer covers dissemination plus every scorer except the local CPU model at its tail.
Latency is therefore not what limits this strategy; costs and signal strength are
([portfolio_alpha](../reports/tables/portfolio_alpha.md)).

A strategy that traded within seconds of the release would need the dictionary or a
synchronous API model, and its first constraint would be EDGAR dissemination, not scoring.

## Cost per signal

| Scorer | USD per document, batch | USD per document, sync | Share of events traded | USD per traded signal |
|---|---|---|---|---|
| Dictionary | 0 (local CPU) | 0 | 39.6% | 0 |
| FinBERT | 0 (local CPU) | 0 | 39.9% | 0 |
| Claude Haiku 5.5 | $0.00015 | $0.00029 | 37.0% | $0.0008 |
| Claude Haiku 4.5 | $0.0016 | $0.0032 (twice the batch rate; no sync calls made) | 36.4% | $0.0087 |
| Llama 3.1 8B | 0 (local CPU) | 0 | 23.0% (ties, see the contamination protocol) | 0 |

"Traded" means the event entered the top or bottom trailing quintile. Local scorers cost
electricity and machine time, which are not priced here: FinBERT takes about 14 hours per 50,000
documents on 8 threads, and Llama about 9 hours per 1,000.
