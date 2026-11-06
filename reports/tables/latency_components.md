<!-- command=lta eval latency | source=lta repro | provider=licensed -->

| component | measured | n | p50_ms | p95_ms | p99_ms |
|---|---|---|---|---|---|
| EDGAR dissemination (acceptance to public) | False | 0 |  |  |  |
| EDGAR fetch and parse (20 live filings) | True | 20 | 375.679 | 494.101 | 546.248 |
| gate and order decision | True | 1000 | 0.001418 | 0.00153306 | 0.00303605 |
