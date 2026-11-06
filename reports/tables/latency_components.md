<!-- command=lta eval latency | run_id=f5c27e5c-6a54-4691-824d-858b2503d787 | git_sha=7fefad323e988cd2b058591f8c52259ca13a5329 | provider=licensed -->

| component | measured | n | p50_ms | p95_ms | p99_ms |
|---|---|---|---|---|---|
| EDGAR dissemination (acceptance to public) | False | 0 |  |  |  |
| EDGAR fetch and parse (20 live filings) | True | 20 | 373.586 | 426.708 | 478.777 |
| gate and order decision | True | 1000 | 0.001423 | 0.00151805 | 0.00228957 |
