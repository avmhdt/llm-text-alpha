<!-- command=lta eval latency | source=lta repro | provider=licensed -->

| scorer | fetch_parse_p50_ms | score_p50_ms | decision_p50_ms | total_p50_ms | total_p95_ms |
|---|---|---|---|---|---|
| lm | 375.679 | 1.47039 | 0.001418 | 377.15 | 498.551 |
| finbert | 375.679 | 1075.3 | 0.001418 | 1450.98 | 2826.44 |
| llm | 375.679 | 1303.27 | 0.001418 | 1678.95 | 2224.07 |
| llm_llama | 375.679 | 32589.1 | 0.001418 | 32964.8 | 47185.3 |
