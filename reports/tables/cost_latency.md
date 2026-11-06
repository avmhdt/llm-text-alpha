<!-- command=lta eval latency | source=lta repro | provider=licensed -->

| scorer | model_id | usd_per_doc_batch | usd_per_doc_sync | n | p50_ms | p95_ms | p99_ms | batch_turnaround_max_s | traded_share | usd_per_traded_signal |
|---|---|---|---|---|---|---|---|---|---|---|
| lm | lm-LM_MasterDictionary@e2d13286 | 0 | 0 | 44464 | 1.47039 | 4.44843 | 7.53259 |  | 0.395824 | 0 |
| finbert | ProsusAI/finbert@0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031 | 0 | 0 | 44464 | 1075.3 | 2332.34 | 3495.63 |  | 0.399036 | 0 |
| llm | claude-haiku-5-5 | 0.000147438 | 0.000294247 | 200 | 1303.27 | 1729.97 | 2283.75 | 659.222 | 0.36968 | 0.000795951 |
| llm_h45 | claude-haiku-4-5 | 0.00158941 | 0.00317883 | 0 |  |  |  | 354.301 | 0.364315 | 0.00872549 |
| llm_llama | llama-3.1-8b-instruct-q4_k_m |  | 0 | 2426 | 32589.1 | 46691.2 | 57595.2 |  | 0.229494 | 0 |
