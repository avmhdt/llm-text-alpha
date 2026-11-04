<!-- command=python benchmarks/bench_scorers.py --n 500 --threads 8 | run_id=bdd7e0f6-f6f2-4d1b-a2d8-086ee9ef2d21 | git_sha=84a57712b273102b7ec1c64c58d3760882232283-dirty | provider=reference hardware in docs/hardware.md; live recorders share the machine -->

| scorer | docs | docs_per_s | p50_ms | p95_ms | p99_ms | hours_per_100k_docs | settings |
|---|---|---|---|---|---|---|---|
| lm | 500 | 581.035 | 1.32601 | 3.97983 | 6.56528 | 0.0478074 | lm-LM_MasterDictionary@e2d13286 |
| finbert | 500 | 0.966209 | 1048.8 | 1162.13 | 1259.47 | 28.7493 | ProsusAI/finbert@0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031, CPU, 8 threads, max_chunks 7 |
