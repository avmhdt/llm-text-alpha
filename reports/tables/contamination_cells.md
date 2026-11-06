<!-- command=lta eval contamination --models claude-haiku-5-5,claude-haiku-4-5,llama-3.1-8b-instruct-q4_k_m | source=lta repro | provider=licensed -->

| scorer | text | window | n | spread | spread_lo95 | spread_hi95 | t_spread | ic | ic_lo95 | ic_hi95 | underpowered | model_id |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llm | raw | pre | 19303 | 0.00208285 | -0.00089693 | 0.00506263 | 1.37003 | 0.0157725 | 0.00166542 | 0.0298733 | False | claude-haiku-5-5 |
| llm | raw | post | 937 | -0.000630006 | -0.0170121 | 0.0157521 | -0.0753755 | 0.0116844 | -0.0524002 | 0.0756731 | False | claude-haiku-5-5 |
| llm | masked | pre | 19303 | 0.00303301 | -3.94351e-05 | 0.00610546 | 1.93484 | 0.0153704 | 0.00126324 | 0.0294715 | False | claude-haiku-5-5 |
| llm | masked | post | 937 | 0.00848654 | -0.00865076 | 0.0256238 | 0.970609 | 0.0264622 | -0.037647 | 0.0903544 | False | claude-haiku-5-5 |
| llm_h45 | raw | pre | 1441 | 0.00100465 | -0.0101528 | 0.0121621 | 0.176484 | 0.00440907 | -0.0472422 | 0.0560368 | False | claude-haiku-4-5 |
| llm_h45 | raw | post | 1431 | 0.00215244 | -0.014099 | 0.0184039 | 0.259594 | 0.00601497 | -0.0458199 | 0.0578176 | False | claude-haiku-4-5 |
| llm_h45 | masked | pre | 1441 | 0.00749496 | -0.00334984 | 0.0183398 | 1.35458 | 0.0219595 | -0.0297146 | 0.0735166 | False | claude-haiku-4-5 |
| llm_h45 | masked | post | 1431 | 0.000953428 | -0.0132392 | 0.015146 | 0.131668 | 0.00440373 | -0.0474277 | 0.0562115 | False | claude-haiku-4-5 |
| llm_llama | raw | pre | 543 | nan | nan | nan | nan | -0.0254908 | -0.109402 | 0.0587808 | False | llama-3.1-8b-instruct-q4_k_m |
| llm_llama | raw | post | 579 | nan | nan | nan | nan | -0.0484319 | -0.129407 | 0.0331847 | False | llama-3.1-8b-instruct-q4_k_m |
| llm_llama | masked | pre | 543 | nan | nan | nan | nan | -0.00700555 | -0.0910974 | 0.0771855 | False | llama-3.1-8b-instruct-q4_k_m |
| llm_llama | masked | post | 579 | nan | nan | nan | nan | -0.0255075 | -0.106771 | 0.0560946 | False | llama-3.1-8b-instruct-q4_k_m |
