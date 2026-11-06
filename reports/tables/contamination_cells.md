<!-- command=lta eval contamination --models claude-haiku-5-5,claude-haiku-4-5,llama-3.1-8b-instruct-q4_k_m | run_id=a430b96c-5153-4d48-9315-b6b7bba292eb | git_sha=7fefad323e988cd2b058591f8c52259ca13a5329-dirty | provider=licensed -->

| scorer | text | window | n | spread | spread_lo95 | spread_hi95 | t_spread | ic | ic_lo95 | ic_hi95 | underpowered | model_id |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llm | raw | pre | 19297 | 0.00205769 | -0.00092278 | 0.00503817 | 1.35317 | 0.0157623 | 0.00165306 | 0.0298653 | False | claude-haiku-5-5 |
| llm | raw | post | 937 | -0.000630006 | -0.0170121 | 0.0157521 | -0.0753755 | 0.0116844 | -0.0524002 | 0.0756731 | False | claude-haiku-5-5 |
| llm | masked | pre | 19297 | 0.00300772 | -6.55128e-05 | 0.00608094 | 1.91822 | 0.0153207 | 0.00121134 | 0.029424 | False | claude-haiku-5-5 |
| llm | masked | post | 937 | 0.00848654 | -0.00865076 | 0.0256238 | 0.970609 | 0.0264622 | -0.037647 | 0.0903544 | False | claude-haiku-5-5 |
| llm_h45 | raw | pre | 1439 | 0.000516671 | -0.0106542 | 0.0116875 | 0.0906533 | 0.00253112 | -0.0491517 | 0.0542004 | False | claude-haiku-4-5 |
| llm_h45 | raw | post | 1431 | 0.00215244 | -0.014099 | 0.0184039 | 0.259594 | 0.00601497 | -0.0458199 | 0.0578176 | False | claude-haiku-4-5 |
| llm_h45 | masked | pre | 1439 | 0.0070185 | -0.00383931 | 0.0178763 | 1.26695 | 0.0199641 | -0.031745 | 0.0715666 | False | claude-haiku-4-5 |
| llm_h45 | masked | post | 1431 | 0.000953428 | -0.0132392 | 0.015146 | 0.131668 | 0.00440373 | -0.0474277 | 0.0562115 | False | claude-haiku-4-5 |
| llm_llama | raw | pre | 542 | nan | nan | nan | nan | -0.0270629 | -0.111033 | 0.0572908 | False | llama-3.1-8b-instruct-q4_k_m |
| llm_llama | raw | post | 579 | nan | nan | nan | nan | -0.0484319 | -0.129407 | 0.0331847 | False | llama-3.1-8b-instruct-q4_k_m |
| llm_llama | masked | pre | 542 | nan | nan | nan | nan | -0.00875585 | -0.0929105 | 0.075523 | False | llama-3.1-8b-instruct-q4_k_m |
| llm_llama | masked | post | 579 | nan | nan | nan | nan | -0.0255075 | -0.106771 | 0.0560946 | False | llama-3.1-8b-instruct-q4_k_m |
