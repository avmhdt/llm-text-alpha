<!-- command=lta eval contamination --models claude-haiku-5-5,claude-haiku-4-5,llama-3.1-8b-instruct-q4_k_m | source=lta repro | provider=licensed -->

| scorer | baseline | n_pre | n_post | did | did_lo95 | did_hi95 | contamination_found | did_ic | did_ic_lo95 | did_ic_hi95 | contamination_found_ic | model_id |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llm | lm | 19249 | 937 | -0.00726211 | -0.0302543 | 0.0145146 | False | -0.0218132 | -0.116201 | 0.062451 | False | claude-haiku-5-5 |
| llm | finbert | 19249 | 937 | -0.0104157 | -0.0284469 | 0.00767384 | False | -0.0482677 | -0.124322 | 0.0194102 | False | claude-haiku-5-5 |
| llm_h45 | lm | 1437 | 1431 | 0.00226875 | -0.0227119 | 0.0272161 | False | 0.0451341 | -0.0464248 | 0.146258 | False | claude-haiku-4-5 |
| llm_h45 | finbert | 1437 | 1431 | -0.0145818 | -0.0376602 | 0.00758825 | False | -0.0393299 | -0.11583 | 0.041774 | False | claude-haiku-4-5 |
| llm_llama | lm | 541 | 579 | nan | nan | nan | False | 0.13944 | -0.0290673 | 0.280575 | False | llama-3.1-8b-instruct-q4_k_m |
| llm_llama | finbert | 541 | 579 | nan | nan | nan | False | -0.00846723 | -0.13752 | 0.110529 | False | llama-3.1-8b-instruct-q4_k_m |
