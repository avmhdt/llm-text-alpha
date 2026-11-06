<!-- command=lta eval contamination --models claude-haiku-5-5,claude-haiku-4-5,llama-3.1-8b-instruct-q4_k_m | run_id=a430b96c-5153-4d48-9315-b6b7bba292eb | git_sha=7fefad323e988cd2b058591f8c52259ca13a5329-dirty | provider=licensed -->

| window | n | accuracy | lo95 | hi95 | model_id | probe |
|---|---|---|---|---|---|---|
| pre | 225 | 0.751111 | 0.690717 | 0.803074 | claude-haiku-5-5 | reid:company_correct |
| post | 75 | 0.76 | 0.65221 | 0.842453 | claude-haiku-5-5 | reid:company_correct |
| all | 300 | 0.753333 | 0.701554 | 0.798707 | claude-haiku-5-5 | reid:company_correct |
| pre | 217 | 0.152074 | 0.11038 | 0.205872 | claude-haiku-5-5 | reid:quarter_correct |
| post | 70 | 0.0285714 | 0.00787045 | 0.0983244 | claude-haiku-5-5 | reid:quarter_correct |
| all | 287 | 0.121951 | 0.0890066 | 0.164883 | claude-haiku-5-5 | reid:quarter_correct |
| pre | 223 | 0.44843 | 0.384577 | 0.51403 | claude-haiku-5-5 | recall:direction_correct |
| post | 65 | 0.446154 | 0.331693 | 0.566624 | claude-haiku-5-5 | recall:direction_correct |
| all | 288 | 0.447917 | 0.391544 | 0.50566 | claude-haiku-5-5 | recall:direction_correct |
| pre | 150 | 0.56 | 0.480047 | 0.636956 | claude-haiku-4-5 | reid:company_correct |
| post | 150 | 0.513333 | 0.434016 | 0.591984 | claude-haiku-4-5 | reid:company_correct |
| all | 300 | 0.536667 | 0.480131 | 0.592275 | claude-haiku-4-5 | reid:company_correct |
| pre | 144 | 0.194444 | 0.138095 | 0.266674 | claude-haiku-4-5 | reid:quarter_correct |
| post | 143 | 0.00699301 | 0.00123547 | 0.0385462 | claude-haiku-4-5 | reid:quarter_correct |
| all | 287 | 0.101045 | 0.0712782 | 0.141352 | claude-haiku-4-5 | reid:quarter_correct |
| pre | 148 | 0.493243 | 0.413891 | 0.572937 | claude-haiku-4-5 | recall:direction_correct |
| post | 140 | 0.464286 | 0.383731 | 0.546748 | claude-haiku-4-5 | recall:direction_correct |
| all | 288 | 0.479167 | 0.422124 | 0.536757 | claude-haiku-4-5 | recall:direction_correct |
| pre | 55 | 0.327273 | 0.218129 | 0.45897 | llama-3.1-8b-instruct-q4_k_m | reid:company_correct |
| post | 186 | 0.268817 | 0.210255 | 0.336736 | llama-3.1-8b-instruct-q4_k_m | reid:company_correct |
| all | 241 | 0.282158 | 0.229099 | 0.342053 | llama-3.1-8b-instruct-q4_k_m | reid:company_correct |
| pre | 52 | 0.0192308 | 0.00340272 | 0.101207 | llama-3.1-8b-instruct-q4_k_m | reid:quarter_correct |
| post | 177 | 0.0451977 | 0.0230774 | 0.0866408 | llama-3.1-8b-instruct-q4_k_m | reid:quarter_correct |
| all | 229 | 0.0393013 | 0.0208118 | 0.0729928 | llama-3.1-8b-instruct-q4_k_m | reid:quarter_correct |
| pre | 74 | 0.459459 | 0.350732 | 0.572189 | llama-3.1-8b-instruct-q4_k_m | recall:direction_correct |
| post | 214 | 0.448598 | 0.383452 | 0.515557 | llama-3.1-8b-instruct-q4_k_m | recall:direction_correct |
| all | 288 | 0.451389 | 0.394931 | 0.509126 | llama-3.1-8b-instruct-q4_k_m | recall:direction_correct |
