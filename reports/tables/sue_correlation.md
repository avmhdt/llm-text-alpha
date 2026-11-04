<!-- command=lta eval regress --scorers lm,finbert,llm --se both --window all | run_id=3e325cbe-a399-41ab-8898-ae02c2291edd | git_sha=68f21d2eb28ab73b547aabe6f34d3370b6c0ff0b-dirty | provider=licensed -->

| scorer | window | with | n | spearman |
|---|---|---|---|---|
| lm | all | sue | 41159 | 0.0822074 |
| lm | all | sue_announced | 38496 | 0.107657 |
| finbert | all | sue | 41159 | 0.142629 |
| finbert | all | sue_announced | 38496 | 0.222304 |
| llm | all | sue | 18955 | 0.283308 |
| llm | all | sue_announced | 17752 | 0.507519 |
