<!-- command=lta eval regress --scorers lm,finbert,llm --se both --window all | source=lta repro | provider=licensed -->

| scorer | window | with | n | spearman |
|---|---|---|---|---|
| lm | all | sue | 41159 | 0.0822074 |
| lm | all | sue_announced | 38496 | 0.107657 |
| finbert | all | sue | 41159 | 0.142629 |
| finbert | all | sue_announced | 38496 | 0.222304 |
| llm | all | sue | 18955 | 0.283308 |
| llm | all | sue_announced | 17752 | 0.507519 |
