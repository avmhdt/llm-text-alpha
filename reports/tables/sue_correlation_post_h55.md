<!-- command=lta eval regress --scorers lm,finbert,llm --se both --window post_h55 | run_id=0398465b-b81d-4324-9536-74af26d46fa3 | git_sha=68f21d2eb28ab73b547aabe6f34d3370b6c0ff0b-dirty | provider=licensed -->

| scorer | window | with | n | spearman |
|---|---|---|---|---|
| lm | post_h55 | sue | 969 | 0.0529307 |
| lm | post_h55 | sue_announced | 916 | 0.0585438 |
| finbert | post_h55 | sue | 969 | 0.0712181 |
| finbert | post_h55 | sue_announced | 916 | 0.112163 |
| llm | post_h55 | sue | 969 | 0.325832 |
| llm | post_h55 | sue_announced | 916 | 0.488367 |
