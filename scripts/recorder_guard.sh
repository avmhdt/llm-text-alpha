#!/usr/bin/env bash
# Pause FinBERT and corpus builds while the crypto recorder's ingest p99 is elevated.
# Pause: any venue ingest_us_p99 > 1200 on two readings in a row. Resume: all < 1000.
LOG=/mnt/d/trading-strategy-brainstorm/llm-text-alpha/logs/recorder-guard.txt
high=0; paused=0
while true; do
  worst=$(cd ~/lob-recorder && ./venv/bin/lob record status 2>/dev/null \
    | grep -o "ingest_us_p99=[0-9.]*" | cut -d= -f2 | sort -g | tail -1)
  pids=$(pgrep -f "lta score finbert|lta corpus build" | tr '\n' ' ')
  if [ -z "$worst" ]; then
    echo "$(date -u +%FT%TZ) status unreadable" >> "$LOG"
  elif awk "BEGIN{exit !($worst > 1200)}"; then
    high=$((high+1))
    if [ "$high" -ge 2 ] && [ "$paused" = 0 ] && [ -n "$pids" ]; then
      kill -STOP $pids; paused=1
      echo "$(date -u +%FT%TZ) paused ($worst us) pids $pids" >> "$LOG"
    fi
  else
    high=0
    if [ "$paused" = 1 ] && awk "BEGIN{exit !($worst < 1000)}"; then
      [ -n "$pids" ] && kill -CONT $pids; paused=0
      echo "$(date -u +%FT%TZ) resumed ($worst us)" >> "$LOG"
    fi
  fi
  if ! pgrep -f "finbert_loop.sh" >/dev/null && [ "$paused" = 0 ]; then
    echo "$(date -u +%FT%TZ) finbert loop ended; guard exits" >> "$LOG"; exit 0
  fi
  sleep 300
done
