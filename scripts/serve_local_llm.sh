#!/usr/bin/env bash
# Serve the D-05 local model (Llama 3.1 8B Instruct, GGUF Q4_K_M) with llama.cpp's
# llama-server on CPU, for `lta score llm --provider local` (P-12).
#
# Run inside WSL (Ubuntu), from the repository root:
#
#   bash scripts/serve_local_llm.sh install    # llama.cpp CPU build b11796, about 22 MB
#   bash scripts/serve_local_llm.sh download   # the model, about 4.9 GB at <= 10 MB/s
#   bash scripts/serve_local_llm.sh start      # llama-server in the background on 127.0.0.1:8080
#   bash scripts/serve_local_llm.sh check      # one test request; prints the reply
#   bash scripts/serve_local_llm.sh status     # running or not, and memory free
#   bash scripts/serve_local_llm.sh stop
#
# `serve` runs the server in the foreground instead of `start`.
#
# Everything lands under $DATA_ROOT (default ~/data/llm-text-alpha) on the WSL disk, never in
# git. The weights are under the Llama 3.1 Community License and are never committed.
# HF_TOKEN is read from .env if the download needs it; it is sent only as a header and is
# never printed.
set -euo pipefail

LLAMA_TAG="b11796"                       # released 2026-10-21, commit 0a1ef309
LLAMA_ASSET="llama-${LLAMA_TAG}-bin-ubuntu-x64.tar.gz"
LLAMA_SHA256="70b856d40577f1cf66120f894b2674398175efbae7d9153566872c4b3c1b3d32"
MODEL_REPO="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
MODEL_REVISION="bf5b95e96dac0462e2a09145ec66cae9a3f12067"
MODEL_FILE="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
MODEL_SHA256="7b064f5842bf9532c91456deda288a1b672397a54fa729aa665952863033557c"
MODEL_ALIAS="llama-3.1-8b-instruct-q4_k_m"   # must equal LOCAL_LLM_MODEL and configs/models.yaml

DATA_ROOT="${DATA_ROOT:-$HOME/data/llm-text-alpha}"
ROOT="$DATA_ROOT/local_llm"
BIN_DIR="$ROOT/llama.cpp-$LLAMA_TAG"
MODEL_PATH="$ROOT/models/$MODEL_FILE"
PID_FILE="$ROOT/llama-server.pid"
LOG_FILE="$ROOT/llama-server.log"
HOST="${LOCAL_LLM_HOST:-127.0.0.1}"
PORT="${LOCAL_LLM_PORT:-8080}"
THREADS="${LOCAL_LLM_THREADS:-8}"        # house rule: at most 8 threads
CTX="${LOCAL_LLM_CTX:-8192}"             # 3,000-token documents plus the prompt and answer
MIN_FREE_GB=30                           # house rule: do not start below 30 GB free memory
RATE="10M"                               # house rule: downloads capped at 10 MB/s

die() { echo "error: $*" >&2; exit 1; }

free_gb() { awk '/MemAvailable/ {printf "%d", $2 / 1048576}' /proc/meminfo; }

server_bin() {
  local found
  found="$(find "$BIN_DIR" -type f -name llama-server -perm -u+x 2>/dev/null | head -1)"
  [[ -n "$found" ]] || die "llama-server not found; run: bash scripts/serve_local_llm.sh install"
  echo "$found"
}

hf_header() {
  # Prints a curl header argument when HF_TOKEN is set in the environment or in .env.
  local token="${HF_TOKEN:-}"
  if [[ -z "$token" && -f .env ]]; then
    token="$(grep -E '^HF_TOKEN=' .env | tail -1 | cut -d= -f2- | tr -d '"' | tr -d "'")"
  fi
  [[ -n "$token" ]] && printf 'Authorization: Bearer %s' "$token"
  return 0
}

install() {
  [[ "$THREADS" -le 8 ]] || die "LOCAL_LLM_THREADS must be 8 or fewer"
  mkdir -p "$BIN_DIR"
  local tarball="$ROOT/$LLAMA_ASSET"
  if [[ ! -f "$tarball" ]]; then
    curl -fL --limit-rate "$RATE" -o "$tarball.part" \
      "https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_TAG/$LLAMA_ASSET"
    mv "$tarball.part" "$tarball"
  fi
  echo "$LLAMA_SHA256  $tarball" | sha256sum -c - >/dev/null || die "llama.cpp checksum mismatch"
  tar -xzf "$tarball" -C "$BIN_DIR"
  echo "llama.cpp $LLAMA_TAG installed: $(server_bin)"
  "$(server_bin)" --version 2>&1 | head -2 || true
}

download() {
  mkdir -p "$ROOT/models"
  if [[ -f "$MODEL_PATH" ]] && echo "$MODEL_SHA256  $MODEL_PATH" | sha256sum -c - >/dev/null 2>&1; then
    echo "model already present and verified: $MODEL_PATH"
    return
  fi
  local header
  header="$(hf_header)"
  local url="https://huggingface.co/$MODEL_REPO/resolve/$MODEL_REVISION/$MODEL_FILE"
  echo "downloading $MODEL_FILE (about 4.9 GB) at up to 10 MB/s; resumable"
  if [[ -n "$header" ]]; then
    curl -fL --limit-rate "$RATE" -C - -H "$header" -o "$MODEL_PATH.part" "$url"
  else
    curl -fL --limit-rate "$RATE" -C - -o "$MODEL_PATH.part" "$url"
  fi
  echo "$MODEL_SHA256  $MODEL_PATH.part" | sha256sum -c - >/dev/null || die "model checksum mismatch"
  mv "$MODEL_PATH.part" "$MODEL_PATH"
  echo "model verified: $MODEL_PATH"
}

args() {
  [[ -f "$MODEL_PATH" ]] || die "model missing; run: bash scripts/serve_local_llm.sh download"
  [[ "$THREADS" -le 8 ]] || die "LOCAL_LLM_THREADS must be 8 or fewer"
  local free
  free="$(free_gb)"
  [[ "$free" -ge "$MIN_FREE_GB" ]] || die "only ${free} GB memory free; need ${MIN_FREE_GB} GB"
  echo --model "$MODEL_PATH" --alias "$MODEL_ALIAS" --host "$HOST" --port "$PORT" \
    --threads "$THREADS" --threads-batch "$THREADS" --ctx-size "$CTX" --parallel 1 \
    --jinja
}

serve() {
  # shellcheck disable=SC2046
  exec nice -n 10 "$(server_bin)" $(args)
}

start() {
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "already running, pid $(cat "$PID_FILE")"
    return
  fi
  local bin cmd
  bin="$(server_bin)"
  cmd="$(args)"
  # shellcheck disable=SC2086
  nohup nice -n 10 "$bin" $cmd >"$LOG_FILE" 2>&1 &
  echo $! >"$PID_FILE"
  echo "started pid $(cat "$PID_FILE"); log $LOG_FILE"
  for _ in $(seq 1 120); do
    if curl -fs "http://$HOST:$PORT/health" >/dev/null 2>&1; then
      echo "ready at http://$HOST:$PORT/v1 as $MODEL_ALIAS"
      return
    fi
    sleep 1
  done
  die "server not healthy after 120 s; see $LOG_FILE"
}

check() {
  curl -fs "http://$HOST:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d '{
    "model": "'"$MODEL_ALIAS"'", "temperature": 0, "max_tokens": 40,
    "messages": [{"role": "user", "content": "Reply with the JSON object {\"ok\": true} only."}],
    "response_format": {"type": "json_schema", "json_schema": {"name": "ok", "strict": true,
      "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}},
                 "required": ["ok"], "additionalProperties": false}}}
  }' | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"]); print("usage", d.get("usage"))'
}

status() {
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "running, pid $(cat "$PID_FILE"), http://$HOST:$PORT/v1"
  else
    echo "not running"
  fi
  echo "memory free: $(free_gb) GB"
}

stop() {
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    kill "$(cat "$PID_FILE")" && rm -f "$PID_FILE" && echo "stopped"
  else
    echo "not running"
  fi
}

case "${1:-}" in
  install) install ;;
  download) download ;;
  serve) serve ;;
  start) start ;;
  check) check ;;
  status) status ;;
  stop) stop ;;
  *) echo "usage: bash scripts/serve_local_llm.sh {install|download|start|serve|check|status|stop}" >&2; exit 2 ;;
esac
