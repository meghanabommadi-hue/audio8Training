#!/usr/bin/env bash
# Benchmark server configs: ENGINES AR_THREADS CODEC_THREADS "CONCURRENCY LEVELS"
cd "$(dirname "$0")"
PY=../Audio8_TTS/onnx_runtime_0_1b_int8/.venv/bin/python
run() {
  [ -f server.pid ] && kill "$(cat server.pid)" 2>/dev/null; sleep 3
  ENGINES=$1 AR_THREADS=$2 CODEC_THREADS=$3 setsid nohup $PY -m uvicorn server:app --host 127.0.0.1 --port 8030 > server_sweep.log 2>&1 & echo $! > server.pid
  for i in $(seq 1 150); do curl -sf localhost:8030/health >/dev/null && break; sleep 2; done
  echo "=== engines=$1 ar_threads=$2 codec_threads=$3"
  $PY bench_server.py --concurrency $4 --requests 3 --buffer ${BUFFER:-0.3}
}
run "$@"
