#!/usr/bin/env bash
# Start the streaming prep worker(s) under a supervisor, plus the Hub sync (resumable).
# A worker exits with code 3 after a GPU error (its HIP context may be unusable);
# the supervisor restarts it. Any other exit (all done, low disk) stops that worker.
set -euo pipefail
cd "$(dirname "$0")"
source "$HOME/TTS/.venv/bin/activate"
WORKERS="${WORKERS:-1}"  # 2 workers do not fit in the ~25 GB this shared GPU leaves free
mkdir -p data/logs

supervise() {
  local i=$1
  while true; do
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python prep_stream.py --worker "$i" --num-workers "$WORKERS" \
      >> "data/logs/prep_w$i.log" 2>&1 && code=0 || code=$?
    echo "$(date '+%F %T') worker $i exited with code $code" >> "data/logs/prep_w$i.log"
    [ "$code" -eq 3 ] || break
    sleep 30
  done
}

for ((i = 0; i < WORKERS; i++)); do
  export -f supervise; export WORKERS
  setsid nohup bash -c "supervise $i" > /dev/null 2>&1 &
done
pgrep -f "^python sync_hub.py" > /dev/null || setsid nohup python sync_hub.py --loop 900 >> data/logs/sync_hub.log 2>&1 &
echo "started $WORKERS supervised prep worker(s) + hub sync; logs in data/logs/"
