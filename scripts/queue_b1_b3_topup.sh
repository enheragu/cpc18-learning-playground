#!/bin/bash
# Wait for in-flight Phase C chain + BourginMLP top-up to finish, then run
# B1+B3 ablation top-up to N=800/cell. Designed to run detached via nohup
# so it survives terminal / Claude-session disconnects.
#
# Usage:
#   nohup bash scripts/queue_b1_b3_topup.sh <PID1> [<PID2> ...] &
#   disown
#
# Tail the log to monitor:
#   tail -f logs/queue_b1_b3_topup_*.log

set -u

ROOT=/home/quique/umh/cpc18-learning-playground
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/queue_b1_b3_topup_$(date +%Y%m%d_%H%M%S).log"

# Redirect everything to the log (so nohup's nohup.out stays clean).
exec >>"$LOG" 2>&1
echo "[queue] start $(date -Is)"
echo "[queue] PID self=$$  args=$*"

if [ "$#" -lt 1 ]; then
    echo "[queue] ERROR: pass at least one PID to wait for"
    exit 2
fi

for pid in "$@"; do
    if kill -0 "$pid" 2>/dev/null; then
        echo "[queue] waiting for PID $pid..."
        # tail --pid blocks (no polling) until the PID disappears.
        tail --pid="$pid" -f /dev/null
        echo "[queue] PID $pid finished at $(date -Is)"
    else
        echo "[queue] PID $pid already gone, skipping"
    fi
done

echo "[queue] all parents finished. Starting B1 at $(date -Is)"
cd "$ROOT/src"

/home/quique/umh/venv/bin/python 04_train_ablation.py --study b1 --workers 6
B1_RC=$?
echo "[queue] B1 finished rc=$B1_RC at $(date -Is)"

# Run B3 even if B1 returned non-zero — they're independent studies and
# you'd rather have one finished than neither.
/home/quique/umh/venv/bin/python 04_train_ablation.py --study b3 --workers 6
B3_RC=$?
echo "[queue] B3 finished rc=$B3_RC at $(date -Is)"

echo "[queue] done $(date -Is). B1_RC=$B1_RC B3_RC=$B3_RC"
