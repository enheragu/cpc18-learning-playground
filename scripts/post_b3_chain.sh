#!/bin/bash
# Wait for the running B3 trainer to finish, then top up the few cells
# still at 799 (idempotency picks them up), then run the three analysers
# plus the figure-regen script. Everything runs in series. Designed to
# be launched with nohup + disown so it survives the terminal session.
#
# Usage:
#   nohup bash scripts/post_b3_chain.sh <B3_PID> </dev/null > logs/post_b3_<ts>.log 2>&1 &
#   disown

set -u

ROOT=/home/quique/umh/cpc18-learning-playground
PY=/home/quique/umh/venv/bin/python

if [ "$#" -lt 1 ]; then
    echo "[chain] ERROR: pass the B3 trainer PID as the first argument."
    exit 2
fi
B3_PID="$1"

echo "[chain] start $(date -Is)"
echo "[chain] waiting for B3 PID $B3_PID ..."

if kill -0 "$B3_PID" 2>/dev/null; then
    tail --pid="$B3_PID" -f /dev/null
fi
echo "[chain] B3 finished at $(date -Is)"

cd "$ROOT/src"

echo "[chain] --- TOP-UP B1 missing trials ---"
"$PY" 04_train_ablation.py --study b1 --workers 6
echo "[chain] --- TOP-UP B3 missing trials ---"
"$PY" 04_train_ablation.py --study b3 --workers 6

echo "[chain] --- 06 Phase C decomposition ---"
"$PY" 06_analyze_decomp.py
echo "[chain] --- 07 ablation ANOVA ---"
"$PY" 07_analyze_ablation.py
echo "[chain] --- 08 power analysis ---"
"$PY" 08_power_analysis.py

echo "[chain] --- 02 main analyse (regime 2 + regime 1) ---"
"$PY" 02_analyze.py --mode shared_pool --regime 2
"$PY" 02_analyze.py --mode shared_pool --regime 1

echo "[chain] --- 05 regenerate figures with canonical data ---"
"$PY" 05_make_paper_figures.py

echo "[chain] all done at $(date -Is)"
