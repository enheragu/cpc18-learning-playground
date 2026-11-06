#!/usr/bin/env bash
# Drive the CPC18 sweep on Kaggle's AMD cell.
#
# You cannot ask Kaggle for an AMD host: ~7% of CPU-only sessions land on an
# AMD EPYC 7B12, the rest on Intel Broadwell-EP. So this plays the lottery --
# each attempt pushes the kernel, which checks the vendor and exits within
# seconds if it drew Intel. Only an AMD draw installs anything or trains.
#
# Progress survives across sessions: the trainers skip seeds already present
# (_seedsAlreadyDone), so each winning session resumes from the canonical tree
# and the 12 h session wall costs at most the trials buffered since the last
# flush (FLUSH_EVERY=25 per cell).
#
# Usage:
#   run_amd_sweep.sh setup  --yes
#   run_amd_sweep.sh hunt   --attempts 30 --yes
#   run_amd_sweep.sh status
#
# Uses the standard Kaggle API credentials: ~/.kaggle/kaggle.json, or the
# KAGGLE_USERNAME and KAGGLE_KEY variables.
# merge_sweep.py merges each returned session into the canonical tree; a seed
# returned twice is kept once, after checking that both copies agree bit for bit.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
SWEEP="$HERE/../amd_sweep"
FETCHED="$SWEEP/fetched"
KAGGLE_BIN="${KAGGLE_BIN:-kaggle}"
# Python of the repository's virtual environment: venv/ (as in the README) or .venv/.
if [[ -z "${PY:-}" ]]; then
  PY="$REPO/venv/bin/python"
  [[ -x "$PY" ]] || PY="$REPO/.venv/bin/python"
fi

TAG="kaggle-amd"
DATASET_SLUG="eeha-cpc18-code"
KERNEL_SLUG="eeha-cpc18-amd-sweep"
BUDGET=""            # hours; empty = leave the kernel's own default

ACTION="${1:-}"; shift || true
ATTEMPTS=1; CONFIRM=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --attempts) ATTEMPTS="$2"; shift 2;;
    # A short --budget with a throwaway --tag is how you rehearse the whole
    # path (stack, probe, training, fetch) on a real AMD draw without
    # committing 11 h to it.
    --budget)   BUDGET="$2"; shift 2;;
    --tag)      TAG="$2"; shift 2;;
    --yes)      CONFIRM=1; shift;;
    *) echo "Unknown option: $1" >&2; exit 2;;
  esac
done

# Results for a non-default tag live in their own tree, so a rehearsal can
# never be merged into the real sweep by accident.
if [[ "$TAG" == "kaggle-amd" ]]; then
  CANON="$SWEEP/canonical"
  KERNEL_SLUG="eeha-cpc18-amd-sweep"
else
  CANON="$SWEEP/canonical-$TAG"
  KERNEL_SLUG="eeha-cpc18-$TAG"
fi

say() { printf '\n=== %s ===\n' "$*"; }

setup_kaggle() {
  USER="${KAGGLE_USERNAME:-$("$KAGGLE_BIN" config view 2>/dev/null | awk '/username/{print $3}' || true)}"
  [[ -n "$USER" ]] || { echo "ERROR: no Kaggle API credentials found" >&2; exit 2; }
}

push_dataset() {
  say "Packaging dataset (resume = current canonical tree)"
  local args=(--slug "$DATASET_SLUG")
  [[ -d "$CANON" ]] && args+=(--resume "$CANON")
  "$HERE/make_dataset_cpc18.sh" "${args[@]}"
}

push_kernel() {
  local work; work=$(mktemp -d); trap "rm -rf '$work'" RETURN
  cp "$HERE/amd_sweep_kernel.py" "$work/"
  # Stamped here, never by hand: a kernel pointing at the wrong tag or dataset
  # would write results that silently do not join the rest of the sweep.
  sed -i "s|^TAG = .*|TAG = \"$TAG\"|; s|^DATASET_SLUG = .*|DATASET_SLUG = \"$DATASET_SLUG\"|" \
      "$work/amd_sweep_kernel.py"
  [[ -n "$BUDGET" ]] && sed -i "s|^BUDGET_HOURS = .*|BUDGET_HOURS = $BUDGET|" "$work/amd_sweep_kernel.py"
  cat > "$work/kernel-metadata.json" <<META
{
  "id": "$USER/$KERNEL_SLUG",
  "title": "$KERNEL_SLUG",
  "code_file": "amd_sweep_kernel.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_tpu": false,
  "enable_internet": true,
  "dataset_sources": ["$USER/$DATASET_SLUG"],
  "competition_sources": [],
  "model_sources": [],
  "kernel_sources": []
}
META
  "$KAGGLE_BIN" kernels push -p "$work" >/dev/null
}

wait_kernel() {
  while true; do
    S=$("$KAGGLE_BIN" kernels status "$USER/$KERNEL_SLUG" 2>&1 || true)
    # Lowercased: the CLI reports KernelWorkerStatus.COMPLETE in caps and a
    # case-sensitive match would poll a finished kernel forever.
    case "${S,,}" in
      *complete*) echo "complete"; return 0;;
      *error*)    echo "error";    return 0;;
      *cancel*)   echo "cancel";   return 0;;
    esac
    sleep 20
  done
}

case "$ACTION" in
  setup)
    setup_kaggle
    (( CONFIRM )) || { echo "Pass --yes to upload the dataset." >&2; exit 1; }
    push_dataset
    ;;

  hunt)
    setup_kaggle
    say "AMD lottery: up to $ATTEMPTS attempt(s) as $USER"
    echo "  Intel draws exit in seconds and cost almost nothing."
    echo "  An AMD draw trains for up to 11 h and banks its progress."
    (( CONFIRM )) || { echo; echo "  Aborted: pass --yes." >&2; exit 1; }
    mkdir -p "$CANON" "$FETCHED"

    for i in $(seq 1 "$ATTEMPTS"); do
      printf '\n--- attempt %d/%d (%s) ---\n' "$i" "$ATTEMPTS" "$(date +%H:%M:%S)"
      push_kernel
      st=$(wait_kernel)
      if [[ "$st" != "complete" ]]; then
        echo "  kernel ended: $st -- skipping this attempt"
        continue
      fi
      dest="$FETCHED/$(date +%Y%m%d-%H%M%S)"
      mkdir -p "$dest"
      "$KAGGLE_BIN" kernels output "$USER/$KERNEL_SLUG" -p "$dest" >/dev/null 2>&1 || true

      sum="$dest/sweep_summary.json"
      if [[ ! -f "$sum" ]]; then
        echo "  no summary returned -- treating as a failed attempt"; continue
      fi
      outcome=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['outcome'])" "$sum")
      cpu=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['cpu_model'])" "$sum")
      echo "  drew: $cpu -> $outcome"

      case "$outcome" in
        skipped_wrong_vendor) rm -rf "$dest"; continue;;
        abort_*) echo "  *** $outcome -- stopping, see $sum" >&2; exit 1;;
      esac

      say "AMD session returned -- merging"
      "$PY" -c "import json,sys;d=json.load(open(sys.argv[1]));print('  elapsed',d.get('elapsed_hours'),'h');[print(f'  {k:32s} {v}/800') for k,v in (d.get('trials_by_cell') or {}).items()]" "$sum"
      if [[ -d "$dest/output_data_$TAG" ]]; then
        "$PY" "$HERE/merge_sweep.py" "$dest/output_data_$TAG" "$CANON"
      else
        echo "  WARNING: no output_data_$TAG in the fetch" >&2
      fi
      # Only the real sweep feeds its progress back into the dataset. A
      # rehearsal must not seed the shared dataset's resume/ tree.
      if [[ "$TAG" == "kaggle-amd" ]]; then
        push_dataset    # next attempt resumes from the updated canonical tree
      fi
    done
    ;;

  status)
    [[ -d "$CANON" ]] || { echo "No sweep data yet ($CANON)"; exit 0; }
    say "Canonical AMD sweep progress"
    "$PY" - "$CANON" <<'PYEOF'
import os, re, sys
canon = sys.argv[1]
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d{3}:")
total = 0
n_cells = sum(1 for c in os.listdir(canon) if os.path.isdir(os.path.join(canon, c)))
for cell in sorted(os.listdir(canon)):
    d = os.path.join(canon, cell)
    if not os.path.isdir(d):
        continue
    for y in os.listdir(d):
        if not y.endswith(".yaml"):
            continue
        n = sum(1 for line in open(os.path.join(d, y)) if DATE.match(line))
        total += n
        bar = "#" * (n * 30 // 800)
        print(f"  {cell:34s} {n:4d}/800  {bar}")
print(f"\n  total trials banked: {total} / {n_cells * 800}  ({n_cells} cells)")
PYEOF
    ;;

  *)
    echo "Usage: $0 {setup|hunt|status} [--attempts N] [--yes]" >&2
    exit 2;;
esac
