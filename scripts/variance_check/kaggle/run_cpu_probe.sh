#!/usr/bin/env bash
# Push, run and collect the CPU/SIMD-tier probe on Kaggle. Runs from the control machine.
#
# Purpose: Kaggle does not let you choose the CPU, so the only way to know which
# reproducibility cell a session lands in is to measure it. Each run samples one
# host of the fleet. Repeat to find out whether Kaggle is ONE cell or several --
# if the arithmetic path changes between sessions, the same notebook silently
# changes numerical behaviour.
#
# CPU-only sessions do not consume the ~30 h/week GPU quota, so sampling is cheap.
#
# Usage:
#   run_cpu_probe.sh status                    # credentials, no side effects
#   run_cpu_probe.sh run --yes                 # push + watch + fetch
#   run_cpu_probe.sh run --repeat 4 --yes      # 4 sequential samples
#   run_cpu_probe.sh fetch --slug eeha-cpu-probe-1
#
# Uses the standard Kaggle API credentials: ~/.kaggle/kaggle.json, or the
# KAGGLE_USERNAME and KAGGLE_KEY variables.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROBE="$HERE/../cpu_probe.py"
RESULTS="$HERE/../probe_results"
KAGGLE_BIN="${KAGGLE_BIN:-kaggle}"

ACTION="${1:-}"; shift || true
SLUG=""; REPEAT=1; CONFIRM=0
BASE_SLUG="eeha-cpu-probe"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --slug)    SLUG="$2"; shift 2;;
    --repeat)  REPEAT="$2"; shift 2;;
    --yes)     CONFIRM=1; shift;;
    *) echo "Unknown option: $1" >&2; exit 2;;
  esac
done

say() { printf '\n=== %s ===\n' "$*"; }

setup_kaggle() {
  USER="${KAGGLE_USERNAME:-$("$KAGGLE_BIN" config view 2>/dev/null | awk '/username/{print $3}' || true)}"
  [[ -n "$USER" ]] || { echo "ERROR: no Kaggle API credentials found" >&2; exit 2; }
}

do_push() {
  local slug="$1" tag="$2"
  local work; work=$(mktemp -d); trap "rm -rf '$work'" RETURN

  cp "$PROBE" "$work/cpu_probe.py"
  # The tag is stamped here, never edited by hand in the .py.
  sed -i "s|^PROBE_TAG = .*|PROBE_TAG = \"$tag\"|" "$work/cpu_probe.py"

  cat > "$work/kernel-metadata.json" <<META
{
  "id": "$USER/$slug",
  "title": "$slug",
  "code_file": "cpu_probe.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_tpu": false,
  "enable_internet": true,
  "dataset_sources": [],
  "competition_sources": [],
  "model_sources": [],
  "kernel_sources": []
}
META
  echo "  pushing $USER/$slug  (tag=$tag, CPU-only, internet on)"
  "$KAGGLE_BIN" kernels push -p "$work" | sed 's/^/    /'
}

do_watch() {
  local slug="$1"
  while true; do
    S=$("$KAGGLE_BIN" kernels status "$USER/$slug" 2>&1 || true)
    echo "    $(date +%H:%M:%S)  $S"
    # ${S,,} lowercases: the CLI reports "KernelWorkerStatus.COMPLETE" in caps,
    # and a case-sensitive match here never fires -- the loop then polls a
    # finished kernel forever instead of moving to the next sample.
    case "${S,,}" in *complete*|*error*|*cancel*) break;; esac
    sleep 20
  done
}

do_fetch() {
  local slug="$1"
  mkdir -p "$RESULTS"
  "$KAGGLE_BIN" kernels output "$USER/$slug" -p "$RESULTS" >/dev/null 2>&1 || {
    echo "    (no output to fetch yet)"; return 0; }
  for j in "$RESULTS"/cpu_probe_*.json; do
    [[ -f "$j" ]] || continue
    python3 - "$j" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
if not d.get("on_kaggle"): raise SystemExit
c, f = d["cpu"], d["fingerprint"]
print(f"    {d['tag']:28s} TIER={d['cpu_capability_tier']:8s} {c['cpu_model']}")
print(f"    {'':28s} avx512={'yes' if c['has_avx512f'] else 'no':3s} cores={c['cores_online']} "
      f"virt={c['hypervisor']}  trained={f['trained_hash']}")
PY
  done
}

case "$ACTION" in
  status)
    say "Kaggle credentials"
    setup_kaggle
    echo "  authenticated as $USER"
    echo
    echo "  CPU-only sessions do not consume the GPU quota."
    ;;

  run)
    setup_kaggle
    say "About to push and EXECUTE $REPEAT CPU probe(s) as $USER"
    echo "  Each run installs torch==$( grep -m1 '^TORCH_PIN' "$PROBE" | cut -d'"' -f2 )+cpu and fingerprints the host."
    echo "  Cost: a CPU-only session, ~2-3 min. No GPU quota consumed."
    (( CONFIRM )) || { echo; echo "  Aborted: pass --yes to confirm." >&2; exit 1; }
    for i in $(seq 1 "$REPEAT"); do
      slug="${SLUG:-$BASE_SLUG-$i}"
      # Timestamp down to seconds: samples are meant to be repeated, and two
      # runs on the same day sharing a tag would silently overwrite each other
      # on fetch -- losing exactly the fleet diversity we are sampling for.
      tag="kaggle-$(date +%Y%m%d-%H%M%S)"
      say "sample $i/$REPEAT"
      do_push "$slug" "$tag"
      do_watch "$slug"
      do_fetch "$slug"
    done
    say "Results in $RESULTS"
    ;;

  fetch)
    setup_kaggle
    [[ -n "$SLUG" ]] || { echo "ERROR: missing --slug" >&2; exit 2; }
    say "Fetching $USER/$SLUG"
    do_fetch "$SLUG"
    ;;

  *)
    echo "Usage: $0 {status|run|fetch} [--repeat N] [--slug S] [--yes]" >&2
    exit 2
    ;;
esac
