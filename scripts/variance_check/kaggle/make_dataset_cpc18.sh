#!/usr/bin/env bash
# Package the CPC18 training code + features as a private Kaggle Dataset, so a
# kernel runs EXACTLY the bytes the lab machines ran.
#
# Why a Dataset and not a download inside the kernel: the feature CSV is an
# input variable of the experiment. Fetching it at run time gives no guarantee
# it is the same bytes, which is precisely the assumption this study exists not
# to make. The kernel re-computes the fingerprints below and aborts on mismatch.
#
# Only cpc18_features.csv ships (254 KB) -- all_CPC18_raw_data.csv (65 MB) is
# only read by the exploratory first regime, which does not run here.
#
# Usage:
#   make_dataset_cpc18.sh
#   make_dataset_cpc18.sh --resume <dir>   # seed prior results
#   make_dataset_cpc18.sh --dry-run
#
# Uses the standard Kaggle API credentials: ~/.kaggle/kaggle.json, or the
# KAGGLE_USERNAME and KAGGLE_KEY variables.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
KAGGLE_BIN="${KAGGLE_BIN:-kaggle}"

SLUG="eeha-cpc18-code"; RESUME=""; DRY=0
# Kaggle rejects `copyright-authors` when CREATING a dataset even though it
# lists it; `other` is accepted and, unlike CC0, does not waive rights on our
# own research code. Versioning an existing dataset does not re-validate.
LICENSE="${EEHA_LICENSE:-other}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --slug)    SLUG="$2"; shift 2;;
    --resume)  RESUME="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    *) echo "Unknown option: $1" >&2; exit 2;;
  esac
done

USER="${KAGGLE_USERNAME:-$("$KAGGLE_BIN" config view 2>/dev/null | awk '/username/{print $3}' || true)}"
[[ -n "$USER" ]] || { echo "ERROR: no Kaggle API credentials found" >&2; exit 2; }

say() { printf '\n=== %s ===\n' "$*"; }
STAGE=$(mktemp -d); trap 'rm -rf "$STAGE"' EXIT

say "Staging content"
mkdir -p "$STAGE/repo/data" "$STAGE/repo/mnist-learning-playground"
rsync -a --exclude '__pycache__/' "$REPO/src" "$STAGE/repo/"
rsync -a --exclude '__pycache__/' "$REPO/mnist-learning-playground/src" "$STAGE/repo/mnist-learning-playground/"
cp "$REPO/data/cpc18_features.csv" "$STAGE/repo/data/"
cp "$REPO/requirements.lock" "$STAGE/repo/"
# The probe travels too: the kernel proves which cell it landed in using the
# same fingerprint code the lab machines ran, not a reimplementation.
mkdir -p "$STAGE/repo/scripts/variance_check"
cp "$HERE/../cpu_probe.py" "$STAGE/repo/scripts/variance_check/"
mkdir -p "$STAGE/repo/scripts/variance_check/kaggle"
cp "$HERE/fingerprint_repo.py" "$STAGE/repo/scripts/variance_check/kaggle/"

if [[ -n "$RESUME" ]]; then
  # Recursive: resume data is per-cell (output_data_<tag>/<Condition>/*.yaml).
  # Seeding the wrong cell would make _seedsAlreadyDone skip trials that were
  # never run on this hardware -- a silent hole in the sweep.
  mkdir -p "$STAGE/repo/resume"
  cp -r "$RESUME"/. "$STAGE/repo/resume/"
  echo "  resume: $(find "$STAGE/repo/resume" -name '*.yaml' | wc -l) YAML(s), "\
       "$(find "$STAGE/repo/resume" -mindepth 1 -maxdepth 1 -type d | wc -l) cell dir(s)"
fi

# Fingerprints the kernel re-verifies before training anything. Computed by the
# SAME module the kernel runs (fingerprint_repo.py), so the two sides cannot
# drift apart into a false abort or, worse, a false pass.
FP=$(python3 "$HERE/fingerprint_repo.py" "$STAGE/repo")
TFP=$(awk -F= '/^training_fingerprint=/{print $2}' <<<"$FP")
DFP=$(awk -F= '/^features_fingerprint=/{print $2}' <<<"$FP")
NFILES=$(awk -F= '/^n_code_files=/{print $2}' <<<"$FP")
cat > "$STAGE/repo/FINGERPRINTS.txt" <<EOF
# Reference fingerprints, computed on the control machine at packaging time.
# The kernel recomputes and aborts on mismatch: a mismatch would mean Kaggle is
# running different code or different data from the rest of the study.
training_fingerprint=$TFP
features_fingerprint=$DFP
n_code_files=$NFILES
packaged=$(date -Is)
EOF

TITLE=$(echo "$SLUG" | tr '-' ' ')
cat > "$STAGE/dataset-metadata.json" <<EOF
{"title": "$TITLE", "id": "$USER/$SLUG", "licenses": [{"name": "$LICENSE"}]}
EOF

echo "  code files          : $NFILES"
echo "  training fingerprint: $TFP"
echo "  features fingerprint: $DFP"
echo "  size                : $(du -sh "$STAGE" | cut -f1)"

if (( DRY )); then say "--dry-run: nothing uploaded"; exit 0; fi

say "Uploading as $USER/$SLUG (private)"
if "$KAGGLE_BIN" datasets status "$USER/$SLUG" >/dev/null 2>&1; then
  echo "  exists -> new version"
  "$KAGGLE_BIN" datasets version -p "$STAGE" -m "sync $(date -Is)" --dir-mode zip
else
  echo "  does not exist -> creating"
  "$KAGGLE_BIN" datasets create -p "$STAGE" --dir-mode zip
fi
echo
echo "  Kaggle takes a while to process. Status:"
echo "  $KAGGLE_BIN datasets status $USER/$SLUG"
