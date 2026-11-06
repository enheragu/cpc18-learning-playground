#!/usr/bin/env bash
# Download the canonical BEAST-GB reference (R) from OSF project vw2su
# (Plonsky et al. 2024 replication package). This provides the canonical
# feature-engineering script and the BEAST-GB hyperparameters used as the
# reference implementation in this repository.
#
# NOTE: the precomputed features (data/cpc18_features.csv) are committed,
# so most users do NOT need this. It is only needed to regenerate the
# features from scratch or to inspect the canonical R implementation.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OSF_NODE="vw2su"
TARGET="BEAST_GB demo code.zip"
DEST="$HERE/$TARGET"

echo "Querying OSF node $OSF_NODE for '$TARGET'..."
URL=$(curl -gs "https://api.osf.io/v2/nodes/${OSF_NODE}/files/osfstorage/?page[size]=100" \
  | python3 -c "
import sys, json
data = json.load(sys.stdin).get('data', [])
for f in data:
    if f['attributes']['name'] == '''$TARGET''':
        print(f['links']['download']); break
" || true)

if [ -z "${URL:-}" ]; then
  echo "Could not locate '$TARGET' automatically. Download it manually from:"
  echo "    https://osf.io/${OSF_NODE}/"
  echo "and place it at: $DEST"
  exit 1
fi

echo "Downloading -> $DEST"
curl -L -o "$DEST" "$URL"
echo "Unzipping into $HERE/BEAST_GB_demo/"
unzip -o "$DEST" -d "$HERE/BEAST_GB_demo/"
echo "Done. Canonical BEAST_GB.R is under $HERE/BEAST_GB_demo/."
