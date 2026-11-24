#!/bin/bash
# Fetch the paper's own stress-stretch data and rebuild data/curves.json (the visible curves).
#
# Source: LivingMatterLab/CANN (the authors' repository, cited in the paper's "Data availability"),
# commit d3caca0f, files GICANN/input/CANNsRUBBERdata.xlsx and GICANN/input/CANNsBRAINdata.xlsx,
# i.e. the spreadsheets the paper's notebooks read. Both files are checked against sha256.
#
# data/prepare_data.py extracts only what a method may see (all training-mode curves, the held-out
# mode of the development materials, and the held-out mode's stretch values of the test materials).
# The spreadsheets themselves also contain the hidden test curves, so they are deleted afterwards.
# data/curves.json is also committed, so offline (e.g. on a compute node) this script keeps it.
#
# Download size: ~65 KB. Disk use after cleanup: ~25 KB.

set -euo pipefail
cd "$(dirname "$0")/.."

COMMIT=d3caca0fafa5eaf8cbd3cefb377598f1f3e9b481
BASE="https://raw.githubusercontent.com/LivingMatterLab/CANN/${COMMIT}/GICANN/input"
DL=data/downloads
mkdir -p "$DL"

fetch() {  # name sha256
  curl -fsSL --retry 3 --max-time 60 -o "$DL/$1" "$BASE/$1" || return 1
  echo "$2  $DL/$1" | sha256sum -c --quiet - || { echo "checksum mismatch for $1" >&2; exit 1; }
}

if fetch CANNsRUBBERdata.xlsx 7d2411b03f9f7939f67c0729289000eba3704b87be3a5ca4f88813a18ec87750 \
   && fetch CANNsBRAINdata.xlsx 701f9c7d4495bc55e78f69189719a448b1de81cdf1941e9c556995fa30d4c2bb; then
  python3 data/prepare_data.py --rubber-xlsx "$DL/CANNsRUBBERdata.xlsx" \
    --brain-xlsx "$DL/CANNsBRAINdata.xlsx" --out data/curves.json
else
  echo "download failed (offline?): keeping the committed data/curves.json"
fi
rm -f "$DL"/*.xlsx
python3 data/load_data.py
