#!/bin/bash
# TEMPORARY (red-team iteration): run every red-team check, variant and shortcut.
# Restored to the gym's version at the end of the iteration.
set -uo pipefail
cd "$(dirname "$0")/.."
bash redteam/run_all.sh 2>&1 | tee redteam/run_all.log
