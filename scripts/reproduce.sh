#!/bin/bash
# Reproduce the paper's results end-to-end.
#
# This script runs inside the compute container and should call
# scripts/baseline.sh and scripts/method.sh to run baselines and
# the paper's method, each of which calls scripts/evaluate.sh.
#
# Assumes:
#   - Container is built (environment/container.sif)
#   - Data and models are downloaded (via scripts/download.sh)
#   - Working directory is /home/user

set -e

cd /home/user
PYTHON=${PYTHON:-python3}

# Download data (no-op since data is embedded)
bash scripts/download.sh

# Run baseline (SI-CANN)
echo ""
echo "========================================"
echo "Running SI-CANN baseline..."
echo "========================================"
bash scripts/baseline.sh

# Run the paper's method (GI-CANN)
echo ""
echo "========================================"
echo "Running GI-CANN method..."
echo "========================================"
bash scripts/method.sh

# Evaluate all results
echo ""
echo "========================================"
echo "Evaluating all results..."
echo "========================================"
bash scripts/evaluate.sh

echo ""
echo "========================================"
echo "Reproduction complete!"
echo "========================================"
echo "Results are in scoring/scores.json"
