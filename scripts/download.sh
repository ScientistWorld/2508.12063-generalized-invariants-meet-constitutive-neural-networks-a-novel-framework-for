#!/bin/bash
# Download all data and models needed for this environment.
#
# This paper uses experimental stress-strain data published directly in the paper
# (Table A.5). All data is embedded in the data loading code (data/load_data.py).
# No external downloads are required.
#
# Estimated total download size: 0 MB
# Estimated disk usage: ~0 MB
#
# The only external dependency is PyTorch (installed via setup.sh).

set -e
cd "$(dirname "$0")/.."

echo "No external downloads required."
echo "All data (rubber and brain tissue stress-strain measurements) is embedded in data/load_data.py"
echo "from the paper's Table A.5."
