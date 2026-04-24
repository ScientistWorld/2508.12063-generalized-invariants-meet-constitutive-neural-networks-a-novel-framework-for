#!/bin/bash
# Install Python packages into the container overlay.
#
# This script is run by the system on the login node (which has internet)
# inside the container with a persistent overlay. Changes persist across
# job submissions without rebuilding the container.
#
# IMPORTANT: Install into the SYSTEM Python, not a virtual environment.

set -e

# Minimal runtime dependencies for this reproduction.
python3 -m pip install --disable-pip-version-check --no-input --break-system-packages \
  numpy \
  torch
