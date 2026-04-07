#!/bin/bash
# Install Python packages into the container overlay.
#
# This script is run by the system on the login node (which has internet)
# inside the container with a persistent overlay. Changes persist across
# job submissions without rebuilding the container.
#
# IMPORTANT: Install into the SYSTEM Python, not a virtual environment.
# Use: uv pip install --system <packages>

set -e

# Install PyTorch CPU-only (fast install, small download)
# Using pip since uv may not be available in all container builds
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
