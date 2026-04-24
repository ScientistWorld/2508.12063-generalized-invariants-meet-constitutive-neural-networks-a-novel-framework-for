#!/bin/bash
# Install Python packages into the container overlay.
#
# This script is run by the system on the login node (which has internet)
# inside the container with a persistent overlay. Changes persist across
# job submissions without rebuilding the container.
#
# IMPORTANT: Install into the SYSTEM Python, not a virtual environment.

set -e

# Runtime dependencies are installed in container.def via apt to avoid the
# broken uv/pip wrapper path used by the scheduler's setup phase.
:
