#!/usr/bin/env bash
# Portable environment; set DEVICE=cuda for Linux CUDA 12.1 wheels.
set -euo pipefail
IVIG_ENV_NAME="${IVIG_ENV_NAME:-ivig}"
cd "$(dirname "$0")"
conda create -n "$IVIG_ENV_NAME" python=3.11 -y
if [[ "${DEVICE:-cpu}" == "cuda" ]]; then
    REQUIREMENTS_FILE=requirements-cuda.txt
else
    REQUIREMENTS_FILE=requirements.txt
fi
conda run -n "$IVIG_ENV_NAME" --no-capture-output python -m pip install -r "$REQUIREMENTS_FILE"
echo "Installed. Run: conda activate $IVIG_ENV_NAME"
