#!/bin/bash
# Launch Stack. Dependencies are installed once via `uv sync`, not here.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv &>/dev/null; then
    echo "uv is required: brew install uv" >&2
    exit 1
fi
if [ ! -d .venv ]; then
    echo "First run: creating the environment..."
    uv sync
fi
# Checked separately from .venv on purpose. These two can get out of step: a
# fetch that failed partway, a cleared ~/.posture_monitor, or a model deleted to
# reclaim disk all leave a working venv and no model. Tying the fetch to .venv
# meant the app started anyway and MediaPipe failed later, which is the silent
# failure this rebuild exists to remove.
MODEL="${POSTURE_MODEL_PATH:-$HOME/.posture_monitor/models/pose_landmarker_full.task}"
if [ ! -f "$MODEL" ]; then
    echo "Pose model missing, fetching it..."
    ./scripts/fetch_model.sh
fi
exec uv run python -m posture.ui.app
