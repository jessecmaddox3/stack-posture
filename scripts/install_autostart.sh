#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -x .venv/bin/python ]; then
    echo 'Run Set Up Stack.command first.' >&2
    exit 1
fi
exec .venv/bin/python -m posture.autostart "$PWD"
