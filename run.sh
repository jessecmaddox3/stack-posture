#!/bin/bash
# Returning-user launch: no installation or downloads in a background process.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    echo 'Double-click Set Up Stack.command first.' >&2
    exit 1
fi
exec .venv/bin/python -m posture.ui.app
