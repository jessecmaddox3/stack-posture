#!/bin/bash
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
    echo 'Run Set Up Stack.command first.'
    status=1
else
    .venv/bin/python -m posture.setup
    status=$?
fi
read -r -p 'Press Return to close. ' _
exit "$status"
