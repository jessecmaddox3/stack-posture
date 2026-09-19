#!/bin/bash
cd "$(dirname "$0")" || exit 1
bash scripts/bootstrap.sh
status=$?
echo
read -r -p 'Press Return to close this setup window. ' _
exit "$status"
