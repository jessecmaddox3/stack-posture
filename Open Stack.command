#!/bin/bash
cd "$(dirname "$0")" || exit 1
echo 'Keep this Terminal window open while Stack runs.'
echo 'Click ▤ in the menu bar, then Check Now and Start Monitoring. Use Quit to stop.'
bash run.sh
status=$?
if [ "$status" -ne 0 ]; then read -r -p 'Press Return to close. ' _; fi
exit "$status"
