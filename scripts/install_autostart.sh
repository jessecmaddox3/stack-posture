#!/bin/bash
# Install Stack as a LaunchAgent using bootstrap (launchctl load is deprecated).
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="com.stack-posture.stack"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
DATA_DIR="$HOME/.posture_monitor"

# launchd does not run a login shell, so its PATH is the bare system default and
# does NOT include Homebrew. Resolve uv now, while we still have a real PATH, and
# bake the answer into the plist. Without this, run.sh's `command -v uv` check
# fails at every launch, KeepAlive relaunches it, and the result is a silent
# crash loop once a minute with no menu bar icon and no message.
UV_BIN="$(command -v uv || true)"
if [ -z "$UV_BIN" ]; then
    echo "uv is not on PATH. Install it first: brew install uv" >&2
    exit 1
fi
AGENT_PATH="$(dirname "$UV_BIN"):/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

# Camera access cannot be granted to a launchd process: macOS attributes the TCC
# prompt to the responsible process, and a background agent generally cannot
# raise one. If we install before the app has ever captured interactively, it
# will start at every login and silently never see a camera. Require evidence
# that a real capture has already happened.
if [ ! -f "$DATA_DIR/history.db" ]; then
    echo "No history database yet, so the app has never run successfully." >&2
    echo "Run ./run.sh manually first, grant camera access when macOS asks," >&2
    echo "and confirm the menu bar icon appears. Then install autostart." >&2
    exit 1
fi
CAPTURED=$("$UV_BIN" run python -c "
import sqlite3, sys
from pathlib import Path
conn = sqlite3.connect(Path.home() / '.posture_monitor' / 'history.db')
try:
    print(conn.execute(\"SELECT COUNT(*) FROM checks WHERE outcome = 'OK'\").fetchone()[0])
except sqlite3.Error:
    print(0)
" 2>/dev/null || echo 0)
if [ "$CAPTURED" -lt 1 ]; then
    echo "The database has no successful checks, so the camera has never worked." >&2
    echo "Run ./run.sh, use Check Now, and confirm a reading appears." >&2
    echo "Then install autostart. Diagnostics: uv run python scripts/diagnostics.py" >&2
    exit 1
fi

mkdir -p "$HOME/Library/LaunchAgents" "$DATA_DIR"

cat > "$PLIST" <<PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>${LABEL}</string>
    <key>ProgramArguments</key>
    <array><string>${PROJECT_DIR}/run.sh</string></array>
    <key>WorkingDirectory</key><string>${PROJECT_DIR}</string>
    <key>EnvironmentVariables</key>
    <dict><key>PATH</key><string>${AGENT_PATH}</string></dict>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key>
    <dict><key>SuccessfulExit</key><false/></dict>
    <key>ThrottleInterval</key><integer>60</integer>
    <key>StandardOutPath</key><string>${DATA_DIR}/launch.log</string>
    <key>StandardErrorPath</key><string>${DATA_DIR}/launch_error.log</string>
</dict>
</plist>
PLIST_EOF

launchctl bootout "gui/$(id -u)/${LABEL}" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
launchctl enable "gui/$(id -u)/${LABEL}"

# bootstrap succeeding only means the plist was accepted, not that the app runs.
# Give it a moment, then say what actually happened rather than assuming.
sleep 3
if launchctl print "gui/$(id -u)/${LABEL}" 2>/dev/null | grep -q "state = running"; then
    echo "Installed and running. KeepAlive restarts it on crash, throttled to once a minute."
else
    echo "Installed, but the agent is not running." >&2
    echo "Check ${DATA_DIR}/launch_error.log for the reason." >&2
    echo "Diagnostics: uv run python scripts/diagnostics.py" >&2
fi
echo "Remove with: ./scripts/uninstall_autostart.sh"
