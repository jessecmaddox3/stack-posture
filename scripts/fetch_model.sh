#!/bin/bash
# Download the MediaPipe pose model. Not committed: 9.4MB binary.
set -euo pipefail
DEST="${1:-$HOME/.posture_monitor/models/pose_landmarker_full.task}"
URL="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
EXPECTED_BYTES=9398198

mkdir -p "$(dirname "$DEST")"
if [ -f "$DEST" ] && [ "$(stat -f%z "$DEST")" -eq "$EXPECTED_BYTES" ]; then
    echo "Model already present: $DEST"
    exit 0
fi

# If curl dies mid-transfer, set -e aborts before the size check below, which
# would leave a truncated model on disk. Clean it up on any exit path so a
# corrupt file is never left where the app would load it.
cleanup_partial() {
    if [ -f "$DEST" ] && [ "$(stat -f%z "$DEST")" -ne "$EXPECTED_BYTES" ]; then
        rm -f "$DEST"
        echo "Removed incomplete download: $DEST" >&2
    fi
    return 0
}
trap cleanup_partial EXIT

echo "Downloading pose_landmarker_full.task (9.4MB)..."
curl -fsSL -o "$DEST" "$URL"
ACTUAL=$(stat -f%z "$DEST")
if [ "$ACTUAL" -ne "$EXPECTED_BYTES" ]; then
    echo "Size mismatch: got $ACTUAL, expected $EXPECTED_BYTES" >&2
    rm -f "$DEST"
    exit 1
fi
echo "Done: $DEST"
