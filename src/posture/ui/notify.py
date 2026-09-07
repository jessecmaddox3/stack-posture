"""Plain macOS notification. Fallback for when the styled panel is unavailable."""
from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger("posture")


def notify(title: str, message: str, sound: str = "Funk") -> None:
    safe_title = title.replace('"', '\\"')
    safe_message = message.replace('"', '\\"').replace("\n", " ")
    script = (
        f'display notification "{safe_message}" with title "{safe_title}" '
        f'sound name "{sound}"'
    )
    try:
        subprocess.run(["osascript", "-e", script], capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        logger.exception("notification failed")
