"""API key resolution. Keychain first, environment variable fallback."""
from __future__ import annotations

import os
import subprocess

KEYCHAIN_SERVICE = "posture-monitor-gemini"
ENV_VAR = "POSTURE_GEMINI_API_KEY"


def get_api_key() -> str | None:
    """Return the Gemini API key, or None if it is not configured anywhere."""
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-w"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    value = os.environ.get(ENV_VAR, "").strip()
    return value or None


def store_api_key(key: str) -> None:
    """Write the key into the login keychain, replacing any existing entry."""
    subprocess.run(
        ["security", "add-generic-password", "-U",
         "-s", KEYCHAIN_SERVICE, "-a", os.environ.get("USER", "posture"), "-w", key],
        check=True, capture_output=True,
    )
