#!/usr/bin/env python3
"""Capture one labelled frame per camera index so you can fill in config.toml.

Usage: uv run python scripts/identify_cameras.py
Writes ~/.posture_monitor/camera_previews/cam<N>.jpg and prints what it found.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from posture import cameras  # noqa: E402
from posture.config import DATA_DIR  # noqa: E402


def _camera_hardware_present() -> bool:
    """True when macOS reports camera hardware, even if capture is refused.

    On the FIRST camera access ever, macOS pops an authorisation prompt and
    OpenCV does not wait for the answer: every probe fails within about a
    second, indistinguishable from having no camera at all. system_profiler
    reports the hardware regardless of whether capture is permitted, so it is
    the only way to tell "no camera" apart from "not authorised yet".
    """
    probe = subprocess.run(
        ["system_profiler", "SPCameraDataType"],
        capture_output=True, text=True, timeout=15,
    )
    return "Model ID" in probe.stdout


def _no_camera_message(hardware_present: bool) -> list[str]:
    """Pure: what to tell the user once discovery has already come back empty."""
    if hardware_present:
        return [
            "A camera exists but could not be opened.",
            "On the FIRST run macOS asks for camera access and OpenCV does",
            "not wait for the answer, so every probe fails. Grant access if",
            "prompted, then RUN THIS COMMAND AGAIN.",
            "If it keeps failing, check System Settings > Privacy and",
            "Security > Camera and confirm your terminal app is enabled.",
        ]
    return ["No camera hardware found. Connect a camera and try again."]


def main() -> int:
    out_dir = DATA_DIR / "camera_previews"
    print("This saves UNCROPPED full frames, showing whatever each camera can see")
    print("including the room behind you, so you can tell the angles apart.")
    print(f"They go to {out_dir} and are not sent anywhere.")
    print("Delete them when you are done (the command is printed at the end).\n")
    out_dir.mkdir(parents=True, exist_ok=True)
    found = cameras.discover(max_index=8)
    if not found:
        for line in _no_camera_message(_camera_hardware_present()):
            print(line)
        return 1
    for index in found:
        frame = cameras.capture(index)
        if frame is None:
            print(f"  index {index}: opened during discovery but gave no frame")
            continue
        path = out_dir / f"cam{index}.jpg"
        cv2.imwrite(str(path), frame)
        print(f"  index {index}: {frame.shape[1]}x{frame.shape[0]} -> {path}")
    print(f"\nOpen {out_dir} and note which index is which, then add to config.toml:")
    print("\n[cameras]\nfront = 0\nside = 2\n")
    print("These previews are full frames. Delete them when you are done:")
    print(f"  rm -rf {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
