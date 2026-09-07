"""Webcam capture. Opens, grabs, and releases per sample.

Holding the device open for a faster loop was considered and rejected: it would
stop Zoom and every other app from using the webcam. The cost of reopening is
paid deliberately so the camera stays shareable.
"""
from __future__ import annotations

import contextlib
import logging
import time
from functools import lru_cache

import cv2
import numpy as np

from posture.config import Config
from posture.types import Outcome

logger = logging.getLogger("posture")


@contextlib.contextmanager
def _device(index: int):
    """Always release, including on the not-opened path (spec H2)."""
    cap = cv2.VideoCapture(index)
    try:
        yield cap
    finally:
        cap.release()


def capture(index: int, warmup_frames: int = 3, settle_s: float = 0.15
            ) -> np.ndarray | None:
    """Grab one frame, discarding warmup frames so auto-exposure can settle.

    Two separate mechanisms, because they address different problems. The warmup
    reads flush frames the driver buffered at open time, which are frequently
    black or badly exposed. The short sleep then gives auto-exposure and
    auto-white-balance time to converge, which the reads alone do not guarantee:
    they can all return before the sensor has finished adjusting. 0.15s is a
    conservative starting value, not a measured constant; if captures come back
    dark on a particular camera, raise it.

    Returns None if the device will not open (usually another app holds it) or
    if every read fails.
    """
    with _device(index) as cap:
        if not cap.isOpened():
            logger.debug("camera %s would not open", index)
            return None
        for _ in range(warmup_frames):
            cap.read()
        if settle_s:
            time.sleep(settle_s)
        ok, frame = cap.read()
        if not ok or frame is None:
            logger.debug("camera %s opened but returned no frame", index)
            return None
        return frame


@lru_cache(maxsize=1)
def discover(max_index: int = 8) -> tuple[int, ...]:
    """Probe for working camera indices. Cached: v2 re-probed on every check."""
    found = []
    for index in range(max_index):
        with _device(index) as cap:
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    found.append(index)
    logger.info("discovered cameras: %s", found)
    return tuple(found)


def capture_roles(config: Config) -> tuple[dict[str, np.ndarray], Outcome | None]:
    """Capture one frame per configured role.

    Returns (frames_by_role, outcome). Outcome is None on success. A missing
    side camera is normal (the user undocked) and is not an error; a missing
    front camera means something is wrong.
    """
    frames: dict[str, np.ndarray] = {}
    for role, index in config.camera_indices.items():
        frame = capture(index, warmup_frames=config.capture_warmup_frames)
        if frame is not None:
            frames[role] = frame
        else:
            # A configured camera being absent is normal (undocked, unplugged).
            # It only becomes an outcome when NO camera produced a frame, which
            # is handled below. Do not assume the front camera always exists:
            # a single webcam may be mounted to the side instead.
            logger.debug("camera role %s produced no frame", role)

    if frames:
        return frames, None
    if not discover():
        return {}, Outcome.CAMERA_UNAVAILABLE
    # Devices exist but none would open, so something else holds them.
    return {}, Outcome.CAMERA_BUSY
