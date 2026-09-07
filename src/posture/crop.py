"""Crop to the person before anything leaves the process.

The desk camera sees the room. Cropping to the MediaPipe bounding box removes
what is BESIDE the desk: the rest of the room, the monitors, a doorway, anyone
who walks past to the left or right. It also cuts token cost and removes
distractors that could pull the model's attention off the posture.

It does not remove what is directly BEHIND the desk. The pose landmarks run
ear to hip, so the crop is a tall column, measured at roughly a quarter of the
frame width but three quarters of its height. A colleague standing behind the
chair, or a whiteboard on the wall behind it, is inside that column and gets
uploaded. This cannot be tightened without cutting off the ear or the hip, and
cva_deg and trunk_angle_deg are computed from exactly those. The README states
this limitation outright rather than implying the crop isolates the person.
"""
from __future__ import annotations

import cv2
import numpy as np

from posture.landmarks import PoseDetector
from posture.types import Landmarks

MAX_EDGE_PX = 768  # downscale cap; posture is legible well below this
MIN_CROP_PX = 8    # below this there is no person region, only noise


class CropUnavailable(Exception):
    """No trustworthy person region, so there is nothing safe to send.

    Raised rather than returning the frame. Every caller must treat this as
    "skip this comparison", never as "send what we have".
    """


def crop_to_person(frame: np.ndarray, lm: Landmarks, padding: float = 0.10) -> np.ndarray:
    """Crop to the padded person bounding box.

    Raises CropUnavailable if no person region can be established. It fails
    closed on purpose: the uncropped frame contains the room behind the desk,
    and the cases that land here are exactly the ones where the detector is
    least confident a person is present at all.
    """
    box = PoseDetector.bounding_box(lm, padding=padding)
    if box is None:
        raise CropUnavailable("no landmark cleared the visibility threshold")

    height, width = frame.shape[:2]
    x0, y0, x1, y1 = box
    left = max(0, int(x0 * width))
    top = max(0, int(y0 * height))
    right = min(width, int(x1 * width))
    bottom = min(height, int(y1 * height))

    if right - left < MIN_CROP_PX or bottom - top < MIN_CROP_PX:
        raise CropUnavailable(
            f"person region collapsed to {right - left}x{bottom - top}px"
        )
    return frame[top:bottom, left:right]


def encode_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    """Downscale to MAX_EDGE_PX then JPEG-encode."""
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest > MAX_EDGE_PX:
        scale = MAX_EDGE_PX / longest
        frame = cv2.resize(frame, (int(width * scale), int(height * scale)),
                           interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buffer.tobytes()
