"""MediaPipe Pose wrapper. The only module that knows mediapipe exists.

Measured at roughly 7ms per 720p frame on Apple Silicon, which is what makes a
30 second sample loop free.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from posture.types import Landmarks, Point

logger = logging.getLogger("posture")

MIN_BOX_VISIBILITY = 0.5


class PoseDetector:
    """Context manager wrapping a MediaPipe PoseLandmarker in IMAGE mode."""

    def __init__(self, model_path: Path, min_confidence: float = 0.5) -> None:
        self._model_path = Path(model_path)
        self._min_confidence = min_confidence
        self._landmarker = None

    def __enter__(self) -> "PoseDetector":
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Pose model missing at {self._model_path}. Run scripts/fetch_model.sh"
            )
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self._model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=self._min_confidence,
            min_pose_presence_confidence=self._min_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        return self

    def __exit__(self, *exc_info) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def detect(self, bgr_frame: np.ndarray) -> Landmarks | None:
        """Detect one pose. Returns None when no person is present."""
        if self._landmarker is None:
            raise RuntimeError("PoseDetector used outside its context manager")
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(image)
        if not result.pose_landmarks:
            return None
        raw = result.pose_landmarks[0]
        return Landmarks(points=tuple(
            Point(x=p.x, y=p.y, visibility=getattr(p, "visibility", 1.0)) for p in raw
        ))

    @staticmethod
    def bounding_box(lm: Landmarks, padding: float = 0.08
                     ) -> tuple[float, float, float, float] | None:
        """Normalized (x0, y0, x1, y1) around the confidently visible landmarks.

        None when nothing clears MIN_BOX_VISIBILITY. The previous full-frame
        return was unsafe: it handed back the whole room under the one
        condition that means no person was confidently found.

        `padding` is a fraction of the PERSON BOX's own width and height, not
        of the frame. It used to be frame-relative, which meant padding=0.10
        added 10% of the frame width on each side no matter how small the
        person was. Measured on this repo's synthetic postures, that roughly
        doubled the width of a profile view: a 0.211-wide person box came out
        0.411 wide, so more than half of what was uploaded was room. Scaling
        the margin to the person keeps the same visual proportion at any
        distance from the camera.

        This narrows the box. It does NOT meaningfully shorten it, and no
        setting here can: MediaPipe's pose landmarks span ear to hip, which at
        desk-webcam distance is genuinely about two thirds of the frame
        height. The crop is a tall column around the user, so anything directly
        behind them at their own height is still inside it. See the privacy
        section of the README, which states this rather than implying the crop
        removes everyone else.

        Padded and clamped to the frame. Used to crop before anything is sent
        to Gemini, so the room beside the desk never leaves the machine.
        """
        visible = [p for p in lm.points if p.visibility >= MIN_BOX_VISIBILITY]
        if not visible:
            return None  # no person region; callers must not fall back to the frame
        xs = [p.x for p in visible]
        ys = [p.y for p in visible]
        left, right = min(xs), max(xs)
        top, bottom = min(ys), max(ys)
        pad_x = padding * (right - left)
        pad_y = padding * (bottom - top)
        clamp = lambda v: max(0.0, min(1.0, v))  # noqa: E731
        return (clamp(left - pad_x), clamp(top - pad_y),
                clamp(right + pad_x), clamp(bottom + pad_y))
