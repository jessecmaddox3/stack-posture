"""Shared types. Imported by every module, imports nothing but stdlib."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum


class Rating(str, Enum):
    GOOD = "GOOD"
    DECENT = "DECENT"
    POOR = "POOR"


class Outcome(str, Enum):
    """Explicit per-check result. Never conflate failure with absence."""
    OK = "OK"
    PERSON_ABSENT = "PERSON_ABSENT"
    CAMERA_UNAVAILABLE = "CAMERA_UNAVAILABLE"
    CAMERA_BUSY = "CAMERA_BUSY"
    API_ERROR = "API_ERROR"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    CROP_UNAVAILABLE = "CROP_UNAVAILABLE"


class CameraRole(str, Enum):
    FRONT = "front"
    SIDE = "side"


class LM:
    """MediaPipe Pose landmark indices (33-point model)."""
    NOSE = 0
    LEFT_EYE = 2
    RIGHT_EYE = 5
    LEFT_EAR = 7
    RIGHT_EAR = 8
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24


@dataclass(frozen=True)
class Point:
    """Normalized image coordinates. x,y in [0,1]. y grows DOWNWARD."""
    x: float
    y: float
    visibility: float = 1.0


@dataclass(frozen=True)
class Landmarks:
    points: tuple[Point, ...]

    def __post_init__(self) -> None:
        if len(self.points) != 33:
            raise ValueError(f"expected 33 landmarks, got {len(self.points)}")

    def __getitem__(self, index: int) -> Point:
        return self.points[index]


@dataclass(frozen=True)
class PostureMetrics:
    """All fields optional: which are populated depends on which cameras saw you."""
    shoulder_tilt_deg: float | None = None
    head_tilt_deg: float | None = None
    head_lateral_ratio: float | None = None
    proximity_ratio: float | None = None
    cva_deg: float | None = None
    trunk_angle_deg: float | None = None

    def available(self) -> dict[str, float]:
        return {k: v for k, v in asdict(self).items() if v is not None}
