"""Fabricate Landmarks for known postures so tests need no camera or model.

Coordinates are normalized image space, y grows DOWNWARD. Values are chosen to
be anatomically plausible for a seated person filling most of the frame.
Landmarks that a given view cannot see are given visibility 0.0.
"""
from __future__ import annotations

import math

from posture.types import LM, Landmarks, Point

_HIDDEN = Point(0.0, 0.0, 0.0)


def _blank() -> list[Point]:
    return [_HIDDEN] * 33


def _finish(points: list[Point]) -> Landmarks:
    return Landmarks(points=tuple(points))


def upright_front() -> Landmarks:
    """Square-on, shoulders level, head centred."""
    pts = _blank()
    pts[LM.NOSE] = Point(0.50, 0.30)
    pts[LM.LEFT_EYE] = Point(0.46, 0.27)
    pts[LM.RIGHT_EYE] = Point(0.54, 0.27)
    pts[LM.LEFT_EAR] = Point(0.42, 0.28)
    pts[LM.RIGHT_EAR] = Point(0.58, 0.28)
    pts[LM.LEFT_SHOULDER] = Point(0.35, 0.55)
    pts[LM.RIGHT_SHOULDER] = Point(0.65, 0.55)
    pts[LM.LEFT_HIP] = Point(0.40, 0.95)
    pts[LM.RIGHT_HIP] = Point(0.60, 0.95)
    return _finish(pts)


def tilted_front(tilt_deg: float) -> Landmarks:
    """Rotate the shoulder line by tilt_deg about its midpoint (positive = right side drops)."""
    pts = list(upright_front().points)
    cx, cy = 0.50, 0.55
    half = 0.15
    rad = math.radians(tilt_deg)
    dx, dy = half * math.cos(rad), half * math.sin(rad)
    pts[LM.LEFT_SHOULDER] = Point(cx - dx, cy - dy)
    pts[LM.RIGHT_SHOULDER] = Point(cx + dx, cy + dy)
    return _finish(pts)


def upright_side(cva_deg: float = 62.0) -> Landmarks:
    """Left-facing profile. Only the near-side ear/shoulder/hip are visible.

    Ear is placed so the shoulder->ear line sits cva_deg above horizontal.
    A NEGATIVE cva_deg puts the ear below the shoulder, which is what a head
    collapsed onto the chest looks like. See dropped_head_side.
    """
    pts = _blank()
    shoulder = Point(0.50, 0.55)
    reach = 0.28
    rad = math.radians(cva_deg)
    pts[LM.LEFT_SHOULDER] = shoulder
    pts[LM.LEFT_EAR] = Point(shoulder.x + reach * math.cos(rad),
                             shoulder.y - reach * math.sin(rad))
    pts[LM.LEFT_HIP] = Point(0.50, 0.95)
    pts[LM.NOSE] = Point(pts[LM.LEFT_EAR].x + 0.08, pts[LM.LEFT_EAR].y + 0.02)
    # Far side present but effectively invisible, as MediaPipe reports in profile.
    pts[LM.RIGHT_SHOULDER] = Point(0.50, 0.55, 0.15)
    pts[LM.RIGHT_EAR] = Point(0.45, 0.28, 0.10)
    pts[LM.RIGHT_HIP] = Point(0.50, 0.95, 0.15)
    return _finish(pts)


def forward_head_side(severity: float = 1.0) -> Landmarks:
    """Lower CVA means worse forward head. severity 0 -> 62deg, 1 -> 42deg."""
    return upright_side(cva_deg=62.0 - 20.0 * severity)


def dropped_head_side(cva_deg: float = -61.9) -> Landmarks:
    """Head collapsed forward and DOWN, ear BELOW the shoulder: the worst case.

    The ear is still forward of the shoulder, so this is the continuation of
    forward_head_side past horizontal, not a separate posture. The default is
    the chin-on-chest case; more negative is a deeper collapse.
    """
    return upright_side(cva_deg=cva_deg)


def mirrored(lm: Landmarks) -> Landmarks:
    """Flip a fixture horizontally, as a camera on the opposite side would see it.

    Used to prove the trunk-angle sign is derived from facing direction rather
    than baked into the fixture's arbitrary choice of which way the person faces.
    """
    return Landmarks(points=tuple(
        Point(1.0 - pt.x, pt.y, pt.visibility) for pt in lm.points
    ))


def reclined_side(lean_deg: float = 15.0) -> Landmarks:
    """Trunk rotated BACKWARD about the hip: reclining, not slouching."""
    return slouched_side(-lean_deg)


def slouched_side(lean_deg: float = 20.0) -> Landmarks:
    """Trunk rotated forward about the hip, with the head carried along."""
    pts = list(upright_side().points)
    hip = pts[LM.LEFT_HIP]
    rad = math.radians(lean_deg)

    def rotate(pt: Point) -> Point:
        dx, dy = pt.x - hip.x, pt.y - hip.y
        return Point(hip.x + dx * math.cos(rad) - dy * math.sin(rad),
                     hip.y + dx * math.sin(rad) + dy * math.cos(rad),
                     pt.visibility)

    for idx in (LM.LEFT_SHOULDER, LM.LEFT_EAR, LM.NOSE):
        pts[idx] = rotate(pts[idx])
    return _finish(pts)


def tilted_front_ears(tilt_deg: float) -> Landmarks:
    """Rotate the ear line by tilt_deg about its midpoint (positive = right side drops).

    Mirrors tilted_front's approach, but moves LEFT_EAR/RIGHT_EAR instead of the
    shoulders, so head_tilt_deg can be exercised independently of shoulder_tilt_deg.
    """
    pts = list(upright_front().points)
    cx, cy = 0.50, 0.28
    half = 0.08
    rad = math.radians(tilt_deg)
    dx, dy = half * math.cos(rad), half * math.sin(rad)
    pts[LM.LEFT_EAR] = Point(cx - dx, cy - dy)
    pts[LM.RIGHT_EAR] = Point(cx + dx, cy + dy)
    return _finish(pts)
