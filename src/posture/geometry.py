"""Pure 2D geometry. No imports beyond stdlib math and posture.types.

COORDINATE CONVENTION: inputs are image coordinates, y grows DOWNWARD.
The angle_from_* helpers flip y internally so results read naturally
(a point above another yields a positive elevation).
"""
from __future__ import annotations

import math

from posture.types import Point

_EPS = 1e-9


def distance(a: Point, b: Point) -> float:
    return math.hypot(b.x - a.x, b.y - a.y)


def angle_between_three_points(a: Point, b: Point, c: Point) -> float:
    """Angle at vertex b formed by a-b-c, in [0, 180] degrees.

    Uses atan2(cross, dot) rather than acos(dot/norms) because acos loses
    precision and can go out of domain for near-collinear points.
    Returns 0.0 for degenerate input rather than NaN.
    """
    v1x, v1y = a.x - b.x, a.y - b.y
    v2x, v2y = c.x - b.x, c.y - b.y
    n1 = math.hypot(v1x, v1y)
    n2 = math.hypot(v2x, v2y)
    if n1 < _EPS or n2 < _EPS:
        return 0.0
    cross = abs(v1x * v2y - v1y * v2x)
    dot = v1x * v2x + v1y * v2y
    return math.degrees(math.atan2(cross, dot))


def rotation_angle_deg(a: Point, b: Point) -> float:
    """Angle of the line a->b measured against horizontal, in (-180, 180].

    Raw image coordinates, no y flip. Prefer angle_from_horizontal unless you
    specifically want image-space orientation.
    """
    return math.degrees(math.atan2(b.y - a.y, b.x - a.x))


def angle_from_horizontal(a: Point, b: Point) -> float:
    """Elevation of b relative to a, in (-90, 90] degrees.

    Positive means b sits ABOVE a on screen. This is the craniovertebral angle
    primitive: pass (shoulder, ear).
    """
    dx = b.x - a.x
    dy = -(b.y - a.y)  # flip to math orientation
    if abs(dx) < _EPS and abs(dy) < _EPS:
        return 0.0
    return math.degrees(math.atan2(dy, abs(dx))) if dx != 0 else (90.0 if dy > 0 else -90.0)


def angle_from_vertical(a: Point, b: Point) -> float:
    """Tilt of the line a->b away from vertical, in (-180, 180] degrees.

    Zero means b is directly above a. Positive means b is displaced toward
    greater x. This is the trunk angle primitive: pass (hip, shoulder).
    """
    dx = b.x - a.x
    dy = -(b.y - a.y)
    if abs(dx) < _EPS and abs(dy) < _EPS:
        return 0.0
    return math.degrees(math.atan2(dx, dy))
