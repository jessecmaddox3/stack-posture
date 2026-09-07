import pytest

from posture.geometry import (
    angle_between_three_points, angle_from_horizontal, angle_from_vertical,
    distance, rotation_angle_deg,
)
from posture.types import Point


def p(x, y):
    return Point(x, y)


def test_right_angle_at_vertex():
    assert angle_between_three_points(p(0, 1), p(0, 0), p(1, 0)) == pytest.approx(90.0)


def test_straight_line_is_180():
    assert angle_between_three_points(p(-1, 0), p(0, 0), p(1, 0)) == pytest.approx(180.0)


def test_coincident_points_return_zero_not_nan():
    # Degenerate input must not produce NaN; downstream scoring would silently poison.
    assert angle_between_three_points(p(0, 0), p(0, 0), p(1, 0)) == 0.0


def test_rotation_angle_horizontal_is_zero():
    assert rotation_angle_deg(p(0, 0), p(1, 0)) == pytest.approx(0.0)


def test_angle_from_horizontal_point_above_is_positive():
    # b is ABOVE a in image coords (smaller y), so elevation is positive.
    assert angle_from_horizontal(p(0, 1), p(1, 0)) == pytest.approx(45.0)


def test_angle_from_horizontal_point_below_is_negative():
    assert angle_from_horizontal(p(0, 0), p(1, 1)) == pytest.approx(-45.0)


def test_angle_from_vertical_upright_is_zero():
    # a = hip, b = shoulder directly above it.
    assert angle_from_vertical(p(0.5, 1.0), p(0.5, 0.5)) == pytest.approx(0.0)


def test_angle_from_vertical_forward_lean_is_positive():
    # Shoulder ahead of hip (greater x) means leaning forward.
    assert angle_from_vertical(p(0.5, 1.0), p(0.6, 0.5)) > 0


def test_distance():
    assert distance(p(0, 0), p(3, 4)) == pytest.approx(5.0)


def test_angle_from_horizontal_is_symmetric_in_magnitude():
    a, b = p(0.2, 0.8), p(0.7, 0.3)
    assert abs(angle_from_horizontal(a, b)) == pytest.approx(abs(angle_from_horizontal(b, a)))
