import pytest

from posture.metrics import (
    compute_front_metrics, compute_side_metrics, facing_sign_from, merge_metrics,
)
from posture.types import PostureMetrics
from tests.synthetic import (
    dropped_head_side, forward_head_side, mirrored, reclined_side, slouched_side,
    tilted_front, tilted_front_ears, upright_front, upright_side,
)


def test_level_shoulders_give_zero_tilt():
    m = compute_front_metrics(upright_front())
    assert m.shoulder_tilt_deg == pytest.approx(0.0, abs=0.01)


def test_shoulder_tilt_matches_the_synthetic_rotation():
    # Signed, not abs(): tilted_front(10.0) drops the right shoulder, and
    # shoulder_tilt_deg must come back positive, not merely 10 in magnitude.
    m = compute_front_metrics(tilted_front(10.0))
    assert m.shoulder_tilt_deg == pytest.approx(10.0, abs=0.5)


def test_shoulder_tilt_sign_flips_with_negative_synthetic_rotation():
    m = compute_front_metrics(tilted_front(-10.0))
    assert m.shoulder_tilt_deg == pytest.approx(-10.0, abs=0.5)


def test_head_tilt_matches_the_synthetic_rotation():
    m = compute_front_metrics(tilted_front_ears(10.0))
    assert m.head_tilt_deg == pytest.approx(10.0, abs=0.5)


def test_head_tilt_sign_flips_with_negative_synthetic_rotation():
    m = compute_front_metrics(tilted_front_ears(-10.0))
    assert m.head_tilt_deg == pytest.approx(-10.0, abs=0.5)


def test_centred_head_gives_near_zero_lateral_ratio():
    m = compute_front_metrics(upright_front())
    assert abs(m.head_lateral_ratio) < 0.02


def test_proximity_ratio_matches_the_synthetic_eye_and_shoulder_geometry():
    # upright_front: eye distance 0.08, shoulder width 0.30 -> ratio 0.2667.
    m = compute_front_metrics(upright_front())
    assert m.proximity_ratio == pytest.approx(0.08 / 0.30, abs=0.001)


def test_front_view_yields_no_cva():
    # CVA is a side-view measurement. Front metrics must not invent one.
    assert compute_front_metrics(upright_front()).cva_deg is None


def test_cva_matches_the_synthetic_angle():
    assert compute_side_metrics(upright_side(cva_deg=62.0)).cva_deg == pytest.approx(62.0, abs=0.5)


def test_forward_head_lowers_cva():
    good = compute_side_metrics(upright_side()).cva_deg
    bad = compute_side_metrics(forward_head_side(1.0)).cva_deg
    assert bad < good
    # A 20 degree drop from the fixture's upright 62. No absolute cutoff is
    # asserted: this number is a raw shoulder-to-ear elevation on an
    # uncalibrated 2D view, so only movement relative to a baseline means
    # anything. See compute_side_metrics for why.
    assert bad == pytest.approx(42.0, abs=0.5)


def test_ear_below_the_shoulder_yields_a_negative_cva():
    """The sign is the whole point: below horizontal must read below zero.

    abs() folded this onto the positive range, so a head dropped onto the
    chest reported the same number as a mild forward head.
    """
    cva = compute_side_metrics(dropped_head_side(-61.9)).cva_deg
    assert cva == pytest.approx(-61.9, abs=0.5)
    assert cva < 0.0


def test_cva_decreases_monotonically_from_upright_to_full_collapse():
    """Four postures, worst last. Any fold in the metric breaks this ordering.

    Under abs() the last two came back positive, so a dropped head read as a
    mild forward head, and a deeper collapse read BETTER than a shallower one.
    """
    upright = compute_side_metrics(upright_side(84.3)).cva_deg
    mild = compute_side_metrics(upright_side(61.9)).cva_deg
    dropped = compute_side_metrics(dropped_head_side(-61.9)).cva_deg
    severe = compute_side_metrics(dropped_head_side(-68.2)).cva_deg

    assert upright > mild > dropped > severe
    assert [upright, mild, dropped, severe] == pytest.approx(
        [84.3, 61.9, -61.9, -68.2], abs=0.5)


def test_upright_side_trunk_angle_is_near_zero():
    assert compute_side_metrics(upright_side()).trunk_angle_deg == pytest.approx(0.0, abs=1.0)


def test_slouch_raises_trunk_angle():
    assert compute_side_metrics(slouched_side(20.0)).trunk_angle_deg == pytest.approx(20.0, abs=1.5)


def test_reclining_yields_a_negative_trunk_angle():
    # Reclining must be distinguishable from slouching, not folded together by
    # abs(). Scoring only penalises positive deviation, so this reads as fine.
    assert compute_side_metrics(reclined_side(15.0)).trunk_angle_deg == pytest.approx(-15.0, abs=1.5)


def test_trunk_angle_sign_survives_a_mirrored_camera():
    # A camera on the other side mirrors the image. Leaning forward must still
    # read positive, otherwise the sign means nothing without knowing the setup.
    forward = compute_side_metrics(slouched_side(20.0)).trunk_angle_deg
    forward_mirrored = compute_side_metrics(mirrored(slouched_side(20.0))).trunk_angle_deg
    assert forward > 0 and forward_mirrored > 0
    assert forward_mirrored == pytest.approx(forward, abs=1.5)


def test_facing_sign_and_trunk_measurement_use_the_same_side():
    """A forward lean must never read as a recline because the two halves of the
    calculation picked opposite sides of the body.

    Regression test for a real defect: facing_sign_from ranked sides by ear
    visibility alone while compute_side_metrics ranked by ear plus shoulder. Here
    the ear-only ranking favours LEFT (0.6 vs 0.55) while ear plus shoulder
    favours RIGHT (1.5 vs 0.7), so the old code derived the sign from the left ear
    and measured the trunk from the right shoulder and hip, turning +7.1 into
    -7.1. Every existing fixture keeps ear and shoulder visibility correlated per
    side, so none of them caught it.
    """
    from posture.types import LM, Landmarks, Point

    pts = [Point(0.0, 0.0, 0.0)] * 33
    pts[LM.NOSE] = Point(0.65, 0.30, 0.9)
    pts[LM.RIGHT_EAR] = Point(0.60, 0.30, 0.55)
    pts[LM.RIGHT_SHOULDER] = Point(0.60, 0.55, 0.95)
    pts[LM.RIGHT_HIP] = Point(0.55, 0.95, 0.9)
    pts[LM.LEFT_EAR] = Point(0.70, 0.30, 0.6)
    pts[LM.LEFT_SHOULDER] = Point(0.50, 0.56, 0.1)
    pts[LM.LEFT_HIP] = Point(0.50, 0.95, 0.1)
    lm = Landmarks(points=tuple(pts))

    trunk = compute_side_metrics(lm).trunk_angle_deg
    assert trunk is not None
    assert trunk > 0, "a forward lean must not be reported as a recline"
    assert trunk == pytest.approx(7.1, abs=0.5)


def test_a_frozen_facing_sign_overrides_the_per_frame_estimate():
    # A head turn can invert the per-frame estimate. The calibrated sign must win,
    # so a conversation does not flip the trunk angle while the trunk has not moved.
    lm = slouched_side(20.0)
    assert compute_side_metrics(lm, facing_sign=1.0).trunk_angle_deg == pytest.approx(20.0, abs=1.5)
    assert compute_side_metrics(lm, facing_sign=-1.0).trunk_angle_deg == pytest.approx(-20.0, abs=1.5)


def test_facing_sign_from_reads_the_expected_direction():
    assert facing_sign_from(upright_side()) == 1.0
    assert facing_sign_from(mirrored(upright_side())) == -1.0


def test_facing_sign_from_is_none_without_a_visible_nose():
    from posture.types import LM as _LM
    from posture.types import Landmarks, Point
    pts = list(upright_side().points)
    pts[_LM.NOSE] = Point(pts[_LM.NOSE].x, pts[_LM.NOSE].y, 0.0)
    assert facing_sign_from(Landmarks(points=tuple(pts))) is None


def test_trunk_angle_survives_a_deep_recline():
    # The facing axis must not rotate away as the trunk leans. An
    # ear-versus-shoulder derivation inverted at roughly 28 degrees and reported
    # this posture as +40 (severe slouch), the exact opposite of reality.
    assert compute_side_metrics(reclined_side(40.0)).trunk_angle_deg == pytest.approx(-40.0, abs=1.5)


def test_deep_recline_sign_survives_a_mirrored_camera():
    deep = compute_side_metrics(reclined_side(40.0)).trunk_angle_deg
    deep_mirrored = compute_side_metrics(mirrored(reclined_side(40.0))).trunk_angle_deg
    assert deep < 0 and deep_mirrored < 0
    assert deep_mirrored == pytest.approx(deep, abs=1.5)


def test_trunk_angle_is_none_without_a_visible_nose():
    # The facing axis needs the nose, so without it the sign would be a guess.
    from posture.types import LM as _LM
    from posture.types import Landmarks, Point
    pts = list(upright_side().points)
    pts[_LM.NOSE] = Point(pts[_LM.NOSE].x, pts[_LM.NOSE].y, 0.0)
    assert compute_side_metrics(Landmarks(points=tuple(pts))).trunk_angle_deg is None


def test_trunk_angle_is_none_without_a_visible_ear():
    # The facing axis comes from nose-to-ear, so no ear means no trustworthy sign.
    from posture.types import LM as _LM
    from posture.types import Landmarks, Point
    pts = list(upright_side().points)
    pts[_LM.LEFT_EAR] = Point(pts[_LM.LEFT_EAR].x, pts[_LM.LEFT_EAR].y, 0.0)
    pts[_LM.RIGHT_EAR] = Point(0.0, 0.0, 0.0)
    assert compute_side_metrics(Landmarks(points=tuple(pts))).trunk_angle_deg is None


def test_low_visibility_yields_none_not_a_wrong_number():
    from posture.types import Landmarks, Point
    invisible = Landmarks(points=tuple(Point(0.5, 0.5, 0.0) for _ in range(33)))
    m = compute_side_metrics(invisible)
    assert m.cva_deg is None
    assert m.trunk_angle_deg is None


def test_low_visibility_front_yields_none_not_a_wrong_number():
    from posture.types import Landmarks, Point
    invisible = Landmarks(points=tuple(Point(0.5, 0.5, 0.0) for _ in range(33)))
    m = compute_front_metrics(invisible)
    assert m.shoulder_tilt_deg is None
    assert m.head_tilt_deg is None
    assert m.head_lateral_ratio is None
    assert m.proximity_ratio is None


def test_trunk_angle_is_none_when_the_facing_sign_is_unknown():
    """Without a calibrated sign, omit the measurement rather than guess.

    The per-frame fallback is exactly what freezing exists to avoid, so falling
    back to it silently defeats the design.
    """
    assert compute_side_metrics(upright_side(), facing_sign=None,
                                require_calibrated=True).trunk_angle_deg is None


def test_merge_prefers_populated_fields():
    front = PostureMetrics(shoulder_tilt_deg=3.0)
    side = PostureMetrics(cva_deg=55.0)
    merged = merge_metrics(front, side)
    assert merged.shoulder_tilt_deg == 3.0
    assert merged.cva_deg == 55.0


def test_merge_does_not_overwrite_with_none():
    merged = merge_metrics(PostureMetrics(cva_deg=55.0), PostureMetrics())
    assert merged.cva_deg == 55.0
