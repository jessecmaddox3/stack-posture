import pytest

from posture.metrics import compute_side_metrics
from posture.scoring import Baseline, score
from posture.types import PostureMetrics, Rating
from tests.synthetic import dropped_head_side, upright_side


def baseline_at(**kwargs) -> Baseline:
    return Baseline(id=1, created_at="2026-07-29T09:00:00", label="test",
                    metrics=PostureMetrics(**kwargs), facing_sign=1.0)


def test_metrics_matching_baseline_are_good():
    base = baseline_at(cva_deg=60.0, trunk_angle_deg=2.0, shoulder_tilt_deg=0.0)
    result = score(PostureMetrics(cva_deg=60.0, trunk_angle_deg=2.0, shoulder_tilt_deg=0.0), base)
    assert result.rating is Rating.GOOD
    assert result.driver is None


def test_small_cva_drop_is_decent():
    base = baseline_at(cva_deg=60.0)
    result = score(PostureMetrics(cva_deg=53.0), base)
    assert result.rating is Rating.DECENT
    assert result.driver == "cva_deg"


def test_large_cva_drop_is_poor():
    base = baseline_at(cva_deg=60.0)
    result = score(PostureMetrics(cva_deg=44.0), base)
    assert result.rating is Rating.POOR
    assert result.driver == "cva_deg"


def test_cva_above_baseline_is_never_penalised():
    # Sitting up straighter than baseline is not a fault.
    base = baseline_at(cva_deg=60.0)
    assert score(PostureMetrics(cva_deg=70.0), base).rating is Rating.GOOD


def test_a_collapsed_head_rates_poor_against_an_upright_baseline():
    """End to end from landmarks: the worst posture must score the worst.

    Goes through compute_side_metrics rather than a hand-written PostureMetrics
    because the defect lived in the metric, not in the rule. Scoring a literal
    -60 was always POOR; what the camera actually produced for that posture was
    +60, which scored GOOD.
    """
    base = baseline_at(cva_deg=compute_side_metrics(upright_side(62.0)).cva_deg)
    collapsed = compute_side_metrics(dropped_head_side(-61.9))
    result = score(collapsed, base)
    assert result.rating is Rating.POOR
    assert result.driver == "cva_deg"


def test_a_deeper_collapse_never_scores_better_than_a_shallower_one():
    """The rating must not improve as the head falls further.

    Under abs() the deviation shrank back toward baseline past horizontal, so
    the deepest collapse produced the mildest score.
    """
    base = baseline_at(cva_deg=compute_side_metrics(upright_side(62.0)).cva_deg)
    postures = [upright_side(62.0), upright_side(50.0), dropped_head_side(-61.9),
                dropped_head_side(-68.2)]
    drops = [-score(compute_side_metrics(lm), base).deviations["cva_deg"]
             for lm in postures]
    assert drops == sorted(drops)


def test_worst_metric_drives_the_rating():
    base = baseline_at(cva_deg=60.0, trunk_angle_deg=0.0)
    result = score(PostureMetrics(cva_deg=54.0, trunk_angle_deg=25.0), base)
    assert result.rating is Rating.POOR
    assert result.driver == "trunk_angle_deg"


def test_reclining_is_not_penalised():
    # trunk_angle_deg is signed and the rule direction is +1, so a negative
    # deviation (reclining) must not count against the score at all.
    base = baseline_at(trunk_angle_deg=2.0)
    assert score(PostureMetrics(trunk_angle_deg=-20.0), base).rating is Rating.GOOD


def test_a_reclined_baseline_does_not_penalise_sitting_upright():
    """The whole point of signing the trunk angle.

    A PT-approved reclined baseline of -20 must not make upright posture POOR.
    Penalising any increase over baseline did exactly that.
    """
    base = baseline_at(trunk_angle_deg=-20.0)
    assert score(PostureMetrics(trunk_angle_deg=-10.0), base).rating is Rating.GOOD
    assert score(PostureMetrics(trunk_angle_deg=0.0), base).rating is Rating.GOOD


def test_a_reclined_baseline_still_catches_real_forward_lean():
    # Not a licence to ignore slouching: past neutral it scores normally.
    base = baseline_at(trunk_angle_deg=-20.0)
    assert score(PostureMetrics(trunk_angle_deg=20.0), base).rating is Rating.POOR


def test_a_symmetric_metric_measures_from_its_baseline_not_from_neutral():
    """The neutral floor must apply ONLY to direction=+1 rules.

    Every other test for the symmetric metrics uses a baseline of exactly 0.0,
    where max(baseline, 0.0) is a no-op, so a mutant that wrongly extended the
    floor to direction=0 would pass the whole suite. A NEGATIVE baseline
    discriminates, and it is realistic: a real calibration on this project
    produced shoulder_tilt_deg of -1.91, and a persistent lean can be larger.

    With a baseline of -6, deviations must be measured from -6, not from 0.
    """
    base = baseline_at(shoulder_tilt_deg=-6.0)
    # Sitting exactly as calibrated is GOOD. Under the mutant it would be DECENT.
    assert score(PostureMetrics(shoulder_tilt_deg=-6.0), base).rating is Rating.GOOD
    # Level shoulders are a 6 degree change FROM the calibrated lean, so DECENT.
    # Under the mutant this would read GOOD, hiding a real change.
    assert score(PostureMetrics(shoulder_tilt_deg=0.0), base).rating is Rating.DECENT
    # And a 12 degree swing the other way is POOR, not DECENT.
    assert score(PostureMetrics(shoulder_tilt_deg=6.0), base).rating is Rating.POOR


def test_an_upright_baseline_is_unaffected():
    base = baseline_at(trunk_angle_deg=2.0)
    assert score(PostureMetrics(trunk_angle_deg=25.0), base).rating is Rating.POOR
    assert score(PostureMetrics(trunk_angle_deg=-20.0), base).rating is Rating.GOOD


def test_worst_metric_wins_even_when_severity_decreases_along_iteration_order():
    """The worst metric must win on SEVERITY, not on being evaluated last.

    METRIC_RULES iterates cva_deg first, then trunk_angle_deg, then
    shoulder_tilt_deg. Here the FIRST metric is the worst, so an implementation
    that simply keeps the most recent non-good metric would return the wrong
    answer. Every other multi-metric test in this file happens to have severity
    increasing along iteration order, so they all pass against that mutant: a
    "last non-good wins" implementation was verified to pass all of them.
    """
    base = baseline_at(cva_deg=60.0, shoulder_tilt_deg=0.0)
    result = score(PostureMetrics(cva_deg=45.0, shoulder_tilt_deg=6.0), base)
    assert result.rating is Rating.POOR
    assert result.driver == "cva_deg"


def test_shoulder_tilt_is_penalised_in_both_directions():
    base = baseline_at(shoulder_tilt_deg=0.0)
    assert score(PostureMetrics(shoulder_tilt_deg=12.0), base).rating is Rating.POOR
    assert score(PostureMetrics(shoulder_tilt_deg=-12.0), base).rating is Rating.POOR


def test_metrics_absent_from_the_baseline_are_skipped():
    base = baseline_at(cva_deg=60.0)  # no trunk baseline recorded
    result = score(PostureMetrics(cva_deg=60.0, trunk_angle_deg=40.0), base)
    assert result.rating is Rating.GOOD
    assert "trunk_angle_deg" not in result.deviations


def test_front_camera_only_still_produces_a_rating():
    # One webcam is a supported setup (undocked, or a laptop with no side camera).
    # CVA and trunk angle are unavailable, but shoulder tilt, head tilt, lateral
    # offset, and proximity are all front-camera metrics and must still score.
    base = baseline_at(shoulder_tilt_deg=0.0, head_tilt_deg=0.0,
                       head_lateral_ratio=0.0, proximity_ratio=0.27)
    good = score(PostureMetrics(shoulder_tilt_deg=1.0, head_tilt_deg=2.0,
                                head_lateral_ratio=0.01, proximity_ratio=0.28), base)
    assert good.rating is Rating.GOOD

    bad = score(PostureMetrics(shoulder_tilt_deg=12.0, head_tilt_deg=2.0,
                               head_lateral_ratio=0.01, proximity_ratio=0.28), base)
    assert bad.rating is Rating.POOR
    assert bad.driver == "shoulder_tilt_deg"
    # Nothing side-derived may appear when there is no side camera.
    assert "cva_deg" not in bad.deviations
    assert "trunk_angle_deg" not in bad.deviations


def test_no_comparable_metrics_returns_none_rating():
    # A person detected but no usable measurement must not be scored GOOD by default.
    base = baseline_at(cva_deg=60.0)
    result = score(PostureMetrics(), base)
    assert result.rating is None
    assert result.deviations == {}


def test_deviations_are_reported_for_every_comparable_metric():
    base = baseline_at(cva_deg=60.0, trunk_angle_deg=2.0)
    result = score(PostureMetrics(cva_deg=55.0, trunk_angle_deg=6.0), base)
    assert result.deviations["cva_deg"] == pytest.approx(-5.0)
    assert result.deviations["trunk_angle_deg"] == pytest.approx(4.0)


def test_scoring_is_deterministic():
    base = baseline_at(cva_deg=60.0)
    metrics = PostureMetrics(cva_deg=52.3)
    assert all(score(metrics, base) == score(metrics, base) for _ in range(5))
