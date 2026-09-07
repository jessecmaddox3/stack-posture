import pytest

from posture.ui.labels import baseline_measuring_text, measuring_text


@pytest.mark.parametrize("capability,expected", [
    ("front+side", "Measuring: posture, forward head, and lean"),
    ("side", "Measuring: forward head angle and lean (no front reading)"),
    ("front", "Measuring: posture only (no side reading)"),
    (None, "Measuring: nothing captured in the last check"),
    ("", "Measuring: nothing captured in the last check"),
])
def test_measuring_text_describes_what_was_actually_captured(capability, expected):
    assert measuring_text(capability, lean_measurable=True) == expected


def test_measuring_text_never_claims_a_camera_that_did_not_contribute():
    # The specific failure: after undocking, the line must stop mentioning
    # forward head, because nothing can see it any more.
    front_only = measuring_text("front", lean_measurable=True)
    assert "forward head" not in front_only
    assert "lean" not in front_only


@pytest.mark.parametrize("capability,expected", [
    ("front+side", "Measuring: posture and forward head angle (lean not calibrated)"),
    ("side", "Measuring: forward head angle only "
             "(no front reading, lean not calibrated)"),
])
def test_an_uncalibrated_facing_sign_stops_the_line_promising_lean(capability, expected):
    """H6b drops the trunk angle when the facing sign is ambiguous, and _sample
    passes require_calibrated=True so monitoring never guesses it per frame. The
    camera is there and it works, but lean is not being measured and will not be
    until he recalibrates, so the line must not go on claiming it.
    """
    assert measuring_text(capability, lean_measurable=False) == expected


def test_the_facing_sign_is_what_changes_the_claim_not_the_camera():
    """Positive control. Both calls have a working side camera; only the
    calibration differs. Without this, an implementation that always dropped
    lean would satisfy the test above.
    """
    with_sign = measuring_text("front+side", lean_measurable=True)
    without_sign = measuring_text("front+side", lean_measurable=False)
    assert "lean" in with_sign
    assert "lean not calibrated" in without_sign
    assert "and lean" not in without_sign
    # Forward head does not depend on the facing sign, so it survives.
    assert "forward head" in with_sign and "forward head" in without_sign


def test_lean_measurable_has_no_default_so_a_caller_cannot_forget_it():
    """A default of True would let a new call site silently re-assert the claim
    this argument exists to retract, which is the defect itself."""
    with pytest.raises(TypeError):
        measuring_text("front+side")


def test_baseline_text_is_only_a_fallback_and_can_disagree_with_the_present():
    """This disagreement IS the defect H5 fixed, so pin that they differ.

    A baseline calibrated with a side camera describes forward head and lean
    forever. Once a check has run with the front camera alone, the present is
    ground truth and must say so. If these two ever returned the same string for
    this input, the fallback would be indistinguishable from the live answer and
    the bug would be invisible again.
    """
    calibrated_with_side = baseline_measuring_text({"cva_deg", "trunk_angle_deg",
                                                    "shoulder_tilt_deg"})
    now_front_only = measuring_text("front", lean_measurable=True)
    assert "forward head" in calibrated_with_side
    assert "forward head" not in now_front_only
    assert calibrated_with_side != now_front_only


@pytest.mark.parametrize("metrics,expected", [
    ({"cva_deg", "trunk_angle_deg"}, "Measuring: posture, forward head, and lean"),
    ({"cva_deg"}, "Measuring: posture and forward head angle"),
    ({"trunk_angle_deg"}, "Measuring: posture and trunk lean"),
    ({"shoulder_tilt_deg"}, "Measuring: side-to-side only (no side reading in the baseline)"),
    (set(), "Measuring: side-to-side only (no side reading in the baseline)"),
])
def test_baseline_measuring_text_names_only_what_the_baseline_holds(metrics, expected):
    assert baseline_measuring_text(metrics) == expected
