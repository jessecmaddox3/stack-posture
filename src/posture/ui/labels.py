"""User-facing label strings. Pure, so they can be tested without a window server.

app.py cannot be imported headless, which means anything deciding what the user is
told would otherwise be untestable. Keeping that decision here rather than inline
in the menu bar class is what makes it pinnable.
"""
from __future__ import annotations

from collections.abc import Iterable


def measuring_text(capability: str | None, *, lean_measurable: bool) -> str:
    """Describe what a check with this camera capability actually measured.

    capability is the "+"-joined set of camera roles that produced a usable
    MEASUREMENT, so "front", "side", "front+side", or None when nothing did.
    It is deliberately not the set of cameras that returned a frame: a side
    camera can be connected, focused, and still yield no readable angle, and a
    row claiming "front+side" while carrying no side metric would defeat the
    whole point of the column. The record's `cameras` field still holds what was
    captured, so the two disagreeing is the signal for "connected but useless".

    That distinction is why the strings below say what is MISSING rather than
    why. "no side camera" would be a diagnosis, and often the wrong one.

    lean_measurable is deliberately REQUIRED rather than defaulted to True. A
    side camera is necessary for trunk lean but not sufficient: H6b drops the
    trunk angle whenever calibration could not decide which way the camera
    faces, and monitoring then refuses to guess per frame, so lean is never
    measured no matter how many frames that camera produces. A default of True
    would let a future call site quietly re-assert the claim this argument
    exists to retract, which is the exact failure mode being fixed.
    """
    roles = set((capability or "").split("+")) - {""}
    has_front, has_side = "front" in roles, "side" in roles
    if has_side and not lean_measurable:
        # The side camera is working and the craniovertebral angle does not
        # depend on the facing sign, so forward head IS still measured. Only
        # lean is gone, and saying so names the fix (recalibrate) rather than
        # implying the camera is broken.
        if has_front:
            return "Measuring: posture and forward head angle (lean not calibrated)"
        return ("Measuring: forward head angle only "
                "(no front reading, lean not calibrated)")
    if has_front and has_side:
        return "Measuring: posture, forward head, and lean"
    if has_side:
        return "Measuring: forward head angle and lean (no front reading)"
    if has_front:
        return "Measuring: posture only (no side reading)"
    return "Measuring: nothing captured in the last check"


def baseline_measuring_text(metric_names: Iterable[str]) -> str:
    """Describe what a BASELINE holds. Only a fallback, before any check has run.

    Once a check exists, measuring_text() on its capability is ground truth for
    the present: a baseline calibrated with a side camera goes on describing that
    camera long after it is unplugged.
    """
    available = set(metric_names)
    has_cva = "cva_deg" in available
    has_trunk = "trunk_angle_deg" in available
    if has_cva and has_trunk:
        return "Measuring: posture, forward head, and lean"
    if has_cva:
        return "Measuring: posture and forward head angle"
    if has_trunk:
        return "Measuring: posture and trunk lean"
    return "Measuring: side-to-side only (no side reading in the baseline)"
