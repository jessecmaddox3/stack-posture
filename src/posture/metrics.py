"""Landmarks to measurements. Pure: no cv2, no mediapipe, no I/O.

Front camera contributes shoulder tilt, head tilt, lateral head offset, and
screen proximity. Side camera contributes craniovertebral angle and trunk angle,
the two that carry forward head and slouch. Both are SIGNED: see
compute_side_metrics for why folding either onto a positive range destroys it.

Any metric whose source landmarks are not confidently visible is None rather
than a guess. Downstream scoring skips None; it never substitutes a default.
"""
from __future__ import annotations

from dataclasses import asdict

from posture.geometry import angle_from_horizontal, angle_from_vertical, distance
from posture.types import LM, Landmarks, Point, PostureMetrics

MIN_VISIBILITY = 0.5


def _visible(*points: Point) -> bool:
    return all(pt.visibility >= MIN_VISIBILITY for pt in points)


def _midpoint(a: Point, b: Point) -> Point:
    return Point((a.x + b.x) / 2, (a.y + b.y) / 2, min(a.visibility, b.visibility))


def compute_front_metrics(lm: Landmarks) -> PostureMetrics:
    left_sh, right_sh = lm[LM.LEFT_SHOULDER], lm[LM.RIGHT_SHOULDER]
    left_ear, right_ear = lm[LM.LEFT_EAR], lm[LM.RIGHT_EAR]
    left_eye, right_eye = lm[LM.LEFT_EYE], lm[LM.RIGHT_EYE]
    nose = lm[LM.NOSE]

    shoulder_tilt = head_tilt = lateral = proximity = None

    if _visible(left_sh, right_sh):
        # Negated because angle_from_horizontal reports elevation, and we want
        # "positive = right shoulder dropped" to match tilted_front's convention.
        shoulder_tilt = -angle_from_horizontal(left_sh, right_sh)
        shoulder_width = distance(left_sh, right_sh)

        if _visible(nose) and shoulder_width > 1e-6:
            centre = _midpoint(left_sh, right_sh)
            lateral = (nose.x - centre.x) / shoulder_width

        if _visible(left_eye, right_eye) and shoulder_width > 1e-6:
            proximity = distance(left_eye, right_eye) / shoulder_width

    if _visible(left_ear, right_ear):
        head_tilt = -angle_from_horizontal(left_ear, right_ear)

    return PostureMetrics(
        shoulder_tilt_deg=shoulder_tilt,
        head_tilt_deg=head_tilt,
        head_lateral_ratio=lateral,
        proximity_ratio=proximity,
    )


def _near_side(lm: Landmarks) -> tuple[Point, Point, Point]:
    """Pick the near side of a profile view, returning (ear, shoulder, hip).

    By visibility, never by z: MediaPipe z-depth is unreliable for this
    near/far decision.

    Shared by facing_sign_from and compute_side_metrics so the two can never
    select DIFFERENT sides. An earlier version let facing_sign_from rank by ear
    visibility alone while compute_side_metrics ranked by ear plus shoulder, and
    the two could disagree: the facing sign came from one side's ear while the
    trunk was measured from the other side's shoulder and hip, reporting a
    forward lean as a recline.
    """
    left = lm[LM.LEFT_EAR].visibility + lm[LM.LEFT_SHOULDER].visibility
    right = lm[LM.RIGHT_EAR].visibility + lm[LM.RIGHT_SHOULDER].visibility
    if left >= right:
        return lm[LM.LEFT_EAR], lm[LM.LEFT_SHOULDER], lm[LM.LEFT_HIP]
    return lm[LM.RIGHT_EAR], lm[LM.RIGHT_SHOULDER], lm[LM.RIGHT_HIP]


def facing_sign_from(lm: Landmarks) -> float | None:
    """Which way is forward in this frame: +1 toward greater x, -1 toward lesser.

    Best effort only. The nose-to-ear axis is the facing axis, but a single 2D
    view cannot separate head yaw from trunk lean, so a sustained head turn can
    invert this. Prefer the value frozen at calibration time (Baseline.facing_sign),
    which is captured under controlled conditions. This function exists to
    produce that calibration value, and as a fallback when none is stored.
    """
    nose = lm[LM.NOSE]
    ear, _shoulder, _hip = _near_side(lm)
    if not _visible(nose, ear):
        return None
    return 1.0 if nose.x >= ear.x else -1.0


def compute_side_metrics(lm: Landmarks, facing_sign: float | None = None,
                         require_calibrated: bool = False) -> PostureMetrics:
    """Pick the near side via _near_side.

    facing_sign should come from Baseline.facing_sign, frozen at calibration time
    under controlled conditions. When not supplied, falls back to facing_sign_from,
    a per-frame estimate that is less reliable because a sustained head turn can
    invert it.

    require_calibrated=True disables that fallback: when facing_sign is None,
    trunk scoring is skipped entirely rather than guessing from the per-frame
    estimate. Monitoring passes True, since by the time a baseline exists it
    either has a frozen sign or does not, and guessing is exactly what freezing
    exists to avoid. Calibration itself keeps the default False, because the
    per-frame estimate is where the votes that freeze a sign legitimately come
    from in the first place.
    """
    ear, shoulder, hip = _near_side(lm)

    cva = None
    if _visible(ear, shoulder):
        # SIGNED elevation of the ear above the shoulder, in (-90, 90]. Higher
        # is more upright the whole way down: positive is an ear above the
        # shoulder, zero is level with it, negative is a head collapsed forward
        # and DOWN onto the chest, which is the worst posture this measures.
        #
        # NOT abs(). Folding the sign made a dropped head report the same number
        # as a mild forward head, and worse, INVERTED the metric below the fold:
        # the further the head collapsed, the higher the score climbed. Nothing
        # looked wrong because upright postures sit where abs() is a no-op. This
        # is the same defect the trunk angle had, and the reason it is signed too.
        #
        # This is NOT the clinical craniovertebral angle, despite the name.
        # Clinical CVA is C7-to-tragus; this is MediaPipe shoulder-to-ear on an
        # uncalibrated 2D projection, so published cutoffs (the often quoted 50
        # degrees) do not transfer and none is applied here. Only movement
        # relative to the user's own baseline is meaningful, and only while the
        # camera stays put: moving the side camera shifts this number more than
        # weeks of real change do.
        cva = angle_from_horizontal(shoulder, ear)

    if facing_sign is not None:
        forward = facing_sign
    elif require_calibrated:
        forward = None
    else:
        forward = facing_sign_from(lm)

    trunk = None
    if _visible(shoulder, hip) and forward is not None:
        # SIGNED: positive means leaning forward, negative means reclining.
        # Scoring only penalises deviation above baseline, so reclining never
        # triggers a nudge. Conflating the two (via abs) would nag about using
        # a recliner, which for some users is deliberate and comfortable.
        #
        # The sign has to know which way the person faces, which depends on
        # which side the camera sits on (profiles mirror each other). It must
        # come from a vector that does NOT rotate away as they lean.
        #
        # The nose-to-ear axis IS the facing axis and is roughly horizontal, so
        # it only inverts past about 76 degrees of trunk rotation, far outside
        # any seated posture. An earlier version used ear-versus-shoulder, which
        # is roughly VERTICAL and therefore inverted at only about
        # (cva_deg - 90), around 28 degrees of recline, silently reporting a
        # deep recline as severe forward slouch: the exact opposite of reality,
        # and the precise case this feature exists to handle.
        #
        # A supplied facing_sign (from calibration) takes precedence over the
        # per-frame estimate, since a sustained head turn can invert the latter
        # while the trunk has not moved.
        trunk = forward * angle_from_vertical(hip, shoulder)

    return PostureMetrics(cva_deg=cva, trunk_angle_deg=trunk)


def merge_metrics(*parts: PostureMetrics) -> PostureMetrics:
    """Combine per-camera metrics. Later non-None values win; None never overwrites."""
    merged: dict[str, float | None] = {}
    for part in parts:
        for key, value in asdict(part).items():
            if value is not None:
                merged[key] = value
    return PostureMetrics(**merged)
