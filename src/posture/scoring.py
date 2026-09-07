"""Metrics plus baseline to a rating. Pure and deterministic.

This is the half of the system Gemini cannot replace: the same input always
produces the same output, so a trend line plots the back rather than model
variance.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from posture.types import PostureMetrics, Rating


@dataclass(frozen=True)
class MetricRule:
    """How far a metric may drift from baseline before it counts against you.

    direction: -1 means lower than baseline is worse (craniovertebral angle),
               +1 means higher is worse (trunk lean),
                0 means any deviation is worse (shoulder tilt).
    """
    direction: int
    decent_at: float
    poor_at: float


# Thresholds in the metric's own units (degrees, or a ratio). Starting values,
# expected to be tuned once a week of real data exists.
METRIC_RULES: dict[str, MetricRule] = {
    "cva_deg": MetricRule(direction=-1, decent_at=4.0, poor_at=10.0),
    "trunk_angle_deg": MetricRule(direction=+1, decent_at=8.0, poor_at=18.0),
    "shoulder_tilt_deg": MetricRule(direction=0, decent_at=5.0, poor_at=10.0),
    "head_tilt_deg": MetricRule(direction=0, decent_at=8.0, poor_at=15.0),
    "head_lateral_ratio": MetricRule(direction=0, decent_at=0.10, poor_at=0.20),
    "proximity_ratio": MetricRule(direction=+1, decent_at=0.06, poor_at=0.12),
}

_SEVERITY = {None: 0, Rating.GOOD: 1, Rating.DECENT: 2, Rating.POOR: 3}


@dataclass(frozen=True)
class Baseline:
    id: int
    created_at: str
    label: str
    metrics: PostureMetrics
    # Which way is forward, frozen at calibration time under controlled conditions.
    # A per-frame estimate can invert during a head turn, because one 2D view
    # cannot separate head yaw from trunk lean. None means it was not captured.
    facing_sign: float | None = None


@dataclass(frozen=True)
class LocalAssessment:
    rating: Rating | None
    driver: str | None
    deviations: dict[str, float] = field(default_factory=dict)


def _rate_one(current: float, baseline: float, rule: MetricRule) -> Rating:
    if rule.direction == 0:
        magnitude = abs(current - baseline)
    elif rule.direction < 0:
        # Only drops below baseline count. No neutral floor here, unlike
        # direction +1: cva_deg is signed but its sign is not a direction, it is
        # just the metric continuing past horizontal, and lower is worse the
        # whole way down. A collapsed head at -60 against a baseline of 60 is a
        # 120 degree drop, which is exactly what it should be.
        magnitude = max(0.0, baseline - current)
    else:
        # Only movement past NEUTRAL counts, not movement away from a negative
        # baseline. With a reclined baseline of -20, penalising any increase
        # would score sitting upright as POOR, inverting the point of signing
        # the trunk angle at all.
        floor = max(baseline, 0.0)
        magnitude = max(0.0, current - floor)

    if magnitude >= rule.poor_at:
        return Rating.POOR
    if magnitude >= rule.decent_at:
        return Rating.DECENT
    return Rating.GOOD


def score(metrics: PostureMetrics, baseline: Baseline) -> LocalAssessment:
    """Rate current metrics against the baseline. Worst comparable metric wins.

    Returns rating None when nothing is comparable, so that "we could not
    measure" is never silently reported as "you are sitting well".
    """
    current = asdict(metrics)
    base = asdict(baseline.metrics)

    deviations: dict[str, float] = {}
    worst: Rating | None = None
    driver: str | None = None

    for name, rule in METRIC_RULES.items():
        now, then = current.get(name), base.get(name)
        if now is None or then is None:
            continue
        deviation = now - then
        deviations[name] = deviation
        rating = _rate_one(now, then, rule)
        if _SEVERITY[rating] > _SEVERITY[worst]:
            worst, driver = rating, name

    if not deviations:
        return LocalAssessment(rating=None, driver=None, deviations={})
    return LocalAssessment(
        rating=worst,
        driver=driver if worst in (Rating.DECENT, Rating.POOR) else None,
        deviations=deviations,
    )
