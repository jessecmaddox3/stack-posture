"""Record what good posture looks like for the user.

Scoring is deviation from this baseline rather than a generic ideal, so the
number stays meaningful as the user's achievable range changes over time.
Baselines are versioned; every check records which one scored it.
"""
from __future__ import annotations

import logging
import statistics
import time
from dataclasses import asdict, dataclass, fields, replace

from posture.config import Config
from posture.metrics import (
    compute_front_metrics, compute_side_metrics, facing_sign_from, merge_metrics,
)
from posture.scoring import Baseline
from posture.types import PostureMetrics

logger = logging.getLogger("posture")


def median_metrics(samples: list[PostureMetrics]) -> PostureMetrics:
    """Per-field median across samples, skipping None.

    Median rather than mean: one frame where the user shifts, or one detector
    glitch, would drag a mean and skew every future score.
    """
    if not samples:
        raise ValueError("no samples to build a baseline from")
    result: dict[str, float] = {}
    for field_def in fields(PostureMetrics):
        values = [
            v for v in (asdict(s).get(field_def.name) for s in samples) if v is not None
        ]
        if values:
            result[field_def.name] = statistics.median(values)
    return PostureMetrics(**result)


MIN_FACING_VOTES = 5
FACING_AGREEMENT = 0.8


@dataclass(frozen=True)
class CalibrationSample:
    metrics: PostureMetrics
    facing_sign: float | None


def majority_facing_sign(samples: list["CalibrationSample"]) -> float | None:
    """Majority vote over the calibration window, or None if never determined.

    Facing is a property of the camera position, not of the frame, so it is
    decided once here under controlled conditions (upright, looking at the
    screen) and then frozen. A per-frame estimate can invert during a head turn.

    Fails closed: fewer than MIN_FACING_VOTES usable votes, or a winning side
    that does not hold at least FACING_AGREEMENT of them, yields None rather
    than a guess. A weak or tied vote previously became +1, which on a -1 rig
    would invert every future trunk reading with nothing looking wrong.
    """
    votes = [s.facing_sign for s in samples if s.facing_sign is not None]
    if len(votes) < MIN_FACING_VOTES:
        return None
    positive = sum(1 for v in votes if v > 0)
    negative = len(votes) - positive
    if positive >= len(votes) * FACING_AGREEMENT:
        return 1.0
    if negative >= len(votes) * FACING_AGREEMENT:
        return -1.0
    return None


def run_calibration(config: Config, detector, conn, label: str,
                    seconds: float = 10, interval_s: float = 0.5) -> Baseline | None:
    """Collect, aggregate, and persist a baseline. None if too few usable samples.

    Deliberately separated from calibrate.py's main(), which only wires real
    dependencies. Keeping the orchestration here means it can be tested with a
    stub detector, a monkeypatched capture_roles, and an in-memory database, with
    no camera and no model. That matters because this function is where the frozen
    facing sign reaches storage, and a silent failure there would send every later
    trunk measurement back to the unreliable per-frame estimate.
    """
    from posture.store.queries import save_baseline

    samples = collect_baseline_samples(config, detector, seconds=seconds,
                                       interval_s=interval_s)
    if len(samples) < 5:
        return None
    metrics = median_metrics([s.metrics for s in samples])
    facing = majority_facing_sign(samples)
    if facing is None and metrics.trunk_angle_deg is not None:
        # Without a trusted sign we have promised not to use trunk angle at all:
        # Monitor passes require_calibrated=True so live readings are None, and
        # score() skips any metric where either side is None. Keeping a value
        # here would print a figure directly above a note saying it is omitted,
        # and persist a reference that nothing can ever be compared against.
        metrics = replace(metrics, trunk_angle_deg=None)
    return save_baseline(conn, metrics, label=label, facing_sign=facing)


def collect_baseline_samples(
    config: Config, detector, seconds: float = 10, interval_s: float = 0.5
) -> list[CalibrationSample]:
    """Sample metrics repeatedly while the user holds their target posture."""
    from posture import cameras

    samples: list[CalibrationSample] = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        frames, outcome = cameras.capture_roles(config)
        if outcome is not None or not frames:
            time.sleep(interval_s)
            continue
        parts, facing = [], None
        for role, frame in frames.items():
            lm = detector.detect(frame)
            if lm is None:
                continue
            if role == "front":
                parts.append(compute_front_metrics(lm))
            else:
                facing = facing_sign_from(lm)
                parts.append(compute_side_metrics(lm, facing_sign=facing))
        if parts:
            samples.append(CalibrationSample(merge_metrics(*parts), facing))
        time.sleep(interval_s)
    return samples
