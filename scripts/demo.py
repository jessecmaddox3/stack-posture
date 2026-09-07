#!/usr/bin/env python3
"""Run an offline, synthetic scoring demo without a camera or account."""
from __future__ import annotations

from posture.scoring import Baseline, score
from posture.types import PostureMetrics


def main() -> int:
    baseline = Baseline(
        id=1,
        created_at="2026-01-01T09:00:00Z",
        label="synthetic upright baseline",
        metrics=PostureMetrics(cva_deg=62.0, trunk_angle_deg=0.0),
    )
    current = PostureMetrics(cva_deg=49.0, trunk_angle_deg=20.0)
    assessment = score(current, baseline)
    print("Stack offline demo")
    print(f"Baseline: {baseline.metrics.available()}")
    print(f"Current:  {current.available()}")
    print(f"Rating:   {assessment.rating.value if assessment.rating else 'unavailable'}")
    print(f"Driver:   {assessment.driver or 'none'}")
    print("This uses synthetic measurements only. No camera, Keychain, files, or network.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
