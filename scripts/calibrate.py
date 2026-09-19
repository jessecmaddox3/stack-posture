#!/usr/bin/env python3
"""Record a posture baseline.

Usage:  uv run python scripts/calibrate.py "new desk position"

Sit the way your physical therapist wants you to sit, then hold still for ten
seconds. Nothing is uploaded and no image is written.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from posture.baseline import run_calibration  # noqa: E402
from posture.config import Config  # noqa: E402
from posture.landmarks import PoseDetector  # noqa: E402
from posture.store.db import connect, migrate  # noqa: E402


def main() -> int:
    label = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    config = Config.load()

    print("Sit the way you want to be sitting. Hold still for 10 seconds.")
    for count in (3, 2, 1):
        print(f"  {count}...")
        time.sleep(1)
    print("Recording...")

    conn = connect(config.db_path)
    migrate(conn)
    try:
        with PoseDetector(config.model_path) as detector:
            baseline = run_calibration(config, detector, conn, label=label)
    finally:
        conn.close()

    if baseline is None:
        print("\nNot enough usable samples. Check your camera framing and lighting.")
        return 1

    metrics = baseline.metrics
    facing = baseline.facing_sign

    print(f"\nBaseline #{baseline.id} saved as '{label}':")
    for name, value in metrics.available().items():
        print(f"  {name:22} {value:.2f}")

    if metrics.cva_deg is None:
        print("\nNote: no craniovertebral angle recorded. Either no side camera was")
        print("connected, or it could not see your ear and shoulder clearly enough.")
        print("Front-only scoring cannot see forward head posture at all.")
    if facing is None:
        print("\nNote: facing direction not captured. Either no side camera was")
        print("connected, it could not see your ear and nose clearly enough, or its")
        print("votes were too inconsistent to trust (not enough usable frames, or")
        print("no clear majority on which way you face).")
        print("Trunk angle will be omitted, not estimated, until this is resolved.")
        print("Re-run with the side camera well framed and hold still to fix.")
    else:
        print(f"\nFacing direction frozen at {facing:+.0f} (from the side camera).")
    return 0


if __name__ == "__main__":
    from posture.instance import instance_lock
    with instance_lock():
        raise SystemExit(main())
