#!/usr/bin/env python3
"""Send one identical frame to Gemini N times and report the spread of ratings.

Usage:
    uv run python scripts/repeatability.py tests/fixtures/forward_head_mild_01.jpg
    uv run python scripts/repeatability.py <image> --runs 10

Costs real money (roughly a cent for the default 10 runs). This informs whether
Gemini is repeatable enough to be the instrument behind the trend chart, or
whether it belongs on the coaching side only.

Scope honestly: N runs of ONE frame measures run-to-run variance at a single
point, not repeatability across the range. Treat a clean result as a reason to
test more borderline frames, never as a finished answer.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from posture.config import Config, DATA_DIR  # noqa: E402
from posture.crop import CropUnavailable, crop_to_person, encode_jpeg  # noqa: E402
from posture.gemini import GeminiClient  # noqa: E402
from posture.landmarks import PoseDetector  # noqa: E402


def summarise(results: list[dict], model: str) -> tuple[int, list[str]]:
    """Turn collected runs into an exit code and the lines to print.

    Pure and network free on purpose. The interesting cases here are the failure
    ones, and they are unreachable in a test if this logic lives inline in main().
    """
    lines: list[str] = []
    ratings = [r["rating"] for r in results if "rating" in r]
    counts = collections.Counter(ratings)
    issues = {r["issue"].strip().lower() for r in results if "issue" in r}

    lines.append("=" * 60)
    if not ratings:
        # Every run failed, so there is no spread to report. This is the FIRST
        # thing that happens if the configured model ID is wrong, which is the
        # single most likely failure on a fresh key. Say so instead of dividing
        # by zero or indexing an empty Counter.
        errors = collections.Counter(r["error"] for r in results if "error" in r)
        lines.append(f"All {len(results)} runs failed: {dict(errors)}")
        if "MODEL_UNAVAILABLE" in errors:
            lines.append(f"\n{model} is not a live model. Update "
                          "gemini_model in src/posture/config.py and run this again.")
        lines.append("\nNo repeatability conclusion is possible from zero successful runs.")
        return 1, lines

    lines.append(f"Ratings across {len(ratings)} successful runs: {dict(counts)}")
    lines.append(f"Distinct issue phrasings: {len(issues)}")
    agreement = counts.most_common(1)[0][1] / len(ratings)
    if len(counts) == 1:
        lines.append(f"\nVERDICT: identical on all {len(ratings)} runs of this ONE frame.")
        lines.append("That is suggestive, not sufficient. A single unambiguous frame is")
        lines.append("repeatable for almost any rater. Before trusting Gemini as the trend")
        lines.append("instrument, run this again on several borderline frames: the boundary")
        lines.append("is where a wandering rater actually shows up.")
    else:
        lines.append(f"\nVERDICT: NOT repeatable. Modal rating appeared "
                      f"{agreement:.0%} of the time.")
        lines.append("Confirms that the trend should be driven by the local measurement.")
    return 0, lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--role", default="front", choices=["front", "side"])
    args = parser.parse_args()

    if not args.image.exists():
        print(f"No such image: {args.image}")
        return 1

    config = Config.load()
    frame = cv2.imread(str(args.image))
    if frame is None:
        print(f"Could not decode {args.image}")
        return 1

    with PoseDetector(config.model_path) as detector:
        landmarks = detector.detect(frame)
    if landmarks is None:
        print("No person detected in that image, so there is nothing to compare.")
        return 1

    try:
        jpeg = encode_jpeg(crop_to_person(frame, landmarks))
    except CropUnavailable as exc:
        # Task 13 fails closed rather than handing back the uncropped frame.
        # Report it the same way as the no-person case instead of dying on an
        # unhandled traceback.
        print(f"Could not isolate a person region in that image: {exc}")
        return 1
    client = GeminiClient(config)
    if not client.available:
        print("No API key configured.")
        return 1

    print(f"Sending the SAME cropped frame {args.runs} times to "
          f"{config.gemini_model}...\n")

    results = []
    for run_index in range(1, args.runs + 1):
        verdict, outcome = client.assess({args.role: jpeg})
        if outcome is not None:
            print(f"  run {run_index:2d}: FAILED ({outcome.value})")
            results.append({"run": run_index, "error": outcome.value})
            continue
        rating = verdict.rating.value if verdict.rating else "NO_PERSON"
        print(f"  run {run_index:2d}: {rating:7s} | {verdict.issue}")
        results.append({"run": run_index, "rating": rating, "issue": verdict.issue,
                        "details": verdict.details, "tip": verdict.tip})

    print()
    exit_code, lines = summarise(results, config.gemini_model)
    for line in lines:
        print(line)
    if exit_code != 0:
        return exit_code

    counts = collections.Counter(r["rating"] for r in results if "rating" in r)
    out_path = DATA_DIR / "repeatability.json"
    out_path.write_text(json.dumps(
        {"image": str(args.image), "model": config.gemini_model,
         "runs": results, "counts": dict(counts)}, indent=2))
    print(f"\nFull results: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
