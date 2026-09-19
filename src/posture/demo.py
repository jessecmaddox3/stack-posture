"""Deterministic synthetic demo. No camera, credentials, personal state or network."""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

from posture.scoring import Baseline, score
from posture.types import PostureMetrics


def dashboard_data() -> dict:
    baseline = Baseline(1, "2026-01-01T09:00:00", "Invented example", PostureMetrics(cva_deg=62))
    labels, values, cva, recent = [], [], [], []
    for day in range(14):
        label = (date(2026, 1, 1) + timedelta(days=day)).isoformat()
        angle = [51, 53, 58, 55, 54, 59, 60, 58, 59, 61, 58, 60, 61, 60][day]
        result = score(PostureMetrics(cva_deg=angle), baseline)
        labels.append(label)
        values.append({"GOOD": 100, "DECENT": 60, "POOR": 20}[result.rating.value])
        cva.append({"date": label, "cva": angle})
        recent.append({"ts": label + "T10:00:00", "outcome": "OK", "cameras": '["side"]',
                       "capability": "side", "local_rating": result.rating.value,
                       "local_driver": result.driver, "gemini_rating": None,
                       "gemini_issue": None, "gemini_details": None, "gemini_tip": None,
                       "is_comparison_sample": False, "api_error": None})
    return {
        "generated_at": "2026-01-14 (synthetic example)", "capability": "side",
        "excluded_checks": 0, "today_score": values[-1],
        "today_checked_on_other_setup": False,
        "average_score": round(sum(values) / len(values), 1),
        "average_score_label": "Average score (invented history)",
        "average_score_note": "All measurements in this preview are invented.",
        "total_checks": 14, "total_checks_scored": 14, "total_checks_recorded": 14,
        "total_checks_unscored": 0,
        "trend": round(sum(values[7:]) / 7 - sum(values[:7]) / 7, 1), "trend_note": None,
        "daily": [], "daily_labels": labels,
        "daily_segments": [{"baseline_id": 1, "label": "Invented example", "data": values}],
        "baseline_breaks": [], "cva": cva, "recent": list(reversed(recent)),
        "agreement": {"sampled": 0, "compared": 0, "matched": 0, "rate": None,
                      "no_verdict": 0, "no_verdict_reasons": {}, "matrix": []},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard", type=Path, help="Write an offline HTML preview here")
    args = parser.parse_args()
    baseline = Baseline(1, "2026-01-01T09:00:00Z", "Synthetic upright baseline",
                        PostureMetrics(cva_deg=62.0, trunk_angle_deg=0.0))
    current = PostureMetrics(cva_deg=49.0, trunk_angle_deg=20.0)
    assessment = score(current, baseline)
    print("Stack offline demo")
    print(f"Baseline: {baseline.metrics.available()}")
    print(f"Current:  {current.available()}")
    print(f"Rating:   {assessment.rating.value if assessment.rating else 'unavailable'}")
    print(f"Driver:   {assessment.driver or 'none'}")
    print("Synthetic measurements only. No camera, Keychain, personal state or network.")
    if args.dashboard:
        from posture.dashboard.generate import render_dashboard
        render_dashboard(dashboard_data(), args.dashboard)
        # Clear top-of-page label, using the same actual dashboard UI below it.
        html = args.dashboard.read_text(encoding="utf-8")
        html = html.replace("const DATA =", "Chart.defaults.animation = false;\nconst DATA =")
        html = html.replace('<body>', '<body><aside style="padding:16px;text-align:center;'
                            'background:#fff4d9;color:#302a20;font:16px system-ui">'
                            '<strong>Synthetic demo.</strong> Every measurement is invented. '
                            'This page does not use your camera or upload anything.</aside>')
        args.dashboard.write_text(html, encoding="utf-8")
        print(f"Open this file in a browser: {args.dashboard}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
