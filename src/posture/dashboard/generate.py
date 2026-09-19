"""Build the dashboard. Chart.js is inlined so the page works offline."""
from __future__ import annotations

import collections
import json
import logging
import webbrowser
from datetime import date, timedelta
from pathlib import Path

from posture.store.queries import (
    agreement_matrix,
    baseline_changed_between,
    comparison_sample_counts,
    cva_series,
    daily_scores,
    recent_checks,
    total_checks_recorded,
)

MIN_TREND_DAYS = 3  # below this a week's mean is noise, not a measurement

logger = logging.getLogger("posture")

TEMPLATE_PATH = Path(__file__).parent / "template.html"
VENDOR_PATH = Path(__file__).parent / "vendor" / "chart.umd.min.js"
LICENSES_PATH = Path(__file__).parent / "vendor" / "LICENSES.txt"


def _dominant_capability(scores: list[dict]) -> str | None:
    """The camera setup behind the most checks in the window.

    Every headline number is computed for ONE capability. A front-only score and
    a two-camera score are not the same measurement: only the two-camera one was
    checked against craniovertebral and trunk angle. Averaging them together
    makes a change of desk look like a change of back.
    """
    counts: collections.Counter = collections.Counter()
    for row in scores:
        counts[row["capability"]] += row["total"]
    return counts.most_common(1)[0][0] if counts else None


def _chronological(rows: list[dict]) -> list[dict]:
    """Oldest first, and within one date the older baseline first.

    daily_scores can now return two rows for one date when a recalibration
    happened mid-day, so "reverse the DESC query" is no longer a total order.
    baseline_id is monotonic (it is the baselines rowid), which makes it a
    usable tiebreak for which ruler came first. None sorts before any id: a
    check recorded before any calibration existed genuinely predates them.
    """
    return sorted(rows, key=lambda r: (
        r["date"], -1 if r["baseline_id"] is None else r["baseline_id"]))


def _mean_score_by_date(rows: list[dict]) -> dict[str, float]:
    """One score per date, weighted by how many checks produced it.

    Rows are per (date, baseline) now, so a plain dict comprehension would
    silently keep whichever one came last. This is used ONLY by the trend,
    which is suppressed outright whenever a recalibration falls inside its
    window, so the weighting never has to reconcile two rulers: any date
    carrying two baselines is a recalibration date, and baseline_changed_between
    sees it and blanks the trend before this number can reach the page.
    """
    totals: dict[str, float] = {}
    weights: dict[str, int] = {}
    for row in rows:
        totals[row["date"]] = totals.get(row["date"], 0.0) + row["score"] * row["total"]
        weights[row["date"]] = weights.get(row["date"], 0) + row["total"]
    return {day: totals[day] / weights[day] for day in totals}


def _baseline_segments(chrono: list[dict]) -> list[dict]:
    """Split the plotted history into runs that share one baseline.

    A recalibration rebases every score after it, so a line drawn straight
    across one joins two different rulers with a single stroke: the reader sees
    a rise or a fall that is an artefact of the reference changing, not of the
    back changing. The trend card already refuses that comparison; the chart
    sitting under it must not quietly make it anyway.

    Each run becomes its own dataset padded with nulls at every other x
    position, so Chart.js (with spanGaps off) leaves a visible gap at the
    boundary and gives each run its own colour and legend entry. The older
    history is still drawn, because it is still worth seeing; what is removed
    is only the claim that the two runs sit on one scale.
    """
    segments: list[dict] = []
    for index, row in enumerate(chrono):
        if not segments or segments[-1]["baseline_id"] != row["baseline_id"]:
            segments.append({
                "baseline_id": row["baseline_id"],
                "start_date": row["date"],
                "data": [None] * len(chrono),
            })
        segments[-1]["data"][index] = row["score"]
        segments[-1]["end_date"] = row["date"]
    # Legend text names each run by a date the chart can actually back. A
    # positional "Baseline 2" would read as the baselines table's id 2, which
    # it is not: the window may open part-way through a much later baseline.
    # The last run's start IS a real switch date (a run after the first only
    # begins where the baseline changed); an earlier run's start may predate
    # the window, so those are named by the day they ended instead.
    for index, segment in enumerate(segments):
        if len(segments) == 1:
            segment["label"] = "Score"
        elif index == len(segments) - 1:
            segment["label"] = f"Current baseline (from {segment['start_date']})"
        else:
            segment["label"] = f"Earlier baseline (until {segment['end_date']})"
    return segments


def _trend(by_date: dict[str, float], today: date) -> tuple[float | None, str | None]:
    """Mean of the last 7 CALENDAR days minus the 7 before, or None and a reason.

    Indexing rows rather than dates is the trap here. daily_scores returns one
    row per (date, capability), so slicing the newest 14 rows can silently span
    4 calendar days when the desk setup varies. v2 had a different version of
    the same bug: it compared the newest 7 against the OLDEST 7 of the last 30
    and called it a 7-day trend (spec M7).

    Returns None with a reason rather than a number whenever the window is too
    sparse to mean anything. A trend built from two days of data is noise
    wearing the costume of a measurement.
    """
    def window(start_offset: int) -> list[float]:
        days = [(today - timedelta(days=start_offset + i)).isoformat() for i in range(7)]
        return [by_date[d] for d in days if d in by_date]

    recent, previous = window(0), window(7)
    if len(recent) < MIN_TREND_DAYS or len(previous) < MIN_TREND_DAYS:
        return None, (
            f"Not enough history yet. A trend needs at least {MIN_TREND_DAYS} days "
            f"of checks in each week, and there are {len(recent)} and {len(previous)}."
        )
    return round(sum(recent) / len(recent) - sum(previous) / len(previous), 1), None


def build_dashboard_data(conn) -> dict:
    scores = daily_scores(conn, days=60)
    capability = _dominant_capability(scores)
    # Everything below is for this capability only. Rows from other desk setups
    # are counted and reported, never silently folded in.
    kept = [row for row in scores if row["capability"] == capability]
    excluded = sum(row["total"] for row in scores if row["capability"] != capability)

    chrono = _chronological(kept)
    segments = _baseline_segments(chrono)
    # The first date of every run after the first: where the ruler changed.
    breaks = [segment["start_date"] for segment in segments[1:]]

    # The ruler the most recent scores were measured against. Derived from the
    # plotted rows rather than from the baselines table so that every label
    # below is true of the number it sits on: if the newest check on this setup
    # used baseline 4, "current baseline" means baseline 4, full stop.
    current_baseline_id = chrono[-1]["baseline_id"] if chrono else None
    on_current = [row for row in chrono if row["baseline_id"] == current_baseline_id]

    by_date = _mean_score_by_date(kept)
    # Today's card reports one ruler. On a day split by a recalibration the
    # honest answer is the score against the baseline in force now, not the
    # blend of before and after.
    current_by_date = {row["date"]: row["score"] for row in on_current}

    today = date.today()
    today_score = current_by_date.get(today.isoformat())
    # today_score can be None for two different reasons: no checks happened
    # today at all, or they happened on a setup other than the dominant one.
    # Those are not the same fact, and only the second one is explained
    # anywhere else on the page (the capability line's excluded-checks
    # sentence), so the card needs to be able to tell them apart too.
    today_checked_on_other_setup = today_score is None and any(
        row["date"] == today.isoformat() and row["capability"] != capability
        for row in scores
    )

    trend, trend_note = _trend(by_date, today)

    # A recalibration rebases every score after it, so a trend spanning one is
    # comparing two different rulers. This matters most in exactly the situation
    # the app is for: recalibrating after a change in the back itself.
    changed_on = baseline_changed_between(
        conn, (today - timedelta(days=13)).isoformat(), today.isoformat())
    if trend is not None and changed_on:
        trend, trend_note = None, (
            f"Baseline was recalibrated on {changed_on}, so scores before and "
            "after it are measured against different references. The trend will "
            "return once there are two full weeks on the current baseline."
        )

    # The average is scoped to the current baseline, and the label says so.
    # Averaging across a recalibration produces a number measured against no
    # reference that exists, under a label ("last 60 days") that quietly
    # promises otherwise. Scoping keeps the number meaningful and keeps the
    # older history visible on the chart, where it belongs.
    ordered = [row["score"] for row in on_current]
    if breaks:
        average_label = f"Average score (current baseline, since {breaks[-1]})"
        average_note = (
            f"Checks from before the recalibration on {breaks[-1]} were scored "
            "against a different reference, so they are not in this average. "
            "They are still drawn on the chart above, as a separate line."
        )
    else:
        average_label = "Average score (last 60 days)"
        average_note = None

    matrix = agreement_matrix(conn)
    matched = sum(count for (local, gem), count in matrix.items() if local == gem)
    # The matrix only ever holds sampled checks that produced BOTH ratings, so
    # its total is not the size of the comparison sample. Taking the denominator
    # from the matrix made the rate describe the calls that succeeded and
    # nothing else, with the failures invisible. comparison_sample_counts
    # carries both figures and the reasons for the difference.
    samples = comparison_sample_counts(conn)

    scored = sum(row["total"] for row in scores)
    recorded = total_checks_recorded(conn, days=60)

    return {
        "generated_at": today.isoformat(),
        "capability": capability,
        "excluded_checks": excluded,
        "today_score": today_score,
        "today_checked_on_other_setup": today_checked_on_other_setup,
        "average_score": round(sum(ordered) / len(ordered), 1) if ordered else None,
        "average_score_label": average_label,
        "average_score_note": average_note,
        "total_checks": sum(row["total"] for row in kept),
        # Every OTHER capability's SCORED checks too, in the same 60-day
        # window. Kept separate from total_checks (dominant capability only)
        # so the template never has to guess which one a label is allowed to
        # claim. "Scored" because this still only counts what daily_scores
        # sees: OK checks with a rating.
        "total_checks_scored": scored,
        # The genuine count of every check in the window, any outcome. This
        # is what keeps the headline honest against the Recent checks table,
        # which is unfiltered: a PERSON_ABSENT or API_ERROR row visible there
        # would otherwise be invisible to every number on this page.
        "total_checks_recorded": recorded,
        "total_checks_unscored": recorded - scored,
        "trend": trend,
        "trend_note": trend_note,
        "daily": chrono,
        # One x label per plotted row, so a date split by a recalibration keeps
        # both of its points instead of one overwriting the other.
        "daily_labels": [row["date"] for row in chrono],
        "daily_segments": segments,
        "baseline_breaks": breaks,
        "cva": list(reversed(cva_series(conn, days=60))),
        "recent": recent_checks(conn, limit=50),
        "agreement": {
            # Every check the roll selected. This is the number a reader means
            # by "how many were compared", and the one that was missing.
            "sampled": samples["sampled"],
            # Of those, the ones that produced both ratings. The rate's real
            # denominator, named so the card can label it instead of implying
            # it covers the whole sample.
            "compared": samples["compared"],
            "matched": matched,
            "rate": (matched / samples["compared"]) if samples["compared"] else None,
            "no_verdict": samples["no_verdict"],
            "no_verdict_reasons": samples["no_verdict_reasons"],
            "matrix": [{"local": k[0], "gemini": k[1], "count": v} for k, v in matrix.items()],
        },
    }


def generate(conn, output_path: Path) -> Path:
    data = build_dashboard_data(conn)
    return render_dashboard(data, output_path)


def render_dashboard(data: dict, output_path: Path) -> Path:
    """Render supplied measurements; the demo supplies wholly invented data."""
    html = TEMPLATE_PATH.read_text(encoding="utf-8")
    html = html.replace("/*__CHARTJS__*/", VENDOR_PATH.read_text(encoding="utf-8"))
    # Escape angle brackets and ampersands as \uXXXX so no HTML-looking substring
    # survives in the payload. Still valid JSON, and the template additionally
    # renders every string through textContent rather than innerHTML.
    payload = (json.dumps(data)
               .replace("<", "\\u003c")
               .replace(">", "\\u003e")
               .replace("&", "\\u0026"))
    html = html.replace("__DATA_JSON__", payload)
    # Retain the permission notices when an HTML dashboard is shared on its own.
    html = html.replace("</body>", "<!--\n" + LICENSES_PATH.read_text(encoding="utf-8")
                        + "\n-->\n</body>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("dashboard written to %s", output_path)
    return output_path


def open_dashboard(conn, output_path: Path) -> None:
    webbrowser.open(f"file://{generate(conn, output_path)}")
