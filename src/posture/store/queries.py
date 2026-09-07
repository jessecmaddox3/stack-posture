"""Reads and writes. Daily figures are derived from checks, never stored twice.

v2 kept a daily_summaries table alongside checks, giving two sources of truth
that could disagree (spec M10).
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import date, datetime

from posture.scoring import Baseline
from posture.types import Outcome, PostureMetrics, Rating


@dataclass(frozen=True)
class CheckRecord:
    ts: str
    date: str
    outcome: Outcome
    baseline_id: int | None
    cameras: list[str]
    metrics: PostureMetrics | None
    local_rating: Rating | None
    local_driver: str | None
    gemini_rating: Rating | None
    gemini_issue: str | None
    gemini_details: str | None
    gemini_tip: str | None
    gemini_model: str | None
    is_comparison_sample: bool
    api_error: str | None


def _enum_value(value) -> str | None:
    return value.value if value is not None else None


# Which PostureMetrics fields each camera role can produce. This mirrors
# metrics.compute_front_metrics and metrics.compute_side_metrics; the two are
# pinned together by test_store.py's capability-metric-map test, so adding a
# metric there without adding it here fails loudly rather than quietly making
# capability under-report.
_ROLE_METRICS: dict[str, tuple[str, ...]] = {
    "front": ("shoulder_tilt_deg", "head_tilt_deg", "head_lateral_ratio",
              "proximity_ratio"),
    "side": ("cva_deg", "trunk_angle_deg"),
}


def _capability(metrics: PostureMetrics | None) -> str | None:
    """Which camera roles this row actually carries a MEASUREMENT from.

    Not which cameras were configured, not which returned a frame, and not
    which detected a person: which produced a number that is in this row.
    Those come apart in a case that happens for real. A side camera can see
    the user perfectly well and still yield neither a craniovertebral angle (ear
    or shoulder under metrics.MIN_VISIBILITY) nor a trunk angle (calibration
    never froze a facing sign, and _sample passes require_calibrated=True so
    monitoring refuses to guess). A row like that holds exactly as much
    side-derived information as a row taken with no side camera at all.

    This column exists so a two-camera trend is never polluted by front-only
    measurements: daily_scores GROUPS BY it and every headline number on the
    dashboard is filtered on it. Stamping "front+side" on a row whose cva_deg
    and trunk_angle_deg are both NULL defeats that, and defeats it silently, in
    the one direction nothing else on the page can catch. The row still records
    the cameras that returned a frame, in the cameras column, so the two can be
    compared when a setup misbehaves.

    Deriving it at write time (rather than out of the metrics JSON on every
    query) keeps the grouping key a plain column that SQL can GROUP BY.
    """
    if metrics is None:
        return None
    available = metrics.available()
    roles = [role for role, names in _ROLE_METRICS.items()
             if any(name in available for name in names)]
    return "+".join(sorted(roles)) if roles else None


def insert_check(conn: sqlite3.Connection, record: CheckRecord) -> int:
    cursor = conn.execute(
        """INSERT INTO checks
           (ts, date, outcome, baseline_id, cameras, metrics, local_rating,
            local_driver, gemini_rating, gemini_issue, gemini_details, gemini_tip,
            gemini_model, is_comparison_sample, api_error, capability)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            record.ts, record.date, record.outcome.value, record.baseline_id,
            json.dumps(record.cameras),
            json.dumps(asdict(record.metrics)) if record.metrics else None,
            _enum_value(record.local_rating), record.local_driver,
            _enum_value(record.gemini_rating), record.gemini_issue,
            record.gemini_details, record.gemini_tip, record.gemini_model,
            int(record.is_comparison_sample), record.api_error,
            _capability(record.metrics),
        ),
    )
    conn.commit()
    return cursor.lastrowid


def save_baseline(conn: sqlite3.Connection, metrics: PostureMetrics, label: str,
                  facing_sign: float | None = None) -> Baseline:
    """Store a new baseline and deactivate every previous one.

    facing_sign is captured once during calibration and frozen, because a
    per-frame estimate can invert during a head turn.
    """
    created = datetime.now().isoformat(timespec="seconds")
    conn.execute("UPDATE baselines SET active = 0")
    cursor = conn.execute(
        "INSERT INTO baselines (created_at, label, metrics, facing_sign, active) "
        "VALUES (?,?,?,?,1)",
        (created, label, json.dumps(asdict(metrics)), facing_sign),
    )
    conn.commit()
    return Baseline(id=cursor.lastrowid, created_at=created, label=label,
                    metrics=metrics, facing_sign=facing_sign)


def active_baseline(conn: sqlite3.Connection) -> Baseline | None:
    row = conn.execute(
        "SELECT id, created_at, label, metrics, facing_sign FROM baselines "
        "WHERE active = 1 ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    return Baseline(id=row["id"], created_at=row["created_at"], label=row["label"],
                    metrics=PostureMetrics(**json.loads(row["metrics"])),
                    facing_sign=row["facing_sign"])


def daily_scores(conn: sqlite3.Connection, days: int = 30) -> list[dict]:
    """Average local score per day, stratified by camera setup AND baseline.

    A GOOD with both cameras and a GOOD with only the front camera do not mean
    the same thing: only the two-camera row was checked against craniovertebral
    and trunk angle. Grouping by capability as well as date keeps a trend from
    silently changing meaning when the desk setup changes, rather than the back.
    Only OK checks with a rating count.

    baseline_id is in the grouping for the same reason. A recalibration rebases
    every score after it, so two checks on one day either side of one are
    measured against different references; averaging them yields a number that
    is true of neither ruler. With baseline_id in the GROUP BY a single point
    can never mix two baselines, which is what lets the dashboard draw the
    history as separate runs rather than one continuous, uninterpretable line.
    """
    rows = conn.execute(
        """SELECT date, capability, baseline_id,
                  ROUND(AVG(CASE local_rating
                       WHEN 'GOOD' THEN 100.0 WHEN 'DECENT' THEN 60.0
                       WHEN 'POOR' THEN 20.0 END), 1) AS score,
                  COUNT(*) AS total,
                  SUM(local_rating = 'GOOD')   AS good,
                  SUM(local_rating = 'DECENT') AS decent,
                  SUM(local_rating = 'POOR')   AS poor
           FROM checks
           WHERE outcome = 'OK' AND local_rating IS NOT NULL
             AND date >= DATE('now', 'localtime', ?)
           GROUP BY date, capability, baseline_id
           ORDER BY date DESC, capability, baseline_id""",
        # DATE(...) truncates to midnight and >= is inclusive of that day, so
        # '-N days' spans N+1 calendar days (today plus N days back). Using
        # N-1 here makes days=60 mean exactly 60 calendar days, not 61.
        (f'-{int(days) - 1} days',),
    ).fetchall()
    return [dict(r) for r in rows]


def total_checks_recorded(conn: sqlite3.Connection, days: int = 60) -> int:
    """Every check in the window, regardless of outcome or rating.

    daily_scores only ever sees OK checks that produced a rating, so a
    PERSON_ABSENT, CAMERA_BUSY, CAMERA_UNAVAILABLE, or API_ERROR row is
    invisible to it even though it is a real event. That same row still shows
    up in recent_checks, which is unfiltered, so a headline total built only
    from summing daily_scores would disagree with what the Recent checks
    table on the page visibly shows.
    """
    row = conn.execute(
        """SELECT COUNT(*) AS n FROM checks
           WHERE date >= DATE('now', 'localtime', ?)""",
        (f'-{int(days) - 1} days',),
    ).fetchone()
    return row["n"] or 0


def cva_series(conn: sqlite3.Connection, days: int = 30) -> list[dict]:
    """Mean daily craniovertebral angle over the last `days` CALENDAR days.

    Days with no side-camera reading are simply absent, so the chart shows a
    gap rather than interpolating across a camera that was not connected.

    Bounded by date, not by row count. LIMIT applies to ROWS, so a sparse
    history (one reading every few days) stretched the series far past the
    window: a reviewer reproduced 60 rows spanning 296 calendar days under a
    heading that says "last 60 days". That is the same bug already fixed in
    daily_scores, and the date bound here is deliberately identical to it.

    AVG, not a median: the name says what the SQL does.
    """
    rows = conn.execute(
        """SELECT date, ROUND(AVG(json_extract(metrics, '$.cva_deg')), 1) AS cva
           FROM checks
           WHERE outcome = 'OK' AND json_extract(metrics, '$.cva_deg') IS NOT NULL
             AND date >= DATE('now', 'localtime', ?)
           GROUP BY date ORDER BY date DESC""",
        # Same off-by-one reasoning as daily_scores: DATE() truncates to
        # midnight and >= includes that day, so '-N days' would span N+1.
        (f'-{int(days) - 1} days',),
    ).fetchall()
    return [dict(r) for r in rows]


def baseline_changed_between(conn: sqlite3.Connection, start: str, end: str) -> str | None:
    """First date in the range whose checks use a different baseline than the
    range's earliest, or None if one baseline covers the whole span.

    A recalibration rebases every score after it. Comparing across one is
    comparing two different rulers, which matters most in precisely the case
    this app exists for: recalibrating after the back itself has changed.
    """
    rows = conn.execute(
        """SELECT date, baseline_id FROM checks
           WHERE date BETWEEN ? AND ? AND baseline_id IS NOT NULL
           GROUP BY date, baseline_id ORDER BY date""",
        (start, end),
    ).fetchall()
    if not rows:
        return None
    first = rows[0]["baseline_id"]
    for row in rows:
        if row["baseline_id"] != first:
            return row["date"]
    return None


def agreement_matrix(conn: sqlite3.Connection) -> dict[tuple[str, str], int]:
    """Local versus Gemini counts over the random comparison sample ONLY."""
    rows = conn.execute(
        """SELECT local_rating, gemini_rating, COUNT(*) AS n
           FROM checks
           WHERE is_comparison_sample = 1 AND outcome = 'OK'
             AND local_rating IS NOT NULL AND gemini_rating IS NOT NULL
           GROUP BY local_rating, gemini_rating"""
    ).fetchall()
    return {(r["local_rating"], r["gemini_rating"]): r["n"] for r in rows}


def comparison_sample_counts(conn: sqlite3.Connection) -> dict:
    """Every check the roll put in the comparison set, split by what became of it.

    agreement_matrix answers a CONDITIONAL question: of the sampled checks that
    produced both a local rating and a Gemini verdict, how often did the two
    match. Reading its total as the size of the comparison sample is a selection
    effect, not a rounding error. A sampled check that came back API_ERROR or
    MODEL_UNAVAILABLE, or that the crop refused locally (CROP_UNAVAILABLE),
    simply is not in the matrix, so the rate describes only the calls that
    worked. Reproduced by a reviewer: twenty checks sampled, the page saying
    "4 of 5", with nothing on it hinting at the other fifteen.

    That gap is not a random subset either. A crop refusal tracks poor
    detection, which tracks the extreme postures the comparison exists to
    check, so the silently dropped rows are disproportionately the interesting
    ones. The same reasoning already keeps a crop-refused row IN the comparison
    set at write time (see Monitor._maybe_assess_with_gemini); this is the read
    side of that decision finally being honoured.

    Returned shape:
      sampled            every row the roll selected, matrix or not
      compared           rows carrying both ratings, always equal to
                         sum(agreement_matrix(conn).values())
      no_verdict         sampled - compared
      no_verdict_reasons [{"reason": str, "count": int}], commonest first

    The WHERE clause is deliberately character-for-character the one in
    agreement_matrix, so the two can never end up describing different
    populations, which is the failure this function exists to make impossible.
    """
    rows = conn.execute(
        """SELECT CASE
                    WHEN local_rating IS NOT NULL AND gemini_rating IS NOT NULL
                         THEN 'COMPARED'
                    WHEN api_error IS NOT NULL THEN api_error
                    WHEN local_rating IS NULL THEN 'NO_LOCAL_RATING'
                    ELSE 'NO_VERDICT'
                  END AS disposition,
                  COUNT(*) AS n
           FROM checks
           WHERE is_comparison_sample = 1 AND outcome = 'OK'
           GROUP BY disposition"""
    ).fetchall()

    counts = {row["disposition"]: row["n"] for row in rows}
    compared = counts.pop("COMPARED", 0)
    sampled = compared + sum(counts.values())
    # Commonest first, then alphabetically, so the page's ordering is stable
    # rather than whatever the query planner happened to emit.
    reasons = [{"reason": reason, "count": n}
               for reason, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    return {"sampled": sampled, "compared": compared,
            "no_verdict": sampled - compared, "no_verdict_reasons": reasons}


def recent_checks(conn: sqlite3.Connection, limit: int = 50) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM checks ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        item["cameras"] = json.loads(item["cameras"] or "[]")
        item["metrics"] = json.loads(item["metrics"]) if item["metrics"] else None
        out.append(item)
    return out


def today_counts(conn: sqlite3.Connection) -> dict[str, int]:
    row = conn.execute(
        """SELECT SUM(local_rating = 'GOOD')   AS good,
                  SUM(local_rating = 'DECENT') AS decent,
                  SUM(local_rating = 'POOR')   AS poor
           FROM checks WHERE date = ? AND outcome = 'OK'""",
        (date.today().isoformat(),),
    ).fetchone()
    return {k: (row[k] or 0) for k in ("good", "decent", "poor")}


def coaching_calls_today(conn: sqlite3.Connection) -> int:
    """Gated coaching calls that actually left the machine today.

    Counts ATTEMPTS, not successes: a call that came back API_ERROR still
    uploaded a cropped photo and still spent a request. Counting only successes
    would let a failing API burn an unbounded number of uploads in a day, which
    is precisely the case a budget exists to bound.

    CROP_UNAVAILABLE is excluded because it is a LOCAL refusal: crop_to_person
    declined to isolate a person region, so nothing was sent and nothing was
    billed. Charging for it would let a run of undetectable frames silently
    starve the coaching the budget is meant to allow.

    Comparison samples are excluded because the budget bounds the coaching path
    only. Letting gated calls eat into the sampled ones would truncate the
    comparison set earlier on worse-posture days, which is exactly the
    posture-correlated censoring the unbiased roll exists to prevent.
    """
    row = conn.execute(
        """SELECT COUNT(*) AS n FROM checks
           WHERE date = ? AND is_comparison_sample = 0
             AND (gemini_model IS NOT NULL
                  OR (api_error IS NOT NULL AND api_error != ?))""",
        (date.today().isoformat(), Outcome.CROP_UNAVAILABLE.value),
    ).fetchone()
    return row["n"] or 0


def seconds_since_last_api_call(conn: sqlite3.Connection) -> float:
    row = conn.execute(
        "SELECT ts FROM checks WHERE gemini_model IS NOT NULL ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return float("inf")
    return (datetime.now() - datetime.fromisoformat(row["ts"])).total_seconds()
