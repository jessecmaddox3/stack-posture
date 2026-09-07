from datetime import date, timedelta

import pytest

from posture.store.db import connect, migrate
from posture.store.queries import (
    CheckRecord, baseline_changed_between, coaching_calls_today, cva_series,
    daily_scores, insert_check, save_baseline, total_checks_recorded,
)
from posture.types import Outcome, PostureMetrics, Rating


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "test.db")
    migrate(c)
    yield c
    c.close()


def _metrics_for(roles: list[str], cva: float | None) -> PostureMetrics:
    """Metrics a check on these camera roles could actually have produced.

    capability is derived from the metrics a row CARRIES, not from the cameras
    that returned a frame, so a fixture claiming ["front", "side"] while holding
    no side-derived number describes a row the monitor cannot write. Building
    the metrics from the roles keeps these fixtures reproducible by the real
    pipeline, which is what makes assertions about capability mean anything.
    """
    if "side" in roles and cva is None:
        cva = 58.0
    return PostureMetrics(
        shoulder_tilt_deg=2.0 if "front" in roles else None,
        cva_deg=cva if "side" in roles else None,
    )


def _add(conn, *, day, rating, cameras=None, baseline_id=None, cva=None):
    roles = cameras if cameras is not None else (
        ["front", "side"] if cva is not None else ["front"])
    insert_check(conn, CheckRecord(
        ts=f"{day}T10:00:00", date=day, outcome=Outcome.OK, baseline_id=baseline_id,
        cameras=roles, metrics=_metrics_for(roles, cva),
        local_rating=rating, local_driver=None, gemini_rating=None,
        gemini_issue=None, gemini_details=None, gemini_tip=None,
        gemini_model=None, is_comparison_sample=False, api_error=None))


def test_daily_scores_bounds_by_date_not_row_count(conn):
    # Two desk setups on the same day produce two rows. A row-based limit would
    # halve the window silently.
    for offset in range(5):
        day = (date.today() - timedelta(days=offset)).isoformat()
        _add(conn, day=day, rating=Rating.GOOD, cameras=["front", "side"])
        _add(conn, day=day, rating=Rating.POOR, cameras=["front"])
    rows = daily_scores(conn, days=5)
    assert len({row["date"] for row in rows}) == 5
    assert len(rows) == 10


def test_daily_scores_never_mixes_two_baselines_into_one_point(conn):
    # A recalibration rebases every score after it. Two checks on the same day,
    # same camera setup, either side of a recalibration, are measured against
    # different references: averaging them produces a number that is true of
    # neither ruler. Without baseline_id in the GROUP BY this is a single row
    # scoring 60.0, the arithmetic mean of a POOR on the old baseline and a
    # GOOD on the new one, which is exactly the blend the page must never make.
    old = save_baseline(conn, PostureMetrics(cva_deg=60.0), "before").id
    new = save_baseline(conn, PostureMetrics(cva_deg=58.0), "after").id
    day = date.today().isoformat()
    _add(conn, day=day, rating=Rating.POOR, baseline_id=old)
    _add(conn, day=day, rating=Rating.GOOD, baseline_id=new)

    rows = daily_scores(conn, days=60)

    assert len(rows) == 2
    assert {(r["baseline_id"], r["score"]) for r in rows} == {(old, 20.0), (new, 100.0)}
    # Every row must name the single baseline its score was measured against,
    # so nothing downstream has to guess which ruler a point belongs to.
    for row in rows:
        assert "baseline_id" in row
        assert row["total"] == 1


def test_daily_scores_keeps_one_row_per_day_when_the_baseline_is_stable(conn):
    # The split above must not fragment ordinary history: one baseline all
    # week is still one point per day per setup.
    only = save_baseline(conn, PostureMetrics(cva_deg=60.0), "only").id
    day = date.today().isoformat()
    _add(conn, day=day, rating=Rating.POOR, baseline_id=only)
    _add(conn, day=day, rating=Rating.GOOD, baseline_id=only)
    rows = daily_scores(conn, days=60)
    assert len(rows) == 1
    assert rows[0]["score"] == 60.0
    assert rows[0]["baseline_id"] == only


def test_cva_series_bounds_by_date_not_row_count(conn):
    # Same bug class already fixed in daily_scores: LIMIT applied to ROWS lets
    # a sparse history stretch far past the window the header promises. One
    # reading every fifth day for 300 days returns 60 rows under a row limit,
    # spanning 296 calendar days under a heading that says "last 60 days".
    today = date.today()
    for offset in range(0, 300, 5):
        _add(conn, day=(today - timedelta(days=offset)).isoformat(),
             rating=Rating.GOOD, cva=50.0)

    rows = cva_series(conn, days=60)

    assert rows, "expected some readings inside the window"
    oldest = date.fromisoformat(min(r["date"] for r in rows))
    newest = date.fromisoformat(max(r["date"] for r in rows))
    span = (newest - oldest).days + 1
    assert span <= 60, f"series spans {span} calendar days under a 60-day window"
    assert oldest >= today - timedelta(days=59)


def test_baseline_changed_between_finds_the_switch(conn):
    # Real baselines rows: checks.baseline_id is a FOREIGN KEY and foreign_keys
    # is ON, so invented integers raise IntegrityError.
    first = save_baseline(conn, PostureMetrics(cva_deg=60.0), "before").id
    second = save_baseline(conn, PostureMetrics(cva_deg=58.0), "after").id
    _add(conn, day="2026-07-01", rating=Rating.GOOD, baseline_id=first)
    _add(conn, day="2026-07-05", rating=Rating.GOOD, baseline_id=second)
    assert baseline_changed_between(conn, "2026-07-01", "2026-07-31") == "2026-07-05"


def test_baseline_changed_between_is_none_on_one_baseline(conn):
    only = save_baseline(conn, PostureMetrics(cva_deg=60.0), "only").id
    _add(conn, day="2026-07-01", rating=Rating.GOOD, baseline_id=only)
    _add(conn, day="2026-07-05", rating=Rating.GOOD, baseline_id=only)
    assert baseline_changed_between(conn, "2026-07-01", "2026-07-31") is None


def test_total_checks_recorded_counts_every_outcome(conn):
    # daily_scores only ever sees OK checks with a rating. A PERSON_ABSENT or
    # API_ERROR row is still a real check, and it still shows up in
    # recent_checks (unfiltered), so a genuine total must count it too.
    today = date.today().isoformat()
    _add(conn, day=today, rating=Rating.GOOD)
    insert_check(conn, CheckRecord(
        ts=f"{today}T11:00:00", date=today, outcome=Outcome.PERSON_ABSENT,
        baseline_id=None, cameras=["front"], metrics=None, local_rating=None,
        local_driver=None, gemini_rating=None, gemini_issue=None,
        gemini_details=None, gemini_tip=None, gemini_model=None,
        is_comparison_sample=False, api_error=None))
    insert_check(conn, CheckRecord(
        ts=f"{today}T12:00:00", date=today, outcome=Outcome.API_ERROR,
        baseline_id=None, cameras=["front"], metrics=None, local_rating=None,
        local_driver=None, gemini_rating=None, gemini_issue=None,
        gemini_details=None, gemini_tip=None, gemini_model=None,
        is_comparison_sample=False, api_error="timeout"))
    assert total_checks_recorded(conn, days=60) == 3
    # daily_scores must NOT see the two failed checks, or this test would not
    # be discriminating between the two queries at all.
    assert sum(row["total"] for row in daily_scores(conn, days=60)) == 1


# --- what the daily coaching budget actually counts ---

def _api_row(conn, *, day, comparison, model="gemini-3.5-flash-lite", api_error=None):
    insert_check(conn, CheckRecord(
        ts=f"{day}T10:00:00", date=day, outcome=Outcome.OK, baseline_id=None,
        cameras=["front"], metrics=PostureMetrics(), local_rating=Rating.POOR,
        local_driver=None, gemini_rating=None, gemini_issue=None,
        gemini_details=None, gemini_tip=None, gemini_model=model,
        is_comparison_sample=comparison, api_error=api_error))


def test_coaching_calls_today_counts_only_gated_calls_from_today(conn):
    today = date.today().isoformat()
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    _api_row(conn, day=today, comparison=False)      # counts
    _api_row(conn, day=today, comparison=True)       # comparison sample, does not
    _api_row(conn, day=yesterday, comparison=False)  # yesterday, does not
    _add(conn, day=today, rating=Rating.POOR)        # no call at all, does not
    assert coaching_calls_today(conn) == 1


def test_a_failed_gated_call_still_spends_budget(conn):
    # The photo left the machine and the request was billed as an attempt. The
    # budget bounds uploads and spend, so counting only successes would let a
    # failing API burn the day unbounded.
    today = date.today().isoformat()
    _api_row(conn, day=today, comparison=False, model=None, api_error="API_ERROR")
    assert coaching_calls_today(conn) == 1


def test_a_crop_refusal_does_not_spend_budget(conn):
    # CROP_UNAVAILABLE means we refused locally and never contacted the API, so
    # nothing was uploaded and nothing was billed. Charging for it would let a
    # run of undetectable frames silently starve real coaching.
    today = date.today().isoformat()
    _api_row(conn, day=today, comparison=False, model=None,
             api_error=Outcome.CROP_UNAVAILABLE.value)
    assert coaching_calls_today(conn) == 0
