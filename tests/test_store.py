from datetime import date

import pytest

from posture.store.db import connect, migrate
from posture.store.migrations import SCHEMA_VERSION
from posture.store.queries import (
    CheckRecord, active_baseline, agreement_matrix, daily_scores, insert_check,
    recent_checks, save_baseline, today_counts,
)
from posture.types import Outcome, PostureMetrics, Rating


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "test.db")
    migrate(c)
    yield c
    c.close()


def rec(**kw):
    defaults = dict(
        ts=f"{date.today().isoformat()}T10:00:00", date=date.today().isoformat(), outcome=Outcome.OK,
        baseline_id=None, cameras=["front"], metrics=PostureMetrics(cva_deg=58.0),
        local_rating=Rating.GOOD, local_driver=None, gemini_rating=None,
        gemini_issue=None, gemini_details=None, gemini_tip=None,
        gemini_model=None, is_comparison_sample=False, api_error=None,
    )
    return CheckRecord(**{**defaults, **kw})


def test_migrate_sets_the_schema_version(conn):
    assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == SCHEMA_VERSION


def test_migrate_is_idempotent(conn):
    migrate(conn)
    migrate(conn)
    assert conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == 1


def test_wal_is_enabled(conn):
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


def test_insert_and_read_back(conn):
    insert_check(conn, rec())
    rows = recent_checks(conn, limit=10)
    assert len(rows) == 1
    assert rows[0]["local_rating"] == "GOOD"
    assert rows[0]["outcome"] == "OK"


def test_metrics_round_trip_as_json(conn):
    insert_check(conn, rec(metrics=PostureMetrics(cva_deg=51.5, trunk_angle_deg=9.0)))
    assert recent_checks(conn, 1)[0]["metrics"]["cva_deg"] == 51.5


def test_save_and_load_active_baseline(conn):
    saved = save_baseline(conn, PostureMetrics(cva_deg=62.0), label="post-op week 3")
    loaded = active_baseline(conn)
    assert loaded.id == saved.id
    assert loaded.metrics.cva_deg == 62.0
    assert loaded.label == "post-op week 3"


def test_saving_a_new_baseline_deactivates_the_old_one(conn):
    first = save_baseline(conn, PostureMetrics(cva_deg=55.0), label="first")
    second = save_baseline(conn, PostureMetrics(cva_deg=62.0), label="second")
    assert active_baseline(conn).id == second.id
    row = conn.execute("SELECT active FROM baselines WHERE id = ?", (first.id,)).fetchone()
    assert row[0] == 0


def test_facing_sign_round_trips(conn):
    # Without this, the calibrated sign is silently lost on read and trunk angle
    # falls back to the per-frame estimate forever, which can invert on a head turn.
    save_baseline(conn, PostureMetrics(cva_deg=60.0), label="side cam", facing_sign=-1.0)
    assert active_baseline(conn).facing_sign == -1.0


def test_facing_sign_is_none_when_not_captured(conn):
    # Front-camera-only calibration cannot determine facing, and that must read
    # back as None rather than as a default direction.
    save_baseline(conn, PostureMetrics(shoulder_tilt_deg=0.0), label="front only")
    assert active_baseline(conn).facing_sign is None


def test_active_baseline_is_none_before_calibration(conn):
    assert active_baseline(conn) is None


def test_daily_scores_ignore_non_ok_outcomes(conn):
    insert_check(conn, rec(local_rating=Rating.GOOD))
    insert_check(conn, rec(outcome=Outcome.API_ERROR, local_rating=None))
    insert_check(conn, rec(outcome=Outcome.PERSON_ABSENT, local_rating=None))
    scores = daily_scores(conn, days=30)
    assert len(scores) == 1
    assert scores[0]["score"] == 100.0


def test_daily_scores_exclude_a_rated_row_with_a_failed_outcome(conn):
    """The outcome filter must do real work, not be shadowed by the null check.

    test_daily_scores_ignore_non_ok_outcomes builds its non-OK rows with
    local_rating=None, so `local_rating IS NOT NULL` already excludes them and
    removing `outcome = 'OK'` changes nothing. Mutation testing confirmed the
    filter could be deleted with all tests still passing. This row is the case
    that discriminates: a genuine local rating alongside a failed outcome.
    """
    insert_check(conn, rec(local_rating=Rating.GOOD))
    insert_check(conn, rec(outcome=Outcome.API_ERROR, local_rating=Rating.POOR))
    scores = daily_scores(conn, days=30)
    assert len(scores) == 1
    assert scores[0]["total"] == 1
    assert scores[0]["score"] == 100.0


def test_agreement_matrix_excludes_failed_outcomes(conn):
    # Same shape of gap on the comparison statistics: a row carrying both ratings
    # but a failed outcome must not contribute to the agreement figure.
    insert_check(conn, rec(is_comparison_sample=True,
                           local_rating=Rating.GOOD, gemini_rating=Rating.GOOD))
    insert_check(conn, rec(outcome=Outcome.API_ERROR, is_comparison_sample=True,
                           local_rating=Rating.POOR, gemini_rating=Rating.POOR))
    matrix = agreement_matrix(conn)
    assert matrix == {("GOOD", "GOOD"): 1}


@pytest.mark.parametrize("ratings,expected", [
    ([Rating.GOOD], 100.0),
    ([Rating.POOR], 20.0),
    ([Rating.GOOD, Rating.POOR], 60.0),
    ([Rating.GOOD, Rating.DECENT, Rating.POOR], 60.0),
])
def test_daily_score_arithmetic(conn, ratings, expected):
    for r in ratings:
        insert_check(conn, rec(local_rating=r))
    assert daily_scores(conn, days=30)[0]["score"] == pytest.approx(expected)


def test_agreement_matrix_counts_only_comparison_samples(conn):
    insert_check(conn, rec(is_comparison_sample=True,
                           local_rating=Rating.GOOD, gemini_rating=Rating.GOOD))
    insert_check(conn, rec(is_comparison_sample=True,
                           local_rating=Rating.GOOD, gemini_rating=Rating.POOR))
    # A gated call must NOT pollute the statistics.
    insert_check(conn, rec(is_comparison_sample=False,
                           local_rating=Rating.POOR, gemini_rating=Rating.POOR))
    matrix = agreement_matrix(conn)
    assert matrix[("GOOD", "GOOD")] == 1
    assert matrix[("GOOD", "POOR")] == 1
    assert ("POOR", "POOR") not in matrix


def test_today_counts(conn):
    today = date.today().isoformat()
    insert_check(conn, rec(date=today, local_rating=Rating.GOOD))
    insert_check(conn, rec(date=today, local_rating=Rating.POOR))
    counts = today_counts(conn)
    assert counts["good"] == 1 and counts["poor"] == 1 and counts["decent"] == 0


def test_date_index_exists(conn):
    names = {r[1] for r in conn.execute("PRAGMA index_list(checks)")}
    assert "idx_checks_date" in names


def test_capability_is_stored_per_check(conn):
    insert_check(conn, rec(cameras=["front"],
                           metrics=PostureMetrics(shoulder_tilt_deg=2.0)))
    insert_check(conn, rec(cameras=["front", "side"],
                           metrics=PostureMetrics(shoulder_tilt_deg=2.0,
                                                  cva_deg=58.0)))
    rows = recent_checks(conn, 10)
    assert {r["capability"] for r in rows} == {"front", "front+side"}


def test_the_capability_metric_map_matches_what_metrics_actually_produces():
    """_ROLE_METRICS is a hand-written mirror of metrics.py. Pin the two together.

    If a new front or side metric is added to metrics.py and not here, capability
    silently under-reports: a row carrying only the new metric would claim no
    camera role at all. This calls the real metric functions on the real
    fixtures and compares the field sets, so drift fails here rather than in the
    stored data months later.
    """
    from posture.metrics import compute_front_metrics, compute_side_metrics
    from posture.store.queries import _ROLE_METRICS
    from tests.synthetic import upright_front, upright_side

    front = compute_front_metrics(upright_front())
    side = compute_side_metrics(upright_side(), facing_sign=1.0)
    assert set(front.available()) == set(_ROLE_METRICS["front"])
    assert set(side.available()) == set(_ROLE_METRICS["side"])
    assert set(_ROLE_METRICS) == {"front", "side"}


def test_daily_scores_are_stratified_by_capability(conn):
    """A one-camera GOOD and a two-camera GOOD do not mean the same thing.

    Averaging them into a single daily number produces a trend whose meaning
    changes with the desk setup rather than with the back.
    """
    today = date.today().isoformat()
    insert_check(conn, rec(date=today, local_rating=Rating.GOOD,
                           cameras=["front"],
                           metrics=PostureMetrics(shoulder_tilt_deg=2.0)))
    insert_check(conn, rec(date=today, local_rating=Rating.POOR,
                           cameras=["front", "side"],
                           metrics=PostureMetrics(shoulder_tilt_deg=2.0,
                                                  cva_deg=58.0)))
    scores = daily_scores(conn, days=30)
    by_capability = {r["capability"]: r for r in scores}
    assert by_capability["front"]["score"] == 100.0
    assert by_capability["front+side"]["score"] == 20.0


# --- F2: agreement_matrix answers a conditional question, and its total was
# being read as the size of the comparison sample.

def test_comparison_sample_counts_denominator_matches_the_matrix_population(conn):
    """`compared` must be exactly the matrix total, or the page has two denominators.

    These are two queries over the same table with the same WHERE clause. If
    they ever drift, the card would print a matched count drawn from one
    population over a denominator drawn from another, which is a worse lie than
    the one F2 fixed because the numbers would look internally consistent.
    """
    from posture.store.queries import comparison_sample_counts

    insert_check(conn, rec(is_comparison_sample=True,
                           local_rating=Rating.GOOD, gemini_rating=Rating.GOOD))
    insert_check(conn, rec(is_comparison_sample=True,
                           local_rating=Rating.POOR, gemini_rating=Rating.GOOD))
    insert_check(conn, rec(is_comparison_sample=True, local_rating=Rating.GOOD,
                           gemini_rating=None, api_error=Outcome.API_ERROR.value))
    insert_check(conn, rec(is_comparison_sample=False,
                           local_rating=Rating.POOR, gemini_rating=Rating.POOR))

    counts = comparison_sample_counts(conn)
    assert counts["compared"] == sum(agreement_matrix(conn).values()) == 2
    assert counts["sampled"] == 3
    assert counts["no_verdict"] == 1
    assert counts["no_verdict_reasons"] == [{"reason": "API_ERROR", "count": 1}]


def test_comparison_sample_counts_excludes_failed_outcomes_like_the_matrix_does(conn):
    """A non-OK row must be out of BOTH, or the denominator grows past the matrix."""
    from posture.store.queries import comparison_sample_counts

    insert_check(conn, rec(is_comparison_sample=True,
                           local_rating=Rating.GOOD, gemini_rating=Rating.GOOD))
    insert_check(conn, rec(outcome=Outcome.CAMERA_BUSY, is_comparison_sample=True,
                           local_rating=None, gemini_rating=None))

    counts = comparison_sample_counts(conn)
    assert counts["sampled"] == 1
    assert counts["compared"] == 1
    assert counts["no_verdict"] == 0


def test_comparison_sample_counts_orders_reasons_commonest_first(conn):
    from posture.store.queries import comparison_sample_counts

    for _ in range(3):
        insert_check(conn, rec(is_comparison_sample=True, local_rating=Rating.GOOD,
                               api_error=Outcome.CROP_UNAVAILABLE.value))
    for _ in range(5):
        insert_check(conn, rec(is_comparison_sample=True, local_rating=Rating.GOOD,
                               api_error=Outcome.API_ERROR.value))

    assert comparison_sample_counts(conn)["no_verdict_reasons"] == [
        {"reason": "API_ERROR", "count": 5},
        {"reason": "CROP_UNAVAILABLE", "count": 3},
    ]
