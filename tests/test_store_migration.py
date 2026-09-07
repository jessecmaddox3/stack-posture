import sqlite3
from pathlib import Path

import pytest

from posture.store.db import connect, migrate
from posture.store.migrations import LEGACY_TABLE


def make_v2_db(path):
    """Build a database with v2's exact schema, as the old app left it."""
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE checks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            rating TEXT NOT NULL,
            issue TEXT DEFAULT '',
            details TEXT DEFAULT '',
            tip TEXT DEFAULT '',
            cameras_used INTEGER DEFAULT 1,
            raw_response TEXT DEFAULT ''
        )
    """)
    conn.execute("""
        CREATE TABLE daily_summaries (
            date TEXT PRIMARY KEY, total_checks INTEGER, good_count INTEGER,
            decent_count INTEGER, poor_count INTEGER, summary_shown INTEGER DEFAULT 0
        )
    """)
    conn.execute(
        "INSERT INTO checks (timestamp, date, time, rating, issue) VALUES (?,?,?,?,?)",
        ("2026-05-01T09:00:00", "2026-05-01", "09:00:00", "POOR", "forward head"),
    )
    conn.commit()
    conn.close()


def test_migrate_over_a_v2_database_does_not_raise(tmp_path):
    """The exact crash a v2 user hits on first launch of v3."""
    db = tmp_path / "history.db"
    make_v2_db(db)
    conn = connect(db)
    migrate(conn)  # previously: OperationalError, no such column is_comparison_sample
    conn.close()


def test_v2_history_is_preserved_under_a_legacy_name(tmp_path):
    db = tmp_path / "history.db"
    make_v2_db(db)
    conn = connect(db)
    migrate(conn)
    rows = conn.execute(f"SELECT rating, issue FROM {LEGACY_TABLE}").fetchall()
    assert len(rows) == 1
    assert rows[0]["rating"] == "POOR"
    conn.close()


def test_v3_tables_exist_and_are_empty_after_migrating_from_v2(tmp_path):
    # V2 ratings scored against no baseline. They are not comparable to v3
    # deviations, so they must not appear in the v3 series.
    db = tmp_path / "history.db"
    make_v2_db(db)
    conn = connect(db)
    migrate(conn)
    assert conn.execute("SELECT COUNT(*) FROM checks").fetchone()[0] == 0
    cols = {r[1] for r in conn.execute("PRAGMA table_info(checks)")}
    assert "is_comparison_sample" in cols
    assert "facing_sign" in {r[1] for r in conn.execute("PRAGMA table_info(baselines)")}
    conn.close()


def test_migrating_twice_is_idempotent(tmp_path):
    db = tmp_path / "history.db"
    make_v2_db(db)
    conn = connect(db)
    migrate(conn)
    migrate(conn)
    assert conn.execute(
        f"SELECT COUNT(*) FROM {LEGACY_TABLE}").fetchone()[0] == 1
    conn.close()


def test_migrating_repeatedly_over_a_live_v3_database_preserves_rows(conn_factory=None):
    """The ordinary case: every launch after the first.

    A mutant that quarantines unconditionally cannot survive this. It attempts
    ALTER TABLE checks RENAME on the very first migrate, before the table exists,
    and raises immediately; on a database that already holds a v3 checks table it
    would instead rename the LIVE table aside and lose the history. Either way the
    user's data is gone. test_migrating_twice_is_idempotent does not catch it,
    because it only counts rows in the quarantined table and never checks that the
    live checks table still holds anything.
    """
    import tempfile
    from posture.store.queries import CheckRecord, insert_check, recent_checks
    from posture.types import Outcome, PostureMetrics, Rating

    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "live.db"
        conn = connect(db)
        migrate(conn)
        insert_check(conn, CheckRecord(
            ts="2026-07-30T10:00:00", date="2026-07-30", outcome=Outcome.OK,
            baseline_id=None, cameras=["front"],
            metrics=PostureMetrics(shoulder_tilt_deg=1.0),
            local_rating=Rating.GOOD, local_driver=None, gemini_rating=None,
            gemini_issue=None, gemini_details=None, gemini_tip=None,
            gemini_model=None, is_comparison_sample=False, api_error=None))
        conn.close()

        for _ in range(3):
            conn = connect(db)
            migrate(conn)
            rows = recent_checks(conn, 10)
            assert len(rows) == 1, "a repeated migrate lost the live checks table"
            assert rows[0]["local_rating"] == "GOOD"
            names = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            assert LEGACY_TABLE not in names
            conn.close()


def test_a_failed_migration_rolls_back_the_rename(tmp_path, monkeypatch):
    """Atomicity, as a test rather than a manual check.

    A half-migrated database is worse than a clean failure: checks renamed away
    with no v3 tables created, and the next launch sees a state neither branch
    expects. Nothing caught a regression here before.
    """
    from posture.store import migrations

    db = tmp_path / "history.db"
    make_v2_db(db)
    broken = dict(migrations.MIGRATIONS)
    broken[1] = list(broken[1]) + ["CREATE TABLE this is not valid sql"]
    monkeypatch.setattr(migrations, "MIGRATIONS", broken)

    conn = connect(db)
    with pytest.raises(sqlite3.OperationalError):
        migrate(conn)
    conn.close()

    # Reopen from disk: nothing may have persisted.
    check = sqlite3.connect(str(db))
    names = {r[0] for r in check.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert LEGACY_TABLE not in names, "the rename survived a failed migration"
    assert "checks" in names
    cols = {r[1] for r in check.execute("PRAGMA table_info(checks)")}
    assert "raw_response" in cols, "the original v2 table was not left intact"
    assert check.execute("SELECT COUNT(*) FROM checks").fetchone()[0] == 1
    check.close()


def test_is_legacy_schema_detects_on_the_missing_v3_column(tmp_path):
    """Pin the detection strategy itself, not only its effects.

    Presence-of-v2-column and absence-of-v3-column agree on every real fixture,
    so nothing distinguished them. This v2-variant table drops raw_response,
    which a presence-based check would miss.
    """
    from posture.store.migrations import _is_legacy_schema

    db = tmp_path / "variant.db"
    raw = sqlite3.connect(str(db))
    raw.execute("CREATE TABLE checks (id INTEGER PRIMARY KEY, rating TEXT)")
    raw.commit()
    raw.close()

    conn = connect(db)
    assert _is_legacy_schema(conn) is True
    conn.close()


def test_is_legacy_schema_is_false_for_a_v3_table(tmp_path):
    conn = connect(tmp_path / "v3.db")
    migrate(conn)
    from posture.store.migrations import _is_legacy_schema
    assert _is_legacy_schema(conn) is False
    conn.close()


def test_a_fresh_database_creates_no_legacy_table(tmp_path):
    conn = connect(tmp_path / "fresh.db")
    migrate(conn)
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert LEGACY_TABLE not in names
    conn.close()
