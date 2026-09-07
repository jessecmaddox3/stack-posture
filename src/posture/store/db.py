"""SQLite connection and migration entry point."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from posture.store import migrations


def connect(path: Path) -> sqlite3.Connection:
    """Open the database with WAL and a busy timeout.

    v2 used a bare connect(), so the dashboard thread and the check thread
    could collide on "database is locked" (spec M13).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def migrate(conn: sqlite3.Connection) -> None:
    """Apply every migration newer than the recorded version. Idempotent.

    Runs as one transaction. Python's sqlite3 module otherwise auto-commits
    each DDL statement individually, which would let a failure part-way
    (e.g. quarantining the legacy table but then failing a later CREATE)
    leave the database in a state neither the v2 nor v3 code path expects.
    An explicit BEGIN keeps every statement here, including the legacy-table
    rename, inside a single all-or-nothing unit.

    Reaches through the `migrations` module rather than importing its names
    directly, so that MIGRATIONS is read fresh on every call; that is what
    lets tests substitute a broken migration set via monkeypatch.
    """
    conn.execute("BEGIN")
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY)")
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        current = row[0] if row else 0

        migrations._quarantine_legacy_checks(conn)

        for version, statements in sorted(migrations.MIGRATIONS.items()):
            if version <= current:
                continue
            for statement in statements:
                conn.execute(statement)

        conn.execute("DELETE FROM schema_version")
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (migrations.SCHEMA_VERSION,))
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
