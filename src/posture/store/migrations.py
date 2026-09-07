"""Versioned schema. Add a new key rather than editing an existing migration."""
from __future__ import annotations

SCHEMA_VERSION = 2

LEGACY_TABLE = "checks_v2"

# The v2 checks table had these columns and none of v3's. Detecting on a column
# that v3 requires is more robust than matching the full v2 set, because it stays
# correct if v2 ever had minor variants.
_V3_REQUIRED_CHECKS_COLUMN = "is_comparison_sample"


def _is_legacy_schema(conn) -> bool:
    """True when a checks table exists and predates v3.

    Detects on the ABSENCE of a v3-required column rather than the presence of a
    v2 one, so it stays correct if v2 had minor variants across installs. Split
    out as its own predicate so that choice can be tested directly rather than
    only through its effects.
    """
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    if "checks" not in tables:
        return False
    columns = {r[1] for r in conn.execute("PRAGMA table_info(checks)")}
    return _V3_REQUIRED_CHECKS_COLUMN not in columns


def _quarantine_legacy_checks(conn) -> None:
    """Rename a v2 checks table aside so the v3 schema can be created.

    V2 ratings came from a vision model scoring against no baseline. V3 ratings
    are deviations from a calibrated personal baseline. Converting one into the
    other would invent comparability that does not exist, so the old history is
    preserved verbatim under a legacy name and kept out of every v3 query.
    """
    if not _is_legacy_schema(conn):
        return
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    if LEGACY_TABLE in tables:
        # Both a legacy checks table AND a previous quarantine. Renaming again
        # would either fail or, worse, need a numbered suffix that silently
        # multiplies copies of the history. No code path in this app produces
        # this state, so say so clearly instead of guessing.
        raise RuntimeError(
            f"{LEGACY_TABLE} already exists alongside a pre-v3 checks table. "
            f"This database is in a state this app does not create. Move "
            f"history.db aside and start fresh, or reconcile the two tables by "
            f"hand; refusing rather than risking your history."
        )
    conn.execute(f"ALTER TABLE checks RENAME TO {LEGACY_TABLE}")
    # v2's other table has no v3 counterpart and is not read by anything.
    if "daily_summaries" in tables:
        conn.execute("ALTER TABLE daily_summaries RENAME TO daily_summaries_v2")


MIGRATIONS: dict[int, list[str]] = {
    1: [
        """
        CREATE TABLE IF NOT EXISTS baselines (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at  TEXT NOT NULL,
            label       TEXT NOT NULL DEFAULT '',
            metrics     TEXT NOT NULL,
            -- Which way is forward, frozen at calibration. NULL means it was not
            -- captured (front camera only, or an older baseline), in which case
            -- trunk angle falls back to a less reliable per-frame estimate.
            facing_sign REAL,
            active      INTEGER NOT NULL DEFAULT 1
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS checks (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            ts                   TEXT NOT NULL,
            date                 TEXT NOT NULL,
            outcome              TEXT NOT NULL,
            baseline_id          INTEGER REFERENCES baselines(id),
            cameras              TEXT NOT NULL DEFAULT '[]',
            metrics              TEXT,
            local_rating         TEXT,
            local_driver         TEXT,
            gemini_rating        TEXT,
            gemini_issue         TEXT,
            gemini_details       TEXT,
            gemini_tip           TEXT,
            gemini_model         TEXT,
            is_comparison_sample INTEGER NOT NULL DEFAULT 0,
            api_error            TEXT
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_checks_date ON checks(date)",
        "CREATE INDEX IF NOT EXISTS idx_checks_sample ON checks(is_comparison_sample)",
        """
        CREATE TABLE IF NOT EXISTS summary_shown (
            date      TEXT PRIMARY KEY,
            shown_at  TEXT NOT NULL
        )
        """,
    ],
    # A one-camera GOOD and a two-camera GOOD do not mean the same thing: GOOD
    # with both cameras means CVA, trunk, and front metrics all passed, GOOD with
    # only the front camera means only the front metrics did. capability records
    # which cameras contributed so a query can stratify on it rather than
    # averaging the two together into a trend that changes meaning with the desk
    # setup. Added as its own migration, per this module's convention, rather
    # than folded into migration 1's CREATE TABLE: that statement already ran on
    # every existing v3 database, so a fresh database and an upgraded one must
    # both reach the same schema through this ALTER TABLE.
    2: [
        "ALTER TABLE checks ADD COLUMN capability TEXT",
        "CREATE INDEX IF NOT EXISTS idx_checks_capability ON checks(date, capability)",
    ],
}
