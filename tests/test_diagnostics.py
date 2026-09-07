from posture.config import Config
from posture.diagnostics import build_diagnostics, redact
from posture.store.db import connect, migrate
from posture.store.queries import save_baseline
from posture.types import PostureMetrics

import pytest


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "diag.db")
    migrate(c)
    yield c
    c.close()


def test_a_google_api_key_is_redacted():
    # The shape Google issues. If one ever reaches a log line, it must not reach
    # the clipboard.
    fake_key = "AIza" + "SyA1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q"
    text = f"calling gemini with key {fake_key}"
    out = redact(text)
    assert "AIzaSy" not in out
    assert "REDACTED" in out


def test_a_bearer_token_is_redacted():
    out = redact("Authorization: Bearer ya29.a0AfH6SMBx-longish-token-value")
    assert "ya29" not in out
    assert "REDACTED" in out


def test_a_bare_oauth_token_without_the_bearer_word_is_redacted():
    # The line above is caught by EITHER the ya29 pattern or the bearer pattern,
    # so deleting one of them alone left the whole suite green: they covered for
    # each other and neither was actually pinned. This line has no "Bearer", so
    # only the ya29 pattern can catch it.
    out = redact("refresh failed for token ya29.a0AfH6SMBx-longish-token-value")
    assert "ya29" not in out
    assert "REDACTED" in out


def test_a_bearer_token_that_is_not_google_shaped_is_redacted():
    # The mirror of the test above: no ya29 prefix, so only the bearer pattern
    # can catch it. Together these two pin each pattern independently.
    out = redact("Authorization: Bearer abc123DEF456ghi789")
    assert "abc123DEF456ghi789" not in out
    assert "REDACTED" in out


def test_an_assignment_to_a_key_like_name_is_redacted():
    for line in ("api_key = hunter2sekrit", "GEMINI_API_KEY: hunter2sekrit",
                 '"apiKey": "hunter2sekrit"'):
        assert "hunter2sekrit" not in redact(line), line


def test_ordinary_text_survives_redaction():
    # Over-redacting makes the bundle useless, which is its own failure.
    text = "2026-07-30 09:15:02 [INFO] posture: calling gemini (comparison sample)"
    assert redact(text) == text


def test_the_bundle_reports_state_not_just_logs(conn):
    save_baseline(conn, PostureMetrics(cva_deg=58.0, shoulder_tilt_deg=1.0), "desk")
    out = build_diagnostics(conn, Config(), log_text="", camera_indices=(0, 1))
    assert "schema" in out.lower()
    assert "baseline" in out.lower()
    assert "cameras" in out.lower()


def test_the_bundle_redacts_its_log_section(conn):
    out = build_diagnostics(
        conn, Config(),
        log_text="key " + "AIza" + "SyA1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q failed",
        camera_indices=())
    assert "AIzaSy" not in out


def test_the_bundle_contains_no_image_data(conn):
    # Frames and JPEG bytes must never be in here. A base64 blob would also make
    # the bundle unpasteable.
    out = build_diagnostics(conn, Config(), log_text="", camera_indices=())
    assert "base64" not in out.lower()
    assert "\xff\xd8" not in out


def test_the_bundle_survives_an_empty_database(conn):
    # The most likely time to ask for diagnostics is before anything works.
    out = build_diagnostics(conn, Config(), log_text="", camera_indices=())
    assert "no baseline" in out.lower()


def test_the_bundle_names_the_configured_model(conn):
    out = build_diagnostics(conn, Config(), log_text="", camera_indices=())
    assert Config().gemini_model in out


def test_the_bundle_reports_both_halves_of_the_call_volume(conn):
    """The sample rate alone never explained the gated calls.

    That is how an uncapped coaching path went unnoticed: the one Gemini number
    in the bundle described the path that was NOT driving the volume. Both
    bounds and the day's spend belong here, so "why is it calling so much" is
    answerable from a pasted bundle.
    """
    from datetime import date

    from posture.store.queries import CheckRecord, insert_check
    from posture.types import Outcome, Rating

    config = Config(comparison_sample_rate=0.15, max_coaching_calls_per_day=7)
    today = date.today().isoformat()
    insert_check(conn, CheckRecord(
        ts=f"{today}T10:00:00", date=today, outcome=Outcome.OK, baseline_id=None,
        cameras=["front"], metrics=PostureMetrics(), local_rating=Rating.POOR,
        local_driver=None, gemini_rating=None, gemini_issue=None,
        gemini_details=None, gemini_tip=None, gemini_model="gemini-3.5-flash-lite",
        is_comparison_sample=False, api_error=None))

    text = build_diagnostics(conn, config, "", ())
    assert "comparison sample rate: 0.15" in text
    assert "coaching budget/day: 7" in text
    assert "coaching calls today: 1" in text
