"""PostureApp launch behaviour that does not require a live AppKit run loop.

PoseDetector needs a real model file on disk and GeminiClient needs a real API
key, so both are patched at their posture.ui.app import site. rumps.App itself
constructs fine without a running NSApplication, which is what makes these
tests possible at all.
"""
from __future__ import annotations

import sqlite3
import threading
import time
import tomllib
from unittest.mock import MagicMock, patch

import pytest

from posture.config import Config
from posture.store.queries import CheckRecord
from posture.types import Outcome


def build_config(tmp_path):
    return Config(
        camera_indices={"front": 0},
        gemini_enabled=True,
        db_path=tmp_path / "app.db",
        model_path=tmp_path / "model.task",
        log_path=tmp_path / "app.log",
    )


@pytest.fixture
def patched_deps():
    with patch("posture.ui.app.PoseDetector") as fake_detector_cls, \
         patch("posture.ui.app.GeminiClient") as fake_gemini_cls, \
         patch("posture.ui.app.notify") as fake_notify, \
         patch("posture.ui.app.AppHelper.callAfter",
               side_effect=lambda fn, *a: fn(*a)) as fake_callafter:
        detector_ctx = MagicMock()
        detector_ctx.__enter__.return_value = MagicMock()
        fake_detector_cls.return_value = detector_ctx
        yield fake_detector_cls, fake_gemini_cls, fake_notify, fake_callafter


@pytest.fixture
def startup_channels():
    """Patch every channel main() reports a startup failure through.

    rumps.alert would put a real modal dialog on the developer's screen and
    block the suite until they dismissed it, and setup_logging would write into
    their real ~/.posture_monitor. Both are patched at their posture.ui.app
    import sites.
    """
    with patch("posture.ui.app.notify") as fake_notify, \
         patch("posture.ui.app.rumps.alert") as fake_alert, \
         patch("posture.ui.app.setup_logging") as fake_setup_logging:
        yield fake_notify, fake_alert, fake_setup_logging


def _alert_text(fake_alert) -> str:
    """Only the alert's own words. Asserting over a whole call object would let
    an unrelated repr (the tmp_path in a kwarg, say) satisfy a substring check.
    """
    kwargs = fake_alert.call_args.kwargs
    return f"{kwargs['title']}\n{kwargs['message']}"


def test_a_malformed_config_file_is_reported_and_does_not_crash_loop(
        tmp_path, startup_channels):
    """A TOML syntax error must not become an invisible relaunch every minute.

    install_autostart.sh sets KeepAlive{SuccessfulExit: false} with
    ThrottleInterval 60, so any non-zero exit before the menu bar icon exists is
    a silent crash loop. main() must survive the raise, tell the user, and
    return normally.
    """
    from posture.ui import app as app_module

    fake_notify, fake_alert, _ = startup_channels
    boom = tomllib.TOMLDecodeError("Expected '=' after a key (at line 3)")
    with patch("posture.ui.app.Config.load", side_effect=boom), \
         patch("posture.ui.app.PostureApp") as fake_app_cls:
        app_module.main()

    fake_app_cls.assert_not_called()
    assert "Expected '=' after a key" in _alert_text(fake_alert)
    assert fake_notify.call_args[0][0] == "Stack could not read its settings"


def test_a_typod_camera_role_is_reported_and_does_not_crash_loop(
        tmp_path, startup_channels):
    """Config.load raises ValueError for an unknown role; main must not die on it."""
    from posture.ui import app as app_module

    fake_notify, fake_alert, _ = startup_channels
    boom = ValueError("unknown camera role(s): ['sied']. Valid roles are "
                      "['front', 'side'].")
    with patch("posture.ui.app.Config.load", side_effect=boom), \
         patch("posture.ui.app.PostureApp") as fake_app_cls:
        app_module.main()

    fake_app_cls.assert_not_called()
    assert "unknown camera role(s): ['sied']" in _alert_text(fake_alert)
    assert fake_notify.called


def test_a_corrupt_database_is_reported_and_does_not_crash_loop(
        tmp_path, startup_channels):
    """The real failure, not a stand-in: migrate() on a file that is not a database.

    connect() succeeds (sqlite opens lazily); migrate() raises
    sqlite3.DatabaseError from inside PostureApp.__init__, before the menu bar
    icon exists.
    """
    from posture.ui import app as app_module

    fake_notify, fake_alert, _ = startup_channels
    config = build_config(tmp_path)
    config.model_path.write_bytes(b"not really a model, but it exists")
    config.db_path.write_bytes(b"this is definitely not a sqlite database")

    with patch("posture.ui.app.Config.load", return_value=config):
        # Proves the guard is what saved us: unguarded, this same call raises.
        with pytest.raises(sqlite3.DatabaseError):
            app_module.PostureApp(config)
        app_module.main()

    assert "file is not a database" in _alert_text(fake_alert)
    assert fake_notify.call_args[0][0] == "Stack could not start"


def test_gemini_preflight_does_not_block_app_launch(tmp_path, patched_deps):
    _, fake_gemini_cls, _, _ = patched_deps
    gate = threading.Event()
    gemini_instance = MagicMock()

    def blocking_preflight():
        gate.wait(timeout=5)
        return None

    gemini_instance.preflight.side_effect = blocking_preflight
    fake_gemini_cls.return_value = gemini_instance

    from posture.ui.app import PostureApp

    started = time.monotonic()
    app = PostureApp(build_config(tmp_path))
    elapsed = time.monotonic() - started

    # __init__ must return long before the gate opens: a synchronous preflight
    # call would block here for up to 5 seconds instead.
    assert elapsed < 1.0
    assert app.status_item.title == "Status: Idle"

    gate.set()
    for _ in range(200):
        if gemini_instance.preflight.called:
            break
        time.sleep(0.01)
    gemini_instance.preflight.assert_called_once()


def test_gemini_preflight_failure_is_visible_in_the_banner_and_the_menu(
        tmp_path, patched_deps):
    """A notification alone is not a signal the user can rely on.

    The banner used to be deliberately left alone here, on the stated grounds
    that _apply_status would overwrite it anyway. That is false: _apply_status
    runs only after a check completes, and __init__ never starts monitoring. So
    a preflight failure at launch left the menu reading "Status: Idle" with
    nothing but a notification macOS is free to suppress.
    """
    _, fake_gemini_cls, fake_notify, _ = patched_deps
    gemini_instance = MagicMock()
    gemini_instance.preflight.return_value = Outcome.API_ERROR
    fake_gemini_cls.return_value = gemini_instance

    from posture.ui.app import PostureApp

    app = PostureApp(build_config(tmp_path))

    for _ in range(200):
        if fake_notify.called:
            break
        time.sleep(0.01)

    fake_notify.assert_called_once()
    title, body = fake_notify.call_args[0]
    assert title == "Gemini unavailable"
    assert "API_ERROR" in body
    assert app.status_item.title == "Status: Gemini unavailable (API_ERROR)"
    assert app.gemini_item.title == (
        "Gemini: NOT working (API_ERROR). Local monitoring continues.")


def test_a_successful_preflight_leaves_the_banner_alone_and_says_so(
        tmp_path, patched_deps):
    """Positive control for the banner change: success must NOT set it.

    Without this, an implementation that stamped the banner on every preflight
    result would satisfy the failure test above.
    """
    _, fake_gemini_cls, fake_notify, _ = patched_deps
    gemini_instance = MagicMock()
    gemini_instance.preflight.return_value = None
    gemini_instance.breaker.is_open.return_value = False
    fake_gemini_cls.return_value = gemini_instance

    from posture.ui.app import PostureApp

    app = PostureApp(build_config(tmp_path))
    for _ in range(200):
        if app.gemini_item.title != "Gemini: checking...":
            break
        time.sleep(0.01)

    assert app.status_item.title == "Status: Idle"
    assert app.gemini_item.title == "Gemini: ok"
    fake_notify.assert_not_called()


def test_a_gemini_failure_after_launch_stays_on_the_menu(tmp_path, patched_deps):
    """The closest thing left to v2's defining failure.

    The agent has been up since Monday, the model is retired on Thursday, the
    breaker trips permanently, and the comparison set silently stops growing.
    The failure is recorded on exactly ONE row, because after that the breaker
    refuses every call and no further api_error is ever written. So a line that
    only read the newest check would announce the failure once and then go back
    to saying everything was fine, all while the menu bar read "Good posture".
    """
    from posture.store.queries import CheckRecord as _CheckRecord, insert_check
    from posture.ui.app import PostureApp

    _, fake_gemini_cls, _, _ = patched_deps
    gemini_instance = MagicMock()
    gemini_instance.preflight.return_value = None
    gemini_instance.breaker.is_open.return_value = False
    fake_gemini_cls.return_value = gemini_instance

    app = PostureApp(build_config(tmp_path))

    def store(**overrides):
        insert_check(app.conn, _CheckRecord(**{
            **dict(ts="2026-07-30T12:00:00", date="2026-07-30", outcome=Outcome.OK,
                   baseline_id=None, cameras=["front"], metrics=None,
                   local_rating=None, local_driver=None, gemini_rating=None,
                   gemini_issue=None, gemini_details=None, gemini_tip=None,
                   gemini_model=None, is_comparison_sample=False, api_error=None),
            **overrides}))

    store(api_error=Outcome.MODEL_UNAVAILABLE.value)
    app._apply_status("Good posture")
    assert app.gemini_item.title == (
        "Gemini: NOT working (MODEL_UNAVAILABLE). Local monitoring continues.")

    # Two further checks, both perfectly normal, because the breaker is refusing
    # to call at all. The warning must survive them.
    store()
    app._apply_status("Good posture")
    store()
    app._apply_status("Good posture")
    assert app.gemini_item.title == (
        "Gemini: NOT working (MODEL_UNAVAILABLE). Local monitoring continues.")

    # And it must clear once a verdict actually comes back, or it is just noise.
    store(gemini_model="gemini-3.5-flash-lite")
    app._apply_status("Good posture")
    assert app.gemini_item.title == "Gemini: ok"


def test_the_gemini_line_says_off_when_gemini_is_disabled(tmp_path, patched_deps):
    """Disabled is a THIRD state, distinct from working and from broken.

    _apply_status is driven here on purpose. Asserting on the freshly
    constructed title alone would pass off the initial MenuItem string and never
    reach the refresh that runs on every check.
    """
    import dataclasses

    from posture.ui.app import PostureApp

    app = PostureApp(dataclasses.replace(build_config(tmp_path), gemini_enabled=False))
    assert app.gemini is None
    app._apply_status("Good posture")
    assert app.gemini_item.title == "Gemini: off"


def test_the_measuring_line_stops_promising_lean_without_a_facing_sign(
        tmp_path, patched_deps):
    """H6b drops the trunk angle from the baseline when the sign is ambiguous.

    The camera-derived line went on claiming lean anyway, forever, contradicting
    the baseline sitting one menu item above it.
    """
    from posture.store.queries import CheckRecord as _CheckRecord, insert_check, save_baseline
    from posture.types import PostureMetrics
    from posture.ui.app import PostureApp

    _, fake_gemini_cls, _, _ = patched_deps
    fake_gemini_cls.return_value = MagicMock(**{"preflight.return_value": None})

    app = PostureApp(build_config(tmp_path))
    save_baseline(app.conn, PostureMetrics(cva_deg=60.0, shoulder_tilt_deg=0.0),
                  "ambiguous facing", facing_sign=None)
    insert_check(app.conn, _CheckRecord(
        ts="2026-07-30T12:00:00", date="2026-07-30", outcome=Outcome.OK,
        baseline_id=1, cameras=["front", "side"],
        # A real front+side check carries front metrics and a craniovertebral
        # angle. metrics=None described a row the monitor cannot write, and
        # capability now names the roles a row actually holds a measurement
        # from, so the fixture has to hold them for the label to be about
        # anything. trunk_angle_deg stays absent, which is the point of this
        # test: the facing sign was never frozen, so lean is not measured.
        metrics=PostureMetrics(shoulder_tilt_deg=2.0, cva_deg=58.0),
        local_rating=None, local_driver=None, gemini_rating=None,
        gemini_issue=None, gemini_details=None, gemini_tip=None,
        gemini_model=None, is_comparison_sample=False, api_error=None))

    app._apply_status("Good posture")
    assert app.measuring_item.title == (
        "Measuring: posture and forward head angle (lean not calibrated)")

    # Positive control: recalibrating with a frozen sign brings lean back, so
    # the line is tracking the sign and not merely refusing to say "lean".
    save_baseline(app.conn, PostureMetrics(cva_deg=60.0, trunk_angle_deg=0.0),
                  "calibrated", facing_sign=1.0)
    app._apply_status("Good posture")
    assert app.measuring_item.title == "Measuring: posture, forward head, and lean"


def a_check_record(**overrides):
    fields = dict(
        ts="2026-07-30T12:00:00",
        date="2026-07-30",
        outcome=Outcome.OK,
        baseline_id=1,
        cameras=["front"],
        metrics=None,
        local_rating=None,
        local_driver=None,
        gemini_rating=None,
        gemini_issue=None,
        gemini_details=None,
        gemini_tip=None,
        gemini_model=None,
        is_comparison_sample=False,
        api_error=None,
    )
    fields.update(overrides)
    return CheckRecord(**fields)


def test_show_nudge_tolerates_a_missing_local_rating(tmp_path, patched_deps):
    # The amendment guards record.local_rating being None (v2's equivalent
    # line raised AttributeError inside the nudge callback for exactly this
    # case). Patched at posture.ui.app.show_popup, its import site: patching
    # posture.ui.popup.show_popup instead would leave app.py's own bound
    # reference untouched and this test would pass without exercising
    # anything.
    from posture.ui.app import PostureApp

    app = PostureApp(build_config(tmp_path))
    record = a_check_record(local_rating=None, local_driver=None,
                             gemini_details=None, gemini_tip=None)

    with patch("posture.ui.app.show_popup") as fake_show_popup:
        app._show_nudge(record)

    fake_show_popup.assert_called_once()
    rating, headline, details, tip = fake_show_popup.call_args[0]
    assert rating is None
    # NOT the POOR headline. The old code fell back to the word "poor" whenever
    # the rating was missing, so an unmeasurable check was announced as poor
    # posture; the headline table keys None separately for exactly that reason.
    assert headline == "Nothing measured"
    assert details == "Driven by posture."
    assert tip == "Feet flat, hips back, ears over shoulders."
    assert fake_show_popup.call_args.kwargs["duration"] == 12.0


def test_gemini_coaching_reaches_the_popup_without_hiding_the_local_driver(
        tmp_path, patched_deps):
    # Local measurement is the primary signal, so its driver is always stated;
    # Gemini's text is additive and labelled, never a silent replacement.
    from posture.ui.app import PostureApp

    app = PostureApp(build_config(tmp_path))
    record = a_check_record(
        local_rating=None, local_driver="head_tilt_deg",
        gemini_details="Shoulders are rolled forward.",
        gemini_tip="Roll them back.")

    with patch("posture.ui.app.show_popup") as fake_show_popup:
        app._show_nudge(record)

    _, _, details, tip = fake_show_popup.call_args[0]
    assert details == "Driven by head tilt deg. Gemini: Shoulders are rolled forward."
    assert tip == "Roll them back."


def test_a_gated_calls_tip_reaches_show_popup_end_to_end(tmp_path, patched_deps):
    """Monitor -> _on_nudge -> _show_nudge -> show_popup, with nothing faked in between.

    The two monitor-level tests pin the record; this one pins the seam, because
    the defect was precisely that the record reaching the popup had every Gemini
    field nulled and the popup silently fell back to its canned line.
    """
    import dataclasses

    import numpy as np

    from posture.ui.app import PostureApp
    from posture.gemini import GeminiVerdict
    from posture.monitor import SlouchTracker
    from posture.scoring import LocalAssessment
    from posture.store.queries import save_baseline
    from posture.types import PostureMetrics, Rating
    from tests.synthetic import upright_front

    _, fake_gemini_cls, _, _ = patched_deps
    client = MagicMock()
    client.available = True
    client.preflight.return_value = None
    client.breaker.is_open.return_value = False
    client.assess.return_value = (
        GeminiVerdict(True, Rating.POOR, "forward head",
                      "Head is well ahead of the shoulders.",
                      "Tuck your chin and sit back."),
        None,
    )
    fake_gemini_cls.return_value = client

    config = dataclasses.replace(
        build_config(tmp_path), comparison_sample_rate=0.0,
        min_seconds_between_api_calls=0)
    app = PostureApp(config)
    save_baseline(app.conn, PostureMetrics(cva_deg=60.0, shoulder_tilt_deg=0.0), "t")
    app.monitor.detector.detect.return_value = upright_front()
    app.monitor._tracker = SlouchTracker(sustained_samples=2, cooldown_s=600)

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    poor = LocalAssessment(Rating.POOR, "cva_deg", {"cva_deg": -12.0})
    with patch("posture.ui.app.show_popup") as fake_show_popup, \
         patch("posture.monitor.score", return_value=poor), \
         patch("posture.cameras.capture_roles", return_value=({"front": frame}, None)):
        app.monitor.run_once()   # gated call happens on this flush
        app.monitor.run_once()   # sustained: the nudge fires

    client.assess.assert_called_once()
    fake_show_popup.assert_called_once()
    _, headline, details, tip = fake_show_popup.call_args[0]
    assert headline == "Time to reset"
    assert tip == "Tuck your chin and sit back."
    assert "Head is well ahead of the shoulders." in details


def test_the_nudge_headlines_describe_the_body_and_never_scold():
    """Pins the whole table, not just the two headlines the tests above reach.

    "Sustained poor posture" was assembled from Rating.value, so any future edit
    that reaches for the rating's own name again would reintroduce a verdict on
    the person. Someone reads this panel many times a day while in pain.
    Every headline here names a position or a next action; none of them
    contains "poor", and none of them is a judgement.
    """
    from posture.ui.app import NUDGE_HEADLINES

    assert NUDGE_HEADLINES == {
        "GOOD": "Nicely stacked",
        "DECENT": "Starting to drift",
        "POOR": "Time to reset",
        None: "Nothing measured",
    }


def test_an_unrecognised_rating_gets_the_no_reading_headline(tmp_path, patched_deps):
    """The .get() fallback, which the None key alone does not exercise.

    A rating this table has never heard of must land on "Nothing measured"
    rather than raising inside the nudge callback, which is the one place in
    this app an exception is invisible to the user.
    """
    from posture.ui.app import PostureApp

    app = PostureApp(build_config(tmp_path))

    class UnknownRating:
        value = "SPLENDID"

    record = a_check_record(local_rating=UnknownRating())
    with patch("posture.ui.app.show_popup") as fake_show_popup:
        app._show_nudge(record)

    assert fake_show_popup.call_args[0][1] == "Nothing measured"
