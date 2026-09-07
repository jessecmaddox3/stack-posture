import dataclasses
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from posture.config import Config
from posture.monitor import Monitor, SlouchTracker
from posture.scoring import LocalAssessment
from posture.store.db import connect, migrate
from posture.store.queries import active_baseline, recent_checks, save_baseline
from posture.types import Outcome, PostureMetrics, Rating
from tests.synthetic import slouched_side, upright_front


# --- SlouchTracker: pure, fake clock ---

def test_tracker_does_not_fire_on_a_single_bad_sample():
    t = SlouchTracker(sustained_samples=3, cooldown_s=600)
    assert t.update(Rating.POOR, now=0.0) is False


def test_tracker_fires_on_the_third_consecutive_bad_sample():
    t = SlouchTracker(sustained_samples=3, cooldown_s=600)
    assert t.update(Rating.POOR, now=0.0) is False
    assert t.update(Rating.POOR, now=30.0) is False
    assert t.update(Rating.POOR, now=60.0) is True


def test_a_good_sample_resets_the_streak():
    t = SlouchTracker(sustained_samples=3, cooldown_s=600)
    t.update(Rating.POOR, now=0.0)
    t.update(Rating.POOR, now=30.0)
    t.update(Rating.GOOD, now=60.0)
    assert t.update(Rating.POOR, now=90.0) is False


def test_decent_counts_toward_the_streak():
    t = SlouchTracker(sustained_samples=2, cooldown_s=600)
    t.update(Rating.DECENT, now=0.0)
    assert t.update(Rating.DECENT, now=30.0) is True


def test_cooldown_suppresses_a_second_nudge():
    t = SlouchTracker(sustained_samples=1, cooldown_s=600)
    assert t.update(Rating.POOR, now=0.0) is True
    assert t.update(Rating.POOR, now=300.0) is False


def test_nudge_resumes_after_the_cooldown_expires():
    t = SlouchTracker(sustained_samples=1, cooldown_s=600)
    t.update(Rating.POOR, now=0.0)
    assert t.update(Rating.POOR, now=601.0) is True


def test_unmeasurable_rating_neither_advances_nor_resets_the_streak():
    """Unmeasurable is not evidence either way.

    It must not count toward the streak, and it must not wipe a genuine one
    either. Glancing away mid-slouch should delay the nudge by one sample, not
    restart the clock. Both halves are pinned here: if it advanced, the third
    call would fire; if it reset, the fourth would not.
    """
    t = SlouchTracker(sustained_samples=3, cooldown_s=600)
    t.update(Rating.POOR, now=0.0)
    t.update(None, now=30.0)
    assert t.update(Rating.POOR, now=60.0) is False
    assert t.update(Rating.POOR, now=90.0) is True


def test_absence_resets_the_streak():
    t = SlouchTracker(sustained_samples=2, cooldown_s=600)
    t.update(Rating.POOR, now=0.0)
    t.reset()
    assert t.update(Rating.POOR, now=30.0) is False


# --- Monitor: integration with fakes ---

@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "m.db")
    migrate(c)
    save_baseline(c, PostureMetrics(cva_deg=60.0, shoulder_tilt_deg=0.0), "test")
    yield c
    c.close()


def build_monitor(conn, config=None, frames=None, landmarks="upright",
                  record_interval_s=None):
    cfg = config or Config(camera_indices={"front": 0}, gemini_enabled=False)
    if record_interval_s is not None:
        cfg = dataclasses.replace(cfg, record_interval_s=record_interval_s)
    detector = MagicMock()
    detector.detect.return_value = upright_front() if landmarks == "upright" else None
    m = Monitor(config=cfg, detector=detector, conn=conn,
                on_nudge=MagicMock(), on_status=MagicMock())
    return m, detector


def test_run_once_records_person_absent_when_no_landmarks(conn):
    m, _ = build_monitor(conn, landmarks=None)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    with patch("posture.cameras.capture_roles", return_value=({"front": frame}, None)):
        record = m.run_once()
    assert record.outcome is Outcome.PERSON_ABSENT
    assert record.local_rating is None


def test_run_once_records_camera_busy(conn):
    m, _ = build_monitor(conn)
    with patch("posture.cameras.capture_roles", return_value=({}, Outcome.CAMERA_BUSY)):
        record = m.run_once()
    assert record.outcome is Outcome.CAMERA_BUSY


def test_run_once_scores_and_persists_an_ok_check(conn):
    m, _ = build_monitor(conn)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    with patch("posture.cameras.capture_roles", return_value=({"front": frame}, None)):
        record = m.run_once()
    assert record.outcome is Outcome.OK
    assert record.local_rating is Rating.GOOD
    assert recent_checks(conn, 1)[0]["outcome"] == "OK"


def test_run_once_records_which_cameras_contributed(conn):
    m, _ = build_monitor(conn, config=Config(camera_indices={"front": 0, "side": 2},
                                             gemini_enabled=False))
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    with patch("posture.cameras.capture_roles",
               return_value=({"front": frame, "side": frame}, None)):
        record = m.run_once()
    assert set(record.cameras) == {"front", "side"}


def test_capability_reflects_the_cameras_that_worked_not_the_ones_configured(conn):
    """Capability exists precisely for the case where a configured camera fails.

    Deriving it from config would agree with reality in every other test, because
    they all have every configured camera working, and would be wrong exactly here.
    Both cameras are configured; only the front produces a frame.
    """
    m, _ = build_monitor(conn, config=Config(camera_indices={"front": 0, "side": 2},
                                             gemini_enabled=False))
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    with patch("posture.cameras.capture_roles", return_value=({"front": frame}, None)):
        m.run_once()

    row = recent_checks(conn, 1)[0]
    assert row["capability"] == "front", "capability must come from actual frames"
    assert set(row["cameras"]) == {"front"}


def test_sample_uses_the_frozen_facing_sign_over_the_frame_estimate(conn):
    """The calibrated sign must win over what this frame happens to estimate.

    This is the test that actually discriminates the frozen sign from the
    per-frame fallback, which no single-frame test in test_baseline.py can do:
    within one frame both paths call facing_sign_from on the same landmarks and
    agree. Here the baseline says facing is -1 while the frame's own estimate is
    +1, which is exactly the head-turn case the freeze exists for. Frozen gives
    -20, the fallback would give +20, so the sign of the result identifies which
    path ran.
    """
    save_baseline(conn, PostureMetrics(trunk_angle_deg=0.0), "mirrored rig",
                  facing_sign=-1.0)
    m, detector = build_monitor(conn)
    detector.detect.return_value = slouched_side(20.0)   # own estimate is +1
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    with patch("posture.cameras.capture_roles", return_value=({"side": frame}, None)):
        record = m.run_once()
    assert record.metrics.trunk_angle_deg == pytest.approx(-20.0, abs=1.5)


def test_sample_omits_trunk_angle_when_no_sign_was_frozen(conn):
    """Monitoring must fail closed, exercised through the real Monitor.

    This is the wiring the defect report named: with no frozen sign, _sample must
    pass require_calibrated=True so the per-frame estimate is not silently used.
    test_sample_uses_the_frozen_facing_sign_over_the_frame_estimate cannot detect
    this, because its baseline HAS a sign, so the check short-circuits before
    require_calibrated is ever consulted.
    """
    save_baseline(conn, PostureMetrics(cva_deg=60.0), "no facing", facing_sign=None)
    m, detector = build_monitor(conn)
    detector.detect.return_value = slouched_side(20.0)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    with patch("posture.cameras.capture_roles", return_value=({"side": frame}, None)):
        record = m.run_once()

    assert record.metrics.trunk_angle_deg is None, "monitoring guessed the sign"
    # CVA does not depend on the facing sign, so it must still be measured.
    assert record.metrics.cva_deg is not None


def test_run_once_without_a_baseline_still_records_the_check(tmp_path):
    c = connect(tmp_path / "nobase.db")
    migrate(c)
    assert active_baseline(c) is None
    m, _ = build_monitor(c)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    with patch("posture.cameras.capture_roles", return_value=({"front": frame}, None)):
        record = m.run_once()
    # Uncalibrated: metrics are captured, but nothing is scored.
    assert record.outcome is Outcome.OK
    assert record.local_rating is None
    assert record.metrics is not None
    c.close()


def test_concurrent_checks_are_serialised(conn):
    """Four threads must never overlap inside a check.

    Every assertion here runs in the MAIN thread. An earlier version asserted
    inside the worker threads, where Python's excepthook downgrades the failure to
    a PytestUnhandledThreadExceptionWarning that pytest does not fail on. That
    version reported "1 passed" even with the lock deleted, so it pinned nothing.
    """
    # record_interval_s=0: window aggregation would otherwise buffer threads
    # 2 through 4 into the first row, collapsing this count to 1 regardless of
    # whether serialisation actually held. Writing every sample keeps the
    # count a faithful proxy for "four checks really ran".
    m, _ = build_monitor(conn, record_interval_s=0)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    concurrent, overlaps, errors = [], [], []

    original = m._perform_check

    def slow(*args, **kwargs):
        concurrent.append(1)
        if len(concurrent) > 1:
            overlaps.append(len(concurrent))
        try:
            time.sleep(0.02)  # widen the window so a missing lock actually races
            return original(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
            raise
        finally:
            concurrent.pop()

    with patch("posture.cameras.capture_roles", return_value=({"front": frame}, None)):
        with patch.object(m, "_perform_check", side_effect=slow):
            threads = [threading.Thread(target=m.run_once) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

    assert errors == [], f"checks raised: {errors}"
    assert overlaps == [], f"checks overlapped, concurrency reached {overlaps}"
    # Four threads that all failed serially would also produce no overlaps, so
    # confirm they did real work.
    assert len(recent_checks(conn, 10)) == 4


def test_stop_is_prompt_and_joins(conn):
    """The sleep must be interruptible, and the worker must actually exit.

    The interval is an hour. An earlier version asserted only is_running, which
    stop() sets unconditionally at its top, so dropping wake.set() left the test
    passing and merely 20x slower. Timing and thread death are what pin it.
    """
    m, _ = build_monitor(conn, config=Config(sample_interval_s=3600,
                                             camera_indices={"front": 0},
                                             gemini_enabled=False))
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    with patch("posture.cameras.capture_roles", return_value=({"front": frame}, None)):
        m.start()
        time.sleep(0.05)  # let the first check finish and enter the interval sleep
        thread = m._thread
        began = time.monotonic()
        m.stop(timeout=5.0)
        elapsed = time.monotonic() - began

    assert m.is_running is False
    assert elapsed < 1.0, f"stop took {elapsed:.2f}s, so the sleep is not interruptible"
    assert thread is not None and not thread.is_alive()
    assert m._thread is None


def test_start_refuses_while_a_previous_thread_is_still_alive(conn):
    """A timed-out stop must not allow a second loop.

    join() neither raises nor returns a status on timeout, so clearing _thread
    unconditionally would let this start() spawn a second worker while the first
    was still blocked, and the stuck one would resume looping once it saw
    _running set True again.
    """
    m, _ = build_monitor(conn)
    release = threading.Event()

    def block(*_args, **_kwargs):
        release.wait(timeout=5.0)
        return ({}, Outcome.CAMERA_BUSY)

    with patch("posture.cameras.capture_roles", side_effect=block):
        m.start()
        time.sleep(0.05)
        m.stop(timeout=0.1)  # times out: the check is blocked in capture
        stuck = m._thread
        assert stuck is not None and stuck.is_alive()

        m.start()
        assert m._thread is stuck, "start() spawned a second loop over a live thread"

        release.set()
        m.stop(timeout=5.0)

    assert m._thread is None


def test_stop_reports_whether_the_worker_actually_exited(conn):
    """Teardown callers must be able to tell.

    quit_app closes the MediaPipe detector and the database after stopping. If a
    check is wedged in camera I/O, doing that races the worker into a native
    binding, so the caller needs to know the join failed.
    """
    m, _ = build_monitor(conn)
    release = threading.Event()

    def block(*_args, **_kwargs):
        release.wait(timeout=5.0)
        return ({}, Outcome.CAMERA_BUSY)

    with patch("posture.cameras.capture_roles", side_effect=block):
        m.start()
        time.sleep(0.05)
        assert m.stop(timeout=0.1) is False, "a timed-out join must report False"
        release.set()
        assert m.stop(timeout=5.0) is True


def test_stop_on_a_never_started_monitor_reports_true(conn):
    m, _ = build_monitor(conn)
    assert m.stop(timeout=1.0) is True


def test_start_reports_whether_monitoring_is_active(conn):
    """The caller must be able to tell a refusal from a success.

    Returning None would let the menu bar flip to "Monitoring" while nothing
    runs. Already-running returns True because the caller wants to know the
    state, not who caused it.
    """
    m, _ = build_monitor(conn, config=Config(sample_interval_s=3600,
                                             camera_indices={"front": 0},
                                             gemini_enabled=False))
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    with patch("posture.cameras.capture_roles", return_value=({"front": frame}, None)):
        assert m.start() is True
        assert m.start() is True  # already running, monitoring is still active
        m.stop(timeout=5.0)


def test_start_reports_false_when_it_refuses_over_a_live_thread(conn):
    m, _ = build_monitor(conn)
    release = threading.Event()

    def block(*_args, **_kwargs):
        release.wait(timeout=5.0)
        return ({}, Outcome.CAMERA_BUSY)

    with patch("posture.cameras.capture_roles", side_effect=block):
        m.start()
        time.sleep(0.05)
        m.stop(timeout=0.1)
        assert m._thread is not None and m._thread.is_alive()
        assert m.start() is False, "refusal must be visible to the caller"
        release.set()
        m.stop(timeout=5.0)


def test_nudge_fires_through_the_monitor_after_sustained_bad_posture(conn):
    """The debounce wired end to end, not just SlouchTracker in isolation."""
    m, _ = build_monitor(conn)
    m._tracker = SlouchTracker(sustained_samples=2, cooldown_s=600)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    bad = LocalAssessment(Rating.POOR, "cva_deg", {"cva_deg": -12.0})

    with patch("posture.monitor.score", return_value=bad):
        with patch("posture.cameras.capture_roles",
                   return_value=({"front": frame}, None)):
            m.run_once()
            assert m.on_nudge.call_count == 0, "fired on the first bad sample"
            m.run_once()
            assert m.on_nudge.call_count == 1


def test_a_busy_camera_resets_the_nudge_streak(conn):
    """A non-OK outcome must clear the streak.

    Otherwise a slouch interrupted by another app grabbing the camera resumes
    mid-count and nudges early, on less evidence than the threshold requires.
    """
    m, _ = build_monitor(conn)
    m._tracker = SlouchTracker(sustained_samples=2, cooldown_s=600)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    bad = LocalAssessment(Rating.POOR, "cva_deg", {"cva_deg": -12.0})

    with patch("posture.monitor.score", return_value=bad):
        with patch("posture.cameras.capture_roles",
                   return_value=({"front": frame}, None)):
            m.run_once()  # streak 1
        with patch("posture.cameras.capture_roles",
                   return_value=({}, Outcome.CAMERA_BUSY)):
            m.run_once()  # must reset
        with patch("posture.cameras.capture_roles",
                   return_value=({"front": frame}, None)):
            m.run_once()  # streak 1 again, not 2

    assert m.on_nudge.call_count == 0


def test_stop_waits_for_a_manual_check_too(conn):
    """quit_app tears down the detector and the database after stop() returns.

    A manual check running outside the scheduled loop must therefore be tracked,
    or teardown races it into the native MediaPipe binding.
    """
    m, _ = build_monitor(conn)
    release = threading.Event()
    entered = threading.Event()

    def block(*_args, **_kwargs):
        entered.set()
        release.wait(timeout=5.0)
        return ({}, Outcome.CAMERA_BUSY)

    with patch("posture.cameras.capture_roles", side_effect=block):
        m.run_once_async()
        assert entered.wait(timeout=2.0), "manual check never started"
        assert m.stop(timeout=0.1) is False, "stop must report a live manual check"
        release.set()
        assert m.stop(timeout=5.0) is True


def test_a_failing_manual_check_is_logged_and_surfaced(conn):
    """A manual check that raises must not vanish.

    The scheduled loop logs its own failures. Without the guard in
    run_once_async, a failed manual check reaches only stderr, which for a login
    item nobody reads. Nothing exercised that branch before.
    """
    m, _ = build_monitor(conn)
    boom = RuntimeError("camera exploded")
    with patch("posture.cameras.capture_roles", side_effect=boom):
        m.run_once_async()
        assert m.stop(timeout=5.0) is True

    # The failure reached the status callback rather than disappearing.
    said = [c.args[0] for c in m.on_status.call_args_list]
    assert any("failed" in str(s).lower() for s in said), said
    # And the worker still cleaned itself up despite raising.
    assert m._workers == set()


def test_manual_checks_do_not_accumulate_threads(conn):
    m, _ = build_monitor(conn)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    with patch("posture.cameras.capture_roles", return_value=({"front": frame}, None)):
        for _ in range(5):
            m.run_once_async()
        m.stop(timeout=5.0)
    assert m._workers == set()


def test_gemini_preflight_runs_on_a_tracked_worker_without_blocking_the_caller(conn):
    """A blocking network call must not run on the caller's thread (AppKit's,

    at launch), and an untracked thread would still be running when quit_app
    tears down shared resources with no way for stop() to know to wait.
    """
    m, _ = build_monitor(conn)
    m.gemini = MagicMock()
    gate = threading.Event()
    m.gemini.preflight.side_effect = lambda: gate.wait(timeout=5.0)

    results = []
    m.run_gemini_preflight_async(results.append)

    # The call must return immediately rather than block on the gate: a
    # synchronous preflight would still be sitting inside gate.wait() here.
    assert m._workers, "the preflight worker should be tracked while it runs"
    gate.set()
    # stop() must be able to join it, exactly like a manual check.
    assert m.stop(timeout=5.0) is True
    assert m._workers == set()


def test_the_dashboard_worker_is_tracked_so_stop_waits_for_it(conn):
    # An untracked dashboard thread would still be reading self.conn while
    # quit_app closes it.
    monitor, _ = build_monitor(conn)
    started, may_finish = threading.Event(), threading.Event()

    def slow_open(_conn, _path):
        started.set()
        may_finish.wait(5.0)

    with patch("posture.dashboard.generate.open_dashboard", slow_open):
        monitor.run_dashboard_async(Path("/tmp/does-not-matter.html"))
        assert started.wait(5.0)
        with monitor._workers_lock:
            assert len(monitor._workers) == 1
        may_finish.set()
    assert monitor.stop(timeout=5.0)
    with monitor._workers_lock:
        assert not monitor._workers


def test_the_diagnostics_worker_is_tracked_so_stop_waits_for_it(conn):
    # An untracked diagnostics thread would still be reading self.conn (and
    # probing cameras) while quit_app closes the connection.
    monitor, _ = build_monitor(conn)
    started, may_finish = threading.Event(), threading.Event()

    def slow_discover(max_index=8):
        started.set()
        may_finish.wait(5.0)
        return ()

    with patch("posture.cameras.discover", slow_discover):
        monitor.run_diagnostics_async(MagicMock())
        assert started.wait(5.0)
        with monitor._workers_lock:
            assert len(monitor._workers) == 1
        may_finish.set()
    assert monitor.stop(timeout=5.0)
    with monitor._workers_lock:
        assert not monitor._workers


def test_a_failing_check_reaches_the_menu_bar_and_does_not_leave_a_stale_reading(conn):
    """Disk fills at 3pm; the menu must not still read "Good posture" all evening.

    The scheduled loop caught and logged the exception and stopped there, so the
    only status the user ever saw was the last successful one. Nothing before
    this exercised the loop's except branch reaching on_status at all.
    """
    m, _ = build_monitor(conn, config=Config(sample_interval_s=3600,
                                             camera_indices={"front": 0},
                                             gemini_enabled=False))
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    # One good check first, so a stale "Good posture" is genuinely available to
    # be left on screen. Without it the test could pass on an empty call list.
    with patch("posture.cameras.capture_roles", return_value=({"front": frame}, None)):
        m.run_once()
    assert m.on_status.call_args_list[-1].args[0] == "Good posture"

    with patch("posture.cameras.capture_roles",
               side_effect=OSError("[Errno 28] No space left on device")):
        m.start()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if m.on_status.call_args_list[-1].args[0] != "Good posture":
                break
            time.sleep(0.01)
        m.stop(timeout=5.0)

    said = m.on_status.call_args_list[-1].args[0]
    assert said != "Good posture", "the menu kept showing the last good reading"
    assert said == "Last check failed: OSError. See the log."


def test_a_status_callback_that_raises_does_not_kill_the_loop(conn):
    """Reporting a failure must not become a second, fatal failure."""
    m, _ = build_monitor(conn, config=Config(sample_interval_s=3600,
                                             camera_indices={"front": 0},
                                             gemini_enabled=False))
    m.on_status = MagicMock(side_effect=RuntimeError("menu is gone"))

    with patch("posture.cameras.capture_roles", side_effect=OSError("boom")):
        m.start()
        time.sleep(0.1)
        alive = m._thread is not None and m._thread.is_alive()
        m.stop(timeout=5.0)

    assert alive, "the loop died while reporting that a check had failed"


def test_a_bad_sample_interval_stops_the_loop_visibly_instead_of_silently(conn):
    """A quoted number in config.toml killed the worker while is_running stayed True.

    sample_interval_s = "30" raises inside Event.wait, which sat OUTSIDE the
    loop's try. The thread died after one check, is_running went on returning
    True, and the menu read "Monitoring" all day over nothing.
    """
    m, _ = build_monitor(conn, config=Config(sample_interval_s="30",  # type: ignore[arg-type]
                                             camera_indices={"front": 0},
                                             gemini_enabled=False))
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    with patch("posture.cameras.capture_roles", return_value=({"front": frame}, None)):
        m.start()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and m._thread.is_alive():
            time.sleep(0.01)
        thread = m._thread
        # Read BEFORE stop(), which sets _running False unconditionally at its
        # top. Reading after would pass whether or not the dead loop cleared it,
        # which is the whole property under test.
        still_claims_running = m.is_running
        m.stop(timeout=5.0)

    assert thread is not None and not thread.is_alive(), "the loop should have died"
    assert still_claims_running is False, "is_running still claims a loop that is gone"
    said = [c.args[0] for c in m.on_status.call_args_list]
    assert "Monitoring stopped: TypeError. See the log." in said, said


def test_a_preflight_callback_that_raises_is_logged_not_dumped_to_stderr(conn):
    """on_result sat outside the try, so a raising callback escaped to the
    default threading excepthook. run_once_async already guards this; these two
    entry points must behave alike."""
    m, _ = build_monitor(conn)
    m.gemini = MagicMock()
    m.gemini.preflight.return_value = None

    seen = []

    def excepthook(args):
        seen.append(args.exc_type)

    original = threading.excepthook
    threading.excepthook = excepthook
    try:
        m.run_gemini_preflight_async(MagicMock(side_effect=RuntimeError("menu is gone")))
        assert m.stop(timeout=5.0) is True
    finally:
        threading.excepthook = original

    assert seen == [], f"the callback exception escaped the worker: {seen}"
    assert m._workers == set()


def test_pausing_and_resuming_starts_a_fresh_window(conn):
    """Stale window state must not survive stop() then start().

    _loop is stubbed out so start() does exactly one thing observable here:
    reset the state. Letting the real loop run would immediately take a check
    and refill the window, which would hide whether start() had cleared it.
    """
    m, _ = build_monitor(conn, record_interval_s=3600)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    with patch("posture.cameras.capture_roles", return_value=({"front": frame}, None)):
        m.run_once()   # first flush always writes, and clears the window
        m.run_once()   # buffered into the still-open window
    assert len(m._window) == 1
    assert m._last_write_at is not None

    assert m.stop(timeout=5.0) is True
    with patch.object(Monitor, "_loop", lambda self: None):
        assert m.start() is True
        m.stop(timeout=5.0)

    assert m._window == [], "the paused window leaked into the next run"
    assert m._last_write_at is None, "the resumed run is still waiting out the old interval"


def test_resuming_does_not_inherit_the_pre_pause_slouch_streak(conn):
    """Pause for a call, resume slouched, and one bad sample must not nudge.

    The streak needs two consecutive samples. One before the pause plus one
    after is not two consecutive samples of anything: the gap is unbounded,
    because monotonic() does not advance while the lid is shut.
    """
    m, _ = build_monitor(conn)
    m._tracker = SlouchTracker(sustained_samples=2, cooldown_s=600)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    bad = LocalAssessment(Rating.POOR, "cva_deg", {"cva_deg": -12.0})

    with patch("posture.monitor.score", return_value=bad), \
         patch("posture.cameras.capture_roles", return_value=({"front": frame}, None)):
        m.run_once()                      # streak 1
        assert m.stop(timeout=5.0) is True
        with patch.object(Monitor, "_loop", lambda self: None):
            assert m.start() is True
            m.stop(timeout=5.0)
        m.run_once()                      # would be streak 2 without the reset

    assert m.on_nudge.call_count == 0


def a_status_record(**overrides):
    from posture.store.queries import CheckRecord
    fields = dict(
        ts="2026-07-30T12:00:00", date="2026-07-30", outcome=Outcome.OK,
        baseline_id=1, cameras=["front"], metrics=None,
        local_rating=Rating.GOOD, local_driver=None,
        gemini_rating=None, gemini_issue=None, gemini_details=None,
        gemini_tip=None, gemini_model=None,
        is_comparison_sample=False, api_error=None,
    )
    fields.update(overrides)
    return CheckRecord(**fields)


def test_status_text_names_a_failing_second_opinion_beside_the_local_reading():
    """"Good posture" alone, while every Gemini call 404s, is v2's failure shape."""
    assert Monitor._status_text(a_status_record()) == "Good posture"
    assert Monitor._status_text(
        a_status_record(api_error=Outcome.MODEL_UNAVAILABLE.value)
    ) == "Good posture (Gemini: MODEL_UNAVAILABLE)"
    assert Monitor._status_text(
        a_status_record(local_rating=Rating.POOR, api_error=Outcome.API_ERROR.value)
    ) == "Poor posture (Gemini: API_ERROR)"


def test_status_text_does_not_blame_gemini_for_a_local_crop_refusal():
    """CROP_UNAVAILABLE shares the api_error column but never reached the API.

    Naming Gemini here would be as false as v2 calling a retired model "no
    person detected", just in the other direction.
    """
    text = Monitor._status_text(
        a_status_record(api_error=Outcome.CROP_UNAVAILABLE.value))
    assert text == "Good posture"


def test_the_ui_does_not_spawn_its_own_check_threads():
    """check_now must delegate to the tracked worker, not raise its own thread.

    app.py cannot be imported without a window server, so this reads the source
    instead. That is a blunt instrument, but the alternative is zero coverage on
    the exact property this task exists to establish: an untracked thread is
    invisible to stop(), so quit can tear down the detector underneath it.
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1]
           / "src" / "posture" / "ui" / "app.py").read_text()
    assert "run_once_async" in src, "check_now should delegate to the tracked worker"
    assert "threading.Thread" not in src, (
        "app.py spawns a thread directly; manual checks must go through "
        "Monitor.run_once_async so stop() can join them"
    )


# --- F1: capability was stamped from DETECTION succeeding, not from a metric
# actually being produced. A side camera that sees the user clearly but yields
# neither a craniovertebral angle nor a trunk angle still made the row claim
# "front+side". daily_scores GROUPS BY that column and every headline number
# on the dashboard is filtered on it, so the one job it has (keeping a
# two-camera trend free of front-only measurements) was defeated silently.

def _invisible_landmarks():
    """A detection with every landmark below the visibility floor.

    This is the case the finding is about: detect() returned something, so the
    old code counted the camera as contributing, but metrics.py refuses to
    measure anything from landmarks it cannot see, so the row carries no
    side-derived number at all.
    """
    from posture.types import Landmarks, Point
    return Landmarks(points=tuple(Point(0.0, 0.0, 0.0) for _ in range(33)))


def _two_camera_monitor(conn, side_landmarks):
    m, detector = build_monitor(
        conn, config=Config(camera_indices={"front": 0, "side": 2},
                            gemini_enabled=False))
    front_frame = np.full((10, 10, 3), 1, dtype=np.uint8)
    side_frame = np.full((10, 10, 3), 2, dtype=np.uint8)
    detector.detect.side_effect = (
        lambda frame: upright_front() if frame[0][0][0] == 1 else side_landmarks)
    return m, {"front": front_frame, "side": side_frame}


def test_capability_drops_a_side_camera_that_produced_no_side_metric(conn):
    m, frames = _two_camera_monitor(conn, _invisible_landmarks())
    with patch("posture.cameras.capture_roles", return_value=(frames, None)):
        m.run_once()

    row = recent_checks(conn, 1)[0]
    # The row genuinely holds no side-derived measurement...
    assert row["metrics"]["cva_deg"] is None
    assert row["metrics"]["trunk_angle_deg"] is None
    # ...so it must not claim one. This is the assertion the finding is about.
    assert row["capability"] == "front"
    # cameras is a different column with a different job: it still records that
    # the side camera returned a frame and a detection, which is what makes the
    # difference between the two columns diagnosable.
    assert set(row["cameras"]) == {"front", "side"}


def test_capability_still_claims_a_side_camera_that_did_produce_an_angle(conn):
    """The fix must not simply delete the side claim everywhere.

    Without this, "return the front roles only" passes the test above.
    """
    m, frames = _two_camera_monitor(conn, slouched_side(20.0))
    with patch("posture.cameras.capture_roles", return_value=(frames, None)):
        m.run_once()

    row = recent_checks(conn, 1)[0]
    assert row["metrics"]["cva_deg"] is not None
    assert row["capability"] == "front+side"


def test_capability_drops_a_front_camera_that_produced_no_front_metric(conn):
    """Symmetric: the column names metrics, not cameras, in both directions."""
    m, detector = build_monitor(
        conn, config=Config(camera_indices={"front": 0, "side": 2},
                            gemini_enabled=False))
    front_frame = np.full((10, 10, 3), 1, dtype=np.uint8)
    side_frame = np.full((10, 10, 3), 2, dtype=np.uint8)
    detector.detect.side_effect = (
        lambda frame: _invisible_landmarks() if frame[0][0][0] == 1
        else slouched_side(20.0))
    with patch("posture.cameras.capture_roles",
               return_value=({"front": front_frame, "side": side_frame}, None)):
        m.run_once()

    row = recent_checks(conn, 1)[0]
    assert row["metrics"]["shoulder_tilt_deg"] is None
    assert row["capability"] == "side"
    assert set(row["cameras"]) == {"front", "side"}


# --- F5: "Not calibrated yet" was shown whenever local_rating was None, which
# covers two different situations with two different fixes. One of them is a
# dead end: telling someone to calibrate when they already have sends them to
# recalibrate, and recalibrating does not help.

def test_an_uncalibrated_check_still_says_to_calibrate():
    """The genuine case must keep its instruction. Pinned so the fix cannot
    simply delete the sentence for everyone."""
    assert Monitor._local_status_text(
        a_status_record(baseline_id=None, local_rating=None)
    ) == "Not calibrated yet"


def test_a_calibrated_check_with_no_comparable_metric_is_not_called_uncalibrated():
    """A baseline exists; it just shares no metric with what was measured.

    This happens for real: a baseline calibrated on the side camera holds
    cva_deg and trunk_angle_deg, and a day the side camera is unplugged
    produces only front metrics, so score() finds nothing to compare and
    returns rating None. The fix is to recalibrate ON THIS SETUP or to
    reconnect the other camera, not to run the same calibration again.
    """
    text = Monitor._local_status_text(
        a_status_record(baseline_id=7, local_rating=None))
    assert text == "Baseline does not cover this camera setup"


def test_the_two_uncalibrated_situations_never_produce_the_same_text():
    """The whole point of the fix: they must be distinguishable on the menu bar."""
    without = Monitor._local_status_text(
        a_status_record(baseline_id=None, local_rating=None))
    with_baseline = Monitor._local_status_text(
        a_status_record(baseline_id=7, local_rating=None))
    assert without != with_baseline
