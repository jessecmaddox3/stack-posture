from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from posture.config import Config
from posture.monitor import Monitor
from posture.store.db import connect, migrate
from posture.store.queries import recent_checks, save_baseline
from posture.types import Outcome, PostureMetrics, Rating
from tests.synthetic import upright_front

FRAME = np.zeros((10, 10, 3), dtype=np.uint8)


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "w.db")
    migrate(c)
    save_baseline(c, PostureMetrics(cva_deg=60.0, shoulder_tilt_deg=0.0), "test")
    yield c
    c.close()


def build(conn, record_interval_s=120):
    cfg = Config(camera_indices={"front": 0}, gemini_enabled=False,
                 sample_interval_s=30, record_interval_s=record_interval_s)
    detector = MagicMock()
    detector.detect.return_value = upright_front()
    return Monitor(config=cfg, detector=detector, conn=conn,
                   on_nudge=MagicMock(), on_status=MagicMock())


def run(monitor):
    with patch("posture.cameras.capture_roles", return_value=({"front": FRAME}, None)):
        return monitor.run_once()


def test_the_first_sample_always_writes_a_row(conn):
    monitor = build(conn)
    run(monitor)
    assert len(recent_checks(conn, 10)) == 1


def test_subsequent_samples_inside_the_window_are_buffered(conn):
    monitor = build(conn, record_interval_s=120)
    run(monitor)
    run(monitor)
    run(monitor)
    assert len(recent_checks(conn, 10)) == 1


def test_a_row_is_written_once_the_window_elapses(conn):
    monitor = build(conn, record_interval_s=120)
    run(monitor)
    run(monitor)
    monitor._last_write_at -= 200  # pretend the window expired
    run(monitor)
    assert len(recent_checks(conn, 10)) == 2


def test_a_zero_record_interval_writes_every_sample(conn):
    monitor = build(conn, record_interval_s=0)
    run(monitor)
    run(monitor)
    run(monitor)
    assert len(recent_checks(conn, 10)) == 3


def test_the_written_row_holds_the_window_median(conn):
    monitor = build(conn, record_interval_s=120)
    run(monitor)                       # flushes immediately, buffer resets
    monitor._window.clear()
    from posture.monitor import SampleResult
    from posture.scoring import LocalAssessment
    # Deliberately asymmetric: median 60, mean 80. Symmetric values like
    # (50, 60, 70) share a mean and median, so the test would pass against a
    # mean-based implementation and prove nothing its name claims.
    for cva in (50.0, 60.0, 130.0):
        monitor._window.append(SampleResult(
            Outcome.OK, PostureMetrics(cva_deg=cva),
            LocalAssessment(Rating.GOOD, None, {}), ["front"], {}, {}))
    monitor._last_write_at -= 200
    run(monitor)
    written = recent_checks(conn, 1)[0]
    # The live sample is a front view and contributes no CVA reading, so the
    # median is over the three planted values (50, 60, 130).
    assert written["metrics"]["cva_deg"] == pytest.approx(60.0)


def test_buffering_does_not_delay_the_nudge(conn):
    from dataclasses import replace

    from posture.monitor import SlouchTracker

    monitor = build(conn, record_interval_s=3600)
    monitor.config = replace(monitor.config, sustained_samples=2)
    monitor._tracker = SlouchTracker(sustained_samples=2, cooldown_s=0)
    with patch("posture.monitor.score") as fake:
        from posture.scoring import LocalAssessment
        fake.return_value = LocalAssessment(Rating.POOR, "cva_deg", {"cva_deg": -12.0})
        run(monitor)
        run(monitor)
    # Only one row written (window is an hour), but the nudge already fired.
    assert len(recent_checks(conn, 10)) == 1
    assert monitor.on_nudge.called


def test_non_ok_samples_are_not_buffered_into_the_median(conn):
    monitor = build(conn, record_interval_s=120)
    run(monitor)                                     # flushes, clearing the window
    run(monitor)                                     # buffered: window now has 1
    assert len(monitor._window) == 1
    monitor.detector.detect.return_value = None      # person leaves
    run(monitor)
    # The absent sample must not join the window, so the count is unchanged.
    assert len(monitor._window) == 1
    assert all(s.outcome is Outcome.OK for s in monitor._window)


def test_the_written_rating_is_reproducible_from_the_written_metrics(conn):
    """A stored row must not contradict its own numbers, and must come from the
    AGGREGATE rather than from any single sample.

    Three wrong implementations all produce GOOD here while the correct one
    produces DECENT, so one assertion catches all three:

      * taking a mode over per-sample ratings (the original defect): the
        per-sample ratings are POOR, DECENT, GOOD, GOOD, so the mode is GOOD.
      * scoring the last sample in the window: that is always the live sample
        appended by _perform_check, which is front-only and scores GOOD.
      * reusing the last sample's own precomputed assessment: likewise GOOD.

    Note the last window entry is NOT one of the planted samples. _perform_check
    appends its live capture after them and before _flush runs, so no reordering
    of the planted list can change what sits last. The discrimination has to come
    from the aggregate scoring differently to a lone GOOD sample, which is why
    the planted CVA values are chosen so their median lands in the DECENT band.
    """
    from posture.monitor import SampleResult
    from posture.scoring import score
    from posture.store.queries import active_baseline

    monitor = build(conn, record_interval_s=120)
    run(monitor)                      # flush, clearing the window
    monitor._window.clear()

    base = active_baseline(conn)
    # Baseline cva is 60. The rule is direction -1, decent_at 4, poor_at 10.
    # 44 scores POOR (16 below), 54 scores DECENT (6 below), 64 scores GOOD.
    # Their median is 54, so the aggregate is DECENT driven by cva_deg, while
    # every wrong implementation above yields GOOD.
    planted = [
        PostureMetrics(cva_deg=44.0, shoulder_tilt_deg=0.0),
        PostureMetrics(cva_deg=54.0, shoulder_tilt_deg=0.0),
        PostureMetrics(cva_deg=64.0, shoulder_tilt_deg=0.0),
    ]
    for m in planted:
        monitor._window.append(SampleResult(
            Outcome.OK, m, score(m, base), ["front"], {}, {}))
    monitor._last_write_at -= 200
    run(monitor)

    row = recent_checks(conn, 1)[0]
    stored = PostureMetrics(**row["metrics"])
    rescored = score(stored, base)

    # The invariant: re-scoring a row's own metrics returns the row's own rating.
    assert row["local_rating"] == (rescored.rating.value if rescored.rating else None)
    assert row["local_driver"] == rescored.driver
    # And it is specifically the aggregate, not a lone sample.
    assert row["local_rating"] == "DECENT"
    assert row["local_driver"] == "cva_deg"
    assert stored.cva_deg == 54.0
