import numpy as np
import pytest

from posture import cameras
from posture.baseline import (
    CalibrationSample, collect_baseline_samples, majority_facing_sign,
    median_metrics, run_calibration,
)
from posture.config import Config
from posture.store.db import connect, migrate
from posture.store.queries import active_baseline
from posture.types import Outcome, PostureMetrics
from tests.synthetic import mirrored, slouched_side, upright_side


def _vote(sign: float | None) -> CalibrationSample:
    return CalibrationSample(PostureMetrics(), sign)


def test_median_of_a_single_sample_is_itself():
    m = median_metrics([PostureMetrics(cva_deg=60.0)])
    assert m.cva_deg == 60.0


def test_median_ignores_none_values_per_field():
    samples = [
        PostureMetrics(cva_deg=60.0, trunk_angle_deg=2.0),
        PostureMetrics(cva_deg=62.0),                      # no trunk reading
        PostureMetrics(cva_deg=58.0, trunk_angle_deg=4.0),
    ]
    m = median_metrics(samples)
    assert m.cva_deg == pytest.approx(60.0)
    assert m.trunk_angle_deg == pytest.approx(3.0)


def test_median_resists_a_single_outlier():
    samples = [PostureMetrics(cva_deg=v) for v in (60.0, 61.0, 59.0, 60.5, 5.0)]
    assert median_metrics(samples).cva_deg == pytest.approx(60.0)


def test_field_with_no_readings_stays_none():
    assert median_metrics([PostureMetrics(cva_deg=60.0)]).trunk_angle_deg is None


def test_empty_sample_list_raises():
    with pytest.raises(ValueError, match="no samples"):
        median_metrics([])


def test_median_keeps_a_negative_cva_negative():
    """cva_deg is signed, so the aggregators must not flatten it.

    Monitor medians a window of samples through this same function before
    scoring it, so a window spent with the head dropped has to survive as a
    negative number or the check reports a posture that never happened.
    """
    samples = [PostureMetrics(cva_deg=v) for v in (-58.0, -62.0, -60.0)]
    assert median_metrics(samples).cva_deg == pytest.approx(-60.0)


def test_even_count_averages_the_middle_two():
    samples = [PostureMetrics(cva_deg=v) for v in (58.0, 60.0, 62.0, 64.0)]
    assert median_metrics(samples).cva_deg == pytest.approx(61.0)


def test_majority_facing_sign_unanimous_positive():
    samples = [CalibrationSample(PostureMetrics(), 1.0) for _ in range(5)]
    assert majority_facing_sign(samples) == 1.0


def test_majority_facing_sign_unanimous_negative():
    samples = [CalibrationSample(PostureMetrics(), -1.0) for _ in range(5)]
    assert majority_facing_sign(samples) == -1.0


def test_majority_facing_sign_all_none():
    samples = [CalibrationSample(PostureMetrics(), None) for _ in range(3)]
    assert majority_facing_sign(samples) is None


def test_majority_facing_sign_empty():
    assert majority_facing_sign([]) is None


def test_an_exact_tie_yields_no_facing_sign():
    """Fail closed. A tie previously became +1, which on a -1 rig would invert
    every future trunk reading with nothing looking wrong."""
    assert majority_facing_sign([_vote(1.0), _vote(-1.0)]) is None


def test_too_few_votes_yields_no_facing_sign():
    # One usable frame out of a ten second window is not a calibration.
    assert majority_facing_sign([_vote(1.0)] + [_vote(None)] * 19) is None


def test_a_weak_majority_yields_no_facing_sign():
    votes = [_vote(1.0)] * 6 + [_vote(-1.0)] * 5
    assert majority_facing_sign(votes) is None


def test_a_clear_majority_freezes_the_sign():
    votes = [_vote(-1.0)] * 9 + [_vote(1.0)]
    assert majority_facing_sign(votes) == -1.0


def test_collect_baseline_samples_carries_the_frame_facing_vote(monkeypatch):
    """The per-frame facing vote must reach CalibrationSample, not be dropped.

    Needs no camera and no model: capture_roles is monkeypatched and the detector
    is a stub object with a detect() method. An earlier review found this path
    entirely untested, and a mutant passing facing_sign=None went undetected.
    """
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    monkeypatch.setattr(cameras, "capture_roles", lambda cfg: ({"side": frame}, None))

    class StubDetector:
        def detect(self, _frame):
            return upright_side()

    cfg = Config(camera_indices={"side": 1})
    samples = collect_baseline_samples(cfg, StubDetector(), seconds=0.2, interval_s=0.01)
    assert samples
    assert samples[0].facing_sign == 1.0


def test_collect_baseline_samples_applies_the_facing_vote_to_trunk_angle(monkeypatch):
    """A mirrored camera flips the vote, and the trunk sign must follow it.

    Without the vote being applied, this forward lean would read negative.
    """
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    monkeypatch.setattr(cameras, "capture_roles", lambda cfg: ({"side": frame}, None))
    lm = mirrored(slouched_side(20.0))

    class StubDetector:
        def detect(self, _frame):
            return lm

    cfg = Config(camera_indices={"side": 1})
    samples = collect_baseline_samples(cfg, StubDetector(), seconds=0.2, interval_s=0.01)
    assert samples[0].facing_sign == -1.0
    assert samples[0].metrics.trunk_angle_deg == pytest.approx(20.0, abs=1.5)


def test_run_calibration_persists_the_frozen_facing_sign(tmp_path, monkeypatch):
    """The frozen sign must survive all the way into the database.

    This is the payoff for signing the trunk angle at all. An in-memory database
    and stubs cover it with no hardware.
    """
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    monkeypatch.setattr(cameras, "capture_roles", lambda cfg: ({"side": frame}, None))

    class StubDetector:
        def detect(self, _frame):
            return mirrored(upright_side())

    conn = connect(tmp_path / "cal.db")
    migrate(conn)
    baseline = run_calibration(Config(camera_indices={"side": 1}), StubDetector(),
                               conn, label="stub", seconds=0.2, interval_s=0.01)
    assert baseline is not None
    assert baseline.facing_sign == -1.0
    assert active_baseline(conn).facing_sign == -1.0
    conn.close()


def test_an_exact_tie_above_the_vote_minimum_still_yields_no_sign():
    """The tie rule itself, not the vote-count gate.

    The other tie test uses two votes, which MIN_FACING_VOTES rejects before the
    tie logic runs, so tie-breaking was never actually exercised. Restoring the
    old "ties favour +1" behaviour passed the entire suite.
    """
    votes = [_vote(1.0)] * 3 + [_vote(-1.0)] * 3
    assert majority_facing_sign(votes) is None


def test_a_rejected_facing_sign_also_drops_the_baseline_trunk_angle(tmp_path, monkeypatch):
    """A baseline must not store a number the app has promised not to use.

    With no trusted sign, live trunk readings are None because Monitor passes
    require_calibrated=True, and scoring skips the metric. Keeping a value here
    makes calibrate.py print a trunk angle directly above a note saying it is
    omitted, and persists a reference nothing can ever be compared against.
    """
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    monkeypatch.setattr(cameras, "capture_roles", lambda cfg: ({"side": frame}, None))

    class AlternatingDetector:
        """Real trunk angles every frame, but no trustworthy facing majority."""

        def __init__(self):
            self.n = 0

        def detect(self, _frame):
            self.n += 1
            lm = slouched_side(20.0)
            return lm if self.n % 2 else mirrored(lm)

    conn = connect(tmp_path / "ambiguous.db")
    migrate(conn)
    baseline = run_calibration(Config(camera_indices={"side": 1}),
                               AlternatingDetector(), conn, label="ambiguous",
                               seconds=0.3, interval_s=0.01)
    assert baseline is not None
    assert baseline.facing_sign is None, "alternating votes should not freeze a sign"
    assert baseline.metrics.trunk_angle_deg is None
    assert active_baseline(conn).metrics.trunk_angle_deg is None
    # The rest of the baseline survives: only the untrustworthy metric is dropped.
    assert baseline.metrics.cva_deg is not None


def test_run_calibration_returns_none_when_too_few_samples(tmp_path, monkeypatch):
    monkeypatch.setattr(cameras, "capture_roles",
                        lambda cfg: ({}, Outcome.CAMERA_BUSY))
    conn = connect(tmp_path / "cal.db")
    migrate(conn)
    result = run_calibration(Config(camera_indices={"side": 1}), object(),
                             conn, label="stub", seconds=0.05, interval_s=0.01)
    assert result is None
    assert active_baseline(conn) is None
    conn.close()
