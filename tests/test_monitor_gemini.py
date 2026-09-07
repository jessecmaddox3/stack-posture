import random
from dataclasses import replace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from posture.config import Config
from posture.crop import CropUnavailable
from posture.gemini import GeminiVerdict
from posture.monitor import Monitor
from posture.store.db import connect, migrate
from posture.store.queries import agreement_matrix, recent_checks, save_baseline
from posture.types import Outcome, PostureMetrics, Rating
from tests.synthetic import upright_front, upright_side

FRAME = np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "g.db")
    migrate(c)
    save_baseline(c, PostureMetrics(cva_deg=60.0, shoulder_tilt_deg=0.0), "test")
    yield c
    c.close()


def build(conn, *, sample_rate, verdict=None, outcome=None, seed=0, record_interval_s=120,
          max_coaching_calls_per_day=1000):
    cfg = Config(camera_indices={"front": 0}, gemini_enabled=True,
                 comparison_sample_rate=sample_rate, min_seconds_between_api_calls=0,
                 record_interval_s=record_interval_s,
                 max_coaching_calls_per_day=max_coaching_calls_per_day)
    detector = MagicMock()
    detector.detect.return_value = upright_front()
    client = MagicMock()
    client.available = True
    client.breaker.is_open.return_value = False
    client.assess.return_value = (
        verdict or GeminiVerdict(True, Rating.DECENT, "rounded shoulders",
                                 "Shoulders are rolled forward.", "Roll them back."),
        outcome,
    )
    monitor = Monitor(config=cfg, detector=detector, conn=conn,
                      on_nudge=MagicMock(), on_status=MagicMock(),
                      gemini_client=client, rng=random.Random(seed))
    return monitor, client


def run(monitor):
    with patch("posture.cameras.capture_roles", return_value=({"front": FRAME}, None)):
        return monitor.run_once()


def test_rate_zero_and_good_posture_means_no_api_call(conn):
    monitor, client = build(conn, sample_rate=0.0)
    record = run(monitor)
    client.assess.assert_not_called()
    assert record.gemini_rating is None
    assert record.is_comparison_sample is False


def test_rate_one_always_calls_and_marks_a_comparison_sample(conn):
    monitor, client = build(conn, sample_rate=1.0)
    record = run(monitor)
    client.assess.assert_called_once()
    assert record.is_comparison_sample is True
    assert record.gemini_rating is Rating.DECENT


def test_gemini_verdict_is_persisted(conn):
    monitor, _ = build(conn, sample_rate=1.0)
    run(monitor)
    row = recent_checks(conn, 1)[0]
    assert row["gemini_rating"] == "DECENT"
    assert row["gemini_issue"] == "rounded shoulders"
    assert row["gemini_model"] is not None


def test_local_rating_survives_a_disagreeing_gemini(conn):
    # The local track is the trend; Gemini never overwrites it.
    monitor, _ = build(conn, sample_rate=1.0)
    record = run(monitor)
    assert record.local_rating is Rating.GOOD
    assert record.gemini_rating is Rating.DECENT


def test_only_sampled_rows_reach_the_agreement_matrix(conn):
    sampled, _ = build(conn, sample_rate=1.0)
    run(sampled)
    unsampled, client = build(conn, sample_rate=0.0)
    # Force a gated call by making the local rating bad.
    with patch("posture.monitor.score") as fake_score:
        from posture.scoring import LocalAssessment
        fake_score.return_value = LocalAssessment(Rating.POOR, "cva_deg", {"cva_deg": -12.0})
        run(unsampled)
    client.assess.assert_called_once()  # gated call happened
    matrix = agreement_matrix(conn)
    assert sum(matrix.values()) == 1  # but only the sampled row counts


def test_api_error_is_recorded_without_losing_the_local_result(conn):
    monitor, _ = build(conn, sample_rate=1.0, verdict=None, outcome=Outcome.API_ERROR)
    record = run(monitor)
    assert record.outcome is Outcome.OK          # local measurement still succeeded
    assert record.gemini_rating is None
    assert record.api_error == "API_ERROR"
    # The name of this test claims the local result survives, not just the
    # outcome flag: pin the actual local rating and metrics too.
    assert record.local_rating is Rating.GOOD
    assert record.metrics is not None


def test_model_unavailable_is_recorded_distinctly(conn):
    monitor, _ = build(conn, sample_rate=1.0, verdict=None,
                       outcome=Outcome.MODEL_UNAVAILABLE)
    record = run(monitor)
    assert record.api_error == "MODEL_UNAVAILABLE"
    assert record.local_rating is Rating.GOOD
    assert record.metrics is not None


def test_open_circuit_suppresses_the_call(conn):
    monitor, client = build(conn, sample_rate=1.0)
    client.breaker.is_open.return_value = True
    record = run(monitor)
    client.assess.assert_not_called()
    # Suppressing the call must not cost the local measurement either.
    assert record.local_rating is Rating.GOOD
    assert record.metrics is not None


def test_person_absent_never_calls_gemini(conn):
    monitor, client = build(conn, sample_rate=1.0)
    monitor.detector.detect.return_value = None
    run(monitor)
    client.assess.assert_not_called()


def test_frames_sent_to_gemini_are_cropped(conn):
    monitor, client = build(conn, sample_rate=1.0)
    big = np.zeros((1080, 1920, 3), dtype=np.uint8)
    with patch("posture.cameras.capture_roles", return_value=({"front": big}, None)):
        monitor.run_once()
    sent = client.assess.call_args[0][0]["front"]
    assert isinstance(sent, bytes)
    assert sent[:2] == b"\xff\xd8"
    # A crop of a 1920x1080 frame encodes far smaller than the full frame would.
    assert len(sent) < 400_000


def test_a_crop_refusal_never_reaches_the_api(conn):
    # Task 13 made crop_to_person raise rather than hand back the uncropped
    # frame. If this stage does not catch it, the exception escapes into the
    # sample thread; if it catches it and sends anyway, a picture of the room
    # goes to Gemini. Neither is acceptable.
    monitor, client = build(conn, sample_rate=1.0)
    with patch("posture.monitor.crop_to_person", side_effect=CropUnavailable("no region")):
        run(monitor)
    client.assess.assert_not_called()


def test_a_crop_refusal_is_recorded_as_a_local_refusal_not_an_api_error(conn):
    # We never contacted Gemini, so blaming Gemini would be false data and would
    # feed the circuit breaker with failures the API never caused.
    monitor, client = build(conn, sample_rate=1.0)
    with patch("posture.monitor.crop_to_person", side_effect=CropUnavailable("no region")):
        run(monitor)
    row = recent_checks(conn, 1)[0]
    assert row["api_error"] == Outcome.CROP_UNAVAILABLE.value
    client.breaker.record_failure.assert_not_called()


def test_a_sampled_row_stays_in_the_comparison_set_even_when_the_crop_refuses(conn):
    # The roll already happened, and it happened before the local rating was
    # consulted. Dropping the row would shrink the comparison set along an axis
    # that is NOT independent of posture: crop failure tracks poor detection,
    # which tracks the extreme postures most worth comparing.
    monitor, client = build(conn, sample_rate=1.0)
    with patch("posture.monitor.crop_to_person", side_effect=CropUnavailable("no region")):
        run(monitor)
    row = recent_checks(conn, 1)[0]
    assert row["is_comparison_sample"] == 1


def test_the_local_result_survives_a_crop_refusal(conn):
    # A refusal to send must not cost us the local measurement, which is the
    # primary signal. Gemini is only ever the second opinion.
    monitor, client = build(conn, sample_rate=1.0)
    with patch("posture.monitor.crop_to_person", side_effect=CropUnavailable("no region")):
        run(monitor)
    row = recent_checks(conn, 1)[0]
    assert row["local_rating"] is not None
    assert row["metrics"]


def test_the_comparison_sample_includes_both_good_and_poor_local_ratings(conn):
    # Pins the unbiased-roll property in BOTH directions. A mutant that leaves
    # the roll first but then filters is_comparison_sample by the rating
    # afterwards (e.g. excluding POOR) would still pass every rate=0.0/rate=1.0
    # test above, because those only ever exercise a single, fixed GOOD rating.
    # This test alternates GOOD and POOR local ratings at a mid rate and checks
    # that a comparison sample was taken of each, not just of the postures
    # local already rated fine.
    from posture.scoring import LocalAssessment

    monitor, client = build(conn, sample_rate=0.5, record_interval_s=0)
    ratings = [Rating.GOOD if i % 2 == 0 else Rating.POOR for i in range(40)]
    with patch("posture.monitor.score") as fake_score:
        for r in ratings:
            fake_score.return_value = LocalAssessment(
                r, "cva_deg" if r is Rating.POOR else None, {})
            run(monitor)

    rows = recent_checks(conn, len(ratings))
    sampled_ratings = {row["local_rating"] for row in rows if row["is_comparison_sample"]}
    assert Rating.GOOD.value in sampled_ratings
    assert Rating.POOR.value in sampled_ratings


# --- the coaching a gated call paid for has to reach the nudge ---

def _poor():
    from posture.scoring import LocalAssessment
    return LocalAssessment(Rating.POOR, "cva_deg", {"cva_deg": -12.0})


def test_a_gated_calls_tip_reaches_the_nudge_record(conn):
    # The whole reason a gated call exists: local rated the posture badly, so we
    # paid for coaching text. If the nudge record does not carry it, the popup
    # falls back to its generic line and the call bought nothing at all.
    from posture.monitor import SlouchTracker

    monitor, client = build(conn, sample_rate=0.0)
    monitor._tracker = SlouchTracker(sustained_samples=2, cooldown_s=600)
    with patch("posture.monitor.score", return_value=_poor()):
        run(monitor)   # flushes, makes the gated call
        run(monitor)   # second bad sample: the nudge fires
    client.assess.assert_called_once()
    monitor.on_nudge.assert_called_once()
    nudged = monitor.on_nudge.call_args[0][0]
    assert nudged.gemini_tip == "Roll them back."
    assert nudged.gemini_details == "Shoulders are rolled forward."
    assert nudged.gemini_rating is Rating.DECENT
    assert nudged.gemini_issue == "rounded shoulders"


def test_a_stale_verdict_never_reaches_a_later_nudge(conn):
    # A verdict describes the window it was taken from. Attaching one from
    # several windows ago would coach him about a posture he has already left.
    from posture.monitor import SlouchTracker

    monitor, client = build(conn, sample_rate=0.0)
    monitor._tracker = SlouchTracker(sustained_samples=1, cooldown_s=0)
    with patch("posture.monitor.score", return_value=_poor()):
        run(monitor)   # nudges, then flushes and makes the gated call
        run(monitor)
        assert monitor.on_nudge.call_args[0][0].gemini_tip is not None
        monitor._last_coaching = replace(
            monitor._last_coaching,
            at=monitor._last_coaching.at - (monitor.config.record_interval_s * 10))
        run(monitor)
    assert monitor.on_nudge.call_args[0][0].gemini_tip is None
    assert monitor._last_coaching is None, "a stale verdict must be dropped, not re-tested"


def test_a_good_sample_discards_the_cached_verdict(conn):
    # He fixed it, then slouched again later. The old verdict describes the
    # earlier slouch, not this one, so it must not be recycled.
    from posture.monitor import SlouchTracker

    monitor, client = build(conn, sample_rate=0.0)
    monitor._tracker = SlouchTracker(sustained_samples=1, cooldown_s=0)
    with patch("posture.monitor.score", return_value=_poor()):
        run(monitor)   # nudges, then flushes and makes the gated call
        run(monitor)
        assert monitor.on_nudge.call_args[0][0].gemini_tip is not None
    from posture.scoring import LocalAssessment
    with patch("posture.monitor.score",
               return_value=LocalAssessment(Rating.GOOD, None, {})):
        run(monitor)
    with patch("posture.monitor.score", return_value=_poor()):
        run(monitor)
    assert monitor.on_nudge.call_args[0][0].gemini_tip is None


def test_the_nudge_record_is_never_marked_a_comparison_sample(conn):
    # on_nudge records are display-only and never inserted, but stamping one
    # would be a lie waiting to be persisted by any future caller that did.
    from posture.monitor import SlouchTracker

    monitor, _ = build(conn, sample_rate=1.0)
    monitor._tracker = SlouchTracker(sustained_samples=2, cooldown_s=600)
    with patch("posture.monitor.score", return_value=_poor()):
        run(monitor)
        run(monitor)
    assert monitor.on_nudge.call_args[0][0].is_comparison_sample is False


def test_a_window_whose_last_tick_has_no_landmarks_still_sends_a_window_frame(conn):
    """Rewritten by F3, because F3 invalidated this test's original premise.

    It previously pinned that a window whose LAST tick found nobody at the desk
    made no Gemini call at all. That followed from _flush carrying frames and
    landmarks from the current tick: with no landmarks there, crop_to_person was
    never attempted, so there was nothing to send.

    F3 removed that coupling. The row _flush writes is about the WINDOW, not
    about the tick that happened to close it, and the window here contains a
    real posture observation with real landmarks. Sending that observation's
    frame is what makes the verdict describe the same thing as the local rating
    it is stored beside. Dropping the row instead would shrink the comparison
    set on exactly the windows that end with the user stepping away, which is a
    posture-correlated exclusion of the kind the unbiased roll exists to
    prevent.

    What survives unchanged is the property the old name was really protecting:
    CROP_UNAVAILABLE means "we had a person region and could not isolate it",
    so it must never be stamped on a tick where nothing was ever cropped.
    """
    monitor, client = build(conn, sample_rate=1.0)
    run(monitor)                                  # flushes immediately
    run(monitor)                                  # buffered: window holds this OK sample
    monitor._last_write_at -= 200                 # pretend the window elapsed
    monitor.detector.detect.return_value = None   # the last tick sees nobody
    client.assess.reset_mock()
    record = run(monitor)

    assert record.outcome is Outcome.OK           # the window's earlier tick still scored
    assert record.api_error is None               # no refusal is invented
    client.assess.assert_called_once()            # the window's own frame went out
    assert record.is_comparison_sample is True
    assert record.gemini_rating is Rating.DECENT


def test_an_empty_window_closed_by_an_absent_tick_never_calls_gemini(conn):
    """The genuine no-observation case, which F3 must not turn into a call.

    Here the window holds nothing at all, so the row written is the raw non-OK
    sample. There is no posture observation anywhere in it, and _flush must not
    reach for a frame it does not have.
    """
    monitor, client = build(conn, sample_rate=1.0, record_interval_s=0)
    monitor.detector.detect.return_value = None
    client.assess.reset_mock()
    record = run(monitor)

    assert record.outcome is Outcome.PERSON_ABSENT
    assert record.api_error is None
    assert record.is_comparison_sample is False
    client.assess.assert_not_called()


# --- the daily coaching budget, end to end through the monitor ---

def test_the_daily_coaching_budget_stops_gated_calls_for_the_rest_of_the_day(conn):
    # Without this the gated path is unbounded: every DECENT or POOR flush calls,
    # so the worst day costs several times the best one and nothing in the config
    # caps it.
    monitor, client = build(conn, sample_rate=0.0, record_interval_s=0,
                            max_coaching_calls_per_day=2)
    with patch("posture.monitor.score", return_value=_poor()):
        for _ in range(10):
            run(monitor)
    assert client.assess.call_count == 2


def test_a_zero_coaching_budget_means_no_gated_call_ever(conn):
    monitor, client = build(conn, sample_rate=0.0, record_interval_s=0,
                            max_coaching_calls_per_day=0)
    with patch("posture.monitor.score", return_value=_poor()):
        for _ in range(5):
            run(monitor)
    client.assess.assert_not_called()


def test_a_spent_coaching_budget_never_suppresses_the_comparison_sample(conn):
    # The budget bounds coaching. If it also censored sampled calls, the
    # comparison set would end early on exactly the days posture is worst, which
    # is the posture-correlated censoring the unbiased roll exists to prevent.
    monitor, client = build(conn, sample_rate=1.0, record_interval_s=0,
                            max_coaching_calls_per_day=0)
    with patch("posture.monitor.score", return_value=_poor()):
        for _ in range(5):
            run(monitor)
    assert client.assess.call_count == 5
    rows = recent_checks(conn, 5)
    assert all(row["is_comparison_sample"] == 1 for row in rows)


def test_comparison_sample_rate_zero_stops_comparison_sampling_only(conn):
    """The explicit contract for rate = 0.0, both halves of it.

    Zero rolls no comparison sample, so nothing reaches the agreement matrix.
    It does NOT silence coaching: that is what max_coaching_calls_per_day is
    for. Overloading the sample rate into a second off switch would make
    "sample nothing, still coach me" unstateable, and would hide the coaching
    path behind a knob whose name says nothing about it.
    """
    monitor, client = build(conn, sample_rate=0.0, record_interval_s=0,
                            max_coaching_calls_per_day=5)
    with patch("posture.monitor.score", return_value=_poor()):
        for _ in range(5):
            run(monitor)
    assert client.assess.call_count == 5           # coaching still happens
    rows = recent_checks(conn, 5)
    assert all(row["is_comparison_sample"] == 0 for row in rows)
    assert agreement_matrix(conn) == {}             # nothing in the comparison set


def test_gemini_disabled_still_overrides_every_other_setting(conn):
    # The master switch. A generous coaching budget and a full sample rate must
    # not resurrect a single call.
    import dataclasses

    monitor, client = build(conn, sample_rate=1.0, record_interval_s=0,
                            max_coaching_calls_per_day=100)
    monitor.config = dataclasses.replace(monitor.config, gemini_enabled=False)
    with patch("posture.monitor.score", return_value=_poor()):
        for _ in range(5):
            run(monitor)
    client.assess.assert_not_called()


# --- F3: the stored local rating is the MEDIAN of a window of samples, but the
# frame sent to Gemini was always the window's NEWEST. During a gradual
# degradation across a window, Gemini therefore sees a systematically worse
# posture than the median it is scored against. That is a bias, not noise, and
# it pushes disagreement into exactly the cell that would justify keeping
# Gemini at all: local GOOD, Gemini POOR.
#
# The reviewer's claim that this is a zero-cost fix was checked: SampleResult
# already carries frames and landmarks for every sample, _perform_check appends
# whole SampleResults to _window, and _flush threw all but the current tick's
# away at self._window.clear(). No extra capture, no extra memory.

# A window whose posture degrades steadily. Median 50.0, newest 40.0, so the
# old code showed Gemini a frame 10.0 degrees worse than the number it was
# compared against.
GRADIENT_CVA = [60.0, 55.0, 50.0, 45.0, 40.0]
WINDOW_MEDIAN_CVA = 50.0
WINDOW_NEWEST_CVA = 40.0


def _tagged(cva: float) -> np.ndarray:
    """A frame whose every pixel is the integer cva it was captured at.

    This is how the frame that reached the API is identified: crop_to_person is
    stubbed to a passthrough and encode_jpeg to a single tag byte, so the bytes
    Gemini received name the sample they came from. Nothing else in the pipeline
    can carry that identity, because a real JPEG of a synthetic frame is opaque.
    """
    return np.full((64, 64, 3), int(cva), dtype=np.uint8)


def _drive_window(monitor, cvas):
    """Run one warm-up flush, then a full window ending in a flush.

    Returns the flushed record. The warm-up matters: the first check of a run is
    always due to write, so without it the window under test would be one sample
    short and its median would fall between two samples.
    """
    monitor.detector.detect.side_effect = (
        lambda frame: upright_side(float(frame[0][0][0])))

    def tick(cva):
        with patch("posture.cameras.capture_roles",
                   return_value=({"side": _tagged(cva)}, None)):
            return monitor.run_once()

    tick(62.0)                              # warm-up: flushes, clears the window
    for cva in cvas[:-1]:
        tick(cva)                           # buffered into the window
    monitor._last_write_at -= 1000          # the record interval has elapsed
    return tick(cvas[-1])                   # appends, then flushes the whole window


@pytest.fixture
def gradient(conn):
    monitor, client = build(conn, sample_rate=1.0, record_interval_s=120)
    with patch("posture.monitor.crop_to_person", side_effect=lambda frame, lm: frame), \
         patch("posture.monitor.encode_jpeg",
               side_effect=lambda img: bytes([int(img[0][0][0])])):
        client.assess.reset_mock()
        record = _drive_window(monitor, GRADIENT_CVA)
    return monitor, client, record


def test_the_frame_sent_to_gemini_matches_the_median_it_is_compared_against(gradient):
    monitor, client, record = gradient

    # The row stores the window MEDIAN, which is what Gemini's verdict is filed
    # beside in the agreement matrix.
    assert record.metrics.cva_deg == pytest.approx(WINDOW_MEDIAN_CVA)

    sent = client.assess.call_args[0][0]["side"]
    frame_cva = float(sent[0])
    assert frame_cva == WINDOW_MEDIAN_CVA, (
        f"Gemini was shown a {frame_cva} degree frame while being compared "
        f"against a {record.metrics.cva_deg} degree median")


def test_the_frame_sent_to_gemini_is_not_the_newest_sample(gradient):
    """The old behaviour, named, so the fix cannot regress quietly.

    Pinned as a distinct value rather than an inequality: an inequality here
    would also pass if the code started sending some third frame for a third
    reason.
    """
    _, client, _ = gradient
    frame_cva = float(client.assess.call_args[0][0]["side"][0])
    assert frame_cva != WINDOW_NEWEST_CVA
    assert frame_cva == WINDOW_MEDIAN_CVA


def test_the_frame_offset_against_the_stored_median_is_zero(gradient):
    """The bias, quantified. Old: 10.0 degrees worse than the stored median."""
    _, client, record = gradient
    frame_cva = float(client.assess.call_args[0][0]["side"][0])
    assert frame_cva - record.metrics.cva_deg == pytest.approx(0.0, abs=1e-9)


def test_an_improving_window_is_debiased_in_the_other_direction_too(conn):
    """The fix must centre the frame, not simply pick an older sample.

    A window improving from 40 to 60 has its newest sample BETTER than the
    median. If the fix were "send the oldest" it would pass every test above
    and fail here, swapping one direction of bias for the other.
    """
    monitor, client = build(conn, sample_rate=1.0, record_interval_s=120)
    with patch("posture.monitor.crop_to_person", side_effect=lambda frame, lm: frame), \
         patch("posture.monitor.encode_jpeg",
               side_effect=lambda img: bytes([int(img[0][0][0])])):
        client.assess.reset_mock()
        record = _drive_window(monitor, list(reversed(GRADIENT_CVA)))

    assert record.metrics.cva_deg == pytest.approx(WINDOW_MEDIAN_CVA)
    assert float(client.assess.call_args[0][0]["side"][0]) == WINDOW_MEDIAN_CVA


def test_a_flat_window_sends_a_frame_from_the_window_it_describes(conn):
    """With every sample identical the choice is arbitrary, but it must be a
    window sample and it must be deterministic across runs."""
    monitor, client = build(conn, sample_rate=1.0, record_interval_s=120)
    with patch("posture.monitor.crop_to_person", side_effect=lambda frame, lm: frame), \
         patch("posture.monitor.encode_jpeg",
               side_effect=lambda img: bytes([int(img[0][0][0])])):
        client.assess.reset_mock()
        _drive_window(monitor, [50.0] * 5)

    assert float(client.assess.call_args[0][0]["side"][0]) == 50.0


def _offsets(cvas):
    """(newest offset, chosen offset) against the stored median, in degrees.

    Drives _representative directly rather than through the monitor, because
    the point here is the selection rule itself over window shapes the
    end-to-end fixture cannot express as cheaply.
    """
    from posture.baseline import median_metrics
    from posture.metrics import compute_side_metrics
    from posture.monitor import SampleResult, _representative

    samples = [
        SampleResult(Outcome.OK,
                     compute_side_metrics(upright_side(cva), facing_sign=None,
                                          require_calibrated=True),
                     None, ["side"], {}, {})
        for cva in cvas
    ]
    median = median_metrics([s.metrics for s in samples])
    return (samples[-1].metrics.cva_deg - median.cva_deg,
            _representative(samples, median).metrics.cva_deg - median.cva_deg)


def test_an_odd_window_puts_the_frame_exactly_on_the_stored_median():
    newest, chosen = _offsets([60.0, 55.0, 50.0, 45.0, 40.0])
    assert newest == pytest.approx(-10.0)
    assert chosen == pytest.approx(0.0, abs=1e-9)


def test_the_even_window_residual_cancels_across_the_direction_of_change():
    """An even window has no middle sample, so a residual offset is unavoidable.

    What makes it acceptable is that its SIGN follows the tie-break, not the
    direction the posture is moving, so a degrading window and an improving one
    lean opposite ways and cancel over a history. The newest-frame offset does
    the opposite: it always points the way the posture is going, which is why it
    accumulated. Both halves are pinned, because a fix that got only one of them
    right would be a bias with better manners.
    """
    down_newest, down_chosen = _offsets([60.0, 56.0, 52.0, 48.0, 44.0, 40.0])
    up_newest, up_chosen = _offsets([40.0, 44.0, 48.0, 52.0, 56.0, 60.0])

    # The old rule: the offset tracks the direction of change and never cancels.
    assert down_newest == pytest.approx(-10.0)
    assert up_newest == pytest.approx(+10.0)

    # The new rule: smaller, and opposite in sign for opposite directions.
    assert down_chosen == pytest.approx(+2.0)
    assert up_chosen == pytest.approx(-2.0)
    assert down_chosen + up_chosen == pytest.approx(0.0, abs=1e-9)
