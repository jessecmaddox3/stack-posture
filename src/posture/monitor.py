"""The sample loop. Runs on a worker thread and never touches AppKit.

v2 mutated menu bar titles directly from worker threads (spec H6), ran manual
and scheduled checks concurrently against the same camera (spec H7), and slept
in uninterruptible one-second ticks so interval changes took up to half an hour
to apply (spec H9). All three are fixed here.
"""
from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Callable

from posture import cameras
from posture.baseline import median_metrics
from posture.config import Config
from posture.crop import CropUnavailable, crop_to_person, encode_jpeg
from posture.gating import decide, roll_sample
from posture.gemini import GeminiClient
from posture.landmarks import PoseDetector
from posture.metrics import compute_front_metrics, compute_side_metrics, merge_metrics
from posture.scoring import METRIC_RULES, LocalAssessment, score
from posture.store.queries import (
    CheckRecord,
    active_baseline,
    coaching_calls_today,
    insert_check,
    seconds_since_last_api_call,
)
from posture.types import Outcome, PostureMetrics, Rating

logger = logging.getLogger("posture")

# How many record windows a Gemini verdict stays usable as nudge coaching.
# A verdict describes the window it was measured from, and the nudge runs on the
# SAMPLE cadence, not the record cadence, so the freshest verdict available to a
# nudge is always at least part of a window old. Two windows (four minutes at the
# shipped record_interval_s of 120s) is long enough that a sustained slouch still
# gets the text that was paid for, and short enough that he is never coached
# about a posture he left several minutes ago. Derived from the cadence rather
# than configured separately so retuning record_interval_s cannot silently turn
# this into hours.
COACHING_TTL_WINDOWS = 2

# api_error values that mean "the second opinion is not working". Deliberately
# NOT every value the column can hold: CROP_UNAVAILABLE is a local refusal to
# send an untrustworthy crop, recorded in the same column, and the API was never
# contacted for it. Shared with the menu bar so the two cannot drift apart.
API_FAILURE_VALUES = frozenset({Outcome.API_ERROR.value, Outcome.MODEL_UNAVAILABLE.value})


@dataclass(frozen=True)
class Coaching:
    """The last Gemini verdict, held so a nudge can show what a call already bought.

    Verdicts arrive on the RECORD cadence (_flush, every record_interval_s) but
    nudges fire on the SAMPLE cadence (every sample_interval_s, once a slouch is
    sustained). The two do not line up, which is why _build_record cannot fill
    these fields in from the sample in front of it: at nudge time there is no
    verdict for THAT sample and there never will be. Carrying the most recent one
    forward, with a freshness bound, is the only way the coaching a gated call
    exists to produce can reach the popup at all.
    """
    rating: Rating | None
    issue: str | None
    details: str | None
    tip: str | None
    model: str | None
    at: float                    # time.monotonic() when the verdict came back


class SlouchTracker:
    """Decide when sustained bad posture warrants a nudge. Pure, fake-clock friendly."""

    def __init__(self, sustained_samples: int, cooldown_s: float) -> None:
        self._needed = max(1, sustained_samples)
        self._cooldown_s = cooldown_s
        self._streak = 0
        self._last_nudge_at: float | None = None

    def reset(self) -> None:
        self._streak = 0

    def update(self, rating: Rating | None, now: float) -> bool:
        """Feed one sample. Returns True when a nudge should fire."""
        if rating in (Rating.DECENT, Rating.POOR):
            self._streak += 1
        elif rating is Rating.GOOD:
            self._streak = 0
        else:
            # Unmeasurable: hold the streak rather than advancing or clearing it.
            # Advancing would let a run of unreadable frames trigger a nudge with
            # no evidence; clearing would let a glance away reset a real slouch.
            return False

        if self._streak < self._needed:
            return False
        if self._last_nudge_at is not None and (now - self._last_nudge_at) < self._cooldown_s:
            return False
        self._last_nudge_at = now
        self._streak = 0
        return True


def _distance_from(metrics: PostureMetrics, reference: PostureMetrics) -> float:
    """How far one sample's metrics sit from a reference, in comparable units.

    Metrics are measured in different things (degrees, ratios), so a raw sum of
    absolute differences would let whichever metric has the largest numeric
    range decide the answer. Each difference is divided by that metric's own
    decent_at threshold, which is the amount of drift scoring already treats as
    the first step away from baseline. That makes a degree of craniovertebral
    angle and a unit of proximity ratio commensurate on the only scale this
    system has an opinion about.

    Metrics missing from either side are skipped, exactly as score() skips
    them: a sample that could not measure something contributes no evidence
    about it rather than a default.
    """
    current, base = asdict(metrics), asdict(reference)
    total = 0.0
    for name, rule in METRIC_RULES.items():
        now, then = current.get(name), base.get(name)
        if now is None or then is None:
            continue
        total += abs(now - then) / rule.decent_at
    return total


def _representative(window: list["SampleResult"],
                    metrics: PostureMetrics) -> "SampleResult":
    """The window sample whose own metrics sit closest to the window median.

    The row written from a window stores the MEDIAN of its samples, and that
    median is the local rating a Gemini verdict is filed beside in the
    agreement matrix. Sending the window's NEWEST frame made those two describe
    different moments, and not symmetrically: while posture degrades across a
    window (which is the normal shape of a slouch developing), the newest frame
    is always worse than the median, so Gemini was systematically shown a worse
    posture than the number it was scored against. Measured on a window
    degrading from 60 to 40 degrees of craniovertebral angle: the newest frame
    was 10.0 degrees below the stored 50.0 median. That bias lands in the local
    GOOD / Gemini POOR cell, which is precisely the cell that would justify
    paying for a second opinion, so the statistic was manufacturing its own
    case for itself.

    Costs nothing: every sample's frames and landmarks are already held in the
    window and were being discarded at _window.clear().

    Ties break toward the window's temporal CENTRE, then toward the earlier
    sample. An even-length window has no single middle sample, so something has
    to decide, and no choice is exactly unbiased there. Measured on a six-sample
    window moving 60 to 40 degrees: the newest frame sits 10.0 degrees from the
    stored median, this sits 2.0. The residual matters less than the number
    suggests, because its SIGN follows the tie-break rather than the direction
    of change: a degrading window resolves 2.0 degrees BETTER than its median
    and an improving one 2.0 degrees WORSE, so the two cancel across a history.
    The newest-frame offset does not cancel, because it always points the same
    way the posture is moving and slouching windows outnumber recovering ones.
    Resolving toward the newest would reinstate exactly that.

    This does not touch the comparison-set ROLL. Which windows get compared is
    decided in _maybe_assess_with_gemini, before the rating is consulted and
    independently of anything here; this only changes WHICH frame of an
    already-selected window is the one described.
    """
    usable = [(index, sample) for index, sample in enumerate(window)
              if sample.metrics is not None]
    centre = (len(window) - 1) / 2
    return min(usable, key=lambda pair: (_distance_from(pair[1].metrics, metrics),
                                         abs(pair[0] - centre), pair[0]))[1]


@dataclass
class SampleResult:
    outcome: Outcome
    metrics: PostureMetrics | None
    assessment: LocalAssessment | None
    cameras: list[str]
    frames: dict
    landmarks: dict          # role -> Landmarks, needed to crop before sending


class Monitor:
    def __init__(self, config: Config, detector: PoseDetector, conn,
                 on_nudge: Callable[[CheckRecord], None],
                 on_status: Callable[[str], None],
                 gemini_client: GeminiClient | None = None,
                 rng: random.Random | None = None) -> None:
        self.config = config
        self.detector = detector
        self.conn = conn
        self.on_nudge = on_nudge
        self.on_status = on_status
        self._rng = rng or random.Random()
        self._tracker = SlouchTracker(config.sustained_samples, config.nudge_cooldown_s)
        self._check_lock = threading.Lock()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._workers: set[threading.Thread] = set()
        self._workers_lock = threading.Lock()
        self._running = False
        self._window: list[SampleResult] = []
        self._last_write_at: float | None = None
        self._last_coaching: Coaching | None = None
        self.gemini = gemini_client

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> bool:
        """Start the loop. True if monitoring is active afterwards, False if not.

        Returning None would let a caller flip its UI to "Monitoring" while
        nothing actually runs, which is a worse failure than the double loop the
        guard below replaced: at least that one over-delivered. Already-running
        returns True, because the question a caller is really asking is whether
        monitoring is on now, not whether this particular call did the starting.

        Refusal happens when a previous stop() timed out and its worker is still
        alive. It resolves itself once that worker's blocking call returns, since
        _running is already False so the loop exits on its next check.
        """
        if self._running:
            return True
        if self._thread is not None and self._thread.is_alive():
            logger.warning("previous monitor thread still running, not starting another")
            return False

        # A pause is a GAP IN OBSERVATION, not a pause in time, so nothing
        # measured before it describes now. Carrying the half-filled window
        # across a pause made the next written row a median of samples from
        # either side of the gap, stamped with a single timestamp as if it were
        # one continuous observation. time.monotonic() does not advance while
        # the system sleeps, so a lid-close gap is unbounded: it can straddle
        # midnight, and a recalibration, and still look like two minutes.
        # _last_write_at goes with it, or the resumed run would keep waiting out
        # a record interval that started before the pause. The nudge streak and
        # the cached Gemini verdict go too: three bad samples from before lunch
        # are not evidence about this afternoon, and coaching text bought for a
        # posture he has since left must not be recycled into the next nudge.
        self._window.clear()
        self._last_write_at = None
        self._tracker.reset()
        self._last_coaching = None

        self._running = True
        self._wake.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="posture-loop")
        self._thread.start()
        return True

    def stop(self, timeout: float = 5.0) -> bool:
        """Stop the loop and wait for the worker to exit.

        If the join times out (a check stuck in camera I/O, say) the thread
        reference is KEPT rather than cleared. join() does not raise or otherwise
        signal a timeout, so clearing unconditionally would let a later start()
        spawn a second loop while the first was still alive. Worse, when the stuck
        thread finally returned it would see _running set True again by that
        start() and carry on looping: two independent loops at double the cadence,
        only one of them joinable.

        Returns False when the join timed out and the worker is still alive, or
        when a manual check (run_once_async) is still running. Callers that are
        about to tear down shared resources (the MediaPipe detector, the database
        connection) MUST check this: closing them while a wedged worker may still
        call into them races a native binding, which is undefined behaviour
        rather than a catchable exception.

        The scheduled-loop join happens first, but even when there is no loop
        thread to join (already stopped, or never started) this still falls
        through to the manual-check join below, rather than returning True early.
        That early return was the exact path a manual check running outside the
        lifecycle could slip through.
        """
        self._running = False
        self._wake.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning("monitor thread did not exit within %.1fs, keeping "
                               "the reference so start() will refuse", timeout)
                return False
            self._thread = None

        with self._workers_lock:
            workers = list(self._workers)
        for worker in workers:
            worker.join(timeout=timeout)
        if any(w.is_alive() for w in workers):
            logger.warning("a manual check is still running after %.1fs", timeout)
            return False
        return True

    def _report_failure(self, headline: str, exc: BaseException) -> None:
        """Put a failure where the user will actually see it: the menu bar.

        Logging alone does not count. Nobody opens the log until they already
        suspect something is wrong, and the whole point of this app is that the
        previous version gave him no reason to suspect anything for two months.
        The exception TYPE goes in the text so the menu says what kind of
        failure it is, not merely that there was one; the log has the traceback.

        on_status is a caller-supplied callback, so it can raise. If it did, and
        we were already inside an except block, that would kill the loop while
        reporting that the loop had a problem.
        """
        try:
            self.on_status(f"{headline}: {type(exc).__name__}. See the log.")
        except Exception:
            logger.exception("status callback failed while reporting %s", headline)

    def _loop(self) -> None:
        try:
            while self._running:
                try:
                    self.run_once()
                except Exception as exc:
                    logger.exception("check failed")
                    # Not just logged. A raising check used to leave the menu
                    # showing the last good reading forever: the disk filled at
                    # 3pm and it still said "Good posture" all evening.
                    self._report_failure("Last check failed", exc)
                # Interruptible sleep, INSIDE the try: stop() and interval
                # changes apply immediately, but a bad sample_interval_s (a
                # quoted number in config.toml, say) raises HERE rather than in
                # the check. Outside the try, that killed the worker after one
                # pass and nothing said so.
                self._wake.wait(timeout=self.config.sample_interval_s)
                self._wake.clear()
        except Exception as exc:
            logger.exception("monitor loop died")
            self._report_failure("Monitoring stopped", exc)
        finally:
            # The loop is gone, whichever way it went, so is_running must stop
            # claiming otherwise. Leaving it True left the menu reading
            # "Monitoring" all day over a thread that had already died, and left
            # start() refusing to restart because it believed one was running.
            self._running = False

    def wake(self) -> None:
        """Interrupt the current sleep, e.g. after an interval change."""
        self._wake.set()

    def run_once(self) -> CheckRecord:
        """One full check. Serialised so a manual check cannot race the loop."""
        with self._check_lock:
            return self._perform_check()

    def run_once_async(self) -> None:
        """Run one check on a TRACKED worker thread.

        Manual checks must be visible to stop(), because quit_app tears down the
        detector and the database once stop() returns. An untracked thread would
        be torn down underneath it, and a concurrent close into the native
        MediaPipe binding is undefined behaviour rather than a catchable
        exception.
        """
        def guarded() -> None:
            try:
                self.run_once()
            except Exception as exc:
                # The scheduled loop logs its own failures, this must too, or a
                # failed manual check reaches only stderr, which for a login item
                # nobody reads.
                logger.exception("manual check failed")
                self._report_failure("Last manual check failed", exc)
            finally:
                with self._workers_lock:
                    self._workers.discard(threading.current_thread())

        thread = threading.Thread(target=guarded, daemon=True, name="posture-manual")
        # Add AND start under the lock. Adding first and starting after would let
        # a concurrent stop() snapshot the set in between and then join a thread
        # that was never started, which raises RuntimeError out of quit_app.
        # Holding the lock across start() is safe: start() does not touch this
        # lock, and if the worker finishes instantly its finally simply waits the
        # microsecond until this block releases.
        with self._workers_lock:
            self._workers.add(thread)
            thread.start()

    def run_gemini_preflight_async(self, on_result: Callable[[Outcome | None], None]) -> None:
        """Run GeminiClient.preflight() on a TRACKED worker thread.

        preflight() is a blocking network call, so running it on the caller's
        thread (typically AppKit's, at launch) would freeze that thread until
        the network answers or times out. It is tracked in the same _workers
        set as a manual check for the same reason run_once_async is: an
        untracked thread would still be running when quit_app tears down
        shared resources, and stop() would have no way to know to wait for it.

        on_result is called from THIS worker thread, not the caller's. Callers
        that need the main thread (AppKit apps do) must marshal inside
        on_result themselves, the same way _on_nudge and _on_status already do.
        """
        def guarded() -> None:
            try:
                outcome = self.gemini.preflight()
            except Exception:
                logger.exception("gemini preflight failed")
                outcome = Outcome.API_ERROR
            finally:
                with self._workers_lock:
                    self._workers.discard(threading.current_thread())
            try:
                on_result(outcome)
            except Exception:
                # Same reasoning as run_once_async: this runs on a worker
                # thread, so an escaping exception goes to the default
                # threading excepthook and out to stderr, which for a login
                # item nobody reads. The two entry points are guarded alike.
                logger.exception("gemini preflight callback failed")

        thread = threading.Thread(target=guarded, daemon=True,
                                  name="posture-gemini-preflight")
        with self._workers_lock:
            self._workers.add(thread)
            thread.start()

    def run_dashboard_async(self, output_path: Path) -> None:
        """Generate and open the dashboard on a TRACKED worker thread.

        Generation reads the whole history and writes a file, so it is far too
        slow for the AppKit thread. It is tracked for the same reason every
        other worker here is: quit_app closes the shared SQLite connection as
        soon as stop() returns, and an untracked thread would still be reading
        from it.
        """
        def guarded() -> None:
            try:
                # Imported here, not at module level, on purpose: this is the
                # patch target tests/test_monitor.py uses to stand in a fake
                # open_dashboard. Hoisting the import to the top of this file
                # would bind the real posture.dashboard.generate.open_dashboard
                # into this closure before any test gets a chance to patch it,
                # which would call webbrowser.open for real during the suite.
                from posture.dashboard.generate import open_dashboard
                open_dashboard(self.conn, output_path)
            except Exception:
                logger.exception("dashboard generation failed")
            finally:
                with self._workers_lock:
                    self._workers.discard(threading.current_thread())

        thread = threading.Thread(target=guarded, daemon=True,
                                  name="posture-dashboard")
        with self._workers_lock:
            self._workers.add(thread)
            thread.start()

    def run_diagnostics_async(self, on_result: Callable[[str], None]) -> None:
        """Build the diagnostics bundle on a TRACKED worker thread.

        Camera discovery probes up to 8 indices and can take seconds, so this
        never belongs on the AppKit thread. Tracked for the same reason as every
        other worker here: quit_app closes the shared SQLite connection as soon
        as stop() returns.
        """
        def guarded() -> None:
            try:
                from posture import cameras
                from posture.diagnostics import build_diagnostics, read_log_tail
                text = build_diagnostics(
                    self.conn, self.config,
                    read_log_tail(self.config.log_path), cameras.discover())
            except Exception as exc:
                logger.exception("diagnostics failed")
                text = f"diagnostics failed: {exc}"
            finally:
                with self._workers_lock:
                    self._workers.discard(threading.current_thread())
            on_result(text)

        thread = threading.Thread(target=guarded, daemon=True,
                                  name="posture-diagnostics")
        with self._workers_lock:
            self._workers.add(thread)
            thread.start()

    def _perform_check(self) -> CheckRecord:
        sample = self._sample()

        # Fast path, every sample: nudge tracking and the menu bar.
        if sample.assessment is not None:
            if sample.assessment.rating is Rating.GOOD:
                # He fixed it. Anything Gemini said about the previous slouch
                # describes a posture he has left, so it must not be recycled
                # into whatever slouch comes next.
                self._last_coaching = None
            if self._tracker.update(sample.assessment.rating, time.monotonic()):
                self.on_nudge(self._with_coaching(
                    self._build_record(sample, active_baseline(self.conn))))
        elif sample.outcome is not Outcome.OK:
            self._tracker.reset()

        if sample.outcome is Outcome.OK:
            self._window.append(sample)

        record = self._flush(sample)
        self.on_status(self._status_text(record or self._build_record(
            sample, active_baseline(self.conn))))
        return record or self._build_record(sample, active_baseline(self.conn))

    def _due_to_write(self, now: float) -> bool:
        if self._last_write_at is None:
            return True
        return (now - self._last_write_at) >= self.config.record_interval_s

    def _flush(self, sample: SampleResult) -> CheckRecord | None:
        """Write one row summarising the window, or None if the window is still open."""
        now = time.monotonic()
        if not self._due_to_write(now):
            return None
        self._last_write_at = now

        baseline = active_baseline(self.conn)
        # _perform_check appends OK samples before calling _flush, so an empty
        # window here means the current sample is non-OK. Writing that raw sample
        # keeps a persistent failure visible in the data: if every sample is
        # CAMERA_BUSY the window never fills, and each due tick records the
        # failure rather than nothing at all.
        window = self._window

        if window:
            metrics = median_metrics([s.metrics for s in window if s.metrics])
            # The frame that may go to Gemini comes from the sample nearest that
            # median, NOT from the current tick. See _representative: the row
            # stores the median, so sending the newest frame described a
            # different moment than the rating it would be compared against, and
            # biased the comparison toward Gemini seeing the worse end of every
            # degrading window.
            chosen = _representative(window, metrics)
            # Score the AGGREGATED metrics rather than taking a mode over the
            # per-sample ratings. Those two do not commute: samples alternating
            # which metric is bad yield a modal POOR whose medians both sit at
            # baseline, so the row contradicts itself. Scoring the aggregate once
            # makes every stored row reproducible from its own numbers, which is
            # what a clinical record has to be.
            aggregated = SampleResult(
                outcome=Outcome.OK,
                metrics=metrics,
                assessment=score(metrics, baseline) if baseline is not None else None,
                cameras=sorted({c for s in window for c in s.cameras}),
                frames=chosen.frames,
                landmarks=chosen.landmarks,
            )
        else:
            aggregated = sample

        record = self._build_record(aggregated, baseline)
        if aggregated.outcome is Outcome.OK:
            record = self._maybe_assess_with_gemini(record, aggregated)
        insert_check(self.conn, record)
        self._window.clear()
        return record

    def _sample(self) -> SampleResult:
        frames, outcome = cameras.capture_roles(self.config)
        if outcome is not None:
            return SampleResult(outcome, None, None, [], {}, {})

        baseline = active_baseline(self.conn)
        parts, used, detected = [], [], {}
        for role, frame in frames.items():
            lm = self.detector.detect(frame)
            if lm is None:
                continue
            used.append(role)
            detected[role] = lm
            if role == "front":
                parts.append(compute_front_metrics(lm))
            else:
                # The sign frozen at calibration, never a per-frame guess: a head
                # turn can invert the per-frame estimate while the trunk has not
                # moved. This is the only place that distinction is observable.
                # require_calibrated=True: by now the baseline either has a
                # frozen sign or it does not, and guessing from the per-frame
                # estimate is exactly what freezing exists to avoid.
                facing = baseline.facing_sign if baseline is not None else None
                parts.append(compute_side_metrics(lm, facing_sign=facing,
                                                  require_calibrated=True))

        if not parts:
            return SampleResult(Outcome.PERSON_ABSENT, None, None, [], frames, {})

        metrics = merge_metrics(*parts)
        assessment = score(metrics, baseline) if baseline is not None else None
        return SampleResult(Outcome.OK, metrics, assessment, used, frames, detected)

    def _maybe_assess_with_gemini(self, record: CheckRecord, sample: SampleResult
                                  ) -> CheckRecord:
        """Decide, then call. The sample roll happens FIRST, independent of the rating.

        Rolling after consulting the rating would bias the comparison set toward
        postures local already flagged, hiding the disagreement that matters most.

        Two paths reach the API from here and they cost very different amounts.
        The comparison sample is bounded by its rate alone. The gated coaching
        call is bounded by config.max_coaching_calls_per_day, read fresh from the
        database on every decision rather than counted in memory: the app is a
        login item that restarts, and an in-memory counter would reset the day's
        spend on every relaunch.
        """
        if self.gemini is None or not self.gemini.available:
            return record

        sampled = roll_sample(self.config.comparison_sample_rate, self._rng)
        local_rating = sample.assessment.rating if sample.assessment else None
        decision = decide(
            local_rating,
            sampled=sampled,
            seconds_since_last_call=seconds_since_last_api_call(self.conn),
            min_gap_s=self.config.min_seconds_between_api_calls,
            gemini_enabled=self.config.gemini_enabled,
            circuit_open=self.gemini.breaker.is_open(time.monotonic()),
            coaching_calls_today=coaching_calls_today(self.conn),
            max_coaching_calls_per_day=self.config.max_coaching_calls_per_day,
        )
        if not decision.should_call:
            logger.debug("skipping gemini (%s)", decision.reason)
            return record

        jpegs = {}
        attempted = False
        for role, frame in sample.frames.items():
            lm = sample.landmarks.get(role)
            if lm is None:
                continue
            attempted = True
            try:
                jpegs[role] = encode_jpeg(crop_to_person(frame, lm))
            except CropUnavailable as exc:
                # No trustworthy person region, so this role has nothing safe to
                # send. Skip the role rather than sending the uncropped frame,
                # which is the room behind the desk.
                logger.info("skipping %s frame: %s", role, exc)

        if not attempted:
            # No role had landmarks at all, so crop_to_person was never called.
            # That is not a crop REFUSAL (which means "we had a person region
            # and could not isolate it"): it is the absence of a posture
            # observation entirely. Stamping CROP_UNAVAILABLE here would be
            # false data, blaming a refusal that never happened, and marking it
            # a comparison sample would compare Gemini against a local rating
            # that was never about an observation at all.
            #
            # This used to be REACHABLE, and it was the common case: _flush
            # carried frames and landmarks from the current tick, so a window
            # whose last tick found nobody arrived here with nothing to send
            # even though earlier ticks in that window had scored fine. F3
            # removed that coupling. _flush now sends the frame of the window
            # sample nearest the median it is writing, and every sample in a
            # window is OK and therefore has landmarks, so this branch is now
            # defensive rather than load-bearing. It is kept because the cost of
            # being wrong about that is a false CROP_UNAVAILABLE in the data
            # the user reads to judge their own progress.
            return record

        if not jpegs:
            # At least one role had landmarks, but crop_to_person found no
            # trustworthy person region in any of them: a genuine refusal, not
            # an absence of an attempt. We never contacted the API, so record
            # that distinctly: calling it an API error would blame Gemini for a
            # local refusal and would feed the circuit breaker with failures the
            # API never caused.
            #
            # Still mark is_comparison_sample. The roll already happened, and it
            # happened BEFORE the local rating was consulted. Dropping the row
            # silently would quietly shrink the comparison set, and crop failure
            # is not independent of posture: it correlates with poor detection,
            # which correlates with the extreme postures we most want compared.
            return replace(record, api_error=Outcome.CROP_UNAVAILABLE.value,
                           is_comparison_sample=decision.is_comparison_sample)

        logger.info("calling gemini (%s)", decision.reason)
        verdict, outcome = self.gemini.assess(jpegs)

        if outcome is not None:
            return replace(record, api_error=outcome.value,
                           is_comparison_sample=decision.is_comparison_sample)

        # Cache every successful verdict, sampled or gated. Reading a verdict
        # back out here cannot bias anything: membership in the comparison set
        # was decided by a roll that already happened, upstream of the rating and
        # upstream of this line. A comparison-sample verdict is coaching text we
        # have already paid for, so refusing to show it would waste a call for no
        # gain in honesty.
        self._last_coaching = Coaching(
            rating=verdict.rating, issue=verdict.issue, details=verdict.details,
            tip=verdict.tip, model=self.config.gemini_model, at=time.monotonic())

        return replace(
            record,
            gemini_rating=verdict.rating,
            gemini_issue=verdict.issue,
            gemini_details=verdict.details,
            gemini_tip=verdict.tip,
            gemini_model=self.config.gemini_model,
            is_comparison_sample=decision.is_comparison_sample,
        )

    def _with_coaching(self, record: CheckRecord) -> CheckRecord:
        """Attach the most recent Gemini verdict to a NUDGE record, if still fresh.

        Only the nudge path uses this. _flush must not: the row it writes is the
        record of one specific window, and stamping it with a verdict measured
        from a different window would fabricate history, and worse, a verdict
        copied onto a row that never made a call would corrupt the agreement
        matrix. Nudge records are display-only and never inserted, which is what
        makes carrying a verdict forward safe here and nowhere else.

        is_comparison_sample is deliberately left alone. The roll that decides
        comparison-set membership happened for the ORIGINAL record, and this
        record is not that observation.
        """
        coaching = self._last_coaching
        if coaching is None:
            return record
        ttl = self.config.record_interval_s * COACHING_TTL_WINDOWS
        if (time.monotonic() - coaching.at) > ttl:
            self._last_coaching = None
            return record
        return replace(
            record,
            gemini_rating=coaching.rating,
            gemini_issue=coaching.issue,
            gemini_details=coaching.details,
            gemini_tip=coaching.tip,
            gemini_model=coaching.model,
        )

    def _build_record(self, sample: SampleResult, baseline) -> CheckRecord:
        now = datetime.now()
        assessment = sample.assessment
        return CheckRecord(
            ts=now.isoformat(timespec="seconds"),
            date=date.today().isoformat(),
            outcome=sample.outcome,
            baseline_id=baseline.id if baseline else None,
            cameras=sample.cameras,
            metrics=sample.metrics,
            local_rating=assessment.rating if assessment else None,
            local_driver=assessment.driver if assessment else None,
            gemini_rating=None, gemini_issue=None, gemini_details=None,
            gemini_tip=None, gemini_model=None,
            is_comparison_sample=False, api_error=None,
        )

    @staticmethod
    def _status_text(record: CheckRecord) -> str:
        """What the menu bar says after a check, INCLUDING a failing second opinion.

        The local reading leads, because it is the primary signal and it stays
        honest whatever Gemini does. But "Good posture" on its own, at a moment
        when every Gemini call is failing, is the shape of v2's defining
        failure: a retired model, a permanently open breaker, a comparison set
        that quietly stopped growing, and nothing on screen that said so.

        Only genuine API failures are named here. CROP_UNAVAILABLE also lands in
        api_error, but it is a LOCAL refusal to send an untrustworthy crop and
        the API was never contacted, so blaming Gemini for it would be false.
        """
        text = Monitor._local_status_text(record)
        if record.api_error in API_FAILURE_VALUES:
            return f"{text} (Gemini: {record.api_error})"
        return text

    @staticmethod
    def _local_status_text(record: CheckRecord) -> str:
        if record.outcome is Outcome.CAMERA_BUSY:
            return "Paused: camera in use"
        if record.outcome is Outcome.CAMERA_UNAVAILABLE:
            return "No camera found"
        if record.outcome is Outcome.PERSON_ABSENT:
            return "No one at the desk"
        if record.local_rating is None:
            # Two different situations reach here, with two different fixes.
            #
            # No baseline at all: calibrate. That instruction is correct and
            # actionable, and it is the only case that should get it.
            #
            # A baseline exists but score() found no metric the two have in
            # common, so it returned rating None rather than pretending. This
            # happens for real: a baseline calibrated with the side camera
            # holds cva_deg and trunk_angle_deg, and a day the side camera is
            # unplugged (or cannot see an ear) produces front metrics only, so
            # there is nothing to compare against. Saying "Not calibrated yet"
            # here is a dead end. He HAS calibrated, so he recalibrates, and
            # recalibrating changes nothing unless he does it on the setup that
            # is actually running now. Naming the real situation is what makes
            # the fix findable.
            if record.baseline_id is None:
                return "Not calibrated yet"
            return "Baseline does not cover this camera setup"
        return {
            Rating.GOOD: "Good posture",
            Rating.DECENT: "Decent posture",
            Rating.POOR: "Poor posture",
        }[record.local_rating]
