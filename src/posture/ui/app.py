"""Menu bar app. The ONLY module allowed to touch AppKit state.

Everything the monitor thread wants to display arrives via callbacks that
marshal onto the main thread; the worker never mutates a menu item directly
(spec H6).
"""
from __future__ import annotations

import logging
import time

import AppKit
import rumps
from PyObjCTools import AppHelper

from posture import config as config_module
from posture.config import Config
from posture.gemini import GeminiClient
from posture.landmarks import PoseDetector
from posture.logging_setup import setup_logging
from posture.monitor import API_FAILURE_VALUES, Monitor
from posture.store.db import connect, migrate
from posture.store.queries import CheckRecord, active_baseline, recent_checks, today_counts
from posture.ui.labels import baseline_measuring_text, measuring_text
from posture.ui.notify import notify
from posture.ui.popup import show_popup

logger = logging.getLogger("posture")

INTERVAL_CHOICES = [15, 30, 60, 120, 300]

APP_NAME = "Stack"

# What the nudge SAYS, keyed by the rating that triggered it. Built from a table
# rather than from the rating's own name: "Sustained poor posture" was assembled
# out of Rating.value, which made the headline a verdict on the person, and a
# verdict someone is going to read many times a day while already in pain.
# These describe the body instead, in the same register as the tip
# underneath, which was always the best line in the panel. None is a real key
# here: an unmeasurable check must not be described as poor posture.
NUDGE_HEADLINES = {
    "GOOD": "Nicely stacked",
    "DECENT": "Starting to drift",
    "POOR": "Time to reset",
    None: "Nothing measured",
}


class PostureApp(rumps.App):
    def __init__(self, config: Config) -> None:
        # "▤" reads as stacked layers, which is the whole idea of the name, and
        # it is monochrome so it inherits the menu bar's own colour in both
        # appearances. The yoga emoji it replaces rendered in full colour, at a
        # size macOS picks, next to a row of monochrome system icons.
        super().__init__(APP_NAME, title="▤", quit_button=None)
        self.config = config
        self.conn = connect(config.db_path)
        migrate(self.conn)

        self._detector_ctx = PoseDetector(config.model_path)
        self.detector = self._detector_ctx.__enter__()

        self.notifications_enabled = True

        self.status_item = rumps.MenuItem("Status: Idle")
        self.status_item.set_callback(None)
        self.stats_item = rumps.MenuItem("Today: no data")
        self.stats_item.set_callback(None)
        self.baseline_item = rumps.MenuItem("Baseline: none")
        self.baseline_item.set_callback(None)
        # Be explicit about what the current setup can see. With a front camera
        # only, forward head posture and trunk lean are not measurable at all,
        # and silently omitting them would overstate what the score means.
        self.measuring_item = rumps.MenuItem("Measuring: unknown")
        self.measuring_item.set_callback(None)
        # A PERMANENT line, not a one-shot notification. v2 died because a
        # retired model turned every call into a silent failure and nothing on
        # screen ever said so. The launch notification alone does not fix that:
        # macOS drops notifications during Do Not Disturb and screen sharing,
        # and the model can just as easily be retired on the Thursday of a week
        # the app has been up since Monday. This line is recomputed on every
        # status update, so a failure that starts mid-run appears here too.
        self.gemini_item = rumps.MenuItem("Gemini: off")
        self.gemini_item.set_callback(None)
        # Last known Gemini failure. Sticky, because MODEL_UNAVAILABLE trips the
        # breaker PERMANENTLY: exactly one api_error row is ever written and
        # every row after it is a plain local check, so reading only the newest
        # row would show the failure once and then go quiet again. Cleared only
        # by evidence that a call has since come back.
        self._gemini_error: str | None = None

        self.toggle_item = rumps.MenuItem("Start Monitoring", callback=self.toggle)
        self.check_now_item = rumps.MenuItem("Check Now", callback=self.check_now)

        self.interval_menu = rumps.MenuItem("Sample Interval")
        for seconds in INTERVAL_CHOICES:
            label = f"{seconds}s" if seconds < 60 else f"{seconds // 60} min"
            item = rumps.MenuItem(label, callback=self.change_interval)
            item.seconds = seconds
            self.interval_menu.add(item)

        self.notif_item = rumps.MenuItem("Notifications: On", callback=self.toggle_notifications)
        self.dashboard_item = rumps.MenuItem("Open Dashboard", callback=self.open_dashboard)
        self.diagnostics_item = rumps.MenuItem(
            "Copy Diagnostics", callback=self.copy_diagnostics)
        self.quit_item = rumps.MenuItem("Quit", callback=self.quit_app)

        self.menu = [
            self.status_item, self.stats_item, self.baseline_item,
            self.measuring_item, self.gemini_item, None,
            self.toggle_item, self.check_now_item, None,
            self.interval_menu, self.notif_item, self.dashboard_item,
            self.diagnostics_item, None,
            self.quit_item,
        ]

        self.monitor = Monitor(
            config=config, detector=self.detector, conn=self.conn,
            on_nudge=self._on_nudge, on_status=self._on_status,
        )
        self.gemini = GeminiClient(config) if config.gemini_enabled else None
        self.monitor.gemini = self.gemini
        if self.gemini is not None:
            self.gemini_item.title = "Gemini: checking..."
            # preflight() is a blocking network call. Running it here, on the
            # AppKit thread before the menu is even live, would freeze launch
            # whenever the network is down or slow. Monitor.run_gemini_preflight_async
            # runs it on a tracked worker thread instead, the same way manual
            # checks already avoid blocking AppKit and stay visible to stop().
            self.monitor.run_gemini_preflight_async(self._on_gemini_preflight_result)
        self._refresh_baseline_label()
        self._mark_interval(config.sample_interval_s)

    # --- callbacks from the worker thread: marshal to main ---

    def _on_status(self, text: str) -> None:
        AppHelper.callAfter(self._apply_status, text)

    def _apply_status(self, text: str) -> None:
        self.status_item.title = f"Status: {text}"
        # Recalibrating while the app runs should be reflected without a restart.
        baseline = self._refresh_baseline_label()
        self._refresh_measuring_from_latest_check(baseline)
        self._refresh_gemini_label()
        counts = today_counts(self.conn)
        self.stats_item.title = (
            f"Today: {counts['good']} good, {counts['decent']} decent, {counts['poor']} poor"
        )

    def _on_nudge(self, record: CheckRecord) -> None:
        if not self.notifications_enabled:
            return
        AppHelper.callAfter(self._show_nudge, record)

    def _show_nudge(self, record: CheckRecord) -> None:
        driver = (record.local_driver or "posture").replace("_", " ")
        # local_rating is expected here, since only a POOR run reaches a nudge,
        # but a missing one must not raise inside the callback. Every other
        # display path in this app tolerates an absent value.
        rating = record.local_rating
        headline = NUDGE_HEADLINES.get(
            rating.value if rating is not None else None, NUDGE_HEADLINES[None])
        # Local measurement is the primary signal, so its driver is always
        # stated. Gemini's text is appended and labelled rather than swapped in:
        # letting it replace the local line would hide the measurement the score,
        # the trend, and the dashboard are all actually built from.
        details = f"Driven by {driver}."
        if record.gemini_details:
            details = f"{details} Gemini: {record.gemini_details}"
        # No leading "Reset:" any more. The badge above it already says "Reset",
        # and repeating the word turned the one genuinely useful line in the
        # panel into a second scold.
        tip = record.gemini_tip or "Feet flat, hips back, ears over shoulders."
        show_popup(rating, headline, details, tip, duration=12.0)

    def _on_gemini_preflight_result(self, outcome) -> None:
        # Called from the worker thread Monitor.run_gemini_preflight_async
        # started, not from AppKit, so every branch marshals. Success is worth
        # marshalling too: it is what turns "Gemini: checking..." into a
        # definite answer instead of leaving that line ambiguous forever.
        AppHelper.callAfter(self._apply_gemini_preflight_result, outcome)

    def _apply_gemini_preflight_result(self, outcome) -> None:
        if outcome is None:
            self._gemini_error = None
            self._refresh_gemini_label()
            return
        self._gemini_error = outcome.value
        self._refresh_gemini_label()
        # The banner IS set here. The earlier claim that _apply_status would
        # overwrite it anyway was false: _apply_status only runs once a check
        # has completed, and __init__ never starts monitoring. Left unset, a
        # preflight failure at launch showed nothing but "Status: Idle" and a
        # notification macOS is free to swallow.
        self.status_item.title = f"Status: Gemini unavailable ({outcome.value})"
        notify("Gemini unavailable",
               f"{outcome.value}. Local monitoring still works.")
        logger.error("gemini preflight failed: %s", outcome.value)

    def _refresh_gemini_label(self) -> None:
        """Keep the Gemini line honest about whether the second opinion works.

        Three sources, in order of how recent their evidence is: the newest
        check row (which is the only place a mid-run failure shows up), then the
        sticky last-known failure, then the circuit breaker, which is open
        whenever calls are being refused even if no row has recorded why yet.
        """
        if self.gemini is None:
            self.gemini_item.title = "Gemini: off"
            return

        latest = recent_checks(self.conn, 1)
        if latest:
            if latest[0]["api_error"] in API_FAILURE_VALUES:
                self._gemini_error = latest[0]["api_error"]
            elif latest[0]["gemini_model"]:
                # A verdict came back, so whatever failed before is over.
                self._gemini_error = None

        if self._gemini_error is None:
            if self.gemini.breaker.is_open(time.monotonic()):
                self.gemini_item.title = (
                    "Gemini: NOT working (repeated failures). "
                    "Local monitoring continues.")
            else:
                self.gemini_item.title = "Gemini: ok"
            return
        self.gemini_item.title = (
            f"Gemini: NOT working ({self._gemini_error}). "
            f"Local monitoring continues.")

    # --- menu actions (already on the main thread) ---

    def toggle(self, _sender) -> None:
        if self.monitor.is_running:
            self.monitor.stop()
            self.toggle_item.title = "Start Monitoring"
            self.status_item.title = "Status: Paused"
        else:
            if active_baseline(self.conn) is None:
                # Starting without a baseline writes rows that can never be
                # scored: they carry no baseline_id and no rating, and scoring
                # cannot be applied retroactively. Refusing is clearer than
                # silently accumulating junk.
                notify("Not calibrated",
                       "Run scripts/calibrate.py first, then start monitoring.")
                self.status_item.title = "Status: Not calibrated"
                return
            if not self.monitor.start():
                # A previous check is still finishing, usually a camera call that
                # outlasted the stop timeout. Say so rather than showing
                # "Monitoring" over a loop that is not running.
                self.status_item.title = "Status: Still finishing the last check"
                notify("Not started yet",
                       "A previous check is still finishing. Try again shortly.")
                return
            self.toggle_item.title = "Stop Monitoring"
            self.status_item.title = "Status: Monitoring"

    def check_now(self, _sender) -> None:
        # Tracked, so quit cannot tear down the detector underneath it. The
        # guarding and logging live in Monitor.run_once_async.
        self.monitor.run_once_async()

    def change_interval(self, sender) -> None:
        from dataclasses import replace
        self.config = replace(self.config, sample_interval_s=sender.seconds)
        self.monitor.config = self.config
        self.monitor.wake()  # apply immediately instead of after the current sleep
        self._mark_interval(sender.seconds)

    def _mark_interval(self, seconds: int) -> None:
        for item in self.interval_menu.values():
            if hasattr(item, "seconds"):
                base = item.title.replace(" ✓", "")
                item.title = base + (" ✓" if item.seconds == seconds else "")

    def toggle_notifications(self, _sender) -> None:
        self.notifications_enabled = not self.notifications_enabled
        state = "On" if self.notifications_enabled else "Off"
        self.notif_item.title = f"Notifications: {state}"

    def open_dashboard(self, _sender) -> None:
        self.monitor.run_dashboard_async(
            self.config.db_path.parent / "dashboard.html")

    def copy_diagnostics(self, _sender) -> None:
        def deliver(text: str) -> None:
            AppHelper.callAfter(self._put_on_clipboard, text)

        self.monitor.run_diagnostics_async(deliver)

    def _put_on_clipboard(self, text: str) -> None:
        pasteboard = AppKit.NSPasteboard.generalPasteboard()
        pasteboard.clearContents()
        pasteboard.setString_forType_(text, AppKit.NSPasteboardTypeString)
        notify(APP_NAME, "Diagnostics copied. Paste them wherever you need.")

    def _refresh_baseline_label(self):
        """Update the baseline line and hand the baseline back to the caller.

        Returned rather than re-queried, because the measuring line needs the
        same baseline and two reads could disagree if a recalibration landed
        between them.
        """
        baseline = active_baseline(self.conn)
        if baseline is None:
            self.baseline_item.title = "Baseline: none (run calibrate.py)"
            self.measuring_item.title = "Measuring: nothing yet"
            return None
        self.baseline_item.title = f"Baseline: {baseline.label} ({baseline.created_at[:10]})"

        # Fallback only, for the window between calibrating and the first check
        # completing. _refresh_measuring_from_latest_check overrides this with
        # what was actually just measured, and takes priority once a check exists.
        self.measuring_item.title = baseline_measuring_text(baseline.metrics.available())
        return baseline

    def _refresh_measuring_from_latest_check(self, baseline) -> None:
        """Describe what the most recent check actually saw, not the baseline.

        A baseline calibrated with a side camera keeps claiming forward head and
        lean are measured even after undocking, because that camera has not
        contributed to any check since. The latest check's capability is ground
        truth for the present; the baseline only describes what was true once, at
        calibration. Left untouched (falls through to the baseline-derived text
        set by _refresh_baseline_label) until at least one check has run.

        The capability alone is not enough, though. It says which cameras
        produced frames, and H6b already established that a side camera whose
        facing sign could not be frozen produces no trunk angle at all: _sample
        passes require_calibrated=True precisely so monitoring will not guess.
        Without the facing sign this line went on promising lean forever, which
        contradicts the baseline that had just refused to hold it.
        """
        latest = recent_checks(self.conn, 1)
        if not latest:
            return
        self.measuring_item.title = measuring_text(
            latest[0]["capability"],
            lean_measurable=baseline is not None and baseline.facing_sign is not None,
        )

    def quit_app(self, _sender) -> None:
        if not self.monitor.stop():
            # A check is wedged, almost always in camera I/O. Closing the detector
            # and the database now would race the worker's next detect() or
            # insert, and a concurrent close into the native MediaPipe binding is
            # undefined behaviour rather than a catchable exception. Leave both to
            # process teardown, which is safe because the thread is a daemon.
            logger.warning("a check is still running, quitting without teardown")
            rumps.quit_application()
            return
        self._detector_ctx.__exit__(None, None, None)
        self.conn.close()
        rumps.quit_application()


def report_startup_failure(headline: str, problem: object, context: str = "") -> None:
    """Say what went wrong, loudly, on the way to a CLEAN exit.

    Exiting non-zero here would be worse than useless. install_autostart.sh
    writes KeepAlive{SuccessfulExit: false} with ThrottleInterval 60, so launchd
    relaunches a non-zero exit every single minute, forever, with no menu bar
    icon and no message. A corrupt database, a malformed config.toml, or one
    typo'd camera role would each produce exactly that: an invisible crash loop.
    The model-missing path already returns 0 for this reason; every other
    startup failure now does the same.

    Three channels, because none of them is sufficient alone:

      the log         complete, but nobody reads it until they already suspect
                      a problem, which is the state we are trying to avoid;
      a notification  immediate, but macOS silently drops notifications during
                      Do Not Disturb, screen sharing, and focus modes;
      a modal alert   cannot be suppressed and holds the screen until it is
                      dismissed. This is the only one that guarantees he finds
                      out at all, and it is the reason exiting silently is not
                      an acceptable answer here.
    """
    if isinstance(problem, BaseException):
        detail = f"{type(problem).__name__}: {problem}"
    else:
        detail = str(problem)
    if context:
        # "file is not a database" does not say WHICH file, and a message that
        # names the fault without naming the thing at fault sends him hunting.
        detail = f"{detail}\n{context}"
    logger.error("startup failed, exiting: %s (%s)", headline, detail,
                 exc_info=isinstance(problem, BaseException))
    notify(headline, detail)
    try:
        rumps.alert(
            title=headline,
            message=(f"{detail}\n\n{APP_NAME} has stopped rather than "
                     f"restarting in a loop. Nothing is being measured until "
                     f"this is fixed.\n\nFor more: "
                     f"uv run python scripts/diagnostics.py"),
        )
    except Exception:
        # An alert needs a window server, which a launchd agent does not always
        # have. Losing the loudest channel must not cost us the other two.
        logger.exception("could not show the startup alert")


def main() -> None:
    try:
        config = Config.load()
    except Exception as exc:
        # setup_logging has not run yet, and it cannot: its path comes from the
        # config that just failed to load. Fall back to the built-in defaults,
        # which cannot raise, so the reason still reaches the log file.
        try:
            setup_logging(Config().log_path)
        except Exception:
            pass
        report_startup_failure(f"{APP_NAME} could not read its settings", exc,
                               # Read through the module, not bound at import:
                               # Config.load() resolves it the same way, so the
                               # path we name is always the path it just read.
                               context=f"Config file: {config_module.CONFIG_PATH}")
        return

    setup_logging(config.log_path)
    logger.info("%s v3 starting", APP_NAME)
    if not config.model_path.exists():
        report_startup_failure(
            f"{APP_NAME} is missing its pose model",
            f"No pose model at {config.model_path}. Run scripts/fetch_model.sh, "
            f"then start it again.")
        return
    try:
        app = PostureApp(config)
    except Exception as exc:
        # Opening and migrating the database, and opening the MediaPipe
        # detector, all happen inside __init__, before the menu bar icon exists.
        # Any of them can fail on a machine that has been running for months.
        report_startup_failure(
            f"{APP_NAME} could not start", exc,
            context=f"Database: {config.db_path}\nModel: {config.model_path}")
        return
    app.run()


if __name__ == "__main__":
    main()
