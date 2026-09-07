"""Tests for the popup panel that do not require a run loop or a live app.

Creating an NSPanel and laying it out with Auto Layout works fine outside an
event loop in this environment: no NSApp.run() is needed for any of it. Only
show_popup() itself needs to be dispatched onto a run loop via
AppHelper.callAfter, and these tests call _show_on_main / _dismiss directly to
avoid that.
"""
from __future__ import annotations

import Foundation
import pytest

from posture.ui import popup


@pytest.fixture(autouse=True)
def _clear_active():
    # _active is module-level state shared across every call. Never let one
    # test's leftover panel bleed into the next.
    popup._active.clear()
    yield
    popup._dismiss()


def test_a_failed_timer_still_leaves_the_panel_dismissable(monkeypatch):
    """Reproduces the orphaned-panel bug: NSTimer scheduling is fallible, and
    if it raises after the panel is already on screen, _active must still
    point at that panel so the next _dismiss() can close it.

    Before the fix, the panel was recorded in _active only after the timer
    call succeeded, so a raise here left _active empty even though a real
    panel was already on screen: the panel could never be found again, and
    every later nudge would stack another permanent window on top of it.

    Both halves matter and must be checked in the same test: checking only
    that _active ends up {} after _dismiss() would pass trivially even on the
    broken code, since _active was already empty going in, there being
    nothing to dismiss. This only means something if _active held the panel
    beforehand.
    """
    class RaisingNSTimer:
        @staticmethod
        def scheduledTimerWithTimeInterval_repeats_block_(*args, **kwargs):
            raise RuntimeError("timer scheduling failed")

    monkeypatch.setattr(Foundation, "NSTimer", RaisingNSTimer)

    # The failure path falls back to notify(), which really does post a macOS
    # notification via osascript. Left unpatched, every run of this suite buzzed
    # the machine with "Test headline. Test details" and a Funk sound, and
    # because osascript is the poster, clicking it opened Script Editor. Capture
    # the call instead, and assert it happened: the user being told is part of
    # the behaviour under test, not an incidental side effect.
    notified = []
    monkeypatch.setattr("posture.ui.notify.notify",
                        lambda title, message: notified.append((title, message)))

    popup._show_on_main("POOR", "Test headline", "Test details", "Test tip", 5.0)

    assert notified, "the failure path must still tell the user something happened"

    panel = popup._active.get("panel")
    assert panel is not None, (
        "the panel was orphaned: it is on screen but _active lost track of it"
    )
    assert popup._active.get("timer") is None

    # Now prove that panel is actually reachable and closeable by the very
    # next call, which is what the next nudge's show_popup does first.
    popup._dismiss()
    assert popup._active == {}
