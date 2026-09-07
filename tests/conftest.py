import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(autouse=True)
def block_user_visible_notifications(monkeypatch):
    """Never let the suite post a real macOS notification.

    notify() shells out to `osascript -e 'display notification ...'`, which is a
    genuinely user-visible side effect: it lands in Notification Center with a
    sound, and because osascript posts it, clicking it opens Script Editor on a
    blank document.

    tests/test_popup.py drives the popup's exception path on purpose, and that
    path falls back to notify(). So every full run of this suite fired a real
    "Test headline. Test details" notification at whoever was using the machine.
    A user seeing these would reasonably assume the old app had come back to life.

    Blocking it here rather than in one test file is deliberate: any future test
    that reaches a notification path should fail loudly rather than quietly
    buzzing someone's desktop.
    """
    real_run = subprocess.run

    def guarded_run(args, *rest, **kwargs):
        if isinstance(args, (list, tuple)) and args and args[0] == "osascript":
            raise AssertionError(
                "a test tried to post a real macOS notification via osascript. "
                "Patch posture.ui.notify.notify in the test instead."
            )
        return real_run(args, *rest, **kwargs)

    monkeypatch.setattr(subprocess, "run", guarded_run)
