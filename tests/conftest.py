import builtins
import socket
import sqlite3
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def pytest_addoption(parser):
    parser.addoption("--native-model-path", default=None,
                     help="Opt into native inference on synthetic frames using this test-only model")


@pytest.fixture
def native_model_path(request):
    supplied = request.config.getoption("--native-model-path")
    if not supplied:
        pytest.skip("native inference requires --native-model-path; no user model is discovered")
    path = Path(supplied).expanduser().resolve()
    if not path.is_file():
        pytest.fail("--native-model-path must name an existing test-only model file")
    return path


@pytest.fixture(autouse=True)
def isolate_devices_network_and_private_state(monkeypatch):
    import cv2
    from posture.config import DATA_DIR

    def blocked(*args, **kwargs):
        raise AssertionError("Tests must mock camera, network and credential access")

    monkeypatch.setattr(cv2, "VideoCapture", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.delenv("POSTURE_GEMINI_API_KEY", raising=False)

    def check_path(path):
        if isinstance(path, (str, Path)):
            value = str(path)
            if value.startswith("file:"):
                value = unquote(urlsplit(value).path)
            if Path(value).expanduser().resolve().is_relative_to(DATA_DIR.resolve()):
                raise AssertionError("Tests must use temporary synthetic state, never the user's data")

    real_open, real_path_open, real_connect = builtins.open, Path.open, sqlite3.connect

    def safe_open(path, *args, **kwargs):
        check_path(path)
        return real_open(path, *args, **kwargs)

    def safe_path_open(path, *args, **kwargs):
        check_path(path)
        return real_path_open(path, *args, **kwargs)

    def safe_connect(path, *args, **kwargs):
        check_path(path)
        return real_connect(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", safe_open)
    monkeypatch.setattr(Path, "open", safe_path_open)
    monkeypatch.setattr(sqlite3, "connect", safe_connect)


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
        if isinstance(args, (list, tuple)) and args and Path(args[0]).name in ("osascript", "security"):
            raise AssertionError(
                "a test tried to access real macOS notifications or Keychain. "
                "Patch the side-effect boundary in the test instead."
            )
        return real_run(args, *rest, **kwargs)

    monkeypatch.setattr(subprocess, "run", guarded_run)
