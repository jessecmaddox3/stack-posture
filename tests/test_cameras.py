import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from posture import cameras
from posture.config import Config
from posture.types import Outcome

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import identify_cameras  # noqa: E402


@pytest.fixture(autouse=True)
def clear_discovery_cache():
    cameras.discover.cache_clear()
    yield
    cameras.discover.cache_clear()


@pytest.fixture(autouse=True)
def block_real_cameras():
    """Hard guard: no test in this file may touch real hardware.

    A test that forgets to mock its camera access would otherwise open the
    physical webcam, trigger a macOS permission prompt, and produce a result that
    depends on whether a camera happens to be attached and authorised. An earlier
    draft of test_capture_roles_reports_camera_busy_when_the_front_camera_is_taken
    did exactly that: it mocked `capture` but not `discover`, so it fell through to
    real hardware and passed or failed nondeterministically.

    Making it impossible beats remembering to mock. Tests that legitimately need
    cv2.VideoCapture patch it themselves, which nests cleanly inside this one.
    """
    def _explode(*args, **kwargs):
        raise AssertionError(
            "this test tried to open a real camera. Mock cv2.VideoCapture, or "
            "patch cameras.discover and cameras.capture."
        )

    with patch("cv2.VideoCapture", side_effect=_explode):
        yield


def fake_capture(*, opened=True, frames=None):
    cap = MagicMock()
    cap.isOpened.return_value = opened
    queue = list(frames or [])

    def read():
        if queue:
            return True, queue.pop(0)
        return False, None

    cap.read.side_effect = read
    return cap


def test_capture_releases_the_device_even_when_it_will_not_open():
    cap = fake_capture(opened=False)
    with patch("cv2.VideoCapture", return_value=cap):
        assert cameras.capture(0) is None
    cap.release.assert_called_once()


def test_capture_releases_the_device_on_success():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    cap = fake_capture(frames=[frame] * 10)
    with patch("cv2.VideoCapture", return_value=cap):
        assert cameras.capture(0, warmup_frames=3) is not None
    cap.release.assert_called_once()


def test_capture_discards_warmup_frames_before_keeping_one():
    frames = [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(5)]
    cap = fake_capture(frames=frames)
    with patch("cv2.VideoCapture", return_value=cap):
        kept = cameras.capture(0, warmup_frames=3)
    # Three discarded, so the kept frame is the fourth (value 3).
    assert kept[0][0][0] == 3


def test_capture_returns_none_when_every_read_fails():
    cap = fake_capture(frames=[])
    with patch("cv2.VideoCapture", return_value=cap):
        assert cameras.capture(0) is None


def test_discover_is_cached_so_probing_happens_once():
    cap = fake_capture(frames=[np.zeros((4, 4, 3), dtype=np.uint8)] * 100)
    with patch("cv2.VideoCapture", return_value=cap) as ctor:
        cameras.discover(max_index=3)
        first_call_count = ctor.call_count
        cameras.discover(max_index=3)
        assert ctor.call_count == first_call_count


def test_capture_roles_reports_camera_busy_when_the_front_camera_is_taken():
    # discover MUST be mocked here. capture_roles consults it on the no-frames
    # path, so without this patch the test reaches the real webcam and its result
    # depends on whether a camera is attached and authorised.
    cfg = Config(camera_indices={"front": 0})
    with patch.object(cameras, "capture", return_value=None):
        with patch.object(cameras, "discover", return_value=(0,)):
            frames, outcome = cameras.capture_roles(cfg)
    assert frames == {}
    assert outcome is Outcome.CAMERA_BUSY


def test_capture_roles_reports_unavailable_when_no_devices_exist():
    # Distinct from CAMERA_BUSY: no devices at all, rather than devices held by
    # another app. The two need different messages in the menu bar, so the
    # distinction has to actually work.
    cfg = Config(camera_indices={"front": 0})
    with patch.object(cameras, "capture", return_value=None):
        with patch.object(cameras, "discover", return_value=()):
            frames, outcome = cameras.capture_roles(cfg)
    assert frames == {}
    assert outcome is Outcome.CAMERA_UNAVAILABLE


def test_capture_roles_succeeds_with_only_the_front_camera():
    cfg = Config(camera_indices={"front": 0, "side": 2})
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    def only_front(index, **kwargs):
        return frame if index == 0 else None

    with patch.object(cameras, "capture", side_effect=only_front):
        frames, outcome = cameras.capture_roles(cfg)
    # A missing side camera is normal (undocked), not an error.
    assert set(frames) == {"front"}
    assert outcome is None


def test_capture_roles_works_with_a_side_camera_only():
    # A single webcam is not necessarily the front one. Someone may mount their
    # only camera to the side, which is the better angle for posture anyway.
    cfg = Config(camera_indices={"side": 1})
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    with patch.object(cameras, "capture", return_value=frame):
        frames, outcome = cameras.capture_roles(cfg)
    assert set(frames) == {"side"}
    assert outcome is None


def test_capture_roles_reports_busy_when_a_lone_side_camera_is_taken():
    cfg = Config(camera_indices={"side": 1})
    with patch.object(cameras, "capture", return_value=None):
        with patch.object(cameras, "discover", return_value=(1,)):
            frames, outcome = cameras.capture_roles(cfg)
    assert frames == {}
    assert outcome is Outcome.CAMERA_BUSY


def test_capture_roles_succeeds_when_only_the_side_camera_produced_a_frame():
    """Both cameras configured, FRONT fails, SIDE succeeds: still a success.

    This is the exact scenario the role-agnostic design exists for, and it had no
    direct coverage. Mutation testing found that a variant requiring front only
    when front is configured passed all 11 tests, because no test had front
    configured-and-failing alongside side configured-and-succeeding. In real use
    this is the docked setup with another app holding the built-in camera.
    """
    cfg = Config(camera_indices={"front": 0, "side": 2})
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    def only_side(index, **kwargs):
        return frame if index == 2 else None

    with patch.object(cameras, "capture", side_effect=only_side):
        frames, outcome = cameras.capture_roles(cfg)
    assert set(frames) == {"side"}
    assert outcome is None


def test_capture_roles_returns_both_when_both_are_present():
    cfg = Config(camera_indices={"front": 0, "side": 2})
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    with patch.object(cameras, "capture", return_value=frame):
        frames, outcome = cameras.capture_roles(cfg)
    assert set(frames) == {"front", "side"}
    assert outcome is None


# --- scripts/identify_cameras.py: distinguishing "no camera" from "not authorised" ---


def test_no_camera_message_names_the_real_no_hardware_case():
    lines = identify_cameras._no_camera_message(hardware_present=False)
    assert any("No camera hardware found" in line for line in lines)
    assert not any("System Settings" in line for line in lines)


def test_no_camera_message_tells_the_user_to_try_again_when_hardware_exists():
    """The first-run permission race: hardware exists, discovery still failed.

    OpenCV does not wait for the macOS authorisation prompt on the very first
    camera access, so every probe fails within about a second. Pointing the user
    at System Settings > Camera is wrong here, since that setting is already
    correct; the actual fix is running the command again after granting access.
    """
    lines = identify_cameras._no_camera_message(hardware_present=True)
    joined = " ".join(lines)
    assert "RUN THIS COMMAND AGAIN" in joined
    assert "No camera hardware found" not in joined


def test_camera_hardware_present_true_when_system_profiler_reports_a_model():
    completed = subprocess.CompletedProcess(
        args=["system_profiler", "SPCameraDataType"], returncode=0,
        stdout="Camera:\n\n    FaceTime HD Camera:\n\n      Model ID: UVC Camera VendorID_1452\n",
    )
    with patch("subprocess.run", return_value=completed) as run:
        assert identify_cameras._camera_hardware_present() is True
    run.assert_called_once_with(
        ["system_profiler", "SPCameraDataType"], capture_output=True, text=True, timeout=15,
    )


def test_camera_hardware_present_false_when_system_profiler_reports_nothing():
    completed = subprocess.CompletedProcess(
        args=["system_profiler", "SPCameraDataType"], returncode=0, stdout="Camera:\n\n",
    )
    with patch("subprocess.run", return_value=completed):
        assert identify_cameras._camera_hardware_present() is False
