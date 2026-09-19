from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from posture.landmarks import PoseDetector
from posture.types import Landmarks, Point

@pytest.mark.native_model
def test_blank_frame_yields_no_person(native_model_path):
    with PoseDetector(native_model_path) as detector:
        blank = np.zeros((720, 1280, 3), dtype=np.uint8)
        assert detector.detect(blank) is None


@pytest.mark.native_model
def test_detector_is_reusable_across_calls(native_model_path):
    with PoseDetector(native_model_path) as detector:
        blank = np.zeros((720, 1280, 3), dtype=np.uint8)
        assert detector.detect(blank) is None
        assert detector.detect(blank) is None


def test_detect_returns_none_when_no_pose_is_found(tmp_path):
    """Cover the None-vs-person branch WITHOUT needing the 9.4MB model.

    Native detect() tests require an explicit test-only model path, so in a fresh clone or
    on CI they all skip and this branch has no coverage at all. Mutation testing
    confirmed that returning an empty Landmarks instead of None would go
    undetected in that environment. A real file satisfies the existence check; the
    native landmarker itself is mocked.
    """
    fake_model = tmp_path / "fake.task"
    fake_model.write_bytes(b"not a real model")
    fake_landmarker = MagicMock()
    fake_landmarker.detect.return_value = MagicMock(pose_landmarks=[])

    with patch("posture.landmarks.vision.PoseLandmarker.create_from_options",
               return_value=fake_landmarker):
        with PoseDetector(fake_model) as detector:
            result = detector.detect(np.zeros((8, 8, 3), dtype=np.uint8))
    assert result is None


def test_detect_maps_raw_landmarks_onto_the_dataclass(tmp_path):
    """Also model-free: verify the mapping, not just the empty case."""
    fake_model = tmp_path / "fake.task"
    fake_model.write_bytes(b"not a real model")
    raw = [MagicMock(x=0.01 * i, y=0.5, visibility=0.9) for i in range(33)]
    fake_landmarker = MagicMock()
    fake_landmarker.detect.return_value = MagicMock(pose_landmarks=[raw])

    with patch("posture.landmarks.vision.PoseLandmarker.create_from_options",
               return_value=fake_landmarker):
        with PoseDetector(fake_model) as detector:
            lm = detector.detect(np.zeros((8, 8, 3), dtype=np.uint8))
    assert lm is not None
    assert len(lm.points) == 33
    assert lm[0].x == pytest.approx(0.0)
    assert lm[10].x == pytest.approx(0.10)
    assert lm[32].visibility == pytest.approx(0.9)


def test_detect_refuses_use_outside_the_context_manager(tmp_path):
    fake_model = tmp_path / "fake.task"
    fake_model.write_bytes(b"not a real model")
    detector = PoseDetector(fake_model)
    with pytest.raises(RuntimeError, match="context manager"):
        detector.detect(np.zeros((8, 8, 3), dtype=np.uint8))


def test_bounding_box_covers_visible_points_with_padding():
    pts = [Point(0.0, 0.0, 0.0)] * 33
    pts[11] = Point(0.40, 0.50, 0.9)
    pts[12] = Point(0.60, 0.50, 0.9)
    pts[0] = Point(0.50, 0.30, 0.9)
    box = PoseDetector.bounding_box(Landmarks(points=tuple(pts)), padding=0.0)
    assert box == pytest.approx((0.40, 0.30, 0.60, 0.50))


def test_padding_is_proportional_to_the_person_not_to_the_frame():
    """Two people, same frame, one five times the size of the other.

    Under the old frame-relative padding both boxes grew by the same absolute
    0.10 on every side, so the small person got a margin as wide as their whole
    body while the large person got a sliver. That is what made the crop a slab
    of room rather than a box around a person. The margin must be 10% of the
    person, so the small box grows by 0.01 and the large one by 0.05.

    Exact equality on purpose: an inequality like "the small box is smaller"
    is satisfied by frame-relative padding too, so it would not catch a
    regression.
    """
    def person(size: float) -> Landmarks:
        pts = [Point(0.0, 0.0, 0.0)] * 33
        pts[11] = Point(0.5 - size / 2, 0.5 - size / 2, 0.9)
        pts[12] = Point(0.5 + size / 2, 0.5 + size / 2, 0.9)
        return Landmarks(points=tuple(pts))

    small = PoseDetector.bounding_box(person(0.10), padding=0.10)
    large = PoseDetector.bounding_box(person(0.50), padding=0.10)
    # Frame-relative padding would give (0.35, 0.35, 0.65, 0.65) and
    # (0.15, 0.15, 0.85, 0.85) instead.
    assert small == pytest.approx((0.44, 0.44, 0.56, 0.56))
    assert large == pytest.approx((0.20, 0.20, 0.80, 0.80))


def test_bounding_box_clamps_to_the_frame():
    pts = [Point(0.0, 0.0, 0.0)] * 33
    pts[11] = Point(0.02, 0.02, 0.9)
    pts[12] = Point(0.98, 0.98, 0.9)
    x0, y0, x1, y1 = PoseDetector.bounding_box(Landmarks(points=tuple(pts)), padding=0.5)
    assert (x0, y0) == (0.0, 0.0)
    assert (x1, y1) == (1.0, 1.0)


def test_bounding_box_ignores_invisible_points():
    pts = [Point(0.0, 0.0, 0.0)] * 33
    pts[11] = Point(0.45, 0.45, 0.9)
    pts[12] = Point(0.55, 0.55, 0.9)
    pts[27] = Point(0.99, 0.99, 0.05)  # ankle, not visible while seated
    x0, y0, x1, y1 = PoseDetector.bounding_box(Landmarks(points=tuple(pts)), padding=0.0)
    assert x1 == pytest.approx(0.55)


def test_bounding_box_of_nothing_is_none_not_the_full_frame():
    pts = tuple(Point(0.0, 0.0, 0.0) for _ in range(33))
    assert PoseDetector.bounding_box(Landmarks(points=pts)) is None
