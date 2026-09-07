import cv2
import numpy as np
import pytest

from posture.crop import MAX_EDGE_PX, CropUnavailable, crop_to_person, encode_jpeg
from posture.types import Landmarks, Point
from tests.synthetic import upright_side


def landmarks_in_the_middle() -> Landmarks:
    pts = [Point(0.0, 0.0, 0.0)] * 33
    pts[11] = Point(0.40, 0.40, 0.9)
    pts[12] = Point(0.60, 0.40, 0.9)
    pts[23] = Point(0.45, 0.70, 0.9)
    pts[24] = Point(0.55, 0.70, 0.9)
    return Landmarks(points=tuple(pts))


def test_crop_is_smaller_than_the_source():
    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    cropped = crop_to_person(frame, landmarks_in_the_middle(), padding=0.05)
    assert cropped.shape[0] < frame.shape[0]
    assert cropped.shape[1] < frame.shape[1]


def test_crop_keeps_the_person_region():
    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    # Paint the person region white, leave the background black.
    frame[160:280, 240:360] = 255
    cropped = crop_to_person(frame, landmarks_in_the_middle(), padding=0.02)
    assert cropped.max() == 255


def test_crop_discards_the_far_background():
    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    frame[0:20, 0:20] = 128  # a whiteboard in the top-left corner
    cropped = crop_to_person(frame, landmarks_in_the_middle(), padding=0.02)
    assert not (cropped == 128).any()


def test_no_visible_landmarks_refuses_rather_than_returning_the_room():
    # The original draft returned the untouched frame here. That is the whole
    # room, handed back precisely when the detector is least sure anyone is in
    # it. Refusing costs one datapoint; returning costs a photo of a colleague.
    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    frame[0:20, 0:20] = 128  # a whiteboard, so a full-frame return is provable
    pts = tuple(Point(0.0, 0.0, 0.0) for _ in range(33))
    with pytest.raises(CropUnavailable):
        crop_to_person(frame, Landmarks(points=pts))


def test_a_degenerate_box_refuses_rather_than_returning_the_room():
    # All visible landmarks collapsed onto one point: the padded box is still
    # tiny, so there is no person region to send.
    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    pts = [Point(0.0, 0.0, 0.0)] * 33
    pts[11] = Point(0.5, 0.5, 0.9)
    pts[12] = Point(0.5, 0.5, 0.9)
    with pytest.raises(CropUnavailable):
        crop_to_person(frame, Landmarks(points=tuple(pts)), padding=0.0)


def test_crop_handles_landmarks_at_the_frame_edge():
    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    pts = [Point(0.0, 0.0, 0.0)] * 33
    pts[11] = Point(0.0, 0.0, 0.9)
    pts[12] = Point(1.0, 1.0, 0.9)
    cropped = crop_to_person(frame, Landmarks(points=tuple(pts)), padding=0.2)
    assert cropped.shape[:2] == (400, 600)


def test_how_much_of_the_frame_actually_leaves_the_machine_is_pinned():
    """Pin the exact upload footprint of a known posture at the production default.

    This is the number that decides whether it is reasonable to run this at
    work, so it is pinned exactly rather than bounded. An inequality such as
    "less than half the frame" is satisfied by the old frame-relative padding
    too (it measured 0.411 wide), so it would let a silent widening through.

    upright_side is a 720p side view of a seated person. Its landmarks span
    0.211 of the frame width and 0.647 of the height before any padding.

    The height is not a defect and is not fixable. MediaPipe's pose landmarks
    run ear to hip, and both ends are load-bearing: cva_deg is the
    shoulder-to-ear line and trunk_angle_deg is the hip-to-shoulder line.
    Shortening this box would crop off a landmark the metrics need. What leaves
    the machine is therefore a tall column, and someone standing directly
    behind the user is inside it. The README says so.
    """
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cropped = crop_to_person(frame, upright_side())  # production default padding

    assert cropped.shape[:2] == (549, 325)
    height_fraction = cropped.shape[0] / frame.shape[0]
    width_fraction = cropped.shape[1] / frame.shape[1]
    assert width_fraction == pytest.approx(0.2539, abs=0.0001)   # was 0.4111
    assert height_fraction == pytest.approx(0.7625, abs=0.0001)  # was 0.7972
    assert width_fraction * height_fraction == pytest.approx(0.1936, abs=0.0001)


def test_encode_jpeg_produces_jpeg_magic_bytes():
    frame = np.full((64, 64, 3), 200, dtype=np.uint8)
    data = encode_jpeg(frame)
    assert data[:2] == b"\xff\xd8"   # SOI
    assert data[-2:] == b"\xff\xd9"  # EOI


def test_lower_quality_produces_smaller_output():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
    assert len(encode_jpeg(frame, quality=40)) < len(encode_jpeg(frame, quality=95))


def test_encode_jpeg_downscales_to_max_edge_px():
    # Pinned regression for Task 13's finding: MAX_EDGE_PX is not exercised by
    # any other test, so silently raising it (e.g. to 100000, disabling the
    # downscale entirely) leaves the rest of the suite green while costing
    # tokens on every single Gemini send. This test fails if that happens.
    frame = np.full((2000, 3000, 3), 128, dtype=np.uint8)  # longest edge 3000 >> MAX_EDGE_PX
    data = encode_jpeg(frame)
    decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert max(decoded.shape[:2]) == MAX_EDGE_PX
