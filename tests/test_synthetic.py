from posture.types import LM
from tests.synthetic import (
    forward_head_side, slouched_side, tilted_front, upright_front, upright_side,
)


def test_all_builders_return_33_points():
    for lm in (upright_front(), tilted_front(5), upright_side(),
               forward_head_side(), slouched_side()):
        assert len(lm.points) == 33


def test_upright_front_shoulders_are_level():
    lm = upright_front()
    assert lm[LM.LEFT_SHOULDER].y == lm[LM.RIGHT_SHOULDER].y


def test_tilted_front_drops_the_right_shoulder():
    lm = tilted_front(10.0)
    assert lm[LM.RIGHT_SHOULDER].y > lm[LM.LEFT_SHOULDER].y


def test_side_view_hides_the_far_side():
    lm = upright_side()
    # Task 4 selects the near side by summing visibility across ear and shoulder,
    # and also reads hip. Verify near/far visibility split for all three pairs.
    threshold = 0.5
    # Near-side (left) landmarks visible.
    assert lm[LM.LEFT_EAR].visibility > threshold
    assert lm[LM.LEFT_SHOULDER].visibility > threshold
    assert lm[LM.LEFT_HIP].visibility > threshold
    # Far-side (right) landmarks invisible.
    assert lm[LM.RIGHT_EAR].visibility < threshold
    assert lm[LM.RIGHT_SHOULDER].visibility < threshold
    assert lm[LM.RIGHT_HIP].visibility < threshold


def test_forward_head_moves_the_ear_forward_and_down():
    good, bad = upright_side(), forward_head_side(1.0)
    assert bad[LM.LEFT_EAR].x > good[LM.LEFT_EAR].x
    assert bad[LM.LEFT_EAR].y > good[LM.LEFT_EAR].y


def test_slouch_moves_the_shoulder_forward_of_the_hip():
    lm = slouched_side(20.0)
    assert lm[LM.LEFT_SHOULDER].x > lm[LM.LEFT_HIP].x
