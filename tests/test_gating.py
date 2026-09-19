import random

from posture.gating import decide, roll_sample
from posture.types import Rating


def d(rating=Rating.GOOD, sampled=False, since=999, gap=60, enabled=True, open_=False,
      spent=0, budget=12):
    return decide(rating, sampled=sampled, seconds_since_last_call=since,
                  min_gap_s=gap, gemini_enabled=enabled, circuit_open=open_,
                  coaching_calls_today=spent, max_coaching_calls_per_day=budget)


def test_good_posture_alone_does_not_trigger_a_call():
    assert d(Rating.GOOD).should_call is False


def test_poor_posture_triggers_a_call():
    assert d(Rating.POOR).should_call is True


def test_decent_posture_triggers_a_call():
    assert d(Rating.DECENT).should_call is True


def test_random_sample_triggers_a_call_even_when_posture_is_good():
    result = d(Rating.GOOD, sampled=True)
    assert result.should_call is True
    assert result.is_comparison_sample is True


def test_gated_call_is_not_a_comparison_sample():
    result = d(Rating.POOR, sampled=False)
    assert result.should_call is True
    assert result.is_comparison_sample is False


def test_sampled_poor_posture_still_counts_as_a_comparison_sample():
    # Sampling is decided independently, so a sampled bad frame is still in the set.
    assert d(Rating.POOR, sampled=True).is_comparison_sample is True


def test_minimum_gap_suppresses_a_gated_call():
    assert d(Rating.POOR, since=10, gap=60).should_call is False


def test_minimum_gap_also_suppresses_a_sampled_call():
    result = d(Rating.GOOD, sampled=True, since=10, gap=60)
    assert result.should_call is False
    assert result.is_comparison_sample is False


def test_disabled_gemini_never_calls():
    assert d(Rating.POOR, sampled=True, enabled=False).should_call is False


def test_truthy_nonboolean_values_cannot_enable_uploads():
    for value in ("false", "true", 1, [], {"enabled": True}):
        assert d(Rating.POOR, sampled=True, enabled=value).should_call is False


def test_open_circuit_never_calls():
    assert d(Rating.POOR, sampled=True, open_=True).should_call is False


def test_unmeasurable_rating_does_not_trigger_a_gated_call():
    assert d(None).should_call is False


def test_unmeasurable_rating_can_still_be_sampled():
    assert d(None, sampled=True).should_call is True


def test_roll_sample_is_deterministic_under_a_seeded_rng():
    a = [roll_sample(0.15, random.Random(7)) for _ in range(20)]
    b = [roll_sample(0.15, random.Random(7)) for _ in range(20)]
    assert a == b


def test_roll_sample_rate_is_approximately_honoured():
    rng = random.Random(1234)
    hits = sum(roll_sample(0.15, rng) for _ in range(10_000))
    assert 0.13 < hits / 10_000 < 0.17


def test_roll_sample_edges():
    assert roll_sample(0.0, random.Random(1)) is False
    assert roll_sample(1.0, random.Random(1)) is True


# --- the daily coaching budget: the only hard bound on the gated path ---

def test_a_gated_call_is_refused_once_the_daily_coaching_budget_is_spent():
    result = d(Rating.POOR, spent=12, budget=12)
    assert result.should_call is False
    assert result.reason == "daily coaching budget spent"


def test_the_last_gated_call_inside_the_budget_still_fires():
    # Boundary. Off by one here either wastes a call every day or loses one.
    assert d(Rating.POOR, spent=11, budget=12).should_call is True


def test_the_first_call_over_the_budget_is_refused():
    assert d(Rating.POOR, spent=13, budget=12).should_call is False


def test_a_zero_coaching_budget_disables_gated_calls_entirely():
    # The documented off switch for coaching, distinct from gemini_enabled
    # (which stops everything) and from comparison_sample_rate (which owns only
    # the comparison path).
    assert d(Rating.POOR, spent=0, budget=0).should_call is False
    assert d(Rating.DECENT, spent=0, budget=0).should_call is False


def test_the_coaching_budget_never_blocks_a_comparison_sample():
    # The roll is unbiased only if its outcome is honoured. Letting a budget
    # spent by GATED calls suppress sampled ones would truncate the comparison
    # set exactly on the days posture is worst, which is posture-correlated
    # censoring of the very set that exists to be unbiased.
    result = d(Rating.POOR, sampled=True, spent=999, budget=12)
    assert result.should_call is True
    assert result.is_comparison_sample is True


def test_a_spent_budget_still_leaves_good_posture_alone():
    # The refusal must come from the gated branch, not swallow every rating.
    assert d(Rating.GOOD, spent=999, budget=12).reason == "local rated acceptable"
