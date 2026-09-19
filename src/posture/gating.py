"""When to call Gemini. Pure, so the sampling policy is fully testable.

There are TWO independent reasons to call, and they are budgeted separately:

- The COMPARISON path: a random sample, taken to measure how often local and
  Gemini agree. Its volume is comparison_sample_rate * record windows per day,
  which does not depend on how the posture actually goes.
- The COACHING path: a gated call, made when local rates the window DECENT or
  POOR, to get text worth showing in the nudge. Its volume DOES depend on the
  posture, so it is capped by max_coaching_calls_per_day. Without that cap the
  worst day costs several times the best one and nothing bounds it.

Three rules keep the local-versus-Gemini comparison honest:

1. The random sample is rolled independently of the local rating, BEFORE this
   function is called. Conditioning it on the rating would bias the comparison
   set toward postures local already flagged, hiding the disagreement that
   matters most (local says fine, Gemini would have caught something).
2. is_comparison_sample is true only for randomly-sampled calls. Gated calls
   produce coaching text and must stay out of the agreement statistics.
3. The coaching budget is checked only in the gated branch, strictly AFTER the
   sampled branch has already returned. A budget that both paths drew from would
   be spent faster on bad-posture days, so the comparison set would stop early
   on precisely those days: posture-correlated censoring, the same bias rule 1
   guards against, arriving through the back door.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from posture.types import Rating

_GATED_RATINGS = (Rating.DECENT, Rating.POOR)


@dataclass(frozen=True)
class GateDecision:
    should_call: bool
    is_comparison_sample: bool
    reason: str


def roll_sample(rate: float, rng: random.Random | None = None) -> bool:
    """Roll for comparison-set inclusion. Call this before scoring is consulted."""
    if rate <= 0.0:
        return False
    if rate >= 1.0:
        return True
    return (rng or random).random() < rate


def decide(
    rating: Rating | None,
    *,
    sampled: bool,
    seconds_since_last_call: float,
    min_gap_s: float,
    gemini_enabled: bool,
    circuit_open: bool,
    coaching_calls_today: int,
    max_coaching_calls_per_day: int,
) -> GateDecision:
    """Decide whether to call, and whether the call counts as a comparison sample.

    coaching_calls_today and max_coaching_calls_per_day are required rather than
    defaulted on purpose: a new call site that forgot them would silently
    reinstate the unbounded gated path this budget exists to close.
    """
    if gemini_enabled is not True:
        return GateDecision(False, False, "gemini disabled")
    if circuit_open:
        return GateDecision(False, False, "circuit breaker open")
    if seconds_since_last_call < min_gap_s:
        return GateDecision(False, False, "within minimum gap between calls")
    if sampled:
        # Before the budget check, always. See rule 3 in the module docstring.
        return GateDecision(True, True, "random comparison sample")
    if rating in _GATED_RATINGS:
        if coaching_calls_today >= max_coaching_calls_per_day:
            return GateDecision(False, False, "daily coaching budget spent")
        return GateDecision(True, False, f"local rated {rating.value}")
    return GateDecision(False, False, "local rated acceptable")
