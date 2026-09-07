"""Popup colour themes. Pure data, no AppKit, so it is testable headlessly."""
from __future__ import annotations

from dataclasses import dataclass

from posture.types import Rating

RGBA = tuple[float, float, float, float]


@dataclass(frozen=True)
class Theme:
    label: str
    background: RGBA
    accent: RGBA
    badge_background: RGBA
    badge_foreground: RGBA
    headline: RGBA
    body: RGBA
    tip_background: RGBA
    tip_foreground: RGBA


# The badge label a rating is DISPLAYED as. Deliberately not Rating.value: the
# enum's values are persisted in SQLite and in the agreement matrix, so they
# cannot be reworded without a migration, and they read as a verdict on the
# person ("POOR") rather than as a description of the body ("Reset"). This is
# the seam where the two are allowed to diverge, and it is the only place the
# user-visible wording lives.
#
# Colours are the Stack palette: warm charcoal surfaces (#1c1a18 / #262320) with
# sage, warm amber, and terracotta as the three semantic colours. Terracotta
# rather than alarm red on purpose. This popup appears many times a day, every
# day, and a colour that spikes the pulse each time is a colour that gets the
# app turned off. Each theme's background and tip box are the base charcoal
# mixed 13% and 26% toward its semantic colour, so all four panels sit on one
# surface instead of four unrelated ones.
THEMES: dict[str, Theme] = {
    "POOR": Theme(
        label="Reset",
        background=(0.193, 0.133, 0.118, 0.97), accent=(0.753, 0.341, 0.275, 1.0),
        badge_background=(0.753, 0.341, 0.275, 1.0), badge_foreground=(1.0, 1.0, 1.0, 1.0),
        headline=(0.922, 0.867, 0.839, 1.0), body=(0.651, 0.585, 0.546, 1.0),
        tip_background=(0.277, 0.164, 0.141, 1.0), tip_foreground=(0.838, 0.604, 0.557, 1.0),
    ),
    "DECENT": Theme(
        label="Drifting",
        background=(0.210, 0.172, 0.127, 0.97), accent=(0.878, 0.643, 0.345, 1.0),
        badge_background=(0.878, 0.643, 0.345, 1.0), badge_foreground=(0.110, 0.102, 0.094, 1.0),
        headline=(0.935, 0.897, 0.846, 1.0), body=(0.663, 0.615, 0.553, 1.0),
        tip_background=(0.310, 0.243, 0.159, 1.0), tip_foreground=(0.907, 0.770, 0.596, 1.0),
    ),
    "GOOD": Theme(
        label="Stacked",
        background=(0.160, 0.178, 0.135, 0.97), accent=(0.498, 0.690, 0.412, 1.0),
        badge_background=(0.498, 0.690, 0.412, 1.0), badge_foreground=(0.110, 0.102, 0.094, 1.0),
        headline=(0.897, 0.902, 0.853, 1.0), body=(0.625, 0.620, 0.560, 1.0),
        tip_background=(0.211, 0.255, 0.177, 1.0), tip_foreground=(0.697, 0.796, 0.632, 1.0),
    ),
    "UNKNOWN": Theme(
        label="No reading",
        background=(0.179, 0.168, 0.157, 0.97), accent=(0.639, 0.612, 0.576, 1.0),
        badge_background=(0.639, 0.612, 0.576, 1.0), badge_foreground=(0.110, 0.102, 0.094, 1.0),
        headline=(0.911, 0.894, 0.869, 1.0), body=(0.639, 0.612, 0.576, 1.0),
        tip_background=(0.247, 0.235, 0.220, 1.0), tip_foreground=(0.775, 0.753, 0.723, 1.0),
    ),
}


def theme_for(rating: Rating | str | None) -> Theme:
    """Map a rating to a theme. An unknown rating gets the UNKNOWN theme.

    v2 fell back to the DECENT theme, so an unmeasurable result was displayed
    with a confident "DECENT" badge (spec M4).
    """
    if rating is None:
        return THEMES["UNKNOWN"]
    key = rating.value if isinstance(rating, Rating) else str(rating).upper()
    return THEMES.get(key, THEMES["UNKNOWN"])
