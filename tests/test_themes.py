from posture.types import Rating
from posture.ui.themes import THEMES, theme_for

# The wording a user actually sees, kept apart from Rating.value on purpose:
# the enum's values are persisted in SQLite and in the agreement matrix, so they
# cannot be reworded, and they read as a verdict rather than a description.
DISPLAY_LABELS = {
    "GOOD": "Stacked",
    "DECENT": "Drifting",
    "POOR": "Reset",
    "UNKNOWN": "No reading",
}


def test_the_display_labels_are_the_stack_wording():
    """Pins all four, so a future edit cannot quietly put the scold back.

    This is the whole point of the rename: "POOR" in a red pill, forty times a
    day, is what makes someone in pain turn the app off.
    The labels have to describe the body ("Drifting", "Reset"), not grade the
    person.
    """
    assert {key: theme.label for key, theme in THEMES.items()} == DISPLAY_LABELS


def test_no_display_label_is_a_rating_name():
    """The regression guard with teeth: deriving a label from the rating again
    (label=rating.value, or .title(), or .capitalize()) fails here even if
    someone also updates the table above to match.
    """
    forbidden = {value for rating in Rating for value in
                 (rating.value, rating.value.title(), rating.value.capitalize())}
    for theme in THEMES.values():
        assert theme.label not in forbidden, theme.label


def test_every_rating_has_a_theme():
    for rating in Rating:
        assert theme_for(rating) is THEMES[rating.value]
        assert theme_for(rating).label == DISPLAY_LABELS[rating.value]


def test_none_maps_to_unknown_not_decent():
    # v2 showed a DECENT badge for unmeasurable results.
    assert theme_for(None).label == "No reading"


def test_unrecognised_string_maps_to_unknown():
    assert theme_for("BANANA").label == "No reading"


def test_string_ratings_are_accepted():
    assert theme_for("poor").label == "Reset"


def test_all_colour_channels_are_in_range():
    for theme in THEMES.values():
        for name, value in vars(theme).items():
            if name == "label":
                continue
            assert len(value) == 4
            assert all(0.0 <= channel <= 1.0 for channel in value), name


def test_the_semantic_colours_stay_ordered_by_lightness():
    """Good must not merely differ from poor, it must read as the lighter one.

    Sage and terracotta are closer in hue than the green and red they replace,
    so hue alone no longer carries the ordering: under simulated protanopia the
    two collapse to within a hair of each other in hue and only lightness is
    left. Relative luminance is the channel that survives every form of colour
    blindness, so it is the one pinned here.
    """
    def luminance(rgba):
        def lin(c):
            return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        r, g, b = (lin(c) for c in rgba[:3])
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    good = luminance(THEMES["GOOD"].accent)
    poor = luminance(THEMES["POOR"].accent)
    # A ratio, not a bare difference: below about 1.5 the two stop being
    # separable at a glance on a dark surface.
    assert (good + 0.05) / (poor + 0.05) >= 1.5
