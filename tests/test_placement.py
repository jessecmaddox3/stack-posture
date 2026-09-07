from posture.ui.placement import Rect, panel_origin, screen_containing

LAPTOP = Rect(0.0, 0.0, 1512.0, 982.0)
EXTERNAL = Rect(1512.0, 0.0, 2560.0, 1440.0)
LEFT_OF_MAIN = Rect(-2560.0, 0.0, 2560.0, 1440.0)


def test_the_mouse_on_the_laptop_picks_the_laptop():
    assert screen_containing((700.0, 400.0), [LAPTOP, EXTERNAL]) == 0


def test_the_mouse_on_the_external_picks_the_external():
    # The exact v2 bug: ignoring origin put this on screen 0 every time.
    assert screen_containing((2000.0, 700.0), [LAPTOP, EXTERNAL]) == 1


def test_a_screen_left_of_the_main_one_is_found():
    # Negative origins are normal on macOS and a naive width-only check misses them.
    assert screen_containing((-1000.0, 700.0), [LAPTOP, LEFT_OF_MAIN]) == 1


def test_a_mouse_outside_every_screen_falls_back_to_the_first():
    assert screen_containing((99999.0, 99999.0), [LAPTOP, EXTERNAL]) == 0


def test_the_panel_sits_inside_the_visible_frame():
    # visible.y of 0 with height 982 models a screen whose menu bar is already
    # excluded. The panel must not extend past the top edge.
    #
    # This must assert the exact origin, not just "fits inside": a "fits
    # inside" inequality check passes even if the margin term is dropped
    # entirely (0 + 1512 - 380 = 1132, and 1132 + 380 <= 1512 is still true,
    # landing the panel flush against the edge with no margin at all). Only
    # an exact assertion catches that.
    visible = Rect(0.0, 0.0, 1512.0, 982.0)
    x, y = panel_origin(visible, width=380.0, height=200.0, margin=20.0)
    assert (x, y) == (1112.0, 762.0)


def test_the_panel_follows_a_non_zero_screen_origin():
    # If origin is dropped, this lands on the laptop instead of the external.
    #
    # This must assert the exact value, not a lower bound: with visible.x
    # dropped, x becomes 2560 - 380 - 20 = 2160.0, which still clears
    # "x >= 1512.0" by pure numeric coincidence. Only the exact expected
    # value (1512 + 2560 - 380 - 20 = 3672.0) actually pins the origin term.
    visible = Rect(1512.0, 0.0, 2560.0, 1400.0)
    x, y = panel_origin(visible, width=380.0, height=200.0, margin=20.0)
    assert (x, y) == (3672.0, 1180.0)


def test_the_panel_follows_a_dock_adjusted_visible_origin():
    # Every fixture above uses visible.y = 0.0, so the y term in panel_origin
    # is never actually exercised: visible.y + visible.height - height - margin
    # and visible.height - height - margin compute the same number when
    # visible.y is 0. That is not a contrived edge case: visibleFrame excludes
    # the Dock, and with the Dock at the bottom (the macOS default), visible.y
    # is the Dock's height, not 0, on every real machine.
    #
    # x = 0 + 1512 - 380 - 20 = 1112.0
    # y = 74 + 908 - 200 - 20 = 762.0
    visible = Rect(0.0, 74.0, 1512.0, 908.0)
    x, y = panel_origin(visible, width=380.0, height=200.0, margin=20.0)
    assert (x, y) == (1112.0, 762.0)


def test_the_panel_follows_a_screen_stacked_above_the_main_one():
    # A screen stacked above the main display sits at y = the main screen's
    # full height, which is a much larger origin than a Dock offset and would
    # be caught by nothing above.
    #
    # x = 0 + 1512 - 380 - 20 = 1112.0
    # y = 982 + 982 - 200 - 20 = 1744.0
    visible = Rect(0.0, 982.0, 1512.0, 982.0)
    x, y = panel_origin(visible, width=380.0, height=200.0, margin=20.0)
    assert (x, y) == (1112.0, 1744.0)


def test_the_mouse_on_a_screen_stacked_above_the_main_one_picks_that_screen():
    # Every screen_containing fixture above uses y = 0.0 for every frame, so
    # an implementation that only ever compares x bounds and ignores y
    # entirely would still pass all of them. Two screens stacked vertically
    # at the same x, with the mouse on the upper one, is the case that
    # actually exercises the y bounds check.
    bottom = Rect(0.0, 0.0, 1512.0, 982.0)
    top = Rect(0.0, 982.0, 1512.0, 982.0)
    assert screen_containing((700.0, 1400.0), [bottom, top]) == 1
