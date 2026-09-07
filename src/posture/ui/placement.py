"""Where the popup goes. Pure geometry so the multi-monitor rule is testable.

v2 read mainScreen().frame().size and ignored origin, so on a two-display desk
the panel opened on the wrong screen (spec M2). None of that logic could be
tested because it lived inside an AppKit call. It lives here now.
"""
from __future__ import annotations

from typing import NamedTuple


class Rect(NamedTuple):
    x: float
    y: float
    width: float
    height: float


def screen_containing(point: tuple[float, float], frames: list[Rect]) -> int:
    """Index of the frame containing the point, or 0 when none does.

    Falling back to index 0 matches AppKit's mainScreen fallback. The caller is
    expected to pass the screens in NSScreen.screens() order.
    """
    x, y = point
    for index, frame in enumerate(frames):
        if (frame.x <= x <= frame.x + frame.width
                and frame.y <= y <= frame.y + frame.height):
            return index
    return 0


def panel_origin(visible: Rect, width: float, height: float, margin: float
                 ) -> tuple[float, float]:
    """Top-right corner of the visible frame, inset by margin.

    Must use the VISIBLE frame, not the full frame: the full frame includes the
    strip under the menu bar, and a panel placed there is partly hidden.
    """
    x = visible.x + visible.width - width - margin
    y = visible.y + visible.height - height - margin
    return (x, y)
