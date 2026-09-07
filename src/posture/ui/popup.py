"""Styled notification panel.

Non-activating, so it never steals keyboard focus. Positioned on the screen
that currently holds the mouse, inside its visibleFrame so it cannot land under
the menu bar. Laid out with NSStackView rather than hand-computed frames.
"""
from __future__ import annotations

import logging

import AppKit
import Foundation
from PyObjCTools import AppHelper

from posture.types import Rating
from posture.ui.placement import Rect, panel_origin, screen_containing
from posture.ui.themes import Theme, theme_for

logger = logging.getLogger("posture")

PANEL_WIDTH = 380.0
PADDING = 18.0
MARGIN = 20.0

_active: dict = {}


def _color(rgba) -> "AppKit.NSColor":
    return AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(*rgba)


def _target_screen() -> "AppKit.NSScreen":
    """The screen under the mouse, falling back to the first one.

    A thin adapter: it reads geometry off NSScreen and hands the decision to
    posture.ui.placement, which is pure and therefore actually tested. v2 used
    mainScreen().frame().size and ignored origin, so on a multi-monitor desk the
    panel appeared on the wrong display (spec M2).
    """
    screens = list(AppKit.NSScreen.screens())
    if not screens:
        return AppKit.NSScreen.mainScreen()
    mouse = AppKit.NSEvent.mouseLocation()
    frames = [Rect(s.frame().origin.x, s.frame().origin.y,
                   s.frame().size.width, s.frame().size.height) for s in screens]
    return screens[screen_containing((mouse.x, mouse.y), frames)]


def _label(text: str, size: float, rgba, *, bold: bool = False, wrap: float | None = None):
    if wrap:
        view = AppKit.NSTextField.wrappingLabelWithString_(text)
        view.setPreferredMaxLayoutWidth_(wrap)
    else:
        view = AppKit.NSTextField.labelWithString_(text)
    font = (AppKit.NSFont.boldSystemFontOfSize_(size) if bold
            else AppKit.NSFont.systemFontOfSize_(size))
    view.setFont_(font)
    view.setTextColor_(_color(rgba))
    view.setBackgroundColor_(AppKit.NSColor.clearColor())
    view.setBezeled_(False)
    view.setEditable_(False)
    return view


def _badge(theme: Theme):
    container = AppKit.NSView.alloc().init()
    container.setWantsLayer_(True)
    container.layer().setBackgroundColor_(_color(theme.badge_background).CGColor())
    container.layer().setCornerRadius_(6.0)

    text = _label(f"  {theme.label}  ", 12.0, theme.badge_foreground, bold=True)
    text.setTranslatesAutoresizingMaskIntoConstraints_(False)
    container.addSubview_(text)
    AppKit.NSLayoutConstraint.activateConstraints_([
        text.leadingAnchor().constraintEqualToAnchor_(container.leadingAnchor()),
        text.trailingAnchor().constraintEqualToAnchor_(container.trailingAnchor()),
        text.topAnchor().constraintEqualToAnchor_constant_(container.topAnchor(), 4.0),
        text.bottomAnchor().constraintEqualToAnchor_constant_(container.bottomAnchor(), -4.0),
    ])
    return container


def _tip_box(theme: Theme, tip: str, width: float):
    box = AppKit.NSView.alloc().init()
    box.setWantsLayer_(True)
    box.layer().setBackgroundColor_(_color(theme.tip_background).CGColor())
    box.layer().setCornerRadius_(8.0)

    text = _label(tip, 12.5, theme.tip_foreground, wrap=width - 24)
    text.setTranslatesAutoresizingMaskIntoConstraints_(False)
    box.addSubview_(text)
    AppKit.NSLayoutConstraint.activateConstraints_([
        text.leadingAnchor().constraintEqualToAnchor_constant_(box.leadingAnchor(), 12.0),
        text.trailingAnchor().constraintEqualToAnchor_constant_(box.trailingAnchor(), -12.0),
        text.topAnchor().constraintEqualToAnchor_constant_(box.topAnchor(), 10.0),
        text.bottomAnchor().constraintEqualToAnchor_constant_(box.bottomAnchor(), -10.0),
    ])
    return box


def show_popup(rating: Rating | str | None, headline: str, details: str,
               tip: str | None, duration: float = 12.0) -> None:
    """Show the panel. Safe to call from any thread."""
    AppHelper.callAfter(_show_on_main, rating, headline, details, tip, duration)


def _dismiss() -> None:
    panel = _active.get("panel")
    if panel is None:
        return
    timer = _active.get("timer")
    if timer is not None:
        timer.invalidate()
    panel.orderOut_(None)
    _active.clear()


def _show_on_main(rating, headline: str, details: str, tip: str | None,
                  duration: float) -> None:
    try:
        _dismiss()
        theme = theme_for(rating)
        content_width = PANEL_WIDTH - PADDING * 2

        stack = AppKit.NSStackView.alloc().init()
        stack.setOrientation_(AppKit.NSUserInterfaceLayoutOrientationVertical)
        stack.setAlignment_(AppKit.NSLayoutAttributeLeading)
        stack.setSpacing_(10.0)
        stack.setTranslatesAutoresizingMaskIntoConstraints_(False)
        stack.setEdgeInsets_((PADDING, PADDING, PADDING, PADDING))

        stack.addArrangedSubview_(_badge(theme))
        if headline:
            stack.addArrangedSubview_(
                _label(headline, 17.0, theme.headline, bold=True, wrap=content_width))
        if details:
            stack.addArrangedSubview_(
                _label(details, 12.5, theme.body, wrap=content_width))
        if tip and tip.strip().lower() != "none":
            stack.addArrangedSubview_(_tip_box(theme, tip, content_width))

        # Build the hierarchy BEFORE measuring. A wrapping label cannot report a
        # height until it knows its width, and its width comes from the
        # constraints below. Measuring first underestimates, and since Gemini's
        # details field is free text of unbounded length, an underestimate clips
        # the message. The panel is created at a provisional height and resized
        # once the real one is known.
        panel = AppKit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            Foundation.NSMakeRect(0.0, 0.0, PANEL_WIDTH, 90.0),
            AppKit.NSWindowStyleMaskBorderless | AppKit.NSWindowStyleMaskNonactivatingPanel,
            AppKit.NSBackingStoreBuffered,
            False,
        )
        panel.setLevel_(AppKit.NSFloatingWindowLevel)
        panel.setOpaque_(False)
        panel.setHasShadow_(True)
        panel.setBackgroundColor_(_color(theme.background))
        panel.setHidesOnDeactivate_(False)
        panel.setBecomesKeyOnlyIfNeeded_(True)

        content = panel.contentView()
        content.setWantsLayer_(True)
        content.layer().setCornerRadius_(14.0)
        content.layer().setMasksToBounds_(True)
        content.addSubview_(stack)

        AppKit.NSLayoutConstraint.activateConstraints_([
            stack.leadingAnchor().constraintEqualToAnchor_(content.leadingAnchor()),
            stack.trailingAnchor().constraintEqualToAnchor_(content.trailingAnchor()),
            stack.topAnchor().constraintEqualToAnchor_(content.topAnchor()),
        ])

        # Now that the stack is constrained to a known width, its wrapping labels
        # can report a real height.
        content.layoutSubtreeIfNeeded()
        height = max(stack.fittingSize().height, 90.0)

        visible = _target_screen().visibleFrame()
        origin_x, origin_y = panel_origin(
            Rect(visible.origin.x, visible.origin.y,
                 visible.size.width, visible.size.height),
            PANEL_WIDTH, height, MARGIN)
        panel.setFrame_display_(
            Foundation.NSMakeRect(origin_x, origin_y, PANEL_WIDTH, height), False)

        # Added after the resize so its hand-computed frame matches the final
        # height. It is the one view here that is not under Auto Layout.
        accent = AppKit.NSView.alloc().initWithFrame_(
            Foundation.NSMakeRect(0, height - 3, PANEL_WIDTH, 3))
        accent.setAutoresizingMask_(AppKit.NSViewMinYMargin | AppKit.NSViewWidthSizable)
        accent.setWantsLayer_(True)
        accent.layer().setBackgroundColor_(_color(theme.accent).CGColor())
        content.addSubview_(accent)

        # orderFrontRegardless, never makeKeyAndOrderFront: typing must not be
        # interrupted (spec M1).
        panel.orderFrontRegardless()

        # Record the panel before scheduling its timer, not after. Timer
        # scheduling is fallible; if it raises, the except clause below still
        # needs _active to point at this panel so the next call's _dismiss()
        # can close it. Without this ordering a failed timer call leaves the
        # panel on screen with nothing in _active, so it can never be found
        # again and every later nudge just stacks another window on top.
        _active["panel"] = panel
        _active["timer"] = None

        timer = Foundation.NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
            float(duration), False, lambda _t: _dismiss())
        _active["timer"] = timer

    except Exception:
        logger.exception("popup failed, falling back to a plain notification")
        from posture.ui.notify import notify
        notify("Stack", f"{headline}. {details}")
