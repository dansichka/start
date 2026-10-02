"""Experimental floating HUD window for macOS (pywebview + PyObjC). Needs the [mac] extra.

The window floats above full-screen call apps and asks macOS not to include it in screen sharing
(NSWindowSharingNone). That request is best-effort: newer macOS versions and some capture APIs may ignore it,
so test with a real share before relying on it. The only surface that is private by construction is a second
device: open the HUD URL on a phone instead.
"""

from __future__ import annotations

from typing import Callable


def apply_window_flags() -> int:
    """Float, join all spaces and request exclusion from sharing; returns how many windows were touched."""
    from AppKit import (
        NSApp,
        NSFloatingWindowLevel,
        NSWindowCollectionBehaviorCanJoinAllSpaces,
        NSWindowCollectionBehaviorFullScreenAuxiliary,
        NSWindowSharingNone,
    )

    windows = list(NSApp.windows())
    for window in windows:
        window.setSharingType_(NSWindowSharingNone)
        window.setLevel_(NSFloatingWindowLevel)
        window.setCollectionBehavior_(NSWindowCollectionBehaviorCanJoinAllSpaces | NSWindowCollectionBehaviorFullScreenAuxiliary)
    return len(windows)


def run_window(url: str, *, width: int = 440, height: int = 640, on_ready: Callable[[], None] | None = None) -> None:
    """Open the HUD in a floating window and block until it is closed (must run on the main thread)."""
    try:
        import webview
    except ImportError as exc:
        raise RuntimeError("не установлен pywebview: pip install 'call-companion[mac]'") from exc

    def started() -> None:
        try:
            from PyObjCTools import AppHelper

            AppHelper.callAfter(apply_window_flags)
        except ImportError:
            pass
        if on_ready:
            on_ready()

    webview.create_window("Компаньон", url, width=width, height=height, on_top=True, resizable=True)
    webview.start(started)
