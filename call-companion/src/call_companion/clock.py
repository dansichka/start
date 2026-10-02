"""A clock the replay advances by hand, so hesitation timers behave the same at any speed."""

from __future__ import annotations


class VirtualClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now
