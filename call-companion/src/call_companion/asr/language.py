"""Keep Whisper's per-segment language guesses from flip-flopping between Russian and Ukrainian."""

from __future__ import annotations

from collections import deque

from ..util import detect_lang


class LanguageTracker:
    """In auto mode, lock onto a language after `lock_after` agreeing segments; unlock when the text disagrees twice."""

    def __init__(self, configured: str = "auto", lock_after: int = 3) -> None:
        self.configured = configured
        self.lock_after = lock_after
        self._recent: deque[str] = deque(maxlen=lock_after)
        self._misses = 0

    def hint(self) -> str | None:
        if self.configured != "auto":
            return self.configured
        if len(self._recent) == self.lock_after and len(set(self._recent)) == 1:
            return self._recent[0]
        return None

    def observe(self, detected: str | None, text: str) -> None:
        if self.configured != "auto":
            return
        locked = self.hint()
        script = detect_lang(text)
        if locked and script not in (locked, "und"):
            self._misses += 1
            if self._misses >= 2:
                self._recent.clear()
                self._misses = 0
            return
        self._misses = 0
        if detected:
            self._recent.append(detected)
