"""Keep the HUD quiet: rate limit, minimum gap and near-duplicate suppression."""

from __future__ import annotations

from collections import deque

from ..config import LiveCfg
from ..util import STOPWORDS, stem, tokenize


DUPLICATE_WINDOW_S = 300.0


def _signature(text: str) -> frozenset[str]:
    return frozenset(stem(t) for t in tokenize(text) if t not in STOPWORDS)


def _overlap(a: frozenset[str], b: frozenset[str], threshold: float = 0.7) -> bool:
    return bool(a) and bool(b) and len(a & b) / len(a | b) >= threshold


def similar(a: str, b: str, threshold: float = 0.7) -> bool:
    return _overlap(_signature(a), _signature(b), threshold)


class WhisperGate:
    def __init__(self, cfg: LiveCfg) -> None:
        self.cfg = cfg
        self.muted = False
        self._times: deque[float] = deque()
        self._recent: deque[tuple[frozenset[str], float]] = deque(maxlen=8)

    def preflight(self, now: float, manual: bool = False, directed: bool = False) -> str | None:
        """Cheap check before spending an LLM call; returns the reason to stay quiet, if any.

        A manual ask always goes through. A direct question to the user is answered unless the user muted the
        companion; the interval and per-minute limits only throttle ambient whispers (names, reminders).
        """
        if manual:
            return None
        if self.muted:
            return "заглушено"
        if directed:
            return None
        while self._times and now - self._times[0] > 60.0:
            self._times.popleft()
        if self._times and now - self._times[-1] < self.cfg.min_gap_s:
            return "слишком рано после прошлой подсказки"
        if len(self._times) >= self.cfg.max_whispers_per_min:
            return "лимит подсказок в минуту"
        return None

    def is_duplicate(self, text: str, now: float) -> bool:
        signature = _signature(text)
        return any(now - at <= DUPLICATE_WINDOW_S and _overlap(signature, old) for old, at in self._recent)

    def record(self, text: str, now: float) -> None:
        self._times.append(now)
        self._recent.append((_signature(text), now))
