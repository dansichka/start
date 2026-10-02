"""Speech-to-text engine interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

SAMPLE_RATE = 16000


@dataclass
class AsrResult:
    text: str
    language: str | None = None
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0


class AsrEngine(Protocol):
    name: str

    def warmup(self) -> None: ...

    def transcribe(self, audio: np.ndarray, *, language: str | None, prompt: str | None) -> AsrResult:
        """audio is float32 mono at 16 kHz in [-1, 1]."""
        ...
