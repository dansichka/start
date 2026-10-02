"""Audio capture interface: two labelled channels, int16 mono at 16 kHz.

The microphone is "me" and the system output is "them", so who is speaking is known without diarization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol

import numpy as np

from ..models import Channel


class CaptureError(Exception):
    pass


@dataclass
class AudioFrame:
    channel: Channel
    pcm: bytes
    t: float


class AudioSource(Protocol):
    def frames(self) -> AsyncIterator[AudioFrame]: ...


def to_mono_16k(pcm: bytes, channels: int, rate: int, target: int = 16000) -> bytes:
    """Downmix interleaved int16 to mono and resample; integer ratios are averaged, others interpolated."""
    usable = len(pcm) - (len(pcm) % (2 * channels))
    samples = np.frombuffer(pcm[:usable], dtype=np.int16).astype(np.float32)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if rate != target and len(samples):
        if rate % target == 0:
            step = rate // target
            samples = samples[: len(samples) // step * step].reshape(-1, step).mean(axis=1)
        else:
            count = int(len(samples) * target / rate)
            samples = np.interp(np.linspace(0, len(samples) - 1, count), np.arange(len(samples)), samples)
    return np.clip(samples, -32768, 32767).astype(np.int16).tobytes()
