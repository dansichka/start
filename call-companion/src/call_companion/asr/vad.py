"""Energy-based voice activity detection and utterance segmentation.

Cheap and dependable on the clean signals a call produces (headset mic, decoded remote audio). The noise
floor is tracked with fast-down / slow-up min-statistics so a fan or a constant hum does not read as speech.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import SAMPLE_RATE

FRAME = SAMPLE_RATE * 30 // 1000
FRAME_S = FRAME / SAMPLE_RATE


@dataclass
class SpeechStart:
    t: float


@dataclass
class Segment:
    t_start: float
    t_end: float
    audio: np.ndarray


class Segmenter:
    def __init__(
        self,
        *,
        min_speech_s: float = 0.4,
        end_silence_s: float = 0.55,
        max_segment_s: float = 12.0,
        preroll_s: float = 0.3,
        start_frames: int = 3,
        margin_db: float = 12.0,
        abs_floor_db: float = -48.0,
    ) -> None:
        self.min_speech_s = min_speech_s
        self.end_silence_frames = max(1, round(end_silence_s / FRAME_S))
        self.max_frames = round(max_segment_s / FRAME_S)
        self.preroll_frames = round(preroll_s / FRAME_S)
        self.start_frames = start_frames
        self.margin_db = margin_db
        self.abs_floor_db = abs_floor_db
        self._tail = np.zeros(0, dtype=np.int16)
        self._odd = b""
        self._origin: float | None = None
        self._index = 0
        self._floor: float | None = None
        self._calibration: list[float] = []
        self._ring: list[np.ndarray] = []
        self._run = 0
        self._active = False
        self._frames: list[np.ndarray] = []
        self._seg_start_index = 0
        self._speech_frames = 0
        self._silence = 0

    def _time(self, index: int) -> float:
        return (self._origin or 0.0) + index * FRAME_S

    def _is_speech(self, frame: np.ndarray) -> bool:
        level = 20 * np.log10(float(np.sqrt(np.mean((frame.astype(np.float32) / 32768.0) ** 2))) + 1e-9)
        if self._floor is None:
            self._calibration.append(level)
            if len(self._calibration) < 10:
                return level > -30.0
            self._floor = float(np.median(self._calibration))
        if level < self._floor:
            self._floor += 0.2 * (level - self._floor)
        else:
            self._floor += 0.0015 * (level - self._floor)
        self._floor = min(max(self._floor, -80.0), -30.0)
        return level > max(self._floor + self.margin_db, self.abs_floor_db)

    def feed(self, pcm: bytes, t: float = 0.0) -> list[SpeechStart | Segment]:
        """Consume int16 little-endian mono samples; `t` stamps the first chunk only (the stream is contiguous)."""
        if self._origin is None:
            self._origin = t
        pcm = self._odd + pcm
        usable = len(pcm) - (len(pcm) % 2)
        self._odd = pcm[usable:]
        samples = np.concatenate([self._tail, np.frombuffer(pcm[:usable], dtype=np.int16)])
        events: list[SpeechStart | Segment] = []
        whole = len(samples) // FRAME
        for n in range(whole):
            frame = samples[n * FRAME : (n + 1) * FRAME]
            self._step(frame, self._is_speech(frame), events)
        self._tail = samples[whole * FRAME :]
        return events

    def _step(self, frame: np.ndarray, speech: bool, events: list[SpeechStart | Segment]) -> None:
        index = self._index
        self._index += 1
        if not self._active:
            self._ring.append(frame)
            self._ring = self._ring[-(self.preroll_frames + self.start_frames) :]
            self._run = self._run + 1 if speech else 0
            if self._run >= self.start_frames:
                self._active = True
                onset = index - self.start_frames + 1
                lead = min(self.preroll_frames, len(self._ring) - self.start_frames)
                self._frames = list(self._ring[len(self._ring) - self.start_frames - lead :])
                self._seg_start_index = onset - lead
                self._speech_frames = self.start_frames
                self._silence = 0
                events.append(SpeechStart(self._time(onset)))
            return
        self._frames.append(frame)
        if speech:
            self._speech_frames += 1
            self._silence = 0
        else:
            self._silence += 1
        if self._silence >= self.end_silence_frames:
            self._finish(index + 1, events, keep_trailing=3)
            self._active, self._ring, self._run = False, [], 0
        elif len(self._frames) >= self.max_frames:
            self._finish(index + 1, events, keep_trailing=0)
            self._seg_start_index = index + 1
            self._frames, self._speech_frames, self._silence = [], 0, 0

    def _finish(self, end_index: int, events: list[SpeechStart | Segment], keep_trailing: int) -> None:
        frames = self._frames
        if self._silence:
            drop = max(self._silence - keep_trailing, 0)
            frames = frames[: len(frames) - drop]
            end_index -= drop
        if self._speech_frames * FRAME_S >= self.min_speech_s and frames:
            audio = np.concatenate(frames).astype(np.float32) / 32768.0
            events.append(Segment(self._time(self._seg_start_index), self._time(end_index), audio))
