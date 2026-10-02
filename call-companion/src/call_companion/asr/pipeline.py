"""Audio frames in, utterances out: per-channel VAD, one decoding worker, hallucination filter."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from ..capture.base import AudioSource
from ..config import AsrCfg
from ..models import Utterance
from ..util import new_id
from .base import AsrEngine, AsrResult
from .hallucination import clean
from .language import LanguageTracker
from .vad import Segment, Segmenter, SpeechStart

log = logging.getLogger(__name__)
CHANNELS = ("me", "them")


class AsrPipeline:
    """One decoding thread keeps a fanless laptop cool; if it falls behind, the user's own speech is dropped first."""

    def __init__(
        self,
        engine: AsrEngine,
        cfg: AsrCfg,
        emit: Callable[[Utterance], None],
        *,
        prompt: Callable[[], str | None] = lambda: None,
        speaker: Callable[[], str] = lambda: "Собеседник",
        on_error: Callable[[Exception], None] = lambda exc: log.warning("asr failed: %s", exc),
        max_backlog_s: float = 30.0,
    ) -> None:
        self.engine = engine
        self.cfg = cfg
        self.emit = emit
        self.prompt = prompt
        self.speaker = speaker
        self.on_error = on_error
        self.max_backlog_s = max_backlog_s
        self._backlog_s = 0.0
        self._segmenters = {
            channel: Segmenter(
                min_speech_s=cfg.min_speech_s, end_silence_s=cfg.end_silence_s, max_segment_s=cfg.max_segment_s
            )
            for channel in CHANNELS
        }
        self._languages = {channel: LanguageTracker(cfg.language) for channel in CHANNELS}

    async def run(self, source: AudioSource) -> None:
        loop = asyncio.get_running_loop()
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr")
        pending: set[asyncio.Task] = set()
        try:
            async for frame in source.frames():
                for event in self._segmenters[frame.channel].feed(frame.pcm, frame.t):
                    if isinstance(event, SpeechStart):
                        if frame.channel == "me":
                            self.emit(self._marker(event.t))
                        continue
                    duration = event.t_end - event.t_start
                    if frame.channel == "me" and self._backlog_s + duration > self.max_backlog_s:
                        log.warning("asr отстаёт, пропускаю свою реплику (%.1f с)", duration)
                        continue
                    self._backlog_s += duration
                    task = loop.create_task(self._decode(pool, frame.channel, event, duration))
                    pending.add(task)
                    task.add_done_callback(pending.discard)
        finally:
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            pool.shutdown(wait=False)

    @staticmethod
    def _marker(t: float) -> Utterance:
        return Utterance(id=new_id("ms"), channel="me", speaker="me", text="", t_start=t, t_end=t, final=False)

    def _transcribe(self, channel: str, segment: Segment) -> tuple[str | None, AsrResult]:
        """Runs on the decoding thread: hint, decode and learn from the result strictly one segment at a time."""
        tracker = self._languages[channel]
        result = self.engine.transcribe(segment.audio, language=tracker.hint(), prompt=self.prompt())
        text = clean(result)
        if text:
            tracker.observe(result.language, text)
        return text, result

    async def _decode(self, pool: ThreadPoolExecutor, channel: str, segment: Segment, duration: float) -> None:
        try:
            text, result = await asyncio.get_running_loop().run_in_executor(pool, self._transcribe, channel, segment)
        except Exception as exc:  # a bad segment must not stop the call
            self.on_error(exc)
            return
        finally:
            self._backlog_s = max(self._backlog_s - duration, 0.0)
        if not text:
            return
        self.emit(
            Utterance(
                id=new_id("u"),
                channel=channel,
                speaker="me" if channel == "me" else self.speaker(),
                text=text,
                t_start=segment.t_start,
                t_end=segment.t_end,
                lang=result.language,
            )
        )
