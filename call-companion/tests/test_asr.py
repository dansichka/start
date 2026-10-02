import asyncio
import time

import numpy as np
import pytest

from call_companion.asr.base import AsrResult
from call_companion.asr.hallucination import clean, repetition_ratio
from call_companion.asr.language import LanguageTracker
from call_companion.asr.pipeline import AsrPipeline
from call_companion.asr.vad import Segment, Segmenter, SpeechStart
from call_companion.capture.base import AudioFrame
from call_companion.config import AsrCfg

RATE = 16000


def tone(seconds: float, amp: float = 0.25, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (amp * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)


def hush(seconds: float, noise: float = 0.0, seed: int = 1) -> np.ndarray:
    n = int(seconds * RATE)
    if not noise:
        return np.zeros(n, dtype=np.int16)
    return (np.random.default_rng(seed).normal(0, noise, n) * 32767).astype(np.int16)


def run(audio: np.ndarray, chunk: int = 1600, t: float = 0.0, **kwargs):
    segmenter = Segmenter(**kwargs)
    events = []
    raw = audio.tobytes()
    for i in range(0, len(raw), chunk * 2):
        events += segmenter.feed(raw[i : i + chunk * 2], t)
    return events


def segments(events) -> list[Segment]:
    return [e for e in events if isinstance(e, Segment)]


# --- segmentation ----------------------------------------------------------------


def test_one_utterance_with_preroll_and_trailing_silence():
    events = run(np.concatenate([hush(1.0), tone(1.0), hush(1.5)]))
    assert [type(e) for e in events] == [SpeechStart, Segment]
    start, segment = events
    assert start.t == pytest.approx(1.0, abs=0.1)
    assert segment.t_start == pytest.approx(0.7, abs=0.1)
    assert segment.t_end == pytest.approx(2.09, abs=0.12)
    assert len(segment.audio) == pytest.approx((segment.t_end - segment.t_start) * RATE, abs=RATE * 0.04)
    assert segment.audio.dtype == np.float32 and np.abs(segment.audio).max() <= 1.0


def test_short_blips_never_become_segments():
    events = run(np.concatenate([hush(1.0), tone(0.2), hush(1.5)]))
    assert segments(events) == []


def test_a_short_pause_stays_inside_one_utterance_a_long_one_splits_it():
    short = run(np.concatenate([hush(0.5), tone(1.0), hush(0.3), tone(1.0), hush(1.5)]))
    assert len(segments(short)) == 1
    long = run(np.concatenate([hush(0.5), tone(1.0), hush(1.0), tone(1.0), hush(1.5)]))
    assert len(segments(long)) == 2


def test_continuous_speech_is_cut_at_the_maximum_length():
    found = segments(run(np.concatenate([hush(0.5), tone(30.0), hush(1.5)]), max_segment_s=12.0))
    assert len(found) == 3
    assert all((s.t_end - s.t_start) <= 12.1 for s in found)
    assert found[1].t_start == pytest.approx(found[0].t_end, abs=0.05)


def test_steady_noise_and_hum_are_not_speech():
    assert run(hush(6.0, noise=0.003)) == []
    hum = np.concatenate([hush(3.0, noise=0.01, seed=2), tone(1.0, amp=0.35), hush(2.0, noise=0.01, seed=3)])
    assert len(segments(run(hum))) == 1


def test_chunk_size_does_not_change_the_result():
    audio = np.concatenate([hush(1.0), tone(1.0), hush(1.0), tone(0.8, freq=330), hush(1.2)])
    baseline = [(round(s.t_start, 3), round(s.t_end, 3), len(s.audio)) for s in segments(run(audio, chunk=1600))]
    assert len(baseline) == 2
    for chunk in (480, 777, 9999, 100):
        assert [(round(s.t_start, 3), round(s.t_end, 3), len(s.audio)) for s in segments(run(audio, chunk=chunk))] == baseline


def test_odd_byte_boundaries_do_not_corrupt_the_stream():
    audio = np.concatenate([hush(1.0), tone(1.0), hush(1.5)])
    raw = audio.tobytes()
    segmenter = Segmenter()
    events = []
    for i in range(0, len(raw), 1001):
        events += segmenter.feed(raw[i : i + 1001])
    reference = segments(run(audio))
    assert len(segments(events)) == 1
    assert len(segments(events)[0].audio) == len(reference[0].audio)


def test_timestamps_follow_the_stream_origin():
    events = run(np.concatenate([hush(1.0), tone(1.0), hush(1.5)]), t=100.0)
    assert segments(events)[0].t_start == pytest.approx(100.7, abs=0.1)


def test_speech_right_from_the_first_frame_is_still_found():
    found = segments(run(np.concatenate([tone(1.5), hush(1.5)])))
    assert len(found) == 1 and found[0].t_start == pytest.approx(0.0, abs=0.05)


# --- hallucination filter -----------------------------------------------------------


def r(text, **kw) -> AsrResult:
    return AsrResult(text=text, **kw)


def test_subtitle_credits_are_always_dropped():
    assert clean(r("Субтитры сделал DimaTorzok")) is None
    assert clean(r("Продолжение следует...")) is None
    assert clean(r("Subtitles by the Amara.org community")) is None


def test_pleasantries_are_dropped_only_when_the_model_doubts_them():
    assert clean(r("Thank you.", no_speech_prob=0.5)) is None
    assert clean(r("Thank you.", avg_logprob=-1.2)) is None
    assert clean(r("Thank you.", no_speech_prob=0.05, avg_logprob=-0.2)) == "Thank you."
    assert clean(r("Спасибо за просмотр", no_speech_prob=0.1)) == "Спасибо за просмотр"


def test_silence_noise_and_loops_are_dropped():
    assert clean(r("")) is None and clean(r(" ... ")) is None
    assert clean(r("что-то", no_speech_prob=0.9, avg_logprob=-1.5)) is None
    assert clean(r("ну " * 12)) is None
    assert clean(r("привет", compression_ratio=3.0)) is None
    assert repetition_ratio("ну ну ну") == 0.0


def test_real_speech_passes_untouched():
    text = "Давайте вернёмся к бюджету первого этапа."
    assert clean(r(text, no_speech_prob=0.02, avg_logprob=-0.3, compression_ratio=1.4)) == text


# --- language tracker -----------------------------------------------------------------


def test_language_locks_after_agreement_and_unlocks_on_contradiction():
    tracker = LanguageTracker("auto", lock_after=3)
    assert tracker.hint() is None
    for _ in range(3):
        tracker.observe("ru", "Расскажите про бюджет")
    assert tracker.hint() == "ru"
    tracker.observe("en", "Can you walk me through the plan")
    assert tracker.hint() == "ru", "one contradiction is tolerated"
    tracker.observe("en", "Let us move on to the budget")
    assert tracker.hint() is None


def test_a_configured_language_is_always_used():
    tracker = LanguageTracker("uk")
    tracker.observe("ru", "привет")
    assert tracker.hint() == "uk"


# --- pipeline ---------------------------------------------------------------------------


class ScriptedSource:
    def __init__(self, frames: list[AudioFrame]) -> None:
        self._frames = frames

    async def frames(self):
        for frame in self._frames:
            yield frame
            await asyncio.sleep(0)


def frames_for(channel: str, audio: np.ndarray, offset: float = 0.0) -> list[AudioFrame]:
    raw, step = audio.tobytes(), 3200
    return [AudioFrame(channel, raw[i : i + step], offset + i / 2 / RATE) for i in range(0, len(raw), step)]


def merged(*streams: list[AudioFrame]) -> list[AudioFrame]:
    return sorted((f for s in streams for f in s), key=lambda f: f.t)


class ScriptedEngine:
    name = "scripted"

    def __init__(self, texts: list[str | Exception], delay: float = 0.0) -> None:
        self.texts = list(texts)
        self.delay = delay
        self.calls: list[dict] = []

    def warmup(self) -> None:
        pass

    def transcribe(self, audio, *, language, prompt):
        self.calls.append({"seconds": len(audio) / RATE, "language": language, "prompt": prompt})
        time.sleep(self.delay)
        item = self.texts.pop(0)
        if isinstance(item, Exception):
            raise item
        return AsrResult(text=item, language="ru")


async def run_pipeline(engine, source, **kwargs):
    out = []
    pipeline = AsrPipeline(engine, AsrCfg(), out.append, **kwargs)
    await pipeline.run(source)
    return out


async def test_pipeline_labels_channels_and_emits_a_speech_start_marker_for_me():
    them = frames_for("them", np.concatenate([hush(0.5), tone(2.0), hush(1.2)]))
    me = frames_for("me", np.concatenate([hush(4.0), tone(1.2), hush(1.2)]))
    engine = ScriptedEngine(["Даня, как дела?", "Всё хорошо"])
    out = await run_pipeline(engine, ScriptedSource(merged(them, me)), prompt=lambda: "Олена, Orion", speaker=lambda: "Олена")
    markers = [u for u in out if not u.final]
    finals = [u for u in out if u.final]
    assert len(markers) == 1 and markers[0].channel == "me" and markers[0].text == ""
    assert [(u.channel, u.speaker, u.text) for u in finals] == [("them", "Олена", "Даня, как дела?"), ("me", "me", "Всё хорошо")]
    assert finals[0].t_start < finals[0].t_end and finals[0].lang == "ru"
    assert all(c["prompt"] == "Олена, Orion" for c in engine.calls)


async def test_pipeline_filters_hallucinations_and_survives_engine_errors():
    audio = np.concatenate([hush(0.5), tone(1.0), hush(1.0), tone(1.0), hush(1.0), tone(1.0), hush(1.2)])
    errors = []
    engine = ScriptedEngine(["Продолжение следует...", RuntimeError("model missing"), "нормальная реплика"])
    out = await run_pipeline(engine, ScriptedSource(frames_for("them", audio)), on_error=errors.append)
    assert [u.text for u in out] == ["нормальная реплика"]
    assert len(errors) == 1 and "model missing" in str(errors[0])


async def test_pipeline_locks_the_language_after_a_few_agreeing_segments():
    audio = np.concatenate([np.concatenate([tone(0.8), hush(1.0)]) for _ in range(5)])
    engine = ScriptedEngine(["Расскажите про бюджет"] * 5)
    await run_pipeline(engine, ScriptedSource(frames_for("them", audio)))
    assert [c["language"] for c in engine.calls] == [None, None, None, "ru", "ru"]


async def test_when_decoding_falls_behind_my_own_speech_is_dropped_first():
    me = frames_for("me", np.concatenate([tone(2.0), hush(1.0), tone(2.0), hush(1.2)]))
    them = frames_for("them", np.concatenate([hush(0.2), tone(2.0), hush(1.0), tone(2.0), hush(1.2)]))
    engine = ScriptedEngine(["a"] * 8, delay=0.15)
    out = await run_pipeline(engine, ScriptedSource(merged(me, them)), max_backlog_s=3.0)
    finals = [u for u in out if u.final]
    assert sum(u.channel == "them" for u in finals) == 2
    assert sum(u.channel == "me" for u in finals) < 2
