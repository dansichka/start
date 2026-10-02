import sys
import types

import numpy as np
import pytest

from call_companion.asr.engines import FasterWhisper, MlxWhisper, build_asr
from call_companion.config import AsrCfg

AUDIO = np.zeros(16000, dtype=np.float32)


def install(monkeypatch, name: str, module: types.ModuleType) -> None:
    monkeypatch.setitem(sys.modules, name, module)


def test_mlx_engine_passes_the_documented_arguments_and_maps_the_result(monkeypatch):
    seen = {}
    fake = types.ModuleType("mlx_whisper")

    def transcribe(audio, **kwargs):
        seen["audio"], seen["kwargs"] = audio, kwargs
        return {
            "text": " Даня, как дела? ",
            "language": "ru",
            "segments": [
                {"no_speech_prob": 0.02, "avg_logprob": -0.2, "compression_ratio": 1.2},
                {"no_speech_prob": 0.10, "avg_logprob": -0.4, "compression_ratio": 1.6},
            ],
        }

    fake.transcribe = transcribe
    install(monkeypatch, "mlx_whisper", fake)
    result = MlxWhisper("mlx-community/whisper-large-v3-turbo").transcribe(AUDIO, language="ru", prompt="Олена, Orion")
    kwargs = seen["kwargs"]
    assert seen["audio"] is AUDIO
    assert kwargs["path_or_hf_repo"] == "mlx-community/whisper-large-v3-turbo"
    assert kwargs["language"] == "ru" and kwargs["initial_prompt"] == "Олена, Orion"
    assert kwargs["temperature"] == 0.0 and kwargs["condition_on_previous_text"] is False
    assert result.text == "Даня, как дела?" and result.language == "ru"
    assert result.no_speech_prob == 0.10 and result.avg_logprob == pytest.approx(-0.3) and result.compression_ratio == 1.6


def test_mlx_engine_leaves_language_detection_to_whisper_when_unset(monkeypatch):
    seen = {}
    fake = types.ModuleType("mlx_whisper")
    fake.transcribe = lambda audio, **kw: seen.update(kw) or {"text": "hi", "segments": []}
    install(monkeypatch, "mlx_whisper", fake)
    result = MlxWhisper("m").transcribe(AUDIO, language=None, prompt=None)
    assert "language" not in seen and seen["initial_prompt"] is None
    assert result.text == "hi" and result.no_speech_prob == 0.0


def test_faster_engine_passes_the_documented_arguments_and_loads_the_model_once(monkeypatch):
    created, calls = [], []

    class Segment:
        def __init__(self, text):
            self.text, self.no_speech_prob, self.avg_logprob, self.compression_ratio = text, 0.05, -0.3, 1.1

    class WhisperModel:
        def __init__(self, name, device, compute_type):
            created.append((name, device, compute_type))

        def transcribe(self, audio, **kwargs):
            calls.append(kwargs)
            return iter([Segment(" Привет "), Segment("мир")]), types.SimpleNamespace(language="ru")

    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = WhisperModel
    install(monkeypatch, "faster_whisper", fake)
    engine = FasterWhisper("small", device="cpu", compute_type="int8")
    first = engine.transcribe(AUDIO, language=None, prompt="Orion")
    engine.transcribe(AUDIO, language="en", prompt=None)
    assert created == [("small", "cpu", "int8")]
    assert calls[0]["initial_prompt"] == "Orion" and calls[0]["language"] is None
    assert calls[0]["beam_size"] == 1 and calls[0]["vad_filter"] is False and calls[0]["condition_on_previous_text"] is False
    assert first.text == "Привет мир" and first.language == "ru"


def test_missing_libraries_give_an_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx_whisper", None)
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    with pytest.raises(RuntimeError, match=r"call-companion\[mac\]"):
        MlxWhisper("m").transcribe(AUDIO, language=None, prompt=None)
    with pytest.raises(RuntimeError, match=r"call-companion\[faster\]"):
        FasterWhisper("small").warmup()


def test_build_asr_picks_the_engine_and_a_sensible_model():
    assert isinstance(build_asr(AsrCfg(engine="mlx")), MlxWhisper)
    fallback = build_asr(AsrCfg(engine="faster"))
    assert isinstance(fallback, FasterWhisper) and fallback.model_name == "large-v3-turbo"
    assert build_asr(AsrCfg(engine="faster", model="small")).model_name == "small"
