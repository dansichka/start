"""Whisper engines. Both libraries are imported lazily so the rest of the package runs without them."""

from __future__ import annotations

from statistics import mean

import numpy as np

from ..config import AsrCfg
from .base import AsrEngine, AsrResult


class MlxWhisper:
    """Apple Silicon: Whisper on the GPU through MLX."""

    name = "mlx"

    def __init__(self, model: str) -> None:
        self.model = model

    def _module(self):
        try:
            import mlx_whisper
        except ImportError as exc:
            raise RuntimeError("не установлен mlx-whisper: pip install 'call-companion[mac]'") from exc
        return mlx_whisper

    def warmup(self) -> None:
        self.transcribe(np.zeros(16000, dtype=np.float32), language="en", prompt=None)

    def transcribe(self, audio: np.ndarray, *, language: str | None, prompt: str | None) -> AsrResult:
        options = {"language": language} if language else {}
        result = self._module().transcribe(
            audio,
            path_or_hf_repo=self.model,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=prompt,
            word_timestamps=False,
            verbose=None,
            **options,
        )
        segments = result.get("segments") or []
        return AsrResult(
            text=result["text"].strip(),
            language=result.get("language"),
            no_speech_prob=max((s.get("no_speech_prob", 0.0) for s in segments), default=0.0),
            avg_logprob=mean([s.get("avg_logprob", 0.0) for s in segments]) if segments else 0.0,
            compression_ratio=max((s.get("compression_ratio", 0.0) for s in segments), default=0.0),
        )


class FasterWhisper:
    """CPU or CUDA fallback (Linux, Windows, Intel Macs)."""

    name = "faster"

    def __init__(self, model: str, device: str = "auto", compute_type: str = "int8") -> None:
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError("не установлен faster-whisper: pip install 'call-companion[faster]'") from exc
            self._model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)
        return self._model

    def warmup(self) -> None:
        self._load()

    def transcribe(self, audio: np.ndarray, *, language: str | None, prompt: str | None) -> AsrResult:
        segments, info = self._load().transcribe(
            audio,
            language=language,
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=prompt,
            vad_filter=False,
            no_speech_threshold=0.6,
        )
        parts = list(segments)
        return AsrResult(
            text=" ".join(p.text.strip() for p in parts).strip(),
            language=info.language,
            no_speech_prob=max((p.no_speech_prob for p in parts), default=0.0),
            avg_logprob=mean([p.avg_logprob for p in parts]) if parts else 0.0,
            compression_ratio=max((p.compression_ratio for p in parts), default=0.0),
        )


def build_asr(cfg: AsrCfg) -> AsrEngine:
    if cfg.engine == "mlx":
        return MlxWhisper(cfg.model)
    model = "large-v3-turbo" if cfg.model.startswith("mlx-community/") else cfg.model
    return FasterWhisper(model)
