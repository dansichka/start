"""LLM backends; the Claude client is imported lazily so tests and replays do not need credentials."""

from .base import LLM, LLMError, LLMRefused, SummaryRequest, WhisperRequest

__all__ = ["LLM", "LLMError", "LLMRefused", "SummaryRequest", "WhisperRequest", "build_llm"]


def build_llm(kind: str, settings) -> LLM:
    if kind == "fake":
        from .fake import FakeLLM

        return FakeLLM()
    if kind == "claude":
        from .anthropic_llm import AnthropicLLM

        return AnthropicLLM(settings.llm)
    raise ValueError(f"неизвестный LLM: {kind!r} (допустимо: claude, fake)")
