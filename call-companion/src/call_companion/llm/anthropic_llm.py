"""Claude implementation: streamed whispers for the live path, structured output for summaries."""

from __future__ import annotations

from typing import Any, AsyncIterator

import anthropic

from ..config import LlmCfg
from ..models import CallSummary
from .base import LLMError, LLMRefused, SummaryRequest, WhisperRequest
from .prompts import SummaryOut, render_summary, render_whisper

FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODELS = frozenset({"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"})


class AnthropicLLM:
    def __init__(self, cfg: LlmCfg, client: anthropic.AsyncAnthropic | None = None) -> None:
        self.cfg = cfg
        self.client = client or anthropic.AsyncAnthropic(timeout=cfg.timeout_s, max_retries=1)

    def _extras(self, model: str, effort: str) -> dict[str, Any]:
        extras: dict[str, Any] = {}
        if not model.startswith("claude-haiku"):
            extras["output_config"] = {"effort": effort}
        return extras

    def _fallbacks(self, model: str) -> dict[str, Any]:
        if self.cfg.fallbacks and model in FALLBACK_MODELS:
            return {"betas": [FALLBACK_BETA], "fallbacks": "default"}
        return {}

    async def whisper(self, req: WhisperRequest) -> AsyncIterator[str]:
        system, messages = render_whisper(req)
        params: dict[str, Any] = dict(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            system=system,
            messages=messages,
            **self._extras(self.cfg.model, self.cfg.effort),
        )
        fallbacks = self._fallbacks(self.cfg.model)
        manager = (
            self.client.beta.messages.stream(**params, **fallbacks)
            if fallbacks
            else self.client.messages.stream(**params)
        )
        async with manager as stream:
            async for text in stream.text_stream:
                yield text
            final = await stream.get_final_message()
        if final.stop_reason == "refusal":
            raise LLMRefused("модель отказалась отвечать")

    async def summarize(self, req: SummaryRequest) -> CallSummary:
        system, messages = render_summary(req)
        model = self.cfg.summary_model
        params: dict[str, Any] = dict(
            model=model,
            max_tokens=self.cfg.summary_max_tokens,
            system=system,
            messages=messages,
            output_format=SummaryOut,
            **self._extras(model, self.cfg.summary_effort),
        )
        fallbacks = self._fallbacks(model)
        result = (
            await self.client.beta.messages.parse(**params, **fallbacks)
            if fallbacks
            else await self.client.messages.parse(**params)
        )
        if result.stop_reason == "refusal":
            raise LLMRefused("модель отказалась оформлять итоги")
        if result.parsed_output is None:
            raise LLMError("модель не вернула структурированный ответ")
        return result.parsed_output.to_summary(req.meeting.id)
