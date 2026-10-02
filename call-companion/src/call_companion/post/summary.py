"""Turn a finished call into a summary, with a second redaction pass on everything the model returns."""

from __future__ import annotations

from ..config import Settings
from ..live.transcript import render_transcript
from ..llm.base import LLM, SummaryRequest
from ..models import Brief, CallRecord, CallSummary
from ..policy import PolicyEngine


async def summarize_call(
    record: CallRecord,
    llm: LLM,
    settings: Settings,
    policy: PolicyEngine,
    brief: Brief | None = None,
) -> CallSummary:
    request = SummaryRequest(
        user_name=settings.user.name or "пользователь",
        language=settings.user.language,
        meeting=record.meeting,
        transcript=render_transcript(record.utterances, me_label=settings.user.name or "Я"),
        action_candidates=record.action_items,
        brief=brief,
    )
    summary = await llm.summarize(request)
    redact = policy.redact
    return summary.model_copy(
        update={
            "title": redact(summary.title),
            "tldr": redact(summary.tldr),
            "decisions": [redact(d) for d in summary.decisions],
            "open_questions": [redact(q) for q in summary.open_questions],
            "followup_draft": redact(summary.followup_draft),
            "facts_to_remember": [n.model_copy(update={"text": redact(n.text)}) for n in summary.facts_to_remember],
            "action_items": [a.model_copy(update={"text": redact(a.text)}) for a in summary.action_items],
        }
    )
