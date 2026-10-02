"""Wiring: build the runtime from settings and finish a call (summary, storage, write-back proposals)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .bus import EventBus
from .config import Settings
from .kernel import KernelClient, build_kernel
from .live.engine import LiveEngine
from .llm import LLM, build_llm
from .models import CallRecord, CallSummary
from .policy import AuditLog, PolicyEngine, Ticket
from .post.store import SessionStore
from .post.summary import summarize_call
from .post.writeback import plan_writeback, register_executors

log = logging.getLogger(__name__)


@dataclass
class Runtime:
    settings: Settings
    bus: EventBus
    policy: PolicyEngine
    kernel: KernelClient
    llm: LLM
    engine: LiveEngine
    store: SessionStore


@dataclass
class CallOutcome:
    record: CallRecord
    summary: CallSummary | None
    tickets: list[Ticket]
    session_dir: Path


def build_runtime(
    settings: Settings,
    *,
    llm: LLM | str = "claude",
    kernel: KernelClient | None = None,
    clock: Callable[[], float] | None = None,
    ephemeral: bool = False,
) -> Runtime:
    """ephemeral keeps the audit trail in memory only (tests, demos)."""
    audit = AuditLog(None if ephemeral else settings.data_dir / "audit.jsonl")
    policy = PolicyEngine(settings.policy, settings.privacy, audit)
    kernel = kernel or build_kernel(settings)
    model = build_llm(llm, settings) if isinstance(llm, str) else llm
    bus = EventBus()
    engine = LiveEngine(settings, kernel, model, policy, bus, clock=clock)
    store = SessionStore(
        settings.data_dir,
        keep_transcripts=settings.privacy.store_transcripts,
        retention_days=settings.privacy.retention_days,
    )
    return Runtime(settings, bus, policy, kernel, model, engine, store)


async def finish_call(rt: Runtime, *, summarize: bool = True) -> CallOutcome:
    record = await rt.engine.stop()
    session_dir = rt.store.dir_for(record.session_id)
    summary: CallSummary | None = None
    if summarize and record.utterances:
        try:
            summary = await summarize_call(record, rt.llm, rt.settings, rt.policy, rt.engine.brief)
        except Exception as exc:  # the call is over; a failed summary must not lose the transcript
            log.warning("summary failed: %s: %s", type(exc).__name__, exc)
            rt.bus.publish("error", message=f"Итоги не получились: {type(exc).__name__}: {exc}")
    register_executors(rt.policy, rt.kernel, session_dir)
    tickets = await plan_writeback(rt.engine, summary) if summary else []
    rt.store.save(record, summary)
    rt.engine.summary = summary.model_dump(mode="json") if summary else None
    rt.bus.publish("summary", summary=rt.engine.summary, session_dir=str(session_dir))
    return CallOutcome(record, summary, tickets, session_dir)
