"""Write-back after a call. Everything here is a proposal that the policy gates; nothing is sent."""

from __future__ import annotations

from pathlib import Path

from ..kernel.base import KernelClient
from ..live.engine import LiveEngine
from ..models import CallSummary, Note, ProposedAction
from ..policy import PolicyEngine, Ticket
from ..util import clip, new_id


def register_executors(policy: PolicyEngine, kernel: KernelClient, session_dir: Path) -> None:
    async def save_note(action: ProposedAction) -> str:
        return await kernel.save_note(Note.model_validate(action.payload))

    async def create_task(action: ProposedAction) -> str:
        return await kernel.submit_action(action)

    async def save_draft(action: ProposedAction) -> str:
        path = session_dir / "followup.md"
        path.write_text(str(action.payload.get("body", "")) + "\n", encoding="utf-8")
        path.chmod(0o600)
        return str(path)

    policy.register_executor("kernel.note", save_note)
    policy.register_executor("kernel.task", create_task)
    policy.register_executor("email.draft", save_draft)


async def plan_writeback(engine: LiveEngine, summary: CallSummary) -> list[Ticket]:
    """Propose notes, tasks and a follow-up draft; local drafts run at once, the rest wait for a human."""
    meeting_id = engine.meeting.id if engine.meeting else None
    tickets: list[Ticket] = []
    for note in summary.facts_to_remember:
        tickets.append(
            engine.propose(
                ProposedAction(
                    id=new_id("act"),
                    capability="kernel.note",
                    title=clip(note.text, 90),
                    payload=note.model_copy(update={"meeting_id": meeting_id}).model_dump(mode="json"),
                    rationale="факт из созвона",
                )
            )
        )
    for item in summary.action_items:
        tickets.append(
            engine.propose(
                ProposedAction(
                    id=new_id("act"),
                    capability="kernel.task",
                    title=clip(item.text, 90),
                    payload={
                        "text": item.text,
                        "owner": item.owner,
                        "due": item.due,
                        "meeting_id": meeting_id,
                        "source": "call-companion",
                    },
                    rationale="обещание из созвона",
                )
            )
        )
    if summary.followup_draft:
        draft = engine.propose(
            ProposedAction(
                id=new_id("act"),
                capability="email.draft",
                title="Черновик follow-up (не отправляется)",
                payload={"body": summary.followup_draft},
            )
        )
        if draft.status == "approved":
            draft = await engine.policy.execute(draft.id)
        tickets.append(draft)
    return tickets
