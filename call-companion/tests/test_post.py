import json
import os
import stat
import time
from datetime import datetime, timezone

import pytest

from call_companion.app import build_runtime
from call_companion.config import PrivacyCfg, Settings, UserCfg
from call_companion.kernel.memory import InMemoryKernel
from call_companion.llm.fake import FakeLLM
from call_companion.models import ActionItem, CallRecord, CallSummary, Meeting, Note, Utterance
from call_companion.policy import PolicyEngine
from call_companion.post.store import SessionStore
from call_companion.post.summary import summarize_call
from call_companion.post.writeback import plan_writeback, register_executors


def make_record(session_id: str = "call_1") -> CallRecord:
    now = datetime.now(timezone.utc)
    return CallRecord(
        session_id=session_id,
        meeting=Meeting(id="m1", title="Статус"),
        started_at=now,
        ended_at=now,
        utterances=[
            Utterance(id="u1", channel="them", speaker="Олена", text="Привет", t_start=0, t_end=1),
            Utterance(id="u2", channel="me", speaker="me", text="Я пришлю план до пятницы", t_start=2, t_end=4),
        ],
        action_items=[ActionItem(id="a1", text="Я пришлю план до пятницы", owner="я", due="до пятницы")],
    )


SUMMARY = CallSummary(
    title="Статус",
    tldr="Сроки подтверждены.",
    action_items=[ActionItem(id="a1", text="Прислать план миграции", owner="я", due="до пятницы")],
    facts_to_remember=[Note(text="Релиз до 15 октября", tags=["orion"])],
    followup_draft="Спасибо за созвон!",
)


def test_store_keeps_no_transcript_by_default(tmp_path):
    path = SessionStore(tmp_path).save(make_record(), CallSummary(title="T", tldr="x"))
    assert {p.name for p in path.iterdir()} == {"meta.json", "actions.json", "summary.json"}
    assert stat.S_IMODE(path.stat().st_mode) == 0o700
    assert stat.S_IMODE((path / "summary.json").stat().st_mode) == 0o600
    assert json.loads((path / "meta.json").read_text(encoding="utf-8"))["transcript_stored"] is False


def test_store_keeps_the_transcript_only_when_opted_in(tmp_path):
    path = SessionStore(tmp_path, keep_transcripts=True).save(make_record(), None)
    rows = [json.loads(line) for line in (path / "transcript.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["text"] for r in rows] == ["Привет", "Я пришлю план до пятницы"]
    assert stat.S_IMODE((path / "transcript.jsonl").stat().st_mode) == 0o600
    assert not (path / "summary.json").exists()


def test_purge_removes_only_expired_sessions(tmp_path):
    store = SessionStore(tmp_path, retention_days=14)
    old, new = store.save(make_record("old"), None), store.save(make_record("new"), None)
    ancient = time.time() - 30 * 86400
    os.utime(old, (ancient, ancient))
    assert store.purge() == 1 and not old.exists() and new.exists()
    assert SessionStore(tmp_path / "nowhere").purge() == 0


class LeakyLLM(FakeLLM):
    async def summarize(self, req):
        return CallSummary(
            title="Итоги",
            tldr="пароль: hunter2",
            followup_draft="Вот токен: token: abc123xyz",
            facts_to_remember=[Note(text="карта 4111 1111 1111 1111")],
            action_items=[ActionItem(id="a", text="проверить api_key=sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123")],
        )


async def test_the_model_summary_is_redacted_a_second_time():
    summary = await summarize_call(make_record(), LeakyLLM(), Settings(), PolicyEngine(), None)
    dump = summary.model_dump_json()
    for secret in ("hunter2", "abc123xyz", "4111", "sk-ant-api03"):
        assert secret not in dump


async def test_summary_receives_the_whole_call_and_the_action_candidates():
    llm = FakeLLM()
    seen = {}

    async def spy(req):
        seen["transcript"], seen["actions"] = req.transcript, req.action_candidates
        return CallSummary(title="x")

    llm.summarize = spy
    await summarize_call(make_record(), llm, Settings(user=UserCfg(name="Даня")), PolicyEngine(), None)
    assert seen["transcript"].splitlines() == ["[Олена] Привет", "[Даня] Я пришлю план до пятницы"]
    assert seen["actions"][0].text == "Я пришлю план до пятницы"


async def make_runtime(tmp_path):
    settings = Settings(user=UserCfg(name="Даня"), privacy=PrivacyCfg(data_dir=str(tmp_path)))
    kernel = InMemoryKernel()
    runtime = build_runtime(settings, llm=FakeLLM(), kernel=kernel, ephemeral=True)
    await runtime.engine.start(Meeting(id="m1", title="Статус"))
    register_executors(runtime.policy, kernel, runtime.store.dir_for("s1"))
    return runtime, kernel


async def test_write_back_gates_everything_except_the_local_draft(tmp_path):
    runtime, kernel = await make_runtime(tmp_path)
    tickets = await plan_writeback(runtime.engine, SUMMARY)
    status = {t.action.capability: t.status for t in tickets}
    assert status == {"kernel.note": "pending", "kernel.task": "pending", "email.draft": "executed"}
    assert (runtime.store.dir_for("s1") / "followup.md").read_text(encoding="utf-8").strip() == "Спасибо за созвон!"
    assert kernel.source.notes == [] and kernel.source.actions == []


async def test_approved_items_reach_the_kernel_with_provenance(tmp_path):
    runtime, kernel = await make_runtime(tmp_path)
    tickets = {t.action.capability: t for t in await plan_writeback(runtime.engine, SUMMARY)}
    note = await runtime.engine.decide(tickets["kernel.note"].id, True, "hud")
    task = await runtime.engine.decide(tickets["kernel.task"].id, True, "cli")
    assert note.status == task.status == "executed"
    saved = kernel.source.notes[0]
    assert saved.text == "Релиз до 15 октября" and saved.meeting_id == "m1" and saved.untrusted
    assert kernel.source.actions[0].payload == {
        "text": "Прислать план миграции", "owner": "я", "due": "до пятницы", "meeting_id": "m1", "source": "call-companion",
    }


async def test_rejected_items_are_never_written_and_voice_cannot_approve(tmp_path):
    runtime, kernel = await make_runtime(tmp_path)
    tickets = {t.action.capability: t for t in await plan_writeback(runtime.engine, SUMMARY)}
    with pytest.raises(PermissionError):
        await runtime.engine.decide(tickets["kernel.note"].id, True, "voice")
    rejected = await runtime.engine.decide(tickets["kernel.note"].id, False, "hud")
    assert rejected.status == "rejected"
    assert kernel.source.notes == [] and kernel.source.actions == []


async def test_an_offline_kernel_fails_the_write_without_losing_the_ticket(tmp_path):
    runtime, kernel = await make_runtime(tmp_path)
    tickets = {t.action.capability: t for t in await plan_writeback(runtime.engine, SUMMARY)}
    kernel.fail = True
    done = await runtime.engine.decide(tickets["kernel.note"].id, True, "hud")
    assert done.status == "failed" and "injected failure" in done.result
    assert runtime.policy.audit.verify() == (True, None)
