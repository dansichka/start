"""The whole pipeline on the fictional demo call: replay, whispers, summary, gated write-back."""

import json

from call_companion.app import finish_call

from .support import demo_runtime, play_demo

EXPECTED = [("entity", "fact"), ("stall", "answer"), ("stall", "answer"), ("entity", "reminder")]


async def test_the_demo_call_produces_the_expected_whispers(tmp_path):
    runtime, clock = await demo_runtime(tmp_path)
    await play_demo(runtime, clock)
    shown = runtime.engine.suggestions
    assert [(s.trigger, s.kind) for s in shown] == EXPECTED
    assert "18 400 EUR" in shown[2].text, "Max says 18 500 - the companion shows the agreed figure"
    assert "план миграции" in shown[3].text
    assert 20.0 <= shown[1].created_at <= 20.6, "the stall fires hesitation_s after the question ends"
    assert [(a.owner, a.due) for a in runtime.engine.actions] == [("я", "до пятницы"), ("я", None)]


async def test_the_replay_is_deterministic(tmp_path):
    runs = []
    for name in ("one", "two"):
        runtime, clock = await demo_runtime(tmp_path / name)
        await play_demo(runtime, clock)
        runs.append([(s.trigger, s.kind, s.text, s.evidence) for s in runtime.engine.suggestions])
    assert runs[0] == runs[1] and len(runs[0]) == 4


async def test_private_and_secret_items_never_reach_the_model_in_the_demo(tmp_path):
    runtime, clock = await demo_runtime(tmp_path)
    await play_demo(runtime, clock)
    seen = {item.id for request in runtime.llm.requests for _, item in request.evidence}
    assert seen and not seen & {"tg:private-77", "stripe:ch-9"}
    prompt_text = " ".join(request.transcript for request in runtime.llm.requests)
    assert "4111" not in prompt_text


async def test_the_user_speaking_cancels_a_pending_stall(tmp_path):
    runtime, clock = await demo_runtime(tmp_path)
    await play_demo(runtime, clock)
    stalls = [s for s in runtime.engine.suggestions if s.trigger == "stall"]
    assert len(stalls) == 2, "the user answered at 25.0 s only after the first stall had fired, and at 44.0 s after the second"


async def test_finishing_the_call_proposes_but_never_writes(tmp_path):
    runtime, clock = await demo_runtime(tmp_path)
    await play_demo(runtime, clock)
    outcome = await finish_call(runtime)
    by_status = {}
    for ticket in outcome.tickets:
        by_status.setdefault(ticket.status, []).append(ticket.action.capability)
    assert sorted(by_status["pending"]) == ["kernel.note", "kernel.note", "kernel.task", "kernel.task"]
    assert by_status["executed"] == ["email.draft"]
    assert (outcome.session_dir / "followup.md").exists()
    assert not (outcome.session_dir / "transcript.jsonl").exists(), "raw transcripts are opt-in"
    assert not (tmp_path / "pack.jsonl.outbox.jsonl").exists()


async def test_approvals_write_to_the_outbox_and_unsupported_actions_fail_cleanly(tmp_path):
    runtime, clock = await demo_runtime(tmp_path)
    await play_demo(runtime, clock)
    outcome = await finish_call(runtime)
    note = next(t for t in outcome.tickets if t.action.capability == "kernel.note")
    task = next(t for t in outcome.tickets if t.action.capability == "kernel.task")
    assert (await runtime.engine.decide(note.id, True, "cli")).status == "executed"
    outbox = (tmp_path / "pack.jsonl.outbox.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(outbox[0])["type"] == "note" and json.loads(outbox[0])["untrusted"] is True
    assert (await runtime.engine.decide(task.id, True, "cli")).status == "failed"
    assert runtime.policy.audit.verify() == (True, None)


async def test_the_transcript_is_stored_only_when_the_user_opted_in(tmp_path):
    runtime, clock = await demo_runtime(tmp_path, store_transcripts=True)
    await play_demo(runtime, clock)
    outcome = await finish_call(runtime)
    rows = (outcome.session_dir / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 13 and json.loads(rows[0])["speaker"] == "Олена"


async def test_a_failing_summary_does_not_lose_the_call(tmp_path):
    from .test_engine import BrokenLLM

    runtime, clock = await demo_runtime(tmp_path, llm=BrokenLLM())
    await play_demo(runtime, clock)
    outcome = await finish_call(runtime)
    assert outcome.summary is None and outcome.tickets == []
    assert any(e.type == "error" and "Итоги не получились" in e.data["message"] for e in runtime.bus.history())
    assert (outcome.session_dir / "meta.json").exists()
