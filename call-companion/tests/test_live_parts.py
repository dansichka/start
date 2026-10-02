import asyncio

import pytest

from call_companion.bus import EventBus
from call_companion.config import LiveCfg
from call_companion.live.entities import EntityIndex, latin_to_cyrillic
from call_companion.live.gate import WhisperGate
from call_companion.live.transcript import TranscriptBuffer
from call_companion.live.triggers import (
    TriggerDetector,
    TriggerKind,
    find_commitment,
    is_filler,
    is_question,
)
from call_companion.models import Entity, Utterance


def utt(text, channel="them", t=0.0, dur=2.0, speaker=None, final=True) -> Utterance:
    return Utterance(
        id=f"u{t}-{channel}",
        channel=channel,
        speaker=speaker or ("me" if channel == "me" else "Олена"),
        text=text,
        t_start=t,
        t_end=t + dur,
        final=final,
    )


# --- transcript -------------------------------------------------------------


def test_transcript_window_and_rendering():
    buffer = TranscriptBuffer(window_s=30)
    buffer.add(utt("давно это было", t=0))
    buffer.add(utt("Как дела с релизом?", t=60))
    buffer.add(utt("Всё по плану", channel="me", t=63))
    buffer.add(utt("Отлично", channel="them", t=65.5, speaker="Олена"))
    text = buffer.render()
    assert "давно" not in text
    assert text.splitlines() == ["[Олена] Как дела с релизом?", "[Я] Всё по плану", "[Олена] Отлично"]
    assert buffer.last("me").text == "Всё по плану"
    assert buffer.last().text == "Отлично"


def test_transcript_joins_same_speaker_and_tracks_partials():
    buffer = TranscriptBuffer()
    buffer.add(utt("Раз", t=0, dur=1))
    buffer.add(utt("два", t=1.2, dur=1))
    assert buffer.render() == "[Олена] Раз два"
    buffer.add(utt("", channel="me", t=5, final=False))
    assert buffer.speaking("me") and not buffer.speaking("them")
    buffer.add(utt("Привет", channel="me", t=5))
    assert not buffer.speaking("me")
    assert len(buffer.finals) == 3


# --- event bus ---------------------------------------------------------------


async def test_bus_delivers_replays_and_drops_for_slow_consumers():
    bus = EventBus(history=10, queue_size=3)
    bus.publish("a", n=1)
    stream = bus.subscribe(after=0)
    first = await anext(stream)
    assert (first.type, first.data) == ("a", {"n": 1})
    bus.publish("b")
    assert (await anext(stream)).type == "b"
    for i in range(10):
        bus.publish("burst", i=i)
    got = [(await anext(stream)).data["i"] for _ in range(3)]
    assert got == [7, 8, 9]
    await stream.aclose()


async def test_bus_subscribe_without_after_skips_history():
    bus = EventBus()
    bus.publish("old")
    stream = bus.subscribe()
    task = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)
    bus.publish("new")
    assert (await asyncio.wait_for(task, 1)).type == "new"
    await stream.aclose()


async def test_bus_heartbeat_does_not_kill_the_subscription():
    bus = EventBus()
    stream = bus.subscribe(heartbeat_s=0.02)
    assert await anext(stream) is None
    assert await anext(stream) is None
    task = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)
    bus.publish("late", n=1)
    got = await asyncio.wait_for(task, 1)
    while got is None:
        got = await asyncio.wait_for(anext(stream), 1)
    assert got.type == "late"
    await stream.aclose()


# --- entities ------------------------------------------------------------------


@pytest.fixture
def index() -> EntityIndex:
    return EntityIndex(
        [
            Entity(name="Orion", kind="project", aliases=["Орион"]),
            Entity(name="Acme Logistics", kind="org", aliases=["Акме"]),
            Entity(name="Олена Коваленко", kind="person"),
            Entity(name="Max Weber", kind="person", aliases=["Макс Вебер"]),
        ]
    )


def names(found) -> list[str]:
    return [m.entity.name for m in found]


def test_transliteration():
    assert latin_to_cyrillic("Max Weber") == "макс вебер"
    assert latin_to_cyrillic("Orion") == "орион"
    assert latin_to_cyrillic("Acme") == "акме"
    assert latin_to_cyrillic("Oleksandr") == "олександр"
    assert latin_to_cyrillic("Shevchenko") == "шевченко"


def test_finds_people_in_any_case_by_first_or_last_name(index):
    for phrase in ("Олена, привет", "спросим у Олены", "договорились с Оленой", "Коваленко прислала", "это Олену касается"):
        assert names(index.find(phrase)) == ["Олена Коваленко"], phrase
    assert names(index.find("Максу нужен план")) == ["Max Weber"]
    assert names(index.find("Вебер согласен")) == ["Max Weber"]


def test_finds_projects_in_latin_and_cyrillic_and_orgs(index):
    assert names(index.find("статус по Orion")) == ["Orion"]
    assert names(index.find("релиз Ориона")) == ["Orion"]
    assert names(index.find("клиент Акме доволен")) == ["Acme Logistics"]
    assert names(index.find("Acme Logistics прислали")) == ["Acme Logistics"]


def test_org_needs_all_words_and_unrelated_text_matches_nothing(index):
    assert names(index.find("Logistics вообще")) == []
    assert names(index.find("погода хорошая, давайте обсудим бюджет")) == []
    assert not EntityIndex()


# --- questions, fillers, commitments ---------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Как у вас дела с релизом?",
        "Скажите, а когда будет готово",
        "а что по бюджету первого этапа",
        "Коли ви зможете надіслати план?",
        "Can you walk me through the timeline",
        "So how are we doing on the integration",
        "Это правда?",
    ],
)
def test_questions_are_detected(text):
    assert is_question(text)


@pytest.mark.parametrize(
    "text",
    ["Что-то пошло не так.", "Мы всё сделали.", "Отлично, спасибо!", "Да", "", "We shipped it yesterday."],
)
def test_statements_are_not_questions(text):
    assert not is_question(text)


def test_fillers():
    assert is_filler("Ну... эм")
    assert is_filler("hmm")
    assert not is_filler("Ну давайте так")
    assert not is_filler("")


def test_commitments_in_three_languages_with_deadlines():
    c = find_commitment("Я пришлю обновлённый план до пятницы", "me")
    assert c and c.due == "до пятницы" and c.owner == "me"
    assert find_commitment("Ми надішлемо специфікацію до понеділка", "Олена").due == "до понеділка"
    c = find_commitment("I'll send the contract by Friday", "Max")
    assert c and c.due == "by Friday"
    assert find_commitment("Мы отправим отчёт", "Олена").due is None
    assert find_commitment("Погода хорошая", "me") is None


# --- trigger detector ---------------------------------------------------------------


def detector(index, **cfg) -> TriggerDetector:
    return TriggerDetector(LiveCfg(**cfg), ["Даня", "Daniil"], index)


def kinds(triggers) -> list[TriggerKind]:
    return [t.kind for t in triggers]


def test_unanswered_question_becomes_a_stall_after_the_pause(index):
    d = detector(index, hesitation_s=3.0)
    assert d.on_utterance(utt("Даня, как у вас дела с релизом?", t=10, dur=3), now=13) == []
    assert d.tick(15.9) == []
    stall = d.tick(16.1)
    assert kinds(stall) == [TriggerKind.STALL]
    assert "релизом" in stall[0].question
    assert d.tick(30) == []


def test_user_starting_to_answer_cancels_the_stall(index):
    d = detector(index, hesitation_s=3.0)
    d.on_utterance(utt("Как дела с релизом?", t=10, dur=3), now=13)
    d.on_utterance(utt("", channel="me", t=14, final=False), now=14)
    assert d.tick(30) == []


def test_filler_does_not_count_as_an_answer(index):
    d = detector(index, hesitation_s=3.0)
    d.on_utterance(utt("Как дела с релизом?", t=10, dur=3), now=13)
    d.on_utterance(utt("", channel="me", t=14, final=False), now=14)
    d.on_utterance(utt("Ну... эм", channel="me", t=14, dur=1.5), now=15.5)
    assert d.tick(17) == []
    assert kinds(d.tick(18.6)) == [TriggerKind.STALL]


def test_real_answer_after_filler_cancels_for_good(index):
    d = detector(index, hesitation_s=3.0)
    d.on_utterance(utt("Как дела с релизом?", t=10, dur=3), now=13)
    d.on_utterance(utt("", channel="me", t=14, final=False), now=14)
    d.on_utterance(utt("Всё идёт по плану, на этой неделе закрываем", channel="me", t=14, dur=4), now=18)
    assert d.tick(40) == []


def test_follow_up_sentence_from_the_same_speaker_extends_the_wait(index):
    d = detector(index, hesitation_s=3.0)
    d.on_utterance(utt("Как дела с релизом?", t=10, dur=2), now=12)
    d.on_utterance(utt("Мне важно понять сроки", t=12.5, dur=2), now=14.5)
    assert d.tick(16) == []
    assert kinds(d.tick(17.6)) == [TriggerKind.STALL]


def test_always_mode_fires_immediately_and_manual_mode_stays_silent(index):
    d = detector(index, assist="always")
    assert kinds(d.on_utterance(utt("Как дела с релизом Orion?"), now=2)) == [TriggerKind.QUESTION]
    quiet = detector(index, assist="manual")
    assert quiet.on_utterance(utt("Как дела с релизом Orion?"), now=2) == []
    assert quiet.tick(100) == []
    assert kinds(quiet.on_utterance(utt("Я пришлю план до пятницы"), now=3)) == [TriggerKind.COMMITMENT]


def test_a_question_keeps_entities_that_are_in_cooldown(index):
    d = detector(index, assist="always", entity_cooldown_s=120)
    assert kinds(d.on_utterance(utt("Вернёмся к Orion"), now=10)) == [TriggerKind.ENTITY]
    trigger = d.on_utterance(utt("Даня, что нового по Orion?"), now=20)[0]
    assert trigger.kind is TriggerKind.QUESTION and [e.name for e in trigger.entities] == ["Orion"]
    assert d.on_utterance(utt("Так вот, Orion важен"), now=30) == []


def test_entity_cooldown_slides_while_the_topic_stays_alive(index):
    d = detector(index, entity_cooldown_s=120)
    first = d.on_utterance(utt("Давайте вернёмся к Orion"), now=10)
    assert kinds(first) == [TriggerKind.ENTITY] and first[0].entities[0].name == "Orion"
    assert d.on_utterance(utt("Так вот, Orion важен"), now=50) == []
    assert d.on_utterance(utt("И снова про Orion"), now=140) == []
    assert kinds(d.on_utterance(utt("А теперь снова про Orion"), now=270)) == [TriggerKind.ENTITY]


def test_question_carries_the_entities_it_mentions(index):
    d = detector(index, assist="always")
    trigger = d.on_utterance(utt("Даня, что по Orion?"), now=1)[0]
    assert trigger.kind is TriggerKind.QUESTION
    assert [e.name for e in trigger.entities] == ["Orion"]


def test_in_a_group_call_only_questions_with_my_name_count(index):
    speakers = {"n": 3}
    d = TriggerDetector(LiveCfg(assist="always"), ["Даня"], index, remote_speakers=lambda: speakers["n"])
    assert d.on_utterance(utt("Как дела с релизом?"), now=1) == []
    assert kinds(d.on_utterance(utt("Даня, как дела с релизом?"), now=2)) == [TriggerKind.QUESTION]
    assert kinds(d.on_utterance(utt("Дане нужно ответить: как дела?"), now=3)) == [TriggerKind.QUESTION]
    speakers["n"] = 1
    assert kinds(d.on_utterance(utt("А как дела с бюджетом?"), now=4)) == [TriggerKind.QUESTION]


def test_my_own_promises_are_recorded_even_in_manual_mode(index):
    d = detector(index, assist="manual")
    trigger = d.on_utterance(utt("Я отправлю договор до понедельника", channel="me"), now=5)[0]
    assert trigger.commitment.owner == "me" and trigger.commitment.due == "до понедельника"


def test_manual_trigger_has_no_gating(index):
    trigger = detector(index).manual(" что с бюджетом? ")
    assert trigger.kind is TriggerKind.MANUAL and trigger.question == "что с бюджетом?" and trigger.directed


# --- gate ------------------------------------------------------------------------------


def test_gate_rate_limit_gap_and_manual_bypass():
    gate = WhisperGate(LiveCfg(max_whispers_per_min=2, min_gap_s=10))
    assert gate.preflight(0) is None
    gate.record("первая подсказка", 0)
    assert gate.preflight(5) is not None
    assert gate.preflight(11) is None
    gate.record("вторая подсказка", 11)
    assert "лимит" in gate.preflight(25)
    assert gate.preflight(25, manual=True) is None
    assert gate.preflight(72) is None


def test_gate_mute_blocks_everything_but_manual():
    gate = WhisperGate(LiveCfg())
    gate.muted = True
    assert gate.preflight(100) == "заглушено"
    assert gate.preflight(100, directed=True) == "заглушено"
    assert gate.preflight(100, manual=True) is None


def test_direct_questions_ignore_the_interval_and_the_minute_limit_but_not_mute():
    gate = WhisperGate(LiveCfg(max_whispers_per_min=1, min_gap_s=30))
    gate.record("фоновая подсказка", 0)
    assert gate.preflight(2) is not None
    assert gate.preflight(2, directed=True) is None


def test_gate_suppresses_near_duplicates_only_within_the_window():
    gate = WhisperGate(LiveCfg())
    gate.record("Бюджет этапа 1 — 18 400 EUR, оплата 50/50", 0)
    assert gate.is_duplicate("Бюджет этапа 1: 18 400 EUR, оплата 50/50", 30)
    assert not gate.is_duplicate("Бюджет этапа 1: 18 400 EUR, оплата 50/50", 400)
    assert not gate.is_duplicate("Релиз нужен до 15 октября", 30)
    assert not gate.is_duplicate("", 30)
