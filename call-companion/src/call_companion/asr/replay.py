"""Replay a scripted call (JSONL) through the engine on a virtual clock.

Each line: {"t": 12.0, "channel": "them" | "me", "speaker": "Олена", "text": "...", "dur": 4.1}
`dur` defaults to a speaking-rate estimate. For "me" lines a silent speech-start marker is emitted at `t`,
exactly like the live VAD does, so a hesitation timer is cancelled the moment the user starts talking.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from ..clock import VirtualClock
from ..live.engine import LiveEngine
from ..models import Utterance

WORDS_PER_SECOND = 2.6


def load_transcript(path: str | Path) -> list[Utterance]:
    utterances: list[Utterance] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        text = row["text"]
        channel = row.get("channel", "them")
        start = float(row["t"])
        duration = float(row.get("dur") or max(1.0, len(text.split()) / WORDS_PER_SECOND))
        utterances.append(
            Utterance(
                id=f"r{number}",
                channel=channel,
                speaker=row.get("speaker") or ("me" if channel == "me" else "them"),
                text=text,
                t_start=start,
                t_end=start + duration,
            )
        )
    return utterances


def _speech_start(utterance: Utterance) -> Utterance:
    return Utterance(
        id=f"{utterance.id}-start",
        channel="me",
        speaker="me",
        text="",
        t_start=utterance.t_start,
        t_end=utterance.t_start,
        final=False,
    )


async def replay(
    engine: LiveEngine,
    utterances: list[Utterance],
    clock: VirtualClock,
    *,
    speed: float = 0.0,
    step: float = 0.25,
    settle: bool = True,
    tail_s: float | None = None,
) -> None:
    """speed 0 runs as fast as possible and settles after every event (deterministic); speed N paces real time."""
    timeline: list[tuple[float, int, Utterance]] = []
    for utterance in utterances:
        if utterance.channel == "me":
            timeline.append((utterance.t_start, 0, _speech_start(utterance)))
        timeline.append((utterance.t_end, 1, utterance))
    timeline.sort(key=lambda entry: (entry[0], entry[1]))

    now = 0.0

    async def advance(target: float) -> None:
        nonlocal now
        while now + step < target:
            now += step
            clock.now = now
            engine.tick()
            if speed:
                await asyncio.sleep(step / speed)
            if settle:
                await engine.settle()
        if speed:
            await asyncio.sleep(max(target - now, 0.0) / speed)
        now = max(now, target)
        clock.now = now

    for moment, _, utterance in timeline:
        await advance(moment)
        engine.feed(utterance)
        if settle:
            await engine.settle()
    await advance(now + (tail_s if tail_s is not None else engine.settings.live.hesitation_s + 1.0))
    if settle:
        await engine.settle()
