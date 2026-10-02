"""Rolling transcript with one pending (partial) utterance per channel."""

from __future__ import annotations

from ..models import Channel, Utterance


class TranscriptBuffer:
    def __init__(self, window_s: float = 90.0) -> None:
        self.window_s = window_s
        self._finals: list[Utterance] = []
        self._partials: dict[str, Utterance] = {}

    def add(self, utterance: Utterance) -> None:
        if utterance.final:
            self._finals.append(utterance)
            self._partials.pop(utterance.channel, None)
        else:
            self._partials[utterance.channel] = utterance

    @property
    def finals(self) -> list[Utterance]:
        return list(self._finals)

    def speaking(self, channel: Channel) -> bool:
        return channel in self._partials

    def last(self, channel: Channel | None = None) -> Utterance | None:
        for utterance in reversed(self._finals):
            if channel is None or utterance.channel == channel:
                return utterance
        return None

    def recent(self, window_s: float | None = None) -> list[Utterance]:
        if not self._finals:
            return []
        horizon = self._finals[-1].t_end - (window_s or self.window_s)
        return [u for u in self._finals if u.t_end >= horizon]

    def render(self, window_s: float | None = None, me_label: str = "Я") -> str:
        """Plain-text transcript; consecutive utterances of one speaker are joined."""
        lines: list[tuple[str, str, float]] = []
        for utterance in self.recent(window_s):
            who = me_label if utterance.channel == "me" else utterance.speaker
            if lines and lines[-1][0] == who and utterance.t_start - lines[-1][2] < 2.0:
                lines[-1] = (who, f"{lines[-1][1]} {utterance.text}", utterance.t_end)
            else:
                lines.append((who, utterance.text, utterance.t_end))
        return "\n".join(f"[{who}] {text}" for who, text, _ in lines)


def render_transcript(utterances: list[Utterance], me_label: str = "Я") -> str:
    """The whole call as plain text, for the summary."""
    buffer = TranscriptBuffer(window_s=float("inf"))
    for utterance in utterances:
        buffer.add(utterance)
    return buffer.render(me_label=me_label)
