"""Drop the phantom text Whisper invents on silence, music and noise."""

from __future__ import annotations

import re
from collections import Counter

from ..util import normalize, tokenize
from .base import AsrResult

# Subtitle credits and sign-offs Whisper learned from video subtitles; none is plausible live speech.
ALWAYS = {
    "продолжение следует",
    "субтитры сделал dimatorzok",
    "субтитры создавал dimatorzok",
    "субтитры делал dimatorzok",
    "редактор субтитров а семкин корректор а егорова",
    "субтитры подогнал",
    "subtitles by the amara org community",
    "amara org",
}
# Short pleasantries that are real speech sometimes; dropped only when the model itself doubts there was speech.
AMBIGUOUS = {
    "спасибо за просмотр",
    "подписывайтесь на канал",
    "дякуємо за перегляд",
    "підписуйтесь на канал",
    "thank you",
    "thanks for watching",
    "thank you for watching",
    "bye",
}
MAX_COMPRESSION = 2.4


def _plain(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", normalize(text)).split())


def repetition_ratio(text: str) -> float:
    """Share of the text taken by its most frequent word, for loops like 'ну ну ну ну ну'."""
    words = tokenize(text)
    if len(words) < 8:
        return 0.0
    return Counter(words).most_common(1)[0][1] / len(words)


def clean(result: AsrResult) -> str | None:
    """Return the text to keep, or None when the segment is silence in disguise."""
    text = result.text.strip()
    if not text or not re.search(r"\w", text):
        return None
    if result.no_speech_prob > 0.6 and result.avg_logprob < -1.0:
        return None
    plain = _plain(text)
    if plain in ALWAYS:
        return None
    if plain in AMBIGUOUS and (result.no_speech_prob > 0.3 or result.avg_logprob < -0.8):
        return None
    if result.compression_ratio > MAX_COMPRESSION or repetition_ratio(text) > 0.6:
        return None
    return text
