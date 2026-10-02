"""Text helpers shared by retrieval, triggers and prompts."""

from __future__ import annotations

import json
import re
import unicodedata
import uuid
from typing import Any

_TOKEN = re.compile(r"[^\W_]+(?:['’ʼ`][^\W_]+)*", re.UNICODE)
_UK_LETTERS = set("іїєґ")
_RU_LETTERS = set("ыэъё")
_UK_HINTS = {"що", "як", "це", "та", "чи", "але", "який", "коли", "тому", "буде", "можна", "дякую", "будь", "ласка"}
_RU_HINTS = {"что", "как", "это", "и", "но", "когда", "потому", "будет", "можно", "спасибо", "пожалуйста", "тоже"}

STOPWORDS = frozenset(
    """
    и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было вот от меня еще нет о из ему
    теперь когда даже ну вдруг ли если уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя ничего ей может они
    тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому этого
    какой совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех никогда можно при наконец два об другой
    хоть после над больше тот через эти нас про всего них какая много разве три эту моя впрочем хорошо свою этой перед иногда лучше
    чуть том нельзя такой им более всегда конечно всю между это вот ну да нет
    і й та але чи що як це він вона воно вони ми ви ти я мене тебе його її нас вас їх мій твій наш ваш цей той така такий
    коли де куди хто чому тому бо якщо вже ще тут там дуже все всі був була було були буде будуть є для про від до за при під над
    the a an and or but if then else of to in on at by for with from as is are was were be been being it its this that these those
    i you he she we they me him her us them my your his our their not no yes do does did done have has had will would can could
    should shall may might must so than too very just also about into over after before again there here what which who whom when
    where why how all any both each few more most other some such only own same
    """.split()
)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower().replace("ё", "е")


def tokenize(text: str) -> list[str]:
    tokens = []
    for raw in _TOKEN.findall(normalize(text)):
        if raw.endswith(("'s", "’s", "ʼs", "`s")):
            raw = raw[:-2]
        tokens.append(re.sub(r"['’ʼ`]", "", raw))
    return tokens


def stem(token: str) -> str:
    """Cheap stemmer: good enough to match Slavic and English inflections in BM25."""
    if len(token) <= 3 or not token.isalpha():
        return token
    if token.isascii():
        for suffix in ("ing", "ed", "es", "s", "e"):
            if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                return token[: -len(suffix)]
        return token
    if token[-1] in "аяуюыиеоэіїєьй":
        token = token[:-1]
    return token[:5]


_CASE_ENDINGS = tuple(
    sorted(
        "ами ями ові еві ою ею ёю ой ей ёй ом ем ём ам ям ах ях ым им ів а я у ю е и ы о ь й і є ї".split(),
        key=len,
        reverse=True,
    )
)


_VOWELS = "аеёиоуыэюяіїє"


def name_stem(token: str) -> str:
    """Strip the case ending and a trailing vowel: Олена, Оленой, Олену all become 'олен'.

    The second step makes Алексей (stem 'алекс') and Алексея ('алексе' -> 'алекс') agree.
    """
    if token.isascii():
        return token
    for ending in _CASE_ENDINGS:
        if token.endswith(ending) and len(token) - len(ending) >= 3:
            token = token[: -len(ending)]
            break
    if len(token) > 3 and token[-1] in _VOWELS:
        token = token[:-1]
    return token


def name_key(text: str) -> frozenset[str]:
    return frozenset(name_stem(t) for t in tokenize(text) if len(t) >= 3)


def names_match(a: str, b: str) -> bool:
    """True when one name's stems are contained in the other's ("Олена" ~ "Олена Коваленко")."""
    ka, kb = name_key(a), name_key(b)
    return bool(ka) and bool(kb) and (ka <= kb or kb <= ka)


def keywords(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


def detect_lang(text: str) -> str:
    """Return ru / uk / en, or und when there is nothing to go on."""
    lowered = text.lower()
    letters = [c for c in lowered if c.isalpha()]
    if not letters:
        return "und"
    cyrillic = sum("а" <= c <= "я" or c in "ёіїєґ" for c in letters)
    if cyrillic < len(letters) / 2:
        return "en"
    uk = sum(c in _UK_LETTERS for c in letters)
    ru = sum(c in _RU_LETTERS for c in letters)
    if uk != ru:
        return "uk" if uk > ru else "ru"
    words = set(tokenize(lowered))
    uk_hits, ru_hits = len(words & _UK_HINTS), len(words & _RU_HINTS)
    return "uk" if uk_hits > ru_hits else "ru"


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0] or text[:limit]
    return cut.rstrip(",.;:- ") + "…"
