"""Strip secrets from text before it leaves the machine or reaches the screen. Standard library only."""

from __future__ import annotations

import re


def _luhn_ok(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2:
            value = value * 2 - 9 if value > 4 else value * 2
        total += value
    return total % 10 == 0


def _iban_ok(raw: str) -> bool:
    iban = raw.replace(" ", "").upper()
    if not 15 <= len(iban) <= 34:
        return False
    rearranged = iban[4:] + iban[:4]
    try:
        return int("".join(str(int(c, 36)) for c in rearranged)) % 97 == 1
    except ValueError:
        return False


_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ключ", re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_\-]{20,}\b")),
    ("ключ", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b|\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("ключ", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    ("ключ", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("токен", re.compile(r"\b\d{8,10}:[A-Za-z0-9_-]{35}\b")),
    ("токен", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    ("токен", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{20,}")),
]
_KEY_VALUE = re.compile(r"(?i)\b(pass(?:word)?|passwd|pwd|пароль|secret|token|api[_-]?key)\b(\s*[:=]\s*)(\S+)")
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b")


class Redactor:
    def __init__(self, extra_patterns: list[str] | None = None) -> None:
        self._extra = [re.compile(p) for p in (extra_patterns or [])]

    def __call__(self, text: str) -> str:
        if not text:
            return text
        for label, pattern in _SECRET_PATTERNS:
            text = pattern.sub(f"[скрыто:{label}]", text)
        text = _KEY_VALUE.sub(lambda m: f"{m.group(1)}{m.group(2)}[скрыто]", text)
        text = _CARD.sub(lambda m: "[скрыто:карта]" if _luhn_ok(re.sub(r"\D", "", m.group())) else m.group(), text)
        text = _IBAN.sub(lambda m: "[скрыто:iban]" if _iban_ok(m.group()) else m.group(), text)
        for pattern in self._extra:
            text = pattern.sub("[скрыто]", text)
        return text
