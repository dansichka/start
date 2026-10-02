from datetime import datetime, timedelta, timezone

import pytest

from call_companion.models import ContextItem, Entity, Meeting, Participant

NOW = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)


def item(id: str, text: str, *, title: str = "", source: str = "note", kind: str = "note", entities=(), sensitivity="work", days_ago: float = 1.0, **extra) -> ContextItem:
    return ContextItem(
        id=id,
        source=source,
        kind=kind,
        title=title,
        text=text,
        entities=list(entities),
        sensitivity=sensitivity,
        ts=NOW - timedelta(days=days_ago),
        **extra,
    )


@pytest.fixture
def items() -> list[ContextItem]:
    return [
        item(
            "card:101",
            "Олена написала 12.09: релиз интеграции Orion нужен до 15 октября, иначе срываем запуск на складе в Гамбурге.",
            title="Олена Коваленко — сроки",
            source="telegram",
            kind="message",
            entities=["Олена Коваленко", "Orion", "Acme Logistics"],
            days_ago=20,
        ),
        item(
            "note:55",
            "Согласованный бюджет этапа 1 — 18 400 EUR, оплата 50/50. Этап 2 (аналитика) оценён в 9 200 EUR, не согласован.",
            title="Договорённости по бюджету Orion",
            entities=["Orion", "Acme Logistics"],
            days_ago=9,
        ),
        item(
            "gh:pr-214",
            "Webhook-синхронизация заказов смержена 28.09. Остался rate limit CRM: 60 запросов в минуту, нужен батчинг.",
            title="PR #214: синхронизация заказов",
            source="github",
            kind="pr",
            entities=["Orion"],
            days_ago=4,
        ),
        item(
            "card:130",
            "Обещал Максу Веберу прислать обновлённый план миграции данных до пятницы.",
            title="Обещал Максу",
            source="kernel",
            kind="open_loop",
            entities=["Max Weber", "Orion"],
            days_ago=3,
        ),
        item(
            "note:70",
            "Предложить поэтапную сдачу: ядро синхронизации до 15 октября, аналитика позже.",
            kind="talking_point",
            entities=["Orion"],
            days_ago=2,
        ),
        item("tg:private1", "Личное: друг зовёт в отпуск в Грузию.", source="telegram", sensitivity="private", days_ago=1),
        item("stripe:9", "Выплата 3 120 EUR на карту 4111 1111 1111 1111.", source="stripe", sensitivity="secret", days_ago=1),
    ]


@pytest.fixture
def entities() -> list[Entity]:
    return [
        Entity(name="Orion", kind="project", aliases=["Орион"], summary="Интеграция CRM со складом"),
        Entity(name="Acme Logistics", kind="org", aliases=["Acme", "Акме"]),
        Entity(name="Олена Коваленко", kind="person", aliases=["Олена"]),
        Entity(name="Max Weber", kind="person", aliases=["Макс Вебер", "Макс"]),
    ]


@pytest.fixture
def meeting() -> Meeting:
    return Meeting(
        id="m1",
        title="Статус Orion",
        start=NOW,
        attendees=[Participant(name="Олена Коваленко", org="Acme Logistics"), Participant(name="Max Weber")],
        agenda="Сроки и бюджет этапа 1",
        mode="client",
    )
