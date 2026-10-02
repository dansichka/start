from datetime import timedelta

from call_companion.bm25 import Bm25Index
from call_companion.kernel_api import MemoryDataSource, build_brief
from call_companion.models import SearchFilters

from .conftest import NOW, item


def ids(results) -> list[str]:
    return [r.id for r in results]


def test_ranks_relevant_items_first(items):
    index = Bm25Index()
    index.extend(items)
    assert ids(index.search("какой бюджет первого этапа", k=3))[0] == "note:55"
    assert ids(index.search("rate limit CRM батчинг", k=3))[0] == "gh:pr-214"


def test_matches_russian_inflections(items):
    index = Bm25Index()
    index.extend(items)
    assert ids(index.search("про сроки интеграции", k=2))[0] == "card:101"
    assert "card:130" in ids(index.search("обещания Максу про миграцию", k=3))


def test_no_overlap_returns_nothing(items):
    index = Bm25Index()
    index.extend(items)
    assert index.search("квантовая хромодинамика", k=5) == []
    assert index.search("   ", k=5) == []


def test_secret_never_returned_and_private_needs_permission(items):
    index = Bm25Index()
    index.extend(items)
    assert "stripe:9" not in ids(index.search("выплата карта EUR", k=10, filters=SearchFilters(max_sensitivity="private")))
    assert "tg:private1" not in ids(index.search("отпуск Грузия друг", k=10, filters=SearchFilters(max_sensitivity="work")))
    assert "tg:private1" in ids(index.search("отпуск Грузия друг", k=10, filters=SearchFilters(max_sensitivity="private")))


def test_filters_by_source_entity_and_age(items):
    index = Bm25Index()
    index.extend(items)
    assert ids(index.search("Orion", k=10, filters=SearchFilters(sources=["github"]))) == ["gh:pr-214"]
    assert set(ids(index.search("Orion", k=10, filters=SearchFilters(entities=["Max Weber"])))) == {"card:130"}
    recent = SearchFilters(since=NOW - timedelta(days=5))
    assert "card:101" not in ids(index.search("Orion сроки", k=10, filters=recent))


def test_newer_items_win_ties():
    old = item("old", "дедлайн релиза в пятницу", days_ago=120)
    new = item("new", "дедлайн релиза в пятницу", days_ago=1)
    index = Bm25Index()
    index.extend([old, new])
    assert ids(index.search("дедлайн релиза", k=2)) == ["new", "old"]


def test_scores_are_attached(items):
    index = Bm25Index()
    index.extend(items)
    results = index.search("бюджет", k=3)
    assert all(r.score > 0 for r in results)
    assert results == sorted(results, key=lambda r: r.score, reverse=True)


def test_brief_puts_attendee_items_first_and_extracts_loops(items, entities, meeting):
    source = MemoryDataSource(items, entities)
    brief = source.brief(meeting, "work")
    got = ids(brief.items)
    assert {"card:101", "card:130"} <= set(got)
    assert "tg:private1" not in got and "stripe:9" not in got
    assert any("план миграции" in loop for loop in brief.open_loops)
    assert any("поэтапную сдачу" in point for point in brief.talking_points)
    assert {"Orion", "Олена Коваленко", "Max Weber"} <= {e.name for e in brief.entities}


def test_brief_respects_sensitivity_limit(items, entities, meeting):
    brief = build_brief(Bm25Index(), items, entities, meeting, "public")
    assert brief.items == []


def test_upcoming_window(meeting):
    from datetime import datetime, timezone

    soon = meeting.model_copy(update={"id": "soon", "start": datetime.now(timezone.utc) + timedelta(minutes=10)})
    later = meeting.model_copy(update={"id": "later", "start": datetime.now(timezone.utc) + timedelta(hours=3)})
    source = MemoryDataSource(meetings=[soon, later])
    assert [m.id for m in source.upcoming(30)] == ["soon"]
