import pytest

from call_companion.kernel.memory import InMemoryKernel
from call_companion.kernel_api import MemoryDataSource
from call_companion.live.retrieval import ContextRetriever
from call_companion.models import Brief, Entity
from call_companion.policy import PolicyEngine

from .conftest import item


def retriever(kernel, brief=None, mode="client", **kwargs) -> ContextRetriever:
    r = ContextRetriever(kernel, PolicyEngine(), mode=mode, ignore=["Даня"], **kwargs)
    r.set_brief(brief)
    return r


async def test_relevant_items_are_labelled_and_ranked(items):
    evidence = await retriever(InMemoryKernel(MemoryDataSource(items))).retrieve("Даня, какой бюджет первого этапа?")
    assert list(evidence.labels) == ["E1"] and evidence.labels["E1"].id == "note:55"
    assert not evidence.degraded and evidence.latency_ms >= 0


async def test_unrelated_questions_find_nothing(items):
    evidence = await retriever(InMemoryKernel(MemoryDataSource(items))).retrieve("как тебе квантовая хромодинамика?")
    assert evidence.labels == {}
    assert (await retriever(InMemoryKernel()).retrieve("   ")).labels == {}


async def test_the_users_own_name_is_not_a_search_term():
    only_name = [item("n", "Даня сходил в магазин", entities=[])]
    evidence = await retriever(InMemoryKernel(MemoryDataSource(only_name))).retrieve("Даня, привет")
    assert evidence.labels == {}


async def test_entity_linked_items_come_first(items):
    orion = Entity(name="Orion", kind="project")
    evidence = await retriever(InMemoryKernel(MemoryDataSource(items))).retrieve("Давайте про Orion", [orion])
    assert evidence.labels and all("Orion" in i.entities for i in evidence.items)


async def test_kernel_outage_is_flagged_and_the_brief_still_answers(items, entities, meeting):
    kernel = InMemoryKernel(MemoryDataSource(items, entities))
    brief = await kernel.brief(meeting)
    kernel.fail = True
    evidence = await retriever(kernel, brief).retrieve("какой бюджет первого этапа?")
    assert evidence.degraded and evidence.labels["E1"].id == "note:55"


async def test_slow_kernel_is_cut_by_the_budget(items):
    kernel = InMemoryKernel(MemoryDataSource(items), latency_s=1.0)
    evidence = await retriever(kernel, budget_s=0.1).retrieve("какой бюджет первого этапа?")
    assert evidence.degraded and evidence.labels == {}


async def test_the_same_item_from_brief_and_kernel_appears_once(items, meeting):
    kernel = InMemoryKernel(MemoryDataSource(items))
    brief = Brief(meeting=meeting, items=[items[1]])
    evidence = await retriever(kernel, brief).retrieve("какой бюджет первого этапа?")
    assert [i.id for i in evidence.items].count("note:55") == 1


async def test_remote_results_are_filtered_by_mode_and_redacted():
    rows = [
        item("w", "Бюджет проекта. token: abc123", title="Бюджет"),
        item("p", "Бюджет личный", sensitivity="private"),
    ]
    evidence = await retriever(InMemoryKernel(MemoryDataSource(rows))).retrieve("Бюджет проекта")
    assert [i.id for i in evidence.items] == ["w"] and "abc123" not in evidence.items[0].text
    internal = await retriever(InMemoryKernel(MemoryDataSource(rows)), mode="internal").retrieve("Бюджет личный")
    assert "p" in [i.id for i in internal.items]
