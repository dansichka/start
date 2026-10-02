"""Real Chromium against the real HUD server: rendering, reload hydration, clicks and XSS safety.

Set HUD_SCREENSHOTS=/some/dir to also save screenshots; CHROMIUM_PATH overrides the browser binary.
"""

import asyncio
import glob
import os
from pathlib import Path

import pytest

pytest.importorskip("playwright")
from playwright.async_api import async_playwright  # noqa: E402

from call_companion.app import finish_call  # noqa: E402
from call_companion.models import ProposedAction  # noqa: E402
from call_companion.ui.server import create_app  # noqa: E402

from .support import demo_runtime, play_demo, running  # noqa: E402

CHROMIUM = os.environ.get("CHROMIUM_PATH") or next(iter(sorted(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome"))), None)
pytestmark = pytest.mark.skipif(not CHROMIUM, reason="нет Chromium")
SHOTS = os.environ.get("HUD_SCREENSHOTS")
TOKEN = "tok-browser"


@pytest.fixture
async def finished(tmp_path):
    runtime, clock = await demo_runtime(tmp_path)
    await play_demo(runtime, clock)
    await finish_call(runtime)
    app = create_app(runtime.engine, runtime.bus, runtime.settings, token=TOKEN)
    async with running(app) as base:
        yield runtime, base


@pytest.fixture
async def browser():
    async with async_playwright() as playwright:
        instance = await playwright.chromium.launch(executable_path=CHROMIUM, args=["--no-sandbox"])
        yield instance
        await instance.close()


async def open_page(browser, base, query="", **context):
    page = await (await browser.new_context(**context)).new_page()
    page.set_default_timeout(6000)
    page.errors = []
    page.on("pageerror", lambda error: page.errors.append(str(error)))
    page.on("console", lambda msg: msg.type == "error" and page.errors.append(msg.text))
    await page.goto(f"{base}/?token={TOKEN}{query}")
    await page.wait_for_selector(".whisper")
    return page


async def shot(page, name):
    if SHOTS:
        Path(SHOTS).mkdir(parents=True, exist_ok=True)
        await page.screenshot(path=str(Path(SHOTS) / f"{name}.png"))


async def eventually(predicate, timeout=4.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        assert asyncio.get_running_loop().time() < deadline, "условие не выполнилось вовремя"
        await asyncio.sleep(0.05)


async def test_hud_shows_the_finished_demo_call_and_survives_a_reload(finished, browser):
    runtime, base = finished
    page = await open_page(browser, base, viewport={"width": 1280, "height": 900})
    for attempt in ("first load", "after reload"):
        assert await page.inner_text("#status-text") == "Звонок завершён", attempt
        assert await page.inner_text("#title") == "Статус Orion"
        kinds = await page.locator(".whisper .kind").all_inner_texts()
        assert kinds == ["НАПОМИНАНИЕ", "ОТВЕТ", "ОТВЕТ", "ФАКТ"], "newest whisper first"
        chips = " | ".join(await page.locator(".whisper .chip").all_inner_texts())
        assert "Договорённости по бюджету Orion" in chips and "kernel" in chips
        assert await page.locator(".line").count() == 13
        assert await page.locator(".approval[data-status=pending]").count() == 4
        assert await page.locator("#summary-wrap").is_visible()
        assert "план миграции" in await page.inner_text("#actions")
        if attempt == "first load":
            await shot(page, "hud-light")
            await page.reload()
            await page.wait_for_selector(".whisper")
    assert page.errors == []


async def test_untrusted_text_is_rendered_as_text_not_markup(finished, browser):
    runtime, base = finished
    page = await open_page(browser, base)
    bus = runtime.bus
    bus.publish("utterance", id="evil1", channel="them", speaker="<b>Evil</b>", text='<img src=x onerror="window.__xss=1"> привет', t_start=0, t_end=1, final=True)
    bus.publish(
        "suggestion", id="evil2", kind="warn", final=True, trigger="manual", evidence=[],
        text='<script>window.__xss=2</script><img src=x onerror="window.__xss=3">',
        sources=[{"id": "x", "source": "<i>src</i>", "title": '"><script>window.__xss=4</script>', "ts": None, "url": None}],
    )
    runtime.engine.propose(ProposedAction(id="a", capability="kernel.note", title='<img src=x onerror="window.__xss=5">'))
    await page.wait_for_selector("#whispers .k-warn")
    await page.wait_for_selector(".approval:has-text('<img src=x')")
    assert await page.evaluate("window.__xss === undefined")
    assert "<img src=x" in await page.inner_text("#lines") and "<b>Evil</b>" in await page.inner_text("#lines")
    assert "<script>" in await page.inner_text("#whispers .k-warn .text")
    assert await page.locator("#whispers img, #lines img, .approval img, #whispers script, #lines b").count() == 0
    assert page.errors == []


async def test_partial_whispers_stream_in_and_can_be_cancelled(finished, browser):
    runtime, base = finished
    page = await open_page(browser, base)
    runtime.bus.publish("suggestion", id="p1", kind="answer", text="Согласованный бю", final=False, trigger="stall")
    await page.wait_for_selector(".whisper.live")
    assert await page.inner_text(".whisper.live .text") == "Согласованный бю"
    runtime.bus.publish("suggestion", id="p1", kind="answer", text="Согласованный бюджет — 18 400 EUR", final=True, trigger="stall", sources=[])
    await page.wait_for_selector(".whisper.live", state="detached")
    assert await page.inner_text(".whisper:first-child .text") == "Согласованный бюджет — 18 400 EUR"
    runtime.bus.publish("suggestion", id="p2", kind="fact", text="временная", final=False, trigger="entity")
    await page.wait_for_selector("[data-whisper=p2]")
    runtime.bus.publish("suggestion_cancel", id="p2")
    await page.wait_for_selector("[data-whisper=p2]", state="detached")
    runtime.bus.publish("clear")
    await page.wait_for_selector(".whisper", state="detached")
    assert await page.is_visible("#empty")


async def test_controls_drive_the_engine(finished, browser):
    runtime, base = finished
    page = await open_page(browser, base)
    await page.click("#mute")
    await eventually(lambda: runtime.engine.gate.muted)
    await page.wait_for_function("document.querySelector('#mute').getAttribute('aria-pressed') === 'true'")
    await page.select_option("#mode", "internal")
    await eventually(lambda: runtime.engine.mode == "internal")
    count = len(runtime.engine.suggestions)
    await page.fill("#ask-text", "какой бюджет этапа 1?")
    await page.press("#ask-text", "Enter")
    await eventually(lambda: len(runtime.engine.suggestions) == count + 1)
    assert runtime.engine.suggestions[-1].trigger == "manual"
    await page.wait_for_function("document.querySelector('.whisper .kind') !== null && document.querySelectorAll('.whisper').length >= 5")
    await page.keyboard.press("Escape")
    await page.wait_for_selector(".whisper", state="detached")
    assert page.errors == []


async def test_approvals_can_be_decided_one_by_one_or_all_at_once(finished, browser):
    runtime, base = finished
    page = await open_page(browser, base)
    await page.locator(".approval").first.get_by_role("button", name="Разрешить").click()
    await page.wait_for_selector(".approval[data-status=executed]")
    await page.locator(".approval[data-status=pending]").first.get_by_role("button", name="Отклонить").click()
    await page.wait_for_selector(".approval[data-status=rejected]")
    assert await page.is_visible("#approve-all")
    await page.click("#approve-all")
    await page.wait_for_function("document.querySelectorAll('.approval[data-status=pending]').length === 0")
    statuses = await page.locator(".approval").evaluate_all("nodes => nodes.map(n => n.dataset.status)")
    assert sorted(set(statuses)) == ["executed", "failed", "rejected"], "tasks fail on a snapshot-only kernel, cleanly"
    assert not await page.is_visible("#approve-all")


async def test_layout_fits_a_phone_and_compact_mode_hides_the_sidebar(finished, browser):
    runtime, base = finished
    phone = await open_page(
        browser, base, viewport={"width": 390, "height": 844}, device_scale_factor=2, is_mobile=True, has_touch=True, color_scheme="dark"
    )
    assert await phone.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
    await shot(phone, "hud-phone")
    compact = await open_page(browser, base, "&compact=1", viewport={"width": 440, "height": 620}, color_scheme="dark")
    assert not await compact.is_visible("#side")
    await shot(compact, "hud-compact")
    dark = await open_page(browser, base, viewport={"width": 1100, "height": 760}, color_scheme="dark")
    assert await dark.evaluate("getComputedStyle(document.body).backgroundColor") == "rgb(14, 16, 21)"
    await shot(dark, "hud-dark")
