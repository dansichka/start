import sys
import time
import types

import pytest

from call_companion import cli
from call_companion.ui import macos


def install(monkeypatch, name: str, module: types.ModuleType) -> None:
    monkeypatch.setitem(sys.modules, name, module)


class Window:
    def __init__(self):
        self.calls = []

    def setSharingType_(self, value): self.calls.append(("sharing", value))
    def setLevel_(self, value): self.calls.append(("level", value))
    def setCollectionBehavior_(self, value): self.calls.append(("behavior", value))


def test_window_flags_float_the_window_and_request_exclusion_from_sharing(monkeypatch):
    windows = [Window(), Window()]
    appkit = types.ModuleType("AppKit")
    appkit.NSApp = types.SimpleNamespace(windows=lambda: windows)
    appkit.NSWindowSharingNone = 0
    appkit.NSFloatingWindowLevel = 3
    appkit.NSWindowCollectionBehaviorCanJoinAllSpaces = 1
    appkit.NSWindowCollectionBehaviorFullScreenAuxiliary = 256
    install(monkeypatch, "AppKit", appkit)
    assert macos.apply_window_flags() == 2
    assert windows[0].calls == [("sharing", 0), ("level", 3), ("behavior", 257)]


def test_run_window_opens_a_floating_window_and_applies_the_flags_when_started(monkeypatch):
    seen = {"flags": 0, "ready": 0}
    webview = types.ModuleType("webview")
    webview.create_window = lambda title, url, **kw: seen.update(title=title, url=url, kwargs=kw)
    webview.start = lambda func: func()
    helper = types.ModuleType("PyObjCTools.AppHelper")
    helper.callAfter = lambda fn: seen.update(flags=seen["flags"] + 1)
    package = types.ModuleType("PyObjCTools")
    package.AppHelper = helper
    install(monkeypatch, "webview", webview)
    install(monkeypatch, "PyObjCTools", package)
    install(monkeypatch, "PyObjCTools.AppHelper", helper)
    macos.run_window("http://127.0.0.1:7777/?token=t", on_ready=lambda: seen.update(ready=1))
    assert seen["url"].startswith("http://127.0.0.1:7777") and seen["kwargs"]["on_top"] is True
    assert seen["flags"] == 1 and seen["ready"] == 1


def test_run_window_without_pywebview_explains_the_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "webview", None)
    with pytest.raises(RuntimeError, match=r"call-companion\[mac\]"):
        macos.run_window("http://x")


def test_windowed_run_opens_the_window_after_the_hud_is_up_and_stops_when_it_closes(monkeypatch):
    events = []

    def fake_window(url, **kwargs):
        events.append(("window", url))
        time.sleep(0.2)

    monkeypatch.setattr(macos, "run_window", fake_window)

    async def session(on_ready=None, stop=None):
        events.append("hud up")
        on_ready("http://127.0.0.1:7777/?token=t")
        await stop.wait()
        events.append("session stopped")

    assert cli.run_windowed(session) == 0
    assert events == ["hud up", ("window", "http://127.0.0.1:7777/?token=t"), "session stopped"]


def test_windowed_run_reports_a_session_that_failed_to_start(monkeypatch, capsys):
    monkeypatch.setattr(macos, "run_window", lambda url, **kw: pytest.fail("no window without a HUD"))

    async def session(on_ready=None, stop=None):
        raise RuntimeError("порт занят")

    assert cli.run_windowed(session) == 1
    assert "порт занят" in capsys.readouterr().err
