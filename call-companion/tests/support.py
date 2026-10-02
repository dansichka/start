"""Shared helpers for tests that run the demo call and a real HUD server."""

import asyncio
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn

from call_companion.app import Runtime, build_runtime
from call_companion.asr.replay import load_transcript, replay
from call_companion.clock import VirtualClock
from call_companion.config import PrivacyCfg, Settings, UserCfg
from call_companion.kernel.snapshot import SnapshotKernel
from call_companion.llm.fake import FakeLLM
from call_companion.models import Meeting

EXAMPLES = Path(__file__).parents[1] / "examples"


async def demo_runtime(tmp_path: Path, *, llm=None, mode=None, store_transcripts: bool = False, **settings) -> tuple[Runtime, VirtualClock]:
    """The fictional demo kernel is copied so write-back never touches the repository."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    pack = tmp_path / "pack.jsonl"
    shutil.copy(EXAMPLES / "sample_kernel_pack.jsonl", pack)
    cfg = Settings(
        user=UserCfg(name="Даня", aliases=["Daniil"]),
        privacy=PrivacyCfg(data_dir=str(tmp_path / "data"), store_transcripts=store_transcripts),
        **settings,
    )
    clock = VirtualClock()
    runtime = build_runtime(cfg, llm=llm or FakeLLM(), kernel=SnapshotKernel(pack), clock=clock, ephemeral=True)
    meeting = Meeting.model_validate_json((EXAMPLES / "meeting.json").read_text(encoding="utf-8"))
    await runtime.engine.start(meeting, mode=mode)
    return runtime, clock


async def play_demo(runtime: Runtime, clock: VirtualClock) -> None:
    await replay(runtime.engine, load_transcript(EXAMPLES / "sample_call.jsonl"), clock)


@asynccontextmanager
async def running(app):
    """Serve an ASGI app on a free local port; yields its base URL."""
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", lifespan="off", timeout_graceful_shutdown=1)
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await task
