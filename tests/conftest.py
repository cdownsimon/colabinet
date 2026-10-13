import asyncio
import gc
import os
import sys
import threading

import pytest

from colabinet.db import Database
from colabinet.store import Store

# pytest's faulthandler_timeout shows threads; an asyncio hang is a task awaiting something, so a test still running
# after this many seconds also prints every running loop's pending tasks.
HANG_REPORT_S = float(os.environ.get("COLABINET_HANG_REPORT_S", "120"))


def report_pending_tasks(nodeid: str) -> None:
    loops = [obj for obj in gc.get_objects() if isinstance(obj, asyncio.AbstractEventLoop) and obj.is_running()]
    for loop in loops:
        tasks = list(asyncio.all_tasks(loop))
        print(f"\n{nodeid} still running: {len(tasks)} pending asyncio tasks", file=sys.stderr)
        for task in tasks:
            print(f"--- {task!r}", file=sys.stderr)
            task.print_stack(limit=15, file=sys.stderr)
        woke = threading.Event()
        loop.call_soon_threadsafe(woke.set)
        print(f"event loop responsive: {woke.wait(5)}", file=sys.stderr, flush=True)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    done = threading.Event()

    def watchdog() -> None:
        if not done.wait(HANG_REPORT_S):
            capture = item.config.pluginmanager.getplugin("capturemanager")
            with capture.global_and_fixture_disabled():  # a hung run never shows captured output
                report_pending_tasks(item.nodeid)

    threading.Thread(target=watchdog, name="hang-report", daemon=True).start()
    try:
        yield
    finally:
        done.set()


class FakeClock:
    def __init__(self, t: float = 1_800_000_000.0) -> None:
        self.t = t

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds

    def day_bounds(self, t: float) -> tuple[float, float]:
        start = t - (t % 86400)
        return start, start + 86400

    def day_key(self, t: float) -> str:
        return str(int(t // 86400))


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
async def store(tmp_path):
    db = Database(tmp_path / "t.db")
    db.init()
    yield Store(db)
    db.close()
