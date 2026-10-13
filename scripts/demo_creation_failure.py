"""Review failed-creation diagnostics with a fake CLI only; no real Colab runtime is allocated."""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
import tempfile
from pathlib import Path

from colabinet.api import create_app
from colabinet.net import start_server, stop_server


async def main(port: int) -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from tests.integration.conftest import make_world

    data = root / "data"
    data.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="creation-error-demo-", dir=data) as directory:
        world = await make_world(Path(directory))
        world.settings.reconcile_s = world.settings.usage_s = 30
        world.settings.guard_s = 5
        server = task = None
        stop = asyncio.Event()
        for kind in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_running_loop().add_signal_handler(kind, stop.set)
        try:
            world.fake.fault(new="not_available", new_error=(
                "[colab] Creating session 'failed-demo'...\n"
                "Traceback (most recent call last):\n"
                "  allocate_runtime(gpu='A100')\n"
                "Authorization: Bearer example-secret\n"
                "QuotaError: Backend rejected accelerator 'A100'. You may not have quota."
            ))
            sid = await world.sup.create(world.request(name="failed-demo", gpu="A100"))
            await world.wait_state(sid, "failed")
            app = create_app(world.sup, "demo-token", port, csrf="demo-csrf")
            server, task = await start_server(app, port, lifespan=True)
            print(f"Fake-only diagnostics preview: http://127.0.0.1:{port}/sessions/{sid}", flush=True)
            await stop.wait()
        finally:
            if server is not None:
                await stop_server(server, task)
            await world.sup.shutdown()
            world.fake.kill_all()
            world.store.db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7870)
    asyncio.run(main(parser.parse_args().port))
