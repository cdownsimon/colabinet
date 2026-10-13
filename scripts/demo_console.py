"""Fake-only console on port 7869. No real Colab executable or runtime is used."""

from __future__ import annotations

import argparse
import asyncio
import random
import signal
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import httpx

from colabinet.api import create_app
from colabinet.net import first_free, start_server, stop_server


async def main(port: int) -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from tests.integration.conftest import make_world

    data = root / "data"
    data.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="demo-", dir=data) as directory:
        world = await make_world(Path(directory))
        world.settings.keepalive_s = 5
        world.settings.proxy_port_start = first_free(34000, 34999)
        world.settings.hidden_port_range = (35000, 35999)
        server = task = None
        stop = asyncio.Event()
        for signal_kind in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_running_loop().add_signal_handler(signal_kind, stop.set)
        try:
            sessions = []
            for name, preset in (("llm", "qwen38-27b-abl-q4k"), ("coder", "qwen38-27b-abl-exl3-4bpw")):
                request = replace(world.request(name=name, preset=preset), vm_port=first_free(32000, 32999))
                sid = await world.sup.create(request)
                await world.wait_state(sid, "ready", timeout=60)
                # Fake VMs run locally; use representative allocations instead of the developer's host RAM.
                await world.store.update_session(sid, ram_gb=52.0)
                sessions.append(sid)
            async with httpx.AsyncClient() as client:
                for sid in sessions:
                    row = await world.store.session(sid)
                    for _ in range(12):
                        await client.post(f"http://127.0.0.1:{row['proxy_port']}/v1/chat/completions",
                                          json={"stream": True, "x_tokens": random.randint(3, 12), "x_delay": 0.01,
                                                "x_shape": "tabby" if sid == sessions[1] else "llamacpp"},
                                          headers={"User-Agent": random.choice(
                                              ["pi/0.99", "curl/8.5", "openai-python"])})
            request = replace(world.request(name="old-cpu", gpu="CPU"), vm_port=first_free(32000, 32999))
            old = await world.sup.create(request)
            await world.wait_state(old, "ready", timeout=60)
            await world.store.update_session(old, ram_gb=12.8)
            await world.sup.stop(old)
            state = world.fake.read()
            state["balance"] = 8.2
            world.fake.write(state)
            world.sup._activity.pop(sessions[0], None)
            await world.store.update_session(sessions[0], idle_min=30, last_activity=world.sup.now() - 60)
            app = create_app(world.sup, "demo-token", port, csrf="demo-csrf")
            server, task = await start_server(app, port, lifespan=True)
            print(f"demo console on http://127.0.0.1:{port} (Ctrl-C to stop)", flush=True)
            await stop.wait()
        finally:
            if server is not None:
                await stop_server(server, task)
            await world.sup.shutdown()
            world.fake.kill_all()
            world.store.db.close()
            print("demo stopped; owned fake processes cleaned up", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7869)
    asyncio.run(main(parser.parse_args().port))
