import asyncio
import time
from dataclasses import dataclass

import pytest

from colabinet.alerts import Alerts, NullNotifier
from colabinet.colab_cli import ColabCLI
from colabinet.config import REPO_ROOT, Clock, Settings
from colabinet.db import Database
from colabinet.engines import VM_ROOT, LlamaCppEngine, StrataEngine
from colabinet.net import first_free
from colabinet.presets import load_presets
from colabinet.proxy import ProxyManager
from colabinet.store import Store
from colabinet.supervisor import CreateRequest, Supervisor
from colabinet.tunnel import LocalTunnels
from tests.fakes import FAKES, FakeColab


class FakeEngineAdapter(LlamaCppEngine):
    """llama.cpp adapter whose setup script is tests/fakes/fake_setup.py (starts tests/fakes/fake_engine.py)."""

    script = FAKES / "fake_setup.py"

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def setup_env(self, spec, vm_port, run_id):
        env = super().setup_env(spec, vm_port, run_id)
        env["FAKE_ENGINE"] = str(FAKES / "fake_engine.py")
        if self.fail:
            env["FAKE_SETUP_FAIL"] = "1"
        return env


class FakeExl3Adapter(FakeEngineAdapter):
    """Stands in for the ExLlamaV3 adapter: same fake VM setup, but the proxy injects include_usage for it."""

    name = "exllamav3"
    inject_usage_option = True


class FakeStrataAdapter(FakeEngineAdapter):
    """Fake runtime for Strata lifecycle checks; not a GPU inference or vision substitute."""

    name = "strata"
    validate_spec = staticmethod(StrataEngine.validate_spec)


@dataclass
class World:
    fake: FakeColab
    sup: Supervisor
    store: Store
    settings: Settings
    engine: FakeEngineAdapter

    def request(self, name="llm", gpu="L4", preset="qwen38-27b-abl-q4k", **kw) -> CreateRequest:
        return CreateRequest(name=name, gpu_class=gpu, preset_id=preset, confirm=True,
                             vm_port=first_free(30000, 30999), **kw)

    async def wait_state(self, sid, *states, timeout=30.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            row = await self.store.session(sid)
            if row["state"] in states:
                return row
            await asyncio.sleep(0.05)
        raise AssertionError(f"session {sid} stuck in {row['state']}, wanted {states}")

    async def wait_for(self, predicate, timeout=15.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if await predicate():
                return
            await asyncio.sleep(0.05)
        raise AssertionError("condition not met")


def fast_settings(tmp_path, fake) -> Settings:
    return Settings(
        data_dir=tmp_path / "data", colab_bin=str(fake.bin), keepalive_s=0.3, setup_poll_s=0.1, reconcile_s=0.3,
        guard_s=0.1, usage_s=0.5, stop_retry_s=0.1, endpoint_wait_s=2, create_slow_s=1, create_abandon_s=2,
        grace_s=1.0, gpu_check_limit_s=20, setup_limit_s=20, tunnel_alert_s=5, notify=False,
        proxy_port_start=first_free(28000, 28899), hidden_port_range=(29000, 29999),
    )


async def make_world(tmp_path, fake=None, db=None, engine=None, engines=None) -> World:
    fake = fake or FakeColab(tmp_path)
    settings = fast_settings(tmp_path, fake)
    if db is None:
        db = Database(settings.db_path)
        db.init()
    store = Store(db)
    engine = engine or FakeEngineAdapter()
    sup = Supervisor(settings=settings, store=store, cli=ColabCLI(str(fake.bin), now=time.time),
                     tunnels=LocalTunnels(), proxies=ProxyManager(), alerts=Alerts(NullNotifier(), time.time),
                     clock=Clock(), presets=load_presets(REPO_ROOT / "presets.toml"),
                     engines=engines or {"llamacpp": engine, "exllamav3": FakeExl3Adapter(),
                                         "strata": FakeStrataAdapter()})
    await sup.start()
    return World(fake, sup, store, settings, engine)


@pytest.fixture
async def world(tmp_path):
    w = await make_world(tmp_path)
    yield w
    await w.sup.shutdown()
    w.fake.kill_all()
    w.store.db.close()
    assert not [h for h in w.fake.hazards if h.startswith(("auto_create", "overwrite", "new_killed"))]


__all__ = ["VM_ROOT", "World", "make_world"]
