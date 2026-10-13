import asyncio
import os

import pytest

from colabinet.colab_cli import ColabCLI, redact, scrub
from tests.fakes import FakeColab


@pytest.fixture
def fake(tmp_path):
    f = FakeColab(tmp_path)
    yield f
    f.kill_all()


@pytest.fixture
def cli(fake):
    return ColabCLI(str(fake.bin), now=lambda: 1.0)


def test_scrub_hides_runtime_proxy_tokens_in_output():
    out = "Upload failed: 500 for url: https://x.colab.dev/api/contents/a?authuser=0&colab-runtime-proxy-token=ey.J.k"
    assert scrub(out).endswith("authuser=0&colab-runtime-proxy-token=***") and "ey.J.k" not in scrub(out)


def test_redact_hides_secret_env_values():
    assert redact(["exec", "--env", "HF_TOKEN=abc", "--env", "LLM_PORT=8090"]) == \
        "exec --env HF_TOKEN=*** --env LLM_PORT=8090"


async def test_new_listing_and_stop(cli, fake):
    proc = await cli.spawn_new("llm", "L4")
    out, _ = await proc.communicate()
    assert proc.returncode == 0 and b"Session READY." in out
    listing = await cli.listing()
    assert listing.trusted
    rt = listing.named("llm")
    assert rt.hardware == "L4" and rt.endpoint.startswith("gpu-l4-")
    assert listing.usage.active_assignments == 1
    assert (await cli.stop("llm")).ok
    after = await cli.listing()
    assert after.trusted and after.runtimes == []


async def test_hidden_auth_failure_is_untrusted(cli, fake):
    proc = await cli.spawn_new("llm", "L4")
    await proc.communicate()
    st = fake.read()
    st["store"] = {}  # the CLI lost its local entry: the VM shows up as "[?]"...
    st["faults"]["hide_when_store_empty"] = True  # ...unless auth fails, then it prints "No active sessions"
    fake.write(st)
    listing = await cli.listing()
    assert listing.runtimes == [] and not listing.trusted


async def test_exec_parses_status_line(cli, fake, tmp_path):
    proc = await cli.spawn_new("llm", "L4")
    await proc.communicate()
    script = tmp_path / "s.py"
    script.write_text('import os, json\nprint("COLABINET_STATUS " + json.dumps({"x": os.environ["X"]}))\n')
    res = await cli.exec_file("llm", script, {"X": "42"})
    assert res.ok and res.status == {"x": "42"}


async def test_exec_drop_is_reported_not_raised(cli, fake, tmp_path):
    proc = await cli.spawn_new("llm", "L4")
    await proc.communicate()
    fake.fault(exec_drop=1)
    script = tmp_path / "s.py"
    script.write_text("print('hi')\n")
    res = await cli.exec_file("llm", script)
    assert not res.ok and res.status is None and "Connection was lost" in res.out


async def test_watchdog_kills_hung_command(fake, tmp_path):
    cli = ColabCLI(str(fake.bin), now=lambda: 1.0, timeouts={"exec": 0.5})
    proc = await cli.spawn_new("llm", "CPU")
    await proc.communicate()
    script = tmp_path / "slow.py"
    script.write_text("import time; time.sleep(30)\n")
    res = await cli.exec_file("llm", script)
    assert res.timed_out and not res.ok


async def test_cancel_propagates_even_when_the_kill_fails(fake, tmp_path, monkeypatch):
    """macOS answers EPERM when killpg hits a group of zombies. If that error replaced the CancelledError, a
    background loop would log it and keep going, and shutdown would wait for that loop forever."""
    cli = ColabCLI(str(fake.bin), now=lambda: 1.0)
    proc = await cli.spawn_new("llm", "CPU")
    await proc.communicate()
    script = tmp_path / "slow.py"
    script.write_text("import time; time.sleep(30)\n")
    real_killpg = os.killpg

    def killpg_then_eperm(pid, sig):
        real_killpg(pid, sig)
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(os, "killpg", killpg_then_eperm)
    task = asyncio.create_task(cli.exec_file("llm", script))
    await asyncio.sleep(1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)


async def test_exec_env_persists_in_the_kernel_like_the_real_cli(cli, fake, tmp_path):
    proc = await cli.spawn_new("llm", "CPU")
    await proc.communicate()
    script = tmp_path / "s.py"
    script.write_text('import os, json\nprint("COLABINET_STATUS " + json.dumps({"a": os.environ.get("A")}))\n')
    assert (await cli.exec_file("llm", script, {"A": "1"})).status == {"a": "1"}
    assert (await cli.exec_file("llm", script)).status == {"a": "1"}  # --env sets os.environ in the kernel


async def test_version(cli):
    assert await cli.version() == "0.7.4"
