"""Long-tunnel check: keep an SSH tunnel to a live Colab session open for N minutes and log whether it still
forwards (the runtime token lasts ~1 h; does the tunnel survive it?). Also runs the keep-alive exec every ~100 s.

    uv run python scripts/tunnel_check.py SESSION [--minutes 75]

SESSION must already exist (`colab new -s SESSION`); the script refuses to open a tunnel for a name that a trusted
listing does not show, because `colab ssh --proxy-mode -s NAME` would create a new paid VM.
"""

from __future__ import annotations

import argparse
import asyncio
import tempfile
import time
from pathlib import Path

import httpx

from colabinet.colab_cli import ColabCLI
from colabinet.config import Settings
from colabinet.engines.base import VM_DIR, VM_ROOT
from colabinet.net import first_free
from colabinet.tunnel import SshTunnels

DUMMY_SERVER = (
    "import http.server\n"
    "http.server.ThreadingHTTPServer(('127.0.0.1', 8090), http.server.SimpleHTTPRequestHandler).serve_forever()\n"
)


async def listed_endpoint(cli: ColabCLI, name: str) -> str | None:
    listing = await cli.listing()
    rt = listing.named(name) if listing.trusted else None
    return rt.endpoint if rt else None


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--minutes", type=float, default=75)
    a = ap.parse_args()
    s = Settings()
    cli = ColabCLI(s.colab_bin, now=time.time, log=print)
    endpoint = await listed_endpoint(cli, a.session)
    if not endpoint:
        raise SystemExit(f"{a.session!r} is not in a trusted listing - refusing (proxy-mode would create a VM)")
    prep = await cli.exec_file(a.session, VM_DIR / "prepare.py")  # uploads need the directory to exist
    if not (prep.status and prep.status.get("prepared")):
        raise SystemExit(f"prepare failed: {prep.out}")
    with tempfile.TemporaryDirectory() as d:
        server = Path(d) / "dummy_server.py"
        server.write_text(DUMMY_SERVER)
        r = await cli.upload(a.session, server, f"{VM_ROOT}/dummy_server.py")
        if not r.ok:
            raise SystemExit(f"upload failed: {r.out}")
    r = await cli.exec_file(a.session, VM_DIR / "bg.py",
                            {"SCRIPT": f"{VM_ROOT}/dummy_server.py", "OUT": f"{VM_ROOT}/dummy.out"})
    print(r.out.strip())
    hidden = first_free(23000, 23999)
    tunnels = SshTunnels(s.colab_bin, s.ssh_key, s.log_dir)
    pid = await tunnels.start(a.session, hidden, 8090)
    t0, last_keep, ok, failed = time.time(), 0.0, 0, 0
    try:
        async with httpx.AsyncClient() as c:
            while time.time() - t0 < a.minutes * 60:
                try:
                    good = (await c.get(f"http://127.0.0.1:{hidden}/", timeout=10)).status_code == 200
                except httpx.HTTPError:
                    good = False
                ok, failed = ok + good, failed + (not good)
                alive = tunnels.alive(pid, a.session)
                print(f"{(time.time() - t0) / 60:5.1f} min  http={'ok' if good else 'FAIL'}  ssh_alive={alive}",
                      flush=True)
                if not alive:
                    if await listed_endpoint(cli, a.session) == endpoint:  # safe tunnel rule
                        print("ssh exited - restarting it")
                        pid = await tunnels.start(a.session, hidden, 8090)
                    else:
                        print("session no longer listed with the same endpoint - ending the check")
                        break
                if time.time() - last_keep >= 100:
                    st = await cli.exec_file(a.session, VM_DIR / "status.py", {"VM_PORT": "8090"})
                    last_keep = time.time()
                    print("  keep-alive", "ok" if st.status else f"FAILED: {st.out[-160:]}")
                await asyncio.sleep(60)
    finally:
        await tunnels.stop(pid, a.session)
    print(f"done: {ok} ok, {failed} failed over {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    asyncio.run(main())
