"""`colabinet serve`: single instance, API token, supervisor, API server, signal handling (spec 5.5)."""

from __future__ import annotations

import asyncio
import fcntl
import logging
import os
import secrets
import signal
import time
from typing import IO

from .accounts import Accounts
from .alerts import Alerts, MacNotifier, NullNotifier
from .api import create_app
from .colab_cli import ColabCLI
from .config import Clock, Settings
from .db import Database
from .doctor import require_colab
from .engines import Engine
from .net import start_server, stop_server
from .presets import PresetError, load_presets
from .proxy import ProxyManager
from .store import Store
from .supervisor import Supervisor
from .tunnel import SshTunnels, Tunnels

log = logging.getLogger("colabinet")


class AlreadyRunning(RuntimeError):
    pass


def acquire_lock(settings: Settings) -> IO:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    f = open(settings.lock_path, "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as e:
        f.close()
        raise AlreadyRunning(f"another colabinet is running (lock {settings.lock_path})") from e
    f.write(str(os.getpid()))
    f.flush()
    return f


def write_token(settings: Settings) -> str:
    token = secrets.token_urlsafe(32)
    fd = os.open(settings.token_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(token)
    os.chmod(settings.token_path, 0o600)
    return token


def setup_logging(settings: Settings) -> None:
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger("colabinet")
    root.setLevel(logging.INFO)
    if not root.handlers:
        for h in (logging.FileHandler(settings.log_dir / "colabinet.log"), logging.StreamHandler()):
            h.setFormatter(fmt)
            root.addHandler(h)


def preflight(settings: Settings, alerts: Alerts) -> None:
    require_colab(settings)
    if not settings.ssh_key.exists():
        alerts.raise_("preflight", f"SSH key {settings.ssh_key} missing (ed25519 needed; RSA is rejected)",
                      key="preflight_ssh")


def build_supervisor(settings: Settings, store: Store, *, tunnels: Tunnels | None = None,
                     engines: dict[str, Engine] | None = None, alerts: Alerts | None = None) -> Supervisor:
    try:
        presets = load_presets(settings.presets_path)
    except PresetError as e:
        presets = e
        log.error("presets: %s", e)
    alerts = alerts or Alerts(MacNotifier() if settings.notify else NullNotifier(), time.time)
    cli = ColabCLI(settings.colab_bin, now=time.time, log=lambda m: log.info(m), auth=settings.colab_auth)
    return Supervisor(settings=settings, store=store, cli=cli,
                      tunnels=tunnels or SshTunnels(settings.colab_bin, settings.ssh_key, settings.log_dir,
                                                   auth=settings.colab_auth),
                      proxies=ProxyManager(), alerts=alerts, clock=Clock(), presets=presets, engines=engines,
                      hf_token=os.environ.get("HF_TOKEN"),
                      accounts=Accounts(settings.data_dir) if settings.colab_auth == "oauth2" else None)


async def serve(settings: Settings, *, tunnels: Tunnels | None = None, engines: dict[str, Engine] | None = None,
                stop_event: asyncio.Event | None = None, ready: asyncio.Event | None = None) -> list[str]:
    lock = acquire_lock(settings)
    try:
        setup_logging(settings)
        token = write_token(settings)
        db = Database(settings.db_path)
        db.init()
        sup = build_supervisor(settings, Store(db), tunnels=tunnels, engines=engines)
        preflight(settings, sup.alerts)
        await sup.start()
        csrf = secrets.token_urlsafe(24)
        server, task = await start_server(create_app(sup, token, settings.ui_port, csrf=csrf), settings.ui_port,
                                         lifespan=True)
        stop_event = stop_event or asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop_event.set)
            except (NotImplementedError, RuntimeError):
                pass
        log.info("colabinet console on http://127.0.0.1:%s (API token in %s)", settings.ui_port, settings.token_path)
        if ready:
            ready.set()
        await stop_event.wait()
        remaining = await sup.shutdown(stop_all=settings.stop_on_exit)
        if remaining:
            log.warning("sessions still running (they bill until stopped; stop them in the Colab web UI): "
                        "%s", ", ".join(remaining))
        await stop_server(server, task)
        db.close()
        return remaining
    finally:
        lock.close()
