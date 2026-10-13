"""Session lifecycle, keep-alive, reconcile, guards and recovery (spec §5). The only unit that decides to spend or
stop. Every state change is written to the database before the action it describes."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
import math
import os
import signal
import statistics
from collections import deque
from dataclasses import dataclass
from typing import Any

import httpx

from . import billing
from .accounts import Accounts
from .activity import Activity
from .alerts import Alerts
from .cli_parse import Listing, Usage
from .colab_cli import ColabCLI
from .config import GPU_CLASSES, Settings, detected_class
from .diagnostics import bounded_diagnostic
from .engines import ENGINES, VM_ROOT, Engine, RunSpec
from .engines.base import VM_DIR
from .guards import evaluate, evaluate_daily
from .net import first_free, port_free
from .presets import PresetError, Presets, load_presets
from .proxy import ProxyManager, Route, SessionProxy
from .stats import RequestRecord
from .store import LIVE_STATES, Store
from .tunnel import SshTunnels, Tunnels, cmdline

log = logging.getLogger("colabinet.supervisor")

DRIVEN = {"creating", "gpu_check", "gpu_mismatch", "setting_up", "reconfiguring", "setup_failed", "tunnelling",
          "ready", "engine_down"}
OPEN_ROWS = "state NOT IN ('stopped', 'failed') OR stop_unconfirmed = 1 OR name_reserved = 1"
OBSERVE_INTERVAL_S = 15.0
OBSERVE_LEASE_S = 30.0


def stop_if_cancelled() -> None:
    """Raise CancelledError when this task was cancelled but carried on, so a cancelled loop always stops.

    A library can swallow a cancellation: on Python 3.12.3, httpx (through anyio) returned a normal response to a
    session driver cancelled during its health check, and the driver kept looping, so shutdown waited forever."""
    task = asyncio.current_task()
    if task is not None and task.cancelling():
        raise asyncio.CancelledError


class CreateRefused(Exception):
    pass


@dataclass
class CreateRequest:
    name: str
    gpu_class: str
    preset_id: str | None = None
    engine: str = "llamacpp"
    repo: str | None = None
    file: str | None = None
    revision: str | None = None
    parallel: int | None = None
    alias: str | None = None
    ctx: int | None = None
    kv: str | None = None
    refusal_projection: bool | None = None
    extra_args: tuple[str, ...] = ()
    proxy_port: int | None = None
    vm_port: int = 8090
    idle_min: int | None = None
    deadline: float | None = None
    cu_cap: float | None = None
    confirm: bool = False


class Supervisor:
    def __init__(self, *, settings: Settings, store: Store, cli: ColabCLI, tunnels: Tunnels, proxies: ProxyManager,
                 alerts: Alerts, clock, presets: Presets | PresetError, engines: dict[str, Engine] | None = None,
                 http: httpx.AsyncClient | None = None, hf_token: str | None = None,
                 accounts: Accounts | None = None) -> None:
        self.s = settings
        self.store = store
        self.cli = cli
        self.tunnels = tunnels
        self.accounts = accounts
        self.account_id = accounts.active if accounts else "default"
        self.account_change_pending = False
        if accounts:
            self.cli = accounts.cli(cli, self.account_id, initial=True)
            self._account_tunnels()
        self.proxies = proxies
        self.alerts = alerts
        self.clock = clock
        self.presets = presets
        self.engines = engines or ENGINES
        # A completed stream can leave a stale connection behind the SSH tunnel.
        # Open a fresh upstream connection rather than retrying an inference POST.
        self.http = http or httpx.AsyncClient(timeout=httpx.Timeout(5, read=None),
                                             limits=httpx.Limits(max_keepalive_connections=0))
        self.hf_token = hf_token
        self.usage: Usage | None = None
        self.cli_version: str | None = None
        self.usage_at: float | None = None
        self._drivers: dict[int, asyncio.Task] = {}
        self._new: dict[int, asyncio.Task] = {}  # sid -> task awaiting `colab new` (rc, output)
        self._new_pids: dict[int, int] = {}
        self._activity: dict[int, Activity] = {}
        self._view: dict[int, dict[str, Any]] = {}  # synchronous snapshot for the proxy's route()
        self._wake: dict[int, asyncio.Event] = {}
        self._health_probe_pending: set[int] = set()
        self._decision: dict[int, str] = {}
        self._pending_stop: dict[int, str] = {}
        self._misses: dict[int, int] = {}
        self._unhealthy: dict[int, int] = {}
        self._last_exec: dict[int, float] = {}
        self._logs: dict[int, deque[str]] = {}
        self._log_counts: dict[int, int] = {}
        self._observed_until: dict[int, float] = {}
        self._resource_samples: dict[int, dict[str, Any]] = {}
        self._listing_lock = asyncio.Lock()
        self._create_lock = asyncio.Lock()
        self._session_locks: dict[int, asyncio.Lock] = {}
        self._recovery_lock = asyncio.Lock()
        self._loops: list[asyncio.Task] = []
        self._paused_for_account: set[int] = set()

    # ------------------------------------------------------------------------------------------------- helpers
    def now(self) -> float:
        return self.clock.now()

    def _account_tunnels(self) -> None:
        if isinstance(self.tunnels, SshTunnels):
            self.tunnels.command_prefix = self.cli.command()

    async def begin_account_change(self, reconnect: str | None = None) -> None:
        async with self._create_lock:
            rows = await self.store.sessions(OPEN_ROWS)
            if rows and not (reconnect == self.account_id and reconnect
                             and all(row["account_id"] == reconnect for row in rows)):
                raise CreateRefused("Stop all managed sessions and wait for confirmed termination before "
                                    "adding, switching, or logging out of an account.")
            self.account_change_pending = True

    async def activate_account(self, account_id: str | None, cli: ColabCLI, usage: Usage | None,
                               save, *, reconnect: bool = False) -> None:
        # Creation and external stops cannot race a change; drain account-scoped background reads first.
        async with self._create_lock, self._listing_lock:
            rows = await self.store.sessions(OPEN_ROWS)
            if rows and not (reconnect and account_id == self.account_id
                             and all(row["account_id"] == account_id for row in rows)):
                raise CreateRefused("Stop all managed sessions before changing accounts.")
            cli.command()  # validate the launcher before persisting a changed account
            selected = save()
            self.account_id = selected if selected is not None else account_id
            self.cli = cli
            self._account_tunnels()
            self.usage, self.usage_at = usage, self.now() if usage else None
            self.alerts.clear("low_balance")
            for alert in self.alerts.list():
                if alert.key.startswith("external:") or alert.key in ("assignments_hidden", "account_reconnect"):
                    self.alerts.clear(alert.key)

    async def resume_account_sessions(self) -> None:
        async with self._recovery_lock:
            if not self.cli.enabled:
                return
            for sid in list(self._paused_for_account):
                async with self._session_locks.setdefault(sid, asyncio.Lock()):
                    row = await self.store.session(sid)
                    self._own_account(row)
                    if row["state"] == "stopping":
                        self._drivers[sid] = asyncio.create_task(self._stop_flow(sid))
                    elif row["state"] in LIVE_STATES:
                        await self._bind_proxy(row)
                        if row["state"] != "creating":
                            self._start_driver(sid)
                    self._paused_for_account.discard(sid)

    def _own_account(self, row: dict | None) -> None:
        if row and (self.account_id is None or row["account_id"] != self.account_id):
            raise CreateRefused("Switch to the account that owns this session in Settings.")

    async def fresh_listing(self) -> Listing:
        async with self._listing_lock:
            listing = await self.cli.listing()
            if listing.usage:
                self.usage, self.usage_at = listing.usage, listing.taken_at
        return listing

    async def _update(self, sid: int, **fields: Any) -> None:
        await self.store.update_session(sid, **fields)
        view = self._view.setdefault(sid, {})
        view.update({k: v for k, v in fields.items() if k in ("state", "hidden_port", "colab_name", "stop_reason")})

    async def _event(self, sid: int | None, kind: str, **detail: Any) -> None:
        await self.store.event(self.now(), kind, session_id=sid, **detail)

    async def _transition(self, sid: int, state: str, **fields: Any) -> None:
        row = await self.store.session(sid)
        now = self.now()
        await self._event(sid, "state", frm=row["state"] if row else None, to=state, **{
            k: v for k, v in fields.items() if k in ("stop_reason",)})
        if state == "tunnelling":
            fields.setdefault("last_activity", now)
            self._activity.setdefault(sid, Activity()).reset(now)
        await self._update(sid, state=state, state_since=now, **fields)
        log.info("session %s: %s -> %s", sid, row["state"] if row else None, state)

    def _start_driver(self, sid: int) -> None:
        task = self._drivers.get(sid)
        if task and not task.done():
            return
        self._drivers[sid] = asyncio.create_task(self._drive(sid), name=f"drive-{sid}")

    async def _cancel_driver(self, sid: int) -> None:
        task = self._drivers.pop(sid, None)
        if task and task is not asyncio.current_task():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _bind_proxy(self, row: dict[str, Any]) -> None:
        sid = row["id"]
        self._activity.setdefault(sid, Activity(row.get("last_activity")))
        self._view.setdefault(sid, {}).update(state=row["state"], hidden_port=row["hidden_port"],
                                              colab_name=row["colab_name"])
        if self.proxies.bound(sid):
            return
        app = SessionProxy(row["proxy_port"], lambda: self.route(sid), self.record, self._activity[sid], self.http,
                           self.now, on_upstream_error=lambda: self._wake_session(sid))
        try:
            await self.proxies.bind(sid, row["proxy_port"], app)
        except OSError as e:
            self.alerts.raise_("proxy_port", f"{row['colab_name']}: cannot listen on port {row['proxy_port']}: {e}",
                               sid)

    def _wake_session(self, sid: int, *, probe_health: bool = True) -> None:
        if probe_health:
            self._health_probe_pending.add(sid)
        self._wake.setdefault(sid, asyncio.Event()).set()

    async def _sleep_or_wake(self, sid: int, seconds: float) -> bool:
        """Sleep up to `seconds`; True if woken early by a decision, upstream error or observation."""
        ev = self._wake.setdefault(sid, asyncio.Event())
        try:
            await asyncio.wait_for(ev.wait(), seconds)
        except TimeoutError:
            return False
        ev.clear()
        return True

    def _sample_interval(self, sid: int) -> float:
        if self._observed_until.get(sid, 0) > self.now():
            return min(self.s.keepalive_s, OBSERVE_INTERVAL_S)
        return self.s.keepalive_s

    def _sample_due(self, sid: int) -> bool:
        return self.now() - self._last_exec.get(sid, 0) >= self._sample_interval(sid)

    def _sample_delay(self, sid: int) -> float:
        return max(0.05, self._sample_interval(sid) - (self.now() - self._last_exec.get(sid, 0)))

    async def observe_session(self, sid: int) -> bool:
        row = await self.store.session(sid)
        if row is None:
            return False
        if row["state"] in DRIVEN and row["endpoint"]:
            now = self.now()
            first = self._observed_until.get(sid, 0) <= now
            self._observed_until[sid] = now + OBSERVE_LEASE_S
            if first or self._sample_due(sid):
                self._wake_session(sid, probe_health=False)
        return True

    def resource_sample(self, sid: int) -> dict[str, Any] | None:
        return self._resource_samples.get(sid)

    # ----------------------------------------------------------------------------------------- proxy callbacks
    def route(self, sid: int) -> Route:
        v = self._view.get(sid, {})
        state, name, run_id = v.get("state"), v.get("colab_name"), v.get("run_id")
        if state == "ready":
            engine = self.engines.get(v.get("engine") or "")
            return Route("forward", v.get("hidden_port"), run_id=run_id,
                         inject_usage=bool(getattr(engine, "inject_usage_option", False)))
        if state == "engine_down":
            return Route("down", message=f"session {name}: the engine is down", run_id=run_id)
        detail = v.get("progress") or state
        return Route("unavailable", message=f"session {name} is {detail}", run_id=run_id)

    async def record(self, rec: RequestRecord, run_id: int) -> None:
        await self.store.insert_request(rec.as_row(run_id))

    # ---------------------------------------------------------------------------------------------- create
    def _run_spec(self, req: CreateRequest) -> RunSpec:
        if req.refusal_projection is not None and type(req.refusal_projection) is not bool:
            raise CreateRefused("refusal_projection must be a boolean or null")
        if req.preset_id:
            if isinstance(self.presets, PresetError):
                raise CreateRefused(f"presets.toml is invalid: {self.presets}")
            preset = self.presets.presets.get(req.preset_id)
            if not preset:
                raise CreateRefused(f"unknown preset {req.preset_id!r}")
            return RunSpec.from_preset(preset, req.gpu_class, self.presets.engines, ctx=req.ctx, kv=req.kv,
                                       parallel=req.parallel, extra_args=tuple(req.extra_args) or None,
                                       refusal_projection=req.refusal_projection)
        if req.engine not in self.engines:
            raise CreateRefused(f"unknown engine {req.engine!r}")
        if req.engine in ("llamacpp", "strata") and not (req.repo and req.file):
            raise CreateRefused(f"a custom {req.engine} model needs a Hugging Face repo and a GGUF file")
        if req.engine == "exllamav3" and not (req.repo and req.revision):
            raise CreateRefused("a custom ExLlamaV3 model needs a Hugging Face repo and a revision (e.g. 4.00bpw)")
        pins = {} if isinstance(self.presets, PresetError) else self.presets.engines.get(req.engine, {})
        release = pins.get("release") or (f"{pins['wheel']}@{pins['tabbyapi_commit']}"
                                           if pins.get("wheel") and pins.get("tabbyapi_commit") else None)
        default_kv = {"llamacpp": "f16", "strata": "int8"}.get(req.engine, "FP16")
        return RunSpec(engine=req.engine, repo=req.repo, file=req.file, revision=req.revision,
                       alias=req.alias or (req.repo.split("/")[-1] if req.engine == "strata" else req.file
                                           or req.repo.split("/")[-1]),
                       ctx=req.ctx if req.ctx is not None else (131072 if req.engine == "strata" else 32768),
                       kv=req.kv or default_kv, parallel=req.parallel if req.parallel is not None else 1,
                       vision=req.engine == "strata", refusal_projection=req.refusal_projection or False,
                       extra_args=tuple(req.extra_args), release=release)

    def _validate_spec(self, spec: RunSpec, gpu_class: str) -> None:
        if spec.engine not in self.engines:
            raise CreateRefused(f"engine {spec.engine!r} is not available")
        if spec.ctx <= 0 or spec.parallel <= 0:
            raise CreateRefused("context and parallel slots must be positive")
        if type(spec.refusal_projection) is not bool:
            raise CreateRefused("refusal_projection must be a boolean")
        if spec.engine == "exllamav3" and gpu_class == "CPU":
            raise CreateRefused("ExLlamaV3 requires a GPU runtime")
        if spec.engine != "strata":
            if spec.refusal_projection:
                raise CreateRefused("experimental refusal projection is supported only by Strata")
            return
        if gpu_class == "CPU":
            raise CreateRefused("Strata requires a GPU runtime")
        try:
            self.engines["strata"].validate_spec(spec)
        except ValueError as exc:
            raise CreateRefused(str(exc)) from exc

    SETTING_KEYS = {"daily_cap_cu": float, "low_balance_cu": float, "idle_min_default": int, "notify": bool,
                    "eligible_gpus": str}

    async def settings_values(self) -> dict[str, Any]:
        defaults = {"daily_cap_cu": self.s.daily_cap_cu, "low_balance_cu": self.s.low_balance_cu,
                    "idle_min_default": self.s.idle_min_default, "notify": self.s.notify,
                    "eligible_gpus": "G4,A100,L4,T4"}
        out: dict[str, Any] = {}
        for key, kind in self.SETTING_KEYS.items():
            raw = await self.store.get_setting(key)
            if raw is None:
                out[key] = defaults[key]
            elif kind is bool:
                out[key] = raw == "1"
            else:
                out[key] = kind(raw)
        return out

    async def update_settings(self, **values: Any) -> dict[str, Any]:
        validated: dict[str, str] = {}
        for key, value in values.items():
            kind = self.SETTING_KEYS.get(key)
            if kind is None or value is None:
                continue
            if kind is bool:
                raw = "1" if value in (True, "1", "true", "on") else "0"
            else:
                parsed = kind(value)
                if kind in (int, float):
                    if not math.isfinite(parsed):
                        raise CreateRefused(f"{key} must be finite")
                    if parsed < 0:
                        raise CreateRefused(f"{key} can't be negative")
                raw = str(parsed)
            validated[key] = raw
        await self.store.set_settings(validated)
        current = await self.settings_values()
        self.alerts.notify_enabled = current["notify"]
        return current

    async def spend_today(self, now: float | None = None) -> float:
        now = now if now is not None else self.now()
        start, end = self.clock.day_bounds(now)
        rows = await self.store.sessions("created_at < ? AND (stopped_at IS NULL OR stopped_at > ?)", (end, start))
        return billing.spend_between(rows, start, end, now)

    async def cap_today(self, now: float | None = None) -> float:
        now = now if now is not None else self.now()
        base = float(await self.store.get_setting("daily_cap_cu", str(self.s.daily_cap_cu)))
        extra = float(await self.store.get_setting(f"daily_extra:{self.clock.day_key(now)}", "0"))
        return base + extra if base > 0 else 0.0

    async def create(self, req: CreateRequest) -> int:
        if not req.confirm:
            raise CreateRefused("creating a session spends compute units: pass confirm=true")
        if req.gpu_class not in GPU_CLASSES:
            raise CreateRefused(f"unknown GPU class {req.gpu_class!r}")
        if req.vm_port == 8080 or not 1024 <= req.vm_port <= 65535:
            raise CreateRefused("VM port must be 1024-65535 and not 8080 (taken on Colab VMs)")
        spec = self._run_spec(req)
        self._validate_spec(spec, req.gpu_class)
        cap = await self.cap_today()
        if cap > 0 and await self.spend_today() >= cap:
            raise CreateRefused(f"today's spend reached the daily cap ({cap:.2f} CU); raise it for today to continue")
        async with self._create_lock:
            if self.account_id is None or not self.cli.enabled or self.account_change_pending:
                raise CreateRefused("Finish Google sign-in or select an account in Settings before creating a session.")
            listing = await self.fresh_listing()
            if not listing.trusted:
                raise CreateRefused("could not get a trusted `colab sessions` + `colab usage` listing; try again")
            if listing.named(req.name):
                raise CreateRefused(f"a runtime named {req.name!r} is already listed; `colab new` would orphan it")
            if await self.store.name_in_use(req.name):
                raise CreateRefused(f"name {req.name!r} is in use by another session")
            live = await self.store.sessions(OPEN_ROWS)
            taken = {r["proxy_port"] for r in live} | self.proxies.ports()
            if req.proxy_port:
                if req.proxy_port in taken or not port_free(req.proxy_port):
                    raise CreateRefused(f"proxy port {req.proxy_port} is not free")
                proxy_port = req.proxy_port
            else:
                proxy_port = first_free(self.s.proxy_port_start, self.s.proxy_port_start + 100, taken)
            hidden = first_free(*self.s.hidden_port_range, {r["hidden_port"] for r in live})
            rate, source = await self.store.rate(req.gpu_class)
            now = self.now()
            idle_default = (await self.settings_values())["idle_min_default"]
            sid = await self.store.insert_session(
                account_id=self.account_id,
                colab_name=req.name, gpu_requested=req.gpu_class, gpu_class=req.gpu_class, proxy_port=proxy_port,
                hidden_port=hidden, vm_port=req.vm_port, state="creating", state_since=now, created_at=now,
                rate_cu_h=rate, rate_source=source, assignments_before=listing.usage.active_assignments,
                usage_rate_before=listing.usage.rate_per_hour, name_reserved=1,
                idle_min=idle_default if req.idle_min is None else req.idle_min,
                deadline=req.deadline, cu_cap=req.cu_cap,
            )
            run_id = await self.store.insert_run(
                session_id=sid, engine=spec.engine, preset_id=spec.preset_id, repo=spec.repo,
                file_or_rev=spec.file or spec.revision, mmproj=spec.mmproj, alias=spec.alias, ctx=spec.ctx,
                kv=spec.kv, parallel=spec.parallel, extra_args=json.dumps(list(spec.extra_args)), started_at=now,
                status="pending", spec_json=json.dumps(dataclasses.asdict(spec)),
            )
            await self._event(sid, "created", gpu=req.gpu_class, rate=rate, name=req.name)
            row = await self.store.session(sid)
            self._view[sid] = {"state": "creating", "hidden_port": hidden, "colab_name": req.name, "run_id": run_id}
            await self._bind_proxy(row)
            try:
                proc = await self.cli.spawn_new(req.name, req.gpu_class)
            except OSError as exc:
                message = bounded_diagnostic(f"Could not launch colab new: {exc}")
                await self._update(sid, name_reserved=0, create_log=message)
                await self._fail_create(row, message, listing)
                return sid
            self._new_pids[sid] = proc.pid
            await self._update(sid, new_pid=proc.pid)
            self._new[sid] = asyncio.create_task(self._run_new(sid, proc))
        self._start_driver(sid)
        return sid

    async def _run_new(self, sid: int, proc: asyncio.subprocess.Process) -> tuple[int, str]:
        out, _ = await proc.communicate()
        self._new_pids.pop(sid, None)
        output = bounded_diagnostic(out.decode(errors="replace"))
        await self._update(sid, new_pid=None, new_exit=proc.returncode, name_reserved=0, create_log=output)
        return proc.returncode, output

    async def abandon(self, sid: int) -> None:
        row = await self.store.session(sid)
        if not row or row["state"] != "creating":
            raise CreateRefused("only a session that is still creating can be abandoned")
        await self._fail_create(row, "abandoned by the user", await self.fresh_listing())

    async def _fail_create(self, row: dict[str, Any], message: str, listing: Listing) -> None:
        sid = row["id"]
        message = bounded_diagnostic(message, 2000)
        await self._update(sid, create_error=message)
        run = await self.store.current_run(sid)
        if run and run["status"] == "pending":
            await self.store.update_run(run["id"], status="failed", error=message)
        if listing.trusted and listing.usage and listing.usage.active_assignments > (row["assignments_before"] or 0):
            self.alerts.raise_("create_orphan", f"{row['colab_name']}: a VM may have been created without a name - "
                               "stop it in the Colab web UI", sid)
        await self._finish(sid, "failed", None, vm_created=False, message=message)

    # ---------------------------------------------------------------------------------------------- driver
    async def _drive(self, sid: int) -> None:
        handlers = {"creating": self._creating, "gpu_check": self._gpu_check, "gpu_mismatch": self._waiting,
                    "setup_failed": self._waiting, "setting_up": self._setting_up, "reconfiguring": self._setting_up,
                    "tunnelling": self._tunnelling, "ready": self._ready, "engine_down": self._ready}
        while True:
            stop_if_cancelled()
            row = await self.store.session(sid)
            if row is None or row["state"] not in DRIVEN:
                return
            try:
                nxt = await handlers[row["state"]](row)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("session %s: driver error in %s", sid, row["state"])
                await asyncio.sleep(self.s.setup_poll_s)
                continue
            if nxt is None:
                return
            current = await self.store.session(sid)
            if current is None or current["state"] not in DRIVEN:
                return
            if nxt != current["state"]:
                await self._transition(sid, nxt)

    async def _creating(self, row: dict[str, Any]) -> str | None:
        sid, name = row["id"], row["colab_name"]
        task = self._new.get(sid)
        if task is None:
            return None  # a create from before a restart: the reconcile loop resolves it
        endpoint = None
        while not task.done():
            stop_if_cancelled()
            await asyncio.wait({task}, timeout=self.s.setup_poll_s)
            if task.done():
                break
            cur = await self.store.session(sid)
            if cur["state"] != "creating":
                return None  # abandoned or stopped while `colab new` runs; _run_new still clears the name
            elapsed = self.now() - row["created_at"]
            if elapsed >= self.s.create_slow_s:
                extra = " - you can Abandon it" if elapsed >= self.s.create_abandon_s else ""
                self.alerts.raise_("create_slow", f"{name}: create is taking long ({elapsed / 60:.0f} min){extra}",
                                   sid, actions=["abandon"] if extra else [])
            listing = await self.fresh_listing()
            rt = listing.named(name) if listing.trusted else None
            if rt:  # registered: safe to end `colab new` now
                endpoint = rt.endpoint
                pid = self._new_pids.get(sid)
                if pid:
                    with contextlib.suppress(OSError):
                        os.killpg(pid, signal.SIGTERM)
                await task
                break
        rc, out = task.result()
        self._new.pop(sid, None)
        self.alerts.clear(f"create_slow:{sid}")
        cur = await self.store.session(sid)
        if cur["state"] != "creating":
            return None
        if sid in self._pending_stop:
            endpoint = endpoint or await self._wait_endpoint(sid, name, self.s.setup_poll_s * 3)
        if endpoint is None and rc == 0:
            endpoint = await self._wait_endpoint(sid, name, self.s.endpoint_wait_s)
            if endpoint is None:
                self.alerts.raise_("endpoint_missing", f"{name}: VM created but not listed - check `colab sessions`"
                                   " or the Colab web UI", sid)
                return None
        elif endpoint is None:
            listing = await self.fresh_listing()
            rt = listing.named(name) if listing.trusted else None
            if not rt:
                await self._fail_create(cur, out.strip() or f"colab new exited {rc} without any output", listing)
                return None
            endpoint = rt.endpoint
        await self._update(sid, endpoint=endpoint)
        await self._event(sid, "vm_created", endpoint=endpoint)
        if sid in self._pending_stop:
            await self.stop(sid, self._pending_stop.pop(sid))
            return None
        return "gpu_check"

    async def _wait_endpoint(self, sid: int, name: str, limit: float) -> str | None:
        deadline = self.now() + limit
        while True:
            stop_if_cancelled()
            listing = await self.fresh_listing()
            rt = listing.named(name) if listing.trusted else None
            if rt:
                return rt.endpoint
            if self.now() >= deadline:
                return None
            await asyncio.sleep(self.s.setup_poll_s)

    async def _status(self, row: dict[str, Any], **env: Any) -> dict | None:
        sid = row["id"]
        self._last_exec[sid] = self.now()
        if self._observed_until.get(sid, 0) > self.now():
            listing = await self.fresh_listing()
            runtime = listing.named(row["colab_name"]) if listing.trusted else None
            if runtime is None or runtime.endpoint != row["endpoint"]:
                return None
        # `colab exec --env` sets os.environ inside the persistent kernel, so every key status.py reads is passed
        # on every call (otherwise e.g. WITH_TORCH=1 from the GPU check would stick to every keep-alive).
        run = await self.store.current_run(sid)
        full = {"VM_PORT": row["vm_port"], "EVENTS_OFFSET": row["event_offset"], "LOG_OFFSET": row["log_offset"],
                "ENGINE_LOG": f"{VM_ROOT}/engine.log", "EVENTS": f"{VM_ROOT}/setup.jsonl", "WITH_TORCH": "0", **env,
                "ENGINE_KIND": run["engine"] if run else "", "ENGINE_MODEL": run["alias"] or "" if run else ""}
        res = await self.cli.exec_file(row["colab_name"], VM_DIR / "status.py", {k: str(v) for k, v in full.items()})
        if res.status is None:
            return None
        await self._update(sid, last_exec_ok=self.now(), log_offset=res.status.get("log_offset", 0))
        buf = self._logs.setdefault(sid, deque(maxlen=500))
        lines = res.status.get("log_lines") or []
        buf.extend(lines)
        self._log_counts[sid] = self._log_counts.get(sid, 0) + len(lines)
        marker = res.status.get("marker") or {}
        resources = res.status.get("resources")
        if (isinstance(resources, dict) and marker.get("session_id") == sid
                and marker.get("endpoint") == row["endpoint"]):
            self._resource_samples[sid] = {"sampled_at": self.now(), **resources}
        return res.status

    async def _prepare_vm(self, row: dict[str, Any]) -> bool:
        """Create /content/colabinet (+ templates/) and write the session marker, before any upload: `colab upload`
        fails with HTTP 500 when the parent directory is missing. bg.py and status.py are never uploaded -
        `colab exec -f` sends the local file's text."""
        marker = json.dumps({"session_id": row["id"], "endpoint": row["endpoint"]})
        res = await self.cli.exec_file(row["colab_name"], VM_DIR / "prepare.py", {"MARKER": marker})
        return bool(res.status and res.status.get("prepared"))

    async def _gpu_check(self, row: dict[str, Any]) -> str:
        sid = row["id"]
        if self.now() - row["state_since"] > self.s.gpu_check_limit_s:
            return await self._setup_failed(row, "the GPU check did not complete in time")
        if not await self._prepare_vm(row):
            await asyncio.sleep(self.s.setup_poll_s)
            return "gpu_check"
        st = await self._status(row, WITH_TORCH="1")
        if st is None:
            await asyncio.sleep(self.s.setup_poll_s)
            return "gpu_check"
        gpu = st.get("gpu") or {}
        cls = detected_class(gpu.get("name"), gpu.get("vram_mib"), st.get("ram_gb"))
        await self._update(sid, gpu_name=gpu.get("name"), vram_mib=gpu.get("vram_mib"), ram_gb=st.get("ram_gb"),
                           driver=gpu.get("driver"), torch_cuda=st.get("torch_cuda"), gpu_class=cls)
        await self._settle_rate(await self.store.session(sid), cls)
        run = await self.store.current_run(sid)
        if cls == "CPU" and run and run["engine"] == "strata":
            return await self._setup_failed(row, "Strata requires a GPU runtime; the detected runtime is CPU")
        if cls != row["gpu_requested"]:
            self.alerts.raise_("gpu_mismatch", f"{row['colab_name']}: asked for {row['gpu_requested']}, got {cls} "
                               f"({gpu.get('name') or 'no GPU'}) - Continue or Stop", sid,
                               actions=["gpu-continue", "stop"])
            return "gpu_mismatch"
        return "setting_up"

    async def _settle_rate(self, row: dict[str, Any], cls: str) -> None:
        sid = row["id"]
        if cls not in GPU_CLASSES:
            return
        table, source = await self.store.rate(cls)
        rate, used = table, source
        listing = await self.fresh_listing()
        others = await self.store.db.fetchone(
            "SELECT COUNT(*) AS n FROM events WHERE kind IN ('state', 'created') AND session_id != ? AND ts >= ?",
            (sid, row["created_at"]))
        if listing.trusted and listing.usage and row["usage_rate_before"] is not None and others["n"] == 0:
            delta = listing.usage.rate_per_hour - row["usage_rate_before"]
            if abs(delta - table) <= 0.25 * table:
                rate, used = round(delta, 2), "usage_delta"
                await self.store.set_rate(cls, rate, "learned", self.now())
        await self._update(sid, rate_cu_h=rate, rate_source=used)

    async def _waiting(self, row: dict[str, Any]) -> str | None:
        """gpu_mismatch / setup_failed: keep the VM alive until the user decides; the guard loop enforces grace."""
        sid = row["id"]
        woke = await self._sleep_or_wake(sid, self._sample_delay(sid))
        decision = self._decision.pop(sid, None) if woke else None
        self._health_probe_pending.discard(sid)
        if decision == "continue" and row["state"] == "gpu_mismatch":
            self.alerts.clear(f"gpu_mismatch:{sid}")
            return "setting_up"
        if decision == "retry" and row["state"] == "setup_failed":
            self.alerts.clear(f"setup_failed:{sid}")
            run = await self.store.current_run(sid)
            await self.store.update_run(run["id"], status="pending", error=None)
            return "setting_up"
        if self._sample_due(sid):
            await self._status(row)
        return row["state"]

    async def _setup_failed(self, row: dict[str, Any], message: str) -> str:
        sid = row["id"]
        run = await self.store.current_run(sid)
        if run:
            await self.store.update_run(run["id"], status="failed", error=message[:1000])
        self.alerts.raise_("setup_failed", f"{row['colab_name']}: setup failed: {message[:200]}", sid,
                           actions=["retry-setup", "stop"])
        return "setup_failed"

    async def _setting_up(self, row: dict[str, Any]) -> str:
        sid, name = row["id"], row["colab_name"]
        run = await self.store.current_run(sid)
        self._view.setdefault(sid, {}).update(run_id=run["id"], engine=run["engine"])
        spec = RunSpec(**{**json.loads(run["spec_json"]), "extra_args": tuple(json.loads(run["spec_json"])
                                                                                 ["extra_args"])})
        try:
            self._validate_spec(spec, row["gpu_class"] or row["gpu_requested"])
        except CreateRefused as exc:
            return await self._setup_failed(row, str(exc))
        engine = self.engines[spec.engine]
        env = engine.setup_env(spec, row["vm_port"], run["id"])
        if self.hf_token:
            env["HF_TOKEN"] = self.hf_token
        if run["status"] != "started":
            for local, remote in engine.uploads(spec):
                if not (await self.cli.upload(name, local, remote)).ok:
                    return await self._setup_failed(row, f"upload of {local.name} failed")
            await self._update(sid, event_offset=0)
            row = await self.store.session(sid)
            started = await self.cli.exec_file(name, VM_DIR / "bg.py", env)
            if not started.ok:  # the exec may have dropped after starting the job: check before retrying
                await asyncio.sleep(self.s.setup_poll_s)
                probe = await self._status(row, EVENTS=env["EVENTS"])
                if not (probe and probe.get("events")):
                    started = await self.cli.exec_file(name, VM_DIR / "bg.py", env)
                    if not started.ok:
                        return await self._setup_failed(row, f"could not start setup: {started.out[-200:]}")
            await self.store.update_run(run["id"], status="started")
        t0 = self.now()
        while True:
            stop_if_cancelled()
            await asyncio.sleep(self.s.setup_poll_s)
            row = await self.store.session(sid)
            if row["state"] not in ("setting_up", "reconfiguring"):
                return row["state"]
            st = await self._status(row, EVENTS=env["EVENTS"])
            if st:
                await self._update(sid, event_offset=st.get("events_offset", row["event_offset"]))
                for ev in st.get("events", []):
                    nxt = await self._setup_event(row, run["id"], ev)
                    if nxt:
                        return nxt
            if self.now() - max(t0, row["state_since"]) > self.s.setup_limit_s:
                return await self._setup_failed(row, "setup took longer than the setup limit")

    async def _setup_event(self, row: dict[str, Any], run_id: int, ev: dict[str, Any]) -> str | None:
        sid, kind = row["id"], ev.get("event")
        await self._event(sid, f"vm:{kind}", **{k: v for k, v in ev.items() if k not in ("event", "log_tail")})
        run = await self.store.current_run(sid)
        if kind == "install_done":
            await self.store.update_run(run_id, install_s=ev.get("seconds"), engine_version=ev.get("release"))
            self._view[sid]["progress"] = "downloading the model"
        elif kind == "download":
            self._view[sid]["progress"] = (f"downloading {ev.get('file')} ({ev.get('done_gb')} GB, "
                                           f"{ev.get('mb_s')} MB/s)")
        elif kind == "download_done":
            await self.store.update_run(run_id, download_s=(run["download_s"] or 0) + (ev.get("seconds") or 0),
                                        download_gb=(run["download_gb"] or 0) + (ev.get("gb") or 0))
            self._view[sid]["progress"] = "loading the model"
        elif kind == "server_ready":
            await self.store.update_run(run_id, load_s=ev.get("seconds"), vram_used_mib=ev.get("vram_used_mib"),
                                        status="ready")
            self._view[sid]["progress"] = "connecting the tunnel"
            return "tunnelling"
        elif kind == "error":
            return await self._setup_failed(row, ev.get("message") or "setup error")
        return None

    async def _safe_to_tunnel(self, row: dict[str, Any]) -> bool:
        """Safe tunnel rule: allowed state + a fresh trusted listing showing NAME with our endpoint."""
        if row["state"] not in ("tunnelling", "ready", "engine_down") or not row["endpoint"]:
            return False
        listing = await self.fresh_listing()
        rt = listing.named(row["colab_name"]) if listing.trusted else None
        return bool(rt and rt.endpoint == row["endpoint"])

    async def _ensure_tunnel(self, row: dict[str, Any]) -> bool:
        sid, name = row["id"], row["colab_name"]
        if self.tunnels.alive(row["tunnel_pid"], name):
            return True
        if row["tunnel_pid"]:
            err = self.tunnels.last_error(name) or ""
            if "429" in err:
                self.alerts.raise_("ssh_429", f"{name}: Colab allows one SSH connection per runtime - close any other "
                                   f"`colab ssh -s {name}` (e.g. a terminal) so colabinet's tunnel can connect", sid)
        if not await self._safe_to_tunnel(row):
            return False
        await self.tunnels.stop(row["tunnel_pid"], name)
        pid = await self.tunnels.start(name, row["hidden_port"], row["vm_port"])
        await self._update(sid, tunnel_pid=pid)
        await self._event(sid, "tunnel_started", pid=pid)
        return True

    async def _healthy(self, row: dict[str, Any]) -> bool:
        try:
            r = await self.http.get(f"http://127.0.0.1:{row['hidden_port']}/health", timeout=5)
            if r.status_code != 200:
                return False
            run = await self.store.current_run(row["id"])
            if run and run["engine"] == "strata":
                # The Python frontend stays healthy after Strata's native engine dies; only loaded models count.
                models = await self.http.get(f"http://127.0.0.1:{row['hidden_port']}/v1/models", timeout=5)
                if models.status_code != 200:
                    return False
                data = models.json().get("data")
                return isinstance(data, list) and any(isinstance(model, dict) and model.get("id")
                    and (not run["alias"] or model["id"] == run["alias"]) for model in data)
            return True
        except (httpx.HTTPError, ValueError, AttributeError):
            return False

    async def _tunnelling(self, row: dict[str, Any]) -> str:
        sid = row["id"]
        if await self._ensure_tunnel(row) and await self._healthy(await self.store.session(sid)):
            now = self.now()
            run = await self.store.current_run(sid)
            await self.store.update_run(run["id"], ready_at=now)
            await self._update(sid, ready_at=now)
            self.alerts.clear(f"tunnel_slow:{sid}")
            self.alerts.clear(f"ssh_429:{sid}")
            return "ready"
        if self.now() - row["state_since"] > self.s.tunnel_alert_s:
            self.alerts.raise_("tunnel_slow", f"{row['colab_name']}: the tunnel is not connecting", sid)
        await asyncio.sleep(min(self.s.setup_poll_s, self.s.keepalive_s))
        if self._sample_due(sid):
            await self._status(row)
        return "tunnelling"

    async def _ready(self, row: dict[str, Any]) -> str | None:
        sid, name = row["id"], row["colab_name"]
        run = await self.store.current_run(sid)
        self._view.setdefault(sid, {}).update(run_id=run["id"], engine=run["engine"])
        await self._sleep_or_wake(sid, self._sample_delay(sid))
        row = await self.store.session(sid)
        probe_health = sid in self._health_probe_pending
        self._health_probe_pending.discard(sid)
        if row["state"] not in ("ready", "engine_down"):
            return row["state"]
        st = await self._status(row) if probe_health or self._sample_due(sid) else None
        if st is not None:
            marker = st.get("marker") or {}
            if marker.get("session_id") != sid:
                self.alerts.raise_("foreign_vm", f"{name}: the VM answering to this name is not ours - stopping it",
                                   sid)
                await self.stop(sid, "foreign_vm", foreign=True)
                return None
            if st.get("engine_healthy"):
                self._unhealthy[sid] = 0
                if row["state"] == "engine_down":
                    self.alerts.clear(f"engine_down:{sid}")
                    return "ready"
            else:
                self._unhealthy[sid] = self._unhealthy.get(sid, 0) + 1
                if self._unhealthy[sid] >= 2 and row["state"] == "ready":
                    self.alerts.raise_("engine_down", f"{name}: the engine stopped answering", sid,
                                       actions=["retry-setup", "stop"])
                    return "engine_down"
        if not self.tunnels.alive(row["tunnel_pid"], name) or not await self._healthy(row):
            await self._ensure_tunnel(row)
        else:
            self.alerts.clear(f"ssh_429:{sid}")
        return row["state"]

    # ---------------------------------------------------------------------------------------- user actions
    async def keep(self, sid: int) -> None:
        self._own_account(await self.store.session(sid))
        now = self.now()
        self._activity.setdefault(sid, Activity()).reset(now)
        await self._update(sid, last_activity=now)
        self.alerts.clear(f"idle_warn:{sid}")

    async def decide(self, sid: int, decision: str) -> None:
        async with self._session_locks.setdefault(sid, asyncio.Lock()):
            await self._decide_locked(sid, decision)

    async def _decide_locked(self, sid: int, decision: str) -> None:
        row = await self.store.session(sid)
        self._own_account(row)
        expected = {"continue": "gpu_mismatch", "retry": ("setup_failed", "engine_down")}[decision]
        if not row or row["state"] not in (expected if isinstance(expected, tuple) else (expected,)):
            raise CreateRefused(f"session is {row['state'] if row else 'missing'}; nothing to {decision}")
        run = await self.store.current_run(sid)
        old = json.loads(run["spec_json"])
        self._validate_spec(RunSpec(**{**old, "extra_args": tuple(old["extra_args"])}),
                            row["gpu_class"] or row["gpu_requested"])
        if row["state"] == "engine_down":
            run = await self.store.current_run(sid)
            await self.store.update_run(run["id"], status="pending")
            await self._transition(sid, "setting_up")
            await self._cancel_driver(sid)
            self._start_driver(sid)
            return
        self._decision[sid] = decision
        self._wake_session(sid)

    async def reconfigure(self, sid: int, *, preset_id: str | None = None, ctx: int | None = None,
                          kv: str | None = None, extra_args: tuple[str, ...] | None = None,
                          refusal_projection: bool | None = None) -> int:
        """Same VM, new run (spec 5.2 step 7): another model, context, KV type or extra args. Installs and downloads
        that are already on the VM are skipped; the proxy answers 503 until the new run is ready."""
        async with self._session_locks.setdefault(sid, asyncio.Lock()):
            return await self._reconfigure_locked(sid, preset_id=preset_id, ctx=ctx, kv=kv, extra_args=extra_args,
                                                   refusal_projection=refusal_projection)

    async def _reconfigure_locked(self, sid: int, *, preset_id: str | None, ctx: int | None,
                                  kv: str | None, extra_args: tuple[str, ...] | None,
                                  refusal_projection: bool | None) -> int:
        if refusal_projection is not None and type(refusal_projection) is not bool:
            raise CreateRefused("refusal_projection must be a boolean or null")
        row = await self.store.session(sid)
        self._own_account(row)
        if not row or row["state"] not in ("ready", "engine_down", "setup_failed"):
            raise CreateRefused(f"only a ready, engine-down or setup-failed session can be reconfigured "
                                f"(this one is {row['state'] if row else 'missing'})")
        run = await self.store.current_run(sid)
        old = json.loads(run["spec_json"])
        old["extra_args"] = tuple(old["extra_args"])
        overrides = {"ctx": ctx, "kv": kv, "extra_args": tuple(extra_args) if extra_args is not None else None,
                     "refusal_projection": refusal_projection}
        if preset_id and preset_id != old.get("preset_id"):
            if isinstance(self.presets, PresetError):
                raise CreateRefused(f"presets.toml is invalid: {self.presets}")
            preset = self.presets.presets.get(preset_id)
            if not preset:
                raise CreateRefused(f"unknown preset {preset_id!r}")
            if overrides["extra_args"] == ():  # an empty form field keeps the new preset's own engine args
                overrides["extra_args"] = None
            spec = RunSpec.from_preset(preset, row["gpu_class"] or row["gpu_requested"], self.presets.engines,
                                       **overrides)
        else:
            spec = RunSpec(**{**old, **{k: v for k, v in overrides.items() if v is not None}})
        self._validate_spec(spec, row["gpu_class"] or row["gpu_requested"])
        await self._cancel_driver(sid)
        now = self.now()
        run_id = await self.store.replace_run(
            sid, now, engine=spec.engine, preset_id=spec.preset_id, repo=spec.repo,
            file_or_rev=spec.file or spec.revision, mmproj=spec.mmproj, alias=spec.alias, ctx=spec.ctx, kv=spec.kv,
            parallel=spec.parallel,
            extra_args=json.dumps(list(spec.extra_args)), started_at=now, status="pending",
            spec_json=json.dumps(dataclasses.asdict(spec)),
        )
        await self._event(sid, "reconfigure", run_id=run_id, ctx=spec.ctx, kv=spec.kv, model=spec.alias)
        self._view.setdefault(sid, {}).update(run_id=run_id, progress="reconfiguring")
        self.alerts.clear_session(sid, {"engine_down", "setup_failed"})
        await self._transition(sid, "reconfiguring")
        self._start_driver(sid)
        return run_id

    async def extend_daily_cap(self, cu: float = 5.0) -> float:
        key = f"daily_extra:{self.clock.day_key(self.now())}"
        extra = float(await self.store.get_setting(key, "0")) + cu
        await self.store.set_setting(key, str(extra))
        self.alerts.clear("daily_cap")
        return await self.cap_today()

    async def stop_external(self, name: str, expected_account: str | None = None) -> None:
        async with self._create_lock:
            if expected_account is not None and expected_account != self.account_id:
                raise CreateRefused("The active account changed. Refresh Overview before stopping this runtime.")
            if self.account_id is None or self.account_change_pending:
                raise CreateRefused("Finish the account change in Settings first.")
            if await self.store.name_in_use(name):
                raise CreateRefused(f"{name!r} is a managed session: stop it by id")
            listing = await self.fresh_listing()
            if not (listing.trusted and listing.named(name)):
                raise CreateRefused(f"no listed runtime named {name!r}")
            await self.cli.stop(name)
            await self._event(None, "external_stopped", name=name, account_id=self.account_id)

    # ------------------------------------------------------------------------------------------------ stop
    async def stop(self, sid: int, reason: str = "user", foreign: bool = False) -> None:
        async with self._session_locks.setdefault(sid, asyncio.Lock()):
            await self._stop_locked(sid, reason, foreign)

    async def _stop_locked(self, sid: int, reason: str, foreign: bool) -> None:
        row = await self.store.session(sid)
        self._own_account(row)
        if row is None or row["state"] in ("stopped", "failed", "stopping"):
            return
        if not self.cli.enabled:
            raise CreateRefused("Sign in again to this session's Google account in Settings before stopping it.")
        if row["state"] == "creating":
            self._pending_stop[sid] = reason  # can't stop by name until the VM is listed (safe stop rule)
            await self._event(sid, "stop_pending", reason=reason)
            return
        await self._transition(sid, "stopping", stop_reason=reason)
        await self._cancel_driver(sid)
        await self._stop_flow(sid, foreign)

    async def _stop_flow(self, sid: int, foreign: bool = False) -> None:
        row = await self.store.session(sid)
        name, endpoint = row["colab_name"], row["endpoint"]
        await self.tunnels.stop(row["tunnel_pid"], name)
        await self._update(sid, tunnel_pid=None)
        confirmed = False
        for attempt in range(3):
            listing = await self.fresh_listing()
            if listing.trusted:
                rt = listing.named(name)
                if rt and (rt.endpoint == endpoint or foreign):
                    await self.cli.stop(name)  # safe stop rule satisfied
                after = await self.fresh_listing()
                if after.trusted and not any(r.endpoint == endpoint for r in after.runtimes) and \
                        (not foreign or after.named(name) is None):
                    confirmed = True
                    break
            await asyncio.sleep(self.s.stop_retry_s)
        if confirmed:
            await self._finish(sid, "stopped", row["stop_reason"], vm_created=True)
            return
        await self._update(sid, state="stopped", state_since=self.now(), stop_unconfirmed=1)
        await self._event(sid, "stop_unconfirmed")
        await self.proxies.unbind(sid)
        self.alerts.raise_("stop_unconfirmed", f"{name}: stop not confirmed - may still be billing. Try "
                           f"`colab stop -s {name}`; colabinet keeps retrying", sid)

    async def _retry_unconfirmed(self, row: dict[str, Any], listing: Listing) -> None:
        sid, name, endpoint = row["id"], row["colab_name"], row["endpoint"]
        ours = next((r for r in listing.runtimes if r.endpoint == endpoint), None)
        if ours is None:
            await self._close(sid, row, row["stop_reason"], vm_created=True)
            await self._update(sid, stop_unconfirmed=0)
            self.alerts.clear(f"stop_unconfirmed:{sid}")
            await self._event(sid, "stop_confirmed")
        elif ours.name == name:
            await self.cli.stop(name)
        else:  # our VM lost its name ("[?]"): a by-name stop can't reach it any more
            self.alerts.raise_("stop_unconfirmed", f"{name}: still billing but no longer stoppable by name - stop it "
                               "in the Colab web UI", sid)
            if self.now() - (row["last_exec_ok"] or row["created_at"]) > billing.RECLAIM_S:
                await self._close(sid, row, "unreachable", vm_created=True)
                await self._update(sid, stop_unconfirmed=0)

    async def _close(self, sid: int, row: dict[str, Any], reason: str | None, vm_created: bool) -> float:
        t = billing.close_time(now=self.now(), created_at=row["created_at"], vm_created=vm_created, reason=reason,
                               last_exec_ok=row["last_exec_ok"])
        await self._update(sid, stopped_at=t)
        await self.store.close_open_runs(sid, t)
        return t

    async def _finish(self, sid: int, state: str, reason: str | None, vm_created: bool, message: str = "") -> None:
        row = await self.store.session(sid)
        await self._close(sid, row, reason, vm_created)
        await self._transition(sid, state, stop_reason=reason)
        if message:
            await self._event(sid, "finish_message", message=message[:500])
        await self._cancel_driver(sid)
        await self.tunnels.stop(row["tunnel_pid"], row["colab_name"])
        await self.proxies.unbind(sid)
        self.alerts.clear_session(sid, {"idle_warn", "create_slow", "engine_down", "gpu_mismatch", "setup_failed",
                                        "tunnel_slow", "endpoint_missing"})
        self._activity.pop(sid, None)

    # ---------------------------------------------------------------------------------------------- loops
    async def reconcile_once(self) -> None:
        async with self._create_lock:
            await self._reconcile_account()

    async def _reconcile_account(self) -> None:
        await self.resume_account_sessions()
        cli = self.cli
        listing = await self.fresh_listing()
        if not listing.trusted or cli is not self.cli:
            return
        rows = await self.store.sessions(OPEN_ROWS)
        known = {r["endpoint"] for r in rows if r["endpoint"]}
        for row in rows:
            if row["account_id"] != self.account_id:
                continue
            sid, name, state = row["id"], row["colab_name"], row["state"]
            if row["name_reserved"] and sid not in self._new and row["new_pid"]:
                line = cmdline(row["new_pid"])
                if not (line and f"new -s {name}" in line):
                    await self._update(sid, name_reserved=0, new_pid=None)
            if row["stop_unconfirmed"]:
                await self._retry_unconfirmed(row, listing)
                continue
            if state in ("stopping", "stopped", "failed"):
                continue
            if state == "creating":
                if sid in self._new or (row["new_pid"] and cmdline(row["new_pid"])):
                    continue
                rt = listing.named(name)
                if rt and rt.endpoint not in known - {row["endpoint"]}:
                    await self._update(sid, endpoint=rt.endpoint, name_reserved=0)
                    if sid in self._pending_stop:
                        await self._transition(sid, "gpu_check")
                        await self.stop(sid, self._pending_stop.pop(sid))
                    else:
                        await self._transition(sid, "gpu_check")
                        self._start_driver(sid)
                elif row["new_exit"] == 0:
                    self.alerts.raise_("endpoint_missing", f"{name}: VM created but not listed", sid)
                else:
                    self._misses[sid] = self._misses.get(sid, 0) + 1
                    if self._misses[sid] >= 2:
                        await self._fail_create(row, "colab new did not complete (found after restart)", listing)
                continue
            named = listing.named(name)
            if named and row["endpoint"] and named.endpoint != row["endpoint"]:
                self.alerts.raise_("foreign_vm", f"{name}: another VM now uses this name - stopping it", sid)
                await self.cli.stop(name)  # the VM under our NAME is not ours (e.g. auto-created); stop it
                await self._event(sid, "foreign_vm_stopped", endpoint=named.endpoint)
            if any(r.endpoint == row["endpoint"] for r in listing.runtimes):
                self._misses[sid] = 0
                continue
            self._misses[sid] = self._misses.get(sid, 0) + 1
            if self._misses[sid] >= 2:
                await self._finish(sid, "stopped", "reclaimed", vm_created=True)
        seen = set()
        for rt in listing.runtimes:
            if rt.endpoint in known:
                continue
            seen.add(rt.endpoint)
            if rt.name:
                self.alerts.raise_("external_runtime", f"runtime {rt.name!r} ({rt.hardware}) is billing but not "
                                   "managed by colabinet", key=f"external:{rt.endpoint}", actions=["stop-external"],
                                   target=rt.name)
            else:
                self.alerts.raise_("unknown_runtime", f"an unnamed runtime {rt.endpoint} ({rt.hardware}) is billing - "
                                   "stop it in the Colab web UI", key=f"external:{rt.endpoint}")
        for alert in self.alerts.list():
            if alert.key.startswith("external:") and alert.key.split(":", 1)[1] not in seen:
                self.alerts.clear(alert.key)
        hidden = listing.usage.active_assignments - len(listing.runtimes) if listing.usage else 0
        if hidden > 0:
            self.alerts.raise_("unknown_runtime", f"{hidden} billing assignment(s) are not shown by `colab sessions` - "
                               "check the Colab web UI", key="assignments_hidden")
        else:
            self.alerts.clear("assignments_hidden")

    async def guards_once(self) -> None:
        if not self.cli.enabled:
            return
        now = self.now()
        rows = await self.store.live_sessions()
        for row in rows:
            sid = row["id"]
            act = self._activity.get(sid)
            last = act.effective(now) if act else (row["last_activity"] or 0.0)
            d = evaluate(row, now, last, self.s.grace_s)
            if d.action == "stop":
                await self._event(sid, "guard", reason=d.reason, message=d.message)
                await self.stop(sid, d.reason)
            elif d.action == "warn":
                self.alerts.raise_("idle_warn", f"{row['colab_name']}: {d.message}", sid, actions=["keep"])
            else:
                self.alerts.clear(f"idle_warn:{sid}")
            if act and act.last > (row["last_activity"] or 0):
                await self._update(sid, last_activity=act.last)
        daily = evaluate_daily(await self.spend_today(now), await self.cap_today(now))
        if daily.action == "none":
            self.alerts.clear("daily_cap")
            return
        self.alerts.raise_("daily_cap", daily.message, key="daily_cap", actions=["extend-daily-cap"])
        if daily.action == "stop":
            for row in rows:
                await self.stop(row["id"], "daily_cap")

    async def usage_once(self) -> None:
        async with self._listing_lock:
            usage = await self.cli.usage()
            if usage is None:
                return
            now = self.now()
            self.usage, self.usage_at = usage, now
            key = f"balance_start:{self.account_id}:{self.clock.day_key(now)}"
            if await self.store.get_setting(key) is None:
                await self.store.set_setting(key, str(usage.balance))
            threshold = float(await self.store.get_setting("low_balance_cu", str(self.s.low_balance_cu)))
            if usage.balance < threshold:
                self.alerts.raise_("low_balance", f"balance is {usage.balance:.2f} CU (below {threshold:.0f}); at 0 "
                                   "Colab applies free-tier rules and SSH stops working", key="low_balance")
            else:
                self.alerts.clear("low_balance")

    async def _loop(self, fn, every: float, name: str) -> None:
        while True:
            stop_if_cancelled()
            try:
                await fn()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("%s failed", name)
            await asyncio.sleep(every)

    # -------------------------------------------------------------------------------------------- lifecycle
    async def start(self) -> None:
        """Recovery (spec 5.5), then the background loops."""
        self.alerts.notify_enabled = (await self.settings_values())["notify"]
        self.cli_version = await self.cli.version()
        for row in await self.store.sessions(OPEN_ROWS):
            self._own_account(row)
            sid, name, state = row["id"], row["colab_name"], row["state"]
            if row["tunnel_pid"]:
                await self.tunnels.stop(row["tunnel_pid"], name)
                await self._update(sid, tunnel_pid=None)
            run = await self.store.current_run(sid)
            self._view[sid] = {"state": state, "hidden_port": row["hidden_port"], "colab_name": name,
                               "run_id": run["id"] if run else None, "engine": run["engine"] if run else None}
            if not self.cli.enabled:
                self._paused_for_account.add(sid)
                self.alerts.raise_("account_reconnect", "Google credentials are missing. Sessions may still bill. "
                                   "Use Settings → Google accounts → Sign in again for their owning account.",
                                   key="account_reconnect")
                continue
            if state == "stopping":
                asyncio.create_task(self._stop_flow(sid))
            elif state in LIVE_STATES:
                await self._bind_proxy(row)
                if state != "creating":
                    self._start_driver(sid)
        self._loops = [
            asyncio.create_task(self._loop(self.reconcile_once, self.s.reconcile_s, "reconcile")),
            asyncio.create_task(self._loop(self.guards_once, self.s.guard_s, "guards")),
            asyncio.create_task(self._loop(self.usage_once, self.s.usage_s, "usage")),
        ]

    async def shutdown(self, stop_all: bool = False) -> list[str]:
        for t in self._loops:
            t.cancel()
        await asyncio.gather(*self._loops, return_exceptions=True)
        live = await self.store.live_sessions()
        if stop_all:
            for row in live:
                if row["state"] != "creating":
                    await self.stop(row["id"], "user")
        for task in list(self._drivers.values()):
            task.cancel()
        for task in list(self._new.values()):
            task.cancel()  # the `colab new` process itself keeps running in its own process group
        await asyncio.gather(*self._drivers.values(), *self._new.values(), return_exceptions=True)
        await self.proxies.close()
        remaining = await self.store.live_sessions()
        return [f"{r['colab_name']} ({r['state']})" for r in remaining]

    # ---------------------------------------------------------------------------------------------- views
    async def state(self) -> dict[str, Any]:
        now = self.now()
        rows = await self.store.sessions(f"{OPEN_ROWS} OR stopped_at > ?", (now - 86400,))
        sessions = []
        for r in rows:
            run = await self.store.current_run(r["id"])
            sessions.append({
                "id": r["id"], "name": r["colab_name"], "state": r["state"], "gpu_requested": r["gpu_requested"],
                "gpu_class": r["gpu_class"], "gpu_name": r["gpu_name"], "rate_cu_h": r["rate_cu_h"],
                "cu": round(billing.session_cu(r, now), 3),
                "up_s": round(billing.billing_end(r, now) - r["created_at"]),
                "endpoint_url": f"http://127.0.0.1:{r['proxy_port']}/v1", "model": run["alias"] if run else None,
                "stop_reason": r["stop_reason"], "stop_unconfirmed": bool(r["stop_unconfirmed"]),
                "progress": self._view.get(r["id"], {}).get("progress"), "idle_min": r["idle_min"],
            })
        return {
            "account_id": self.account_id,
            "sessions": sessions,
            "alerts": [dataclasses.asdict(a) for a in self.alerts.list()],
            "usage": dataclasses.asdict(self.usage) if self.usage else None,
            "usage_age_s": round(now - self.usage_at) if self.usage_at else None,
            "daily": {"spend": round(await self.spend_today(now), 3), "cap": await self.cap_today(now)},
        }

    def reload_presets(self) -> Presets | PresetError:
        try:
            self.presets = load_presets(self.s.presets_path)
        except PresetError as e:
            self.presets = e
        return self.presets

    def logs(self, sid: int) -> list[str]:
        return list(self._logs.get(sid, []))

    def logs_after(self, sid: int, after: int = 0) -> tuple[list[str], int]:
        lines = self.logs(sid)
        cursor = self._log_counts.get(sid, len(lines))
        pending = max(0, cursor - after) if after <= cursor else len(lines)
        return (lines[-pending:] if pending else []), cursor

    async def stats(self, days: int = 30) -> list[dict[str, Any]]:
        since = self.now() - days * 86400
        rows = await self.store.db.fetchall(
            "SELECT s.gpu_class, r.alias, r.kv, q.completion_tokens, q.prompt_tokens, q.engine_gen_ms, q.ttft_ms "
            "FROM requests q JOIN runs r ON r.id = q.run_id JOIN sessions s ON s.id = r.session_id "
            "WHERE q.ts >= ? AND q.status < 400 AND (q.client IS NULL OR q.client NOT LIKE 'colabinet-%')", (since,))
        groups: dict[tuple, list[dict]] = {}
        for r in rows:
            groups.setdefault((r["gpu_class"], r["alias"], r["kv"]), []).append(r)
        out = []
        for (gpu, alias, kv), items in sorted(groups.items(), key=lambda kv_: str(kv_[0])):
            tps = [i["completion_tokens"] / i["engine_gen_ms"] * 1000 for i in items
                   if i["completion_tokens"] and i["engine_gen_ms"]]
            ttft = [i["ttft_ms"] for i in items if i["ttft_ms"] is not None]
            out.append({"gpu": gpu, "model": alias, "kv": kv, "requests": len(items),
                        "tokens_in": sum(i["prompt_tokens"] or 0 for i in items),
                        "tokens_out": sum(i["completion_tokens"] or 0 for i in items),
                        "gen_tps_p50": round(statistics.median(tps), 1) if tps else None,
                        "ttft_ms_p50": round(statistics.median(ttft)) if ttft else None})
        return out
