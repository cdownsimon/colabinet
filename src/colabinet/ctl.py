"""`colabinet ctl ...`: a small client for the running server (plan 1 has no UI)."""

from __future__ import annotations

import argparse
import sys
from typing import Any

import httpx

from .config import Settings


class CtlError(Exception):
    pass


class Client:
    def __init__(self, settings: Settings) -> None:
        try:
            token = settings.token_path.read_text().strip()
        except FileNotFoundError as e:
            raise CtlError(f"no API token at {settings.token_path}: is `colabinet serve` running?") from e
        self.http = httpx.Client(base_url=f"http://127.0.0.1:{settings.ui_port}",
                                 headers={"X-Colabinet-Token": token}, timeout=httpx.Timeout(10, read=180))

    def call(self, method: str, path: str, body: dict | None = None) -> Any:
        try:
            r = self.http.request(method, path, json=body)
        except httpx.ConnectError as e:
            raise CtlError("cannot reach colabinet: is `colabinet serve` running?") from e
        data = r.json()
        if r.status_code >= 400:
            raise CtlError(data.get("error") if isinstance(data, dict) else str(data))
        return data


def _table(rows: list[dict], cols: list[str]) -> str:
    if not rows:
        return "(none)"
    widths = {c: max(len(c), *(len(str(r.get(c, "") if r.get(c) is not None else "-")) for r in rows)) for c in cols}
    lines = ["  ".join(c.ljust(widths[c]) for c in cols)]
    for r in rows:
        lines.append("  ".join(str(r.get(c) if r.get(c) is not None else "-").ljust(widths[c]) for c in cols))
    return "\n".join(lines)


def confirm(prompt: str, yes: bool) -> None:
    if yes:
        return
    if input(f"{prompt}\nType yes to continue: ").strip().lower() != "yes":
        raise CtlError("cancelled")


def run(argv: list[str], settings: Settings, out=print) -> int:
    ap = argparse.ArgumentParser(prog="colabinet ctl")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    c = sub.add_parser("create")
    c.add_argument("--name", required=True)
    c.add_argument("--gpu", required=True, help="CPU, T4, T4-HIGHMEM, L4, A100, A100-80, G4")
    c.add_argument("--preset")
    c.add_argument("--engine", choices=("llamacpp", "exllamav3", "strata"), default="llamacpp")
    c.add_argument("--revision", help="EXL3 revision")
    c.add_argument("--parallel", type=int)
    c.add_argument("--refusal-projection", action=argparse.BooleanOptionalAction, default=None,
                   help="Strata only: experimental projection reduces refusals and changes model behavior")
    c.add_argument("--repo")
    c.add_argument("--file")
    c.add_argument("--alias")
    c.add_argument("--ctx", type=int)
    c.add_argument("--kv")
    c.add_argument("--extra-arg", action="append", default=[])
    c.add_argument("--proxy-port", type=int)
    c.add_argument("--vm-port", type=int, default=8090)
    c.add_argument("--idle", type=int, help="idle stop in minutes (0 = off)")
    c.add_argument("--deadline", help="HH:MM local")
    c.add_argument("--cap", type=float, help="per-session CU cap")
    c.add_argument("--yes", action="store_true")
    for name in ("stop", "gpu-continue", "retry-setup"):
        p = sub.add_parser(name)
        p.add_argument("id", type=int)
        p.add_argument("--yes", action="store_true")
    for name in ("keep", "abandon", "logs"):
        sub.add_parser(name).add_argument("id", type=int)
    se = sub.add_parser("stop-external")
    se.add_argument("name")
    se.add_argument("--yes", action="store_true")
    ex = sub.add_parser("extend-daily-cap")
    rc = sub.add_parser("reconfigure")
    rc.add_argument("id", type=int)
    rc.add_argument("--preset")
    rc.add_argument("--ctx", type=int)
    rc.add_argument("--kv")
    rc.add_argument("--extra-arg", action="append", default=None)
    rc.add_argument("--refusal-projection", action=argparse.BooleanOptionalAction, default=None,
                    help="Strata only: experimental projection reduces refusals and changes model behavior")
    rc.add_argument("--yes", action="store_true")
    restart = sub.add_parser("restart-engine")
    restart.add_argument("id", type=int)
    restart.add_argument("--yes", action="store_true")
    ex.add_argument("--cu", type=float, default=5.0)
    sub.add_parser("alerts")
    sub.add_parser("usage")
    st = sub.add_parser("stats")
    st.add_argument("--days", type=int, default=30)
    a = ap.parse_args(argv)

    try:
        cl = Client(settings)
        if a.cmd == "list":
            state = cl.call("GET", "/api/state")
            out(_table(state["sessions"], ["id", "name", "state", "gpu_class", "cu", "rate_cu_h", "endpoint_url",
                                           "model", "stop_reason"]))
            daily = state["daily"]
            out(f"today: {daily['spend']:.2f} / {daily['cap']:.2f} CU; alerts: {len(state['alerts'])}")
        elif a.cmd == "create":
            rates = cl.call("GET", "/api/rates")
            state = cl.call("GET", "/api/state")
            rate = rates.get(a.gpu, {}).get("rate_cu_h")
            if rate is None:
                raise CtlError(f"unknown GPU {a.gpu!r}; choose from {', '.join(rates)}")
            idle = 30 if a.idle is None else a.idle
            lines = [f"Create {a.gpu} session {a.name!r} at {rate:.2f} CU/h.",
                     f"Idle stop {idle} min -> at most ~{rate * idle / 60:.2f} CU unused." if idle else
                     "Idle stop is OFF.",
                     f"Today so far {state['daily']['spend']:.2f} of {state['daily']['cap']:.2f} CU (daily cap)."]
            if a.cap:
                lines.append(f"Session cap {a.cap} CU -> stops after ~{a.cap / rate * 60:.0f} min.")
            if not idle and not a.deadline and not a.cap:
                lines.append("WARNING: no idle stop, deadline or cap - only the daily cap will stop this session.")
            if a.refusal_projection:
                lines.append("WARNING: experimental refusal projection reduces refusals and changes behavior; "
                             "it does not guarantee an uncensored model.")
            confirm("\n".join(lines), a.yes)
            body = {"name": a.name, "gpu": a.gpu, "preset": a.preset, "repo": a.repo, "file": a.file,
                    "alias": a.alias, "ctx": a.ctx, "kv": a.kv, "extra_args": a.extra_arg,
                    "engine": a.engine, "revision": a.revision, "parallel": a.parallel,
                    "refusal_projection": a.refusal_projection,
                    "proxy_port": a.proxy_port, "vm_port": a.vm_port, "idle_min": a.idle, "deadline": a.deadline,
                    "cu_cap": a.cap, "confirm": True}
            out(f"created session {cl.call('POST', '/api/sessions', body)['id']}")
        elif a.cmd == "reconfigure":
            warning = ("\nExperimental refusal projection reduces refusals and changes behavior; "
                       "it does not guarantee an uncensored model." if a.refusal_projection else "")
            confirm(f"Reconfigure session {a.id} on the same billing VM?{warning}", a.yes)
            body = {"preset": a.preset, "ctx": a.ctx, "kv": a.kv, "extra_args": a.extra_arg,
                    "refusal_projection": a.refusal_projection, "confirm": True}
            out(f"started run {cl.call('POST', f'/api/sessions/{a.id}/reconfigure', body)['run_id']}")
        elif a.cmd == "restart-engine":
            confirm(f"Restart the engine for session {a.id} on the same billing VM?", a.yes)
            out(f"started run {cl.call('POST', f'/api/sessions/{a.id}/restart-engine', {'confirm': True})['run_id']}")
        elif a.cmd in ("stop", "gpu-continue", "retry-setup"):
            confirm(f"{a.cmd} session {a.id}?", a.yes)
            cl.call("POST", f"/api/sessions/{a.id}/{a.cmd}", {"confirm": True})
            out("ok")
        elif a.cmd in ("keep", "abandon"):
            cl.call("POST", f"/api/sessions/{a.id}/{a.cmd}")
            out("ok")
        elif a.cmd == "logs":
            out("\n".join(cl.call("GET", f"/api/sessions/{a.id}/logs")["lines"]))
        elif a.cmd == "stop-external":
            confirm(f"stop the unmanaged runtime {a.name!r}?", a.yes)
            cl.call("POST", f"/api/external/{a.name}/stop", {"confirm": True})
            out("ok")
        elif a.cmd == "extend-daily-cap":
            out(f"today's cap is now {cl.call('POST', '/api/daily-cap/extend', {'cu': a.cu})['cap']:.2f} CU")
        elif a.cmd == "alerts":
            out(_table(cl.call("GET", "/api/alerts"), ["level", "kind", "session_id", "message"]))
        elif a.cmd == "usage":
            state = cl.call("GET", "/api/state")
            u = state["usage"] or {}
            out(f"balance {u.get('balance', '?')} CU, rate {u.get('rate_per_hour', '?')}/h, "
                f"{u.get('active_assignments', '?')} assignments (read {state['usage_age_s']} s ago); "
                f"today {state['daily']['spend']:.2f} / {state['daily']['cap']:.2f} CU")
        elif a.cmd == "stats":
            out(_table(cl.call("GET", f"/api/stats?days={a.days}"),
                       ["gpu", "model", "kv", "requests", "tokens_in", "tokens_out", "gen_tps_p50", "ttft_ms_p50"]))
    except CtlError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0
