"""A fake `colab` executable that reproduces google-colab-cli 0.7.4 behaviour, including its hazards.

State lives in the JSON file named by $FAKE_COLAB_STATE:
  store   {name: endpoint}          the CLI's local session store
  server  {endpoint: {...}}          assignments that exist (and bill) on Colab's side
  faults  {...}                      injected failures (see FAULTS below)
  hazards [...]                      things colabinet must never cause: auto_create, overwrite, new_killed
Each endpoint gets a sandbox directory that stands in for the VM's /content; remote paths and --env values that
start with /content are mapped into it, and uploaded/exec'd scripts run locally inside it.
"""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

STATE = Path(os.environ.get("FAKE_COLAB_STATE", "colab_state.json"))
RATES = {"NONE": 0.08, "T4": 1.07, "L4": 1.54, "A100": 5.30, "G4": 8.90}
GPU = {  # accelerator, shape -> nvidia-smi "name, total MiB"
    ("L4", "STANDARD"): "NVIDIA L4, 23034",
    ("A100", "STANDARD"): "NVIDIA A100-SXM4-40GB, 40960",
    ("A100", "HIGH_RAM"): "NVIDIA A100-SXM4-80GB, 81920",
    ("T4", "STANDARD"): "Tesla T4, 15360",
    ("T4", "HIGH_RAM"): "Tesla T4, 15360",
    ("G4", "STANDARD"): "NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887",
}
FAULTS = {
    "new": None,  # None | "not_available" | "crash_after_assign" | "hang" | "slow"
    "new_delay": 0.0,
    "assign_accel": None,  # hand out this accelerator instead of the requested one (GPU-mismatch tests)
    "hide_when_store_empty": False,  # `sessions` prints "No active sessions" when the store is empty
    "usage_fail": False,
    "stop_noop": False,  # `stop` reports success but the assignment stays
    "exec_drop": 0,  # the next N execs die with "Connection was lost."
}
EMPTY = "[colab] No active sessions found on server."


def default_state() -> dict:
    return {"store": {}, "server": {}, "balance": 200.0, "next_id": 1, "faults": dict(FAULTS), "hazards": [],
            "calls": []}


@contextmanager
def locked():
    STATE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        st = json.loads(STATE.read_text()) if STATE.exists() else default_state()
        yield st
        save(STATE, st)


def save(path: Path, st: dict) -> None:
    """Replace the state file in one step: tests read it without the lock and must never see it half-written."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(st, indent=1))
    os.replace(tmp, path)


def sandbox(endpoint: str) -> Path:
    return STATE.parent / "vm" / endpoint


def to_local(path: str, endpoint: str) -> str:
    return str(sandbox(endpoint) / path[len("/content/"):]) if path.startswith("/content/") else path


def opt(args: list[str], flag: str) -> str | None:
    return args[args.index(flag) + 1] if flag in args else None


def prune(st: dict) -> int:
    stale = [n for n, ep in st["store"].items() if ep not in st["server"]]
    for n in stale:
        del st["store"][n]
    return len(stale)


def cmd_new(args: list[str]) -> int:
    name = opt(args, "-s")
    accel = (opt(args, "--gpu") or "NONE").upper()
    shape = "HIGH_RAM" if "--high-mem" in args and accel not in ("L4", "G4") else "STANDARD"
    with locked() as st:
        f = st["faults"]
        if name in st["store"]:
            st["hazards"].append(f"overwrite:{name}")
        if f["new"] == "not_available":
            print(f.get("new_error") or f"[colab] Backend rejected accelerator '{accel}'. You may not have quota.",
                  file=sys.stderr)
            return 1
        accel = (f["assign_accel"] or accel).upper()
        endpoint = f"{'m' if accel == 'NONE' else 'gpu-' + accel.lower()}-s-{st['next_id']}"
        st["next_id"] += 1
        st["server"][endpoint] = {"accel": accel, "shape": shape, "pids": []}
        mode, delay = f["new"], f["new_delay"]
    print(f"[colab] Creating session '{name}'...", flush=True)
    if mode == "crash_after_assign":
        return 1
    if mode == "hang":
        def killed(*_):
            with locked() as st2:
                st2["hazards"].append(f"new_killed:{name}")
            sys.exit(143)

        signal.signal(signal.SIGTERM, killed)
        signal.signal(signal.SIGINT, killed)
        time.sleep(3600)
    if mode == "slow":
        time.sleep(delay)
    with locked() as st:
        st["store"][name] = endpoint
    print("[colab] Session READY.")
    return 0


def cmd_sessions(args: list[str]) -> int:
    with locked() as st:
        pruned = prune(st)
        if pruned:
            print(f"[colab] Pruned {pruned} stale local session(s).")
        if (not st["store"] and st["faults"]["hide_when_store_empty"]) or not st["server"]:
            print(EMPTY)
            return 0
        names = {ep: n for n, ep in st["store"].items()}
        for ep, a in st["server"].items():
            hw = "CPU" if a["accel"] == "NONE" else a["accel"]
            shape = "High-RAM" if a["shape"] == "HIGH_RAM" else "Standard"
            variant = "DEFAULT" if a["accel"] == "NONE" else "GPU"
            print(f"[{names.get(ep, '?')}] {ep} | Hardware: {hw} | Shape: {shape} | Variant: {variant}")
    return 0


def cmd_usage(args: list[str]) -> int:
    with locked() as st:
        if st["faults"]["usage_fail"]:
            print("[colab] Warning: failed to fetch compute-unit info: boom", file=sys.stderr)
            return 1
        rate = sum(RATES[a["accel"]] * (1.27 if a["shape"] == "HIGH_RAM" else 1.0) for a in st["server"].values())
        print(f"Current balance: {st['balance']:.2f} compute units\nUsage rate: {rate:.2f}/hr\n"
              f"Active assignments: {len(st['server'])}")
    return 0


def session_endpoint(st: dict, name: str) -> str | None:
    ep = st["store"].get(name)
    if ep and ep in st["server"]:
        return ep
    if ep:
        del st["store"][name]
    return None


def cmd_upload(args: list[str]) -> int:
    name, local, remote = opt(args, "-s"), args[-2], args[-1]
    with locked() as st:
        ep = session_endpoint(st, name)
    if not ep:
        print(f"[colab] Session '{name}' not found.")
        return 1
    dest = Path(to_local(remote, ep))
    if not dest.parent.is_dir():  # the real Contents API does not create parent directories (seen on a real runtime)
        print(f"[colab] Upload failed: 500 Server Error: Internal Server Error for url: .../api/contents{remote}")
        return 1
    dest.write_bytes(Path(local).read_bytes())
    print(f"[colab] Uploaded {local} -> {remote}")
    return 0


def cmd_exec(args: list[str]) -> int:
    name, file = opt(args, "-s"), opt(args, "-f")
    envs = [args[i + 1] for i, a in enumerate(args) if a == "--env"]
    with locked() as st:
        ep = session_endpoint(st, name)
        if ep and st["faults"]["exec_drop"] > 0:
            st["faults"]["exec_drop"] -= 1
            print("RuntimeError: Connection was lost.")
            return 1
        server = st["server"].get(ep) if ep else None
        if server is not None:  # --env sets os.environ in the persistent kernel: values stick between execs
            kernel_env = server.setdefault("kernel_env", {})
            for item in envs:
                k, v = item.split("=", 1)
                kernel_env[k] = to_local(v, ep)
    if not ep:
        print(f"[colab] Session '{name}' appears to be lost (404/401). Cleaning up.")
        return 1
    box = sandbox(ep)
    (box / "bin").mkdir(parents=True, exist_ok=True)
    gpu = GPU.get((server["accel"], server["shape"]))
    smi = box / "bin" / "nvidia-smi"
    if gpu and not smi.exists():
        smi.write_text(f"#!/bin/sh\necho '{gpu}, 512, 550.54, 72, 61'\n")
        smi.chmod(0o755)
    env = {**os.environ, **server.get("kernel_env", {})}
    env["PATH"] = f"{box / 'bin'}:{env['PATH']}"
    env["COLABINET_ROOT"] = str(box / "colabinet")
    p = subprocess.run([sys.executable, file], cwd=box, env=env, capture_output=True, text=True)
    sys.stdout.write(p.stdout + p.stderr)
    for line in p.stdout.splitlines():
        if line.startswith("COLABINET_BG_PID "):
            with locked() as st:
                pid = int(line.split()[1])
                st["all_pids"] = st.get("all_pids", []) + [pid]
                if ep in st["server"]:
                    st["server"][ep]["pids"].append(pid)
    return p.returncode


def cmd_stop(args: list[str]) -> int:
    name = opt(args, "-s")
    with locked() as st:
        ep = st["store"].get(name)
        if not ep:
            print(f"[colab] Session '{name}' not found.")
            return 0
        print(f"[colab] Stopping session '{name}'...")
        for pid in st["server"].get(ep, {}).get("pids", []):
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if not st["faults"]["stop_noop"]:
            st["server"].pop(ep, None)
        del st["store"][name]
    print("[colab] Session terminated.")
    return 0


def cmd_ssh(args: list[str]) -> int:
    name = opt(args, "-s")
    with locked() as st:
        prune(st)
        if name not in st["store"]:  # the real CLI auto-creates a (billing) VM here
            st["hazards"].append(f"auto_create:{name}")
            endpoint = f"m-s-{st['next_id']}"
            st["next_id"] += 1
            st["server"][endpoint] = {"accel": "NONE", "shape": "STANDARD", "pids": []}
            st["store"][name] = endpoint
    return 1  # the fake can't bridge SSH; integration tests use the local tunnel


def main() -> int:
    args = sys.argv[1:]
    if args and args[0] in ("--auth=oauth2", "--auth=adc"):
        args = args[1:]
    with locked() as st:
        st["calls"].append(args)
    commands = {"new": cmd_new, "sessions": cmd_sessions, "usage": cmd_usage, "upload": cmd_upload,
                "exec": cmd_exec, "stop": cmd_stop, "ssh": cmd_ssh, "version": lambda _a: print("Version: 0.7.4")}
    if not args or args[0] not in commands:
        print(f"fake colab: unsupported {args}", file=sys.stderr)
        return 2
    return commands[args[0]](args[1:])


if __name__ == "__main__":
    sys.exit(main())
