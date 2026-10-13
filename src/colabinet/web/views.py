"""Read models for the console pages: plain dicts assembled from the supervisor and the database."""

from __future__ import annotations

import dataclasses
import json
import math
import time
from typing import Any

from .. import billing
from ..config import GPU_CLASSES, SYSTEM_RAM_GB, VRAM_GB
from ..fit import EXL3_OVERHEAD_GB, OVERHEAD_GB, Fit, estimate, kv_gb
from ..metrics import cu_by_group
from ..presets import Measured, Preset, PresetError
from ..store import LIVE_STATES
from ..supervisor import OPEN_ROWS, Supervisor

SETUP_STEPS = [("vm_created", "VM"), ("state:setting_up", "GPU check"), ("vm:install_done", "install"),
               ("vm:download_done", "download"), ("vm:server_ready", "load"), ("state:ready", "tunnel")]
_EVIDENCE_BADGES = {
    "tested": ("GPU tested", "secondary"),
    "reported": ("Reported GPU run", "outline"),
    "build_only": ("Build verified", "outline"),
}


def state_variant(state: str, unconfirmed: bool = False) -> str:
    if unconfirmed or state in ("gpu_mismatch", "setup_failed", "engine_down", "failed"):
        return "destructive"
    if state == "ready":
        return "default"
    if state == "stopped":
        return "outline"
    return "secondary"


def gpu_choices(eligible_setting: str) -> list[str]:
    """GPU classes the operator allows for new sessions. CPU is always allowed as the fallback."""
    eligible = {group.strip().upper() for group in eligible_setting.split(",")} | {"CPU"}
    return [group for group in GPU_CLASSES if group.split("-")[0] in eligible]


async def base_context(sup: Supervisor) -> dict[str, Any]:
    now = sup.now()
    live = await sup.store.live_sessions()
    daily_spend, daily_cap = await sup.spend_today(now), await sup.cap_today(now)
    values = await sup.settings_values()
    return {
        "balance": sup.usage.balance if sup.usage else None,
        "usage_age_s": round(now - sup.usage_at) if sup.usage_at else None,
        "burn_rate": round(sum(row["rate_cu_h"] for row in live if row["state"] != "stopping"), 2),
        "live_count": len(live),
        "daily": {"spend": daily_spend, "cap": daily_cap},
        "alert_count": len(sup.alerts.list()),
        "command_sessions": [{"id": row["id"], "name": row["colab_name"], "state": row["state"],
                              "endpoint": f"http://127.0.0.1:{row['proxy_port']}/v1"} for row in live],
        "new_gpus": gpu_choices(values["eligible_gpus"]),
    }


async def session_rows(sup: Supervisor, where: str, params: tuple = (), limit: int | None = None) -> list[dict]:
    now = sup.now()
    sql = f"SELECT * FROM sessions WHERE {where} ORDER BY id DESC" + (f" LIMIT {int(limit)}" if limit else "")
    rows = await sup.store.db.fetchall(sql, params)
    accounts = {row["id"]: row["email"] for row in sup.accounts.rows} if sup.accounts else {}
    out = []
    for row in rows:
        run = await sup.store.current_run(row["id"])
        if run:
            run["refusal_projection"] = json.loads(run["spec_json"]).get("refusal_projection", False)
        creation = await sup.store.creation_diagnostics(row["id"]) if row["state"] == "failed" else None
        stats = await sup.store.db.fetchone(
            "SELECT COUNT(*) AS n, SUM(completion_tokens) AS out, median(gen_tps) AS tps FROM request_stats "
            "WHERE session_id = ? AND is_playground = 0 AND status < 400", (row["id"],))
        idle_left = None
        if row["state"] in ("tunnelling", "ready", "engine_down") and row["idle_min"]:
            act = sup._activity.get(row["id"])
            last = max(act.effective(now) if act else 0.0, row["last_activity"] or 0.0)
            idle_left = max(0, round(row["idle_min"] * 60 - (now - last)))
        out.append({
            **row, "account_email": accounts.get(row["account_id"], "Shared CLI account"),
            "run": run, "cu": billing.session_cu(row, now), "up_s": billing.billing_end(row,
                now) - row["created_at"],
            "endpoint_url": f"http://127.0.0.1:{row['proxy_port']}/v1", "variant":
                state_variant(row["state"], bool(row["stop_unconfirmed"])), "requests": stats["n"],
            "tokens_out": stats["out"] or 0, "gen_tps": stats["tps"], "idle_left_s": idle_left,
            "progress": sup._view.get(row["id"], {}).get("progress"),
            "creation_error": creation["error"] if creation else "",
        })
    return out


async def overview(sup: Supervisor) -> dict[str, Any]:
    now = sup.now()
    start, _ = sup.clock.day_bounds(now)
    live = await session_rows(sup, OPEN_ROWS)
    recent = await session_rows(sup, "state IN ('stopped', 'failed') AND stop_unconfirmed = 0", limit=5)
    today = await sup.store.db.fetchone(
        "SELECT COUNT(*) AS n, SUM(completion_tokens) AS out, SUM(prompt_tokens) AS inp FROM request_stats "
        "WHERE ts >= ? AND is_playground = 0", (start,))
    thirty = await sup.store.sessions("created_at < ? AND (stopped_at IS NULL OR stopped_at > ?)",
                                      (now, now - 30 * 86400))
    balance_start = await sup.store.get_setting(f"balance_start:{sup.account_id}:{sup.clock.day_key(now)}")
    base = await base_context(sup)
    burn = base["burn_rate"]
    return {
        "account_id": sup.account_id,
        "kpis": {
            "live": len([row for row in live if row["state"] in LIVE_STATES]),
            "ready": len([row for row in live if row["state"] == "ready"]),
            "burn_rate": burn,
            "balance": base["balance"],
            "balance_age_s": base["usage_age_s"],
            "hours_left": (base["balance"] / burn) if base["balance"] and burn else None,
            "cu_today": base["daily"]["spend"], "cap_today": base["daily"]["cap"],
            "balance_drop": (float(balance_start) - base["balance"]) if balance_start and base["balance"] else None,
            "cu_30d": billing.spend_between(thirty, now - 30 * 86400, now, now),
            "requests_today": today["n"], "tokens_out_today": today["out"] or 0, "tokens_in_today": today["inp"] or 0,
        },
        "alerts": [dataclasses.asdict(argument) for argument in sup.alerts.list()],
        "sessions": live + recent,
        "activity": await activity(sup, limit=15),
        "cu_chart": await cu_per_day(sup, days=14),
    }


async def activity(sup: Supervisor, limit: int = 15, session_id: int | None = None) -> list[dict[str, Any]]:
    where, params = ("e.session_id = ?", (session_id,)) if session_id else ("1=1", ())
    rows = await sup.store.db.fetchall(
        f"SELECT e.*, s.colab_name FROM events e LEFT JOIN sessions s ON s.id = e.session_id WHERE {where} "
        f"AND e.kind NOT IN ('vm:download') ORDER BY e.id DESC LIMIT ?", (*params, limit))
    return [{**row, "detail": json.loads(row["detail_json"]) if row["detail_json"] else {}} for row in rows]


async def cu_per_day(sup: Supervisor, days: int = 14) -> dict[str, Any]:
    now = sup.now()
    end = sup.clock.day_bounds(now)[1]
    start = end - days * 86400
    runs = await sup.store.db.fetchall(
        "SELECT r.started_at, COALESCE(r.ended_at, s.stopped_at) AS ended_at, s.rate_cu_h, s.gpu_class AS gpu, "
        "r.alias AS model, r.kv, r.engine, s.colab_name AS session FROM runs r JOIN sessions s ON s.id = r.session_id")
    cu = cu_by_group(runs, ["day", "gpu"], start, now, now, sup.clock.day_bounds, sup.clock.day_key)
    labels = [sup.clock.day_key(start + index * 86400 + 43200) for index in range(days)]
    gpus = sorted({key[1] for key in cu})
    return {"labels": labels, "rows": [{"label": detail[5:], **{group: round(cu.get((detail, group), 0.0),
        3) for group in gpus}}
                                       for detail in labels], "series": gpus}


def pi_provider_json(session: dict[str, Any]) -> str:
    run = session["run"] or {}
    model = run.get("alias") or "model"
    cfg = {"colab": {"baseUrl": session["endpoint_url"], "api": "openai-completions", "apiKey": "none",
                     "models": [{"id": model, "name": f"{model} (colabinet {session['colab_name']})",
                                 "reasoning": True, "contextWindow": run.get("ctx") or 32768,
                                 "maxTokens": 32000}]}}
    return json.dumps(cfg, indent=2)


async def session_detail(sup: Supervisor, sid: int) -> dict[str, Any] | None:
    rows = await session_rows(sup, "id = ?", (sid,))
    if not rows:
        return None
    statement = rows[0]
    runs = await sup.store.db.fetchall("SELECT * FROM runs WHERE session_id = ? ORDER BY id DESC", (sid,))
    events = await sup.store.db.fetchall("SELECT kind FROM events WHERE session_id = ?", (sid,))
    kinds = {exc["kind"] for exc in events}
    state_events = await sup.store.db.fetchall(
        "SELECT detail_json FROM events WHERE session_id = ? AND kind = 'state'", (sid,))
    reached = {f"state:{json.loads(exc['detail_json'])['to']}" for exc in state_events if exc["detail_json"]}
    steps = [{"label": label, "done": key in kinds or key in reached} for key, label in SETUP_STEPS]
    speed = await sup.store.db.fetchall(
        "SELECT run_id, model, kv, prompt_bucket, COUNT(*) AS n, median(gen_tps) AS gen, median(prefill_tps) AS pre, "
        "median(ttft_ms) AS ttft FROM request_stats WHERE session_id = ? AND is_playground = 0 AND status < 400 "
        "GROUP BY run_id, prompt_bucket ORDER BY run_id DESC, prompt_bucket", (sid,))
    guards = {
        "idle_min": statement["idle_min"], "idle_left_s": statement["idle_left_s"],
            "deadline": statement["deadline"], "cu_cap": statement["cu_cap"],
        "cap_pct": min(100, round(100 * statement["cu"] / statement["cu_cap"])) if statement["cu_cap"] else None,
        "last_exec_ok": statement["last_exec_ok"],
    }
    return {"s": statement, "runs": runs, "steps": steps, "speed": speed, "guards": guards,
            "creation": await sup.store.creation_diagnostics(sid),
            "pi_json": pi_provider_json(statement),
            "resources": await session_resources(sup, sid),
                "alerts": [dataclasses.asdict(argument) for argument in sup.alerts.list()
                                                         if argument.session_id == sid]}


async def session_resources(sup: Supervisor, sid: int) -> dict[str, Any] | None:
    row = await sup.store.session(sid)
    if row is None:
        return None
    sample = sup.resource_sample(sid) or {}
    received = sample.get("sampled_at")
    age = max(0, sup.now() - received) if received is not None else None
    resources = {
        **sample, "sampled_at": received, "stale": age is None or age > 45,
        "ended": row["state"] in ("stopping", "stopped", "failed"), "cpu_only": row["gpu_class"] == "CPU",
    }
    for name in ("ram", "disk"):
        reading = dict(resources.get(name) or {})
        total, used = reading.get("total_bytes"), reading.get("used_bytes")
        reading["percent"] = min(100, max(0, 100 * used / total)) if total and used is not None else None
        resources[name] = reading
    resources["cpu"] = resources.get("cpu") or {}
    resources["gpu"] = resources.get("gpu") or {}
    gpu = resources["gpu"]
    resources["vram_pct"] = min(100, max(0, 100 * gpu["used_mib"] / gpu["vram_mib"])) if (
        gpu.get("vram_mib") and gpu.get("used_mib") is not None) else None
    return resources


async def requests_of(sup: Supervisor, sid: int, limit: int = 200) -> list[dict[str, Any]]:
    return await sup.store.db.fetchall(
        "SELECT * FROM request_stats WHERE session_id = ? ORDER BY id DESC LIMIT ?", (sid, limit))


def fit_presentation(engine: str, fit: Fit, preset: Preset | None, gpu: str) -> dict[str, str]:
    """Show fit estimates or recorded GPU evidence, never infer evidence from fallback defaults."""
    if engine == "strata":
        if fit.verdict == "cpu":
            return {"label": "GPU required", "variant": "destructive", "message": "Strata requires a GPU runtime"}
        evidence = preset.evidence.get(gpu) if preset else None
        if evidence:
            label, variant = _EVIDENCE_BADGES[evidence.status]
            return {"label": label, "variant": variant,
                    "message": evidence.message + " Recorded evidence is not a RAM/VRAM estimate or a guarantee "
                               "for the requested context, PLE mode or refusal projection."}
        return {"label": "Not benchmarked", "variant": "outline",
                "message": "No recorded benchmark for this preset/GPU pair. Strata uses a GPU expert cache, "
                           "host RAM and a shared 28.8 GB PLE table. Download size is not VRAM residency; "
                           "check available host RAM."}
    label = {"fits": "Fits in VRAM", "tight": "Tight on VRAM", "spills": "CPU offload required",
             "cpu": "CPU only", "unknown": "Size unknown"}[fit.verdict]
    variant = "secondary" if fit.verdict == "fits" else "outline"
    message = fit.message
    if engine == "exllamav3":
        if fit.verdict == "cpu":
            label, variant, message = "GPU required", "destructive", "ExLlamaV3 requires a GPU runtime"
        elif fit.verdict == "spills":
            label, variant = "Exceeds VRAM", "destructive"
            message = "Estimated weights and KV cache exceed VRAM; reduce context/cache size or choose a larger GPU"
    elif fit.verdict == "spills":
        message += "; sufficient host RAM is required for offloaded weights"
    if fit.verdict in ("fits", "tight") and fit.kv_gb is None:
        label += " (weights only)"
    return {"label": label, "variant": variant, "message": message}


# Strata's own hardware policy, from setup.py at the pinned release (d6708a4): host RAM and expert arena, in GiB.
# Run on each VM before downloading; 2026-10-13 runs on T4, T4-HIGHMEM, L4 and A100s all matched it.
STRATA_MODELS = {"IQ2_XS": (48, 35.5), "IQ3_XXS": (60, 42.9), "IQ3_S": (62, 50.3)}


def strata_policy(quant: str, ram_gib: float, vram_gib: float, low_ram: str = "auto") -> str | None:
    """Why Strata would refuse this quant on this machine, or None when its policy accepts it."""
    ram_need, arena = STRATA_MODELS[quant]

    def low_fits():
        return ram_gib - 6 + max(0.0, vram_gib - 5) >= arena

    low = low_ram == "on" or (low_ram == "auto" and ram_gib < arena + 10)
    if low and not low_fits():
        return (f"Strata {quant} cannot fit about {ram_gib:.0f} GiB host RAM and {vram_gib:.0f} GiB VRAM under its "
                "low-RAM policy")
    if not low and ram_gib < ram_need - 4:
        hint = "; low-RAM mode would fit: add extra engine args --low-ram on" if low_fits() else ""
        return f"Strata {quant} requires about {ram_need} GiB host RAM; this class has about {ram_gib:.0f}{hint}"
    return None


def strata_low_ram(args: tuple[str, ...]) -> str:
    try:
        from ..engines.strata import _split_setup_args
        return _split_setup_args(tuple(args))[0]
    except ValueError:
        return "auto"


def gpu_fit(engine: str, gpu: str, preset: Preset | None, weights_gb: float | None, kv_bytes_per_token: int | None,
            ctx: int, kv: str, host_ram_gb: float | None, strata_quant: str | None = None,
            low_ram: str = "auto") -> dict[str, Any]:
    """How one model configuration fits one GPU class, and whether the console offers that class for it.

    status is "blocked" when the runtime cannot run the model at all (shown as a destructive badge), "ok" when it
    fits, and "warn" for everything that runs with a caveat, such as llama.cpp offloading layers to the CPU."""
    fit = estimate(gpu, weights_gb, ctx, kv, kv_bytes_per_token,
                   overhead_gb=EXL3_OVERHEAD_GB if engine == "exllamav3" else OVERHEAD_GB)
    shown = fit_presentation(engine, fit, preset, gpu)
    if engine == "llamacpp" and weights_gb and host_ram_gb:
        # Offloaded layers live in host RAM; past VRAM + RAM, llama.cpp pages weights from disk and is unusable.
        need = weights_gb + (kv_gb(ctx, kv, kv_bytes_per_token) or 0.0) + OVERHEAD_GB
        room = VRAM_GB.get(gpu, 0.0) + host_ram_gb
        if need > room:
            # Measured 2026-10-13: the 111 GB Flash-Next GGUF paged from disk and generated 0.24 tok/s on a T4
            # (16 GB VRAM, 13.6 GB RAM) and 1.8 tok/s on a T4-HIGHMEM (54.8 GB RAM), whose 0.6 tok/s prompt
            # processing would take nearly half an hour for a 1K-token prompt.
            shown = {"label": "Too large", "variant": "destructive",
                     "message": f"Needs about {need:.0f} GB, but this runtime has about {room:.0f} GB of VRAM "
                                "and host RAM together; llama.cpp would page weights from disk and crawl"}
    if engine == "strata" and strata_quant in STRATA_MODELS and host_ram_gb and VRAM_GB.get(gpu):
        refusal = strata_policy(strata_quant, host_ram_gb * 1e9 / 2**30, VRAM_GB[gpu], low_ram)
        if refusal:
            shown = {"label": "Not enough RAM", "variant": "destructive", "message": refusal}
    return {"gpu": gpu, "verdict": fit.verdict, "ctx": ctx, "kv": kv, "kv_gb": fit.kv_gb,
            "status": {"destructive": "blocked", "secondary": "ok"}.get(shown["variant"], "warn"),
            "measured": measured_speed(preset.measured.get(gpu) if preset else None), **shown}


def tps(value: float) -> str:
    return f"{value:,.1f}" if value >= 10 or value == round(value, 1) else f"{value:.2g}"


def measured_speed(speed: Measured | None) -> dict[str, str] | None:
    """A preset's measured speed on one class, as display text: shown on the GPU choices, the review and Models."""
    if speed is None:
        return None
    prefill = f"{speed.prefill_tps:,.0f}" if speed.prefill_tps else None
    detail = f"{tps(speed.gen_tps)} tok/s generation" + (f", {prefill} tok/s prefill on a 4K-token prompt"
                                                           if prefill else "")
    return {"gen": tps(speed.gen_tps), "prefill": prefill, "date": speed.date, "note": speed.note,
            "detail": detail + (f" ({speed.note})" if speed.note else "")}


def fit_matrix(preset, host_ram: dict[str, float] | None = None) -> list[dict[str, Any]]:
    out = []
    strata = preset.engine == "strata"
    for gpu in GPU_CLASSES:
        detail = preset.defaults_for(gpu)
        row = gpu_fit(preset.engine, gpu, preset, None if strata else preset.weights_gb,
                      None if strata else preset.kv_bytes_per_token_f16, detail.ctx, detail.kv,
                      (host_ram or {}).get(gpu), strata_quant(preset.file), strata_low_ram(detail.extra_args))
        out.append({**row, "kv_unknown": row["kv_gb"] is None and gpu != "CPU" and not strata})
    return out


def strata_quant(file: str | None) -> str | None:
    return file.split("/", 1)[0] if file and file.split("/", 1)[0] in STRATA_MODELS else None


def session_fit(engine: str, preset: Preset | None, gpus: list[str], host_ram: dict[str, float],
                ctx: int | None = None, kv: str | None = None, weights_gb: float | None = None,
                extra_args: tuple[str, ...] | None = None, file: str | None = None) -> dict[str, dict]:
    """Fit of the New session form's model on each offered GPU, honouring any context, KV cache or engine-arg
    override (engine args replace a preset's own, as they do at create).

    A preset supplies its own size and per-GPU defaults; a custom model only has the size looked up for it."""
    strata = engine == "strata"
    if preset:
        weights_gb, file = preset.weights_gb, preset.file
    out = {}
    for gpu in gpus:
        defaults = preset.defaults_for(gpu) if preset else None
        args = extra_args if extra_args else (defaults.extra_args if defaults else ())
        out[gpu] = gpu_fit(
            engine, gpu, preset, None if strata else weights_gb,
            None if strata or not preset else preset.kv_bytes_per_token_f16,
            ctx or (defaults.ctx if defaults else 131072 if strata else 32768),
            kv or (defaults.kv if defaults else {"exllamav3": "FP16", "strata": "int8"}.get(engine, "f16")),
            host_ram.get(gpu), strata_quant(file) if strata else None, strata_low_ram(args))
    return out


def default_gpu(fits: dict[str, dict]) -> str:
    """The class a new session starts on when none was asked for, and the one offered instead of a class that cannot
    run the model: L4 (else the cheapest GPU offered) unless it cannot run it, then the cheapest GPU class that fits,
    then the cheapest that runs it at all. `fits` holds the offered classes, cheapest first."""
    preferred = "L4" if "L4" in fits else next((gpu for gpu in fits if gpu != "CPU"), "CPU")
    if fits.get(preferred, {}).get("status", "blocked") != "blocked":
        return preferred
    for wanted in ("ok", "warn"):
        for gpu, row in fits.items():
            if row["status"] == wanted and gpu != "CPU":
                return gpu
    return preferred


async def host_ram(sup: Supervisor) -> dict[str, float]:
    """Host RAM per class: the latest valid reading from a recorded session, else the reference value."""
    return {gpu: ram for gpu, (ram, _) in (await host_ram_sources(sup)).items()}


async def host_ram_sources(sup: Supervisor) -> dict[str, tuple[float, str]]:
    observations = await sup.store.db.fetchall(
        "SELECT gpu_class, ram_gb FROM sessions WHERE ram_gb > 0 AND gpu_class IS NOT NULL "
        "ORDER BY created_at DESC, id DESC")
    observed: dict[str, float] = {}
    for observation in observations:
        if math.isfinite(observation["ram_gb"]):
            observed.setdefault(observation["gpu_class"], observation["ram_gb"])
    return {gpu: (observed[gpu], "observed") if gpu in observed else (SYSTEM_RAM_GB[gpu], "reference")
            for gpu in GPU_CLASSES}


async def models(sup: Supervisor) -> dict[str, Any]:
    if isinstance(sup.presets, PresetError):
        return {"error": str(sup.presets), "presets": [], "custom": []}
    best = await sup.store.db.fetchall(
        "SELECT r.preset_id, rs.gpu, median(rs.gen_tps) AS tps, COUNT(*) AS n, SUM(rs.completion_tokens) AS out "
        "FROM request_stats rs JOIN runs r ON r.id = rs.run_id WHERE rs.status < 400 AND rs.is_playground = 0 "
        "GROUP BY r.preset_id, rs.gpu")
    by_preset: dict[str, list] = {}
    for b in best:
        by_preset.setdefault(b["preset_id"], []).append(b)
    ram = await host_ram(sup)
    presets = [{"p": preset, "fit": fit_matrix(preset, ram), "measured": by_preset.get(preset.id, [])}
               for preset in sup.presets.presets.values()]
    custom = await sup.store.db.fetchall(
        "SELECT engine, repo, file_or_rev, alias, MAX(started_at) AS last FROM runs WHERE preset_id IS NULL "
        "GROUP BY engine, repo, file_or_rev ORDER BY last DESC LIMIT 10")
    return {"error": None, "presets": presets, "custom": custom}


def custom_preset_toml(run: dict[str, Any]) -> str:
    ident = (run["alias"] or run["repo"].split("/")[-1]).replace("/", "-").lower()
    key = "file" if run["engine"] in ("llamacpp", "strata") else "revision"
    return (f'[[preset]]\nid = "{ident}"\nengine = "{run["engine"]}"\nlabel = "{run["alias"] or run["repo"]}"\n'
            f'repo = "{run["repo"]}"\n{key} = "{run["file_or_rev"]}"\nalias = "{run["alias"]}"\n'
            + ('vision = true\n\n[preset.gpu.A100]\nctx = 131072\nkv = "int8"\n'
               if run["engine"] == "strata" else ''))


async def settings(sup: Supervisor) -> dict[str, Any]:
    rates = [{"gpu": group, **dict(zip(("rate", "source"), await sup.store.rate(group))), "vram": VRAM_GB[group]}
             for group in GPU_CLASSES]
    learned = await sup.store.db.fetchall("SELECT gpu, updated_at FROM gpu_rates")
    updated = {row["gpu"]: row["updated_at"] for row in learned}
    ram = await host_ram_sources(sup)
    for row in rates:
        row["updated_at"] = updated.get(row["gpu"])
        row["ram_gb"], row["ram_source"] = ram[row["gpu"]]
    db_size = sup.store.db.path.stat().st_size if sup.store.db.path.exists() else 0
    config = await sup.settings_values()
    gpu_types = [gpu for gpu in GPU_CLASSES if gpu != "CPU" and "-" not in gpu]
    eligible_gpus = {gpu.strip().upper() for gpu in config["eligible_gpus"].split(",")}
    return {"config": config, "gpu_types": gpu_types, "eligible_gpus": eligible_gpus,
            "rates": rates, "cli_version": sup.cli_version,
            "colab_auth": sup.s.colab_auth,
            "colab_bin": sup.s.colab_bin, "ssh_key": str(sup.s.ssh_key), "ssh_key_ok": sup.s.ssh_key.exists(),
            "hf_token": bool(sup.hf_token), "db_path": str(sup.store.db.path), "db_mb": round(db_size / 1e6, 2),
            "stop_on_exit": sup.s.stop_on_exit, "setup_limit_s": sup.s.setup_limit_s,
            "grace_s": sup.s.grace_s, "proxy_port_start": sup.s.proxy_port_start,
            "hidden_port_range": sup.s.hidden_port_range}


def now_str(cursor: float | None) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(cursor)) if cursor else "-"
