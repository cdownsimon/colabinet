"""The console (spec §9, §10): server-rendered pages, htmx partials that poll, and form actions.

Every state-changing request goes through htmx or app.js with the per-process CSRF token in X-CSRF-Token (checked
by the API middleware); results come back as toasts via the HX-Trigger header, errors as 4xx JSON {"error": ...}.
"""

from __future__ import annotations

import json
import shlex
import time
from typing import Any

from fastapi import APIRouter, FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..api import parse_deadline
from ..config import GPU_CLASSES, PACKAGE_DIR, VRAM_GB
from ..diagnostics import redact_diagnostic
from ..fit import hf_file_gb
from ..metrics import CHARTS, GROUPS, MEASURES, TIME_PRESETS, QueryError, QuerySpec, to_csv
from ..presets import PresetError
from ..supervisor import CreateRefused, CreateRequest, Supervisor
from . import services, views
from .login import GoogleLogin, LoginError

TEMPLATES_DIR = PACKAGE_DIR / "templates"
STATIC_DIR = PACKAGE_DIR / "static"
PLAYGROUND_UA = "colabinet-playground"
TOAST_STATES = {"ready": ("success", "is ready"), "stopped": ("info", "stopped"),
                "setup_failed": ("error", "setup failed"), "engine_down": ("error", "engine is down"),
                "gpu_mismatch": ("warning", "got a different GPU"), "failed": ("error", "could not be created")}

def _form_projection(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError("refusal projection must be on, off or unchanged")



def _ago(timestamp: float | None) -> str:
    if not timestamp:
        return "-"
    statement = max(0, time.time() - timestamp)
    if statement < 60:
        return f"{statement:.0f}s ago"
    if statement < 3600:
        return f"{statement / 60:.0f} min ago"
    if statement < 86400:
        return f"{statement / 3600:.1f} h ago"
    return f"{statement / 86400:.0f} d ago"


def _dur(statement: float | None) -> str:
    if statement is None:
        return "-"
    statement = max(0, int(statement))
    if statement < 60:
        return f"{statement}s"
    if statement < 3600:
        return f"{statement // 60}m {statement % 60:02d}s"
    return f"{statement // 3600}h {statement % 3600 // 60:02d}m"


def _num(value: Any, digits: int = 1) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _when(timestamp: float | None) -> str:
    return time.strftime("%H:%M:%S", time.localtime(timestamp)) if timestamp else "-"


def make_templates() -> Jinja2Templates:
    templates = Jinja2Templates(directory=TEMPLATES_DIR)
    env = templates.env
    env.filters.update(ago=_ago, dur=_dur, num=_num, when=_when, day=views.now_str, tojson_attr=json.dumps)
    env.globals.update(gpu_classes=GPU_CLASSES, vram=VRAM_GB, state_variant=views.state_variant)
    return templates


def toast(response: Response, category: str, title: str, description: str = "") -> Response:
    response.headers["HX-Trigger"] = json.dumps({"cb-toast": {"category": category, "title": title,
                                                              "description": description}})
    return response


def _form_list(form, key: str) -> list[str]:
    return [value for value in form.getlist(key) if value]


def navigate(path: str, replace: bool = False) -> Response:
    response = Response(status_code=204)
    response.headers["HX-Location"] = json.dumps({
        "path": path, "source": "#cb-view", "target": "#cb-view", "select": "#cb-view > *",
        "selectOOB": "#cmdk:outerHTML", "swap": "innerHTML show:top",
        "push": False if replace else "true", "replace": "true" if replace else "false",
    })
    return response


def spec_from_form(form) -> dict[str, Any]:
    filters: dict[str, Any] = {key: _form_list(form, f"f_{key}") for key in ("gpu", "model", "engine", "kv", "session",
                                                                          "client", "path")}
    filters = {key: value for key, value in filters.items() if value}
    filters["status_class"] = form.get("status_class", "all")
    filters["exclude_playground"] = form.get("exclude_playground") == "on"
    # Keep chart selection and CSV columns stable when controls are grouped differently in the UI.
    measure_order = {measure: index for index, measure in enumerate([*MEASURES, "cu", "out_per_cu"])}
    measures = sorted(_form_list(form, "measures"), key=lambda measure: measure_order.get(measure, len(measure_order)))
    return {"time": form.get("time", "30d"), "group_by": _form_list(form, "group_by"),
            "measures": measures, "chart": form.get("chart", "bar"), "filters": filters}


def mount_ui(app: FastAPI, sup: Supervisor, csrf: str) -> None:
    templates = make_templates()
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    router = APIRouter()
    http = sup.http

    login = app.state.google_login = GoogleLogin(sup)

    async def page(request: Request, name: str, active: str, **ctx: Any) -> HTMLResponse:
        base = await views.base_context(sup)
        return templates.TemplateResponse(request, name, {"active": active, "csrf": csrf, "base": base, **ctx})

    def partial(request: Request, name: str, **ctx: Any) -> HTMLResponse:
        return templates.TemplateResponse(request, name, ctx)

    async def refused(fn, *args, **kw):
        try:
            return await fn(*args, **kw), None
        except (CreateRefused, QueryError) as exc:
            return None, JSONResponse({"error": str(exc)}, 409)


    @router.get("/", response_class=HTMLResponse)
    async def overview(request: Request):
        return await page(request, "pages/overview.html", "overview", overview_data=await views.overview(sup))

    @router.get("/sessions", response_class=HTMLResponse)
    async def sessions(request: Request, tab: str = "live"):
        return await page(request, "pages/sessions.html", "sessions", tab=tab)

    @router.get("/sessions/{sid}", response_class=HTMLResponse)
    async def session(request: Request, sid: int, tab: str = "overview"):
        detail = await views.session_detail(sup, sid)
        if detail is None:
            return HTMLResponse("session not found", 404)
        presets = [] if isinstance(sup.presets, PresetError) else list(sup.presets.presets.values())
        return await page(request, "pages/session.html", "sessions", detail=detail, tab=tab, presets=presets)

    @router.get("/new", response_class=HTMLResponse)
    async def new(request: Request, preset: str | None = None, gpu: str | None = None):
        presets = [] if isinstance(sup.presets, PresetError) else list(sup.presets.presets.values())
        rates = {group: (await sup.store.rate(group))[0] for group in GPU_CLASSES}
        values = await sup.settings_values()
        gpus = views.gpu_choices(values["eligible_gpus"])
        selected = next((model for model in presets if model.id == preset), None)
        if selected is None and preset != "custom":
            selected = presets[0] if presets else None
            preset = selected.id if selected else "custom"
        engine = selected.engine if selected else "llamacpp"
        fits = views.session_fit(engine, selected, gpus, await views.host_ram(sup))
        # A class asked for by name (the command menu, a link) is kept even if it cannot run this model: the form
        # flags it rather than silently choosing another. Without one, start on a class that can run it.
        if gpu is None:
            gpu = views.default_gpu(fits)
        elif gpu not in gpus:
            gpu = next((group for group in gpus if group != "CPU"), "CPU")
        return await page(request, "pages/new.html", "new", presets=presets, preset=preset, gpu=gpu, rates=rates,
                          gpus=gpus, engine=engine, idle_default=values["idle_min_default"], gpu_fit=fits,
                          preset_error=str(sup.presets) if isinstance(sup.presets, PresetError) else None)

    @router.get("/playground", response_class=HTMLResponse)
    async def playground(request: Request, session: int | None = None):
        live = [statement for statement in await views.session_rows(sup, "state = 'ready'")]
        return await page(request, "pages/playground.html", "playground", sessions=live, selected=session)

    @router.get("/metrics", response_class=HTMLResponse)
    async def metrics(request: Request, saved: int | None = None):
        spec = QuerySpec()
        if saved:
            row = await sup.store.db.fetchone("SELECT * FROM saved_queries WHERE id = ?", (saved,))
            if row:
                spec = QuerySpec.parse(json.loads(row["spec_json"]))
        options = {}
        for key, col in (("gpu", "gpu"), ("model", "model"), ("engine", "engine"), ("kv", "kv"),
                         ("session", "session"), ("client", "client"), ("path", "path")):
            rows = await sup.store.db.fetchall(
                f"SELECT DISTINCT {col} AS v FROM request_stats WHERE {col} IS NOT NULL ORDER BY 1 LIMIT 50")
            options[key] = [item["v"] for item in rows]
        return await page(request, "pages/metrics.html", "metrics", spec=spec, options=options,
                          groups=list(GROUPS), measures=list(MEASURES) + ["cu", "out_per_cu"],
                          times=list(TIME_PRESETS), charts=sorted(CHARTS), saved=await sup.store.saved_queries(),
                          saved_id=saved)

    @router.get("/models", response_class=HTMLResponse)
    async def models(request: Request):
        measure = await views.models(sup)
        for client in measure["custom"]:
            client["toml"] = views.custom_preset_toml(client)
        return await page(request, "pages/models.html", "models", models_context=measure)

    @router.get("/settings", response_class=HTMLResponse)
    async def settings(request: Request):
        if login.state == "idle":
            await login.check()
        response = await page(request, "pages/settings.html", "settings", st=await views.settings(sup),
                              account=login.snapshot())
        response.headers["Cache-Control"] = "no-store"
        return response

    def account_card(request: Request) -> HTMLResponse:
        response = partial(request, "partials/google_account.html", account=login.snapshot())
        response.headers["Cache-Control"] = "no-store"
        return response

    @router.get("/ui/auth", response_class=HTMLResponse)
    async def account_status(request: Request):
        return account_card(request)

    @router.post("/ui/actions/auth/{action}")
    async def account_action(request: Request, action: str):
        form = await request.form()
        try:
            if action == "start":
                await login.start(str(form.get("account_id", "")) or None)
            elif action == "check":
                await login.check()
            elif action == "code":
                await login.submit(str(form.get("flow_id", "")), str(form.get("code", "")))
            elif action == "cancel":
                await login.cancel(str(form.get("flow_id", "")))
            elif action == "switch":
                await login.switch(str(form.get("account_id", "")))
            elif action == "logout":
                await login.logout(str(form.get("account_id", "")))
            else:
                return JSONResponse({"error": "unknown account action"}, 404)
        except (LoginError, CreateRefused) as exc:
            return JSONResponse({"error": str(exc)}, 409, headers={"Cache-Control": "no-store"})
        except OSError:
            return JSONResponse({"error": "The account change could not be saved. Check data-directory permissions."},
                                409, headers={"Cache-Control": "no-store"})
        return account_card(request)


    @router.get("/ui/sidebar-status", response_class=HTMLResponse)
    async def sidebar_status(request: Request):
        return partial(request, "partials/sidebar_status.html", base=await views.base_context(sup))

    @router.get("/ui/overview/{part}", response_class=HTMLResponse)
    async def overview_part(request: Request, part: str):
        if part not in ("kpis", "attention", "sessions", "activity", "chart"):
            return HTMLResponse("", 404)
        return partial(request, f"partials/overview_{part}.html", overview_data=await views.overview(sup))

    @router.get("/ui/sessions/table", response_class=HTMLResponse)
    async def sessions_table(request: Request, tab: str = "live", query: str = Query(default="", alias="q")):
        where = {"live": "state NOT IN ('stopped', 'failed') OR stop_unconfirmed = 1",
                 "history": "state IN ('stopped', 'failed') AND stop_unconfirmed = 0"}.get(tab, "1=1")
        rows = await views.session_rows(sup, where, limit=200)
        total = len(rows)
        # Normalise once, then filter and display the same value. Filtering on the raw
        # query while showing the stripped one made a whitespace-only search look like an
        # empty list, and the table then claimed nothing was billing.
        query = query.strip()
        if query:
            ql = query.lower()
            rows = [item for item in rows if ql in " ".join(str(value or "") for value in (
                item["colab_name"], (item["run"] or {}).get("alias"), item["gpu_class"], item["gpu_name"],
                (item["run"] or {}).get("engine"), item["stop_reason"])).lower()]
        return partial(request, "partials/sessions_table.html", rows=rows, tab=tab, query=query, total=total)

    @router.get("/ui/sessions/{sid}/{part}", response_class=HTMLResponse)
    async def session_part(request: Request, sid: int, part: str, after: int = 0, after_event: int = 0):
        if part == "resources":
            resources = await views.session_resources(sup, sid)
            return partial(request, "partials/session_resources.html", resources=resources) if resources else (
                HTMLResponse("session not found", 404))
        if part == "controls":
            session = await sup.store.session(sid)
            return partial(request, "partials/session_controls.html", session=session, refresh_badge=True) if (
                session) else HTMLResponse("session not found", 404)
        if part == "logs.txt":
            creation = await sup.store.creation_diagnostics(sid)
            if creation is None:
                return HTMLResponse("session not found", 404)
            events = await sup.store.db.fetchall(
                "SELECT * FROM events WHERE session_id = ? ORDER BY id DESC LIMIT 200", (sid,))
            lines = ["Creation log", f"Exit code: {creation['exit_code']}" if creation["exit_code"] is not None
                     else "Exit code: not recorded", creation["output"] or creation["error"] or "No output captured.",
                     "", "Lifecycle events (latest 200)"]
            lines.extend(f"{views.now_str(event['ts'])} {event['kind']} {event['detail_json'] or ''}"
                         for event in reversed(events))
            lines.extend(["", "Engine log (latest 500 lines)", *sup.logs(sid)])
            return PlainTextResponse(redact_diagnostic("\n".join(lines)) + "\n", headers={
                "Content-Disposition": f'attachment; filename="session-{sid}.log"',
            })
        if part == "status":
            detail = await views.session_detail(sup, sid)
            return partial(request, "partials/session_status.html", detail=detail) if detail else HTMLResponse("", 404)
        if part == "runs":
            runs = await sup.store.db.fetchall("SELECT * FROM runs WHERE session_id = ? ORDER BY id DESC", (sid,))
            return partial(request, "partials/session_runs.html", runs=runs)
        if part == "requests":
            return partial(request, "partials/session_requests.html", rows=await views.requests_of(sup, sid))
        if part == "logs":
            creation = await sup.store.creation_diagnostics(sid)
            if creation is None:
                return HTMLResponse("session not found", 404)
            lines, cursor = sup.logs_after(sid, after)
            events = await sup.store.events(sid, after_event)
            events = [{**event, "detail_json": redact_diagnostic(event["detail_json"] or "")} for event in events]
            event_cursor = events[-1]["id"] if events else after_event
            response = partial(request, "partials/session_logs.html", events=events, lines=lines,
                               creation=creation)
            response.headers["X-Log-Cursor"] = str(cursor)
            response.headers["X-Event-Cursor"] = str(event_cursor)
            return response
        return HTMLResponse("", 404)

    @router.post("/ui/sessions/{sid}/watch")
    async def watch_session(sid: int):
        if not await sup.observe_session(sid):
            return HTMLResponse("session not found", 404)
        return Response(status_code=204)

    @router.post("/ui/new/estimate", response_class=HTMLResponse)
    async def new_estimate(request: Request):
        try:
            form = await request.form()
            gpu = form.get("gpu", "L4")
            rate = (await sup.store.rate(gpu))[0] if gpu in GPU_CLASSES else None
            preset = None if isinstance(sup.presets, PresetError) else sup.presets.presets.get(form.get("preset", ""))
            engine = preset.engine if preset else form.get("engine", "llamacpp")
            ctx_override = int(form["ctx"]) if form.get("ctx") else None
            weights = None
            if not preset and engine == "llamacpp" and form.get("repo") and form.get("file"):
                weights = await hf_file_gb(http, form["repo"], form["file"])
            projection = _form_projection(form.get("refusal_projection"))
            if projection is None:
                projection = preset.refusal_projection if preset else False
            if projection and engine != "strata":
                return JSONResponse({"error": "experimental refusal projection is supported only by Strata"}, 409)
            # Every offered class is assessed with the form's settings, so the GPU choices can grey out
            # the ones this configuration cannot run; the review below details the selected one.
            offered = views.gpu_choices((await sup.settings_values())["eligible_gpus"])
            fits = views.session_fit(engine, preset, offered + ([] if gpu in offered else [gpu]),
                                     await views.host_ram(sup), ctx_override, form.get("kv") or None, weights,
                                     tuple(shlex.split(form.get("extra_args") or "")), form.get("file") or None)
            fit = fits[gpu]
            # What the form offers instead when the selected class cannot run this: the same choice as its default.
            suggest = views.default_gpu({group: fits[group] for group in offered})
            if suggest == gpu or fits[suggest]["status"] == "blocked":
                suggest = ""
            ctx, kv = fit["ctx"], fit["kv"]
            idle = int(form.get("idle_min") or 0)
            cap = float(form.get("cu_cap") or 0) or None
            cost = {"rate": rate, "idle_waste": rate * idle / 60 if rate and idle else None,
                    "cap_minutes": cap / rate * 60 if cap and rate else None, "idle": idle, "cap": cap,
                    "no_limit": not idle and not cap and not form.get("deadline"),
                    "endpoint": form.get("proxy_port") or f"next free from {sup.s.proxy_port_start}"}
            daily = {"spend": await sup.spend_today(), "cap": await sup.cap_today()}
            if rate and daily["cap"]:
                left = daily["cap"] - daily["spend"]
                daily["minutes_left"] = max(0.0, left / rate * 60)
            return partial(request, "partials/new_estimate.html", fit=fit, fit_display=fit, gpu_fit=fits,
                           suggest=suggest, cost=cost, daily=daily, gpu=gpu,
                           ctx=ctx, kv=kv, preset=preset, engine=engine, refusal_projection=projection)
        except (ValueError, TypeError):
            return JSONResponse({"error": "check context, KV cache and cost fields"}, 409)

    @router.post("/ui/metrics/run", response_class=HTMLResponse)
    async def metrics_run(request: Request):
        form = await request.form()
        try:
            spec = QuerySpec.parse(spec_from_form(form))
        except QueryError as exc:
            return partial(request, "partials/metrics_result.html", error=str(exc))
        res = await services.query_metrics(sup, json.loads(spec.to_json()))
        return partial(request, "partials/metrics_result.html", res=res, spec=spec, spec_json=spec.to_json(),
                       error=None)

    @router.post("/ui/metrics/sql", response_class=HTMLResponse)
    async def metrics_sql(request: Request):
        form = await request.form()
        try:
            res = await services.query_sql(sup, form.get("sql", ""))
        except QueryError as exc:
            return partial(request, "partials/sql_result.html", error=str(exc))
        return partial(request, "partials/sql_result.html", res=res, error=None)

    @router.get("/ui/metrics/export.csv")
    async def metrics_export(spec: str):
        try:
            query = QuerySpec.parse(json.loads(spec))
        except (QueryError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, 400)
        res = await services.query_metrics(sup, json.loads(query.to_json()))
        return PlainTextResponse(to_csv(res), media_type="text/csv",
                                 headers={"Content-Disposition": 'attachment; filename="colabinet-metrics.csv"'})

    @router.post("/ui/metrics/save")
    async def metrics_save(request: Request):
        form = await request.form()
        name = (form.get("name") or "").strip()
        if not name:
            return JSONResponse({"error": "give the query a name"}, 409)
        try:
            spec = QuerySpec.parse(spec_from_form(form))
        except QueryError as exc:
            return JSONResponse({"error": str(exc)}, 409)
        qid = await services.save_query(sup, name, json.loads(spec.to_json()))
        resp = navigate(f"/metrics?saved={qid}" if qid else "/metrics")
        return toast(resp, "success", f"Saved “{name}”")

    @router.post("/ui/metrics/saved/{qid}/delete")
    async def metrics_delete(qid: int):
        await sup.store.delete_query(qid)
        return navigate("/metrics")

    @router.get("/ui/toasts")
    async def toasts(after: int = -1):
        last = (await sup.store.db.fetchone("SELECT COALESCE(MAX(id), 0) AS m FROM events"))["m"]
        if after < 0:
            return {"last": last, "toasts": []}
        rows = await sup.store.db.fetchall(
            "SELECT e.id, e.kind, e.detail_json, s.colab_name FROM events e "
            "LEFT JOIN sessions s ON s.id = e.session_id "
            "WHERE e.id > ? AND e.kind IN ('state', 'stop_unconfirmed') ORDER BY e.id LIMIT 20", (after,))
        out = []
        for row in rows:
            detail = json.loads(row["detail_json"]) if row["detail_json"] else {}
            if row["kind"] == "stop_unconfirmed":
                out.append({"category": "error", "title": f"{row['colab_name']}: stop not confirmed",
                            "description": "It may still be billing - see Needs attention"})
            elif detail.get("to") in TOAST_STATES:
                cat, text = TOAST_STATES[detail["to"]]
                reason = f" ({detail['stop_reason']})" if detail.get("stop_reason") else ""
                out.append({"category": cat, "title": f"{row['colab_name']} {text}{reason}", "description": ""})
        return {"last": rows[-1]["id"] if len(rows) == 20 else last, "toasts": out}


    @router.post("/ui/actions/create")
    async def create(request: Request):
        form = await request.form()
        services.require_confirmation(form.get("confirm") == "yes")
        preset = form.get("preset") or None
        try:
            extra = tuple(shlex.split(form.get("extra_args") or ""))
            req = CreateRequest(
                name=(form.get("name") or "").strip(), gpu_class=form.get("gpu", ""),
                preset_id=None if preset in (None, "custom") else preset, engine=form.get("engine") or "llamacpp",
                repo=form.get("repo") or None, file=form.get("file") or None, revision=form.get("revision") or None,
                alias=form.get("alias") or None, ctx=int(form["ctx"]) if form.get("ctx") else None,
                kv=form.get("kv") or None, extra_args=extra,
                refusal_projection=_form_projection(form.get("refusal_projection")),
                parallel=int(form["parallel"]) if form.get("parallel") else None,
                proxy_port=int(form["proxy_port"]) if form.get("proxy_port") else None,
                vm_port=int(form.get("vm_port") or 8090),
                idle_min=int(form["idle_min"]) if form.get("idle_min") not in (None, "") else None,
                deadline=parse_deadline(form.get("deadline") or None),
                cu_cap=float(form["cu_cap"]) if form.get("cu_cap") else None, confirm=form.get("confirm") == "yes",
            )
        except ValueError as exc:
            return JSONResponse({"error": f"check the form: {exc}"}, 409)
        if not req.name:
            return JSONResponse({"error": "give the session a name"}, 409)
        sid, err = await refused(sup.create, req)
        if err:
            return err
        return navigate(f"/sessions/{sid}")

    @router.post("/ui/actions/sessions/{sid}/{action}")
    async def session_action(request: Request, sid: int, action: str):
        form = await request.form()
        if action in ("stop", "gpu-continue", "retry-setup", "reconfigure", "restart-engine"):
            services.require_confirmation(form.get("confirm") == "yes")
        row = await sup.store.session(sid)
        name = row["colab_name"] if row else f"#{sid}"
        if action == "stop":
            _, err = await refused(sup.stop, sid, "user")
            msg = ("info", f"Stopping {name}")
        elif action == "abandon":
            _, err = await refused(sup.abandon, sid)
            msg = ("warning", f"Abandoned {name}", "If a VM appears later it will be listed as unmanaged")
        elif action == "keep":
            _, err = await refused(sup.keep, sid)
            msg = ("success", f"{name}: idle timer reset")
        elif action == "gpu-continue":
            _, err = await refused(sup.decide, sid, "continue")
            msg = ("info", f"{name}: continuing on the assigned GPU")
        elif action == "retry-setup":
            _, err = await refused(sup.decide, sid, "retry")
            msg = ("info", f"{name}: retrying setup")
        elif action == "restart-engine":
            _, err = await refused(sup.reconfigure, sid)
            msg = ("info", f"{name}: restarting engine")
        elif action == "reconfigure":
            try:
                raw_extra = form.get("extra_args")
                extra = tuple(shlex.split(raw_extra)) if raw_extra is not None else None
                ctx = int(form["ctx"]) if form.get("ctx") else None
                projection = _form_projection(form.get("refusal_projection"))
            except ValueError as exc:
                return JSONResponse({"error": f"check the form: {exc}"}, 409)
            _, err = await refused(sup.reconfigure, sid, preset_id=form.get("preset") or None, ctx=ctx,
                                   kv=form.get("kv") or None, extra_args=extra, refusal_projection=projection)
            msg = ("info", f"{name}: reconfiguring", "The endpoint answers 503 until the new run is ready")
        else:
            return JSONResponse({"error": f"unknown action {action!r}"}, 404)
        if err:
            return err
        return toast(Response(status_code=204), *msg)

    @router.post("/ui/actions/stop-all")
    async def stop_all(request: Request):
        services.require_confirmation((await request.form()).get("confirm") == "yes")
        rows = await sup.store.live_sessions()
        for row in rows:
            await sup.stop(row["id"], "user")
        return toast(Response(status_code=204), "info", f"Stopping {len(rows)} session(s)")

    @router.post("/ui/actions/external/{name}/stop")
    async def stop_external(request: Request, name: str):
        form = await request.form()
        services.require_confirmation(form.get("confirm") == "yes")
        _, err = await refused(sup.stop_external, name, expected_account=str(form.get("account_id", "")))
        return err or toast(Response(status_code=204), "info", f"Asked Colab to stop {name}")

    @router.post("/ui/actions/daily-cap/extend")
    async def extend():
        cap = await sup.extend_daily_cap(5.0)
        return toast(Response(status_code=204), "success", f"Today's cap is now {cap:.2f} CU")

    @router.post("/ui/actions/presets/reload")
    async def reload_presets():
        result = sup.reload_presets()
        if isinstance(result, PresetError):
            return JSONResponse({"error": f"presets.toml: {result}"}, 409)
        return toast(navigate("/models", replace=True), "success", f"Loaded {len(result.presets)} preset(s)")

    @router.post("/ui/actions/settings")
    async def save_settings(request: Request):
        form = await request.form()
        values = {key: form[key] for key in ("daily_cap_cu", "low_balance_cu", "idle_min_default") if form.get(key)}
        values["eligible_gpus"] = ",".join(part.strip().upper() for value in _form_list(form, "eligible_gpus")
                                           for part in value.split(",") if part.strip())
        values["notify"] = form.get("notify") == "on"
        try:
            await sup.update_settings(**values)
        except (CreateRefused, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, 409)
        # The page in front of the operator still holds the GPU list rendered before this save,
        # so the command menu is re-sent as an out-of-band swap. 204 would be dropped by htmx
        # before the swap is processed, hence the rendered body. Replacing the whole dialog
        # matches the soft-navigation refresh and re-initialises the menu's item cache.
        menu = partial(request, "partials/cmdk.html", base=await views.base_context(sup), oob=True)
        return toast(menu, "success", "Settings saved")


    @router.post("/ui/playground/send")
    async def playground_send(request: Request):
        return await services.send_playground(sup, await services.read_json(request))

    @router.get("/ui/playground/models")
    async def playground_models(session_id: int):
        row = await sup.store.session(session_id)
        if not row or row["state"] != "ready":
            return JSONResponse({"error": "pick a session that is ready"}, 409)
        resp = await http.get(f"http://127.0.0.1:{row['proxy_port']}/v1/models", headers={"User-Agent": PLAYGROUND_UA})
        return JSONResponse({"status": resp.status_code, "body": resp.json()})

    app.include_router(router)
