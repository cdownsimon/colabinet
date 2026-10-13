"""JSON API on 127.0.0.1:<ui_port> (spec §10). Every request must carry a local Host header; every state-changing
request also needs X-Colabinet-Token (from <data_dir>/api.token) and, if present, a local Origin. Plan 2 adds the
browser's CSRF token as an alternative secret."""

from __future__ import annotations

import dataclasses
import hmac
import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, StrictBool

from .config import SEED_RATES
from .metrics import QueryError, to_csv
from .presets import PresetError
from .supervisor import CreateRefused, CreateRequest, Supervisor
from .web import services

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


@asynccontextmanager
async def _lifespan(app: FastAPI):
    try:
        yield
    finally:
        if login := getattr(app.state, "google_login", None):
            await login.close()


class Confirm(BaseModel):
    confirm: StrictBool = False


class CreateBody(BaseModel):
    name: str
    gpu: str
    preset: str | None = None
    engine: str = "llamacpp"
    revision: str | None = None
    parallel: int | None = Field(default=None, gt=0)
    repo: str | None = None
    file: str | None = None
    alias: str | None = None
    ctx: int | None = None
    kv: str | None = None
    refusal_projection: StrictBool | None = None
    extra_args: list[str] = []
    proxy_port: int | None = None
    vm_port: int = 8090
    idle_min: int | None = None
    deadline: str | None = None  # "HH:MM" local time: today, or tomorrow if already past
    cu_cap: float | None = None
    confirm: StrictBool = False


class ReconfigureBody(Confirm):
    preset: str | None = None
    ctx: int | None = Field(default=None, gt=0)
    kv: str | None = None
    extra_args: list[str] | None = None
    refusal_projection: StrictBool | None = None


class Extend(BaseModel):
    cu: float = 5.0


def parse_deadline(hhmm: str | None, now: datetime | None = None) -> float | None:
    if not hhmm:
        return None
    now = now or datetime.now()
    h, m = (int(x) for x in hhmm.split(":"))
    t = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if t <= now:
        t += timedelta(days=1)
    return t.timestamp()


def create_app(sup: Supervisor, token: str, ui_port: int, csrf: str | None = None) -> FastAPI:
    """JSON API; with `csrf`, also the console (web/app.py), whose browser requests carry X-CSRF-Token."""
    app = FastAPI(title="colabinet", docs_url=None, redoc_url=None, openapi_url="/api/openapi.json", lifespan=_lifespan)
    hosts = {f"127.0.0.1:{ui_port}", f"localhost:{ui_port}"}
    origins = {f"http://{h}" for h in hosts}

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        if request.headers.get("host") not in hosts:
            return JSONResponse({"error": "request rejected: Host must be 127.0.0.1 or localhost"}, 403)
        if request.method not in SAFE_METHODS:
            origin = request.headers.get("origin")
            if origin is not None and origin not in origins:
                return JSONResponse({"error": "request rejected: cross-site Origin"}, 403)
            api_ok = hmac.compare_digest(request.headers.get("x-colabinet-token", ""), token)
            csrf_ok = bool(csrf) and hmac.compare_digest(request.headers.get("x-csrf-token", ""), csrf or "")
            if not (api_ok or csrf_ok):
                return JSONResponse({"error": "request rejected: missing or wrong X-Colabinet-Token / X-CSRF-Token"},
                                    403)
        return await call_next(request)

    @app.exception_handler(QueryError)
    @app.exception_handler(CreateRefused)
    async def refused(_request: Request, exc: CreateRefused):
        return JSONResponse({"error": str(exc)}, 409)

    def need_confirm(body: Confirm | CreateBody) -> None:
        services.require_confirmation(body.confirm)

    @app.get("/api/health")
    async def health():
        return {"ok": True}

    @app.get("/api/state")
    async def state():
        return await sup.state()

    @app.get("/api/alerts")
    async def alerts():
        return [dataclasses.asdict(a) for a in sup.alerts.list()]

    @app.get("/api/stats")
    async def stats(days: int = 30):
        return await sup.stats(days)

    @app.get("/api/rates")
    async def rates():
        return {gpu: dict(zip(("rate_cu_h", "source"), await sup.store.rate(gpu))) for gpu in SEED_RATES}

    @app.get("/api/presets")
    async def presets():
        if isinstance(sup.presets, PresetError):
            return JSONResponse({"error": str(sup.presets)}, 500)
        return [{"id": p.id, "engine": p.engine, "label": p.label, "weights_gb": p.weights_gb, "licence": p.licence}
                for p in sup.presets.presets.values()]

    @app.get("/api/sessions/{sid}/logs")
    async def logs(sid: int):
        return {"lines": sup.logs(sid)}

    @app.post("/api/sessions")
    async def create(body: CreateBody):
        need_confirm(body)
        req = CreateRequest(
            name=body.name, gpu_class=body.gpu, preset_id=body.preset, repo=body.repo, file=body.file,
            alias=body.alias, ctx=body.ctx, kv=body.kv, extra_args=tuple(body.extra_args),
            engine=body.engine, revision=body.revision, parallel=body.parallel,
            refusal_projection=body.refusal_projection,
            proxy_port=body.proxy_port, vm_port=body.vm_port, idle_min=body.idle_min,
            deadline=parse_deadline(body.deadline), cu_cap=body.cu_cap, confirm=True,
        )
        return {"id": await sup.create(req)}

    @app.post("/api/sessions/stop-all")
    async def stop_all(body: Confirm):
        need_confirm(body)
        rows = await sup.store.live_sessions()
        for row in rows:
            await sup.stop(row["id"], "user")
        return {"stopped": [r["id"] for r in rows]}

    @app.post("/api/sessions/{sid}/stop")
    async def stop(sid: int, body: Confirm):
        need_confirm(body)
        await sup.stop(sid, "user")
        return {"ok": True}

    @app.post("/api/sessions/{sid}/abandon")
    async def abandon(sid: int):
        await sup.abandon(sid)
        return {"ok": True}

    @app.post("/api/sessions/{sid}/keep")
    async def keep(sid: int):
        await sup.keep(sid)
        return {"ok": True}

    @app.post("/api/sessions/{sid}/gpu-continue")
    async def gpu_continue(sid: int, body: Confirm):
        need_confirm(body)
        await sup.decide(sid, "continue")
        return {"ok": True}

    @app.post("/api/sessions/{sid}/retry-setup")
    async def retry_setup(sid: int, body: Confirm):
        need_confirm(body)
        await sup.decide(sid, "retry")
        return {"ok": True}

    @app.post("/api/external/{name}/stop")
    async def stop_external(name: str, body: Confirm):
        need_confirm(body)
        await sup.stop_external(name)
        return {"ok": True}

    @app.post("/api/daily-cap/extend")
    async def extend(body: Extend):
        return {"cap": await sup.extend_daily_cap(body.cu)}

    @app.post("/api/sessions/{sid}/reconfigure")
    async def reconfigure(sid: int, body: ReconfigureBody):
        need_confirm(body)
        return {"run_id": await sup.reconfigure(sid, preset_id=body.preset, ctx=body.ctx, kv=body.kv,
                                                extra_args=body.extra_args,
                                                refusal_projection=body.refusal_projection)}

    @app.post("/api/sessions/{sid}/restart-engine")
    async def restart_engine(sid: int, body: Confirm):
        need_confirm(body)
        return {"run_id": await sup.reconfigure(sid)}

    @app.get("/api/settings")
    async def settings():
        return await sup.settings_values()

    @app.post("/api/settings")
    async def update_settings(request: Request):
        body = await services.read_json(request)
        if not isinstance(body, dict):
            raise QueryError("settings must be an object")
        try:
            return await sup.update_settings(**body)
        except (ValueError, TypeError):
            raise QueryError("invalid settings value") from None

    @app.post("/api/presets/reload")
    async def reload_presets():
        result = sup.reload_presets()
        if isinstance(result, PresetError):
            raise QueryError(str(result))
        return {"count": len(result.presets)}

    @app.post("/api/metrics/query")
    async def metrics_query(request: Request):
        return services.result_json(await services.query_metrics(sup, await services.read_json(request)))

    @app.post("/api/metrics/sql")
    async def metrics_sql(request: Request):
        body = await services.read_json(request)
        if not isinstance(body, dict):
            raise QueryError("SQL request must be an object")
        return services.result_json(await services.query_sql(sup, body.get("sql")))

    @app.get("/api/metrics/export.csv")
    async def metrics_export(spec: str):
        try:
            raw = json.loads(spec)
        except ValueError:
            raise QueryError("invalid query JSON") from None
        result = await services.query_metrics(sup, raw)
        return PlainTextResponse(to_csv(result), media_type="text/csv",
                                 headers={"Content-Disposition": 'attachment; filename="colabinet-metrics.csv"'})

    @app.get("/api/saved-queries")
    async def saved_queries():
        return await sup.store.saved_queries()

    @app.post("/api/saved-queries")
    async def save_query(request: Request):
        body = await services.read_json(request)
        if not isinstance(body, dict):
            raise QueryError("saved query must be an object")
        return {"id": await services.save_query(sup, body.get("name"), body.get("spec"))}

    @app.delete("/api/saved-queries/{qid}")
    async def delete_query(qid: int):
        await sup.store.delete_query(qid)
        return {"ok": True}

    @app.post("/api/playground/send")
    async def playground_send(request: Request):
        return await services.send_playground(sup, await services.read_json(request))

    if csrf:
        from .web.app import mount_ui

        mount_ui(app, sup, csrf)
    return app
