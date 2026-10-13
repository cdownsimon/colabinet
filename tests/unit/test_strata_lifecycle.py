import dataclasses
import json
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from pydantic import ValidationError

from colabinet.alerts import Alerts, NullNotifier
from colabinet.api import CreateBody, ReconfigureBody
from colabinet.config import REPO_ROOT, Settings
from colabinet.presets import load_presets
from colabinet.supervisor import CreateRefused, CreateRequest, Supervisor

REPO = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
FILE = "IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf"
PRESET = "qwen38-flash-next-strata-iq2-xs"


@pytest.fixture
def supervisor(store, clock, tmp_path, monkeypatch):
    sup = Supervisor(settings=Settings(data_dir=tmp_path), store=store, cli=None, tunnels=None, proxies=None,
                     alerts=Alerts(NullNotifier(), clock.now), clock=clock,
                     presets=load_presets(REPO_ROOT / "presets.toml"), http=object())
    monkeypatch.setattr(sup, "_start_driver", Mock())
    monkeypatch.setattr(sup, "_cancel_driver", AsyncMock())
    return sup


def request(**overrides):
    fields = dict(name="strata", gpu_class="A100", engine="strata", repo=REPO, file=FILE, confirm=True)
    return CreateRequest(**{**fields, **overrides})


async def seed_run(sup, *, gpu="A100", state="ready", projection=True, old_spec=False):
    spec = sup._run_spec(request(refusal_projection=projection))
    payload = dataclasses.asdict(spec)
    if old_spec:
        payload.pop("refusal_projection")
    sid = await sup.store.insert_session(account_id="default", colab_name="strata", gpu_requested="A100",
        gpu_class=gpu, proxy_port=3080, hidden_port=23000, vm_port=8090, state=state,
        state_since=sup.now(), created_at=sup.now(), rate_cu_h=5, rate_source="seed", idle_min=30)
    await sup.store.insert_run(session_id=sid, engine="strata", repo=REPO, file_or_rev=FILE,
        alias=spec.alias, ctx=spec.ctx, kv=spec.kv, parallel=1, started_at=sup.now(), status="ready",
        spec_json=json.dumps(payload))
    return sid


def test_custom_strata_defaults_and_preset_projection_override(supervisor):
    preset = supervisor.presets.presets[PRESET]
    supervisor.presets.presets[PRESET] = dataclasses.replace(preset, refusal_projection=True)
    assert supervisor._run_spec(request(preset_id=PRESET)).refusal_projection is True
    assert supervisor._run_spec(request(preset_id=PRESET, refusal_projection=False)).refusal_projection is False


@pytest.mark.parametrize("overrides", [
    {"gpu_class": "CPU"}, {"parallel": 2}, {"kv": "f16"}, {"ctx": 0}, {"ctx": 262145},
    {"repo": "other/model"}, {"file": FILE.replace("00001", "00002")},
    {"file": FILE.replace("IQ2_XS", "Q4_K_M")}, {"revision": "main"},
    {"extra_args": ("--context", "512")}, {"refusal_projection": "true"},
    {"engine": "llamacpp", "refusal_projection": True},
])
async def test_invalid_create_is_refused_before_listing_or_billing(supervisor, monkeypatch, overrides):
    cap = AsyncMock(side_effect=AssertionError("billing must not be reached"))
    listing = AsyncMock(side_effect=AssertionError("listing must not be reached"))
    monkeypatch.setattr(supervisor, "cap_today", cap)
    monkeypatch.setattr(supervisor, "fresh_listing", listing)
    with pytest.raises(CreateRefused):
        await supervisor.create(request(**overrides))
    assert await supervisor.store.sessions() == []
    cap.assert_not_awaited()
    listing.assert_not_awaited()


@pytest.mark.parametrize("overrides", [{"ctx": 0}, {"kv": "f16"}, {"ctx": 262145},
    {"extra_args": ("--vision", "off")}, {"refusal_projection": "true"}])
async def test_invalid_reconfigure_leaves_current_run_and_driver_untouched(supervisor, overrides):
    sid = await seed_run(supervisor)
    before = await supervisor.store.current_run(sid)
    with pytest.raises(CreateRefused):
        await supervisor.reconfigure(sid, **overrides)
    assert await supervisor.store.current_run(sid) == before
    assert (await supervisor.store.session(sid))["state"] == "ready"
    supervisor._cancel_driver.assert_not_awaited()
    supervisor._start_driver.assert_not_called()


@pytest.mark.parametrize("projection", [None, False, True])
async def test_restart_and_explicit_projection_overrides_persist(supervisor, projection):
    sid = await seed_run(supervisor)
    first = await supervisor.store.current_run(sid)
    rid = await supervisor.reconfigure(sid, refusal_projection=projection)
    run = await supervisor.store.current_run(sid)
    spec = json.loads(run["spec_json"])
    assert rid != first["id"] and rid == run["id"]
    assert spec["refusal_projection"] is (True if projection is None else projection)
    assert (spec["vision"], spec["ctx"], spec["kv"]) == (True, 131072, "int8")
    assert (await supervisor.store.session(sid))["state"] == "reconfiguring"


async def test_new_preset_uses_its_projection_default_not_previous_model(supervisor):
    sid = await seed_run(supervisor, projection=True)
    await supervisor.reconfigure(sid, preset_id="qwen38-flash-next-strata-iq3-s")
    assert json.loads((await supervisor.store.current_run(sid))["spec_json"])["refusal_projection"] is False


async def test_old_spec_without_projection_can_restart(supervisor):
    sid = await seed_run(supervisor, old_spec=True)
    await supervisor.reconfigure(sid)
    assert json.loads((await supervisor.store.current_run(sid))["spec_json"])["refusal_projection"] is False


@pytest.mark.parametrize("action", ["continue", "retry", "restart"])
async def test_cpu_strata_cannot_continue_retry_or_restart(supervisor, action):
    state = {"continue": "gpu_mismatch", "retry": "setup_failed", "restart": "ready"}[action]
    sid = await seed_run(supervisor, gpu="CPU", state=state)
    before = await supervisor.store.current_run(sid)
    with pytest.raises(CreateRefused, match="GPU"):
        if action == "restart":
            await supervisor.reconfigure(sid)
        else:
            await supervisor.decide(sid, action)
    assert await supervisor.store.current_run(sid) == before
    assert (await supervisor.store.session(sid))["state"] == state
    assert sid not in supervisor._decision


async def test_detected_cpu_is_setup_failure_not_continue_prompt(supervisor, monkeypatch):
    sid = await seed_run(supervisor, state="gpu_check")
    row = await supervisor.store.session(sid)
    monkeypatch.setattr(supervisor, "_prepare_vm", AsyncMock(return_value=True))
    monkeypatch.setattr(supervisor, "_status", AsyncMock(return_value={"gpu": {}, "ram_gb": 12}))
    monkeypatch.setattr(supervisor, "_settle_rate", AsyncMock())
    assert await supervisor._gpu_check(row) == "setup_failed"
    assert (await supervisor.store.session(sid))["gpu_class"] == "CPU"
    assert (await supervisor.store.current_run(sid))["status"] == "failed"
    assert not any(alert.kind == "gpu_mismatch" for alert in supervisor.alerts.list())


@pytest.mark.parametrize("body_class,base", [(CreateBody, {"name": "s", "gpu": "A100"}), (ReconfigureBody, {})])
@pytest.mark.parametrize("value", ["true", "false", 1, 0, [], {}])
def test_json_projection_requires_real_boolean_or_null(body_class, base, value):
    with pytest.raises(ValidationError):
        body_class(**base, refusal_projection=value)


@pytest.mark.parametrize("data", [[], [{"id": "wrong"}], [{"id": ""}], [None], None])
async def test_frontend_health_without_loaded_model_is_not_tunnel_ready(supervisor, data):
    sid = await seed_run(supervisor)
    row = await supervisor.store.session(sid)
    def respond(request):
        return httpx.Response(200, json={"status": "ok"} if request.url.path == "/health" else {"data": data})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        supervisor.http = http
        assert await supervisor._healthy(row) is False


async def test_native_death_drives_ready_session_to_engine_down(supervisor, monkeypatch):
    sid = await seed_run(supervisor)
    row = await supervisor.store.session(sid)
    alias = (await supervisor.store.current_run(sid))["alias"]
    models = [{"id": alias}]
    def respond(request):
        return httpx.Response(200, json={"status": "ok"} if request.url.path == "/health" else {"data": models})
    monkeypatch.setattr(supervisor, "_sample_due", lambda session_id: True)
    monkeypatch.setattr(supervisor, "_ensure_tunnel", AsyncMock())
    monkeypatch.setattr(supervisor, "_sleep_or_wake", AsyncMock(return_value=False))
    supervisor.tunnels = Mock()
    supervisor.tunnels.alive.return_value = True
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        supervisor.http = http
        assert await supervisor._healthy(row) is True
        models.clear()  # Frontend remains HTTP 200, while the native engine disappears.
        assert await supervisor._healthy(row) is False
        async def status(current, **env):
            return {"engine_healthy": await supervisor._healthy(current), "marker": {"session_id": sid}}
        monkeypatch.setattr(supervisor, "_status", status)
        assert await supervisor._ready(row) == "ready"
        assert await supervisor._ready(row) == "engine_down"
    assert any(alert.kind == "engine_down" for alert in supervisor.alerts.list())


@pytest.mark.parametrize("loaded", [True, False])
def test_vm_status_distinguishes_live_native_engine_from_frontend(tmp_path, monkeypatch, loaded):
    import io
    import urllib.request

    from tests.unit.test_vm_resources import vm_status
    class Reply(io.BytesIO):
        status = 200
    def urlopen(url, **kwargs):
        payload = {"status": "ok"} if url.endswith("/health") else {
            "data": [{"id": "expected"}] if loaded else []}
        return Reply(json.dumps(payload).encode())
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("ENGINE_KIND", "strata")
    monkeypatch.setenv("ENGINE_MODEL", "expected")
    assert vm_status(tmp_path, monkeypatch, None)["engine_healthy"] is loaded


async def test_retry_keeps_projection_and_does_not_replace_the_run(supervisor, monkeypatch):
    sid = await seed_run(supervisor, state="setup_failed", projection=True)
    before = await supervisor.store.current_run(sid)
    await supervisor.decide(sid, "retry")
    monkeypatch.setattr(supervisor, "_sleep_or_wake", AsyncMock(return_value=True))
    assert await supervisor._waiting(await supervisor.store.session(sid)) == "setting_up"
    after = await supervisor.store.current_run(sid)
    assert after["id"] == before["id"] and after["spec_json"] == before["spec_json"]
    assert json.loads(after["spec_json"])["refusal_projection"] is True
    assert after["status"] == "pending"


def test_custom_strata_toml_round_trips_file_and_gpu_vision(supervisor, tmp_path):
    from colabinet.web.views import custom_preset_toml
    toml = custom_preset_toml({"engine": "strata", "repo": REPO, "file_or_rev": FILE, "alias": "strata"})
    path = tmp_path / "custom.toml"
    path.write_text(toml)
    preset = load_presets(path).presets["strata"]
    spec = supervisor._run_spec(request())
    assert preset.file == spec.file and preset.revision is None and preset.vision is True
    from colabinet.engines.base import RunSpec
    exported = RunSpec.from_preset(preset, "A100", {})
    supervisor._validate_spec(exported, "A100")
    assert (exported.ctx, exported.kv, exported.parallel) == (131072, "int8", 1)


async def test_projection_opt_in_cannot_switch_to_a_non_strata_engine(supervisor):
    sid = await seed_run(supervisor)
    before = await supervisor.store.current_run(sid)
    with pytest.raises(CreateRefused, match="only by Strata"):
        await supervisor.reconfigure(sid, preset_id="qwen38-27b-abl-exl3-4bpw", refusal_projection=True)
    assert await supervisor.store.current_run(sid) == before
    supervisor._cancel_driver.assert_not_awaited()
    await supervisor.reconfigure(sid, preset_id="qwen38-27b-abl-exl3-4bpw")
    run = await supervisor.store.current_run(sid)
    assert run["engine"] == "exllamav3" and json.loads(run["spec_json"])["refusal_projection"] is False


async def test_stock_engine_health_does_not_require_strata_loaded_models(supervisor):
    sid = await seed_run(supervisor)
    run = await supervisor.store.current_run(sid)
    await supervisor.store.update_run(run["id"], engine="llamacpp")
    def respond(request):
        return httpx.Response(200, json={"status": "ok"} if request.url.path == "/health" else {"data": []})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        supervisor.http = http
        assert await supervisor._healthy(await supervisor.store.session(sid)) is True
