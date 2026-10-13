"""Every console page and partial renders against a seeded world; actions need the CSRF token."""

import asyncio
import html
import json
import re

import httpx
import pytest

from colabinet.api import create_app
from tests.integration.conftest import make_world

pytestmark = pytest.mark.slow
CSRF, TOKEN = "csrf-test-token", "api-test-token"


@pytest.fixture
async def console(tmp_path):
    world = await make_world(tmp_path)
    try:
        sid = await world.sup.create(world.request())
        row = await world.wait_state(sid, "ready")
        async with httpx.AsyncClient() as client:
            await client.post(f"http://127.0.0.1:{row['proxy_port']}/v1/chat/completions",
                              json={"stream": True, "x_tokens": 4}, headers={"User-Agent": "pi/0.99"})
        app = create_app(world.sup, TOKEN, 7860, csrf=CSRF)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                    base_url="http://127.0.0.1:7860") as client:
            yield world, client, sid
    finally:
        await world.sup.shutdown()
        world.fake.kill_all()
        world.store.db.close()


PAGES = {"/": ["Overview", "Sessions", "CU per day", "Activity"],
         "/sessions": ["Live", "History"], "/new": ["1 · GPU", "Create…", "L4"],
         "/playground": ["Playground", "Send"], "/metrics": ["Builder", "SQL (read-only)", "Group by"],
         "/models": ["Qwen3.8-27B abliterated Q4_K", "fit (preset defaults)", "EXL3"],
         "/settings": ["Daily CU cap", "GPU rates", "0.7.4"]}


async def test_pages_render(console):
    world, client, sid = console
    for path, needles in PAGES.items():
        response = await client.get(path)
        assert response.status_code == 200, (path, response.text[:300])
        for needle in needles:
            assert needle in response.text, (path, needle)
        assert f'"X-CSRF-Token": "{CSRF}"' in response.text
    response = await client.get(f"/sessions/{sid}")
    assert response.status_code == 200 and "http://127.0.0.1:" in response.text and "pi provider JSON" in response.text
    for tab in ("runs", "requests", "logs"):
        assert (await client.get(f"/sessions/{sid}?tab={tab}")).status_code == 200
    assert (await client.get("/sessions/999")).status_code == 404


async def test_partials_render(console):
    world, client, sid = console
    for part in ("kpis", "attention", "sessions", "activity", "chart"):
        assert (await client.get(f"/ui/overview/{part}")).status_code == 200
    response = await client.get("/ui/sessions/table?tab=live&q=llm")
    assert "llm" in response.text and "ready" in response.text
    response = await client.get(f"/ui/sessions/{sid}/requests")
    assert "pi/0.99" in response.text and "engine" in response.text
    for part in ("status", "runs", "logs"):
        assert (await client.get(f"/ui/sessions/{sid}/{part}")).status_code == 200
    assert "Balance" in (await client.get("/ui/sidebar-status")).text
    response = await client.get("/ui/overview/chart")
    assert "data-cb-chart" in response.text and "L4" in response.text


async def test_actions_need_csrf(console):
    world, client, sid = console
    assert (await client.post(f"/ui/actions/sessions/{sid}/keep")).status_code == 403
    response = await client.post(f"/ui/actions/sessions/{sid}/keep", headers={"X-CSRF-Token": CSRF})
    assert response.status_code == 204 and "idle timer reset" in response.headers["HX-Trigger"]
    response = await client.post(f"/ui/actions/sessions/{sid}/keep",
                     headers={"X-CSRF-Token": CSRF, "Origin": "https://x.example"})
    assert response.status_code == 403


async def test_estimate_and_create_validation(console):
    world, client, _ = console
    headers = {"X-CSRF-Token": CSRF}
    preset = "qwen38-27b-abl-q4k"
    response = await client.post("/ui/new/estimate", data={"gpu": "L4", "preset": preset, "idle_min": "30"},
        headers=headers)
    assert response.status_code == 200 and "Fit on L4: Tight on VRAM" in response.text and "1.54 CU/h" in response.text
    response = await client.post("/ui/new/estimate", data={"gpu": "L4", "preset": preset, "kv": "f16",
        "idle_min": "0"}, headers=headers)
    assert "CPU offload required" in response.text and "Nothing will stop this session" in response.text
    response = await client.post("/ui/actions/create", data={"name": "llm2", "gpu": "L4", "preset": preset},
        headers=headers)
    assert response.status_code == 409 and "confirm" in response.json()["error"]
    response = await client.post("/ui/actions/create", data={"name": "llm", "gpu": "L4", "preset": preset,
        "confirm": "yes"},
                     headers=headers)
    err = response.json()["error"]
    assert response.status_code == 409 and ("in use" in err or "listed" in err)


async def test_custom_engine_review_uses_its_own_kv_default(console):
    _, client, _ = console
    for engine, cache in [("llamacpp", "f16"), ("exllamav3", "FP16")]:
        response = await client.post("/ui/new/estimate", headers={"X-CSRF-Token": CSRF},
                                     data={"preset": "custom", "engine": engine, "gpu": "L4"})
        assert response.status_code == 200
        assert f"KV cache: {cache}" in response.text


async def test_exl3_review_explains_gpu_requirement_and_vram_limits(console):
    _, client, _ = console
    for gpu, expected in [("CPU", "GPU required"), ("T4", "Exceeds VRAM")]:
        response = await client.post("/ui/new/estimate", headers={"X-CSRF-Token": CSRF},
                                     data={"preset": "qwen38-27b-abl-exl3-4bpw", "gpu": gpu})
        assert response.status_code == 200
        assert expected in response.text
        assert "llama.cpp will offload" not in response.text and "runs on the CPU" not in response.text


async def test_metrics_run_sql_save_export(console):
    world, client, _ = console
    headers = {"X-CSRF-Token": CSRF}
    form = {"time": "30d", "group_by": ["gpu", "model_kv"],
            "measures": ["requests", "gen_tps_p50", "prefill_tps_p90", "cu"],
            "chart": "bar", "exclude_playground": "on", "status_class": "all"}
    response = await client.post("/ui/metrics/run", data=form, headers=headers)
    assert response.status_code == 200 and "L4" in response.text and "data-cb-chart" in response.text
    spec = re.search(r"spec=([^\"]+)\"", response.text).group(1)
    csv = await client.get(f"/ui/metrics/export.csv?spec={spec}")
    assert csv.status_code == 200 and csv.text.startswith("gpu,model_kv,requests")
    assert "prefill_tps_p90" in csv.text.splitlines()[0]
    response = await client.post("/ui/metrics/save", data={**form, "name": "speed"}, headers=headers)
    assert response.status_code == 204
    assert json.loads(response.headers["HX-Location"])["path"].startswith("/metrics?saved=")
    assert "speed" in (await client.get("/metrics")).text
    response = await client.post("/ui/metrics/sql", data={"sql": "SELECT COUNT(*) AS n FROM requests"}, headers=headers)
    assert response.status_code == 200 and "<td>1</td>" in response.text
    response = await client.post("/ui/metrics/sql", data={"sql": "DELETE FROM requests"}, headers=headers)
    assert "SQL error" in response.text


async def test_settings_save_and_toast_feed(console):
    world, client, _ = console
    headers = {"X-CSRF-Token": CSRF}
    form = {"daily_cap_cu": "6", "low_balance_cu": "10", "idle_min_default": "25", "eligible_gpus": "L4,A100"}
    response = await client.post("/ui/actions/settings", data=form, headers=headers)
    assert response.status_code == 200 and (await world.sup.settings_values())["daily_cap_cu"] == 6.0
    # The save answers with the refreshed command palette as an out-of-band swap, so the page
    # in front of the operator stops offering a class it just disallowed.
    assert 'id="cmdk-gpus"' in response.text and 'hx-swap-oob="outerHTML"' in response.text
    assert set(re.findall(r'href="/new\?gpu=([^"]+)"', response.text)) == {"CPU", "L4", "A100", "A100-80"}
    page = (await client.get("/settings")).text
    assert 'name="daily_cap_cu" type="number" step="0.5" min="0" value="6.0"' in page
    first = (await client.get("/ui/toasts?after=-1")).json()
    assert first["toasts"] == [] and first["last"] > 0
    feed = (await client.get("/ui/toasts?after=0")).json()
    assert any("is ready" in cursor["title"] for cursor in feed["toasts"])


async def test_gpu_checkboxes_save_multiple_selections_and_allow_none(console):
    world, client, _ = console
    headers = {"X-CSRF-Token": CSRF}
    response = await client.post("/ui/actions/settings", data={"eligible_gpus": ["L4", "A100"]},
                                 headers=headers)
    assert response.status_code == 200
    assert re.findall(r'href="/new\?gpu=([^"]+)"', response.text) == ["CPU", "L4", "A100", "A100-80"]
    assert (await world.sup.settings_values())["eligible_gpus"] == "L4,A100"
    page = (await client.get("/settings")).text
    checkboxes = re.findall(r'<input[^>]*type="checkbox"[^>]*name="eligible_gpus"[^>]*>', page)
    assert len(checkboxes) == 4
    checked = {re.search(r'value="([^"]+)"', field).group(1) for field in checkboxes if "checked" in field}
    assert checked == {"L4", "A100"}
    page = (await client.get("/new")).text
    assert set(re.findall(r'name="gpu" value="([^"]+)"', page)) == {"CPU", "L4", "A100", "A100-80"}

    response = await client.post("/ui/actions/settings", data={"idle_min_default": "25"}, headers=headers)
    assert response.status_code == 200
    # Every family unchecked is a real state: only the CPU fallback is left, in the palette too.
    assert re.findall(r'href="/new\?gpu=([^"]+)"', response.text) == ["CPU"]
    assert (await world.sup.settings_values())["eligible_gpus"] == ""
    page = (await client.get("/settings")).text
    assert all("checked" not in field for field in re.findall(
        r'<input[^>]*type="checkbox"[^>]*name="eligible_gpus"[^>]*>', page))
    page = (await client.get("/new")).text
    assert re.findall(r'name="gpu" value="([^"]+)"', page) == ["CPU"]
    assert re.search(r'name="gpu" value="CPU"[^>]*checked', page)


async def test_playground_relays_through_the_proxy(console):
    world, client, sid = console
    headers = {"X-CSRF-Token": CSRF}
    body = {"session_id": sid, "prompt": "hi", "stream": False, "max_tokens": 5}
    response = await client.post("/ui/playground/send", json=body, headers=headers)
    assert response.status_code == 200 and response.json()["status"] == 200
    # The proxy records the request just after relaying the last byte, so the row can land a moment later.
    for _ in range(50):
        rows = await world.store.db.fetchall("SELECT client FROM requests WHERE client = 'colabinet-playground'")
        if rows:
            break
        await asyncio.sleep(0.05)
    assert rows
    response = await client.get(f"/ui/playground/models?session_id={sid}")
    assert response.json()["body"]["data"][0]["id"]


async def test_every_compute_action_requires_confirmation(console, monkeypatch):
    from unittest.mock import AsyncMock

    world, client, sid = console
    methods = {name: AsyncMock(return_value=42) for name in ("create", "stop", "decide", "reconfigure",
        "stop_external")}
    for name, method in methods.items():
        monkeypatch.setattr(world.sup, name, method)
    actions = [("/ui/actions/create", "/api/sessions", {"name": "approved", "gpu": "L4",
        "preset": "qwen38-27b-abl-q4k"}),
               ("/ui/actions/stop-all", "/api/sessions/stop-all", {}),
               ("/ui/actions/external/external/stop", "/api/external/external/stop", {})]
    actions += [(f"/ui/actions/sessions/{sid}/{action}", f"/api/sessions/{sid}/{action}", {})
                for action in ("stop", "gpu-continue", "retry-setup", "reconfigure", "restart-engine")]
    for ui_path, api_path, fields in actions:
        for confirmation in (None, False, "false"):
            data = dict(fields)
            if confirmation is not None:
                data["confirm"] = confirmation
            for path, keyword in ((ui_path, "data"), (api_path, "json")):
                response = await client.post(path, **{keyword: data}, headers={"X-CSRF-Token": CSRF})
                assert response.status_code in (409, 422), (path, confirmation, response.text)
                assert not any(method.called for method in methods.values()), path
        response = await client.post(ui_path, data={**fields, "confirm": "yes"}, headers={"X-CSRF-Token": CSRF})
        assert response.status_code == 204, (ui_path, response.text)
        response = await client.post(api_path, json={**fields, "confirm": True}, headers={"X-Colabinet-Token": TOKEN})
        assert response.status_code == 200, (api_path, response.text)
        assert any(method.called for method in methods.values()), api_path
        for method in methods.values():
            method.reset_mock()


async def test_console_rejects_bad_host_and_wrong_token(console):
    _, client, sid = console
    path = f"/ui/actions/sessions/{sid}/keep"
    assert (await client.post(path, headers={"X-CSRF-Token": "wrong"})).status_code == 403
    assert (await client.post(path, headers={"X-CSRF-Token": CSRF, "Host": "evil:7860"})).status_code == 403
    assert (await client.get("/", headers={"Host": "evil:7860"})).status_code == 403


async def test_json_api_metrics_settings_and_saved_queries(console):
    world, client, _ = console
    headers = {"X-Colabinet-Token": TOKEN}
    spec = {"time": {"preset": "30d"}, "group_by": ["gpu"], "measures": ["requests", "gen_tps_p50"]}
    response = await client.post("/api/metrics/query", json=spec, headers=headers)
    assert response.status_code == 200 and "L4" in response.text
    response = await client.post("/api/metrics/sql", json={"sql": "SELECT COUNT(*) AS total FROM requests"},
        headers=headers)
    assert response.status_code == 200 and response.json()["rows"] == [[1]]
    assert (await client.post("/api/metrics/sql", json={"sql": "DELETE FROM requests"},
        headers=headers)).status_code == 409
    assert (await client.post("/api/metrics/query", json={"filters": []}, headers=headers)).status_code == 409
    response = await client.post("/api/saved-queries", json={"name": "api-speed", "spec": spec}, headers=headers)
    assert response.status_code == 200
    query_id = response.json()["id"]
    assert "api-speed" in (await client.get("/api/saved-queries")).text
    assert (await client.delete(f"/api/saved-queries/{query_id}", headers=headers)).status_code == 200
    assert not await world.store.saved_queries()
    import json

    response = await client.get("/api/metrics/export.csv", params={"spec": json.dumps(spec)})
    assert response.status_code == 200 and response.text.startswith("gpu,requests,gen_tps_p50")
    assert (await client.post("/api/settings", json={"idle_min_default": 22}, headers=headers)).status_code == 200
    assert (await client.get("/api/settings")).json()["idle_min_default"] == 22


async def test_toast_cursor_does_not_skip_a_large_batch(console):
    world, client, sid = console
    cursor = (await client.get("/ui/toasts")).json()["last"]
    for _ in range(25):
        await world.store.event(world.sup.now(), "state", session_id=sid, to="ready")
    first = (await client.get(f"/ui/toasts?after={cursor}")).json()
    second = (await client.get(f"/ui/toasts?after={first['last']}")).json()
    assert len(first["toasts"]) + len(second["toasts"]) == 25


async def test_log_cursor_returns_only_new_lines(console):
    world, client, sid = console
    first = (await client.get(f"/ui/sessions/{sid}/logs?after=0"))
    assert first.status_code == 200 and "X-Log-Cursor" in first.headers
    response = await client.get(f"/ui/sessions/{sid}/logs?after={first.headers['X-Log-Cursor']}")
    assert response.status_code == 200 and "beforeend:#session-log-lines" not in response.text
    assert response.headers["X-Log-Cursor"] == first.headers["X-Log-Cursor"]


async def test_streaming_playground_and_invalid_input(console):
    world, client, sid = console
    for path in ("/api/playground/send", "/ui/playground/send"):
        response = await client.post(path, json={"session_id": sid, "prompt": "hello", "stream": True, "max_tokens": 2},
                                     headers={"X-CSRF-Token": CSRF})
        assert response.status_code == 200 and "data: [DONE]" in response.text
        assert (await client.post(path, json={"session_id": []},
            headers={"X-CSRF-Token": CSRF})).status_code in (409, 422)
    rows = await world.store.db.fetchall("SELECT client FROM requests WHERE client = 'colabinet-playground'")
    assert len(rows) == 2


async def test_empty_pages_selection_and_invalid_estimate(tmp_path):
    world = await make_world(tmp_path)
    try:
        app = create_app(world.sup, TOKEN, 7860, csrf=CSRF)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                    base_url="http://127.0.0.1:7860") as client:
            for path in PAGES:
                assert (await client.get(path)).status_code == 200
            preset_id = "qwen38-27b-abl-exl3-4bpw"
            html = (await client.get(f"/new?gpu=L4&preset={preset_id}")).text
            assert len(re.findall(r'type="radio"[^>]*checked', html)) == 1
            assert re.search(r'type="radio"[^>]*value="L4"[^>]*checked', html)
            assert re.search(r'<option[^>]*value="' + preset_id + r'"[^>]*selected', html)
            response = await client.post("/ui/new/estimate", data={"ctx": "invalid"},
                                         headers={"X-CSRF-Token": CSRF})
            assert response.status_code == 409
    finally:
        await world.sup.shutdown()
        world.fake.kill_all()
        world.store.db.close()


async def test_playground_closes_upstream_when_iterator_is_cancelled():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    from colabinet.web.services import send_playground

    class Stream(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            yield b"data: first\n\n"
            yield b"data: second\n\n"

        async def aclose(self):
            self.closed = True

    stream = Stream()
    upstream = httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"})
    http = SimpleNamespace(build_request=Mock(return_value=httpx.Request("POST", "http://127.0.0.1:3080")),
                           send=AsyncMock(return_value=upstream))
    store = SimpleNamespace(session=AsyncMock(return_value={"id": 1, "state": "ready", "proxy_port": 3080}),
                            current_run=AsyncMock(return_value={"alias": "model"}))
    response = await send_playground(SimpleNamespace(http=http, store=store), {"session_id": 1})
    assert await anext(response.body_iterator) == b"data: first\n\n"
    await response.body_iterator.aclose()
    assert stream.closed


async def test_malformed_json_is_rejected_before_new_api_operations(console, monkeypatch):
    from unittest.mock import AsyncMock

    from colabinet.web import services

    world, _, _ = console
    operations = [AsyncMock() for _ in range(5)]
    monkeypatch.setattr(world.sup, "update_settings", operations[0])
    for name, operation in zip(("query_metrics", "query_sql", "save_query", "send_playground"), operations[1:]):
        monkeypatch.setattr(services, name, operation)
    app = create_app(world.sup, TOKEN, 7860, csrf=CSRF)
    endpoints = ["/api/settings", "/api/metrics/query", "/api/metrics/sql", "/api/saved-queries",
                 "/api/playground/send", "/ui/playground/send"]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
                                base_url="http://127.0.0.1:7860") as client:
        for content in (b"{broken", b"", b"\xff"):
            responses = [await client.post(endpoint, content=content,
                                           headers={"X-CSRF-Token": CSRF, "Content-Type": "application/json"})
                         for endpoint in endpoints]
            assert [response.status_code for response in responses] == [409] * len(endpoints)
            assert all(response.json()["error"] == "invalid JSON body" for response in responses)
    assert not any(operation.called for operation in operations)


@pytest.mark.parametrize("entrypoint", ["api", "form"])
@pytest.mark.parametrize("custom", [False, True])
async def test_strata_create_projection_reconfigure_restart_and_review(console, entrypoint, custom):
    from colabinet.net import first_free
    world, client, _ = console
    fields = {"name": f"strata-{entrypoint}", "gpu": "L4", "preset": "qwen38-flash-next-strata-iq2-xs",
              "vm_port": first_free(30000, 30999), "refusal_projection": True, "confirm": True}
    if custom:
        fields.update(preset="custom" if entrypoint == "form" else None, engine="strata",
            repo="ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
            file="IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf")
    if entrypoint == "api":
        response = await client.post("/api/sessions", json=fields, headers={"X-Colabinet-Token": TOKEN})
        assert response.status_code == 200, response.text
        sid = response.json()["id"]
    else:
        form = {key: str(value) for key, value in fields.items()}
        form.update(confirm="yes", refusal_projection="true")
        review = await client.post("/ui/new/estimate", data=form, headers={"X-CSRF-Token": CSRF})
        assert review.status_code == 200
        response = await client.post("/ui/actions/create", data=form, headers={"X-CSRF-Token": CSRF})
        assert response.status_code == 204, response.text
        sid = (await world.store.db.fetchone("SELECT id FROM sessions WHERE colab_name = ?", (fields["name"],)))["id"]
    await world.wait_state(sid, "ready")
    first = await world.store.current_run(sid)
    spec = json.loads(first["spec_json"])
    assert (spec["engine"], spec["vision"], spec["refusal_projection"], spec["kv"], spec["ctx"]) == (
        "strata", True, True, "int8", 131072)
    if entrypoint == "api":
        response = await client.post(f"/api/sessions/{sid}/reconfigure",
            json={"confirm": True, "refusal_projection": False}, headers={"X-Colabinet-Token": TOKEN})
    else:
        response = await client.post(f"/ui/actions/sessions/{sid}/reconfigure",
            data={"confirm": "yes", "refusal_projection": "false"}, headers={"X-CSRF-Token": CSRF})
    assert response.status_code in (200, 204), response.text
    await world.wait_state(sid, "ready")
    second = await world.store.current_run(sid)
    assert second["id"] != first["id"] and json.loads(second["spec_json"])["refusal_projection"] is False
    response = await client.post(f"/api/sessions/{sid}/restart-engine", json={"confirm": True},
                                headers={"X-Colabinet-Token": TOKEN})
    assert response.status_code == 200
    await world.wait_state(sid, "ready")
    restarted = await world.store.current_run(sid)
    assert restarted["id"] != second["id"]
    assert json.loads(restarted["spec_json"])["refusal_projection"] is False


def _gpu_inputs(page):
    """The New session GPU radios as {class: (checked, disabled)}."""
    return {gpu: ("checked" in attrs, "disabled" in attrs)
            for gpu, attrs in re.findall(r'name="gpu" value="([A-Za-z0-9-]+)"([^>]*)>', page)}


async def test_new_session_greys_out_classes_that_cannot_run_the_model(console):
    _, client, _ = console
    inputs = _gpu_inputs((await client.get("/new?preset=qwen38-27b-exl3-4bpw")).text)
    assert inputs["L4"] == (True, False)
    assert {gpu for gpu, (_, disabled) in inputs.items() if disabled} == {"CPU", "T4", "T4-HIGHMEM"}
    # Launching a preset that L4 cannot hold starts on the cheapest class that can.
    inputs = _gpu_inputs((await client.get("/new?preset=qwen38-flash-next-abl-q4k-xl")).text)
    assert [gpu for gpu, (checked, _) in inputs.items() if checked] == ["A100"]
    assert inputs["L4"] == (False, True)


async def test_new_session_keeps_an_explicit_class_and_offers_one_that_fits(console):
    _, client, _ = console
    # A class asked for by name stays selected and submittable, even when it cannot run the model.
    inputs = _gpu_inputs((await client.get("/new?gpu=CPU&preset=qwen38-27b-q4km")).text)
    assert inputs["CPU"] == (True, False)
    response = await client.post("/ui/new/estimate", headers={"X-CSRF-Token": CSRF},
                                 data={"preset": "qwen38-27b-q4km", "gpu": "CPU"})
    assert response.status_code == 200 and "CPU cannot run this configuration" in response.text
    marker = re.search(r'id="gpu-fit" hidden data-fit="([^"]*)" data-suggest="([^"]*)"', response.text)
    fit = json.loads(html.unescape(marker.group(1)))
    assert marker.group(2) == "L4"
    assert fit["CPU"]["status"] == "blocked" and fit["L4"]["status"] != "blocked"
    response = await client.post("/ui/new/estimate", headers={"X-CSRF-Token": CSRF},
                                 data={"preset": "qwen38-27b-q4km", "gpu": "L4"})
    assert 'data-suggest=""' in response.text and "cannot run this configuration" not in response.text


async def test_measured_speeds_show_on_gpu_choices_review_and_models(console):
    _, client, _ = console
    page = (await client.get("/new?preset=qwen38-27b-abl-q4k")).text
    speeds = dict(re.findall(r'name="gpu" value="([A-Za-z0-9-]+)".*?class="cb-gpu-speed"[^>]*>([^<]*)<', page, re.S))
    assert speeds["L4"] == "13.1 tok/s measured" and speeds["A100"] == "43.0 tok/s measured"
    assert speeds["CPU"] == ""  # killed for memory: nothing measured, and greyed out
    response = await client.post("/ui/new/estimate", headers={"X-CSRF-Token": CSRF},
                                 data={"preset": "qwen38-27b-abl-q4k", "gpu": "L4"})
    assert "<dt>Measured</dt>" in response.text
    assert ("<strong>13.1 tok/s</strong> generation, 660 tok/s prefill on a 4K-token prompt, on L4 at the "
            "preset's defaults (2026-10-13)") in response.text
    marker = re.search(r'id="gpu-fit" hidden data-fit="([^"]*)"', response.text)
    fit = json.loads(html.unescape(marker.group(1)))
    assert fit["G4"]["measured"]["gen"] == "67.8" and fit["CPU"]["measured"] is None
    # A greyed-out class keeps its measurement: it shows why the class is unsuitable.
    response = await client.post("/ui/new/estimate", headers={"X-CSRF-Token": CSRF},
                                 data={"preset": "qwen38-flash-next-abl-q4k-xl", "gpu": "A100"})
    fit = json.loads(html.unescape(re.search(r'data-fit="([^"]*)"', response.text).group(1)))
    assert fit["L4"]["status"] == "blocked" and fit["L4"]["measured"]["gen"] == "3.0"
    models = (await client.get("/models")).text
    assert "Measured <strong>13.1</strong> tok/s<br>Prefill 660 tok/s" in models
    assert "Measured: engine-reported generation speed" in models


async def test_exllamav3_create_on_cpu_is_refused(console):
    _, client, _ = console
    response = await client.post("/ui/actions/create", headers={"X-CSRF-Token": CSRF}, data={
        "name": "exl-cpu", "gpu": "CPU", "preset": "qwen38-27b-exl3-4bpw", "confirm": "yes"})
    assert response.status_code == 409 and "requires a GPU runtime" in response.json()["error"]
