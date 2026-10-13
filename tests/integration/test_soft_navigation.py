import asyncio
import json

import pytest
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from colabinet.api import create_app
from colabinet.net import first_free, start_server, stop_server
from tests.fakes.browser import require_browser, run_browser
from tests.integration.conftest import make_world

pytestmark = pytest.mark.slow


@pytest.fixture
async def console(tmp_path):
    require_browser()
    world = await make_world(tmp_path)
    world.settings.hidden_port_range = (43000, 43999)
    world.settings.keepalive_s = 5
    world.settings.reconcile_s = 5
    world.settings.usage_s = 5
    world.settings.endpoint_wait_s = 10
    server = task = None
    try:
        sid = await world.sup.create(world.request())
        await world.wait_state(sid, "ready")
        port = first_free(50000, 50999)
        app = create_app(world.sup, "navigation-test-token", port, csrf="navigation-test-csrf")
        state = {"requests": [], "cancelled_streams": 0, "stream_started": 0}

        @app.middleware("http")
        async def observe(request, call_next):
            state["requests"].append({"method": request.method, "path": request.url.path,
                                      "boosted": request.headers.get("HX-Boosted") == "true"})
            if request.query_params.get("delay"):
                await asyncio.sleep(float(request.query_params["delay"]))
            if request.url.path == "/ui/actions/create" and state.get("slow_create"):
                await asyncio.sleep(0.5)
            if request.url.path == "/ui/test/navigation-state":
                return JSONResponse(state)
            if request.url.path == "/ui/test/not-a-page":
                return HTMLResponse("not a console page")
            if request.headers.get("HX-History-Restore-Request") == "true" and state.get("invalid_history"):
                markup = '<main id="cb-view"></main>' if state["invalid_history"] == "empty" else "not a console page"
                return HTMLResponse(markup, status_code=503 if state["invalid_history"] == "http" else 200)
            if request.url.path == "/ui/playground/send" and state.get("slow_stream"):
                async def stream():
                    state["stream_started"] += 1
                    try:
                        for number in range(100):
                            chunk = {"choices": [{"delta": {"content": f"piece {number} "}}]}
                            yield f"data: {json.dumps(chunk)}\n\n"
                            await asyncio.sleep(0.05)
                        yield "data: [DONE]\n\n"
                    finally:
                        state["cancelled_streams"] += 1
                return StreamingResponse(stream(), media_type="text/event-stream")
            return await call_next(request)

        server, task = await start_server(app, port)
        yield world, f"http://127.0.0.1:{port}", tmp_path, state
    finally:
        if server is not None:
            await stop_server(server, task)
        await world.sup.shutdown()
        for row in await world.store.sessions():
            await world.sup.tunnels.stop(row["tunnel_pid"], row["colab_name"])
        world.fake.kill_all()
        world.store.db.close()


NAVIGATION = r"""
const delay = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));
async function settled(path, selector) {
  for (let attempt = 0; attempt < 200; attempt++) {
    const ready = await evaluate(`location.pathname + location.search === ${JSON.stringify(path)} &&
      document.readyState === 'complete' && !!document.querySelector(${JSON.stringify(selector)})`);
    if (ready) { await delay(50); return; }
    await delay(30);
  }
  const state = await evaluate(`(async () => ({url: location.href, title: document.title,
    heading: document.querySelector('.cb-crumbs h1')?.textContent,
    dialog: !!document.getElementById('reconfigure-dialog'),
    requests: await (await fetch('/ui/test/navigation-state')).json()}))()`);
  throw Error(`Navigation did not finish: ${path}: ${JSON.stringify(state)}`);
}
async function click(selector, path, marker) {
  await evaluate(`setTimeout(() => document.querySelector(${JSON.stringify(selector)}).click(), 0); true`);
  await delay(50);
  await settled(path, marker);
}
await evaluate(`window.__shell = { sidebar: document.getElementById('sidebar'),
  main: document.querySelector('main'), toaster: document.getElementById('toaster') }; true`);
"""


async def browse(console, path, script):
    world, url, directory, state = console
    return await asyncio.to_thread(run_browser, directory, url + path, NAVIGATION + script)


async def test_navigation_keeps_the_shell_and_updates_url_title_and_active_link(console):
    result = await browse(console, "/", r"""
await evaluate(`window.__errors = []; addEventListener('error', (event) => window.__errors.push(event.message)); true`);
await click('#sidebar a[href="/sessions"]', '/sessions', '#sessions-table');
await click('#sidebar a[href="/metrics"]', '/metrics', '#metrics-form');
await click('#sidebar a[href="/models"]', '/models', '.cb-crumbs h1');
return await evaluate(`({ sidebarKept: window.__shell?.sidebar === document.getElementById('sidebar'),
  mainKept: window.__shell?.main === document.querySelector('main'),
  toasterKept: window.__shell?.toaster === document.getElementById('toaster'),
  title: document.title, active: document.querySelector('#sidebar a[aria-current="page"]')?.getAttribute('href'),
  documentLoads: performance.getEntriesByType('navigation').length,
  heading: document.querySelector('.cb-crumbs h1').textContent,
  errors: window.__errors,
  headingFocused: document.activeElement === document.querySelector('.cb-crumbs h1') })`);
""")
    assert result == {"sidebarKept": True, "mainKept": True, "toasterKept": True, "title": "Models · colabinet",
                      "active": "/models", "documentLoads": 1, "heading": "Models", "headingFocused": True,
                      "errors": []}


async def test_session_tabs_restart_page_lifecycle_and_reconfigure_dialog(console):
    result = await browse(console, "/sessions?tab=all", r"""
await settled('/sessions?tab=all', '#sessions-table tbody tr');
await click('#sessions-table a[href="/sessions/1?reconfigure=1"]',
  '/sessions/1?reconfigure=1', '#reconfigure-dialog');
const opened = await evaluate(`document.getElementById('reconfigure-dialog').open`);
await evaluate(`document.getElementById('reconfigure-dialog').close(); true`);
await click('a[role="tab"][href="/sessions/1?tab=logs"]', '/sessions/1?tab=logs', '#session-logs');
await evaluate(`Array.from(document.querySelectorAll('button')).find((button) =>
  button.textContent === 'Pause').click(); true`);
await click('#sidebar a[href="/"]', '/', '#ov-kpis');
await click('#sidebar a[href="/sessions"]', '/sessions', '#sessions-table');
await settled('/sessions', '#sessions-table tbody tr');
await click('#sessions-table a[href="/sessions/1"]', '/sessions/1', '#session-watch');
await click('a[role="tab"][href="/sessions/1?tab=logs"]', '/sessions/1?tab=logs', '#session-logs');
return await evaluate(`({ shellKept: !!window.__shell, reconfigureOpened: ${opened},
  pauseReset: Array.from(document.querySelectorAll('button')).some((button) =>
    button.textContent === 'Pause' && button.getAttribute('aria-pressed') === 'false'),
  resourcesPresent: !!document.querySelector('#session-resources .card'),
  eventsLoaded: !!document.querySelector('#session-log-events li') })`);
""")
    assert result == {"shellKept": True, "reconfigureOpened": True, "pauseReset": True,
                      "resourcesPresent": True, "eventsLoaded": True}


async def test_back_and_forward_keep_shell_without_caching_playground_content(console):
    result = await browse(console, "/playground", r"""
await evaluate(`document.getElementById('pg-prompt').value = 'private-navigation-prompt';
  document.getElementById('pg-output').textContent = 'private-navigation-reply'; true`);
await click('#sidebar a[href="/metrics"]', '/metrics', '#metrics-form');
await evaluate(`history.back(); true`);
await settled('/playground', '#pg-prompt');
const returned = await evaluate(`({ shellKept: !!window.__shell,
  promptReset: document.getElementById('pg-prompt').value === 'Say hello in five words.',
  replyReset: document.getElementById('pg-output').textContent === '',
  privateStored: [localStorage, sessionStorage].some((storage) => Object.values(storage).some((value) =>
    value.includes('private-navigation-prompt') || value.includes('private-navigation-reply'))),
  historyCached: [localStorage, sessionStorage].some((storage) => storage.getItem('htmx-history-cache') !== null) })`);
await evaluate(`history.forward(); true`);
await settled('/metrics', '#metrics-form');
return { ...returned, forwardKeptShell: await evaluate(`!!window.__shell`),
  forwardTitle: await evaluate(`document.title`) };
""")
    assert result == {"shellKept": True, "promptReset": True, "replyReset": True, "privateStored": False,
                      "historyCached": False, "forwardKeptShell": True, "forwardTitle": "Metrics · colabinet"}


async def test_playground_examples_initialize_on_soft_navigation_and_history_restore(console):
    world, _, _, _ = console
    row = await world.store.session(1)
    run = await world.store.current_run(1)
    result = await browse(console, "/", r"""
async function examples() {
  return await evaluate(`({ python: document.getElementById('pg-python-example').textContent,
    curl: document.getElementById('pg-curl-example').textContent,
    copyEnabled: !document.getElementById('pg-copy-example').disabled,
    language: document.getElementById('pg-example-language').value,
    pythonVisible: !document.getElementById('pg-python-panel').hidden,
    curlVisible: !document.getElementById('pg-curl-panel').hidden,
    shellKept: window.__shell.main === document.querySelector('main') })`);
}
await click('#sidebar a[href="/playground"]', '/playground', '#pg-python-example');
const first = await examples();
await evaluate(`(() => {
  const prompt = document.getElementById('pg-prompt');
  prompt.value = 'Changed before leaving';
  prompt.dispatchEvent(new Event('input', { bubbles: true }));
  const language = document.getElementById('pg-example-language');
  language.value = 'curl';
  language.dispatchEvent(new Event('change', { bubbles: true }));
})()`);
const edited = await examples();
await click('#sidebar a[href="/metrics"]', '/metrics', '#metrics-form');
await evaluate(`history.back(); true`);
await settled('/playground', '#pg-python-example');
return { first, edited, returned: await examples() };
""")
    for state in (result["first"], result["returned"]):
        assert state["copyEnabled"] is True and state["shellKept"] is True
        assert state["language"] == "python" and state["pythonVisible"] is True and state["curlVisible"] is False
        for language in ("python", "curl"):
            assert f"http://127.0.0.1:{row['proxy_port']}/v1/chat/completions" in state[language]
            assert run["alias"] in state[language]
            assert "Say hello in five words." in state[language]
            assert "Changed before leaving" not in state[language]
    for language in ("python", "curl"):
        assert "Changed before leaving" in result["edited"][language]
    assert result["edited"]["language"] == "curl" and result["edited"]["curlVisible"] is True
    assert result["edited"]["pythonVisible"] is False


async def test_session_controls_work_after_soft_navigation(console):
    result = await browse(console, "/", r"""
await click('#ov-sessions a[href="/sessions/1"]', '/sessions/1', '#session-controls');
await evaluate(`document.getElementById('session-action-stop').click();
  document.getElementById('stop-confirm').click(); true`);
for (let attempt = 0; attempt < 200; attempt++) {
  if (await evaluate(`document.getElementById('session-controls')?.dataset.state === 'stopped'`)) break;
  await delay(30);
}
return await evaluate(`({ shellKept: !!window.__shell,
  state: document.getElementById('session-controls').dataset.state,
  badge: document.getElementById('session-state').textContent.trim(),
  stopRemoved: !document.getElementById('session-action-stop') })`);
""")
    assert result == {"shellKept": True, "state": "stopped", "badge": "stopped", "stopRemoved": True}


async def test_navigation_loading_latest_request_and_errors_keep_current_page(console):
    result = await browse(console, "/", r"""
await evaluate(`document.querySelector('#sidebar a[href="/models"]').href = '/models?delay=0.5';
  htmx.process(document.querySelector('#sidebar a[href="/models?delay=0.5"]'));
  document.querySelector('#sidebar a[href="/models?delay=0.5"]').click(); true`);
await delay(80);
const busy = await evaluate(`!document.getElementById('cb-loading').hidden &&
  document.getElementById('cb-view').getAttribute('aria-busy') === 'true'`);
await click('#sidebar a[href="/metrics"]', '/metrics', '#metrics-form');
await delay(600);
const latest = await evaluate(`location.pathname === '/metrics' &&
  document.querySelector('.cb-crumbs h1').textContent === 'Metrics'`);
await evaluate(`document.querySelector('#sidebar a[href="/sessions"]').href = '/sessions/9999';
  htmx.process(document.querySelector('#sidebar a[href="/sessions/9999"]'));
  document.querySelector('#sidebar a[href="/sessions/9999"]').click(); true`);
await delay(250);
const errorKeptPage = await evaluate(`location.pathname === '/metrics' && !!document.getElementById('metrics-form')`);
await evaluate(`document.querySelector('#sidebar a[href="/settings"]').href = '/ui/test/not-a-page';
  htmx.process(document.querySelector('#sidebar a[href="/ui/test/not-a-page"]'));
  document.querySelector('#sidebar a[href="/ui/test/not-a-page"]').click(); true`);
await delay(250);
return { busy, latest, errorKeptPage, ...await evaluate(`({ shellKept: !!window.__shell,
  invalidKeptPage: location.pathname === '/metrics' && !!document.getElementById('metrics-form'),
  idle: document.getElementById('cb-loading').hidden &&
    !document.getElementById('cb-view').hasAttribute('aria-busy') })`) };
""")
    assert result == {"busy": True, "latest": True, "errorKeptPage": True, "shellKept": True,
                      "invalidKeptPage": True, "idle": True}


async def test_leaving_playground_cancels_stream_without_touching_returned_page(console):
    console[3]["slow_stream"] = True
    result = await browse(console, "/playground", r"""
await evaluate(`window.__errors = []; addEventListener('unhandledrejection', (event) =>
  window.__errors.push(String(event.reason))); document.getElementById('pg-send').click(); true`);
for (let attempt = 0; attempt < 100; attempt++) {
  if (await evaluate(`document.getElementById('pg-output').textContent.includes('piece')`)) break;
  await delay(30);
}
const started = await evaluate(`document.getElementById('pg-output').textContent.includes('piece')`);
await click('#sidebar a[href="/metrics"]', '/metrics', '#metrics-form');
await click('#sidebar a[href="/playground"]', '/playground', '#pg-output');
await delay(300);
return { started, ...await evaluate(`(async () => ({ shellKept: !!window.__shell,
  outputEmpty: document.getElementById('pg-output').textContent === '',
  sendEnabled: !document.getElementById('pg-send').disabled, errors: window.__errors,
  cancelled: (await (await fetch('/ui/test/navigation-state')).json()).cancelled_streams === 1 }))()`) };
""")
    assert result == {"started": True, "shellKept": True, "outputEmpty": True,
                      "sendEnabled": True, "errors": [], "cancelled": True}


async def test_saved_queries_and_preset_reload_use_soft_navigation(console):
    result = await browse(console, "/metrics", r"""
await evaluate(`document.querySelector('[name="name"]').value = 'navigation speed';
  document.querySelector('button[hx-post="/ui/metrics/save"]').click(); true`);
await settled('/metrics?saved=1', '.cb-saved');
const saved = await evaluate(`!!window.__shell && document.querySelector('.cb-saved').value === '1'`);
await click('#sidebar a[href="/models"]', '/models', '.cb-crumbs h1');
await evaluate(`document.querySelector('button[hx-post="/ui/actions/presets/reload"]').click(); true`);
await delay(300);
const reloaded = await evaluate(`!!window.__shell && document.title === 'Models · colabinet'`);
await click('a[href="/new?preset=qwen38-27b-abl-q4k"]', '/new?preset=qwen38-27b-abl-q4k', '#new-form');
const preset = await evaluate(`document.getElementById('preset').value === 'qwen38-27b-abl-q4k' &&
  document.getElementById('new-form').dataset.custom === 'false'`);
await click('#sidebar a[href="/metrics"]', '/metrics', '.cb-saved');
await evaluate(`document.querySelector('.cb-saved').value = '1';
  document.querySelector('.cb-saved').dispatchEvent(new Event('change')); true`);
await settled('/metrics?saved=1', '.cb-saved');
await evaluate(`window.confirm = () => true;
  document.querySelector('button[hx-post="/ui/metrics/saved/1/delete"]').click(); true`);
await settled('/metrics', '#metrics-form');
return { saved, reloaded, preset, ...await evaluate(`({ shellKept: !!window.__shell,
  savedDeleted: !document.querySelector('.cb-saved'),
  sidebarKept: window.__shell?.sidebar === document.getElementById('sidebar') })`) };
""")
    assert result == {"saved": True, "reloaded": True, "preset": True,
                      "shellKept": True, "savedDeleted": True, "sidebarKept": True}


async def test_session_watch_and_polling_stop_on_leave_and_restart_once_on_back(console):
    result = await browse(console, "/sessions/1?tab=logs", r"""
await delay(150);
await click('#sidebar a[href="/metrics"]', '/metrics', '#metrics-form');
await delay(150);
const before = await evaluate(`(async () => (await (await fetch('/ui/test/navigation-state')).json()).requests
  .filter((request) => request.path.startsWith('/ui/sessions/1/')).length)()`);
await delay(2200);
const after = await evaluate(`(async () => (await (await fetch('/ui/test/navigation-state')).json()).requests
  .filter((request) => request.path.startsWith('/ui/sessions/1/')).length)()`);
await evaluate(`history.back(); true`);
await settled('/sessions/1?tab=logs', '#session-logs');
return { pollingStopped: before === after, ...await evaluate(`(async () => ({
  shellKept: !!window.__shell,
  watches: (await (await fetch('/ui/test/navigation-state')).json()).requests
    .filter((request) => request.path === '/ui/sessions/1/watch').length }))()`) };
""")
    assert result == {"pollingStopped": True, "shellKept": True, "watches": 2}


async def test_navigation_supersedes_pending_history_restore(console):
    result = await browse(console, "/", r"""
await evaluate(`document.querySelector('#sidebar a[href="/models"]').href = '/models?delay=0.5';
  htmx.process(document.querySelector('#sidebar a[href="/models?delay=0.5"]')); true`);
await click('#sidebar a[href="/models?delay=0.5"]', '/models?delay=0.5', '.cb-crumbs h1');
await click('#sidebar a[href="/metrics"]', '/metrics', '#metrics-form');
await evaluate(`history.back(); true`);
await delay(80);
const busy = await evaluate(`!document.getElementById('cb-loading').hidden`);
await click('#sidebar a[href="/sessions"]', '/sessions', '#sessions-table');
await delay(600);
return { busy, ...await evaluate(`({ shellKept: !!window.__shell,
  latest: location.pathname === '/sessions' && !!document.getElementById('sessions-table'),
  idle: document.getElementById('cb-loading').hidden })`) };
""")
    assert result == {"busy": True, "shellKept": True, "latest": True, "idle": True}


async def test_command_menu_theme_and_playground_requests_work_after_navigation(console):
    result = await browse(console, "/", r"""
const dark = await evaluate(`basecoat.theme.toggle(); document.documentElement.classList.contains('dark')`);
await evaluate(`document.querySelector('button[aria-label="Search"]').click(); true`);
await click('#cmdk a[href="/new"]', '/new', '#new-form');
await evaluate(`cbNavigate('/new?preset=custom'); true`);
await settled('/new?preset=custom', '#new-form[data-custom="true"]');
const custom = await evaluate(`getComputedStyle(document.getElementById('repo').parentElement).display !== 'none'`);
await click('#sidebar a[href="/playground"]', '/playground', '#pg-send');
await evaluate(`document.getElementById('pg-send').click(); true`);
for (let attempt = 0; attempt < 100; attempt++) {
  if (await evaluate(`document.getElementById('pg-metrics').textContent.includes('HTTP 200')`)) break;
  await delay(30);
}
const streamed = await evaluate(`document.getElementById('pg-output').textContent.length > 0 &&
  document.getElementById('pg-metrics').textContent.includes('HTTP 200')`);
await evaluate(`document.getElementById('pg-stream').checked = false;
  document.getElementById('pg-send').click(); true`);
for (let attempt = 0; attempt < 100; attempt++) {
  if (await evaluate(`!document.getElementById('pg-send').disabled`)) break;
  await delay(30);
}
await evaluate(`cbPlaygroundModels(); true`);
for (let attempt = 0; attempt < 100; attempt++) {
  if (await evaluate(`document.getElementById('pg-raw-out').textContent.includes('"data"')`)) break;
  await delay(30);
}
return { custom, streamed, ...await evaluate(`({ shellKept: !!window.__shell,
  themeKept: document.documentElement.classList.contains('dark') === ${dark},
  commandClosed: !document.getElementById('cmdk').open,
  nonstreamed: !!document.getElementById('pg-output').textContent &&
    document.getElementById('pg-metrics').textContent.includes('HTTP 200'),
  modelsLoaded: document.getElementById('pg-raw-out').textContent.includes('"data"') })`) };
""")
    assert result == {"custom": True, "streamed": True, "shellKept": True, "themeKept": True,
                      "commandClosed": True, "nonstreamed": True, "modelsLoaded": True}
    rows = await console[0].store.db.fetchall("SELECT * FROM requests WHERE client = 'colabinet-playground'")
    assert len(rows) == 3


async def test_create_action_opens_fake_session_without_document_reload(console):
    vm_port = console[0].request().vm_port
    result = await browse(console, "/new", r"""
await evaluate(`document.getElementById('name').value = 'soft-navigation-created';
  document.getElementById('vm_port').value = VM_PORT;
  document.querySelector('button[hx-post="/ui/actions/create"]').click(); true`);
await settled('/sessions/2', '#session-controls');
return await evaluate(`({ shellKept: !!window.__shell,
  heading: document.querySelector('.cb-crumbs h1').textContent,
  commandUpdated: !!document.querySelector('#cmdk a[href="/sessions/2"]'),
  active: document.querySelector('#sidebar a[aria-current="page"]').getAttribute('href') })`);
""".replace("VM_PORT", str(vm_port)))
    assert result == {"shellKept": True, "heading": "soft-navigation-created", "commandUpdated": True,
                      "active": "/sessions"}
    await console[0].wait_state(2, "ready")


async def test_download_links_are_not_boosted(console):
    result = await browse(console, "/sessions/1?tab=logs", r"""
await send('Browser.setDownloadBehavior', { behavior: 'deny' });
const plainLog = await evaluate(`!document.querySelector('a[href="/ui/sessions/1/logs.txt"]')
  .hasAttribute('hx-boost')`);
await evaluate(`document.querySelector('a[href="/ui/sessions/1/logs.txt"]').click(); true`);
await delay(150);
await click('#sidebar a[href="/metrics"]', '/metrics', '#metrics-result a');
const plainCSV = await evaluate(`!document.querySelector('#metrics-result a').hasAttribute('hx-boost')`);
await evaluate(`document.querySelector('#metrics-result a').click(); true`);
await delay(150);
return { plainLog, plainCSV, ...await evaluate(`(async () => ({ shellKept: !!window.__shell,
  stillMetrics: location.pathname === '/metrics' && !!document.getElementById('metrics-form'),
  downloads: (await (await fetch('/ui/test/navigation-state')).json()).requests
    .filter((request) => request.path.endsWith('/logs.txt') || request.path.endsWith('/export.csv')) }))()`) };
""")
    assert result == {"plainLog": True, "plainCSV": True, "shellKept": True, "stillMetrics": True,
                      "downloads": [{"method": "GET", "path": "/ui/sessions/1/logs.txt", "boosted": False},
                                    {"method": "GET", "path": "/ui/metrics/export.csv", "boosted": False}]}


async def test_direct_links_and_native_navigation_work_without_javascript(console):
    result = await browse(console, "/", r"""
const origin = await evaluate(`location.origin`);
await send('Emulation.setScriptExecutionDisabled', { value: true });
await send('Page.navigate', { url: origin + '/sessions/1?tab=logs' });
await settled('/sessions/1?tab=logs', '#session-logs');
const direct = await evaluate(`typeof htmx === 'undefined' && !!document.getElementById('session-controls')`);
await evaluate(`document.querySelector('#sidebar a[href="/models"]').click(); true`);
await settled('/models', '.cb-crumbs h1');
return { direct, ...await evaluate(`({ native: typeof htmx === 'undefined' && !window.__shell,
  heading: document.querySelector('.cb-crumbs h1').textContent })`) };
""")
    assert result == {"direct": True, "native": True, "heading": "Models"}


async def test_leaving_page_unregisters_basecoat_charts(console):
    result = await browse(console, "/", r"""
await settled('/', 'canvas[data-chart-initialized]');
await evaluate(`window.__charts = Array.from(document.querySelectorAll('canvas[data-chart-initialized]')); true`);
await click('#sidebar a[href="/metrics"]', '/metrics', '#metrics-form');
return await evaluate(`({ shellKept: !!window.__shell, chartsExisted: window.__charts.length > 0,
  unregistered: window.__charts.every((canvas) => !Chart.getChart(canvas) && !canvas._destroy &&
    !canvas._basecoatChartConfig && !canvas.hasAttribute('data-chart-initialized')) })`);
""")
    assert result == {"shellKept": True, "chartsExisted": True, "unregistered": True}


@pytest.mark.parametrize("failure", ["malformed", "empty", "http", "offline"])
async def test_failed_history_restore_keeps_current_page_and_clears_loading(console, failure):
    console[3]["invalid_history"] = None if failure == "offline" else failure
    setup = r"""
await send('Network.enable');
await send('Network.emulateNetworkConditions', {
  offline: true, latency: 0, downloadThroughput: 0, uploadThroughput: 0,
});
""" if failure == "offline" else ""
    result = await browse(console, "/models", r"""
await click('#sidebar a[href="/metrics"]', '/metrics', '#metrics-form');
await evaluate(`window.__errors = []; addEventListener('error', (event) => window.__errors.push(event.message)); true`);
SETUP
await evaluate(`history.back(); true`);
await delay(300);
return await evaluate(`({ shellKept: !!window.__shell, pageKept: !!document.getElementById('metrics-form'),
  urlKept: location.pathname === '/metrics', title: document.title, errors: window.__errors,
  idle: document.getElementById('cb-loading').hidden &&
    !document.getElementById('cb-view').hasAttribute('aria-busy') })`);
""".replace("SETUP", setup))
    assert result == {"shellKept": True, "pageKept": True, "urlKept": True, "title": "Metrics · colabinet",
                      "errors": [], "idle": True}


async def test_late_create_response_does_not_override_new_navigation(console):
    console[3]["slow_create"] = True
    vm_port = console[0].request().vm_port
    result = await browse(console, "/new", r"""
await evaluate(`document.getElementById('name').value = 'late-navigation-created';
  document.getElementById('vm_port').value = VM_PORT;
  document.querySelector('button[hx-post="/ui/actions/create"]').click(); true`);
await delay(80);
await click('#sidebar a[href="/metrics"]', '/metrics', '#metrics-form');
await delay(900);
return await evaluate(`({ shellKept: !!window.__shell,
  pageKept: location.pathname === '/metrics' && !!document.getElementById('metrics-form') })`);
""".replace("VM_PORT", str(vm_port)))
    assert result == {"shellKept": True, "pageKept": True}
    assert (await console[0].wait_state(2, "ready"))["colab_name"] == "late-navigation-created"


async def test_preset_reload_replaces_url_without_adding_history(console):
    result = await browse(console, "/models?retained=1", r"""
const entries = await evaluate(`history.length`);
await evaluate(`document.querySelector('button[hx-post="/ui/actions/presets/reload"]').click(); true`);
await delay(300);
return await evaluate(`({ shellKept: !!window.__shell, path: location.pathname + location.search,
  replaced: history.length === ${entries} })`);
""")
    assert result == {"shellKept": True, "path": "/models", "replaced": True}


async def test_history_restore_refreshes_command_menu_sessions(console):
    result = await browse(console, "/models", r"""
await click('#sidebar a[href="/metrics"]', '/metrics', '#metrics-form');
await evaluate(`cbConfirmStop(1, 'llm'); document.getElementById('stop-confirm').click(); true`);
for (let attempt = 0; attempt < 100; attempt++) {
  const stopped = await evaluate(`(async () => !(await (await fetch('/models')).text())
    .includes('href="/sessions/1"'))()`);
  if (stopped) break;
  await delay(30);
}
await evaluate(`history.back(); true`);
await settled('/models', '.cb-crumbs h1');
return await evaluate(`({ shellKept: !!window.__shell,
  staleCommandRemoved: !document.querySelector('#cmdk a[href="/sessions/1"]'),
  noLiveBadge: document.getElementById('nav-live-count').hidden })`);
""")
    assert result == {"shellKept": True, "staleCommandRemoved": True, "noLiveBadge": True}
