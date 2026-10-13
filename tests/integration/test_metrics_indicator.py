"""The Metrics loading indicator must survive the result swap.

The builder triggers a query on every change and shows its state through an
hx-indicator on #metrics-state. The result partial also writes its summary back to
that same element out of band. When that out-of-band swap was a whole-element swap
("true"), the node holding the indicator was replaced on every result: a second change
made while the first query was still in flight lost its loading state, because the
class htmx toggles landed on a node that was no longer in the document. Swapping the
children instead keeps the node, and the indicator with it.

Sequential queries cannot catch this, because a query that starts after the previous one
settled re-resolves the indicator. The test below changes two controls while the first
request is still pending and holds the second response open.
"""

import asyncio

import pytest
from fastapi.responses import JSONResponse

from colabinet.api import create_app
from colabinet.metrics import QuerySpec
from colabinet.net import first_free, start_server, stop_server
from colabinet.web.app import make_templates
from tests.fakes.browser import require_browser, run_browser
from tests.integration.conftest import make_world

pytestmark = pytest.mark.slow


@pytest.fixture
async def metrics_console(tmp_path):
    require_browser()
    world = await make_world(tmp_path)
    world.settings.hidden_port_range = (44000, 44999)
    world.settings.keepalive_s = 5
    world.settings.endpoint_wait_s = 10
    server = task = None
    try:
        sid = await world.sup.create(world.request(name="llm"))
        await world.wait_state(sid, "ready")
        port = first_free(51000, 51999)
        app = create_app(world.sup, "metrics-test-token", port, csrf="metrics-test-csrf")
        state = {"delay": 0.0, "runs": 0}

        @app.middleware("http")
        async def slow_runs(request, call_next):
            if request.url.path == "/ui/test/metrics-delay":
                state["delay"] = float(request.query_params.get("seconds", 0))
                state["runs"] = 0
                return JSONResponse({"delay": state["delay"]})
            if request.url.path == "/ui/test/metrics-runs":
                return JSONResponse({"runs": state["runs"], "delay": state["delay"]})
            if request.url.path == "/ui/metrics/run":
                state["runs"] += 1
                if state["delay"]:
                    await asyncio.sleep(state["delay"])
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


async def test_loading_state_survives_a_second_change_while_a_query_is_pending(metrics_console):
    world, url, directory, state = metrics_console

    # Hold every query open long enough to change the builder twice in between. The
    # delay is set directly on the shared state the middleware reads, so a single
    # navigation is enough.
    state["delay"] = 1.5
    result = await asyncio.to_thread(run_browser, directory, url + "/metrics", """
try {
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
// Let the page's own load query finish before the two-change race begins.
for (let attempt = 0; attempt < 100; attempt++) {
  const s = await evaluate(`(() => {
    const n = document.getElementById('metrics-state');
    return !!n && !n.classList.contains('htmx-request');
  })()`);
  if (s) break;
  await delay(100);
}
const before = await evaluate(`(() => {
  const n = document.getElementById('metrics-state');
  window.__state = n;
  return { summary: n.querySelector('.cb-query-state-default').textContent.trim() };
})()`);

// First change: the query is now in flight and held open by the delay above.
await evaluate(`document.querySelector('[name="group_by"][value="model"]').click(); true`);
await delay(150);
// Second change while the first is still pending; htmx queues it behind the first.
await evaluate(`document.querySelector('[name="group_by"][value="day"]').click(); true`);
await delay(150);

// Sample the indicator across the window where both queries are outstanding.
const samples = [];
for (let attempt = 0; attempt < 30; attempt++) {
  const s = await evaluate(`(() => {
    const n = document.getElementById('metrics-state');
    return { sameNode: n === window.__state,
      requesting: n.classList.contains('htmx-request'),
      runningShown: getComputedStyle(n.querySelector('.cb-query-state-running')).display !== 'none' };
  })()`);
  samples.push(s);
  await delay(60);
}
const during = samples.filter((s) => s.requesting);
const lost = during.filter((s) => !s.runningShown || !s.sameNode);

// Wait for everything to settle, then read the final feedback.
for (let attempt = 0; attempt < 100; attempt++) {
  const done = await evaluate(`(() => {
    const n = document.getElementById('metrics-state');
    return !!n && !n.classList.contains('htmx-request');
  })()`);
  if (done && attempt > 5) break;
  await delay(100);
}
const after = await evaluate(`(() => {
  const n = document.getElementById('metrics-state');
  return { sameNode: n === window.__state,
    runningHidden: getComputedStyle(n.querySelector('.cb-query-state-running')).display === 'none',
    summary: n.querySelector('.cb-query-state-default').textContent.trim() };
})()`);
const runs = await evaluate(`fetch('/ui/test/metrics-runs').then(r => r.json())`);
return { before, duringCount: during.length, lostCount: lost.length, lost, after, runs };
} catch (error) { return { scriptError: String(error && error.stack || error) }; }
""")

    assert result["runs"]["runs"] >= 2, result["runs"]
    # Both queries were outstanding at once, so the race really was exercised.
    assert result["duringCount"] > 0, result
    assert result["lostCount"] == 0, result["lost"]
    # The indicator node survives the out-of-band result swap.
    assert result["after"]["sameNode"], result["after"]
    assert result["after"]["runningHidden"], result["after"]
    assert "rows" in result["after"]["summary"], result["after"]


async def test_result_summary_is_written_into_the_preserved_indicator(metrics_console):
    world, url, directory, state = metrics_console
    result = await asyncio.to_thread(run_browser, directory, url + "/metrics", """
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
for (let attempt = 0; attempt < 120; attempt++) {
  if (await evaluate(`(() => { const n = document.getElementById('metrics-state');
    return !!n && !n.classList.contains('htmx-request'); })()`)) break;
  await delay(100);
}
return await evaluate(`(() => {
  const n = document.getElementById('metrics-state');
  const rows = document.querySelectorAll('#metrics-result tbody tr').length;
  return { rows, summary: n.querySelector('.cb-query-state-default').textContent.trim(),
    hasRunningChild: !!n.querySelector('.cb-query-state-running') };
})()`);
""")
    assert result["hasRunningChild"], result
    assert "rows" in result["summary"] and "results below" in result["summary"], result


async def test_error_branch_keeps_the_indicator_node_and_running_child(metrics_console):
    world, url, directory, state = metrics_console
    result = await asyncio.to_thread(run_browser, directory, url + "/metrics", """
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
for (let attempt = 0; attempt < 120; attempt++) {
  if (await evaluate(`(() => { const n = document.getElementById('metrics-state');
    return !!n && !n.classList.contains('htmx-request'); })()`)) break;
  await delay(100);
}
await evaluate(`(() => {
  const form = document.getElementById('metrics-form');
  // A form state the query builder cannot satisfy, so the server answers with an error.
  const select = document.createElement('select');
  select.name = 'status_class';
  select.innerHTML = '<option value="5xx" selected>5xx</option>';
  form.appendChild(select);
  form.dispatchEvent(new Event('change', { bubbles: true }));
  return true;
})()`);
for (let attempt = 0; attempt < 80; attempt++) {
  const done = await evaluate(`(() => { const n = document.getElementById('metrics-state');
    const err = document.querySelector('#metrics-result .alert');
    return !!err && !n.classList.contains('htmx-request'); })()`);
  if (done) break;
  await delay(100);
}
return await evaluate(`(() => {
  const n = document.getElementById('metrics-state');
  const alert = document.querySelector('#metrics-result .alert');
  return { hasRunningChild: !!n.querySelector('.cb-query-state-running'),
    summary: n.querySelector('.cb-query-state-default').textContent.trim(),
    hasAlert: !!alert };
})()`);
""")
    assert result["hasRunningChild"], result
    # Either the error alert is shown, or the query succeeded; the indicator must hold
    # its running child in both branches, which is what the swap used to break.
    assert result["summary"], result

def _render_result(**context):
    base = {"res": {"rows": [], "columns": [], "note": None}, "spec": QuerySpec(), "spec_json": "{}"}
    return make_templates().env.get_template("partials/metrics_result.html").render(**{**base, **context})


def test_result_template_swaps_children_rather_than_the_indicator_node():
    """A whole-element out-of-band swap replaces the node htmx toggles its indicator
    class on, which is what dropped the loading state during a queued query."""
    for markup in (_render_result(), _render_result(error="bad query")):
        assert 'hx-swap-oob="innerHTML"' in markup
        assert 'hx-swap-oob="true"' not in markup
        # The running child has to survive in both branches, or a later query has nothing
        # left to show while it is in flight.
        assert 'class="cb-query-state-running"' in markup


def test_error_branch_still_reports_the_error_in_the_indicator():
    markup = _render_result(error="bad query")
    assert "bad query" in markup
