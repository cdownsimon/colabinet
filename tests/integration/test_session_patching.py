import asyncio
import json

import pytest
from fastapi.responses import Response

from colabinet.api import create_app
from colabinet.net import first_free, start_server, stop_server
from tests.fakes.browser import require_browser, run_browser
from tests.integration.conftest import make_world

pytestmark = pytest.mark.slow


@pytest.fixture
async def preview(tmp_path):
    require_browser()
    world = await make_world(tmp_path)
    world.settings.hidden_port_range = (41000, 41999)
    world.settings.keepalive_s = 5
    world.settings.reconcile_s = 5
    world.settings.usage_s = 5
    world.settings.endpoint_wait_s = 10
    server = task = None
    try:
        for name in ("llm", "coder"):
            sid = await world.sup.create(world.request(name=name))
            await world.wait_state(sid, "ready")
        port = first_free(48000, 48999)
        app = create_app(world.sup, "patch-test-token", port, csrf="patch-test-csrf")
        action_requests = {"count": 0}

        @app.post("/ui/test/delayed-action")
        async def delayed_action():
            action_requests["count"] += 1
            await asyncio.sleep(1)
            return Response(status_code=204)

        @app.get("/ui/test/action-count")
        async def action_count():
            return action_requests

        server, task = await start_server(app, port)
        yield world, f"http://127.0.0.1:{port}", tmp_path
    finally:
        if server is not None:
            await stop_server(server, task)
        await world.sup.shutdown()
        for row in await world.store.sessions():
            await world.sup.tunnels.stop(row["tunnel_pid"], row["colab_name"])
        world.fake.kill_all()
        world.store.db.close()


async def browse(preview, path, script):
    world, url, directory = preview
    return await asyncio.to_thread(run_browser, directory, url + path, script)


async def test_detail_poll_updates_counters_without_replacing_static_content(preview):
    result = await browse(preview, "/sessions/1", r"""
return await evaluate(`(async () => {
  const status = document.getElementById('session-status');
  const endpoint = document.getElementById('ep-url');
  const card = endpoint.closest('.card');
  const counter = Array.from(status.querySelectorAll('dt')).find((node) => node.textContent === 'Up / cost')
    .nextElementSibling;
  const counterText = counter.textContent;
  const counterNode = counter.firstChild;
  const keep = Array.from(status.querySelectorAll('button')).find((node) => node.textContent === 'Keep');
  keep.focus();
  const range = document.createRange();
  range.selectNodeContents(endpoint);
  const selection = getSelection();
  selection.removeAllRanges();
  selection.addRange(range);
  const selected = selection.toString();
  scrollTo({ top: 200, behavior: 'instant' });
  const scroll = scrollY;
  let staticMutations = 0;
  const observer = new MutationObserver((records) => staticMutations += records.length);
  observer.observe(card, { subtree: true, childList: true, characterData: true, attributes: true });
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(Error('No status poll completed')), 8000);
    document.addEventListener('htmx:afterSettle', function completed(event) {
      if (event.target.id !== 'session-status') return;
      clearTimeout(timer);
      document.removeEventListener('htmx:afterSettle', completed);
      resolve();
    });
  });
  observer.disconnect();
  const updated = Array.from(status.querySelectorAll('dt')).find((node) => node.textContent === 'Up / cost')
    .nextElementSibling;
  return { counterChanged: updated.textContent !== counterText, counterNodeKept: updated.firstChild === counterNode,
    endpointKept: document.getElementById('ep-url') === endpoint,
    cardKept: document.getElementById('ep-url').closest('.card') === card,
    focusKept: document.activeElement === keep, selectionKept: getSelection().toString() === selected,
    scrollKept: scrollY === scroll, staticMutations };
})()`);
""")
    assert result == {"counterChanged": True, "counterNodeKept": True, "endpointKept": True, "cardKept": True,
                      "focusKept": True, "selectionKept": True, "scrollKept": True, "staticMutations": 0}, result


async def test_sessions_table_keeps_rows_focus_and_horizontal_scroll_during_poll(preview):
    result = await browse(preview, "/sessions?tab=all", r"""
return await evaluate(`(async () => {
  const target = document.getElementById('sessions-table');
  for (let attempt = 0; attempt < 100 && target.querySelectorAll('tbody tr').length !== 2; attempt++)
    await new Promise((resolve) => setTimeout(resolve, 50));
  const table = target.querySelector('table');
  const row = table.querySelector('tbody tr');
  const name = row.querySelector('strong');
  const open = Array.from(row.querySelectorAll('a')).find((node) => node.textContent === 'Reconfigure');
  const column = Array.from(table.querySelectorAll('thead th')).findIndex((node) => node.textContent.trim() === 'Up');
  const counter = row.children[column].firstChild;
  const oldCounter = counter.nodeValue;
  open.focus();
  const range = document.createRange();
  range.selectNodeContents(name);
  getSelection().removeAllRanges();
  getSelection().addRange(range);
  const selected = getSelection().toString();
  const container = target.querySelector('.table-container');
  container.scrollLeft = 200;
  const scroll = container.scrollLeft;
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(Error('No sessions poll completed')), 8000);
    document.addEventListener('htmx:afterSettle', function completed(event) {
      if (event.target.id !== 'sessions-table') return;
      if (target.querySelector('tbody tr')?.children[column].textContent === oldCounter) return;
      clearTimeout(timer);
      document.removeEventListener('htmx:afterSettle', completed);
      resolve();
    });
  });
  const updated = target.querySelector('tbody tr');
  return { tableKept: target.querySelector('table') === table, rowKept: updated === row,
    counterNodeKept: updated.children[column].firstChild === counter,
    counterChanged: updated.children[column].textContent !== oldCounter,
    focusKept: document.activeElement === open, selectionKept: getSelection().toString() === selected,
    scrollKept: target.querySelector('.table-container').scrollLeft === scroll };
})()`);
""")
    assert result == {"tableKept": True, "rowKept": True, "counterNodeKept": True, "counterChanged": True,
                      "focusKept": True, "selectionKept": True, "scrollKept": True}


async def test_identical_resource_update_does_not_mutate_cards(preview):
    result = await browse(preview, "/sessions/1", r"""
return await evaluate(`(async () => {
  const target = document.getElementById('session-resources');
  const cards = Array.from(target.querySelectorAll('.card'));
  let mutations = 0;
  const observer = new MutationObserver((records) => mutations += records.length);
  cards.forEach((card) => observer.observe(card, {
    subtree: true, childList: true, characterData: true, attributes: true,
  }));
  htmx.swap(target, target.innerHTML, { swapStyle: target.getAttribute('hx-swap') }, { contextElement: target });
  await new Promise((resolve) => requestAnimationFrame(resolve));
  observer.disconnect();
  return { cardsKept: cards.every((card, index) => target.querySelectorAll('.card')[index] === card), mutations };
})()`);
""")
    assert result == {"cardsKept": True, "mutations": 0}


async def test_stopping_session_updates_controls_and_badge_without_replacing_panels(preview):
    result = await browse(preview, "/sessions/1", r"""
return await evaluate(`(async () => {
  const controls = document.getElementById('session-controls');
  const state = document.querySelector('#session-state .badge');
  const endpoint = document.getElementById('ep-url');
  const headers = { 'X-CSRF-Token': document.body.dataset.csrf };
  const response = await fetch('/ui/actions/sessions/1/stop', {
    method: 'POST', headers, body: new URLSearchParams({ confirm: 'yes' }),
  });
  if (!response.ok) throw Error(await response.text());
  for (const id of ['session-controls', 'session-status', 'session-resources']) {
    const target = document.getElementById(id);
    await htmx.ajax('GET', target.getAttribute('hx-get'), {
      source: target, target, swap: target.getAttribute('hx-swap'),
    });
  }
  return { controlsKept: document.getElementById('session-controls') === controls,
    stateKept: document.querySelector('#session-state .badge') === state,
    state: document.getElementById('session-controls').dataset.state,
    badge: document.getElementById('session-state').textContent.trim(),
    actions: document.querySelectorAll('#session-controls button, #session-controls a').length,
    endpointKept: document.getElementById('ep-url') === endpoint,
    resourcesEnded: document.getElementById('session-resources').textContent.includes('Session ended') };
})()`);
""")
    assert result == {"controlsKept": True, "stateKept": True, "state": "stopped", "badge": "stopped",
                      "actions": 0, "endpointKept": True, "resourcesEnded": True}


async def test_new_session_row_does_not_replace_existing_rows_or_actions(preview):
    world, url, directory = preview
    vm_port = first_free(37000, 37999)
    payload = json.dumps({"name": "newer", "gpu": "L4", "preset": "qwen38-27b-abl-q4k",
                          "vm_port": vm_port, "confirm": True})
    result = await browse(preview, "/sessions?tab=all", r"""
return await evaluate(`(async () => {
  const target = document.getElementById('sessions-table');
  for (let attempt = 0; attempt < 100 && target.querySelectorAll('tbody tr').length !== 2; attempt++)
    await new Promise((resolve) => setTimeout(resolve, 50));
  const original = Array.from(target.querySelectorAll('tbody tr'));
  const open = Array.from(original[0].querySelectorAll('a')).find((node) => node.textContent === 'Reconfigure');
  const response = await fetch('/api/sessions', { method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': document.body.dataset.csrf },
    body: JSON.stringify(CREATE_REQUEST),
  });
  if (!response.ok) throw Error(await response.text());
  open.focus();
  await htmx.ajax('GET', target.getAttribute('hx-get'), {
    source: target, target, swap: target.getAttribute('hx-swap'),
  });
  const rows = Array.from(target.querySelectorAll('tbody tr'));
  return { names: rows.map((row) => row.querySelector('strong').textContent),
    originalsKept: original.every((row) => rows.includes(row)), focusKept: document.activeElement === open,
    linkKept: Array.from(target.querySelectorAll('a')).includes(open) };
})()`);
""".replace("CREATE_REQUEST", payload))
    assert result == {"names": ["newer", "coder", "llm"], "originalsKept": True, "focusKept": True, "linkKept": True}


async def test_overview_counter_and_attention_polls_preserve_cards_rows_and_actions(preview):
    world, url, directory = preview
    world.sup.alerts.raise_("monitor_test", "Unchanged warning", session_id=1, actions=["keep"])
    result = await browse(preview, "/", r"""
return await evaluate(`(async () => {
  const attention = document.getElementById('ov-attention');
  const alert = attention.querySelector('[hx-post="/ui/actions/sessions/1/keep"]').closest('.alert');
  const action = alert.querySelector('button');
  const kpis = Array.from(document.querySelectorAll('#ov-kpis .card'));
  const rows = Array.from(document.querySelectorAll('#ov-sessions tbody tr'));
  const counter = rows[0].children[5].firstChild;
  const previous = counter.nodeValue;
  action.focus();
  const range = document.createRange();
  range.selectNodeContents(alert.querySelector('p'));
  getSelection().removeAllRanges();
  getSelection().addRange(range);
  let mutations = 0;
  const observer = new MutationObserver((records) => mutations += records.length);
  observer.observe(alert, { subtree: true, childList: true, characterData: true, attributes: true });
  await new Promise((resolve, reject) => {
    const seen = new Set();
    const timer = setTimeout(() => reject(Error('Overview polls did not complete')), 9000);
    document.addEventListener('htmx:afterSettle', function completed(event) {
      if (['ov-sessions', 'ov-attention', 'ov-kpis', 'sidebar-status'].includes(event.target.id))
        seen.add(event.target.id);
      if (seen.size !== 4) return;
      clearTimeout(timer);
      document.removeEventListener('htmx:afterSettle', completed);
      resolve();
    });
  });
  observer.disconnect();
  return { cardsKept: kpis.every((card, index) => document.querySelectorAll('#ov-kpis .card')[index] === card),
    rowsKept: rows.every((row, index) => document.querySelectorAll('#ov-sessions tbody tr')[index] === row),
    counterKept: document.querySelector('#ov-sessions tbody tr').children[5].firstChild === counter,
    counterChanged: counter.nodeValue !== previous, alertKept: attention.contains(alert),
    actionKept: attention.contains(action), focusKept: document.activeElement === action,
    selection: getSelection().toString(), mutations };
})()`);
""")
    assert result == {"cardsKept": True, "rowsKept": True, "counterKept": True, "counterChanged": True,
                      "alertKept": True, "actionKept": True, "focusKept": True,
                      "selection": "Unchanged warning", "mutations": 0}


async def test_patching_during_an_action_keeps_htmx_request_serialization(preview):
    result = await browse(preview, "/sessions/1", r"""
return await evaluate(`(async () => {
  const target = document.createElement('section');
  const markup = '<button id="delayed-action" class="btn" hx-post="/ui/test/delayed-action" ' +
    'hx-swap="none" hx-trigger="click queue:none">Slow action</button>';
  target.innerHTML = markup;
  document.body.appendChild(target);
  htmx.process(target);
  const action = target.firstElementChild;
  let reinitialized = 0;
  document.addEventListener('htmx:afterProcessNode', (event) => {
    if (event.target === action) reinitialized++;
  });
  action.click();
  await new Promise((resolve) => setTimeout(resolve, 100));
  htmx.swap(target, markup, { swapStyle: 'cb-patch' }, { contextElement: target });
  await new Promise((resolve) => setTimeout(resolve, 50));
  action.click();
  await new Promise((resolve) => setTimeout(resolve, 150));
  const response = await fetch('/ui/test/action-count');
  const count = (await response.json()).count;
  await new Promise((resolve) => setTimeout(resolve, 1200));
  return { nodeKept: target.firstElementChild === action, reinitialized, count };
})()`);
""")
    assert result == {"nodeKept": True, "reinitialized": 0, "count": 1}
