"""The palette's per-session commands must still run after the dialog moved to a partial.

The dialog was relocated verbatim, but the inline Stop handler lost its closing quote while it
was being copied, so `cbConfirmStop(1, 'llm)` was a syntax error rather than a call. Clicking
"Stop llm" then closed the menu and did nothing else: no confirmation, no stop. Nothing else
caught it, because the regression only shows when the browser parses the handler.

These regressions drive the real console in headless Chrome and assert the observable
outcome - the confirmation opens for the right session, cancelling keeps the session billing,
and confirming stops it.
"""

import asyncio

import pytest

from colabinet.api import create_app
from colabinet.net import first_free, start_server, stop_server
from tests.fakes.browser import require_browser, run_browser
from tests.integration.conftest import make_world

pytestmark = pytest.mark.slow


@pytest.fixture
async def console(tmp_path):
    require_browser()
    world = await make_world(tmp_path)
    world.settings.hidden_port_range = (45000, 45999)
    port = first_free(52000, 52999)
    app = create_app(world.sup, "stop-menu-token", port, csrf="stop-menu-csrf")
    server = task = None
    try:
        sid = await world.sup.create(world.request(name="llm"))
        await world.wait_state(sid, "ready")
        server, task = await start_server(app, port)
        yield world, f"http://127.0.0.1:{port}", tmp_path, sid
    finally:
        if server is not None:
            await stop_server(server, task)
        await world.sup.shutdown()
        for row in await world.store.sessions():
            await world.sup.tunnels.stop(row["tunnel_pid"], row["colab_name"])
        world.fake.kill_all()
        world.store.db.close()


HELPERS = r"""
const delay = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));
// An inline handler that does not parse never runs, and nothing in the page reports it.
async function unparseableHandlers() {
  return await evaluate(`(() => {
    const broken = [];
    for (const node of document.querySelectorAll('#cmdk [onclick]')) {
      const source = node.getAttribute('onclick');
      try { new Function(source); } catch (error) { broken.push(source + '  ->  ' + error.message); }
    }
    return broken;
  })()`);
}
async function clickCommand(label) {
  return await evaluate(`(() => {
    const dialog = document.getElementById('cmdk');
    if (!dialog.open) dialog.showModal();
    const item = [...dialog.querySelectorAll('[role="menuitem"]')]
      .find((node) => node.textContent.trim() === ${JSON.stringify(label)});
    if (!item) return 'missing';
    item.click();
    return 'clicked';
  })()`);
}
async function clickDialogButton(dialog, label) {
  return await evaluate(`(() => {
    const button = [...document.querySelectorAll('#' + ${JSON.stringify(dialog)} + ' button')]
      .find((node) => node.textContent.trim() === ${JSON.stringify(label)});
    if (!button) return 'missing';
    button.click();
    return 'clicked';
  })()`);
}
async function settled(check, attempts = 120) {
  for (let attempt = 0; attempt < attempts; attempt++) {
    if (await evaluate(check)) return true;
    await delay(50);
  }
  return false;
}
"""


async def browse(console, script):
    world, url, directory, sid = console
    return await asyncio.to_thread(run_browser, directory, url + "/", HELPERS + script)


async def test_every_palette_inline_handler_is_parseable(console):
    """Direct guard for the dropped quote: the handler must be a call, not a syntax error."""
    result = await browse(console, "return await unparseableHandlers();")
    assert result == [], result


async def test_palette_stop_opens_the_confirmation_for_that_session(console):
    world, _, _, sid = console
    result = await browse(console, r"""
const clicked = await clickCommand('Stop llm');
const opened = await settled(`document.getElementById('stop-dialog').open`);
return { clicked, opened,
  menuClosed: await evaluate(`!document.getElementById('cmdk').open`),
  name: await evaluate(`document.getElementById('stop-name').textContent`),
  title: await evaluate(`document.getElementById('stop-title').textContent`),
  post: await evaluate(`document.getElementById('stop-confirm').getAttribute('hx-post')`) };
""")
    assert result["clicked"] == "clicked"
    assert result["opened"] is True
    assert result["menuClosed"] is True
    assert result["name"] == "llm"
    assert "Stop llm?" == result["title"]
    assert result["post"] == f"/ui/actions/sessions/{sid}/stop"


async def test_cancelling_the_palette_stop_keeps_the_session_live(console):
    world, _, _, sid = console
    result = await browse(console, r"""
await clickCommand('Stop llm');
await settled(`document.getElementById('stop-dialog').open`);
const cancelled = await clickDialogButton('stop-dialog', 'Cancel');
const closed = await settled(`!document.getElementById('stop-dialog').open`);
return { cancelled, closed, state: await evaluate(`document.getElementById('stop-name').textContent`) };
""")
    assert result["cancelled"] == "clicked"
    assert result["closed"] is True
    # Cancelling is not a stop: the session must still be there and still billing.
    assert [row["state"] for row in await world.store.live_sessions()] == ["ready"]
    assert (await world.store.session(sid))["state"] == "ready"


async def test_confirming_the_palette_stop_stops_the_session(console):
    world, _, _, sid = console
    result = await browse(console, r"""
await clickCommand('Stop llm');
await settled(`document.getElementById('stop-dialog').open`);
const confirmed = await clickDialogButton('stop-dialog', 'Stop');
return { confirmed, closed: await settled(`!document.getElementById('stop-dialog').open`) };
""")
    assert result["confirmed"] == "clicked"
    assert result["closed"] is True
    # "stopping" is still a billing state, so the stop only counts once Colab confirms it.
    assert (await world.wait_state(sid, "stopped", timeout=30))["state"] == "stopped"
    assert await world.store.live_sessions() == []
