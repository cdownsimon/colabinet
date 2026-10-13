"""The command menu must follow a Settings save while the page stays open.

The save form posts with hx-swap="none", so before the out-of-band refresh the open document
kept the GPU list it was rendered with: unchecking G4 left "New session on G4" in the palette
until something else re-rendered the page, and following it landed on a form that silently
selected T4 instead.

These regressions drive the real console in headless Chrome and read the live DOM, because the
stale item was only ever reachable in the open document - a fresh GET after the save sees the
new list whether or not the swap works. Every save asserts a sentinel set before the click, so
a passing test cannot have quietly reloaded the page.
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
    world.settings.hidden_port_range = (44000, 44999)
    port = first_free(51000, 51999)
    app = create_app(world.sup, "palette-live-token", port, csrf="palette-live-csrf")
    server = task = None
    try:
        server, task = await start_server(app, port)
        yield world, f"http://127.0.0.1:{port}", tmp_path
    finally:
        if server is not None:
            await stop_server(server, task)
        await world.sup.shutdown()
        world.fake.kill_all()
        world.store.db.close()


HELPERS = r"""
const delay = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));
const palette = async () => await evaluate(`[...document.querySelectorAll('#cmdk-gpus a[role="menuitem"]')]
  .map((link) => link.getAttribute('href').split('=')[1])`);
async function setBoxes(classes, checked) {
  await evaluate(`(() => { for (const gpu of ${JSON.stringify(classes)}) {
    const box = document.querySelector('input[name="eligible_gpus"][value="' + gpu + '"]');
    if (!box) throw Error('no checkbox for ' + gpu);
    box.checked = ${JSON.stringify(checked)};
  } return true; })()`);
}
// Save with the real button and wait for htmx to report the request finished. A sentinel set
// beforehand is read back afterwards: a reload would drop it.
async function save() {
  const armed = await evaluate(`(() => {
    window.__sentinel = 'same-document';
    window.__seen = [];
    window.__save = new Promise((resolve) => {
      const finish = (value) => resolve(value);
      setTimeout(() => finish({ ok: false, status: 0, timedOut: true, seen: window.__seen }), 5000);
      document.body.addEventListener('htmx:afterRequest', (event) => {
        const url = event.detail.xhr?.responseURL || event.detail.pathInfo?.requestPath || '';
        window.__seen.push(url);
        if (url.includes('/ui/actions/settings')) {
          finish({ ok: event.detail.successful, status: event.detail.xhr.status, seen: window.__seen });
        }
      });
    });
    const button = document.querySelector('form[hx-post="/ui/actions/settings"] button[type="submit"]');
    if (!button) return 'no save button';
    button.click();
    return 'clicked';
  })()`);
  if (armed !== 'clicked') return { armed, ok: false, status: 0 };
  const result = await evaluate(`window.__save`);
  await delay(80);
  result.sameDocument = (await evaluate(`window.__sentinel`)) === 'same-document';
  return result;
}
// Open the palette and type, exactly as a person would, then read what is still offered.
async function search(text) {
  return await evaluate(`(() => {
    const dialog = document.getElementById('cmdk');
    if (!dialog.open) dialog.showModal();
    const input = document.getElementById('cmdk-input');
    input.value = ${JSON.stringify(text)};
    input.dispatchEvent(new Event('input', { bubbles: true }));
    return [...dialog.querySelectorAll('#cmdk-gpus a[role="menuitem"]')]
      .filter((item) => item.getAttribute('aria-hidden') !== 'true')
      .map((item) => item.getAttribute('href').split('=')[1]);
  })()`);
}
async function closePalette() {
  await evaluate(`(() => { const dialog = document.getElementById('cmdk');
    if (dialog.open) dialog.close(); return true; })()`);
}
"""


async def browse(console, script):
    world, url, directory = console
    return await asyncio.to_thread(run_browser, directory, url + "/settings", HELPERS + script)


async def test_unchecking_a_gpu_drops_it_from_the_open_palette(console):
    world, _, _ = console
    assert "G4" in (await world.sup.settings_values())["eligible_gpus"]
    result = await browse(console, r"""
const before = await palette();
await setBoxes(['G4'], false);
const outcome = await save();
return { before, save: outcome, after: await palette(), search: await search('G4') };
""")
    assert "G4" in result["before"]
    # The save finished and the document was never reloaded, so this is the live page.
    assert result["save"]["ok"] is True and result["save"]["sameDocument"] is True
    assert "G4" not in result["after"]
    assert result["after"] == ["CPU", "T4", "T4-HIGHMEM", "L4", "A100", "A100-80"]
    # The reviewer's own repro: open Search and type G4, with no reload in between.
    assert result["search"] == []
    # A rendered body is what carries the out-of-band swap; 204 is dropped by htmx.
    assert result["save"]["status"] == 200


async def test_re_enabling_a_gpu_brings_it_back_without_a_reload(console):
    result = await browse(console, r"""
await setBoxes(['G4'], false);
await save();
const dropped = await palette();
await setBoxes(['G4'], true);
const second = await save();
return { dropped, second, after: await palette(), search: await search('G4') };
""")
    assert "G4" not in result["dropped"]
    assert result["second"]["ok"] is True and result["second"]["sameDocument"] is True
    assert "G4" in result["after"]
    assert result["after"] == ["CPU", "T4", "T4-HIGHMEM", "L4", "A100", "A100-80", "G4"]
    assert result["search"] == ["G4"]
    assert result["second"]["status"] == 200


async def test_filter_still_works_after_the_palette_is_replaced(console):
    """The whole dialog is re-sent, so the menu's item cache is rebuilt rather than stale."""
    result = await browse(console, r"""
await setBoxes(['G4'], false);
const outcome = await save();
return { save: outcome, g4: await search('G4'), a100: await search('A100'), page: await search('settings') };
""")
    assert result["save"]["sameDocument"] is True
    assert result["g4"] == []
    assert result["a100"] == ["A100", "A100-80"]
    assert result["page"] == []


async def test_cpu_only_palette_offers_cpu_and_that_command_selects_cpu(console):
    result = await browse(console, r"""
await setBoxes(['T4', 'L4', 'A100', 'G4'], false);
const outcome = await save();
const offered = await palette();
await search('CPU');
await evaluate(`document.querySelector('#cmdk-gpus a[href="/new?gpu=CPU"]').click(); true`);
for (let attempt = 0; attempt < 100; attempt++) {
  if (await evaluate(`location.pathname === '/new'`)) break;
  await delay(30);
}
return { save: outcome, offered, path: await evaluate(`location.pathname + location.search`),
  selected: await evaluate(`[...document.querySelectorAll('#new-form input[name="gpu"]')]
    .filter((input) => input.checked).map((input) => input.value)`) };
""")
    assert result["save"]["ok"] is True and result["save"]["sameDocument"] is True
    assert result["offered"] == ["CPU"]
    assert result["path"] == "/new?gpu=CPU"
    assert result["selected"] == ["CPU"]
    assert result["save"]["status"] == 200


async def test_every_offered_command_on_the_open_page_selects_its_gpu(console):
    """No offered destination may fall through to the /new substitution."""
    result = await browse(console, r"""
await setBoxes(['G4'], false);
const outcome = await save();
const offered = await palette();
const selections = [];
for (const gpu of offered) {
  await evaluate(`(() => { const dialog = document.getElementById('cmdk');
    if (!dialog.open) dialog.showModal(); return true; })()`);
  await evaluate(`(() => { const link = document.querySelector('#cmdk-gpus a[href="/new?gpu='
    + ${JSON.stringify(gpu)} + '"]');
    if (!link) return false; link.click(); return true; })()`);
  for (let attempt = 0; attempt < 100; attempt++) {
    if (await evaluate(`location.pathname === '/new'`)) break;
    await delay(30);
  }
  selections.push([gpu, await evaluate(`[...document.querySelectorAll('#new-form input[name="gpu"]')]
    .filter((input) => input.checked).map((input) => input.value)`)]);
  await closePalette();
  await evaluate(`history.replaceState({}, '', '/settings'); true`);
  await delay(30);
}
return { offered, selections };
""")
    assert result["offered"] == ["CPU", "T4", "T4-HIGHMEM", "L4", "A100", "A100-80"]
    assert result["selections"] == [[gpu, [gpu]] for gpu in result["offered"]]
