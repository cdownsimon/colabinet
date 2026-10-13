import json

import pytest

from colabinet.web.app import make_templates
from tests.fakes.browser import run_browser


def patch_page(content):
    return f'''<!doctype html><html><head>
<script src="/static/vendor/htmx.min.js" defer></script><script src="/static/app.js" defer></script>
</head><body hx-ext="cb-patch" data-csrf="test-csrf">{content}</body></html>'''


@pytest.mark.parametrize("native_move", [True, False])
def test_keyed_reordering_and_removal_preserve_focus_selection_and_existing_rows(tmp_path, native_move):
    page = patch_page('''<section id="patch-target"><table><tbody>
<tr data-cb-key="first"><td>First</td><td><button>Open first</button></td></tr>
<tr data-cb-key="second"><td><span>Selected model</span></td><td><button>Open second</button></td></tr>
<tr data-cb-key="removed"><td>Remove me</td></tr>
</tbody></table></section>''')
    result = run_browser(tmp_path, page, r"""
return await evaluate(`(async () => {
  if (!NATIVE_MOVE) Element.prototype.moveBefore = undefined;
  const target = document.getElementById('patch-target');
  const first = target.querySelector('[data-cb-key="first"]');
  const second = target.querySelector('[data-cb-key="second"]');
  const removed = target.querySelector('[data-cb-key="removed"]');
  const button = second.querySelector('button');
  button.focus();
  const range = document.createRange();
  range.selectNodeContents(second.querySelector('span'));
  getSelection().removeAllRanges();
  getSelection().addRange(range);
  htmx.swap(target, '<table><tbody>' +
    '<tr data-cb-key="second"><td><span>Selected model</span></td><td><button>Open second</button></td></tr>' +
    '<tr data-cb-key="first"><td>First updated</td><td><button>Open first</button></td></tr>' +
    '<tr data-cb-key="added"><td>Added</td></tr>' +
    '</tbody></table>', { swapStyle: 'cb-patch' }, { contextElement: target });
  await new Promise((resolve) => setTimeout(resolve, 30));
  return { order: Array.from(target.querySelectorAll('tr')).map((row) => row.dataset.cbKey),
    firstKept: target.querySelector('[data-cb-key="first"]') === first,
    secondKept: target.querySelector('[data-cb-key="second"]') === second,
    removed: !removed.isConnected, firstUpdated: first.firstElementChild.textContent === 'First updated',
    focusKept: document.activeElement === button, selection: getSelection().toString() };
})()`);
""".replace("NATIVE_MOVE", str(native_move).lower()))
    assert result == {"order": ["second", "first", "added"], "firstKept": True, "secondKept": True,
                      "removed": True, "firstUpdated": True, "focusKept": True, "selection": "Selected model"}


def test_conditional_cards_keep_endpoint_and_initialize_new_actions(tmp_path):
    page = patch_page('''<section id="patch-target">
<section data-cb-key="endpoint"><code id="endpoint">http://127.0.0.1:3080/v1</code><button>Copy</button></section>
<section data-cb-key="cost">0.100 CU</section></section>''')
    result = run_browser(tmp_path, page, r"""
return await evaluate(`(async () => {
  const target = document.getElementById('patch-target');
  const endpoint = document.getElementById('endpoint');
  const card = endpoint.parentElement;
  const copy = card.querySelector('button');
  const original = card.outerHTML;
  copy.focus();
  const range = document.createRange();
  range.selectNodeContents(endpoint);
  getSelection().removeAllRanges();
  getSelection().addRange(range);
  let mutations = 0;
  const observer = new MutationObserver((records) => mutations += records.length);
  observer.observe(card, { subtree: true, childList: true, characterData: true, attributes: true });
  const actionRequests = [];
  document.addEventListener('htmx:beforeRequest', (event) => {
    actionRequests.push(event.detail.requestConfig.path);
    event.preventDefault();
  });
  htmx.swap(target, '<div data-cb-key="alert"><button hx-post="/retry">Retry</button></div>' +
    '<section data-cb-key="setup">Setting up</section>' + original +
    '<section data-cb-key="cost">0.200 CU</section>',
    { swapStyle: 'cb-patch' }, { contextElement: target });
  await new Promise((resolve) => setTimeout(resolve, 30));
  target.querySelector('[hx-post]').click();
  htmx.swap(target, original + '<section data-cb-key="cost">0.300 CU</section>',
    { swapStyle: 'cb-patch' }, { contextElement: target });
  await new Promise((resolve) => setTimeout(resolve, 30));
  observer.disconnect();
  return { endpointKept: document.getElementById('endpoint') === endpoint,
    cardKept: endpoint.parentElement === card, focusKept: document.activeElement === copy,
    selection: getSelection().toString(), mutations, actionRequests,
    conditionalRemoved: !target.querySelector('[data-cb-key="alert"], [data-cb-key="setup"]'),
    cost: target.querySelector('[data-cb-key="cost"]').textContent };
})()`);
""")
    assert result == {"endpointKept": True, "cardKept": True, "focusKept": True,
                      "selection": "http://127.0.0.1:3080/v1", "mutations": 0, "actionRequests": ["/retry"],
                      "conditionalRemoved": True, "cost": "0.300 CU"}


@pytest.mark.parametrize("selected_reading", ["temperature", "cores"])
def test_missing_resource_counters_keep_unchanged_readings_and_selection(tmp_path, selected_reading):
    resources = {
        "cpu": {"percent": 25, "cores": 4},
        "ram": {"used_bytes": 8 * 1073741824, "total_bytes": 16 * 1073741824,
                "available_bytes": 8 * 1073741824, "percent": 50},
        "disk": {"used_bytes": 25 * 1073741824, "total_bytes": 100 * 1073741824,
                 "free_bytes": 75 * 1073741824, "percent": 25},
        "gpu": {"name": "L4", "utilization_pct": 30, "temperature_c": 61, "used_mib": 8192, "vram_mib": 16384},
        "vram_pct": 50, "sampled_at": None, "ended": False, "stale": False, "cpu_only": False,
    }
    template = make_templates().env.get_template("partials/session_resources.html")
    known = template.render(resources=resources)
    for field in ("cpu", "ram", "disk"):
        resources[field]["percent"] = None
    resources["gpu"]["utilization_pct"] = resources["vram_pct"] = None
    missing = template.render(resources=resources)
    result = run_browser(tmp_path, patch_page(f'<section id="patch-target">{known}</section>'), r"""
const missing = MISSING_READINGS;
return await evaluate(`(async () => {
  const target = document.getElementById('patch-target');
  const cards = Array.from(target.querySelectorAll('.card'));
  const temperature = cards[3].querySelector('.cb-resource-value .cb-right');
  const cores = cards[0].querySelector('section > p');
  const selected = SELECTED_READING === 'temperature' ? temperature : cores;
  const range = document.createRange();
  range.selectNodeContents(selected);
  getSelection().removeAllRanges();
  getSelection().addRange(range);
  const original = getSelection().toString();
  htmx.swap(target, ${JSON.stringify(missing)}, { swapStyle: 'cb-patch' }, { contextElement: target });
  await new Promise((resolve) => setTimeout(resolve, 30));
  return { cardsKept: cards.every((card, index) => target.querySelectorAll('.card')[index] === card),
    temperatureKept: target.querySelectorAll('.card')[3].querySelector('.cb-resource-value .cb-right') === temperature,
    coresKept: target.querySelectorAll('.card')[0].querySelector('section > p') === cores,
    selectionKept: getSelection().toString() === original,
    progressRemoved: target.querySelectorAll('[role="progressbar"]').length === 0,
    unknown: target.querySelectorAll('.cb-resource-value')[0].textContent.trim() === '—' };
})()`);
""".replace("SELECTED_READING", json.dumps(selected_reading)).replace("MISSING_READINGS", json.dumps(missing)))
    assert result == {"cardsKept": True, "temperatureKept": True, "coresKept": True, "selectionKept": True,
                      "progressRemoved": True, "unknown": True}


def test_changed_htmx_action_attributes_rewire_the_retained_button(tmp_path):
    page = patch_page('''<section id="patch-target">
<button data-cb-key="action" hx-post="/before" hx-swap="none">Act</button></section>''')
    result = run_browser(tmp_path, page, r"""
return await evaluate(`(async () => {
  const target = document.getElementById('patch-target');
  const action = target.querySelector('button');
  const requests = [];
  document.addEventListener('htmx:beforeRequest', (event) => {
    requests.push(event.detail.requestConfig.path);
    event.preventDefault();
  });
  action.click();
  htmx.swap(target, '<button data-cb-key="action" hx-post="/after" hx-swap="none">Act</button>',
    { swapStyle: 'cb-patch' }, { contextElement: target });
  await new Promise((resolve) => setTimeout(resolve, 30));
  action.click();
  return { buttonKept: target.querySelector('button') === action, requests };
})()`);
""")
    assert result == {"buttonKept": True, "requests": ["/before", "/after"]}
