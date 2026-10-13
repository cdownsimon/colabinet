"""Layout regressions for the wide session tables.

The session lists carry the only per-row actions in the console (Open / Reconfigure / Stop).
They used to sit in a 13-column table that overflowed its scroll container, so at common
window widths the actions were pushed past the right edge and Stop became unreachable.

A second regression hid the Created column below 1500px instead of moving it, which
removed the only visible date a failed creation had. These tests pin both: actions stay
in view, and every row keeps its date at every width, run or no run.
"""

import pytest

from colabinet.web.app import make_templates
from tests.fakes.browser import run_browser


def _session(index, state="ready", run=True):
    return {
        "id": index, "colab_name": f"session-{index}", "account_email": "owner@example.com",
        "state": state, "stop_unconfirmed": False, "progress": None,
        "creation_error": "RuntimeError: no runtime available" if state == "failed" else None,
        "stop_reason": "user" if state == "stopped" else None,
        "gpu_name": "NVIDIA L4" if state != "failed" else None,
        "gpu_class": "L4" if state != "failed" else None, "gpu_requested": "L4",
        "run": ({"engine": "exllamav3", "alias": "huihui.ai/qwen3.8-27b-abliterated-very-long-suffix",
                 "repo": "huihui-ai/Qwen3.8-27B-abliterated-exl3", "ctx": 65536, "kv": "FP16"} if run else None),
        "proxy_port": 34000 + index, "created_at": 1791000000.0, "up_s": 425, "idle_left_s": 1670.0,
        "cu": 0.501, "requests": 12, "tokens_out": 86, "gen_tps": 10.0,
    }


# Widths covering large desktop, laptop, small tablet, tablet and phone.
# At >=48rem the sidebar is a column and is included in the fixture; below that
# basecoat overlays it, so those widths are rendered without it.
WIDTHS_WITH_SIDEBAR = [1920, 1440, 1280, 1024, 820]
WIDTHS_OVERLAY = [700, 600, 430, 390]


def _shell(markup, sidebar=True):
    # The partials assume base.html, so the page shell, viewport meta and both stylesheets
    # have to be present for the layout under test to be the layout the console ships.
    # At >=48rem the sidebar is a real column and is reproduced here; below that basecoat
    # turns it into an overlay driven by its own JavaScript, which a static fixture cannot
    # run, so those widths are exercised without the column to match the shipped result.
    dark = "<script>try{document.documentElement.classList.add('dark')}catch(e){}</script>"
    aside = (
        '<aside class="sidebar" data-side="left" aria-hidden="false" id="sidebar" '
        'data-sidebar-initialized="true" data-basecoat-component="sidebar"><nav aria-label="colabinet">'
        '<header><a href="/" class="cb-brand"><span><strong>colabinet</strong>'
        "<small>Colab LLM console</small></span></a></header>"
        '<section class="scrollbar"></section>'
        '<footer><div class="cb-row cb-sidebar-actions">'
        '<button type="button" class="btn" data-variant="ghost" data-size="sm">Theme</button>'
        '<hr class="cb-rule"><button type="button" class="btn cb-stop-all" data-variant="destructive" '
        'data-size="sm">Stop all</button></div></footer></nav></aside>'
    ) if sidebar else ""
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<link rel="stylesheet" href="/static/vendor/basecoat-nova.cdn.min.css">'
        '<link rel="stylesheet" href="/static/app.css">' + dark + "</head><body>" + aside +
        '<main id="cb-view" class="cb-main"><header class="cb-topbar">'
        '<button type="button" class="btn" data-variant="ghost" data-size="icon">menu</button>'
        '<div class="cb-crumbs"><h1>Sessions</h1></div>'
        '<div class="cb-top-actions"><button type="button" class="btn" data-variant="outline" '
        'data-size="sm">Search</button></div></header>'
        '<div class="cb-page" data-page="sessions"><section class="card">' + markup + "</section></div>"
        "</main></body></html>"
    )


def _render(template, sidebar=True, **context):
    base = {"csrf": "test-csrf", "active": "sessions", "query": "", "total": 1,
            "base": {"balance": None, "burn_rate": 0, "live_count": 2, "alert_count": 0,
                     "command_sessions": [], "daily": {"spend": 0, "cap": 10}}}
    return _shell(make_templates().env.get_template(template).render(**{**base, **context}), sidebar)


def _table(sidebar=True, **context):
    return _render("partials/sessions_table.html", sidebar=sidebar, **context)


MEASURE = r"""
  const readings = [];
  for (const width of WIDTHS) {
    await send('Emulation.setDeviceMetricsOverride', {
      width, height: 1000, deviceScaleFactor: 1, mobile: width < 560,
    });
    await evaluate(`new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)))`);
    readings.push(await evaluate(`(() => ({
      width: innerWidth,
      documentWidth: document.documentElement.scrollWidth,
      rows: Array.from(document.querySelectorAll('tbody tr')).map((row) => {
        const cell = row.querySelector('.cb-sticky-actions');
        const bounds = cell.getBoundingClientRect();
        return {
          left: bounds.left, right: bounds.right, width: bounds.width,
          sticky: getComputedStyle(cell).position,
          // Every date the row carries, whether shown in its own column or inline.
          dates: Array.from(row.querySelectorAll('.cb-col-optional, .cb-created-compact'))
            .filter((node) => getComputedStyle(node).display !== 'none')
            .map((node) => node.textContent.trim()).filter(Boolean),
          actions: Array.from(cell.querySelectorAll('a, button')).map((node) => {
            const rect = node.getBoundingClientRect();
            return { text: node.textContent.trim(), left: rect.left, right: rect.right,
              visible: rect.width > 0 && rect.height > 0 };
          }) };
      }),
    }))()`));
  }
  return readings;
"""


@pytest.mark.parametrize("tab,rows", [
    ("all", [_session(1), _session(2)]),
    ("live", [_session(1), _session(2)]),
    # A failed creation has no run, so its date cannot ride along with model metadata.
    ("all", [_session(1, "failed", run=False)]),
    ("history", [_session(3, "failed", run=False), _session(4, "stopped")]),
])
def test_row_actions_stay_inside_the_viewport_at_every_width(tmp_path, tab, rows):
    page = _table(rows=rows, tab=tab, total=len(rows))
    for reading in run_browser(tmp_path, page, MEASURE.replace("WIDTHS", str(WIDTHS_WITH_SIDEBAR))):
        assert reading["documentWidth"] <= reading["width"], reading
        assert reading["rows"], reading
        for row in reading["rows"]:
            assert row["sticky"] == "sticky", reading
            assert row["width"] > 0, reading
            assert row["left"] >= -0.5, reading
            assert row["right"] <= reading["width"] + 0.5, (reading["width"], row)
            assert row["actions"], reading
            for action in row["actions"]:
                assert action["visible"], (reading["width"], action)
                assert action["left"] >= -0.5, (reading["width"], action)
                assert action["right"] <= reading["width"] + 0.5, (reading["width"], action)


@pytest.mark.parametrize("tab,rows", [
    ("all", [_session(1), _session(2)]),
    # The regression this test exists for: no run, so nothing else carries the date.
    ("all", [_session(5, "failed", run=False)]),
    ("history", [_session(6, "stopped")]),
])
def test_every_row_shows_its_created_date_at_every_width(tmp_path, tab, rows):
    page = _table(rows=rows, tab=tab, total=len(rows))
    assert all(rows)
    for reading in run_browser(tmp_path, page, MEASURE.replace("WIDTHS", str(WIDTHS_WITH_SIDEBAR))):
        for row in reading["rows"]:
            assert row["dates"], (reading["width"], tab, row)
            for value in row["dates"]:
                assert "2026" in value or "20" in value, (reading["width"], tab, value)


def test_overview_actions_stay_inside_the_viewport_at_every_width(tmp_path):
    page = _render("partials/overview_sessions.html",
                   overview_data={"sessions": [_session(1), _session(2)]})
    for reading in run_browser(tmp_path, page, MEASURE.replace("WIDTHS", str(WIDTHS_WITH_SIDEBAR))):
        assert reading["documentWidth"] <= reading["width"], reading
        for row in reading["rows"]:
            assert row["sticky"] == "sticky", reading
            assert row["right"] <= reading["width"] + 0.5, (reading["width"], row)
            for action in row["actions"]:
                assert action["visible"] and action["right"] <= reading["width"] + 0.5, (reading["width"], action)


def test_stop_action_is_present_and_reachable_for_a_live_session(tmp_path):
    page = _table(rows=[_session(1)], tab="live", total=1)
    result = run_browser(tmp_path, page, r"""
await evaluate(`new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)))`);
return await evaluate(`(() => {
  const stop = Array.from(document.querySelectorAll('tbody button')).find((node) => node.textContent.trim() === 'Stop');
  const bounds = stop.getBoundingClientRect();
  return { text: stop.textContent.trim(), onclick: stop.getAttribute('onclick'),
    variant: stop.getAttribute('data-variant'),
    inside: bounds.right <= innerWidth + 0.5 && bounds.left >= -0.5 };
})()`);
""")
    assert result["text"] == "Stop"
    assert "cbConfirmStop(1" in result["onclick"]
    assert result["variant"] == "destructive", result
    assert result["inside"], result

@pytest.mark.parametrize("tab,rows", [
    ("all", [_session(1), _session(2)]),
    ("all", [_session(7, "failed", run=False)]),
])
def test_phone_widths_keep_actions_reachable_and_dates_visible(tmp_path, tab, rows):
    """Phone widths render the sidebar as an overlay, which the fixture reproduces by
    omitting the column: a static page cannot run the JavaScript that drives that state."""
    page = _table(sidebar=False, rows=rows, tab=tab, total=len(rows))
    for reading in run_browser(tmp_path, page, MEASURE.replace("WIDTHS", str(WIDTHS_OVERLAY))):
        assert reading["documentWidth"] <= reading["width"], reading
        for row in reading["rows"]:
            assert row["sticky"] == "sticky", reading
            assert row["actions"], reading
            assert row["dates"], (reading["width"], tab, row)
            for action in row["actions"]:
                assert action["visible"], (reading["width"], action)
                assert action["right"] <= reading["width"] + 0.5, (reading["width"], action)
