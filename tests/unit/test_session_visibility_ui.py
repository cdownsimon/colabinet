import json
import shutil
import subprocess

import pytest

from colabinet.config import PACKAGE_DIR

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node is needed for frontend regressions")


def run_frontend(script):
    harness = r"""
const fs = require('node:fs');
const elements = {};
const listeners = {};
const requests = [];
const triggers = [];
global.window = global;
global.location = { origin: 'http://127.0.0.1:7860' };
global.document = {
  body: { dataset: { csrf: 'test-token' } }, visibilityState: 'visible',
  addEventListener: (name, callback) => (listeners[name] ||= []).push(callback),
  getElementById: (id) => elements[id] || null,
};
global.htmx = { process: () => {}, trigger: (target, event) => triggers.push(event) };
global.fetch = async (url, options) => { requests.push({ url, ...options }); return { ok: true }; };
eval(fs.readFileSync(process.argv[1], 'utf8'));
""" + script
    result = subprocess.run([shutil.which("node"), "-e", harness, str(PACKAGE_DIR / "static/app.js")],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_resource_observation_runs_only_for_a_visible_active_session():
    result = run_frontend(r"""
elements['session-watch'] = { dataset: { sessionId: '7' } };
elements['session-controls'] = { dataset: { state: 'ready' } };
(async () => {
  await cbObserveSession();
  document.visibilityState = 'hidden';
  await cbObserveSession();
  document.visibilityState = 'visible';
  elements['session-controls'].dataset.state = 'stopped';
  await cbObserveSession();
  process.stdout.write(JSON.stringify(requests));
})();
""")
    assert result == [{"url": "/ui/sessions/7/watch", "method": "POST", "headers": {
        "X-CSRF-Token": "test-token",
    }}]


def test_log_pause_prevents_requests_and_resume_refreshes_immediately():
    result = run_frontend(r"""
const button = { textContent: 'Pause', setAttribute: () => {} };
let prevented = 0;
cbToggleLogs(button);
for (const listener of listeners['htmx:beforeRequest'] || []) listener({
  detail: { elt: { id: 'session-logs' } }, preventDefault: () => prevented++,
});
const pausedLabel = button.textContent;
cbToggleLogs(button);
process.stdout.write(JSON.stringify({ prevented, pausedLabel, resumedLabel: button.textContent, triggers }));
""")
    assert result == {"prevented": 1, "pausedLabel": "Resume", "resumedLabel": "Pause",
                      "triggers": ["htmx:abort", "refresh"]}


def test_pausing_an_outstanding_log_request_leaves_the_cursor_and_display_unchanged():
    result = run_frontend(r"""
let url = '/ui/sessions/1/logs?after=7&after_event=9';
const target = { id: 'session-logs', getAttribute: () => url, setAttribute: (key, value) => url = value };
elements['session-logs'] = target;
elements['session-log-lines'] = { textContent: 'old line\n', scrollTop: 10, scrollHeight: 900 };
let aborted = false;
htmx.trigger = (element, name) => {
  triggers.push(name);
  if (element === target && name === 'htmx:abort') aborted = true;
};
const button = { textContent: 'Pause', setAttribute: () => {} };
cbToggleLogs(button);
let prevented = false;
for (const listener of listeners['htmx:beforeOnLoad'] || []) listener({
  detail: { elt: target }, preventDefault: () => prevented = true,
});
if (!prevented) elements['session-log-lines'].textContent += 'delayed line\n';
for (const listener of listeners['htmx:afterRequest'] || []) listener({ detail: {
  elt: target, successful: true, xhr: { getResponseHeader: () => '20' },
} });
const paused = { aborted, prevented, url, lines: elements['session-log-lines'].textContent,
  scroll: elements['session-log-lines'].scrollTop };
cbToggleLogs(button);
process.stdout.write(JSON.stringify({ paused, triggers }));
""")
    assert result == {"paused": {"aborted": True, "prevented": True,
        "url": "/ui/sessions/1/logs?after=7&after_event=9", "lines": "old line\n", "scroll": 10},
        "triggers": ["htmx:abort", "refresh"]}


@pytest.mark.parametrize("autoscroll", [True, False])
@pytest.mark.parametrize("connected", [True, False])
def test_log_cursor_retention_and_optional_autoscroll(autoscroll, connected):
    result = run_frontend(r"""
let url = '/ui/sessions/1/logs?after=0&after_event=0';
const target = { id: 'session-logs', isConnected: CONNECTED,
  getAttribute: () => url, setAttribute: (key, value) => url = value };
elements['session-log-lines'] = {
  textContent: Array.from({ length: 600 }, (_, index) => `line ${index}`).join('\n') + '\n',
  scrollTop: 10, scrollHeight: 900,
};
const feed = { children: Array.from({ length: 220 }, () => ({})), scrollTop: 10, scrollHeight: 700 };
Object.defineProperty(feed, 'firstElementChild', { get: () => ({ remove: () => feed.children.shift() }) });
elements['session-log-events'] = feed;
elements['logs-autoscroll'] = { checked: AUTOSCROLL };
for (const listener of listeners['htmx:afterRequest'] || []) listener({ detail: {
  elt: target, successful: true, xhr: { getResponseHeader: (name) => name === 'X-Log-Cursor' ? '600' : '225' },
} });
process.stdout.write(JSON.stringify({ url, firstLine: elements['session-log-lines'].textContent.split('\n')[0],
  lines: elements['session-log-lines'].textContent.trimEnd().split('\n').length,
  events: feed.children.length, logScroll: elements['session-log-lines'].scrollTop, eventScroll: feed.scrollTop }));
    """.replace("AUTOSCROLL", json.dumps(autoscroll)).replace("CONNECTED", json.dumps(connected)))
    if not connected:
        assert result == {"url": "/ui/sessions/1/logs?after=0&after_event=0", "firstLine": "line 0",
                          "lines": 600, "events": 220, "logScroll": 10, "eventScroll": 10}
        return
    assert result["url"] == "/ui/sessions/1/logs?after=600&after_event=225"
    assert result["firstLine"] == "line 100"
    assert result["lines"] == 500
    assert result["events"] == 200
    assert result["logScroll"] == (900 if autoscroll else 10)
    assert result["eventScroll"] == (700 if autoscroll else 10)
