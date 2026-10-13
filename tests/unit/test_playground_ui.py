import json
import shutil
import subprocess

import pytest

from colabinet.config import PACKAGE_DIR


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is needed for the standalone frontend regression")
def test_streamed_playground_end_to_end_speed_includes_ttft():
    harness = r"""
const fs = require('node:fs');
const elements = {};
global.window = global;
global.document = {
  body: { dataset: { csrf: 'test' } },
  addEventListener: () => {},
  getElementById: (id) => elements[id] ||= { textContent: '', hidden: false, value: '' },
  querySelector: () => ({ querySelector: (selector) => document.getElementById(selector.slice(1)) }),
};
const values = { 'pg-session': '1', 'pg-prompt': 'Hello', 'pg-thinking': 'none', 'pg-max': '256' };
for (const [id, value] of Object.entries(values)) document.getElementById(id).value = value;
document.getElementById('pg-stream').checked = true;
const times = [0, 10000, 20000];
global.performance = { now: () => times.shift() };
const events = [
  { choices: [{ delta: { content: 'answer' } }] },
  { choices: [], usage: { prompt_tokens: 12, completion_tokens: 100, completion_time: 10 } },
];
const chunks = events.map((event) => new TextEncoder().encode(`data: ${JSON.stringify(event)}\n\n`));
let position = 0;
global.fetch = async () => ({
  status: 200,
  headers: { get: () => 'text/event-stream' },
  body: { getReader: () => ({ read: async () => position < chunks.length
    ? { value: chunks[position++], done: false } : { done: true } }) },
});
eval(fs.readFileSync(process.argv[1], 'utf8'));
cbPlaygroundSend().then(() => process.stdout.write(JSON.stringify(elements['pg-metrics'].textContent)));
"""
    result = subprocess.run([shutil.which("node"), "-e", harness, str(PACKAGE_DIR / "static/app.js")],
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    metrics = json.loads(result.stdout)
    assert "TTFT 10.00 s" in metrics
    assert "engine 10.0 t/s" in metrics
    assert "end-to-end 5.0 t/s" in metrics
    assert "total 20.00 s" in metrics
