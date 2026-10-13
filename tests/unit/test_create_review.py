import json
import shutil
import subprocess

import pytest

from colabinet.config import PACKAGE_DIR


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is needed for the standalone frontend regression")
@pytest.mark.parametrize("scenario", ["fresh", "invalid", "edited"])
def test_create_review_uses_a_fresh_estimate_for_unchanged_fields(scenario):
    harness = r"""
const fs = require('node:fs');
const scenario = process.argv[2];
const elements = {};
const toasts = [];
global.window = global;
global.document = {
  body: { dataset: { csrf: 'test' } }, addEventListener: () => {},
  getElementById: (id) => elements[id] ||= { textContent: '', hidden: false },
};
const form = document.getElementById('new-form');
form.reportValidity = () => true;
form.values = { idle_min: '20', cu_cap: '0.8' };
global.FormData = class {
  constructor(source) { this.values = { ...source.values }; }
  [Symbol.iterator]() { return Object.entries(this.values)[Symbol.iterator](); }
};
document.getElementById('cost-lines').dataset = { summary: 'L4 at 1.54 CU/h, idle stop 30 min' };
document.getElementById('create-dialog').showModal = () => elements['create-dialog'].open = true;
document.getElementById('toaster').toast = (toast) => toasts.push(toast);
Object.defineProperty(document.getElementById('estimate'), 'innerHTML', {
  set: () => elements['cost-lines'].dataset.summary = 'L4 at 1.54 CU/h, idle stop 20 min, cap 0.8 CU',
});
global.htmx = { process: () => {} };
let fetches = 0;
global.fetch = async (url, options) => {
  fetches++;
  if (url !== '/ui/new/estimate' || options.body.values.idle_min !== '20') throw Error('wrong estimate request');
  if (scenario === 'edited') form.values.idle_min = '0';
  return { ok: scenario !== 'invalid', json: async () => ({ error: 'invalid guard' }), text: async () => 'fresh' };
};
eval(fs.readFileSync(process.argv[1], 'utf8'));
Promise.resolve(cbReviewCreate()).then(() => process.stdout.write(JSON.stringify({
  fetches, summary: elements['create-summary']?.textContent, open: Boolean(elements['create-dialog'].open), toasts,
})));
"""
    result = subprocess.run([shutil.which("node"), "-e", harness, str(PACKAGE_DIR / "static/app.js"), scenario],
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    review = json.loads(result.stdout)
    assert review["fetches"] == 1
    if scenario == "fresh":
        assert review["open"]
        assert "idle stop 20 min, cap 0.8 CU" in review["summary"]
    else:
        assert not review["open"]
        assert review["toasts"]
