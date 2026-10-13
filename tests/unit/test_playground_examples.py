import json
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from colabinet.web.app import make_templates
from tests.fakes.browser import run_browser


@pytest.fixture
def endpoints():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append({"port": self.server.server_port, "path": self.path, "body": body})
            reply = json.dumps({"choices": [{"message": {"content": "example accepted"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            pass

    servers = [ThreadingHTTPServer(("127.0.0.1", 0), Handler) for _ in range(2)]
    threads = [threading.Thread(target=server.serve_forever, daemon=True) for server in servers]
    for thread in threads:
        thread.start()
    try:
        yield [server.server_port for server in servers], requests
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join()


def page(ports):
    return make_templates().env.get_template("pages/playground.html").render(
        csrf="test-csrf", active="playground", selected=2,
        sessions=[{"id": index, "colab_name": f"session-{index}", "proxy_port": port,
                   "gpu_class": "L4", "run": {"alias": f"model-{index}"}}
                  for index, port in enumerate(ports, 1)],
        base={"balance": None, "burn_rate": 0, "live_count": 2, "alert_count": 0,
              "command_sessions": [], "new_gpus": ["CPU"], "daily": {"spend": 0, "cap": 10}},
    )


SNAPSHOT = r"""
const snapshot = () => ({
  python: document.getElementById('pg-python-example')?.textContent || '',
  curl: document.getElementById('pg-curl-example')?.textContent || '',
  copiesEnabled: document.getElementById('pg-copy-example')?.disabled === false,
  languageEnabled: document.getElementById('pg-example-language')?.disabled === false,
  visible: [...document.querySelectorAll('#pg-client-examples pre code')]
    .filter((code) => code.getClientRects().length > 0).map((code) => code.id),
});
"""


def test_copied_examples_execute_for_the_selected_session_and_edited_prompt(tmp_path, endpoints):
    if shutil.which("curl") is None:
        pytest.skip("curl is needed to execute the terminal example")
    ports, requests = endpoints
    sentinel = tmp_path / "should-not-be-created"
    prompt = ("Quotes: 'single' and \"double\" and '''; backslash \\; café.\n"
              "COLABINET_REQUEST\n<script>window.injected = true</script>\n"
              f"$(touch {sentinel}) and `touch {sentinel}`")
    result = run_browser(tmp_path, page(ports), SNAPSHOT + "const prompt = " + json.dumps(prompt) + r""";
const initial = await evaluate(`(${snapshot.toString()})()`);
await evaluate(`(() => {
  document.getElementById('pg-session').value = '1';
  document.getElementById('pg-session').dispatchEvent(new Event('change', { bubbles: true }));
  document.getElementById('pg-prompt').value = ${JSON.stringify(prompt)};
  document.getElementById('pg-prompt').dispatchEvent(new Event('input', { bubbles: true }));
  document.getElementById('pg-thinking').value = 'high';
  document.getElementById('pg-thinking').dispatchEvent(new Event('change', { bubbles: true }));
  document.getElementById('pg-max').value = '37';
  document.getElementById('pg-max').dispatchEvent(new Event('input', { bubbles: true }));
})()`);
const edited = await evaluate(`(${snapshot.toString()})()`);
const copied = await evaluate(`(async () => {
  const values = [];
  Object.defineProperty(navigator, 'clipboard', { configurable: true,
    value: { writeText: async (text) => values.push(text) } });
  for (const language of ['python', 'curl']) {
    const select = document.getElementById('pg-example-language');
    select.value = language;
    select.dispatchEvent(new Event('change', { bubbles: true }));
    document.getElementById('pg-copy-example')?.click();
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  return { values, injected: window.injected === true };
})()`);
return { initial, edited, copied };
""")
    assert result["initial"]["python"], "The ready session needs a runnable Python example"
    assert result["initial"]["curl"], "The ready session needs a runnable curl example"
    assert result["edited"]["copiesEnabled"] is True
    assert result["copied"]["values"] == [result["edited"]["python"], result["edited"]["curl"]]
    assert result["copied"]["injected"] is False
    for version, port, alias, content, limit, thinking in (
        ("initial", ports[1], "model-2", "Say hello in five words.", 256, "none"),
        ("edited", ports[0], "model-1", prompt, 37, "high"),
    ):
        python = subprocess.run([sys.executable, "-c", result[version]["python"]],
                                text=True, capture_output=True, timeout=10)
        assert python.returncode == 0, python.stderr
        assert python.stdout.strip() == "example accepted"
        curl = subprocess.run(["sh", "-c", result[version]["curl"]],
                              text=True, capture_output=True, timeout=10)
        assert curl.returncode == 0, curl.stderr
        assert json.loads(curl.stdout)["choices"][0]["message"]["content"] == "example accepted"
        expected = {"model": alias, "stream": False, "max_tokens": limit,
                    "messages": [{"role": "user", "content": content}],
                    "chat_template_kwargs": {"enable_thinking": thinking != "none"}}
        if thinking != "none":
            expected["reasoning_effort"] = thinking
            expected["chat_template_kwargs"]["reasoning_effort"] = thinking
        assert requests[-2:] == [{"port": port, "path": "/v1/chat/completions", "body": expected}] * 2
        assert not sentinel.exists()


def test_invalid_token_limit_disables_copy_and_restores_examples_when_corrected(tmp_path):
    result = run_browser(tmp_path, page([34001, 34002]), SNAPSHOT + r"""
const initial = await evaluate(`(${snapshot.toString()})()`);
await evaluate(`(() => {
  const control = document.getElementById('pg-max');
  control.value = '0';
  control.dispatchEvent(new Event('input', { bubbles: true }));
})()`);
const invalid = await evaluate(`(${snapshot.toString()})()`);
await evaluate(`(() => {
  const control = document.getElementById('pg-max');
  control.value = '128';
  control.dispatchEvent(new Event('input', { bubbles: true }));
})()`);
const restored = await evaluate(`(${snapshot.toString()})()`);
return { initial, invalid, restored };
""")
    assert result["initial"]["copiesEnabled"] is True
    assert result["invalid"]["copiesEnabled"] is False
    assert result["invalid"]["languageEnabled"] is True
    assert result["invalid"]["visible"] == []
    assert result["invalid"]["python"] == result["invalid"]["curl"] == ""
    assert result["restored"]["copiesEnabled"] is True
    assert result["restored"]["visible"] == ["pg-python-example"]
    assert result["restored"]["python"] and result["restored"]["curl"]


def test_language_dropdown_shows_and_copies_only_the_selected_example(tmp_path):
    result = run_browser(tmp_path, page([34001, 34002]), r"""
return await evaluate(`(async () => {
  const select = document.getElementById('pg-example-language');
  if (!select) return { missingDropdown: true };
  const values = [];
  Object.defineProperty(navigator, 'clipboard', { configurable: true,
    value: { writeText: async (text) => values.push(text) } });
  const snapshot = () => ({ language: select.value,
    visible: [...document.querySelectorAll('#pg-client-examples pre code')]
      .filter((code) => code.getClientRects().length > 0).map((code) => code.id),
    copyLabel: document.getElementById('pg-copy-example').getAttribute('aria-label'),
  });
  const initial = snapshot();
  document.getElementById('pg-copy-example').click();
  await new Promise((resolve) => setTimeout(resolve, 0));
  select.value = 'curl';
  select.dispatchEvent(new Event('change', { bubbles: true }));
  const curl = snapshot();
  document.getElementById('pg-prompt').value = 'Updated while curl is selected';
  document.getElementById('pg-prompt').dispatchEvent(new Event('input', { bubbles: true }));
  document.getElementById('pg-copy-example').click();
  await new Promise((resolve) => setTimeout(resolve, 0));
  select.value = 'python';
  select.dispatchEvent(new Event('change', { bubbles: true }));
  const returned = snapshot();
  return { initial, curl, returned, values,
    python: document.getElementById('pg-python-example').textContent,
    curlText: document.getElementById('pg-curl-example').textContent,
    labelled: !!document.querySelector('label[for="pg-example-language"]'),
    options: [...select.options].map((option) => option.textContent),
  };
})()`);
""")
    assert "missingDropdown" not in result, "Choose one example with a language dropdown"
    assert result["labelled"] is True
    assert result["options"] == ["Python 3", "curl"]
    assert result["initial"] == {"language": "python", "visible": ["pg-python-example"],
                                  "copyLabel": "Copy Python example"}
    assert result["curl"] == {"language": "curl", "visible": ["pg-curl-example"],
                               "copyLabel": "Copy curl example"}
    assert result["returned"] == result["initial"]
    assert "Say hello in five words." in result["values"][0]
    assert result["values"][0].startswith("import json")
    assert result["values"][1] == result["curlText"]
    assert "Updated while curl is selected" in result["python"]
    assert "Updated while curl is selected" in result["curlText"]


def test_example_copy_feedback_names_the_language_and_handles_clipboard_failure(tmp_path):
    result = run_browser(tmp_path, page([34001, 34002]), r"""
return await evaluate(`(async () => {
  const toasts = [];
  cbToast = (...args) => toasts.push(args);
  Object.defineProperty(navigator, 'clipboard', { configurable: true,
    value: { writeText: async () => {} } });
  document.getElementById('pg-copy-example').click();
  await new Promise((resolve) => setTimeout(resolve, 0));
  await cbCopy('http://127.0.0.1:34001/v1');
  const select = document.getElementById('pg-example-language');
  select.value = 'curl';
  select.dispatchEvent(new Event('change', { bubbles: true }));
  document.getElementById('pg-copy-example').click();
  await new Promise((resolve) => setTimeout(resolve, 0));
  Object.defineProperty(navigator, 'clipboard', { configurable: true,
    value: { writeText: async () => { throw new Error('Clipboard unavailable'); } } });
  document.getElementById('pg-copy-example').click();
  await new Promise((resolve) => setTimeout(resolve, 0));
  const panel = document.getElementById('pg-curl-panel');
  panel.focus();
  return { toasts, codeFocusable: document.activeElement === panel,
    codeLabel: panel.getAttribute('aria-label') };
})()`);
""")
    assert result["toasts"][0][:2] == ["success", "Copied Python example"]
    assert result["toasts"][1] == ["success", "Copied", "http://127.0.0.1:34001/v1"]
    assert result["toasts"][2][:2] == ["success", "Copied curl example"]
    assert result["toasts"][3] == ["error", "Could not copy", "Select the text and copy it manually."]
    assert result["codeFocusable"] is True
    assert result["codeLabel"] == "curl example"
