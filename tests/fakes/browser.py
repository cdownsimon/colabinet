import json
import os
import shutil
import signal
import subprocess
from pathlib import Path

import pytest

from colabinet.config import PACKAGE_DIR

from . import FAKES


def require_browser():
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    mac_chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if chrome is None and mac_chrome.is_file():
        chrome = str(mac_chrome)
    node = shutil.which("node")
    if chrome is None or node is None:
        pytest.skip("Chrome and Node are needed for browser regressions")
    result = subprocess.run([node, "-e", "process.exit(typeof WebSocket === 'function' ? 0 : 1)"],
                            capture_output=True, timeout=10)
    if result.returncode:
        pytest.skip("Node with built-in WebSocket is needed for browser regressions")
    return chrome, node


def run_browser(tmp_path, page, script):
    chrome, node = require_browser()
    if not page.startswith(("http://", "https://", "file://")):
        fixture = tmp_path / "page.html"
        fixture.write_text(page.replace("/static/", (PACKAGE_DIR / "static").as_uri() + "/"))
        page = fixture.as_uri()
    source = tmp_path / "browser-test.cjs"
    source.write_text(script)
    # Own process group, so a timeout also ends the Chrome that Node started and the test fails instead of hanging.
    process = subprocess.Popen([node, str(FAKES / "browser.cjs"), chrome, page, str(tmp_path / "chrome"), str(source)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=75)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        pytest.fail(f"browser test timed out after 75 s: {stderr[-2000:]}")
    assert process.returncode == 0, stderr
    return json.loads(stdout)
