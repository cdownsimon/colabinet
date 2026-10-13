import json
import shutil
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
    result = subprocess.run([node, str(FAKES / "browser.cjs"), chrome, page, str(tmp_path / "chrome"), str(source)],
                            capture_output=True, text=True, timeout=75)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)
