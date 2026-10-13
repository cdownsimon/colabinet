import sys

import pytest

from tests.fakes import browser

pytestmark = pytest.mark.slow


def test_browser_waits_for_debugging_port_file_to_finish_writing(tmp_path, monkeypatch):
    chrome, node = browser.require_browser()
    wrapper = tmp_path / "delayed-chrome"
    wrapper.write_text(f"""#!{sys.executable}
import os
import sys
import time
from pathlib import Path

profile = Path(next(argument.split('=', 1)[1] for argument in sys.argv if argument.startswith('--user-data-dir=')))
profile.mkdir(parents=True, exist_ok=True)
(profile / 'DevToolsActivePort').write_text('')
time.sleep(0.2)
os.execv({chrome!r}, [{chrome!r}, *sys.argv[1:]])
""")
    wrapper.chmod(0o700)
    monkeypatch.setattr(browser, "require_browser", lambda: (str(wrapper), node))
    directory = tmp_path / "run"
    directory.mkdir()
    result = browser.run_browser(directory, "<html><head><title>Ready</title></head><body></body></html>",
                                 "return await evaluate('document.title');")
    assert result == "Ready"
