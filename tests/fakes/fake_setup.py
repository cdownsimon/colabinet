# Stand-in for vm/setup_llamacpp.py in integration tests: same events, but starts tests/fakes/fake_engine.py.
# FAKE_SETUP_FAIL=1 makes it report an error instead.
import json
import os
import subprocess
import sys
import time
import urllib.request

E = os.environ.get
EVENTS, LOG, PORT = E("EVENTS"), E("ENGINE_LOG"), E("LLM_PORT")
T0 = time.time()


def event(kind, **kw):
    with open(EVENTS, "a") as f:
        f.write(json.dumps({"t": round(time.time() - T0, 2), "event": kind, **kw}) + "\n")


event("gpu_info", nvidia_smi="fake", torch="2.11.0 12.8")
if E("FAKE_SETUP_FAIL") == "1":
    event("error", message="download of model.gguf failed: 404", log_tail="")
    sys.exit(1)
event("install_done", engine="llamacpp", release=E("LLM_RELEASE"), seconds=0.1, skipped=False)
event("download_done", file=E("LLM_FILE"), gb=0.01, seconds=0.1, mb_s=100.0)
subprocess.Popen([sys.executable, E("FAKE_ENGINE"), "--port", PORT, "--alias", E("LLM_ALIAS", "fake-model")],
                 stdout=open(LOG, "a"), stderr=subprocess.STDOUT)
for _ in range(100):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1) as r:
            if r.status == 200:
                break
    except Exception:
        time.sleep(0.1)
event("server_ready", seconds=0.2, vram_used_mib=1000, modalities={"vision": False}, cmd="fake")
