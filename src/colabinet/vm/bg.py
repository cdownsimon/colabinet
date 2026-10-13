# Runs ON the Colab VM via `colab exec`: start $SCRIPT detached from the Jupyter kernel (output appended to $OUT).
# A long `colab exec` can drop its websocket ("Connection was lost."); a detached job survives that.
import os
import subprocess
import sys

script, out = os.environ["SCRIPT"], os.environ["OUT"]
os.makedirs(os.path.dirname(out), exist_ok=True)
events = os.environ.get("EVENTS")
if events:
    os.makedirs(os.path.dirname(events) or ".", exist_ok=True)
    with open(events, "w"):
        pass
proc = subprocess.Popen(
    [sys.executable, "-u", script],
    stdout=open(out, "a"),
    stderr=subprocess.STDOUT,
    cwd=os.path.dirname(script) or ".",
    start_new_session=True,
)
print(f"COLABINET_BG_PID {proc.pid}")
