# Runs ON the Colab VM via `colab exec` (the CLI sends this file's text, so nothing needs uploading first).
# Creates colabinet's directories and writes the session marker. The Contents API behind `colab upload` does not
# create parent directories (HTTP 500, seen on a real runtime), so this runs before any upload.
import json
import os

ROOT = os.environ.get("COLABINET_ROOT", "/content/colabinet")
for d in (ROOT, os.path.join(ROOT, "templates")):
    os.makedirs(d, exist_ok=True)
marker = os.environ.get("MARKER")
if marker:
    with open(os.path.join(ROOT, "session.json"), "w") as f:
        json.dump(json.loads(marker), f)
print("COLABINET_STATUS " + json.dumps({"prepared": ROOT, "marker": bool(marker)}))
