"""ExLlamaV3 via TabbyAPI (spec §6): uploads vm/setup_exllamav3.py (+ chat template) and maps a RunSpec onto its
EXL_* env. TabbyAPI only reports streamed `usage` when asked, so the proxy adds stream_options.include_usage
(only when the client didn't set stream_options itself)."""

from __future__ import annotations

import json
from pathlib import Path

from ..config import PACKAGE_DIR
from .base import VM_DIR, VM_ROOT, RunSpec

TABBY_COMMIT = "be74bf0a00bcb3a518e6feb7606f150c189be637"
WHEEL = "1.5.2"


class ExLlamaV3Engine:
    name = "exllamav3"
    health_path = "/health"
    inject_usage_option = True
    script = VM_DIR / "setup_exllamav3.py"

    def __init__(self, tabby_commit: str = TABBY_COMMIT, wheel: str = WHEEL) -> None:
        self.tabby_commit, self.wheel = tabby_commit, wheel

    def uploads(self, spec: RunSpec) -> list[tuple[Path, str]]:
        files = [(self.script, f"{VM_ROOT}/{self.script.name}")]
        if spec.template:
            files.append((PACKAGE_DIR / "chat_templates" / spec.template, f"{VM_ROOT}/templates/{spec.template}"))
        return files

    def setup_env(self, spec: RunSpec, vm_port: int, run_id: int) -> dict[str, str]:
        wheel, commit = self.wheel, self.tabby_commit
        if spec.release and "@" in spec.release:
            wheel, commit = spec.release.split("@", 1)
        env = {
            "SCRIPT": f"{VM_ROOT}/{self.script.name}",
            "OUT": f"{VM_ROOT}/setup.out",
            "EVENTS": f"{VM_ROOT}/run-{run_id}.jsonl",
            "ENGINE_LOG": f"{VM_ROOT}/engine.log",
            "EXL_TABBY_COMMIT": commit,
            "EXL_WHEEL": wheel,
            "EXL_REPO": spec.repo,
            "EXL_REVISION": spec.revision or "main",
            "EXL_PORT": str(vm_port),
            "EXL_MAX_SEQ_LEN": str(spec.ctx),
            "EXL_CACHE_MODE": spec.kv,
            "EXL_PARALLEL": str(spec.parallel),
            "EXL_DRAFT_MODE": "disabled",
            "EXL_VISION": "1" if spec.vision else "0",
            "EXL_VISION_OFFLOAD": "1" if spec.vision_offload else "0",
            "EXL_EXTRA": json.dumps(dict(argument.split("=", 1) for argument in spec.extra_args if "=" in argument)),
            "EXL_TEMPLATE": f"{VM_ROOT}/templates/{spec.template}" if spec.template else "",
        }
        return env
