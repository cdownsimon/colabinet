"""llama.cpp adapter: uploads vm/setup_llamacpp.py (+ chat template) and maps a RunSpec onto its LLM_* env."""

from __future__ import annotations

import json
from pathlib import Path

from ..config import PACKAGE_DIR
from .base import VM_DIR, VM_ROOT, RunSpec


class LlamaCppEngine:
    name = "llamacpp"
    health_path = "/health"
    script = VM_DIR / "setup_llamacpp.py"

    def uploads(self, spec: RunSpec) -> list[tuple[Path, str]]:
        files = [(self.script, f"{VM_ROOT}/{self.script.name}"), (VM_DIR / "gguf_files.py", f"{VM_ROOT}/gguf_files.py")]
        if spec.template:
            files.append((PACKAGE_DIR / "chat_templates" / spec.template, f"{VM_ROOT}/templates/{spec.template}"))
        return files

    def setup_env(self, spec: RunSpec, vm_port: int, run_id: int) -> dict[str, str]:
        env = {
            "SCRIPT": f"{VM_ROOT}/{self.script.name}",
            "OUT": f"{VM_ROOT}/setup.out",
            "EVENTS": f"{VM_ROOT}/run-{run_id}.jsonl",
            "ENGINE_LOG": f"{VM_ROOT}/engine.log",
            "LLM_RELEASE": spec.release or "b11273",
            "LLM_REPO": spec.repo,
            "LLM_FILE": spec.file or "",
            "LLM_PORT": str(vm_port),
            "LLM_CTX": str(spec.ctx),
            "LLM_KV": spec.kv,
            "LLM_PARALLEL": str(spec.parallel),
            "LLM_ALIAS": spec.alias,
            "LLM_MMPROJ_FILE": spec.mmproj or "",
            "LLM_MMPROJ_GPU": "1" if spec.mmproj_gpu else "0",
            "LLM_EXTRA_ARGS": json.dumps(list(spec.extra_args)),
        }
        if spec.template:
            env["LLM_TEMPLATE"] = f"{VM_ROOT}/templates/{spec.template}"
        return env
