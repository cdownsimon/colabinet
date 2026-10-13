"""Original Qwen3.8 Strata adapter; native CLI extras plus the upstream low-RAM setup selector."""

from __future__ import annotations

import json
from pathlib import Path

from .base import VM_DIR, VM_ROOT, RunSpec

RELEASE = "d6708a4aae15b4860000d54c8af9e84d684bce09"
REPO = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
QUANTS = ("IQ2_XS", "IQ3_XXS", "IQ3_S")
FILES = {f"{q}/Qwen3.8-Flash-Next-GSQ-RCO-{q}-00001-of-00002.gguf": q for q in QUANTS}
# These options would change managed settings, route setup/server options to the engine, or replace the model.
MANAGED_ARGS = {
    "--pack", "--native", "--ple-gguf", "--mtp", "--max-context", "--kv", "--vision", "--serve",
    "--control-vector", "--control-vector-scaled", "--control-vector-layer-range", "--cvec-mode", "--cvec-dir",
    "--host", "--port", "--config", "--engine", "--model", "--family", "--context", "--data-dir",
    "--experimental-speed-projection", "--no-start", "--build", "--yes", "--setup", "--mmproj",
}


def _split_setup_args(args: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
    mode, native, index, seen = "auto", [], 0, False
    while index < len(args):
        arg = args[index]
        if arg.startswith("--low-ram="):
            raise ValueError("Strata setup options use separate flag/value arguments, e.g. --low-ram on")
        if arg == "--low-ram":
            if seen or index + 1 == len(args) or args[index + 1] not in ("auto", "on", "off"):
                raise ValueError("Strata --low-ram requires one auto, on or off value")
            mode, seen = args[index + 1], True
            index += 2
        else:
            native.append(arg)
            index += 1
    return mode, tuple(native)


def validate_spec(spec: RunSpec) -> None:
    """Reject unsupported requests before lifecycle allocation or mutation."""
    if spec.repo != REPO or spec.file not in FILES:
        raise ValueError("Strata requires the original ISTA-DASLab Qwen3.8-Flash-Next-GSQ-RCO-GGUF repository "
                         "and exact IQ2_XS, IQ3_XXS or IQ3_S first-shard path")
    if spec.parallel != 1:
        raise ValueError("Strata supports parallel=1 only (FIFO serialized requests)")
    if spec.kv not in ("int8", "q4_0", "k8v4"):
        raise ValueError("Strata KV must be int8, q4_0 or k8v4")
    if not isinstance(spec.ctx, int) or isinstance(spec.ctx, bool) or not 0 < spec.ctx <= 262144:
        raise ValueError("Strata context must be a positive integer at most 262144")
    if spec.template or spec.mmproj or spec.mmproj_gpu or spec.vision_offload:
        raise ValueError("Strata manages its template and GPU vision projector; template/mmproj/offload overrides "
                         "are unsupported (use vision=true or false)")
    if spec.revision:
        raise ValueError("Strata upstream setup does not support model revision overrides")
    if spec.release and spec.release != RELEASE:
        raise ValueError(f"Strata source must be pinned to {RELEASE}")
    if not spec.alias or not isinstance(spec.vision, bool) or not isinstance(spec.refusal_projection, bool):
        raise ValueError("Strata requires a model alias and boolean vision/refusal_projection settings")
    for arg in spec.extra_args:
        if not isinstance(arg, str) or not arg or "\x00" in arg:
            raise ValueError("Strata extra_args must be nonempty engine CLI strings")
        if arg.split("=", 1)[0] in MANAGED_ARGS:
            raise ValueError(f"Strata extra_args cannot override managed option {arg!r}")
    _split_setup_args(spec.extra_args)
    for i, arg in enumerate(spec.extra_args):
        if arg == "--ple-io":
            if i + 1 == len(spec.extra_args) or spec.extra_args[i + 1] not in ("direct", "ram", "mmap"):
                raise ValueError("Strata --ple-io requires direct, ram or mmap")
        elif arg.startswith("--ple-io="):
            raise ValueError("Strata engine options use separate flag/value arguments, e.g. --ple-io ram")


class StrataEngine:
    name = "strata"
    health_path = "/health"
    script = VM_DIR / "setup_strata.py"
    validate_spec = staticmethod(validate_spec)

    def uploads(self, spec: RunSpec) -> list[tuple[Path, str]]:
        validate_spec(spec)
        return [(self.script, f"{VM_ROOT}/{self.script.name}")]

    def setup_env(self, spec: RunSpec, vm_port: int, run_id: int) -> dict[str, str]:
        validate_spec(spec)
        low_ram, native_args = _split_setup_args(spec.extra_args)
        # Colab exec retains environment variables: always send explicit on/off and empty extra args.
        return {
            "SCRIPT": f"{VM_ROOT}/{self.script.name}", "OUT": f"{VM_ROOT}/setup.out",
            "EVENTS": f"{VM_ROOT}/run-{run_id}.jsonl", "ENGINE_LOG": f"{VM_ROOT}/engine.log",
            "STRATA_RELEASE": spec.release or RELEASE, "STRATA_REPO": spec.repo, "STRATA_FILE": spec.file,
            "STRATA_QUANT": FILES[spec.file], "STRATA_ALIAS": spec.alias, "STRATA_PORT": str(vm_port),
            "STRATA_CTX": str(spec.ctx), "STRATA_KV": spec.kv,
            "STRATA_VISION": "gpu" if spec.vision else "none",
            "STRATA_PROJECTION": "on" if spec.refusal_projection else "off",
            "STRATA_LOW_RAM": low_ram, "STRATA_EXTRA_ARGS": json.dumps(list(native_args)),
            "STRATA_API_KEY": "", "STRATA_PREBUILT_URL": "", "STRATA_GPU": "",
        }
