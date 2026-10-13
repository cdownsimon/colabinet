"""Model presets from presets.toml (stdlib tomllib). Errors are reported, never fatal to the app."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

KNOWN_ENGINES = {"llamacpp", "exllamav3", "strata"}
FALLBACK_CLASS = {"A100-80": "A100", "T4-HIGHMEM": "T4"}


class PresetError(ValueError):
    pass


@dataclass(frozen=True)
class GpuDefaults:
    ctx: int = 32768
    kv: str = "f16"
    mmproj_gpu: bool = False
    extra_args: tuple[str, ...] = ()  # engine args for this GPU class, e.g. Strata's ("--low-ram", "on")


@dataclass(frozen=True)
class GpuEvidence:
    status: str
    message: str
    performance: str = ""  # Saved benchmark summary, separate from local request aggregates.


@dataclass(frozen=True)
class Measured:
    """Speeds measured on a real runtime of one GPU class at the preset's defaults, engine-reported."""
    gen_tps: float  # generation, on a short prompt
    prefill_tps: float | None = None  # prompt processing on a ~4K-token prompt; None when too slow to try
    date: str = ""
    note: str = ""


@dataclass(frozen=True)
class Preset:
    id: str
    engine: str
    label: str
    repo: str
    file: str | None = None
    revision: str | None = None
    mmproj: str | None = None
    template: str | None = None
    alias: str | None = None
    weights_gb: float | None = None
    kv_bytes_per_token_f16: int | None = None
    licence: str | None = None
    parallel: int = 1
    gpu: dict[str, GpuDefaults] = field(default_factory=dict)
    vision: bool = False  # Bundled ExLlamaV3/Strata vision; llama.cpp uses mmproj instead.
    vision_offload: bool = False  # Keep ExLlamaV3 vision weights in pinned system RAM.
    refusal_projection: bool = False  # Strata's experimental refusal-direction projection, opt-in.
    evidence: dict[str, GpuEvidence] = field(default_factory=dict)
    measured: dict[str, Measured] = field(default_factory=dict)

    def defaults_for(self, gpu_class: str) -> GpuDefaults:
        generic = GpuDefaults(kv={"exllamav3": "FP16", "strata": "int8"}.get(self.engine, "f16"))
        return self.gpu.get(gpu_class) or self.gpu.get(FALLBACK_CLASS.get(gpu_class, "")) or generic


@dataclass(frozen=True)
class Presets:
    engines: dict[str, dict]
    presets: dict[str, Preset]


def load_presets(path: Path) -> Presets:
    try:
        data = tomllib.loads(Path(path).read_text())
    except FileNotFoundError as e:
        raise PresetError(f"{path} not found") from e
    except tomllib.TOMLDecodeError as e:
        raise PresetError(f"{path}: {e}") from e
    presets: dict[str, Preset] = {}
    for i, raw in enumerate(data.get("preset", [])):
        where = f"preset #{i + 1}"
        for key in ("id", "engine", "label", "repo"):
            if not raw.get(key):
                raise PresetError(f"{where}: missing '{key}'")
        if raw["engine"] not in KNOWN_ENGINES:
            raise PresetError(f"{where}: unknown engine {raw['engine']!r}")
        if raw["engine"] in ("llamacpp", "strata") and not raw.get("file"):
            raise PresetError(f"{where}: {raw['engine']} presets need 'file'")
        if raw["engine"] == "exllamav3" and not raw.get("revision"):
            raise PresetError(f"{where}: exllamav3 presets need 'revision' (the bitrate branch, e.g. 4.00bpw)")
        for key in ("vision", "vision_offload", "refusal_projection"):
            if key in raw and not isinstance(raw[key], bool):
                raise PresetError(f"{where}: {key} must be a boolean (true or false)")
        if raw.get("refusal_projection") and raw["engine"] != "strata":
            raise PresetError(f"{where}: refusal_projection is supported only by Strata")
        if raw["id"] in presets:
            raise PresetError(f"{where}: duplicate id {raw['id']!r}")
        fields = {k: v for k, v in raw.items() if k not in ("gpu", "evidence", "measured")}
        try:
            gpu = {}
            for name, values in raw.get("gpu", {}).items():
                extra = values.get("extra_args", [])
                if not isinstance(extra, list) or not all(isinstance(item, str) for item in extra):
                    raise ValueError(f"gpu.{name}.extra_args must be a list of strings")
                gpu[name] = GpuDefaults(**{**GpuDefaults().__dict__, **values, "extra_args": tuple(extra)})
            evidence = {k: GpuEvidence(**v) for k, v in raw.get("evidence", {}).items()}
            if any(e.status not in ("tested", "reported", "build_only") for e in evidence.values()):
                raise ValueError("evidence status must be tested, reported or build_only")
            measured = {k: Measured(**v) for k, v in raw.get("measured", {}).items()}
            for name, speed in measured.items():
                if not all(isinstance(value, (int, float)) and value > 0
                           for value in (speed.gen_tps, speed.prefill_tps) if value is not None):
                    raise ValueError(f"measured.{name}: gen_tps and prefill_tps must be positive numbers")
            presets[raw["id"]] = Preset(**fields, gpu=gpu, evidence=evidence, measured=measured)
        except (TypeError, ValueError) as e:
            raise PresetError(f"{where}: {e}") from e
    return Presets(engines=data.get("engines", {}), presets=presets)
