from .base import VM_ROOT, Engine, RunSpec
from .exllamav3 import ExLlamaV3Engine
from .llamacpp import LlamaCppEngine
from .strata import StrataEngine

ENGINES: dict[str, Engine] = {"llamacpp": LlamaCppEngine(), "exllamav3": ExLlamaV3Engine(), "strata": StrataEngine()}

__all__ = ["ENGINES", "VM_ROOT", "Engine", "ExLlamaV3Engine", "LlamaCppEngine", "StrataEngine", "RunSpec"]
