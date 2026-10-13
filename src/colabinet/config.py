"""Static facts (GPU classes, seed rates) and runtime settings."""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
SOURCE_CHECKOUT = (REPO_ROOT / "pyproject.toml").is_file()
DEFAULT_DATA_DIR = REPO_ROOT / "data" if SOURCE_CHECKOUT else Path.home() / ".local/share/colabinet"
DEFAULT_PRESETS_PATH = REPO_ROOT / "presets.toml" if SOURCE_CHECKOUT else PACKAGE_DIR / "presets.toml"


def default_ssh_public_key(key: Path) -> Path | None:
    """The CLI can use its default public key directly, allowing OpenSSH to authenticate via ssh-agent."""
    default_key = Path.home() / ".ssh/id_ed25519"
    public = default_key.with_name(default_key.name + ".pub")
    if key.resolve() == default_key.resolve() and public.is_file():
        return public
    return None

# Rate classes. A class is what we bill by: the requested one at create, the detected one after the GPU check.
GPU_CLASSES = ("CPU", "T4", "T4-HIGHMEM", "L4", "A100", "A100-80", "G4")

# CU/h seed table (docs/colab-facts.md). Learned rates overwrite these in the gpu_rates table.
SEED_RATES = {
    "CPU": (0.08, "measured"),
    "T4": (1.07, "third-party"),
    "T4-HIGHMEM": (1.27, "third-party"),
    "L4": (1.54, "measured"),
    "A100": (5.30, "measured"),
    "A100-80": (6.77, "measured"),
    "G4": (8.90, "measured"),
}

VRAM_GB = {"CPU": 0.0, "T4": 16.0, "T4-HIGHMEM": 16.0, "L4": 22.5, "A100": 40.0, "A100-80": 80.0, "G4": 96.0}

# Host RAM references (decimal GB) as read on each class by the 2026-10-13 preset test matrix; see
# docs/colab-facts.md. Allocations vary. The console prefers actual readings from recorded sessions.
SYSTEM_RAM_GB = {"CPU": 13.6, "T4": 13.6, "T4-HIGHMEM": 54.8, "L4": 56.9,
                 "A100": 89.6, "A100-80": 179.4, "G4": 189.9}


def cli_flags(gpu_class: str) -> list[str]:
    """`colab new` flags for a rate class."""
    flags = {
        "CPU": [],
        "T4": ["--gpu", "T4"],
        "T4-HIGHMEM": ["--gpu", "T4", "--high-mem"],
        "L4": ["--gpu", "L4"],
        "A100": ["--gpu", "A100"],
        "A100-80": ["--gpu", "A100", "--high-mem"],
        "G4": ["--gpu", "G4"],
    }
    if gpu_class not in flags:
        raise ValueError(f"unknown GPU class {gpu_class!r}")
    return flags[gpu_class]


def detected_class(gpu_name: str | None, vram_mib: int | None, ram_gb: float | None) -> str:
    """Map what nvidia-smi and the RAM size report to a rate class ("UNKNOWN" if it matches none)."""
    if not gpu_name:
        return "CPU"
    name = gpu_name.upper()
    if "A100" in name:
        return "A100-80" if (vram_mib or 0) > 60_000 else "A100"
    if "L4" in name:
        return "L4"
    if "T4" in name:
        return "T4-HIGHMEM" if (ram_gb or 0) > 30 else "T4"
    if "RTX PRO 6000" in name or "BLACKWELL" in name:
        return "G4"
    return "UNKNOWN"


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: DEFAULT_DATA_DIR)
    colab_bin: str = field(default_factory=lambda: shutil.which("colab") or str(Path.home() / ".local/bin/colab"))
    colab_auth: str = "oauth2"
    ssh_key: Path = field(default_factory=lambda: Path.home() / ".ssh/id_ed25519")
    presets_path: Path = field(default_factory=lambda: DEFAULT_PRESETS_PATH)
    ui_port: int = 7860
    proxy_port_start: int = 3080
    hidden_port_range: tuple[int, int] = (23000, 23999)
    vm_port_default: int = 8090
    idle_min_default: int = 30
    daily_cap_cu: float = 10.0
    low_balance_cu: float = 10.0
    setup_limit_s: float = 40 * 60
    gpu_check_limit_s: float = 5 * 60
    grace_s: float = 10 * 60
    tunnel_alert_s: float = 10 * 60
    create_slow_s: float = 5 * 60
    create_abandon_s: float = 30 * 60
    endpoint_wait_s: float = 5 * 60
    keepalive_s: float = 100
    setup_poll_s: float = 15
    reconcile_s: float = 60
    guard_s: float = 30
    usage_s: float = 300
    stop_retry_s: float = 30
    notify: bool = True
    stop_on_exit: bool = False

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir).expanduser().resolve()
        expanded_bin = os.path.expanduser(self.colab_bin)
        self.colab_bin = shutil.which(expanded_bin) or expanded_bin
        self.ssh_key = Path(self.ssh_key).expanduser().resolve()

    @property
    def db_path(self) -> Path:
        return self.data_dir / "colabinet.db"

    @property
    def lock_path(self) -> Path:
        return self.data_dir / "colabinet.lock"

    @property
    def token_path(self) -> Path:
        return self.data_dir / "api.token"

    @property
    def log_dir(self) -> Path:
        return self.data_dir / "logs"


class Clock:
    """Wall-clock time. Guards use wall time because macOS monotonic time stops during sleep."""

    def now(self) -> float:
        return time.time()

    def day_bounds(self, t: float) -> tuple[float, float]:
        """Local-calendar-day [start, end) containing t."""
        lt = time.localtime(t)
        start = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
        nxt = time.localtime(start + 26 * 3600)
        end = time.mktime((nxt.tm_year, nxt.tm_mon, nxt.tm_mday, 0, 0, 0, 0, 0, -1))
        return start, end

    def day_key(self, t: float) -> str:
        return time.strftime("%Y-%m-%d", time.localtime(t))
