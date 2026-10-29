"""Hardware-aware defaults: pick models and settings that fit this machine.

`detect()` inspects RAM, free memory, CPU cores and accelerators (Apple MLX/Metal,
CUDA, MPS). `plan()` turns that into a Plan: which models to run, whether to
unload models between runs (low-memory mode) and which backend to use.

Tiers (override with AIGM_TIER=lite|standard|full):
    lite      < 8 GB RAM or < 2 GB free   -> DEFAULT_MODELS, low-memory
    standard  typical laptop / CPU-only   -> DEFAULT_MODELS
    full      >= 16 GB RAM + accelerator  -> DEFAULT_MODELS

All tiers run the same two models: spectttra-alpha-120s + fakeprint was the best
ensemble in scripts/benchmark_ensemble.py, and it needs ~0.2 GB, so bigger
machines gain nothing from adding the heavier, less accurate AST models. Tiers
are kept so they can diverge again when a model earns a place.

Whatever the tier, models are dropped (lowest priority first) until their
measured memory fits the budget, and explicit model selections get warnings
instead of being silently changed.
"""

import logging
import os
import platform
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional

from aigm_detector.device import is_apple_silicon, mlx_available, resolve_backend

logger = logging.getLogger(__name__)

GB = 1024**3

# Python + torch + transformers resident memory before any model loads
RUNTIME_OVERHEAD_GB = 1.0
# Fraction of free memory we allow ourselves to use
BUDGET_FRACTION = 0.8

# Priority order: when memory is short, models are dropped from the end
TIER_MODELS: Dict[str, List[str]] = {
    "lite": ["spectttra-alpha-120s", "fakeprint"],
    "standard": ["spectttra-alpha-120s", "fakeprint"],
    "full": ["spectttra-alpha-120s", "fakeprint"],
}


@dataclass(frozen=True)
class HardwareProfile:
    """What this machine offers."""

    system: str  # "Darwin", "Linux", "Windows"
    machine: str  # "arm64", "x86_64"
    chip: str  # e.g. "Apple M2 Pro" or CPU brand
    cpu_cores: int
    total_ram_gb: float
    available_ram_gb: float
    apple_silicon: bool
    mlx: bool
    gpu_memory_gb: Optional[float]  # MLX recommended working set, or CUDA VRAM
    torch_device: str  # "cuda", "mps" or "cpu"

    @property
    def accelerated(self) -> bool:
        """True if there is a GPU the models can run on."""
        return self.torch_device in ("cuda", "mps") or (self.apple_silicon and self.mlx)


@dataclass
class Plan:
    """Settings chosen for this machine."""

    tier: str
    backend: str
    models: List[str]
    low_memory: bool
    memory_budget_gb: float
    required_gb: float
    notes: List[str] = field(default_factory=list)  # why these choices were made
    warnings: List[str] = field(default_factory=list)  # risks with an explicit selection
    excluded: Dict[str, str] = field(default_factory=dict)  # model -> reason


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def detect() -> HardwareProfile:
    """Inspect the machine (cached for the process)."""
    from aigm_detector.device import get_device

    apple = is_apple_silicon()
    has_mlx = mlx_available()
    total = _total_ram_gb()
    return HardwareProfile(
        system=platform.system(),
        machine=platform.machine(),
        chip=_chip_name(),
        cpu_cores=os.cpu_count() or 1,
        total_ram_gb=total,
        available_ram_gb=_available_ram_gb(total),
        apple_silicon=apple,
        mlx=has_mlx,
        gpu_memory_gb=_gpu_memory_gb(apple and has_mlx),
        torch_device=get_device().type,
    )


def _total_ram_gb() -> float:
    total = None
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / GB
    except (ValueError, OSError, AttributeError):
        try:
            import psutil

            total = psutil.virtual_memory().total / GB
        except Exception:
            total = 8.0  # conservative guess

    limit = _cgroup_limit_gb()  # containers can be capped below physical RAM
    return min(total, limit) if limit else total


def _cgroup_limit_gb() -> Optional[float]:
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = open(path).read().strip()
        except OSError:
            continue
        if raw.isdigit() and int(raw) < 1 << 60:
            return int(raw) / GB
    return None


def _available_ram_gb(total_gb: float) -> float:
    """Memory that can be used without swapping."""
    try:
        import psutil

        return min(psutil.virtual_memory().available / GB, total_gb)
    except Exception:
        pass

    try:  # Linux
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return min(int(line.split()[1]) * 1024 / GB, total_gb)
    except OSError:
        pass

    if platform.system() == "Darwin":  # free + inactive + speculative + purgeable pages
        try:
            out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=2).stdout
            page = int(out.split("page size of")[1].split()[0])
            pages = 0
            for line in out.splitlines():
                key, _, value = line.partition(":")
                if key.strip() in ("Pages free", "Pages inactive", "Pages speculative",
                                   "Pages purgeable"):
                    pages += int(value.strip().rstrip("."))
            return min(pages * page / GB, total_gb)
        except Exception:
            pass

    return total_gb * 0.5


def _chip_name() -> str:
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=2,
            ).stdout.strip()
            if out:
                return out
        except Exception:
            pass
    if platform.system() == "Linux":
        try:
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def _gpu_memory_gb(use_mlx: bool) -> Optional[float]:
    if use_mlx:
        try:
            import mlx.core as mx

            info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
            size = info.get("max_recommended_working_set_size") or info.get("memory_size")
            if size:
                return size / GB
        except Exception:
            pass
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / GB
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def model_memory_gb(name: str) -> float:
    """Measured peak inference memory (weights included) for a 3-minute track."""
    from aigm_detector.models import get_model

    return float(getattr(get_model(name), "memory_gb", 1.0))


def choose_tier(profile: HardwareProfile) -> str:
    forced = os.environ.get("AIGM_TIER", "").lower()
    if forced in TIER_MODELS:
        return forced
    if profile.total_ram_gb < 8 or profile.available_ram_gb < 2:
        return "lite"
    if profile.total_ram_gb >= 16 and profile.accelerated:
        return "full"
    return "standard"


def memory_budget_gb(profile: HardwareProfile) -> float:
    budget = profile.available_ram_gb * BUDGET_FRACTION
    if profile.gpu_memory_gb and profile.torch_device == "cuda":
        budget = min(budget, profile.gpu_memory_gb)  # discrete VRAM is the tighter limit
    return round(budget, 2)


def _required_gb(models: List[str], low_memory: bool) -> float:
    sizes = [model_memory_gb(m) for m in models] or [0.0]
    return RUNTIME_OVERHEAD_GB + (max(sizes) if low_memory else sum(sizes))


def plan(
    models: "Optional[str | List[str]]" = None,
    backend: Optional[str] = None,
    low_memory: Optional[bool] = None,
    profile: Optional[HardwareProfile] = None,
) -> Plan:
    """Choose models/settings for this machine.

    Args:
        models: None or "auto" lets the machine decide; anything else is honoured
            as given (see resolve_model_names) and only checked for fit.
        backend: "auto", "torch", "mlx" or None.
        low_memory: Force low-memory mode on/off; None decides from the budget.
        profile: Hardware to plan for (default: this machine).
    """
    from aigm_detector.models import resolve_model_names

    profile = profile or detect()
    tier = choose_tier(profile)
    budget = memory_budget_gb(profile)
    chosen_backend = resolve_backend(backend)
    notes = [
        f"{profile.chip}, {profile.total_ram_gb:.0f} GB RAM "
        f"({profile.available_ram_gb:.1f} GB free), "
        f"{'accelerated' if profile.accelerated else 'CPU only'} → tier '{tier}'"
    ]
    warnings: List[str] = []
    excluded: Dict[str, str] = {}

    automatic = models is None or (isinstance(models, str) and models.strip() == "auto")
    if automatic:
        names = list(TIER_MODELS[tier])
        if not profile.accelerated and "ast-60s" in names:
            names.remove("ast-60s")
            excluded["ast-60s"] = "too slow without a GPU (~30 s per track on CPU)"

        auto_low = tier == "lite" if low_memory is None else low_memory
        # Drop lowest-priority models until the set fits the budget
        while len(names) > 1 and _required_gb(names, auto_low) > budget:
            dropped = names.pop()
            excluded[dropped] = f"needs {model_memory_gb(dropped):.1f} GB, budget {budget:.1f} GB"
        if low_memory is None and _required_gb(names, False) > budget:
            auto_low = True
            notes.append("models don't fit in memory together → low-memory mode")
        low = auto_low
    else:
        names = resolve_model_names(models)
        low = bool(low_memory) if low_memory is not None else False
        if low_memory is None and _required_gb(names, False) > budget:
            low = True
            notes.append("selected models don't fit in memory together → low-memory mode")
        if not profile.accelerated and "ast-60s" in names:
            warnings.append("ast-60s is slow without a GPU (~30 s per track on CPU)")

    required = _required_gb(names, low)
    if required > budget:
        warnings.append(
            f"estimated {required:.1f} GB needed but only {budget:.1f} GB budget; "
            "expect swapping — close other apps or choose smaller models"
        )

    if chosen_backend == "torch" and profile.apple_silicon and not profile.mlx:
        notes.append("tip: pip install 'aigm-detector[mlx]' for native Apple Silicon inference")

    return Plan(
        tier=tier,
        backend=chosen_backend,
        models=names,
        low_memory=low,
        memory_budget_gb=budget,
        required_gb=round(required, 2),
        notes=notes,
        warnings=warnings,
        excluded=excluded,
    )
