"""Device and backend selection (PyTorch CPU/CUDA/MPS, or Apple MLX)."""

import logging
import os
import platform
from functools import lru_cache
from typing import Literal, Optional

import torch

logger = logging.getLogger(__name__)

Backend = Literal["torch", "mlx"]
BACKENDS = ("auto", "torch", "mlx")


@lru_cache(maxsize=1)
def get_device() -> torch.device:
    """Get the best available PyTorch device (CUDA > MPS > CPU).

    Override with the AIGM_DEVICE environment variable (e.g. "cpu").
    Cached to avoid repeated checks.
    """
    override = os.environ.get("AIGM_DEVICE")
    if override:
        device = torch.device(override)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"Using torch device: {device}")
    return device


def is_apple_silicon() -> bool:
    """True on macOS running natively on an M-series chip."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


@lru_cache(maxsize=1)
def mlx_available() -> bool:
    """True if the `mlx` package can be imported."""
    try:
        import mlx.core  # noqa: F401
    except Exception:
        return False
    return True


def resolve_backend(requested: Optional[str] = None) -> Backend:
    """Resolve a backend request ("auto", "torch", "mlx") to a concrete backend.

    Resolution order: explicit argument, then AIGM_BACKEND env var, then "auto".
    "auto" picks MLX on Apple Silicon when it is installed, PyTorch otherwise.

    Raises:
        ValueError: Unknown backend name
        RuntimeError: MLX requested but not installed
    """
    requested = (requested or os.environ.get("AIGM_BACKEND") or "auto").lower()

    if requested not in BACKENDS:
        raise ValueError(f"Unknown backend '{requested}'. Choose from: {', '.join(BACKENDS)}")

    if requested == "auto":
        return "mlx" if is_apple_silicon() and mlx_available() else "torch"

    if requested == "mlx" and not mlx_available():
        raise RuntimeError(
            "MLX backend requested but `mlx` is not installed. "
            "Install with: pip install 'aigm-detector[mlx]'"
        )

    return requested  # type: ignore[return-value]


def describe_environment() -> dict:
    """Summarize the runtime environment (used by `aigm-detector doctor`)."""
    info = {
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "apple_silicon": is_apple_silicon(),
        "torch": torch.__version__,
        "torch_device": str(get_device()),
        "mlx": None,
        "mlx_device": None,
        "default_backend": resolve_backend("auto"),
    }
    if mlx_available():
        import mlx.core as mx

        info["mlx"] = getattr(mx, "__version__", "unknown")
        info["mlx_device"] = str(mx.default_device())
    return info
