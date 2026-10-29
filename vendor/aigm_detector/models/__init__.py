"""Model registry and auto-discovery."""

import logging
from typing import Dict, List, Type

from aigm_detector.models.base import BaseDetector

logger = logging.getLogger(__name__)

# Global model registry
MODEL_REGISTRY: Dict[str, Type[BaseDetector]] = {}

# The "default" selection: the best ensemble in scripts/benchmark_ensemble.py.
# Two complementary views: SpecTTTra-α-120s models two minutes of song structure,
# fakeprint looks for vocoder peaks in the average spectrum.
DEFAULT_MODELS: List[str] = ["spectttra-alpha-120s", "fakeprint"]

_builtins_loaded = False


def register_model(cls: Type[BaseDetector]) -> Type[BaseDetector]:
    """Decorator to register a detector model.

    Usage:
        @register_model
        class MyDetector(BaseDetector):
            name = "mymodel"
            ...
    """
    if not hasattr(cls, "name") or not cls.name:
        raise ValueError(f"{cls.__name__} must define 'name' attribute")

    if cls.name in MODEL_REGISTRY:
        logger.warning(f"Model '{cls.name}' already registered. Overwriting.")

    MODEL_REGISTRY[cls.name] = cls
    logger.debug(f"Registered model: {cls.name}")
    return cls


def get_available_models() -> Dict[str, Type[BaseDetector]]:
    """Get all registered models.

    Returns:
        Dictionary mapping model names to detector classes
    """
    _ensure_models_loaded()
    return MODEL_REGISTRY.copy()


def get_model(name: str) -> Type[BaseDetector]:
    """Get a specific model by name.

    Raises:
        KeyError: Model not found
    """
    _ensure_models_loaded()

    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise KeyError(f"Model '{name}' not found. Available: {available}")

    return MODEL_REGISTRY[name]


def resolve_model_names(spec: "str | List[str] | None") -> List[str]:
    """Expand a model selection into registry names.

    Accepts None/"auto" (chosen for this machine, see aigm_detector.resources),
    "default" (DEFAULT_MODELS), "all", or a comma-separated list / list of names.
    Family prefixes also work: "spectttra" selects every variant.
    """
    _ensure_models_loaded()

    if spec is None or (isinstance(spec, str) and spec.strip() == "auto"):
        from aigm_detector.resources import plan

        return plan().models
    items = spec.split(",") if isinstance(spec, str) else list(spec)
    items = [s.strip() for s in items if s.strip()]

    names: List[str] = []
    for item in items:
        if item == "default":
            matches = DEFAULT_MODELS
        elif item == "all":
            matches = list(MODEL_REGISTRY)
        elif item in MODEL_REGISTRY:
            matches = [item]
        else:
            matches = [n for n in MODEL_REGISTRY if n.startswith(item + "-")]
            if not matches:
                logger.warning(f"Model '{item}' not found, skipping")
        names.extend(m for m in matches if m not in names)
    return names


def _ensure_models_loaded():
    """Register the built-in detectors (once)."""
    global _builtins_loaded
    if _builtins_loaded:
        return
    _builtins_loaded = True

    # AST (trained on Suno/Udio/MusicGen + human music)
    try:
        from aigm_detector.models.ast_detector import AST60Detector, ASTDetector

        register_model(ASTDetector)
        register_model(AST60Detector)
    except ImportError as e:
        logger.warning(f"Failed to import ast_detector: {e}")

    # SONICS SpecTTTra family (trained on Suno/Udio + real songs)
    try:
        from aigm_detector.models.spectttra_detector import SPECTTTRA_VARIANTS

        for cls in SPECTTTRA_VARIANTS:
            register_model(cls)
    except ImportError as e:
        logger.warning(f"Failed to import spectttra_detector: {e}")

    # Spectral fakeprint + logistic regression (Afchar et al., ISMIR 2025)
    try:
        from aigm_detector.models.fakeprint_detector import FakeprintDetector

        register_model(FakeprintDetector)
    except ImportError as e:
        logger.warning(f"Failed to import fakeprint_detector: {e}")


__all__ = [
    "MODEL_REGISTRY",
    "DEFAULT_MODELS",
    "register_model",
    "get_available_models",
    "get_model",
    "resolve_model_names",
    "BaseDetector",
]
