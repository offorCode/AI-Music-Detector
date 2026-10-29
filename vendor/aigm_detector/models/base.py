"""Base detector interface and result types."""

import gc
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from aigm_detector.device import Backend, resolve_backend
from aigm_detector.preprocess import AudioInput

logger = logging.getLogger(__name__)


@dataclass
class DetectorResult:
    """Result from a single detector model."""

    model_name: str
    score: float  # p(AI-generated), 0.0-1.0
    confidence: float  # Calibrated confidence (0.0-1.0)
    processing_time_ms: float
    metadata: Dict[str, Any]  # Model-specific extras


class BaseDetector(ABC):
    """Abstract base class for all detection models.

    Provides:
    - Lazy loading with graceful failure handling
    - Backend selection (PyTorch or MLX)
    - Consistent predict interface
    - Automatic error logging
    """

    name: str = ""  # Model identifier (must be set by subclass)
    requires: List[str] = []  # AudioInput fields needed (e.g. ["waveform_16k"])
    description: str = ""  # One-line summary shown by `aigm-detector models`
    repo_id: Optional[str] = None  # Hugging Face checkpoint
    backends: Tuple[str, ...] = ("torch",)  # Backends this detector implements
    memory_gb: float = 1.0  # Peak inference memory incl. weights, 3-min track
    cpu_seconds: Optional[float] = None  # PyTorch CPU latency, 3-min track, 4 cores

    def __init__(self, backend: Optional[str] = None):
        self._loaded: bool = False
        self._load_error: Optional[str] = None
        self.backend: Backend = self._pick_backend(backend)

    def _pick_backend(self, requested: Optional[str]) -> Backend:
        backend = resolve_backend(requested)
        if backend not in self.backends:
            logger.info(f"{self.name}: backend '{backend}' unsupported, falling back to torch")
            return "torch"
        return backend

    def is_available(self) -> bool:
        """Check if model loaded successfully."""
        return self._loaded and self._load_error is None

    def ensure_loaded(self) -> bool:
        """Lazy loading with error handling.

        Returns:
            True if model is loaded and ready, False if loading failed
        """
        if self._loaded:
            return self._load_error is None

        try:
            logger.info(f"Loading model: {self.name} ({self.backend})")
            self._load_weights()
            self._loaded = True
            logger.info(f"Model {self.name} loaded successfully")
            return True

        except Exception as e:
            self._loaded = True  # Don't retry a failed load on every file
            self._load_error = str(e)
            logger.error(f"Model {self.name} failed to load: {e}", exc_info=True)
            return False

    def unload(self) -> None:
        """Release weights so the next predict reloads them (low-memory mode)."""
        self._release_weights()
        self._loaded = False
        self._load_error = None
        gc.collect()

        try:
            import torch

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def predict(self, audio: AudioInput) -> Optional[DetectorResult]:
        """Public predict interface with lazy loading.

        Args:
            audio: Preprocessed audio input

        Returns:
            DetectorResult if successful, None if model unavailable
        """
        if not self.ensure_loaded():
            logger.warning(f"Model {self.name} unavailable, skipping prediction")
            return None

        try:
            start_time = time.perf_counter()
            result = self._predict_impl(audio)
            result.processing_time_ms = (time.perf_counter() - start_time) * 1000
            result.metadata.setdefault("backend", self.backend)
            return result

        except Exception as e:
            logger.error(f"Prediction failed for {self.name}: {e}", exc_info=True)
            return None

    @abstractmethod
    def _load_weights(self) -> None:
        """Internal method to download/load weights.

        Raises:
            Exception if loading fails (will be caught by ensure_loaded)
        """

    @abstractmethod
    def _predict_impl(self, audio: AudioInput) -> DetectorResult:
        """Internal prediction implementation.

        Raises:
            Exception if prediction fails (will be caught by predict)
        """

    def _release_weights(self) -> None:
        """Drop references to loaded weights. Override in subclasses."""

    def __repr__(self) -> str:
        status = "available" if self.is_available() else "unavailable"
        if self._load_error:
            status = f"error: {self._load_error[:50]}"
        return f"<{self.__class__.__name__}({self.name}, {self.backend}): {status}>"
