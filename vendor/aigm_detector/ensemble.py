"""Ensemble detection with weighted fusion."""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Union

from aigm_detector.models import get_model
from aigm_detector.models.base import BaseDetector, DetectorResult
from aigm_detector.preprocess import AudioInput
from aigm_detector.resources import Plan, plan

logger = logging.getLogger(__name__)

Verdict = Literal["AI", "HUMAN", "UNCERTAIN"]


@dataclass
class EnsembleResult:
    """Result from ensemble detector."""

    verdict: Verdict
    ai_probability: float  # 0.0-1.0
    threshold: float
    model_scores: Dict[str, DetectorResult]
    dominant_signal: str  # Model furthest from the decision boundary
    processing_time_ms: float
    warnings: List[str]  # Any warnings (e.g. single model)
    backend: str = "torch"

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable summary (used by CLI --json and the API)."""
        return {
            "verdict": self.verdict,
            "ai_probability": round(self.ai_probability, 4),
            "threshold": self.threshold,
            "backend": self.backend,
            "model_scores": {k: round(v.score, 4) for k, v in self.model_scores.items()},
            "models": {
                k: {
                    "score": round(v.score, 4),
                    "backend": v.metadata.get("backend"),
                    "processing_time_ms": round(v.processing_time_ms, 1),
                    "window_scores": v.metadata.get("window_scores"),
                }
                for k, v in self.model_scores.items()
            },
            "dominant_signal": self.dominant_signal,
            "processing_time_ms": round(self.processing_time_ms, 1),
            "warnings": self.warnings,
        }


class EnsembleDetector:
    """Ensemble detector with configurable fusion strategy."""

    # Equal weights until calibrated on a held-out benchmark
    DEFAULT_WEIGHTS: Dict[str, float] = {}

    def __init__(
        self,
        model_names: Optional[Union[str, List[str]]] = None,
        strategy: Literal["mean", "weighted_mean", "max"] = "weighted_mean",
        weights: Optional[Dict[str, float]] = None,
        threshold: float = 0.5,
        uncertain_margin: float = 0.15,
        backend: Optional[str] = None,
        low_memory: Optional[bool] = None,
    ):
        """Initialize ensemble detector.

        Args:
            model_names: Model names, "all", "default", "auto" or a comma-separated
                string (None = "auto": chosen for this machine's resources)
            strategy: Fusion strategy (weighted_mean, mean, max)
            weights: Custom weights for weighted_mean strategy
            threshold: Classification threshold (default 0.5)
            uncertain_margin: Margin for UNCERTAIN verdict (default ±0.15)
            backend: "auto", "torch" or "mlx" (None = AIGM_BACKEND env or auto)
            low_memory: Unload each model after it runs (None = decide from free memory)
        """
        self.strategy = strategy
        self.threshold = threshold
        self.uncertain_margin = uncertain_margin
        self.plan: Plan = plan(model_names, backend=backend, low_memory=low_memory)
        self.backend = self.plan.backend
        self.low_memory = self.plan.low_memory
        self.weights = weights if weights is not None else self.DEFAULT_WEIGHTS.copy()

        for note in self.plan.notes:
            logger.info(note)
        for warning in self.plan.warnings:
            logger.warning(warning)

        self.models = self._load_models(self.plan.models)

        if not self.models:
            raise RuntimeError("No models available. At least one model must load successfully.")

        model_names_str = ", ".join(m.name for m in self.models)
        logger.info(f"Ensemble initialized with models: {model_names_str} ({self.backend})")
        logger.info(f"Strategy: {strategy}, Threshold: {threshold}")

    def _load_models(self, names: List[str]) -> List[BaseDetector]:
        """Instantiate detector objects (weights load lazily on first predict)."""
        if not names:
            logger.error("No valid models specified")
            return []

        models = []
        for name in names:
            try:
                models.append(get_model(name)(backend=self.backend))
            except Exception as e:
                logger.error(f"Failed to instantiate {name}: {e}")
        return models

    def load(self) -> List[str]:
        """Eagerly load (download/convert) all models. Returns the ones that loaded."""
        return [m.name for m in self.models if m.ensure_loaded()]

    def predict(self, audio: AudioInput) -> EnsembleResult:
        """Run ensemble prediction.

        Raises:
            RuntimeError: No models available
        """
        start_time = time.perf_counter()
        warnings = list(self.plan.warnings)

        results: Dict[str, DetectorResult] = {}
        for model in self.models:
            result = model.predict(audio)
            if result is not None:
                results[model.name] = result
            else:
                warnings.append(f"Model {model.name} failed to predict")

            if self.low_memory:
                model.unload()

        if not results:
            raise RuntimeError("All models failed. No predictions available.")

        if len(results) == 1:
            warnings.append("SINGLE MODEL - LOW CONFIDENCE: Only one model available")

        if audio.duration_seconds < 3.0:
            warnings.append(f"SHORT AUDIO: only {audio.duration_seconds:.1f}s analysed")

        score = self._fuse_scores(results)
        verdict = self._compute_verdict(score)
        dominant = max(results.items(), key=lambda x: abs(x[1].score - 0.5))[0]

        return EnsembleResult(
            verdict=verdict,
            ai_probability=score,
            threshold=self.threshold,
            model_scores=results,
            dominant_signal=dominant,
            processing_time_ms=(time.perf_counter() - start_time) * 1000,
            warnings=warnings,
            backend=self.backend,
        )

    def _fuse_scores(self, results: Dict[str, DetectorResult]) -> float:
        """Apply fusion strategy to combine model scores."""
        if self.strategy == "weighted_mean":
            return self._weighted_mean(results)
        elif self.strategy == "mean":
            return self._mean(results)
        elif self.strategy == "max":
            return self._max(results)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _weighted_mean(self, results: Dict[str, DetectorResult]) -> float:
        """Weighted mean fusion."""
        total_weight = 0.0
        weighted_sum = 0.0

        for name, result in results.items():
            weight = self.weights.get(name, 1.0)  # Default weight 1.0 if not specified
            weighted_sum += result.score * weight
            total_weight += weight

        if total_weight == 0:
            return self._mean(results)

        return weighted_sum / total_weight

    def _mean(self, results: Dict[str, DetectorResult]) -> float:
        """Simple mean fusion."""
        scores = [r.score for r in results.values()]
        return sum(scores) / len(scores)

    def _max(self, results: Dict[str, DetectorResult]) -> float:
        """Maximum score fusion (most suspicious model wins)."""
        return max(r.score for r in results.values())

    def _compute_verdict(self, score: float) -> Verdict:
        """AI / HUMAN, or UNCERTAIN within ±uncertain_margin of the threshold."""
        if abs(score - self.threshold) < self.uncertain_margin:
            return "UNCERTAIN"

        return "AI" if score > self.threshold else "HUMAN"

    def get_loaded_models(self) -> List[str]:
        """Get list of successfully loaded model names."""
        return [m.name for m in self.models if m.is_available()]

    def __repr__(self) -> str:
        loaded = self.get_loaded_models()
        return (
            f"<EnsembleDetector(models={loaded}, strategy={self.strategy}, "
            f"backend={self.backend})>"
        )
