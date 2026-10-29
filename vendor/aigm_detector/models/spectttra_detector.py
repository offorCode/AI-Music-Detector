"""SpecTTTra detectors from SONICS (ICLR 2025).

Paper: https://arxiv.org/abs/2408.14080
Trained on 97K songs (49K Suno/Udio fakes). Six public checkpoints:
alpha/beta/gamma tokenizer variants (fine -> coarse), each for 5 s or 120 s clips.
Runs on PyTorch or MLX without needing the `sonics` / `timm` packages.
"""

import json
import logging
from typing import Any, Dict, Optional, Type

import numpy as np

from aigm_detector.hub import cache_dir, download_file
from aigm_detector.models.base import BaseDetector, DetectorResult
from aigm_detector.models.sonics_net import SonicsFeatureExtractor
from aigm_detector.preprocess import AudioInput, sample_windows

logger = logging.getLogger(__name__)


class SpecTTTraDetector(BaseDetector):
    """SpecTTTra spectro-temporal transformer (SONICS).

    5 s models are scored on up to 12 windows spread over the song (averaged);
    120 s models on the central 2 minutes (zero-padded when shorter), like the
    official demo.
    """

    name = "spectttra-gamma-5s"
    requires = ["waveform_16k"]
    description = "SONICS SpecTTTra-γ, 5s clips (F1 0.76, fastest)"
    repo_id = "awsaf49/sonics-spectttra-gamma-5s"
    backends = ("torch", "mlx")

    memory_gb = 0.08
    cpu_seconds = 0.10

    max_windows = 12
    batch_size = 12

    def __init__(self, backend: Optional[str] = None):
        super().__init__(backend)
        self.model: Any = None
        self.config: Dict[str, Any] = {}
        self.features: Optional[SonicsFeatureExtractor] = None

    def _load_weights(self):
        config_path = download_file(self.repo_id, "config.json")
        self.config = json.loads(config_path.read_text())
        self.features = SonicsFeatureExtractor(self.config)

        if self.backend == "mlx":
            self.model = self._load_mlx()
        else:
            self.model = self._load_torch()

    def _state_dict(self):
        import torch

        path = download_file(self.repo_id, "pytorch_model.bin")
        return torch.load(path, map_location="cpu", weights_only=True)

    def _load_torch(self):
        from aigm_detector.device import get_device
        from aigm_detector.models.sonics_net import TorchSpecTTTra

        self.device = get_device()
        model = TorchSpecTTTra(self.config)
        model.load_sonics_state_dict(self._state_dict())
        return model.to(self.device).eval()

    def _load_mlx(self):
        import mlx.core as mx

        from aigm_detector.mlx_backend.spectttra import convert_spectttra_weights, load_spectttra

        # Convert the pickle checkpoint once; later loads read safetensors directly
        cached = cache_dir() / "mlx" / f"{self.name}.safetensors"
        if not cached.exists():
            logger.info(f"Converting {self.repo_id} to MLX -> {cached}")
            state = {k: v.float().numpy() for k, v in self._state_dict().items()}
            cached.parent.mkdir(parents=True, exist_ok=True)
            mx.save_safetensors(str(cached), convert_spectttra_weights(state, self.config))
        return load_spectttra(self.config, cached)

    def _release_weights(self):
        self.model = None

    @property
    def window_seconds(self) -> float:
        return float(self.config["audio"]["max_time"])

    def _predict_impl(self, audio: AudioInput) -> DetectorResult:
        sr = self.config["audio"]["sample_rate"]
        window = int(self.config["audio"]["max_len"])
        max_windows = self.max_windows if self.window_seconds <= 30 else 1
        windows = sample_windows(audio.numpy_at(sr), window, max_windows)

        logits = []
        for start in range(0, len(windows), self.batch_size):
            specs = self.features(windows[start : start + self.batch_size])
            logits.append(self._run(specs))
        window_scores = 1 / (1 + np.exp(-np.concatenate(logits)[:, 0]))

        score = float(window_scores.mean())
        return DetectorResult(
            model_name=self.name,
            score=score,
            confidence=abs(score - 0.5) * 2,
            processing_time_ms=0.0,  # Set by base class
            metadata={
                "model_id": self.repo_id,
                "window_seconds": self.window_seconds,
                "window_scores": [round(float(s), 4) for s in window_scores],
            },
        )

    def _run(self, specs: np.ndarray) -> np.ndarray:
        """(B, n_mels, frames) -> (B, 1) logits."""
        if self.backend == "mlx":
            import mlx.core as mx

            return np.array(self.model(mx.array(specs)).astype(mx.float32))

        import torch

        with torch.inference_mode():
            return self.model(torch.from_numpy(specs).to(self.device)).float().cpu().numpy()


# Peak MLX memory (GB) and PyTorch CPU seconds for a 3-minute track (4 cores)
_RESOURCES = {
    ("alpha", "5s"): (0.14, 0.30),
    ("beta", "5s"): (0.09, 0.12),
    ("gamma", "5s"): (0.08, 0.10),
    ("alpha", "120s"): (0.16, 0.33),
    ("beta", "120s"): (0.12, 0.21),
    ("gamma", "120s"): (0.12, 0.15),
}

# F1 on the SONICS test set, from the official model card
_VARIANTS = {
    ("alpha", "5s"): "SONICS SpecTTTra-α, 5s clips (F1 0.78)",
    ("beta", "5s"): "SONICS SpecTTTra-β, 5s clips (F1 0.78)",
    ("gamma", "5s"): "SONICS SpecTTTra-γ, 5s clips (F1 0.76, fastest)",
    ("alpha", "120s"): "SONICS SpecTTTra-α, 120s context (F1 0.97, best)",
    ("beta", "120s"): "SONICS SpecTTTra-β, 120s context (F1 0.92)",
    ("gamma", "120s"): "SONICS SpecTTTra-γ, 120s context (F1 0.88)",
}


def _make_variant(variant: str, duration: str, description: str) -> Type[SpecTTTraDetector]:
    name = f"spectttra-{variant}-{duration}"
    cls_name = f"SpecTTTra{variant.capitalize()}{duration.upper()}Detector"
    return type(
        cls_name,
        (SpecTTTraDetector,),
        {
            "name": name,
            "description": description,
            "repo_id": f"awsaf49/sonics-{name}",
            "memory_gb": _RESOURCES[(variant, duration)][0],
            "cpu_seconds": _RESOURCES[(variant, duration)][1],
            "__doc__": description,
        },
    )


SPECTTTRA_VARIANTS = [_make_variant(v, d, desc) for (v, d), desc in _VARIANTS.items()]
