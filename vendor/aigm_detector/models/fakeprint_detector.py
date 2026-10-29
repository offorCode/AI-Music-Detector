"""Spectral "fakeprint" detector (lofcz/ai-music-detector).

Implements the method of Afchar et al., "A Fourier Explanation of AI-music
Artifacts" (ISMIR 2025, https://arxiv.org/abs/2506.19108): neural vocoders leave
regularly spaced peaks in the average spectrum. The fakeprint is the mean dB
spectrum in 1-8 kHz minus its lower envelope, and a logistic regression on that
3585-value vector gives p(AI). Trained on SONICS fakes vs FMA real songs, so it
targets Suno <= 5 and Udio <= 1.5.

Preprocessing mirrors the reference implementation
(github.com/lofcz/ai-music-detector, src/python/inference.py). It is a few FFTs
and a dot product, so it runs on the CPU for every backend.
"""

import json
import logging
from typing import Any, Optional

import numpy as np
import torch
from scipy.ndimage import minimum_filter1d

from aigm_detector.hub import download_snapshot
from aigm_detector.models.base import BaseDetector, DetectorResult
from aigm_detector.preprocess import AudioInput

logger = logging.getLogger(__name__)


class FakeprintDetector(BaseDetector):
    """Logistic regression on the spectral fakeprint (15 KB of weights)."""

    name = "fakeprint"
    requires = ["waveform_16k"]
    description = "Spectral fakeprint + logistic regression (Suno<=5, Udio<=1.5), tiny and fast"
    repo_id = "lofcz/ai-music-detector"
    backends = ("torch",)

    memory_gb = 0.05  # one STFT of the track; weights are 15 KB
    cpu_seconds = 0.3

    sample_rate = 16000
    n_fft = 8192
    max_duration_s = 300  # the reference implementation only reads the first 5 minutes

    def __init__(self, backend: Optional[str] = None):
        super().__init__(backend)
        self.weights: Any = None
        self.bias: float = 0.0
        self.cfg: dict = {}
        self.freq_mask: Optional[np.ndarray] = None

    def _load_weights(self):
        from safetensors.numpy import load_file

        model_dir = download_snapshot(self.repo_id, ["config.json", "model.safetensors"])
        config = json.loads((model_dir / "config.json").read_text())
        self.cfg = config["preprocessing"]
        tensors = load_file(str(model_dir / "model.safetensors"))
        self.weights = tensors["weights"].reshape(-1).astype(np.float64)
        self.bias = float(tensors["bias"].reshape(-1)[0])

        freqs = np.linspace(0, self.cfg["sample_rate"] / 2, num=self.cfg["n_fft"] // 2 + 1)
        self.freq_mask = (freqs >= self.cfg["freq_min"]) & (freqs <= self.cfg["freq_max"])
        if int(self.freq_mask.sum()) != len(self.weights):
            raise ValueError(
                f"fakeprint has {int(self.freq_mask.sum())} bins but the model expects "
                f"{len(self.weights)}"
            )

    def _release_weights(self):
        self.weights = None
        self.freq_mask = None

    def fakeprint(self, wave: np.ndarray) -> np.ndarray:
        """Mean dB spectrum in the artifact band minus its lower hull, normalised to [0, 1]."""
        x = torch.from_numpy(np.ascontiguousarray(wave[: self.max_duration_s * self.sample_rate]))
        # Same as torchaudio.transforms.Spectrogram(n_fft, power=2): hann, hop = n_fft // 2
        window = torch.hann_window(self.n_fft)
        spec = torch.stft(
            x, self.n_fft, hop_length=self.n_fft // 2, window=window, center=True,
            pad_mode="reflect", return_complex=True,
        ).abs().pow(2)
        spec_db = 10 * torch.log10(spec.clamp(min=1e-10, max=1e6))
        mean_db = spec_db.mean(dim=1).numpy()[self.freq_mask]

        hull = minimum_filter1d(mean_db, size=self.cfg["hull_area"], mode="nearest")
        hull = np.clip(hull, self.cfg["min_db"], None)
        residue = np.clip(mean_db - hull, 0, self.cfg["max_db"])
        return (residue / (residue.max() + 1e-6)).astype(np.float32)

    def _predict_impl(self, audio: AudioInput) -> DetectorResult:
        fp = self.fakeprint(audio.numpy_at(self.sample_rate))
        logit = float(fp @ self.weights + self.bias)
        score = float(1.0 / (1.0 + np.exp(-logit)))
        return DetectorResult(
            model_name=self.name,
            score=score,
            confidence=abs(score - 0.5) * 2,
            processing_time_ms=0.0,  # Set by base class
            metadata={"model_id": self.repo_id, "logit": round(logit, 4)},
        )
