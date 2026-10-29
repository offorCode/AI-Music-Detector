"""Audio Spectrogram Transformer detectors (AI-Music-Detection checkpoints).

Fine-tuned from MIT/ast-finetuned-audioset on SleepyJesse/ai_music_large
(10K Suno/Udio/MusicGen songs + 10K human songs). Runs on PyTorch or MLX.
"""

import json
import logging
from typing import Any, Optional

import numpy as np
from transformers import ASTFeatureExtractor

from aigm_detector.hub import download_snapshot
from aigm_detector.models.base import BaseDetector, DetectorResult
from aigm_detector.preprocess import AudioInput, sample_windows

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000  # AST feature extractor / training sample rate
FRAME_SHIFT_S = 0.01  # Kaldi fbank hop (10 ms)


class ASTDetector(BaseDetector):
    """Audio Spectrogram Transformer for AI music detection.

    The model sees `max_length` fbank frames (10.24 s or 60 s). Longer tracks are
    scored on up to `max_windows` windows spread over the song and averaged.
    """

    name = "ast"
    requires = ["waveform_16k"]
    description = "AST fine-tuned on Suno/Udio/MusicGen, 10.24s windows"
    repo_id = "AI-Music-Detection/ai_music_detection_large_10.24s"
    backends = ("torch", "mlx")

    memory_gb = 1.2  # measured MLX peak, 6 windows batched
    cpu_seconds = 5.1

    max_length = 1024  # fbank frames per window
    max_windows = 6
    batch_size = 6

    def __init__(self, backend: Optional[str] = None):
        super().__init__(backend)
        self.model: Any = None
        self.feature_extractor: Optional[ASTFeatureExtractor] = None
        self.ai_index = 1

    @property
    def window_seconds(self) -> float:
        return self.max_length * FRAME_SHIFT_S

    def _load_weights(self):
        model_dir = download_snapshot(self.repo_id, ["config.json", "*.safetensors"])
        config = json.loads((model_dir / "config.json").read_text())
        self.ai_index = int(config.get("label2id", {}).get("ai_generated", 1))

        # Same preprocessing as training (MIT AudioSet defaults), no extra download needed
        self.feature_extractor = ASTFeatureExtractor(max_length=self.max_length)

        if self.backend == "mlx":
            from aigm_detector.mlx_backend.ast import load_ast

            self.model = load_ast(model_dir)
        else:
            from transformers import ASTForAudioClassification

            from aigm_detector.device import get_device

            self.device = get_device()
            self.model = ASTForAudioClassification.from_pretrained(model_dir)
            self.model = self.model.float().to(self.device).eval()

    def _release_weights(self):
        self.model = None

    def _predict_impl(self, audio: AudioInput) -> DetectorResult:
        # Kaldi fbank emits one frame per 10 ms after a 25 ms window
        window = int(self.window_seconds * SAMPLE_RATE) + 240
        windows = sample_windows(audio.numpy_at(SAMPLE_RATE), window, self.max_windows)

        probs = []
        for start in range(0, len(windows), self.batch_size):
            batch = list(windows[start : start + self.batch_size])
            feats = self.feature_extractor(
                batch, sampling_rate=SAMPLE_RATE, return_tensors="np"
            )["input_values"].astype(np.float32)
            probs.append(self._run(feats))
        window_scores = np.concatenate(probs)[:, self.ai_index]

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

    def _run(self, feats: np.ndarray) -> np.ndarray:
        """(B, max_length, 128) fbank -> (B, num_labels) softmax probabilities."""
        if self.backend == "mlx":
            import mlx.core as mx

            probs = mx.softmax(self.model(mx.array(feats)), axis=-1)
            return np.array(probs.astype(mx.float32))

        import torch

        with torch.inference_mode():
            logits = self.model(input_values=torch.from_numpy(feats).to(self.device)).logits
            return logits.softmax(dim=-1).float().cpu().numpy()


class AST60Detector(ASTDetector):
    """AST variant trained on 60 s inputs (max_length=6000)."""

    name = "ast-60s"
    description = "AST fine-tuned on Suno/Udio/MusicGen, 60s windows (slower)"
    repo_id = "AI-Music-Detection/ai_music_detection_large_60s"

    memory_gb = 3.0  # estimate: ~7K-token attention; not yet measured
    cpu_seconds = 30.0

    max_length = 6000
    max_windows = 3
    batch_size = 1  # ~7K tokens per window; keep attention memory bounded
