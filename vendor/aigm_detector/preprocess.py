"""Unified audio loading and lazy preprocessing for all detection models."""

import io
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Union

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from pydub import AudioSegment

logger = logging.getLogger(__name__)

# Suppress librosa warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

SUPPORTED_FORMATS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}

MIN_DURATION_S = 3.0
MAX_DURATION_S = 600.0


@dataclass(frozen=True)
class AudioInput:
    """Immutable mono audio with lazily computed, cached representations.

    Only the decoded waveform is stored eagerly. Resampled versions and
    spectrograms are computed on first access, so models only pay for the
    representations they actually use (e.g. AST and SpecTTTra need 16 kHz only).
    """

    waveform: torch.Tensor  # (1, T) float32 mono at `sample_rate`
    sample_rate: int
    duration_seconds: float
    sample_rate_original: int
    _cache: Dict[str, torch.Tensor] = field(
        default_factory=dict, repr=False, compare=False, hash=False
    )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
        apply_codec_aug: bool = False,
    ) -> "AudioInput":
        """Load and preprocess audio file.

        Args:
            path: Path to audio file
            apply_codec_aug: Apply MP3 codec augmentation (re-encode at 128kbps)

        Returns:
            AudioInput

        Raises:
            ValueError: Unsupported format or invalid audio
            FileNotFoundError: File does not exist
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        if path.suffix.lower() not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {path.suffix}. "
                f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
            )

        waveform, sr = cls._load_audio(path)

        if apply_codec_aug:
            waveform = cls._apply_codec_augmentation(waveform, sr)

        return cls.from_array(waveform, sr)

    @classmethod
    def from_array(
        cls,
        waveform: Union[np.ndarray, torch.Tensor],
        sample_rate: int,
    ) -> "AudioInput":
        """Build from an in-memory waveform shaped (T,) or (channels, T)."""
        waveform = torch.as_tensor(np.asarray(waveform), dtype=torch.float32)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        duration = waveform.shape[1] / sample_rate

        if duration < MIN_DURATION_S:
            logger.warning(
                f"Audio is only {duration:.1f}s (< {MIN_DURATION_S:.0f}s). "
                "Results may be unreliable."
            )

        if duration > MAX_DURATION_S:
            logger.warning(
                f"Audio is {duration / 60:.1f} minutes long. "
                f"Processing first {MAX_DURATION_S / 60:.0f} minutes only."
            )
            waveform = waveform[:, : int(MAX_DURATION_S * sample_rate)]
            duration = MAX_DURATION_S

        return cls(
            waveform=waveform.contiguous(),
            sample_rate=sample_rate,
            duration_seconds=duration,
            sample_rate_original=sample_rate,
        )

    # ------------------------------------------------------------------
    # Lazy representations
    # ------------------------------------------------------------------

    def at(self, sample_rate: int) -> torch.Tensor:
        """Waveform resampled to `sample_rate`, shape (1, T). Cached."""
        key = f"wave_{sample_rate}"
        if key not in self._cache:
            self._cache[key] = self._resample(self.waveform, self.sample_rate, sample_rate)
        return self._cache[key]

    def numpy_at(self, sample_rate: int) -> np.ndarray:
        """Waveform resampled to `sample_rate` as a 1-D float32 numpy array."""
        return self.at(sample_rate)[0].numpy()

    @property
    def waveform_44k(self) -> torch.Tensor:
        return self.at(44100)

    @property
    def waveform_24k(self) -> torch.Tensor:
        return self.at(24000)

    @property
    def waveform_22k(self) -> torch.Tensor:
        return self.at(22050)

    @property
    def waveform_16k(self) -> torch.Tensor:
        return self.at(16000)

    @property
    def mel_22k(self) -> torch.Tensor:
        """Log mel-spectrogram at 22.05 kHz, shape (128, frames). Cached."""
        if "mel_22k" not in self._cache:
            mel = self._compute_mel_spectrogram(self.numpy_at(22050), 22050)
            self._cache["mel_22k"] = torch.from_numpy(mel).float()
        return self._cache["mel_22k"]

    @property
    def stft_magnitude(self) -> torch.Tensor:
        """STFT magnitude at 44.1 kHz, shape (1025, frames). Cached."""
        if "stft" not in self._cache:
            stft = self._compute_stft(self.numpy_at(44100))
            self._cache["stft"] = torch.from_numpy(stft).float()
        return self._cache["stft"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_audio(path: Path) -> tuple[torch.Tensor, int]:
        """Decode audio to a (channels, T) float32 tensor.

        Uses libsndfile first (wav/flac/ogg/mp3), then falls back to librosa's
        audioread/ffmpeg path (m4a and exotic mp3s). torchaudio.load is avoided
        because recent torchaudio releases require the separate torchcodec package.
        """
        try:
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
            return torch.from_numpy(data.T.copy()), sr
        except Exception as sf_error:
            logger.debug(f"soundfile failed on {path.name}: {sf_error}; trying librosa")

        try:
            data, sr = librosa.load(str(path), sr=None, mono=False)
            data = np.atleast_2d(data).astype(np.float32)
            return torch.from_numpy(data), int(sr)
        except Exception as e:
            raise ValueError(f"Failed to load audio file: {e}")

    @staticmethod
    def _resample(
        waveform: torch.Tensor,
        orig_sr: int,
        target_sr: int,
    ) -> torch.Tensor:
        """Resample waveform to target sample rate."""
        if orig_sr == target_sr:
            return waveform
        return AF.resample(waveform, orig_freq=orig_sr, new_freq=target_sr)

    @staticmethod
    def _compute_mel_spectrogram(
        waveform: np.ndarray,
        sr: int,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
    ) -> np.ndarray:
        """Compute log mel-spectrogram using librosa."""
        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        return librosa.power_to_db(mel, ref=np.max)

    @staticmethod
    def _compute_stft(
        waveform: np.ndarray,
        n_fft: int = 2048,
        hop_length: int = 512,
    ) -> np.ndarray:
        """Compute STFT magnitude using librosa."""
        return np.abs(librosa.stft(y=waveform, n_fft=n_fft, hop_length=hop_length))

    @staticmethod
    def _apply_codec_augmentation(waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """Apply MP3 codec augmentation by re-encoding at 128kbps.

        Per ArtifactNet paper, this reduces codec probability drift by ~83%.
        Requires ffmpeg.
        """
        try:
            # (channels, T) -> interleaved int16
            waveform_np = (waveform.numpy().T.clip(-1, 1) * 32767).astype(np.int16)

            audio = AudioSegment(
                waveform_np.tobytes(),
                frame_rate=sr,
                sample_width=2,
                channels=waveform.shape[0],
            )

            buffer = io.BytesIO()
            audio.export(buffer, format="mp3", bitrate="128k")
            buffer.seek(0)
            augmented = AudioSegment.from_mp3(buffer)

            samples = np.array(augmented.get_array_of_samples())
            samples = samples.reshape((-1, augmented.channels)).T
            return torch.from_numpy(samples.astype(np.float32) / 32767.0)

        except Exception as e:
            logger.warning(f"Codec augmentation failed: {e}. Using original audio.")
            return waveform


def sample_windows(audio: np.ndarray, window: int, max_windows: int) -> np.ndarray:
    """Pick up to `max_windows` fixed-length windows spread evenly over a track.

    - Shorter than one window: zero-padded at the end (single window).
    - One window requested: the centre of the track.
    - Otherwise: evenly spaced starts from the beginning to the end.

    Returns:
        Array shaped (num_windows, window)
    """
    n = audio.shape[0]
    if n <= window:
        return np.pad(audio, (0, window - n))[None, :]

    count = max(1, min(max_windows, int(np.ceil(n / window))))
    if count == 1:
        starts = [(n - window) // 2]
    else:
        starts = np.linspace(0, n - window, count).round().astype(int)

    return np.stack([audio[s : s + window] for s in starts])
