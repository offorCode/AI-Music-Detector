"""AIGM Detector - AI-Generated Music Detection Ensemble."""

__version__ = "0.1.0"
__author__ = "AI Music Detector Contributors"

from aigm_detector.ensemble import EnsembleDetector, EnsembleResult
from aigm_detector.models.base import BaseDetector, DetectorResult
from aigm_detector.preprocess import AudioInput
from aigm_detector.youtube import YouTubeDownloader

__all__ = [
    "AudioInput",
    "EnsembleDetector",
    "EnsembleResult",
    "BaseDetector",
    "DetectorResult",
    "YouTubeDownloader",
]
