"""REST API (FastAPI). Run with `aigm-detector serve`.

Configure with environment variables:
    AIGM_MODELS   comma-separated model names (default: DEFAULT_MODELS)
    AIGM_BACKEND  auto | torch | mlx
"""

import os
import tempfile
from pathlib import Path
from threading import Lock
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile

from aigm_detector import __version__
from aigm_detector.ensemble import EnsembleDetector
from aigm_detector.models import get_available_models
from aigm_detector.preprocess import SUPPORTED_FORMATS, AudioInput

app = FastAPI(title="aigm-detector", version=__version__)

_detector: Optional[EnsembleDetector] = None
_lock = Lock()  # Models are not thread-safe; serialize inference


def get_detector() -> EnsembleDetector:
    global _detector
    if _detector is None:
        _detector = EnsembleDetector(model_names=os.environ.get("AIGM_MODELS"))
    return _detector


@app.get("/health")
def health():
    detector = get_detector()
    return {
        "status": "ok",
        "backend": detector.backend,
        "models": [m.name for m in detector.models],
        "loaded": detector.get_loaded_models(),
    }


@app.get("/models")
def models():
    return {
        name: {
            "description": cls.description,
            "backends": list(cls.backends),
            "checkpoint": cls.repo_id,
        }
        for name, cls in get_available_models().items()
    }


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise HTTPException(400, f"Unsupported format '{suffix}'. Use one of {sorted(SUPPORTED_FORMATS)}")

    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(file.file.read())
        tmp.flush()
        try:
            audio = AudioInput.from_file(tmp.name)
        except ValueError as e:
            raise HTTPException(400, str(e))

        with _lock:
            try:
                result = get_detector().predict(audio)
            except RuntimeError as e:
                raise HTTPException(503, str(e))

    return {"file": file.filename, **result.to_dict()}
