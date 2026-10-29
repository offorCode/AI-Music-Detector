"""Checkpoint download and converted-weight cache."""

import os
from pathlib import Path
from typing import List

from huggingface_hub import hf_hub_download, snapshot_download


def cache_dir() -> Path:
    """Where converted (e.g. MLX) weights are stored. Override with AIGM_CACHE_DIR."""
    root = os.environ.get("AIGM_CACHE_DIR") or Path.home() / ".cache" / "aigm_detector"
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_snapshot(repo_id: str, patterns: List[str]) -> Path:
    """Download only the files matching `patterns` and return the local directory."""
    return Path(snapshot_download(repo_id=repo_id, allow_patterns=patterns))


def download_file(repo_id: str, filename: str) -> Path:
    return Path(hf_hub_download(repo_id=repo_id, filename=filename))
