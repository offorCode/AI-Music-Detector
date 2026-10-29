"""YouTube audio download for detection."""

import logging
import tempfile
from pathlib import Path
from typing import Optional

import yt_dlp

logger = logging.getLogger(__name__)


class YouTubeDownloader:
    """Download audio from YouTube URLs."""

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize downloader.

        Args:
            output_dir: Directory to save downloaded files. If None, uses temp directory.
        """
        self.output_dir = output_dir or Path(tempfile.gettempdir()) / "aigm_detector_yt"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download(self, url: str, format: str = "mp3") -> Path:
        """Download audio from YouTube URL.

        Args:
            url: YouTube URL
            format: Output format (mp3, wav, flac)

        Returns:
            Path to downloaded audio file

        Raises:
            ValueError: Invalid URL or download failed
        """
        try:
            output_template = str(self.output_dir / "%(id)s.%(ext)s")

            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": output_template,
                "quiet": True,
                "no_warnings": True,
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": format,
                        "preferredquality": "192",
                    }
                ],
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                logger.info(f"Downloading audio from: {url}")
                info = ydl.extract_info(url, download=True)
                video_id = info["id"]

                # Construct output path
                output_path = self.output_dir / f"{video_id}.{format}"

                if not output_path.exists():
                    raise ValueError(f"Download failed: {output_path} not found")

                logger.info(f"Downloaded: {output_path}")
                return output_path

        except Exception as e:
            raise ValueError(f"Failed to download from YouTube: {e}")

    def cleanup(self):
        """Remove downloaded files."""
        if self.output_dir.exists():
            import shutil

            shutil.rmtree(self.output_dir)
            logger.info(f"Cleaned up: {self.output_dir}")
