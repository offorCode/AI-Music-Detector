# AI-Music Detector

Windows desktop prototype for estimating whether a music track is AI-generated. It uses the SONICS SpecTTTra alpha-120s pretrained classifier through the AIGM Detector package. The app runs inference locally; audio is not uploaded.

## Setup and run

1. Install Python 3.11 (64-bit) and enable **Add Python to PATH**.
2. Double-click `setup_windows.bat`. This creates an isolated `.venv`, installs CPU-only PyTorch, the detector, and imageio-ffmpeg's bundled FFmpeg binary. It needs internet, several minutes, and around 1 GB of disk space.
3. Double-click `run_detector.bat`, or select the `.venv` Python interpreter in VS Code and run `earprint_desktop.py`.
4. Choose a track and click **Detect AI music**. The first analysis downloads the model (about 75 MB) into the current user's local app cache; later runs can work offline.

Supported common formats: WAV, MP3, FLAC, OGG, M4A, AAC, WMA, AIFF, OPUS, MP4, and WEBM, as supported by FFmpeg. Audio is decoded locally to mono 44.1 kHz. Audio from 0.1 seconds to ten minutes is accepted; clips under three seconds are especially unreliable, and this model scores a two-minute section. The splash screen uses OnnTech.png, the Windows icon uses a transparent OnnTech.ico, and OnnTech.mp3 plays briefly at startup. Audio is not uploaded.

The setup script checks the existing `.venv` first and skips installation when its dependencies are already present. Running `earprint_desktop.py` from outside the project virtual environment will relaunch it in `.venv` when available.

## Sharing the install bundle

Run `package_windows.ps1` to create `dist/AI-Music-Detector-source.zip`. Share that ZIP; recipients need 64-bit Python 3.11 and an internet connection for setup and the first model download. This is a setup bundle, not a signed single-file Windows installer.

## Reading the result

The model score is compared with a threshold that includes an uncertainty band. The app reports **Likely AI-generated**, **Likely human-made**, or **Uncertain result**. This is a model estimate, not proof of who made the music. The released checkpoint's reported F1 is 0.97 on its authors' SONICS test set; performance can fall on other generators, edited or compressed audio, excerpts, and human–AI mixes. Do not use a result as proof or to accuse an artist.

## Support the project

AI-Music Detector is free to use. If it is useful to you, you can support its continued development:

- [PayPal](https://www.paypal.com/ncp/payment/75JCGB569TXGN)
- [Ko-fi](https://ko-fi.com/chechris)
- **USDT:** `TEHviarX7EooeCYBuut4SLX8eUDxzsPvLp` — send USDT on the TRON network (TRC20) only.
- **Bitcoin:** `184ZYveWrPm24kutxAP1tUDKB1NVciLbt4` — send BTC on the Bitcoin network only.

For help, contact us on [WhatsApp](https://wa.me/233530842130) Thank you for your support!

## Model source and license

The selected checkpoint is listed as MIT licensed by the [SONICS model card](https://huggingface.co/awsaf49/sonics-spectttra-alpha-120s); the official project lists code and models as MIT while listing its training dataset separately as CC BY-NC 4.0. The app also includes MIT-licensed AIGM Detector source. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Before selling or distributing a proprietary installer, resolve the model and training-data provenance with a licensing professional. The FFmpeg executable installed by `imageio-ffmpeg` reports a GPLv3+ build configuration; its redistribution has obligations. Use a commercially cleared LGPL build or complete the required GPL compliance before shipping it with a paid product. This prototype does not provide legal clearance.
