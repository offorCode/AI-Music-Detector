"""AI Music Detector: local Windows desktop inference prototype."""
from __future__ import annotations

import math
import os
import subprocess
import sys
import threading
import ctypes
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

BG, CARD, INK, MUTED = "#f3f2ec", "#fffefa", "#18231f", "#68756e"
GREEN, DEEP, LINE = "#c9f36a", "#244538", "#dce1d8"
MAX_SECONDS = 600
MIN_AUDIO_SECONDS = 0.1
MODEL_NAME = "spectttra-alpha-120s"
MODEL_CONTEXT_SECONDS = 120
UNCERTAINTY_MARGIN = 0.15


def decode_audio(path: str):
    """Use bundled FFmpeg to decode common formats to mono float32 audio."""
    try:
        import imageio_ffmpeg
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Audio decoder dependencies are missing. Run setup_windows.bat first.") from exc
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = str(Path(ffmpeg_exe).parent)
    paths = os.environ.get("PATH", "").split(os.pathsep)
    if ffmpeg_dir not in paths:
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    command = [ffmpeg_exe, "-hide_banner", "-loglevel", "error", "-nostdin",
               "-threads", "2", "-i", path, "-t", str(MAX_SECONDS), "-vn", "-sn", "-dn",
               "-map", "0:a:0", "-ac", "1", "-ar", "44100", "-f", "f32le", "pipe:1"]
    proc = subprocess.run(command, capture_output=True, check=False)
    if proc.returncode:
        detail = proc.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(detail or "FFmpeg could not decode this audio file.")
    if not proc.stdout:
        raise ValueError("No audio samples were found in this file.")
    audio = np.frombuffer(proc.stdout, dtype="<f4").copy()
    if not audio.size:
        raise ValueError("No audio samples were found in this file.")
    if len(audio) / 44100 < MIN_AUDIO_SECONDS:
        raise ValueError("Audio must be at least 0.1 seconds long to analyze.")
    return audio, 44100


def inspect_signal(audio, sample_rate: int) -> dict:
    import numpy as np
    return {"duration": len(audio) / sample_rate, "rate": sample_rate,
            "peak": float(np.max(np.abs(audio))),
            "rms": float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))),
            "wave": audio[::max(1, len(audio) // 1200)].copy()}


class AIMusicDetectorApp:
    def __init__(self, root: tk.Tk):
        self.root, self.path, self.signal, self.detector, self.busy = root, None, None, None, False
        self._intro_alias = "onntech_intro"
        self._intro_open = False
        self._splash_image = None
        self._set_icon()
        self._show_splash()
        self._build()

    def _asset_path(self, filename: str) -> Path:
        return Path(__file__).resolve().parent / filename

    def _set_icon(self):
        icon = self._asset_path("OnnTech.ico")
        if icon.is_file():
            try:
                self.root.iconbitmap(default=str(icon))
            except tk.TclError:
                pass

    def _show_splash(self):
        self.root.withdraw()
        self.splash = tk.Toplevel(self.root)
        self.splash.title("AI-Music Detector")
        self.splash.configure(bg="#fffefa")
        self.splash.resizable(False, False)
        self.splash.overrideredirect(True)
        self.splash.attributes("-topmost", True)
        width, height = 600, 440
        x = (self.root.winfo_screenwidth() - width) // 2
        y = (self.root.winfo_screenheight() - height) // 2
        self.splash.geometry(f"{width}x{height}+{x}+{y}")
        try:
            self._splash_image = tk.PhotoImage(file=str(self._asset_path("OnnTech.png")))
            tk.Label(self.splash, image=self._splash_image, bg="#fffefa").pack(pady=(28, 12))
        except (tk.TclError, OSError):
            tk.Label(self.splash, text="♫", font=("Segoe UI", 64, "bold"), bg="#fffefa", fg="#f06422").pack(pady=(24, 4))
        tk.Label(self.splash, text="AI-Music Detector", font=("Segoe UI", 24, "bold"),
                 bg="#fffefa", fg="#f06422").pack(pady=(0, 6))
        tk.Label(self.splash, text="Open-source project • Built independently",
                 font=("Segoe UI", 11, "bold"), bg="#fffefa", fg="#f06422").pack()
        tk.Label(self.splash, text="Enjoying the app? Support future development.",
                 font=("Segoe UI", 10), bg="#fffefa", fg="#111111").pack(pady=(8, 0))
        self._play_intro()
        self.root.after(4000, self._close_splash)

    def _play_intro(self):
        track = self._asset_path("OnnTech.mp3")
        if not track.is_file() or sys.platform != "win32":
            return
        try:
            winmm = ctypes.windll.winmm
            command = f'open "{track}" type mpegvideo alias {self._intro_alias}'
            if winmm.mciSendStringW(command, None, 0, None) == 0:
                self._intro_open = True
                winmm.mciSendStringW(f"play {self._intro_alias}", None, 0, None)
        except (AttributeError, OSError):
            self._intro_open = False

    def _stop_intro(self):
        if self._intro_open:
            try:
                winmm = ctypes.windll.winmm
                winmm.mciSendStringW(f"stop {self._intro_alias}", None, 0, None)
                winmm.mciSendStringW(f"close {self._intro_alias}", None, 0, None)
            except (AttributeError, OSError):
                pass
            self._intro_open = False

    def _close_splash(self):
        self._stop_intro()
        if self.splash.winfo_exists():
            self.splash.destroy()
        self.root.deiconify()
        self.root.lift()

    def _build(self):
        self.root.title("AI-Music Detector")
        self.root.geometry("1080x760")
        self.root.minsize(850, 650)
        self.root.configure(bg=BG)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TProgressbar", troughcolor="#e3e5de", background="#85ac65", bordercolor="#e3e5de")
        page = tk.Frame(self.root, bg=BG, padx=32, pady=20)
        page.pack(fill="both", expand=True)
        header = tk.Frame(page, bg=BG)
        header.pack(fill="x", pady=(0, 20))
        tk.Label(header, text="♫", font=("Segoe UI", 16, "bold"), bg=GREEN, fg=DEEP, width=3).pack(side="left", padx=(0, 10))
        tk.Label(header, text="AI-Music Detector", font=("Segoe UI", 16, "bold"), bg=BG, fg=INK).pack(side="left")
        tk.Label(header, text="LOCAL AUDIO ANALYSIS", font=("Segoe UI", 9, "bold"), bg=BG, fg=MUTED).pack(side="right", pady=8)
        tk.Label(page, text="Does it sound made by AI?", font=("Segoe UI", 27, "bold"), bg=BG, fg=INK).pack(anchor="w")
        tk.Label(page, text="Analyze a track with a pretrained music-forensics model.", font=("Segoe UI", 11), bg=BG, fg=MUTED).pack(anchor="w", pady=(4, 18))

        body = tk.Frame(page, bg=BG)
        body.pack(fill="both", expand=True)
        left = tk.Frame(body, bg=CARD, highlightbackground=LINE, highlightthickness=1, padx=19, pady=17)
        left.pack(side="left", fill="both", expand=True, padx=(0, 12))
        right = tk.Frame(body, bg=CARD, highlightbackground=LINE, highlightthickness=1, padx=19, pady=17, width=345)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        tk.Label(left, text="AUDIO FILE", font=("Segoe UI", 9, "bold"), bg=CARD, fg=MUTED).pack(anchor="w")
        tk.Label(left, text="Select a song to analyze", font=("Segoe UI", 16, "bold"), bg=CARD, fg=INK).pack(anchor="w", pady=(5, 12))
        self.file_label = tk.Label(left, text="No audio selected", font=("Segoe UI", 10), bg="#f7f7f2", fg=MUTED, anchor="w", padx=12, pady=13)
        self.file_label.pack(fill="x")
        controls = tk.Frame(left, bg=CARD)
        controls.pack(fill="x", pady=(11, 17))
        self._button(controls, "Choose audio", self.choose_file, True).pack(side="left")
        self._button(controls, "Open in player", self.open_player, False).pack(side="left", padx=8)
        self.analyze_button = self._button(controls, "Detect AI music", self.analyze, True)
        self.analyze_button.pack(side="right")
        self.analyze_button.configure(state="disabled")
        tk.Label(left, text="WAVEFORM", font=("Segoe UI", 9, "bold"), bg=CARD, fg=MUTED).pack(anchor="w", pady=(0, 7))
        self.canvas = tk.Canvas(left, height=175, bg="#f7f7f2", highlightthickness=0)
        self.canvas.pack(fill="x")
        self.canvas.bind("<Configure>", lambda _e: self.draw_waveform())
        self.status = tk.StringVar(value="Choose a file to begin.")
        tk.Label(left, textvariable=self.status, font=("Segoe UI", 9), bg=CARD, fg=MUTED, anchor="w").pack(fill="x", pady=(10, 4))
        self.progress = ttk.Progressbar(left, mode="indeterminate")

        tk.Label(right, text="DETECTION RESULT", font=("Segoe UI", 9, "bold"), bg=CARD, fg=MUTED).pack(anchor="w")
        self.verdict = tk.Label(right, text="Waiting for audio", font=("Segoe UI", 21, "bold"), bg=CARD, fg=INK, wraplength=295, justify="left")
        self.verdict.pack(anchor="w", pady=(10, 4))
        self.probability = tk.Label(right, text="", font=("Segoe UI", 11, "bold"), bg=CARD, fg=DEEP)
        self.probability.pack(anchor="w", pady=(0, 7))
        self.verdict_note = tk.Label(right, text="The detector scores multiple parts of the track.", font=("Segoe UI", 9), bg=CARD, fg=MUTED, wraplength=295, justify="left")
        self.verdict_note.pack(anchor="w", pady=(0, 11))
        self.meter = ttk.Progressbar(right, mode="determinate", maximum=100)
        self.meter.pack(fill="x", pady=(0, 12))
        tk.Frame(right, bg=LINE, height=1).pack(fill="x", pady=(0, 13))
        tk.Label(right, text="TRACK PROPERTIES", font=("Segoe UI", 9, "bold"), bg=CARD, fg=MUTED).pack(anchor="w", pady=(0, 6))
        self.metric_vars = {}
        for key, label in [("duration", "Duration"), ("format", "Input format"), ("rate", "Analysis rate"), ("peak", "Peak level"), ("rms", "Average level")]:
            row = tk.Frame(right, bg=CARD)
            row.pack(fill="x", pady=4)
            tk.Label(row, text=label, font=("Segoe UI", 9), bg=CARD, fg=MUTED).pack(side="left")
            self.metric_vars[key] = tk.StringVar(value="—")
            tk.Label(row, textvariable=self.metric_vars[key], font=("Segoe UI", 9, "bold"), bg=CARD, fg=INK).pack(side="right")
        tk.Frame(right, bg=LINE, height=1).pack(fill="x", pady=(12, 10))
        tk.Label(right, text="MODEL SCORE", font=("Segoe UI", 9, "bold"), bg=CARD, fg=MUTED).pack(anchor="w", pady=(0, 5))
        self.model_text = tk.Label(right, text="The model score will appear here.", font=("Consolas", 8), bg=CARD, fg=MUTED, justify="left", anchor="w")
        self.model_text.pack(fill="x")

        note = tk.Frame(page, bg="#eaf0e7", padx=13, pady=10)
        note.pack(fill="x", pady=(13, 0))
        tk.Label(note, text="Research estimate, not proof. New AI generators, remastering, short clips, and human–AI mixes can fool detectors. Results are probabilistic and should not be used to accuse or make legal claims.", font=("Segoe UI", 9), bg="#eaf0e7", fg="#42564a", wraplength=950, justify="left").pack(anchor="w")

    @staticmethod
    def _button(parent, text, command, primary):
        return tk.Button(parent, text=text, command=command, bg=DEEP if primary else "#eef0e9", fg="white" if primary else INK,
                         activebackground="#315b48" if primary else "#e1e5da", activeforeground="white" if primary else INK,
                         relief="flat", bd=0, padx=13, pady=9, cursor="hand2", font=("Segoe UI", 9, "bold"))

    def choose_file(self):
        path = filedialog.askopenfilename(title="Choose a music file", filetypes=[
            ("Common audio files", "*.wav *.mp3 *.flac *.ogg *.m4a *.aac *.wma *.aiff *.aif *.aifc *.opus *.mp4 *.webm *.mkv *.mka *.caf *.amr *.ape *.mpc *.wv *.ac3 *.dts *.au"),
            ("All files", "*.*")])
        if not path:
            return
        self.path = path
        self.signal = None
        self.file_label.configure(text=Path(path).name, fg=INK)
        self.analyze_button.configure(state="normal")
        self.status.set("Ready. Audio is processed locally.")
        self.verdict.configure(text="Ready to analyze", fg=INK)
        self.probability.configure(text="")
        self.verdict_note.configure(text="The first run downloads detector checkpoints. Later runs can work offline.")
        self.meter["value"] = 0
        self.model_text.configure(text="Segment scores will appear here.")
        for var in self.metric_vars.values():
            var.set("—")
        self.draw_waveform()

    def open_player(self):
        if self.path:
            try:
                os.startfile(self.path)
            except OSError as exc:
                messagebox.showerror("Could not open audio", str(exc))

    def analyze(self):
        if not self.path or self.busy:
            return
        self.busy = True
        self.analyze_button.configure(state="disabled")
        self.progress.pack(fill="x", pady=(0, 2))
        self.progress.start(12)
        self.status.set("Decoding audio…")
        self.verdict.configure(text="Analyzing track…", fg=INK)
        self.probability.configure(text="")
        self.verdict_note.configure(text="Loading the detector and scoring audio windows.")
        threading.Thread(target=self._worker, args=(self.path,), daemon=True).start()

    def _worker(self, path):
        try:
            self._ui(self.status.set, "Decoding audio to mono 44.1 kHz…")
            wave_data, rate = decode_audio(path)
            signal = inspect_signal(wave_data, rate)
            self._ui(self.status.set, "Loading the licensed music model…")
            model_cache = os.path.join(os.environ.get("LOCALAPPDATA", str(Path.home())), "AI-Music Detector", "models")
            os.environ.setdefault("HF_HOME", model_cache)
            vendor_path = str(Path(__file__).resolve().parent / "vendor")
            if vendor_path not in sys.path:
                sys.path.insert(0, vendor_path)
            if self.detector is None:
                torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
                if torch_lib.exists():
                    self._torch_dll_handle = os.add_dll_directory(str(torch_lib))
                from aigm_detector import EnsembleDetector
                self.detector = EnsembleDetector(model_names=MODEL_NAME, backend="torch",
                                                 low_memory=True, uncertain_margin=UNCERTAINTY_MARGIN)
            self._ui(self.status.set, "Scoring the full track (first run downloads about 75 MB)…")
            from aigm_detector import AudioInput
            audio = AudioInput.from_array(wave_data, rate)
            result = self.detector.predict(audio)
            duration = len(wave_data) / rate
            model_scores = {name: score.score for name, score in result.model_scores.items()}
            data = {"result": result, "score": result.ai_probability, "signal": signal,
                    "model_scores": model_scores, "duration": duration,
                    "truncated": duration >= MAX_SECONDS}
            self.root.after(0, self._show_result, data)
        except Exception as exc:
            self.root.after(0, self._show_error, str(exc))

    def _ui(self, fn, *args):
        self.root.after(0, fn, *args)

    def _show_result(self, data):
        result, score, signal = data["result"], data["score"], data["signal"]
        if result.verdict == "AI":
            text, color = "Likely AI-generated", "#9c4b32"
        elif result.verdict == "HUMAN":
            text, color = "Likely human-made", DEEP
        else:
            text, color = "Uncertain result", "#8a6736"
        self.verdict.configure(text=text, fg=color)
        self.probability.configure(text=f"Model AI score: {score:.0%}")
        note = "Model score, not a verified probability of authorship."
        if result.verdict == "UNCERTAIN":
            direction = "leans toward AI" if score >= 0.5 else "leans toward human-made"
            low = round((0.5 - UNCERTAINTY_MARGIN) * 100)
            high = round((0.5 + UNCERTAINTY_MARGIN) * 100)
            note += f"\nThis score {direction}, but {low}–{high}% is the app's uncertain range, so it is not a firm classification."
        if result.warnings:
            note += "\n" + "\n".join(result.warnings)
        if data["duration"] < 3:
            note += "\nVery short clip: results under three seconds may be unreliable. This model zero-pads the clip to its analysis window."
        elif data["duration"] < MODEL_CONTEXT_SECONDS:
            note += (f"\nThis model is designed for {MODEL_CONTEXT_SECONDS}-second audio windows. "
                     "Shorter tracks are padded for analysis, which can make the result less reliable.")
        if data["truncated"]:
            note += f"\nOnly the first {MAX_SECONDS // 60} minutes were analyzed."
        self.verdict_note.configure(text=note)
        self.meter["value"] = score * 100
        self.model_text.configure(text="\n".join(f"{name}: {value:.1%}" for name, value in data["model_scores"].items()))
        self.metric_vars["duration"].set(self._time(data["duration"]))
        self.metric_vars["format"].set(Path(self.path).suffix.upper().lstrip("."))
        self.metric_vars["rate"].set("44.1 kHz mono")
        self.metric_vars["peak"].set(self._db(signal["peak"]))
        self.metric_vars["rms"].set(self._db(signal["rms"]))
        self.signal = signal
        self.status.set(f"Analysis complete in {result.processing_time_ms / 1000:.1f} seconds.")
        self.draw_waveform()
        self._done()

    def _show_error(self, message):
        self.verdict.configure(text="Could not analyze audio", fg="#9c4b32")
        self.verdict_note.configure(text=message[:700])
        self.status.set("Analysis failed.")
        self._done()

    def _done(self):
        self.progress.stop()
        self.progress.pack_forget()
        self.analyze_button.configure(state="normal" if self.path else "disabled")
        self.busy = False

    def draw_waveform(self):
        c = self.canvas
        c.delete("all")
        width, height = c.winfo_width(), c.winfo_height()
        if width < 10 or height < 10:
            return
        mid = height // 2
        c.create_line(0, mid, width, mid, fill="#d9dfd5")
        if self.signal is None:
            c.create_text(width // 2, mid, text="Waveform appears here after analysis", fill="#9aa39c", font=("Segoe UI", 9))
            return
        samples = self.signal["wave"]
        step = max(1, len(samples) // width)
        coords = []
        for x in range(min(width, len(samples))):
            chunk = samples[x * step:(x + 1) * step]
            if len(chunk):
                amp = min(1.0, float(max(abs(chunk.min()), abs(chunk.max()))))
                coords.extend((x, mid - amp * height * .44, x, mid + amp * height * .44))
        if coords:
            c.create_line(*coords, fill="#608455", width=1.2)

    @staticmethod
    def _time(seconds):
        total = int(seconds)
        return f"{total // 60}:{total % 60:02d}"

    @staticmethod
    def _db(value):
        return "−∞ dBFS" if value <= 0 else f"{20 * math.log10(value):.1f} dBFS"


def main():
    app_dir = Path(__file__).resolve().parent
    venv_python = app_dir / ".venv" / "Scripts" / "python.exe"
    if os.name == "nt" and venv_python.is_file():
        try:
            in_project_venv = Path(sys.prefix).resolve() == (app_dir / ".venv").resolve()
        except OSError:
            in_project_venv = False
        if not in_project_venv:
            subprocess.Popen([str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]], cwd=app_dir)
            return
    root = tk.Tk()
    app = AIMusicDetectorApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app._stop_intro(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
