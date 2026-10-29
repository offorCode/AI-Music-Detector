"""Command-line interface for aigm-detector."""

import csv
import logging
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import typer
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from aigm_detector import __version__
from aigm_detector.ensemble import EnsembleDetector, EnsembleResult
from aigm_detector.preprocess import SUPPORTED_FORMATS, AudioInput
from aigm_detector.youtube import YouTubeDownloader

app = typer.Typer(
    name="aigm-detector",
    help="AI-Generated Music Detector - Ensemble detection tool (PyTorch or Apple MLX)",
    no_args_is_help=True,
)
console = Console()

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Shared options
ModelsOpt = typer.Option(
    None,
    "--models",
    "-m",
    help="Model names, a family prefix (e.g. 'spectttra'), 'auto' (fit this machine; default), "
    "'default' or 'all'",
)
BackendOpt = typer.Option(
    None,
    "--backend",
    "-b",
    help="auto | torch | mlx (default: $AIGM_BACKEND or auto = MLX on Apple Silicon)",
)
LowMemOpt = typer.Option(
    None,
    "--low-memory/--no-low-memory",
    help="Unload each model after use (default: decided from free memory)",
)


def _build_detector(
    models: Optional[str], backend: Optional[str], low_memory: Optional[bool] = None
) -> EnsembleDetector:
    detector = EnsembleDetector(model_names=models, backend=backend, low_memory=low_memory)
    names = ", ".join(m.name for m in detector.models)
    extra = " · low-memory" if detector.low_memory else ""
    console.print(
        f"[dim]Models: {names} · backend: {detector.backend}{extra} "
        f"· tier: {detector.plan.tier}[/dim]"
    )
    for warning in detector.plan.warnings:
        console.print(f"[yellow]⚠️  {warning}[/yellow]")
    return detector


def _set_verbosity(verbose: bool) -> None:
    if verbose:
        logging.getLogger("aigm_detector").setLevel(logging.INFO)


@app.command()
def predict(
    audio_paths: List[Path] = typer.Argument(..., help="Path(s) to audio file(s)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show per-model scores"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    models: Optional[str] = ModelsOpt,
    backend: Optional[str] = BackendOpt,
    low_memory: Optional[bool] = LowMemOpt,
):
    """Detect if audio is AI-generated."""
    _set_verbosity(verbose)

    try:
        detector = _build_detector(models, backend, low_memory)

        outputs = []
        for audio_path in audio_paths:
            console.print(f"[dim]Loading {audio_path}...[/dim]")
            audio = AudioInput.from_file(audio_path)
            result = detector.predict(audio)

            if json_output:
                outputs.append({"file": str(audio_path), **result.to_dict()})
            else:
                _print_result_pretty(audio_path, result, verbose=verbose)

        if json_output:
            console.print_json(data=outputs[0] if len(outputs) == 1 else outputs)

    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            logger.exception("Detailed error:")
        raise typer.Exit(1)


@app.command()
def predict_youtube(
    url: str = typer.Argument(..., help="YouTube URL"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show per-model scores"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    models: Optional[str] = ModelsOpt,
    backend: Optional[str] = BackendOpt,
    keep_file: bool = typer.Option(False, "--keep", help="Keep downloaded audio file"),
):
    """Download audio from YouTube and detect if AI-generated."""
    _set_verbosity(verbose)

    downloader = YouTubeDownloader()

    try:
        console.print(f"[dim]Downloading from {url}...[/dim]")
        audio_path = downloader.download(url)
        console.print(f"[dim]Downloaded to {audio_path}[/dim]")

        audio = AudioInput.from_file(audio_path)
        detector = _build_detector(models, backend)
        result = detector.predict(audio)

        if json_output:
            console.print_json(data={"url": url, "file": str(audio_path), **result.to_dict()})
        else:
            _print_result_pretty(audio_path, result, verbose=verbose)

        if not keep_file:
            audio_path.unlink(missing_ok=True)
            console.print("[dim]Cleaned up downloaded file[/dim]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            logger.exception("Detailed error:")
        raise typer.Exit(1)


@app.command()
def predict_batch(
    input_path: Path = typer.Argument(..., help="Directory or glob pattern"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output CSV path"),
    models: Optional[str] = ModelsOpt,
    backend: Optional[str] = BackendOpt,
):
    """Process multiple audio files (models are loaded once and reused)."""
    try:
        if input_path.is_dir():
            files = sorted(
                f for f in input_path.glob("**/*.*") if f.suffix.lower() in SUPPORTED_FORMATS
            )
        else:
            files = sorted(Path(".").glob(str(input_path)))

        if not files:
            console.print(f"[red]No audio files found in {input_path}[/red]")
            raise typer.Exit(1)

        console.print(f"[dim]Found {len(files)} audio files[/dim]")
        detector = _build_detector(models, backend)
        model_names = [m.name for m in detector.models]

        results = []
        for file_path in tqdm(files, desc="Processing"):
            row = {"file": str(file_path)}
            try:
                result = detector.predict(AudioInput.from_file(file_path))
                row.update(
                    verdict=result.verdict,
                    ai_probability=round(result.ai_probability, 4),
                    processing_time_ms=round(result.processing_time_ms, 1),
                    **{n: round(r.score, 4) for n, r in result.model_scores.items()},
                )
            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")
                row.update(verdict="ERROR", ai_probability=None, processing_time_ms=None)
            results.append(row)

        if output:
            fields = ["file", "verdict", "ai_probability", "processing_time_ms", *model_names]
            with output.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(results)
            console.print(f"[green]Results written to {output}[/green]")

        console.print("\n[bold]Summary:[/bold]")
        console.print(f"  Total: {len(results)}")
        for verdict in ("AI", "HUMAN", "UNCERTAIN", "ERROR"):
            console.print(f"  {verdict.title()}: {sum(r['verdict'] == verdict for r in results)}")

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("models")
def list_models():
    """List available detector models."""
    from aigm_detector.models import DEFAULT_MODELS, get_available_models
    from aigm_detector.resources import RUNTIME_OVERHEAD_GB, plan

    auto = plan()
    table = Table(title="Available models")
    table.add_column("Name", style="bold", no_wrap=True)
    table.add_column("Default", justify="center")
    table.add_column("Auto", justify="center")
    table.add_column("Memory", justify="right")
    table.add_column("Fits", justify="center")
    table.add_column("Description")

    for name, cls in get_available_models().items():
        fits = RUNTIME_OVERHEAD_GB + cls.memory_gb <= auto.memory_budget_gb
        table.add_row(
            name,
            "✓" if name in DEFAULT_MODELS else "",
            "✓" if name in auto.models else "",
            f"{cls.memory_gb:.2f} GB",
            "[green]✓[/green]" if fits else "[red]✗[/red]",
            cls.description,
        )
    console.print(table)
    console.print(
        f"[dim]Auto = picked for this machine (tier '{auto.tier}', "
        f"budget {auto.memory_budget_gb:.1f} GB). All models run on torch and mlx. "
        "Checkpoints download from Hugging Face on first use.[/dim]"
    )


@app.command()
def doctor(
    backend: Optional[str] = BackendOpt,
):
    """Show this machine's resources and the settings auto mode picks for it."""
    from aigm_detector.device import describe_environment
    from aigm_detector.hub import cache_dir
    from aigm_detector.resources import detect, plan

    profile = detect()
    info = describe_environment()
    console.print("[bold]Machine[/bold]")
    rows = {
        "chip": profile.chip,
        "platform": info["platform"],
        "cpu_cores": profile.cpu_cores,
        "ram": f"{profile.total_ram_gb:.1f} GB total · {profile.available_ram_gb:.1f} GB free",
        "gpu_memory": f"{profile.gpu_memory_gb:.1f} GB" if profile.gpu_memory_gb else "—",
        "python": info["python"],
        "torch": f"{info['torch']} ({info['torch_device']})",
        "mlx": f"{info['mlx']} ({info['mlx_device']})" if info["mlx"] else "not installed",
        "cache_dir": str(cache_dir()),
    }
    for key, value in rows.items():
        console.print(f"  {key:12s}: {value}")

    auto = plan(backend=backend)
    console.print("\n[bold]Auto settings[/bold]")
    console.print(f"  {'tier':12s}: {auto.tier}")
    console.print(f"  {'backend':12s}: {auto.backend}")
    console.print(f"  {'models':12s}: {', '.join(auto.models)}")
    console.print(f"  {'low_memory':12s}: {auto.low_memory}")
    console.print(
        f"  {'memory':12s}: ~{auto.required_gb:.1f} GB needed · {auto.memory_budget_gb:.1f} GB budget"
    )
    for model, reason in auto.excluded.items():
        console.print(f"  [dim]skipped {model}: {reason}[/dim]")
    for note in auto.notes[1:]:
        console.print(f"  [dim]{note}[/dim]")
    for warning in auto.warnings:
        console.print(f"  [yellow]⚠️  {warning}[/yellow]")
    console.print(
        "\n[dim]Override with --models / --backend / --low-memory, or AIGM_TIER=lite|standard|full[/dim]"
    )


@app.command()
def download(
    models: Optional[str] = ModelsOpt,
    backend: Optional[str] = BackendOpt,
):
    """Pre-download (and convert for MLX) model weights, e.g. before going offline."""
    detector = _build_detector(models, backend)
    for model in detector.models:
        with console.status(f"Fetching {model.name}..."):
            ok = model.ensure_loaded()
        status = "[green]ok[/green]" if ok else f"[red]failed: {model._load_error}[/red]"
        console.print(f"  {model.name:22s} {status}")


@app.command()
def benchmark(
    models: Optional[str] = ModelsOpt,
    backends: str = typer.Option("torch,mlx", help="Comma-separated backends to compare"),
    duration: float = typer.Option(180.0, help="Seconds of synthetic audio (a typical song)"),
    runs: int = typer.Option(3, help="Timed runs per model (after one warm-up)"),
):
    """Compare load time, latency and peak memory across backends."""
    from aigm_detector.device import mlx_available
    from aigm_detector.models import get_model, resolve_model_names

    rng = np.random.default_rng(0)
    samples = rng.standard_normal(int(duration * 16000)).astype(np.float32) * 0.1

    requested = [b.strip() for b in backends.split(",") if b.strip()]
    if "mlx" in requested and not mlx_available():
        console.print("[yellow]MLX not installed, skipping mlx backend[/yellow]")
        requested.remove("mlx")

    table = Table(title=f"{duration:.0f}s of audio · median of {runs} runs")
    for column in ("Model", "Backend", "Load", "1st run", "Median", "Peak mem", "Score"):
        table.add_column(column, justify="left" if column in ("Model", "Backend") else "right")

    for name in resolve_model_names(models):
        scores = []
        for backend_name in requested:
            detector = get_model(name)(backend=backend_name)
            if detector.backend != backend_name:
                table.add_row(name, backend_name, *["n/a"] * 5)
                continue

            audio = AudioInput.from_array(samples, 16000)
            _reset_peak_memory(backend_name)
            t0 = time.perf_counter()
            loaded = detector.ensure_loaded()
            load_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            result = detector.predict(audio) if loaded else None
            first_s = time.perf_counter() - t0
            if result is None:
                table.add_row(name, backend_name, *["failed"] * 5)
                continue

            times = []
            for _ in range(runs):
                t0 = time.perf_counter()
                result = detector.predict(audio)
                times.append(time.perf_counter() - t0)
            scores.append(result.score)
            table.add_row(
                name,
                backend_name,
                f"{load_s:.1f}s",
                f"{first_s:.2f}s",
                f"{np.median(times):.2f}s",
                _peak_memory(backend_name),
                f"{result.score:.4f}",
            )
            detector.unload()

        if len(scores) > 1:
            table.add_row("", "[dim]max Δ[/dim]", "", "", "", "", f"[dim]{max(scores) - min(scores):.1e}[/dim]")

    console.print(table)
    console.print("[dim]Peak mem: MLX allocator / CUDA allocator; MPS shows driver memory; CPU n/a.[/dim]")


def _reset_peak_memory(backend: str) -> None:
    if backend == "mlx":
        import mlx.core as mx

        mx.reset_peak_memory()
        return
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_memory(backend: str) -> str:
    if backend == "mlx":
        import mlx.core as mx

        return f"{mx.get_peak_memory() / 2**30:.2f} GB"
    import torch

    from aigm_detector.device import get_device

    kind = get_device().type
    if kind == "cuda":
        return f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GB"
    if kind == "mps":
        return f"{torch.mps.driver_allocated_memory() / 2**30:.2f} GB"
    return "n/a"


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind host"),
    port: int = typer.Option(8000, help="Bind port"),
):
    """Start FastAPI server."""
    try:
        import uvicorn

        from aigm_detector.api import app as api_app
    except ImportError:
        console.print("[red]FastAPI not installed. Install with: pip install aigm-detector[api][/red]")
        raise typer.Exit(1)

    console.print(f"[green]Starting API server at http://{host}:{port}[/green]")
    uvicorn.run(api_app, host=host, port=port)


@app.command()
def demo(
    port: int = typer.Option(7860, help="Port for Gradio interface"),
    share: bool = typer.Option(False, help="Create public sharing link"),
):
    """Launch Gradio demo."""
    try:
        from aigm_detector.demo import launch
    except ImportError:
        console.print("[red]Gradio not installed. Install with: pip install aigm-detector[demo][/red]")
        raise typer.Exit(1)

    console.print(f"[green]Launching Gradio demo at http://localhost:{port}[/green]")
    launch(port=port, share=share)


@app.command()
def version():
    """Show version."""
    console.print(f"aigm-detector version {__version__}")


def _print_result_pretty(
    file_path: Path,
    result: EnsembleResult,
    verbose: bool = False,
):
    """Pretty-print detection result."""
    if result.verdict == "AI":
        verdict_str = f"[red bold]{result.verdict}[/red bold]"
        confidence_pct = result.ai_probability * 100
    elif result.verdict == "HUMAN":
        verdict_str = f"[green bold]{result.verdict}[/green bold]"
        confidence_pct = (1 - result.ai_probability) * 100
    else:
        verdict_str = f"[yellow bold]{result.verdict}[/yellow bold]"
        confidence_pct = 50.0

    console.print(f"\n[bold]{file_path.name}[/bold]")
    console.print(f"  Verdict    : {verdict_str} ({confidence_pct:.0f}% confidence)")

    if verbose or result.warnings:
        console.print(f"  AI Prob    : {result.ai_probability:.3f}")

        if verbose:
            console.print("\n  [dim]Model Scores:[/dim]")
            for name, det_result in result.model_scores.items():
                filled = int(det_result.score * 20)
                bar = "█" * filled + "░" * (20 - filled)
                windows = det_result.metadata.get("window_scores") or []
                spread = (
                    f" [dim]({len(windows)} windows, {min(windows):.2f}–{max(windows):.2f})[/dim]"
                    if len(windows) > 1
                    else ""
                )
                console.print(
                    f"    {name:22s} {bar} {det_result.score:.2f} "
                    f"[dim]{det_result.processing_time_ms / 1000:.2f}s[/dim]{spread}"
                )

        console.print(
            f"\n  Time       : {result.processing_time_ms / 1000:.2f}s ({result.backend})"
        )

        if result.warnings:
            console.print("\n  [yellow]Warnings:[/yellow]")
            for warning in result.warnings:
                console.print(f"    ⚠️  {warning}")

    console.print()


def main():
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
