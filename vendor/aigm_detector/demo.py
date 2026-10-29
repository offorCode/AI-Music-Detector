"""Gradio web demo. Run with `aigm-detector demo`."""

from functools import lru_cache

import gradio as gr

from aigm_detector.ensemble import EnsembleDetector
from aigm_detector.models import get_available_models
from aigm_detector.preprocess import AudioInput
from aigm_detector.resources import plan


@lru_cache(maxsize=4)
def _detector(models: tuple, backend: str) -> EnsembleDetector:
    return EnsembleDetector(model_names=list(models), backend=backend)


def analyze(audio_path, models, backend):
    if not audio_path:
        return {}, {"message": "Upload an audio file"}
    if not models:
        return {}, {"message": "Select at least one model"}

    result = _detector(tuple(sorted(models)), backend).predict(AudioInput.from_file(audio_path))
    label = {"AI-generated": result.ai_probability, "Human": 1 - result.ai_probability}
    return label, result.to_dict()


def build() -> gr.Blocks:
    with gr.Blocks(title="AIGM Detector") as ui:
        gr.Markdown(
            "# AIGM Detector\nIs this track AI-generated? Research tool — not for rights enforcement."
        )
        with gr.Row():
            with gr.Column():
                audio = gr.Audio(label="Audio", type="filepath")
                models = gr.CheckboxGroup(
                    choices=list(get_available_models()), value=plan().models, label="Models (preselected for this machine)"
                )
                backend = gr.Radio(["auto", "torch", "mlx"], value="auto", label="Backend")
                run = gr.Button("Analyze", variant="primary")
            with gr.Column():
                verdict = gr.Label(label="Verdict", num_top_classes=2)
                details = gr.JSON(label="Details")
        run.click(analyze, inputs=[audio, models, backend], outputs=[verdict, details])
    return ui


def launch(port: int = 7860, share: bool = False) -> None:
    build().launch(server_port=port, share=share)
