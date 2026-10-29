"""Audio Spectrogram Transformer (AST) in MLX.

Mirrors `transformers.ASTForAudioClassification` (inference only) and loads the
same Hugging Face `model.safetensors` checkpoints, from either the transformers
v4 (`encoder.layer.N.attention.attention.query`) or v5 (`layers.N.attention.q_proj`)
parameter layout.
"""

import json
import re
from pathlib import Path
from typing import Dict, Union

import mlx.core as mx
import mlx.nn as nn

PREFIX = "audio_spectrogram_transformer."

# transformers v4 names -> v5 names (which this module mirrors)
_V4_TO_V5 = [
    (r"encoder\.layer\.(\d+)\.attention\.attention\.query\.", r"layers.\1.attention.q_proj."),
    (r"encoder\.layer\.(\d+)\.attention\.attention\.key\.", r"layers.\1.attention.k_proj."),
    (r"encoder\.layer\.(\d+)\.attention\.attention\.value\.", r"layers.\1.attention.v_proj."),
    (r"encoder\.layer\.(\d+)\.attention\.output\.dense\.", r"layers.\1.attention.o_proj."),
    (r"encoder\.layer\.(\d+)\.intermediate\.dense\.", r"layers.\1.mlp.fc1."),
    (r"encoder\.layer\.(\d+)\.output\.dense\.", r"layers.\1.mlp.fc2."),
    (r"encoder\.layer\.(\d+)\.layernorm_(before|after)\.", r"layers.\1.layernorm_\2."),
]


class Attention(nn.Module):
    def __init__(self, dims: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dims // num_heads) ** -0.5
        self.q_proj = nn.Linear(dims, dims)
        self.k_proj = nn.Linear(dims, dims)
        self.v_proj = nn.Linear(dims, dims)
        self.o_proj = nn.Linear(dims, dims)

    def __call__(self, x: mx.array) -> mx.array:
        B, L, D = x.shape
        q, k, v = (
            proj(x).reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
            for proj in (self.q_proj, self.k_proj, self.v_proj)
        )
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, D))


class MLP(nn.Module):
    def __init__(self, dims: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dims, hidden)
        self.fc2 = nn.Linear(hidden, dims)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class Layer(nn.Module):
    def __init__(self, dims: int, num_heads: int, hidden: int, eps: float):
        super().__init__()
        self.layernorm_before = nn.LayerNorm(dims, eps=eps)
        self.attention = Attention(dims, num_heads)
        self.layernorm_after = nn.LayerNorm(dims, eps=eps)
        self.mlp = MLP(dims, hidden)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attention(self.layernorm_before(x))
        return x + self.mlp(self.layernorm_after(x))


class PatchEmbeddings(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.projection = nn.Conv2d(
            1,
            cfg["hidden_size"],
            kernel_size=cfg["patch_size"],
            stride=(cfg["frequency_stride"], cfg["time_stride"]),
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, time, mel) -> NHWC image with H=mel, W=time
        x = x.transpose(0, 2, 1)[..., None]
        x = self.projection(x)  # (B, f, t, D)
        B, f, t, D = x.shape
        return x.reshape(B, f * t, D)  # frequency-major, same order as PyTorch


class Embeddings(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        dims = cfg["hidden_size"]
        f = (cfg["num_mel_bins"] - cfg["patch_size"]) // cfg["frequency_stride"] + 1
        t = (cfg["max_length"] - cfg["patch_size"]) // cfg["time_stride"] + 1
        self.cls_token = mx.zeros((1, 1, dims))
        self.distillation_token = mx.zeros((1, 1, dims))
        self.position_embeddings = mx.zeros((1, f * t + 2, dims))
        self.patch_embeddings = PatchEmbeddings(cfg)

    def __call__(self, x: mx.array) -> mx.array:
        patches = self.patch_embeddings(x)
        B = patches.shape[0]
        cls = mx.broadcast_to(self.cls_token, (B, 1, patches.shape[-1]))
        dist = mx.broadcast_to(self.distillation_token, (B, 1, patches.shape[-1]))
        return mx.concatenate([cls, dist, patches], axis=1) + self.position_embeddings


class Classifier(nn.Module):
    def __init__(self, dims: int, num_labels: int, eps: float):
        super().__init__()
        self.layernorm = nn.LayerNorm(dims, eps=eps)
        self.dense = nn.Linear(dims, num_labels)

    def __call__(self, x: mx.array) -> mx.array:
        return self.dense(self.layernorm(x))


class MLXAST(nn.Module):
    """AST classifier. Input: fbank features (B, max_length, num_mel_bins)."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.config = cfg
        dims = cfg["hidden_size"]
        eps = cfg.get("layer_norm_eps", 1e-12)
        self.embeddings = Embeddings(cfg)
        self.layers = [
            Layer(dims, cfg["num_attention_heads"], cfg["intermediate_size"], eps)
            for _ in range(cfg["num_hidden_layers"])
        ]
        self.layernorm = nn.LayerNorm(dims, eps=eps)
        self.classifier = Classifier(dims, len(cfg.get("id2label", {0: 0, 1: 1})), eps)

    def __call__(self, input_values: mx.array) -> mx.array:
        x = self.embeddings(input_values)
        for layer in self.layers:
            x = layer(x)
        x = self.layernorm(x)
        pooled = (x[:, 0] + x[:, 1]) / 2  # cls + distillation tokens
        return self.classifier(pooled)


def convert_ast_weights(state_dict: Dict[str, "object"]) -> Dict[str, mx.array]:
    """Map a PyTorch/safetensors AST state dict to MLX parameter names/layouts."""
    weights = {}
    for key, value in state_dict.items():
        if not isinstance(value, mx.array):
            value = mx.array(value.detach().cpu().float().numpy() if hasattr(value, "detach") else value)
        key = key[len(PREFIX):] if key.startswith(PREFIX) else key
        for pattern, repl in _V4_TO_V5:
            key = re.sub(pattern, repl, key)
        if key.endswith("patch_embeddings.projection.weight"):
            value = value.transpose(0, 2, 3, 1)  # (O, I, kH, kW) -> (O, kH, kW, I)
        weights[key] = value
    return weights


def load_ast(model_dir: Union[str, Path], dtype: mx.Dtype = mx.float32) -> MLXAST:
    """Load an AST checkpoint directory (config.json + model.safetensors)."""
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text())
    model = MLXAST(cfg)
    weights = convert_ast_weights(mx.load(str(model_dir / "model.safetensors")))
    weights = {k: v.astype(dtype) for k, v in weights.items()}
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model
