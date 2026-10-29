"""SONICS SpecTTTra (ICLR 2025) in MLX.

Mirrors `sonics.models.AudioClassifier` with a SpecTTTra encoder (inference only).
Parameter names follow the original PyTorch state dict so checkpoints convert 1:1.
Input is the normalized, resized mel spectrogram (B, n_mels, frames); see
`aigm_detector.models.sonics_net.SonicsFeatureExtractor`.
"""

import math
from pathlib import Path
from typing import Dict, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def sinusoid_table(num_tokens: int, dims: int) -> np.ndarray:
    position = np.arange(num_tokens, dtype=np.float32)[:, None]
    div_term = np.exp(np.arange(0, dims, 2, dtype=np.float32) * (-math.log(10000.0) / dims))
    pe = np.zeros((num_tokens, dims), dtype=np.float32)
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe[None]


class PositionalEncoding(nn.Module):
    def __init__(self, num_tokens: int, dims: int):
        super().__init__()
        self.pe = mx.array(sinusoid_table(num_tokens, dims))

    def __call__(self, x: mx.array) -> mx.array:
        return x + self.pe


class Tokenizer1D(nn.Module):
    def __init__(self, input_dim: int, dims: int, clip: int, num_clips: int, pre_norm: bool):
        super().__init__()
        self.conv1d = nn.Conv1d(input_dim, dims, clip, stride=clip, bias=not pre_norm)
        self.pos_encoder = PositionalEncoding(num_clips, dims)
        self.norm_pre = nn.LayerNorm(dims, eps=1e-6) if pre_norm else None

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, length, channels) — MLX convs are channels-last
        x = self.pos_encoder(nn.gelu(self.conv1d(x)))
        return self.norm_pre(x) if self.norm_pre is not None else x


class STTokenizer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        spec_dim, temp_dim = m["input_shape"]
        dims, t_clip, f_clip = m["embed_dim"], m["t_clip"], m["f_clip"]
        pre_norm = m.get("pre_norm", False)
        self.num_temporal_tokens = (temp_dim - t_clip) // t_clip + 1
        self.num_spectral_tokens = (spec_dim - f_clip) // f_clip + 1
        self.temporal_tokenizer = Tokenizer1D(
            spec_dim, dims, t_clip, self.num_temporal_tokens, pre_norm
        )
        self.spectral_tokenizer = Tokenizer1D(
            temp_dim, dims, f_clip, self.num_spectral_tokens, pre_norm
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, F, T). Temporal tokens convolve over T with F channels, and vice versa.
        temporal = self.temporal_tokenizer(x.transpose(0, 2, 1))
        spectral = self.spectral_tokenizer(x)
        return mx.concatenate([temporal, spectral], axis=1)


class Attention(nn.Module):
    def __init__(self, dims: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dims // num_heads) ** -0.5
        self.qkv = nn.Linear(dims, dims * 3, bias=False)
        self.proj = nn.Linear(dims, dims)

    def __call__(self, x: mx.array) -> mx.array:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, -1).transpose(2, 0, 3, 1, 4)
        out = mx.fast.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], scale=self.scale)
        return self.proj(out.transpose(0, 2, 1, 3).reshape(B, L, D))


class MLP(nn.Module):
    def __init__(self, dims: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dims, hidden)
        self.fc2 = nn.Linear(hidden, dims)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dims: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dims)
        self.attn = Attention(dims, num_heads)
        self.norm2 = nn.LayerNorm(dims)
        self.mlp = MLP(dims, int(dims * mlp_ratio))

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class Transformer(nn.Module):
    def __init__(self, dims: int, num_heads: int, num_layers: int, mlp_ratio: float):
        super().__init__()
        self.blocks = [Block(dims, num_heads, mlp_ratio) for _ in range(num_layers)]

    def __call__(self, x: mx.array) -> mx.array:
        for block in self.blocks:
            x = block(x)
        return x


class Encoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        self.st_tokenizer = STTokenizer(cfg)
        self.transformer = Transformer(
            m["embed_dim"], m["num_heads"], m["num_layers"], m.get("mlp_ratio", 4.0)
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.transformer(self.st_tokenizer(x))


class MLXSpecTTTra(nn.Module):
    """SpecTTTra classifier returning one logit per example (sigmoid -> p(fake))."""

    def __init__(self, cfg: dict):
        super().__init__()
        if cfg["model"]["name"] != "SpecTTTra":
            raise ValueError(f"Unsupported SONICS architecture: {cfg['model']['name']}")
        self.config = cfg
        self.encoder = Encoder(cfg)
        self.classifier = nn.Linear(cfg["model"]["embed_dim"], cfg["num_classes"])

    def __call__(self, spec: mx.array) -> mx.array:
        return self.classifier(self.encoder(spec).mean(axis=1))


def convert_spectttra_weights(state_dict: Dict[str, np.ndarray], cfg: dict) -> Dict[str, mx.array]:
    """Map a SONICS PyTorch state dict (as numpy arrays) to MLX names/layouts."""
    tokens = {
        "temporal_tokenizer": STTokenizer(cfg).num_temporal_tokens,
        "spectral_tokenizer": STTokenizer(cfg).num_spectral_tokens,
    }
    weights = {}
    for key, value in state_dict.items():
        if not (key.startswith("encoder.") or key.startswith("classifier.")):
            continue  # feature extractor buffers / augmentation are handled elsewhere
        if key.endswith("conv1d.weight"):
            value = value.transpose(0, 2, 1)  # (O, I, K) -> (O, K, I)
        if key.endswith("pos_encoder.pe"):
            which = key.split(".")[2]
            value = value[:, : tokens[which]]  # sinusoid buffers are stored with max_len rows
        weights[key] = mx.array(np.ascontiguousarray(value, dtype=np.float32))
    return weights


def load_spectttra(
    cfg: dict,
    weights: Union[str, Path, Dict[str, mx.array]],
    dtype: mx.Dtype = mx.float32,
) -> MLXSpecTTTra:
    """Build a SpecTTTra model from a config and converted weights (dict or .safetensors)."""
    if not isinstance(weights, dict):
        weights = mx.load(str(weights))
    model = MLXSpecTTTra(cfg)
    model.load_weights([(k, v.astype(dtype)) for k, v in weights.items()], strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model
