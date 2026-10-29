"""Self-contained PyTorch SpecTTTra + SONICS feature extraction.

A dependency-free re-implementation of `sonics.models.AudioClassifier`
(https://github.com/awsaf49/sonics) for inference. It loads the official
`pytorch_model.bin` checkpoints without needing the `sonics` or `timm` packages,
and its parameter names match the original state dict exactly.
"""

import math
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram


class SonicsFeatureExtractor:
    """Waveform -> normalized log-mel spectrogram, resized to the model input shape."""

    def __init__(self, cfg: dict):
        mel = cfg["melspec"]
        self.melspec = MelSpectrogram(
            n_fft=mel["n_fft"],
            hop_length=mel["hop_length"],
            win_length=mel["win_length"],
            n_mels=mel["n_mels"],
            sample_rate=cfg["audio"]["sample_rate"],
            f_min=mel["f_min"],
            f_max=mel["f_max"],
            power=mel["power"],
        )
        self.to_db = AmplitudeToDB(top_db=mel["top_db"])
        self.norm = mel.get("norm", "mean_std")
        self.input_shape = tuple(cfg["model"]["input_shape"])

    @torch.no_grad()
    def __call__(self, windows: np.ndarray) -> np.ndarray:
        """(N, samples) -> (N, n_mels, frames) float32.

        Windows are processed one at a time on purpose: torchaudio's top_db
        clamp is computed over the whole batch, which would couple windows.
        """
        return np.stack([self._single(w) for w in windows])

    def _single(self, window: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(window, dtype=np.float32))[None]
        spec = self.to_db(self.melspec(x))  # (1, F, T)
        if self.norm == "mean_std":
            spec = (spec - spec.mean()) / (spec.std() + 1e-6)
        elif self.norm == "min_max":
            spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-6)
        elif self.norm == "simple":
            spec = (spec - 40) / 80
        spec = F.interpolate(spec[:, None], size=self.input_shape, mode="bilinear")
        return spec[0, 0].numpy()


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, dims: int, num_tokens: int, learnable: bool):
        super().__init__()
        if learnable:
            self.pe = nn.Parameter(torch.zeros(1, num_tokens, dims))
        else:
            position = torch.arange(num_tokens, dtype=torch.float32)[:, None]
            div = torch.exp(torch.arange(0, dims, 2).float() * (-math.log(10000.0) / dims))
            pe = torch.zeros(1, num_tokens, dims)
            pe[0, :, 0::2] = torch.sin(position * div)
            pe[0, :, 1::2] = torch.cos(position * div)
            self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]]


class Tokenizer1D(nn.Module):
    def __init__(self, input_dim, dims, clip, num_clips, pre_norm, pe_learnable):
        super().__init__()
        self.conv1d = nn.Conv1d(input_dim, dims, clip, stride=clip, bias=not pre_norm)
        self.act = nn.GELU()
        self.pos_encoder = LearnedPositionalEncoding(dims, num_clips, pe_learnable)
        self.norm_pre = nn.LayerNorm(dims, eps=1e-6) if pre_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1d(x)).transpose(1, 2)
        return self.norm_pre(self.pos_encoder(x))


class STTokenizer(nn.Module):
    def __init__(self, m: dict):
        super().__init__()
        spec_dim, temp_dim = m["input_shape"]
        t_clip, f_clip = m["t_clip"], m["f_clip"]
        pre_norm, pe_learnable = m.get("pre_norm", False), m.get("pe_learnable", False)
        self.temporal_tokenizer = Tokenizer1D(
            spec_dim, m["embed_dim"], t_clip, (temp_dim - t_clip) // t_clip + 1,
            pre_norm, pe_learnable,
        )
        self.spectral_tokenizer = Tokenizer1D(
            temp_dim, m["embed_dim"], f_clip, (spec_dim - f_clip) // f_clip + 1,
            pre_norm, pe_learnable,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self.temporal_tokenizer(x), self.spectral_tokenizer(x.transpose(1, 2))], dim=1
        )


class Attention(nn.Module):
    def __init__(self, dims: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dims, dims * 3, bias=False)
        self.proj = nn.Linear(dims, dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class Mlp(nn.Module):
    def __init__(self, dims: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dims, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dims: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dims)
        self.attn = Attention(dims, num_heads)
        self.norm2 = nn.LayerNorm(dims)
        self.mlp = Mlp(dims, int(dims * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class Transformer(nn.Module):
    def __init__(self, dims, num_heads, num_layers, mlp_ratio):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Block(dims, num_heads, mlp_ratio) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class Encoder(nn.Module):
    def __init__(self, m: dict):
        super().__init__()
        self.st_tokenizer = STTokenizer(m)
        self.transformer = Transformer(
            m["embed_dim"], m["num_heads"], m["num_layers"], m.get("mlp_ratio", 4.0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer(self.st_tokenizer(x))


class TorchSpecTTTra(nn.Module):
    """SpecTTTra classifier. Input: (B, n_mels, frames) from SonicsFeatureExtractor."""

    def __init__(self, cfg: dict):
        super().__init__()
        if cfg["model"]["name"] != "SpecTTTra":
            raise ValueError(f"Unsupported SONICS architecture: {cfg['model']['name']}")
        self.encoder = Encoder(cfg["model"])
        self.classifier = nn.Linear(cfg["model"]["embed_dim"], cfg["num_classes"])

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(spec).mean(dim=1))

    def load_sonics_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load an official checkpoint, ignoring feature-extractor buffers."""
        own = self.state_dict()
        filtered = {}
        for key, value in state_dict.items():
            if key not in own:
                continue
            if key.endswith("pos_encoder.pe"):
                value = value[:, : own[key].shape[1]]
            filtered[key] = value
        missing = sorted(set(own) - set(filtered))
        if missing:
            raise RuntimeError(f"Checkpoint is missing SpecTTTra weights: {missing[:5]}")
        self.load_state_dict(filtered, strict=True)
