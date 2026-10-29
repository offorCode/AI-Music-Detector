"""Apple MLX implementations of the detector networks.

Only the neural networks run in MLX; feature extraction stays in NumPy/PyTorch
on the CPU (it is cheap and must match training-time preprocessing exactly).
Importing this package requires `mlx` (pip install 'aigm-detector[mlx]').
"""

from aigm_detector.mlx_backend.ast import MLXAST, load_ast
from aigm_detector.mlx_backend.spectttra import MLXSpecTTTra, load_spectttra

__all__ = ["MLXAST", "load_ast", "MLXSpecTTTra", "load_spectttra"]
