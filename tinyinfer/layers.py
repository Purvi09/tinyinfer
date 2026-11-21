"""The building blocks of a Qwen3 transformer, written from scratch in plain PyTorch."""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Rescales each vector to a standard size, then applies a learned per-position weight.

    out = x / sqrt(mean(x^2) + eps) * weight
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Learned during training, loaded from the checkpoint. One number per position.
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()  # do the math in float32 for precision
        rms_inv = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)  # 1 / RMS
        x = x * rms_inv
        return self.weight * x.to(input_dtype)
