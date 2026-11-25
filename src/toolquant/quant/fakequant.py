"""Weight-only per-group symmetric round-to-nearest fake quantization (CLAUDE.md section 5)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import nn

GROUP_SIZE = 128


def fake_quantize_tensor(w: torch.Tensor, bits: int, group_size: int = GROUP_SIZE) -> torch.Tensor:
    """Quantize-dequantize a (out, in) weight, grouping along the input dimension.

    scale = max|w| / (2^(bits-1) - 1) per group; q = clamp(round(w/scale), -2^(bits-1),
    2^(bits-1) - 1); W_hat = q * scale. A trailing partial group is its own group.
    """
    if bits not in (4, 8):
        raise ValueError("bits must be 4 or 8")
    in_dim = w.shape[1]
    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))
    x = w.detach().to(torch.float32)
    pieces = []
    for start in range(0, in_dim, group_size):
        g = x[:, start : start + group_size]
        scale = g.abs().amax(dim=1, keepdim=True) / qmax
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        q = torch.clamp(torch.round(g / scale), qmin, qmax)
        pieces.append(q * scale)
    return torch.cat(pieces, dim=1).to(w.dtype)


@contextmanager
def quantized(
    model: nn.Module,
    module_names: list[str],
    bits: int,
    group_size: int = GROUP_SIZE,
) -> Iterator[None]:
    """Fake-quantize the named linear layers in place; always restore the original weights."""
    mods = dict(model.named_modules())
    saved: dict[str, torch.Tensor] = {}
    try:
        for n in module_names:
            lin = mods[n]
            saved[n] = lin.weight.data.clone()
            lin.weight.data.copy_(fake_quantize_tensor(lin.weight.data, bits, group_size))
        yield
    finally:
        for n, w in saved.items():
            mods[n].weight.data.copy_(w)


@contextmanager
def apply_map(
    model: nn.Module,
    groups: dict[str, list[str]],
    bits_per_group: dict[str, int],
    group_size: int = GROUP_SIZE,
) -> Iterator[None]:
    """Fake-quantize every group at its assigned precision (16 = leave as is); restore on exit."""
    mods = dict(model.named_modules())
    saved: dict[str, torch.Tensor] = {}
    try:
        for g, bits in bits_per_group.items():
            if bits >= 16:
                continue
            for n in groups[g]:
                lin = mods[n]
                saved[n] = lin.weight.data.clone()
                lin.weight.data.copy_(fake_quantize_tensor(lin.weight.data, bits, group_size))
        yield
    finally:
        for n, w in saved.items():
            mods[n].weight.data.copy_(w)
