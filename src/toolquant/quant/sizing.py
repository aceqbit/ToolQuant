"""Size model: size_bits(group) = params * (bits + 16 / group_size). Unquantized params: 16 bits."""

from __future__ import annotations

GROUP_SIZE = 128


def group_bits(params: int, bits: int, group_size: int = GROUP_SIZE) -> float:
    return params * (bits + 16 / group_size)


def total_bits(
    group_params: dict[str, int],
    bits_per_group: dict[str, int],
    other_params: int,
    group_size: int = GROUP_SIZE,
) -> float:
    quant = sum(group_bits(p, bits_per_group[g], group_size) for g, p in group_params.items())
    return quant + other_params * 16


def uniform_bits(
    group_params: dict[str, int], bits: int, other_params: int, group_size: int = GROUP_SIZE
) -> float:
    return total_bits(group_params, {g: bits for g in group_params}, other_params, group_size)


def budget(
    f: float, group_params: dict[str, int], other_params: int, group_size: int = GROUP_SIZE
) -> float:
    """budget(f) = size_int4 + f * (size_int8 - size_int4)."""
    s4 = uniform_bits(group_params, 4, other_params, group_size)
    s8 = uniform_bits(group_params, 8, other_params, group_size)
    return s4 + f * (s8 - s4)


def to_mb(bits: float) -> float:
    return bits / 8 / 1e6
