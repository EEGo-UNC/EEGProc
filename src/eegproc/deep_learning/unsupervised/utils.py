"""Shared helpers for unsupervised deep-learning modules."""

from __future__ import annotations

from collections.abc import Sequence


def _ensure_tuple(value, length: int, name: str):
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = tuple(value)
        if len(result) != length:
            raise ValueError(
                f"{name} must have length {length}, got {len(result)}."
            )
        return result

    return tuple([value] * length)


def _product(values: Sequence[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result