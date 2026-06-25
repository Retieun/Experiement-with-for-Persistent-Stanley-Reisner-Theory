from __future__ import annotations

from typing import Sequence

Window = tuple[float, ...]
LabeledWindow = tuple[Window, str]


def labeled_windows(
    signal: Sequence[float],
    label: str,
    length: int = 4096,
    stride: int | None = None,
) -> tuple[LabeledWindow, ...]:
    if length < 1:
        raise ValueError("length must be at least 1")
    actual_stride = length if stride is None else stride
    if actual_stride < 1:
        raise ValueError("stride must be at least 1")

    values = tuple(float(value) for value in signal)
    return tuple(
        (values[start : start + length], label)
        for start in range(0, len(values) - length + 1, actual_stride)
    )
