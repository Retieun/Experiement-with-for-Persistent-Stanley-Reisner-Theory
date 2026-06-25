from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from psrt_bearing.complex import Point, vietoris_rips_complex
from psrt_bearing.invariants import _reduced_betti_by_dimension


@dataclass(frozen=True)
class BaselineFeatures:
    values: tuple[int, ...]
    labels: tuple[str, ...]


def betti_curve_features(
    points: Iterable[Point],
    radii: Iterable[float],
    max_dim: int = 2,
    homology_dims: Iterable[int] = (0, 1),
) -> BaselineFeatures:
    point_list = [tuple(point) for point in points]
    dims = tuple(homology_dims)
    values: list[int] = []
    labels: list[str] = []

    for radius in radii:
        faces = vietoris_rips_complex(point_list, radius=radius, max_dim=max_dim)
        reduced = _reduced_betti_by_dimension(faces)
        for dim in dims:
            value = reduced.get(dim, 0)
            if dim == 0 and faces:
                value += 1
            values.append(value)
            labels.append(f"r={radius}:H_{dim}")
    return BaselineFeatures(values=tuple(values), labels=tuple(labels))
