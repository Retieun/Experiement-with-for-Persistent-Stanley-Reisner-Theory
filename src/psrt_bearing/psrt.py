from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from psrt_bearing.complex import Point, vietoris_rips_complex
from psrt_bearing.invariants import Face, _close_faces, _n_choose_k


@dataclass(frozen=True)
class VectorFeatures:
    values: tuple[int, ...]
    labels: tuple[str, ...]


def face_vector(faces: Iterable[Iterable[int]], max_dim: int | None = None) -> tuple[int, ...]:
    closed = _close_faces(faces)
    actual_max = max((len(face) - 1 for face in closed), default=-1)
    limit = actual_max if max_dim is None else max_dim
    values = [1]
    values.extend(sum(1 for face in closed if len(face) - 1 == dim) for dim in range(limit + 1))
    return tuple(values)


def h_vector(faces: Iterable[Iterable[int]], max_dim: int | None = None) -> tuple[int, ...]:
    f_values = face_vector(faces, max_dim=max_dim)
    dimension = len(f_values) - 2
    if dimension < 0:
        return (1,)

    h_values: list[int] = []
    for k in range(dimension + 1):
        total = 0
        for i in range(k + 1):
            total += ((-1) ** (k - i)) * _n_choose_k(dimension + 1 - i, k - i) * f_values[i]
        h_values.append(total)
    return tuple(h_values)


def persistent_face_h_features(
    points: Iterable[Point],
    radii: Iterable[float],
    max_dim: int = 2,
) -> VectorFeatures:
    point_list = [tuple(point) for point in points]
    radius_grid = tuple(radii)
    values: list[int] = []
    labels: list[str] = []
    for radius in radius_grid:
        faces = vietoris_rips_complex(point_list, radius=radius, max_dim=max_dim)
        f_values = face_vector(faces, max_dim=max_dim)
        h_values = h_vector(faces, max_dim=max_dim)
        for index, value in enumerate(f_values):
            values.append(value)
            labels.append(f"r={radius}:f_{index - 1}")
        for index, value in enumerate(h_values):
            values.append(value)
            labels.append(f"r={radius}:h_{index}")
    return VectorFeatures(values=tuple(values), labels=tuple(labels))


def facets(faces: Iterable[Iterable[int]]) -> set[Face]:
    closed = _close_faces(faces)
    return {
        face
        for face in closed
        if not any(set(face) < set(other) for other in closed)
    }


def facet_barcodes(
    filtration: Iterable[tuple[float, Iterable[Iterable[int]]]]
) -> dict[Face, tuple[float, float | None]]:
    active: dict[Face, float] = {}
    closed: dict[Face, tuple[float, float | None]] = {}

    for radius, faces_at_radius in filtration:
        current = facets(faces_at_radius)
        for face, birth in list(active.items()):
            if face not in current:
                closed[face] = (birth, radius)
                del active[face]
        for face in current:
            if face not in active and face not in closed:
                active[face] = radius

    for face, birth in active.items():
        closed[face] = (birth, None)
    return dict(sorted(closed.items()))
