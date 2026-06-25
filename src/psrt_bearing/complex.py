from __future__ import annotations

from itertools import combinations
from math import dist
from typing import Iterable, Sequence

Point = Sequence[float]
Face = tuple[int, ...]


def vietoris_rips_complex(points: Iterable[Point], radius: float, max_dim: int = 2) -> set[Face]:
    """Build a Vietoris-Rips complex up to ``max_dim`` for a small point cloud."""
    point_list = [tuple(point) for point in points]
    faces: set[Face] = {(index,) for index in range(len(point_list))}

    max_face_size = min(max_dim + 1, len(point_list))
    for size in range(2, max_face_size + 1):
        for candidate in combinations(range(len(point_list)), size):
            if _is_rips_face(candidate, point_list, radius):
                faces.add(candidate)

    return faces


def induced_subcomplex(faces: Iterable[Iterable[int]], vertices: Iterable[int]) -> set[Face]:
    vertex_set = set(vertices)
    return {tuple(sorted(face)) for face in faces if set(face).issubset(vertex_set)}


def _is_rips_face(candidate: Face, points: list[Point], radius: float) -> bool:
    return all(dist(points[left], points[right]) <= radius for left, right in combinations(candidate, 2))
