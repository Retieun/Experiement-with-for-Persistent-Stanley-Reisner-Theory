from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

from psrt_bearing.complex import Point, vietoris_rips_complex

Face = tuple[int, ...]
BettiTable = dict[tuple[int, int], int]


@dataclass(frozen=True)
class PersistentBettiFeatures:
    values: tuple[int, ...]
    labels: tuple[str, ...]
    subsets_enumerated: int


def graded_betti_numbers(
    faces: Iterable[Iterable[int]], max_subset_card: int | None = None
) -> BettiTable:
    """Compute graded Betti numbers by Hochster's formula over GF(2)."""
    complex_faces = _close_faces(faces)
    vertices = sorted({vertex for face in complex_faces for vertex in face})
    subset_cap = len(vertices) if max_subset_card is None else min(max_subset_card, len(vertices))

    table: defaultdict[tuple[int, int], int] = defaultdict(int)
    table[(0, 0)] = 1

    for cardinality in range(1, subset_cap + 1):
        for subset in combinations(vertices, cardinality):
            induced = {face for face in complex_faces if set(face).issubset(subset)}
            homology = _reduced_betti_by_dimension(induced)
            for dimension, rank in homology.items():
                if rank == 0:
                    continue
                homological_degree = cardinality - dimension - 1
                if homological_degree >= 0:
                    table[(homological_degree, cardinality)] += rank

    return dict(table)


def persistent_graded_betti_features(
    points: Iterable[Point],
    radii: Iterable[float],
    betti_keys: Iterable[tuple[int, int]],
    max_dim: int = 2,
    max_subset_card: int | None = None,
) -> PersistentBettiFeatures:
    point_list = [tuple(point) for point in points]
    radius_grid = tuple(radii)
    keys = tuple(betti_keys)
    values: list[int] = []
    labels: list[str] = []

    for radius in radius_grid:
        faces = vietoris_rips_complex(point_list, radius=radius, max_dim=max_dim)
        table = graded_betti_numbers(faces, max_subset_card=max_subset_card)
        for key in keys:
            values.append(table.get(key, 0))
            labels.append(f"r={radius}:beta_{key[0]}_{key[1]}")

    subset_cap = len(point_list) if max_subset_card is None else min(max_subset_card, len(point_list))
    subsets_per_radius = sum(_n_choose_k(len(point_list), size) for size in range(1, subset_cap + 1))
    return PersistentBettiFeatures(
        values=tuple(values),
        labels=tuple(labels),
        subsets_enumerated=subsets_per_radius * len(radius_grid),
    )


def macaulay2_betti_table(table: BettiTable) -> list[list[int]]:
    if not table:
        return []

    max_column = max(i for i, _j in table)
    max_row = max(j - i for i, j in table)
    rows = [[0 for _ in range(max_column + 1)] for _ in range(max_row + 1)]
    for (homological_degree, internal_degree), value in table.items():
        rows[internal_degree - homological_degree][homological_degree] = value
    return rows


def _close_faces(faces: Iterable[Iterable[int]]) -> set[Face]:
    closed: set[Face] = set()
    for raw_face in faces:
        face = tuple(sorted(raw_face))
        if not face:
            continue
        for size in range(1, len(face) + 1):
            closed.update(tuple(combo) for combo in combinations(face, size))
    return closed


def _reduced_betti_by_dimension(faces: set[Face]) -> dict[int, int]:
    if not faces:
        return {}

    faces_by_dim: defaultdict[int, list[Face]] = defaultdict(list)
    for face in faces:
        faces_by_dim[len(face) - 1].append(face)

    max_dim = max(faces_by_dim)
    ranks = {dim: _boundary_rank(faces_by_dim[dim], faces_by_dim[dim - 1]) for dim in range(1, max_dim + 1)}

    betti: dict[int, int] = {}
    for dim in range(0, max_dim + 1):
        chain_dim = len(faces_by_dim[dim])
        boundary_rank = ranks.get(dim, 0)
        next_boundary_rank = ranks.get(dim + 1, 0)
        value = chain_dim - boundary_rank - next_boundary_rank
        if dim == 0:
            value -= 1
        if value:
            betti[dim] = value
    return betti


def _boundary_rank(domain_faces: list[Face], codomain_faces: list[Face]) -> int:
    if not domain_faces or not codomain_faces:
        return 0

    row_index = {face: index for index, face in enumerate(sorted(codomain_faces))}
    columns: list[int] = []
    for face in sorted(domain_faces):
        column = 0
        for omitted in range(len(face)):
            boundary_face = face[:omitted] + face[omitted + 1 :]
            column ^= 1 << row_index[boundary_face]
        if column:
            columns.append(column)

    return _gf2_rank(columns)


def _gf2_rank(columns: Iterable[int]) -> int:
    pivots: dict[int, int] = {}
    for column in columns:
        vector = column
        while vector:
            pivot = vector.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = vector
                break
            vector ^= pivots[pivot]
    return len(pivots)


def _n_choose_k(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    numerator = 1
    denominator = 1
    for offset in range(1, k + 1):
        numerator *= n - offset + 1
        denominator *= offset
    return numerator // denominator
