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


def persistent_graded_betti_numbers(
    birth_faces: Iterable[Iterable[int]],
    death_faces: Iterable[Iterable[int]],
    max_subset_card: int | None = None,
) -> BettiTable:
    """Compute two-scale persistent graded Betti numbers via induced homology ranks."""
    birth_complex = _close_faces(birth_faces)
    death_complex = _close_faces(death_faces)
    vertices = sorted({vertex for face in death_complex for vertex in face})
    subset_cap = len(vertices) if max_subset_card is None else min(max_subset_card, len(vertices))

    table: defaultdict[tuple[int, int], int] = defaultdict(int)
    table[(0, 0)] = 1

    for cardinality in range(1, subset_cap + 1):
        for subset in combinations(vertices, cardinality):
            birth_induced = {face for face in birth_complex if set(face).issubset(subset)}
            death_induced = {face for face in death_complex if set(face).issubset(subset)}
            for dimension in range(0, cardinality):
                rank = _persistent_reduced_homology_rank(birth_induced, death_induced, dimension)
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


def persistent_graded_betti_pair_features(
    points: Iterable[Point],
    radii: Iterable[float],
    betti_keys: Iterable[tuple[int, int]],
    max_dim: int = 2,
    max_subset_card: int | None = None,
) -> PersistentBettiFeatures:
    point_list = [tuple(point) for point in points]
    radius_grid = tuple(radii)
    keys = tuple(betti_keys)
    requested = _requested_dimensions_by_cardinality(keys)
    complexes = {
        radius: vietoris_rips_complex(point_list, radius=radius, max_dim=max_dim)
        for radius in radius_grid
    }
    values: list[int] = []
    labels: list[str] = []

    for birth_index, birth_radius in enumerate(radius_grid):
        for death_radius in radius_grid[birth_index:]:
            table = _persistent_graded_betti_numbers_for_keys(
                complexes[birth_radius],
                complexes[death_radius],
                requested,
            )
            for key in keys:
                values.append(table.get(key, 0))
                labels.append(f"r={birth_radius}->{death_radius}:beta_{key[0]}_{key[1]}")

    subset_cap = len(point_list) if max_subset_card is None else min(max_subset_card, len(point_list))
    subsets_per_pair = sum(_n_choose_k(len(point_list), size) for size in range(1, subset_cap + 1))
    pair_count = len(radius_grid) * (len(radius_grid) + 1) // 2
    return PersistentBettiFeatures(
        values=tuple(values),
        labels=tuple(labels),
        subsets_enumerated=subsets_per_pair * pair_count,
    )


def _requested_dimensions_by_cardinality(
    betti_keys: tuple[tuple[int, int], ...]
) -> dict[int, set[int]]:
    requested: dict[int, set[int]] = defaultdict(set)
    for homological_degree, cardinality in betti_keys:
        dimension = cardinality - homological_degree - 1
        if dimension < 0:
            raise ValueError(f"Betti key {(homological_degree, cardinality)!r} implies negative homology dimension")
        requested[cardinality].add(dimension)
    return requested


def _persistent_graded_betti_numbers_for_keys(
    birth_faces: Iterable[Iterable[int]],
    death_faces: Iterable[Iterable[int]],
    requested: dict[int, set[int]],
) -> BettiTable:
    birth_complex = _close_faces(birth_faces)
    death_complex = _close_faces(death_faces)
    vertices = sorted({vertex for face in death_complex for vertex in face})
    table: defaultdict[tuple[int, int], int] = defaultdict(int)

    for cardinality, dimensions in requested.items():
        if cardinality > len(vertices):
            continue
        for subset in combinations(vertices, cardinality):
            birth_induced = {face for face in birth_complex if set(face).issubset(subset)}
            death_induced = {face for face in death_complex if set(face).issubset(subset)}
            for dimension in dimensions:
                rank = _persistent_reduced_homology_rank(birth_induced, death_induced, dimension)
                if rank == 0:
                    continue
                homological_degree = cardinality - dimension - 1
                table[(homological_degree, cardinality)] += rank
    return dict(table)


def macaulay2_betti_table(table: BettiTable) -> list[list[int]]:
    if not table:
        return []

    max_column = max(i for i, _j in table)
    max_row = max(j - i for i, j in table)
    rows = [[0 for _ in range(max_column + 1)] for _ in range(max_row + 1)]
    for (homological_degree, internal_degree), value in table.items():
        rows[internal_degree - homological_degree][homological_degree] = value
    return rows


def _persistent_reduced_homology_rank(
    birth_faces: set[Face], death_faces: set[Face], dimension: int
) -> int:
    if not birth_faces or not death_faces:
        return 0
    if dimension == 0:
        return _persistent_reduced_h0_rank(birth_faces, death_faces)

    birth_by_dim = _faces_by_dimension(birth_faces)
    death_by_dim = _faces_by_dimension(death_faces)
    birth_d_faces = sorted(birth_by_dim.get(dimension, []))
    death_d_faces = sorted(death_by_dim.get(dimension, []))
    if not birth_d_faces or not death_d_faces:
        return 0

    birth_boundary = _boundary_columns(
        birth_d_faces,
        sorted(birth_by_dim.get(dimension - 1, [])),
    )
    source_cycle_basis = _kernel_basis(birth_boundary, len(birth_d_faces))
    if not source_cycle_basis:
        return 0

    target_index = {face: index for index, face in enumerate(death_d_faces)}
    source_index = {face: index for index, face in enumerate(birth_d_faces)}
    mapped_cycles = [
        _map_vector_to_target(vector, birth_d_faces, source_index, target_index)
        for vector in source_cycle_basis
    ]
    target_boundaries = _boundary_columns(
        sorted(death_by_dim.get(dimension + 1, [])),
        death_d_faces,
    )

    cycle_rank = _gf2_rank(mapped_cycles)
    boundary_rank = _gf2_rank(target_boundaries)
    combined_rank = _gf2_rank([*mapped_cycles, *target_boundaries])
    intersection_rank = cycle_rank + boundary_rank - combined_rank
    return cycle_rank - intersection_rank


def _persistent_reduced_h0_rank(birth_faces: set[Face], death_faces: set[Face]) -> int:
    birth_vertices = sorted({face[0] for face in birth_faces if len(face) == 1})
    death_vertices = sorted({face[0] for face in death_faces if len(face) == 1})
    if not birth_vertices or not death_vertices:
        return 0

    birth_components = _component_labels(birth_faces, birth_vertices)
    death_components = _component_labels(death_faces, death_vertices)
    mapped_components = {
        death_components[vertex]
        for vertex in birth_vertices
        if vertex in death_components
    }
    if not mapped_components:
        return 0
    return len(mapped_components) - 1


def _component_labels(faces: set[Face], vertices: list[int]) -> dict[int, int]:
    parent = {vertex: vertex for vertex in vertices}

    def find(vertex: int) -> int:
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    vertex_set = set(vertices)
    for face in faces:
        if len(face) == 2 and face[0] in vertex_set and face[1] in vertex_set:
            union(face[0], face[1])

    roots = {vertex: find(vertex) for vertex in vertices}
    root_ids = {root: index for index, root in enumerate(sorted(set(roots.values())))}
    return {vertex: root_ids[root] for vertex, root in roots.items()}


def _faces_by_dimension(faces: set[Face]) -> dict[int, list[Face]]:
    grouped: defaultdict[int, list[Face]] = defaultdict(list)
    for face in faces:
        grouped[len(face) - 1].append(face)
    return dict(grouped)


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


def _boundary_columns(domain_faces: list[Face], codomain_faces: list[Face]) -> list[int]:
    if not domain_faces or not codomain_faces:
        return []

    row_index = {face: index for index, face in enumerate(sorted(codomain_faces))}
    columns: list[int] = []
    for face in sorted(domain_faces):
        column = 0
        for omitted in range(len(face)):
            boundary_face = face[:omitted] + face[omitted + 1 :]
            column ^= 1 << row_index[boundary_face]
        columns.append(column)
    return columns


def _kernel_basis(columns: list[int], domain_dimension: int) -> list[int]:
    rows = [0 for _ in range(max((column.bit_length() for column in columns), default=0))]
    for column_index, column in enumerate(columns):
        vector = column
        while vector:
            row = (vector & -vector).bit_length() - 1
            rows[row] |= 1 << column_index
            vector &= vector - 1

    pivot_rows: dict[int, int] = {}
    for raw_row in rows:
        row = raw_row
        while row:
            pivot = (row & -row).bit_length() - 1
            if pivot not in pivot_rows:
                pivot_rows[pivot] = row
                break
            row ^= pivot_rows[pivot]

    for pivot in sorted(pivot_rows, reverse=True):
        row = pivot_rows[pivot]
        for other_pivot, other_row in list(pivot_rows.items()):
            if other_pivot != pivot and ((other_row >> pivot) & 1):
                pivot_rows[other_pivot] = other_row ^ row

    pivot_columns = set(pivot_rows)
    basis: list[int] = []
    for free_column in range(domain_dimension):
        if free_column in pivot_columns:
            continue
        vector = 1 << free_column
        for pivot, row in pivot_rows.items():
            if (row >> free_column) & 1:
                vector |= 1 << pivot
        basis.append(vector)
    return basis


def _map_vector_to_target(
    vector: int,
    source_faces: list[Face],
    source_index: dict[Face, int],
    target_index: dict[Face, int],
) -> int:
    mapped = 0
    for face in source_faces:
        if (vector >> source_index[face]) & 1:
            mapped |= 1 << target_index[face]
    return mapped


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
