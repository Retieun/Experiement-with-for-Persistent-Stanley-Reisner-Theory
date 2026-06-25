from __future__ import annotations

from hashlib import sha256
from math import dist
from typing import Iterable, Sequence

from psrt_bearing.baselines import betti_curve_features
from psrt_bearing.embedding import farthest_point_sample, takens_embedding
from psrt_bearing.invariants import (
    PersistentBettiFeatures,
    persistent_graded_betti_features,
    persistent_graded_betti_pair_features,
)
from psrt_bearing.psrt import persistent_face_h_features


class FeatureCache:
    def __init__(self) -> None:
        self._items: dict[str, PersistentBettiFeatures] = {}

    def get(self, key: str) -> PersistentBettiFeatures | None:
        return self._items.get(key)

    def set(self, key: str, value: PersistentBettiFeatures) -> PersistentBettiFeatures:
        self._items[key] = value
        return value

    def __len__(self) -> int:
        return len(self._items)


def featurize_window(
    window: Sequence[float],
    radii: Iterable[float] | None,
    betti_keys: Iterable[tuple[int, int]],
    embedding_dim: int = 3,
    delay: int = 1,
    max_points: int = 16,
    max_dim: int = 2,
    max_subset_card: int | None = None,
    cache: FeatureCache | None = None,
    method: str = "psrt-snapshot",
    radius_count: int = 6,
) -> PersistentBettiFeatures:
    if max_points > 24:
        raise ValueError("max_points must be <= 24 for Hochster subset enumeration")

    keys = tuple(betti_keys)
    embedded = takens_embedding(window, dimension=embedding_dim, delay=delay)
    sampled = farthest_point_sample(embedded, max_points=max_points)
    radius_grid = tuple(radii) if radii is not None else _diameter_radius_grid(sampled, radius_count)
    cache_key = _feature_key(
        window,
        radius_grid,
        keys,
        embedding_dim,
        delay,
        max_points,
        max_dim,
        max_subset_card,
        method,
        radius_count,
    )

    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    if method == "psrt-snapshot":
        features = persistent_graded_betti_features(
            sampled,
            radii=radius_grid,
            betti_keys=keys,
            max_dim=max_dim,
            max_subset_card=max_subset_card,
        )
    elif method == "psrt-pairs":
        features = persistent_graded_betti_pair_features(
            sampled,
            radii=radius_grid,
            betti_keys=keys,
            max_dim=max_dim,
            max_subset_card=max_subset_card,
        )
    elif method == "ph":
        baseline = betti_curve_features(sampled, radii=radius_grid, max_dim=max_dim)
        features = PersistentBettiFeatures(baseline.values, baseline.labels, 0)
    elif method == "fh":
        vector_features = persistent_face_h_features(sampled, radii=radius_grid, max_dim=max_dim)
        features = PersistentBettiFeatures(vector_features.values, vector_features.labels, 0)
    else:
        raise ValueError(f"unknown feature method: {method}")

    if cache is not None:
        return cache.set(cache_key, features)
    return features


def _diameter_radius_grid(points: Sequence[Sequence[float]], count: int) -> tuple[float, ...]:
    if count < 1:
        raise ValueError("radius_count must be at least 1")
    if not points:
        return (0.0,)
    diameter = max((dist(left, right) for left in points for right in points), default=0.0)
    if count == 1:
        return (diameter,)
    step = diameter / (count - 1)
    return tuple(round(step * index, 12) for index in range(count))


def _feature_key(
    window: Sequence[float],
    radii: tuple[float, ...],
    betti_keys: tuple[tuple[int, int], ...],
    embedding_dim: int,
    delay: int,
    max_points: int,
    max_dim: int,
    max_subset_card: int | None,
    method: str,
    radius_count: int,
) -> str:
    payload = repr(
        (
            tuple(float(value) for value in window),
            radii,
            betti_keys,
            embedding_dim,
            delay,
            max_points,
            max_dim,
            max_subset_card,
            method,
            radius_count,
        )
    )
    return sha256(payload.encode("utf-8")).hexdigest()
