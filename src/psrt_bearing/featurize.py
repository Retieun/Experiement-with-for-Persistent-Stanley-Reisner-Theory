from __future__ import annotations

from hashlib import sha256
from typing import Iterable, Sequence

from psrt_bearing.embedding import farthest_point_sample, takens_embedding
from psrt_bearing.invariants import PersistentBettiFeatures, persistent_graded_betti_features


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
    radii: Iterable[float],
    betti_keys: Iterable[tuple[int, int]],
    embedding_dim: int = 3,
    delay: int = 1,
    max_points: int = 16,
    max_dim: int = 2,
    max_subset_card: int | None = None,
    cache: FeatureCache | None = None,
) -> PersistentBettiFeatures:
    if max_points > 24:
        raise ValueError("max_points must be <= 24 for Hochster subset enumeration")

    radius_grid = tuple(radii)
    keys = tuple(betti_keys)
    cache_key = _feature_key(window, radius_grid, keys, embedding_dim, delay, max_points, max_dim, max_subset_card)

    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    embedded = takens_embedding(window, dimension=embedding_dim, delay=delay)
    sampled = farthest_point_sample(embedded, max_points=max_points)
    features = persistent_graded_betti_features(
        sampled,
        radii=radius_grid,
        betti_keys=keys,
        max_dim=max_dim,
        max_subset_card=max_subset_card,
    )
    if cache is not None:
        return cache.set(cache_key, features)
    return features


def _feature_key(
    window: Sequence[float],
    radii: tuple[float, ...],
    betti_keys: tuple[tuple[int, int], ...],
    embedding_dim: int,
    delay: int,
    max_points: int,
    max_dim: int,
    max_subset_card: int | None,
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
        )
    )
    return sha256(payload.encode("utf-8")).hexdigest()
