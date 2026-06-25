from psrt_bearing.featurize import FeatureCache, featurize_window


def test_featurize_window_returns_persistent_betti_values_and_uses_cache():
    cache = FeatureCache()

    first = featurize_window(
        [0.0, 1.0, 0.0],
        radii=[0.5, 2.0],
        betti_keys=[(1, 2)],
        embedding_dim=1,
        delay=1,
        max_points=3,
        max_dim=1,
        cache=cache,
    )
    second = featurize_window(
        [0.0, 1.0, 0.0],
        radii=[0.5, 2.0],
        betti_keys=[(1, 2)],
        embedding_dim=1,
        delay=1,
        max_points=3,
        max_dim=1,
        cache=cache,
    )

    assert first.values == (2, 0)
    assert second is first
    assert len(cache) == 1


def test_featurize_window_refuses_unsafe_point_cap():
    try:
        featurize_window([0.0, 1.0, 2.0], radii=[1.0], betti_keys=[(1, 2)], max_points=25)
    except ValueError as error:
        assert "max_points" in str(error)
    else:
        raise AssertionError("expected unsafe max_points to be refused")


def test_featurize_window_supports_true_persistent_pairs_and_diameter_radii():
    result = featurize_window(
        [0.0, 1.0, 2.0, 1.0],
        radii=None,
        betti_keys=[(1, 2)],
        embedding_dim=1,
        delay=1,
        max_points=4,
        max_dim=1,
        method="psrt-pairs",
        radius_count=3,
    )

    assert len(result.values) == 6
    assert result.labels[0].startswith("r=0.0->0.0")


def test_featurize_window_supports_ph_baseline():
    result = featurize_window(
        [0.0, 1.0],
        radii=[0.5, 1.1],
        betti_keys=[(1, 2)],
        embedding_dim=1,
        delay=1,
        max_points=2,
        max_dim=1,
        method="ph",
    )

    assert result.values == (2, 0, 1, 0)
    assert result.labels == ("r=0.5:H_0", "r=0.5:H_1", "r=1.1:H_0", "r=1.1:H_1")
