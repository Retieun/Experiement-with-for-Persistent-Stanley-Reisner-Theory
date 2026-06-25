from psrt_bearing.baselines import betti_curve_features


def test_betti_curve_features_count_components_and_loops_across_radii():
    points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

    features = betti_curve_features(points, radii=[0.5, 1.1, 1.5], max_dim=2, homology_dims=[0, 1])

    assert features.values == (4, 0, 1, 1, 1, 0)
    assert features.labels == (
        "r=0.5:H_0",
        "r=0.5:H_1",
        "r=1.1:H_0",
        "r=1.1:H_1",
        "r=1.5:H_0",
        "r=1.5:H_1",
    )
