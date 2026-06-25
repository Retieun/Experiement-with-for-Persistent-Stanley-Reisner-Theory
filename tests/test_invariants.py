from psrt_bearing.invariants import (
    graded_betti_numbers,
    macaulay2_betti_table,
    persistent_graded_betti_features,
)


def test_hollow_square_has_expected_betti_numbers():
    complex_faces = {
        (0,),
        (1,),
        (2,),
        (3,),
        (0, 1),
        (1, 2),
        (2, 3),
        (0, 3),
    }

    betti = graded_betti_numbers(complex_faces, max_subset_card=4)

    assert betti[(0, 0)] == 1
    assert betti[(1, 2)] == 2
    assert betti[(2, 4)] == 1


def test_persistent_graded_betti_features_flatten_radius_grid():
    points = [(0.0,), (1.0,)]

    features = persistent_graded_betti_features(
        points,
        radii=[0.5, 1.1],
        betti_keys=[(1, 2)],
        max_dim=1,
        max_subset_card=2,
    )

    assert features.values == (1, 0)
    assert features.labels == ("r=0.5:beta_1_2", "r=1.1:beta_1_2")
    assert features.subsets_enumerated == 6


def test_macaulay2_betti_table_uses_row_as_internal_minus_homological_degree():
    table = macaulay2_betti_table({(0, 0): 1, (1, 2): 2, (2, 4): 1})

    assert table[0][0] == 1
    assert table[1][1] == 2
    assert table[2][2] == 1
