from psrt_bearing.invariants import (
    graded_betti_numbers,
    macaulay2_betti_table,
    persistent_graded_betti_numbers,
    persistent_graded_betti_pair_features,
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


def test_paper_example_2_2_face_list_has_expected_hochster_values():
    faces = {
        (0,),
        (1,),
        (2,),
        (3,),
        (4,),
        (5,),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (2, 3),
        (2, 5),
        (0, 1, 2),
        (0, 1, 3),
        (0, 2, 3),
        (1, 2, 3),
    }

    betti = graded_betti_numbers(faces)

    assert betti == {
        (0, 0): 1,
        (1, 2): 5,
        (1, 3): 2,
        (1, 4): 1,
        (2, 3): 6,
        (2, 4): 6,
        (2, 5): 2,
        (3, 4): 2,
        (3, 5): 6,
        (3, 6): 1,
        (4, 6): 2,
    }


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


def test_persistent_graded_betti_numbers_use_inclusion_induced_rank():
    birth_faces = {
        (0,),
        (1,),
        (2,),
        (0, 1),
        (0, 2),
        (1, 2),
    }
    death_faces = birth_faces | {(0, 1, 2)}

    table = persistent_graded_betti_numbers(
        birth_faces,
        death_faces,
        max_subset_card=3,
    )

    assert graded_betti_numbers(birth_faces)[(1, 3)] == 1
    assert table.get((1, 3), 0) == 0


def test_persistent_graded_betti_numbers_compute_h0_image_rank_for_changing_vertices():
    birth_faces = {(0,), (1,)}
    death_faces = {(0,), (1,), (2,)}

    table = persistent_graded_betti_numbers(
        birth_faces,
        death_faces,
        max_subset_card=2,
    )

    assert table.get((1, 2), 0) == 1


def test_persistent_graded_betti_pair_features_flatten_birth_death_grid():
    points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

    features = persistent_graded_betti_pair_features(
        points,
        radii=[1.1, 1.5],
        betti_keys=[(2, 4)],
        max_dim=2,
        max_subset_card=4,
    )

    assert features.values == (1, 0, 0)
    assert features.labels == (
        "r=1.1->1.1:beta_2_4",
        "r=1.1->1.5:beta_2_4",
        "r=1.5->1.5:beta_2_4",
    )


def test_persistent_graded_betti_pair_features_reject_negative_homology_keys():
    try:
        persistent_graded_betti_pair_features(
            [(0.0,), (1.0,)],
            radii=[1.0],
            betti_keys=[(3, 1)],
        )
    except ValueError as error:
        assert "negative homology" in str(error)
    else:
        raise AssertionError("expected impossible Betti key to be rejected")


def test_macaulay2_betti_table_uses_row_as_internal_minus_homological_degree():
    table = macaulay2_betti_table({(0, 0): 1, (1, 2): 2, (2, 4): 1})

    assert table[0][0] == 1
    assert table[1][1] == 2
    assert table[2][2] == 1
