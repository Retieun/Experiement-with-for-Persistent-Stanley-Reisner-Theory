from psrt_bearing.psrt import (
    facet_barcodes,
    face_vector,
    h_vector,
    persistent_face_h_features,
)


def test_face_and_h_vectors_for_single_edge():
    faces = {(0,), (1,), (0, 1)}

    assert face_vector(faces) == (1, 2, 1)
    assert h_vector(faces) == (1, 0)


def test_persistent_face_h_features_flatten_radius_grid():
    points = [(0.0,), (1.0,)]

    features = persistent_face_h_features(points, radii=[0.5, 1.1], max_dim=1)

    assert features.values == (1, 2, 0, 1, 0, 1, 2, 1, 1, 0)
    assert features.labels == (
        "r=0.5:f_-1",
        "r=0.5:f_0",
        "r=0.5:f_1",
        "r=0.5:h_0",
        "r=0.5:h_1",
        "r=1.1:f_-1",
        "r=1.1:f_0",
        "r=1.1:f_1",
        "r=1.1:h_0",
        "r=1.1:h_1",
    )


def test_facet_barcodes_track_when_facets_stop_being_maximal():
    filtration = [
        (0.0, {(0,), (1,)}),
        (1.0, {(0,), (1,), (0, 1)}),
    ]

    assert facet_barcodes(filtration) == {
        (0,): (0.0, 1.0),
        (1,): (0.0, 1.0),
        (0, 1): (1.0, None),
    }
