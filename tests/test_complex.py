from psrt_bearing.complex import vietoris_rips_complex


def test_vietoris_rips_complex_includes_faces_within_radius():
    points = [(0.0,), (1.0,), (3.0,)]

    faces = vietoris_rips_complex(points, radius=1.1, max_dim=2)

    assert (0,) in faces
    assert (1,) in faces
    assert (2,) in faces
    assert (0, 1) in faces
    assert (0, 2) not in faces
    assert (1, 2) not in faces
