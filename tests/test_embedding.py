from psrt_bearing.embedding import farthest_point_sample, takens_embedding


def test_takens_embedding_builds_delay_coordinate_points():
    signal = [0.0, 1.0, 2.0, 3.0, 4.0]

    points = takens_embedding(signal, dimension=3, delay=1)

    assert points == ((0.0, 1.0, 2.0), (1.0, 2.0, 3.0), (2.0, 3.0, 4.0))


def test_farthest_point_sample_is_deterministic_and_keeps_extremes():
    points = [(0.0,), (1.0,), (2.0,), (10.0,)]

    sampled = farthest_point_sample(points, max_points=2)

    assert sampled == ((0.0,), (10.0,))
