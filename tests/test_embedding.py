from psrt_bearing.embedding import farthest_point_sample, plot_embedding, takens_embedding


def test_takens_embedding_builds_delay_coordinate_points():
    signal = [0.0, 1.0, 2.0, 3.0, 4.0]

    points = takens_embedding(signal, dimension=3, delay=1)

    assert points == ((0.0, 1.0, 2.0), (1.0, 2.0, 3.0), (2.0, 3.0, 4.0))


def test_farthest_point_sample_is_deterministic_and_keeps_extremes():
    points = [(0.0,), (1.0,), (2.0,), (10.0,)]

    sampled = farthest_point_sample(points, max_points=2)

    assert sampled == ((0.0,), (10.0,))


def test_plot_embedding_writes_image(tmp_path):
    output = tmp_path / "embedding.png"

    result = plot_embedding([(0.0, 0.0), (1.0, 1.0)], output, title="demo")

    assert result == output
    assert output.exists()
    assert output.stat().st_size > 0
