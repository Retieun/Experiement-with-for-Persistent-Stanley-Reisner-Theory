from psrt_bearing.window import labeled_windows


def test_labeled_windows_slices_signal_with_stride_and_label():
    windows = labeled_windows([0, 1, 2, 3, 4], label="faulty", length=3, stride=2)

    assert windows == (((0.0, 1.0, 2.0), "faulty"), ((2.0, 3.0, 4.0), "faulty"))
