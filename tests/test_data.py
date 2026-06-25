from psrt_bearing.data import extract_drive_end_signal, infer_cwru_label


def test_infer_cwru_label_from_filename():
    assert infer_cwru_label("normal_0.mat") == "healthy"
    assert infer_cwru_label("IR007_0.mat") == "faulty"
    assert infer_cwru_label("B014_0.mat") == "faulty"
    assert infer_cwru_label("OR021_0.mat") == "faulty"


def test_extract_drive_end_signal_prefers_de_channel():
    mat = {"X098_DE_time": [[1.0], [2.0], [3.0]], "X098_FE_time": [[9.0]]}

    assert extract_drive_end_signal(mat) == (1.0, 2.0, 3.0)
