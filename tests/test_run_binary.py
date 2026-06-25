from scripts.run_binary import collect_balanced_windows, write_metrics_json


def test_collect_balanced_windows_spreads_samples_across_recordings():
    recordings = [
        ("normal_a", "healthy", ((0.0,), (1.0,), (2.0,))),
        ("normal_b", "healthy", ((3.0,),)),
        ("fault_a", "faulty", ((4.0,), (5.0,))),
        ("fault_b", "faulty", ((6.0,),)),
    ]

    samples = collect_balanced_windows(recordings, windows_per_class=2)

    assert [(label, source) for _window, label, source in samples] == [
        ("healthy", "normal_a"),
        ("healthy", "normal_b"),
        ("faulty", "fault_a"),
        ("faulty", "fault_b"),
    ]


def test_collect_balanced_windows_requires_two_recordings_per_class():
    recordings = [
        ("normal_a", "healthy", ((0.0,), (1.0,))),
        ("fault_a", "faulty", ((2.0,),)),
        ("fault_b", "faulty", ((3.0,),)),
    ]

    try:
        collect_balanced_windows(recordings, windows_per_class=2)
    except ValueError as error:
        assert "at least two recordings" in str(error)
    else:
        raise AssertionError("expected one-recording class to be rejected")


def test_write_metrics_json_records_parameters_and_metrics(tmp_path):
    output = tmp_path / "metrics.json"

    write_metrics_json(
        output,
        parameters={"method": "ph"},
        metrics={"accuracy": 1.0, "f1": 1.0},
        cv_metrics={"accuracy_mean": 0.5},
        cache_entries=3,
        total_subsets=12,
        feature_vectors=4,
    )

    text = output.read_text(encoding="utf-8")
    assert '"method": "ph"' in text
    assert '"total_subsets_enumerated": 12' in text
