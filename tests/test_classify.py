from psrt_bearing.classify import grouped_cross_validate, grouped_train_test_indices, train_evaluate


def test_train_evaluate_reports_basic_metrics():
    result = train_evaluate(
        features=[(0.0,), (0.1,), (1.0,), (1.1,)],
        labels=["healthy", "healthy", "faulty", "faulty"],
        test_size=0.5,
        random_state=0,
    )

    assert set(result) == {"accuracy", "f1", "confusion_matrix"}
    assert result["accuracy"] >= 0.0


def test_grouped_train_test_indices_keep_sources_together():
    train, test = grouped_train_test_indices(
        labels=["healthy", "healthy", "healthy", "healthy", "faulty", "faulty", "faulty", "faulty"],
        groups=["normal_a", "normal_a", "normal_b", "normal_b", "fault_a", "fault_a", "fault_b", "fault_b"],
        test_size=0.5,
        random_state=0,
    )

    source_by_index = {
        0: "normal_a",
        1: "normal_a",
        2: "normal_b",
        3: "normal_b",
        4: "fault_a",
        5: "fault_a",
        6: "fault_b",
        7: "fault_b",
    }
    train_groups = {source_by_index[index] for index in train}
    test_groups = {source_by_index[index] for index in test}
    assert train_groups.isdisjoint(test_groups)


def test_grouped_train_test_indices_require_two_sources_per_class():
    try:
        grouped_train_test_indices(
            labels=["healthy", "healthy", "faulty", "faulty"],
            groups=["normal_a", "normal_a", "fault_a", "fault_a"],
        )
    except ValueError as error:
        assert "two source groups" in str(error)
    else:
        raise AssertionError("expected grouped split to reject one source per class")


def test_grouped_cross_validate_reports_fold_metrics():
    result = grouped_cross_validate(
        features=[(0.0,), (0.1,), (0.2,), (1.0,), (1.1,), (1.2,)],
        labels=["healthy", "healthy", "healthy", "faulty", "faulty", "faulty"],
        groups=["n1", "n2", "n3", "f1", "f2", "f3"],
        n_splits=3,
        random_state=0,
    )

    assert set(result) == {"accuracy_mean", "f1_mean", "folds"}
    assert len(result["folds"]) == 3
