from psrt_bearing.classify import train_evaluate


def test_train_evaluate_reports_basic_metrics():
    result = train_evaluate(
        features=[(0.0,), (0.1,), (1.0,), (1.1,)],
        labels=["healthy", "healthy", "faulty", "faulty"],
        test_size=0.5,
        random_state=0,
    )

    assert set(result) == {"accuracy", "f1", "confusion_matrix"}
    assert result["accuracy"] >= 0.0
