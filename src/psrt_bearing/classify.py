from __future__ import annotations

from typing import Sequence


def train_evaluate(
    features: Sequence[Sequence[float]],
    labels: Sequence[str],
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict[str, object]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(
        list(features),
        list(labels),
        test_size=test_size,
        random_state=random_state,
        stratify=list(labels),
    )
    classifier = RandomForestClassifier(n_estimators=100, random_state=random_state)
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions, pos_label="faulty")),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=["healthy", "faulty"]).tolist(),
    }
