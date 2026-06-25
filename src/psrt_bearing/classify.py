from __future__ import annotations

from collections import defaultdict
from random import Random
from typing import Sequence


def train_evaluate(
    features: Sequence[Sequence[float]],
    labels: Sequence[str],
    test_size: float = 0.25,
    random_state: int = 42,
    groups: Sequence[str] | None = None,
) -> dict[str, object]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sklearn.model_selection import train_test_split

    feature_list = list(features)
    label_list = list(labels)
    if groups is None:
        x_train, x_test, y_train, y_test = train_test_split(
            feature_list,
            label_list,
            test_size=test_size,
            random_state=random_state,
            stratify=label_list,
        )
    else:
        train_indices, test_indices = grouped_train_test_indices(label_list, groups, test_size, random_state)
        x_train = [feature_list[index] for index in train_indices]
        x_test = [feature_list[index] for index in test_indices]
        y_train = [label_list[index] for index in train_indices]
        y_test = [label_list[index] for index in test_indices]

    classifier = RandomForestClassifier(n_estimators=100, random_state=random_state)
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions, pos_label="faulty")),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=["healthy", "faulty"]).tolist(),
    }


def grouped_train_test_indices(
    labels: Sequence[str],
    groups: Sequence[str],
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[list[int], list[int]]:
    if len(labels) != len(groups):
        raise ValueError("labels and groups must have the same length")

    grouped_indices: dict[str, list[int]] = defaultdict(list)
    grouped_labels: dict[str, str] = {}
    for index, (label, group) in enumerate(zip(labels, groups)):
        grouped_indices[group].append(index)
        grouped_labels.setdefault(group, label)

    groups_by_label: dict[str, list[str]] = defaultdict(list)
    for group, label in grouped_labels.items():
        groups_by_label[label].append(group)

    rng = Random(random_state)
    test_groups: set[str] = set()
    train_groups: set[str] = set()
    for label, label_groups in groups_by_label.items():
        if len(label_groups) < 2:
            raise ValueError(f"grouped split needs at least two source groups for class {label!r}")
        shuffled = list(label_groups)
        rng.shuffle(shuffled)
        test_count = max(1, round(len(shuffled) * test_size))
        if test_count >= len(shuffled) and len(shuffled) > 1:
            test_count = len(shuffled) - 1
        test_groups.update(shuffled[:test_count])
        train_groups.update(shuffled[test_count:])
        if not shuffled[test_count:]:
            train_groups.update(shuffled[:test_count])

    train = [index for group in sorted(train_groups) for index in grouped_indices[group]]
    test = [index for group in sorted(test_groups) for index in grouped_indices[group]]
    return train, test
