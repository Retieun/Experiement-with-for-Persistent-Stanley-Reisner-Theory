from __future__ import annotations

from math import dist
from pathlib import Path
from typing import Iterable, Sequence

Point = tuple[float, ...]


def takens_embedding(signal: Sequence[float], dimension: int = 3, delay: int = 1) -> tuple[Point, ...]:
    if dimension < 1:
        raise ValueError("dimension must be at least 1")
    if delay < 1:
        raise ValueError("delay must be at least 1")

    last_start = len(signal) - (dimension - 1) * delay
    if last_start <= 0:
        return ()

    return tuple(
        tuple(float(signal[start + axis * delay]) for axis in range(dimension))
        for start in range(last_start)
    )


def farthest_point_sample(points: Iterable[Sequence[float]], max_points: int) -> tuple[Point, ...]:
    point_list = [tuple(float(value) for value in point) for point in points]
    if max_points < 1:
        raise ValueError("max_points must be at least 1")
    if len(point_list) <= max_points:
        return tuple(point_list)

    selected = [point_list[0]]
    remaining = point_list[1:]

    while len(selected) < max_points:
        next_point = max(
            remaining,
            key=lambda point: min(dist(point, chosen) for chosen in selected),
        )
        selected.append(next_point)
        remaining.remove(next_point)

    return tuple(selected)


def estimate_tau(signal: Sequence[float], max_lag: int = 128) -> int:
    """Estimate delay as the first non-positive autocorrelation lag."""
    values = [float(value) for value in signal]
    if len(values) < 3:
        return 1

    mean = sum(values) / len(values)
    centered = [value - mean for value in values]
    denominator = sum(value * value for value in centered)
    if denominator == 0:
        return 1

    limit = min(max_lag, len(values) - 1)
    for lag in range(1, limit + 1):
        numerator = sum(centered[index] * centered[index + lag] for index in range(len(values) - lag))
        if numerator / denominator <= 0:
            return lag
    return 1


def plot_embedding(
    points: Iterable[Sequence[float]],
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    point_list = [tuple(float(value) for value in point) for point in points]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(5, 4))
    if point_list and len(point_list[0]) >= 3:
        axis = fig.add_subplot(111, projection="3d")
        axis.scatter(
            [point[0] for point in point_list],
            [point[1] for point in point_list],
            [point[2] for point in point_list],
            s=14,
        )
        axis.set_zlabel("x(t+2tau)")
    else:
        axis = fig.add_subplot(111)
        xs = [point[0] for point in point_list]
        ys = [point[1] if len(point) > 1 else index for index, point in enumerate(point_list)]
        axis.scatter(xs, ys, s=14)
        axis.set_ylabel("x(t+tau)" if point_list and len(point_list[0]) > 1 else "index")
    axis.set_xlabel("x(t)")
    if title:
        axis.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output
