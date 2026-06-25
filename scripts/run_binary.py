from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from psrt_bearing.classify import train_evaluate
from psrt_bearing.data import load_cwru_mat
from psrt_bearing.embedding import estimate_tau
from psrt_bearing.featurize import FeatureCache, featurize_window
from psrt_bearing.window import labeled_windows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PSRT binary CWRU demo.")
    parser.add_argument("data_dir", type=Path, help="Directory containing CWRU .mat files.")
    parser.add_argument("--window-length", type=int, default=4096)
    parser.add_argument("--windows-per-class", type=int, default=20)
    parser.add_argument("--embedding-dim", type=int, default=3)
    parser.add_argument("--max-points", type=int, default=16)
    parser.add_argument("--max-dim", type=int, default=2)
    parser.add_argument("--max-subset-card", type=int, default=4)
    args = parser.parse_args()

    windows_by_label: dict[str, list[tuple[float, ...]]] = {"healthy": [], "faulty": []}
    for mat_path in sorted(args.data_dir.glob("*.mat")):
        signal, label = load_cwru_mat(mat_path)
        if label not in windows_by_label:
            continue
        remaining = args.windows_per_class - len(windows_by_label[label])
        if remaining <= 0:
            continue
        sliced = labeled_windows(signal, label=label, length=args.window_length)
        windows_by_label[label].extend(window for window, _ in sliced[:remaining])

    labeled = [
        (window, label)
        for label, windows in windows_by_label.items()
        for window in windows[: args.windows_per_class]
    ]
    if len({label for _window, label in labeled}) < 2:
        raise SystemExit("Need at least one healthy and one faulty window.")

    cache = FeatureCache()
    features: list[tuple[int, ...]] = []
    labels: list[str] = []
    for window, label in labeled:
        tau = estimate_tau(window)
        result = featurize_window(
            window,
            radii=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            betti_keys=((1, 2), (2, 3), (2, 4)),
            embedding_dim=args.embedding_dim,
            delay=tau,
            max_points=args.max_points,
            max_dim=args.max_dim,
            max_subset_card=args.max_subset_card,
            cache=cache,
        )
        features.append(result.values)
        labels.append(label)
        print(f"{label}: enumerated {result.subsets_enumerated} subsets")

    metrics = train_evaluate(features, labels)
    print(f"accuracy: {metrics['accuracy']:.3f}")
    print(f"f1: {metrics['f1']:.3f}")
    print(f"confusion_matrix: {metrics['confusion_matrix']}")


if __name__ == "__main__":
    main()
