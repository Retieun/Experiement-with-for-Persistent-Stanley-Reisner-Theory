from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from psrt_bearing.classify import grouped_cross_validate, train_evaluate
from psrt_bearing.data import load_cwru_mat, recording_id
from psrt_bearing.embedding import estimate_tau
from psrt_bearing.featurize import DiskFeatureCache, FeatureCache, featurize_window
from psrt_bearing.window import labeled_windows

Window = tuple[float, ...]
RecordingWindows = tuple[str, str, tuple[Window, ...]]
LabeledSample = tuple[Window, str, str]


def collect_balanced_windows(
    recordings: list[RecordingWindows],
    windows_per_class: int,
) -> list[LabeledSample]:
    by_label: dict[str, list[tuple[str, tuple[Window, ...]]]] = {"healthy": [], "faulty": []}
    for source, label, windows in recordings:
        if label in by_label and windows:
            by_label[label].append((source, windows))

    samples: list[LabeledSample] = []
    for label, source_windows in by_label.items():
        if len(source_windows) < 2:
            raise ValueError(f"class {label!r} needs at least two recordings for grouped splitting")

        selected: list[LabeledSample] = []
        offsets = {source: 0 for source, _windows in source_windows}
        while len(selected) < windows_per_class:
            progressed = False
            for source, windows in source_windows:
                offset = offsets[source]
                if offset < len(windows):
                    selected.append((windows[offset], label, source))
                    offsets[source] = offset + 1
                    progressed = True
                    if len(selected) == windows_per_class:
                        break
            if not progressed:
                break
        samples.extend(selected)
    return samples


def write_metrics_json(
    output_path: Path,
    parameters: dict[str, Any],
    metrics: dict[str, Any],
    cv_metrics: dict[str, Any] | None,
    cache_entries: int,
    total_subsets: int,
    feature_vectors: int,
) -> Path:
    payload = {
        "parameters": parameters,
        "metrics": metrics,
        "cv_metrics": cv_metrics,
        "cache_entries": cache_entries,
        "total_subsets_enumerated": total_subsets,
        "feature_vectors": feature_vectors,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PSRT binary CWRU demo.")
    parser.add_argument("data_dir", type=Path, help="Directory containing CWRU .mat files.")
    parser.add_argument("--window-length", type=int, default=4096)
    parser.add_argument("--windows-per-class", type=int, default=20)
    parser.add_argument("--embedding-dim", type=int, default=3)
    parser.add_argument("--max-points", type=int, default=16)
    parser.add_argument("--max-dim", type=int, default=2)
    parser.add_argument("--max-subset-card", type=int, default=4)
    parser.add_argument(
        "--method",
        choices=("psrt-snapshot", "psrt-pairs", "ph", "fh"),
        default="psrt-pairs",
    )
    parser.add_argument("--radius-count", type=int, default=6)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--cv-folds", type=int, default=0)
    parser.add_argument("--metrics-json", type=Path)
    args = parser.parse_args()

    recordings: list[RecordingWindows] = []
    for mat_path in sorted(args.data_dir.glob("*.mat")):
        signal, label = load_cwru_mat(mat_path)
        if label not in {"healthy", "faulty"}:
            continue
        sliced = labeled_windows(signal, label=label, length=args.window_length)
        source = recording_id(mat_path)
        recordings.append((source, label, tuple(window for window, _ in sliced)))

    labeled = collect_balanced_windows(recordings, windows_per_class=args.windows_per_class)
    if len({label for _window, label, _source in labeled}) < 2:
        raise SystemExit("Need at least one healthy and one faulty window.")

    cache = DiskFeatureCache(args.cache_dir) if args.cache_dir else FeatureCache()
    features: list[tuple[int, ...]] = []
    labels: list[str] = []
    groups: list[str] = []
    total_subsets = 0
    for window, label, source in labeled:
        tau = estimate_tau(window)
        result = featurize_window(
            window,
            radii=None,
            betti_keys=((1, 2), (2, 3), (2, 4)),
            embedding_dim=args.embedding_dim,
            delay=tau,
            max_points=args.max_points,
            max_dim=args.max_dim,
            max_subset_card=args.max_subset_card,
            cache=cache,
            method=args.method,
            radius_count=args.radius_count,
            normalize=args.normalize,
        )
        features.append(result.values)
        labels.append(label)
        groups.append(source)
        total_subsets += result.subsets_enumerated
        print(f"{source} {label}: enumerated {result.subsets_enumerated} subsets")

    metrics = train_evaluate(features, labels, groups=groups)
    print(f"feature_method: {args.method}")
    print(f"normalized: {args.normalize}")
    print(f"feature_vectors: {len(features)}")
    print(f"cache_entries: {len(cache)}")
    print(f"total_subsets_enumerated: {total_subsets}")
    print(f"accuracy: {metrics['accuracy']:.3f}")
    print(f"f1: {metrics['f1']:.3f}")
    print(f"confusion_matrix: {metrics['confusion_matrix']}")
    cv = None
    if args.cv_folds:
        cv = grouped_cross_validate(features, labels, groups, n_splits=args.cv_folds)
        print(f"cv_accuracy_mean: {cv['accuracy_mean']:.3f}")
        print(f"cv_f1_mean: {cv['f1_mean']:.3f}")
    if args.metrics_json:
        write_metrics_json(
            args.metrics_json,
            parameters={
                "data_dir": str(args.data_dir),
                "window_length": args.window_length,
                "windows_per_class": args.windows_per_class,
                "embedding_dim": args.embedding_dim,
                "max_points": args.max_points,
                "max_dim": args.max_dim,
                "max_subset_card": args.max_subset_card,
                "method": args.method,
                "radius_count": args.radius_count,
                "normalize": args.normalize,
                "cv_folds": args.cv_folds,
            },
            metrics=metrics,
            cv_metrics=cv,
            cache_entries=len(cache),
            total_subsets=total_subsets,
            feature_vectors=len(features),
        )


if __name__ == "__main__":
    main()
