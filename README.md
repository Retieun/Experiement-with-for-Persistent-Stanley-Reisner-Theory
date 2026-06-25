# psrt-bearing

`psrt-bearing` is a methods-demonstration project for applying persistent
Stanley-Reisner theory to bearing-fault detection on the CWRU vibration
benchmark.

The pipeline is intended to:

- slice CWRU bearing vibration recordings into labeled windows,
- embed each 1D vibration window as a Takens delay-coordinate point cloud,
- build Vietoris-Rips complexes across a small filtration grid,
- compute persistent graded Betti-number features using Hochster's formula, and
- train a simple healthy-vs-faulty classifier on those features.

This project is not intended to claim state-of-the-art bearing-fault detection.
The goal is a clean, reproducible implementation of the paper's method on a
recognized real-world signal dataset.

## Current Status

Implemented:

- exact graded Betti numbers via Hochster's formula over GF(2),
- Macaulay2-style Betti-table layout, where rows are `j - i`,
- small Vietoris-Rips complex construction for capped point clouds,
- Takens delay embedding and deterministic farthest-point downsampling,
- window slicing, CWRU `.mat` loading helpers, feature caching, and
- a RandomForest binary classifier wrapper.

The arXiv HTML exposes the Example 2.2 Betti table, but not a machine-readable
face list for the pictured six-vertex complex. The test suite therefore covers
known exact fixtures, including a hollow square, until that face list is pinned
down from the paper figure or source.

## Usage

Install the package in editable mode:

```bash
python -m pip install -e ".[dev]"
```

Run tests:

```bash
python -m pytest
```

Run the binary CWRU demo against a directory of `.mat` files:

```bash
python scripts/run_binary.py path/to/cwru-mat-files
```

## Caveats

- Graded Betti features are computed on downsampled point clouds, so the
  representation is coarse by construction.
- Persistent homology may already separate the healthy and faulty classes. The
  graded-Betti pipeline is demonstrated here, not claimed superior.
- The Hochster-style computation enumerates vertex subsets, so point-cloud sizes
  must remain small.

## Reference

This implementation is based on:

Faisal Suwayyid and Guo-Wei Wei. "Persistent Stanley-Reisner Theory."
arXiv:2503.23482, 2025. <https://arxiv.org/abs/2503.23482>

The PDF referenced for this project is available at
<https://arxiv.org/pdf/2503.23482>.
