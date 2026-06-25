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
- true two-radius persistent graded Betti features using inclusion-induced
  homology ranks,
- Macaulay2-style Betti-table layout, where rows are `j - i`,
- persistent `f`/`h` vector features and facet barcode utilities,
- a persistent homology Betti-curve baseline,
- small Vietoris-Rips complex construction for capped point clouds,
- Takens delay embedding and deterministic farthest-point downsampling,
- optional per-window standardization before embedding,
- window slicing, CWRU `.mat` loading helpers, memory/disk feature caching, and
- a RandomForest binary classifier wrapper with source-recording grouped
  train/test splits and grouped cross-validation.

The test suite includes the face list from the paper source for Example 2.2.
The raw Hochster values are pinned directly; the paper's printed table appears
to use a layout that is not fully consistent with its surrounding text for the
later entries, so the fixture checks the mathematical `(i, j)` values.

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

By default the CLI uses the paper-faithful `psrt-pairs` feature method, which
computes persistent graded Betti ranks over all `(birth_radius, death_radius)`
pairs in a diameter-scaled radius grid. Other feature methods are available:

```bash
python scripts/run_binary.py path/to/cwru-mat-files --method psrt-snapshot
python scripts/run_binary.py path/to/cwru-mat-files --method ph
python scripts/run_binary.py path/to/cwru-mat-files --method fh
```

The CLI reports the number of feature vectors, cache entries, and total
Hochster subsets enumerated. Classification splits are grouped by source
recording so windows from the same `.mat` file do not appear in both train and
test sets. Use `--normalize` to standardize each vibration window before
embedding, `--cache-dir .cache/features` to persist expensive feature vectors,
and `--cv-folds 5` to also report grouped cross-validation means.

## Caveats

- Graded Betti features are computed on downsampled point clouds, so the
  representation is coarse by construction.
- Persistent homology may already separate the healthy and faulty classes. The
  `--method ph` baseline is included to check that directly.
- The Hochster-style computation enumerates vertex subsets, so point-cloud sizes
  must remain small.

## Reference

This implementation is based on:

Faisal Suwayyid and Guo-Wei Wei. "Persistent Stanley-Reisner Theory."
arXiv:2503.23482, 2025. <https://arxiv.org/abs/2503.23482>

The PDF referenced for this project is available at
<https://arxiv.org/pdf/2503.23482>.
