# psrt-bearing

`psrt-bearing` is a small methods project around persistent Stanley-Reisner
theory and bearing vibration data. The basic question is modest: can the
invariants from Suwayyid and Wei's PSRT paper be made into a working,
inspectable pipeline on a familiar fault-detection dataset?

The pipeline slices CWRU bearing recordings into windows, turns each 1D signal
window into a Takens delay point cloud, builds Vietoris-Rips complexes across a
short radius grid, and then computes graded Betti-style features for a simple
healthy-vs-faulty classifier.

This is not a benchmark-chasing project. A good raw-signal model will almost
certainly do better. The point here is to make the algebraic pipeline explicit
enough that the tradeoffs are visible.

## Current Status

The current implementation has the core pieces in place:

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

The test suite also pins the face list from Example 2.2 in the paper source.
For that fixture I check the raw Hochster `(i, j)` values directly. The printed
table in the paper seems to mix conventions in the later rows, so the test
sticks to the formula rather than the formatting.

## Usage

For local work I usually install it editable:

```bash
python -m pip install -e ".[dev]"
```

Then run the tests:

```bash
python -m pytest
```

To run the binary demo, point the script at a folder of CWRU `.mat` files:

```bash
python scripts/run_binary.py path/to/cwru-mat-files
```

By default the CLI uses `psrt-pairs`, which is the closest option here to the
paper's persistent graded-Betti definition. It computes ranks over all
`(birth_radius, death_radius)` pairs in a diameter-scaled radius grid.

Other feature sets are available for comparison:

```bash
python scripts/run_binary.py path/to/cwru-mat-files --method psrt-snapshot
python scripts/run_binary.py path/to/cwru-mat-files --method ph
python scripts/run_binary.py path/to/cwru-mat-files --method fh
```

The script reports feature counts, cache entries, and the number of Hochster
subsets enumerated. Splits are grouped by source recording, so windows from the
same `.mat` file do not land in both train and test.

Useful flags:

- `--normalize` standardizes each vibration window before embedding.
- `--cache-dir .cache/features` keeps expensive feature vectors on disk.
- `--cv-folds 5` adds grouped cross-validation means.

To save a run as JSON:

```bash
python scripts/run_binary.py path/to/cwru-mat-files --metrics-json artifacts/run.json
```

The notebook at `notebooks/01_demo.ipynb` is the more visual route. It loads a
few CWRU windows, plots one healthy and one faulty Takens embedding, featurizes
with PSRT pairs, and trains the grouped classifier.

## Mini CWRU Smoke Test

I tested the pipeline on a tiny official CWRU subset with two normal recordings
and two 0.007 inch inner-race fault recordings:

```text
Normal_0.mat
Normal_1.mat
IR007_0.mat
IR007_1.mat
```

This is only a smoke test. It checks that loading, grouped splitting,
featurization, caching, and classification work end to end. It is much too
small to report as a serious benchmark.

The fast persistent-homology baseline run was:

```bash
python scripts/run_binary.py E:\datasets\cwru-mini --windows-per-class 2 --max-points 8 --radius-count 3 --normalize --method ph --metrics-json artifacts\cwru-mini-ph.json
```

Result:

```text
feature_vectors: 4
accuracy: 1.000
f1: 1.000
confusion_matrix: [[1, 0], [0, 1]]
```

The small PSRT-pairs run was:

```bash
python scripts/run_binary.py E:\datasets\cwru-mini --windows-per-class 2 --max-points 8 --radius-count 3 --normalize --method psrt-pairs --cache-dir artifacts\mini-cache --metrics-json artifacts\cwru-mini-psrt.json
```

Result:

```text
feature_vectors: 4
total_subsets_enumerated: 3888
accuracy: 1.000
f1: 1.000
confusion_matrix: [[1, 0], [0, 1]]
```

A slightly larger mini run used four windows per class:

```bash
python scripts/run_binary.py E:\datasets\cwru-mini --windows-per-class 4 --max-points 12 --radius-count 4 --normalize --method psrt-pairs --cache-dir artifacts\mini-cache-12 --metrics-json artifacts\cwru-mini-psrt-12.json
```

Result:

```text
feature_vectors: 8
total_subsets_enumerated: 63440
accuracy: 1.000
f1: 1.000
confusion_matrix: [[2, 0], [0, 2]]
```

## Caveats

- The point clouds are downsampled before the graded Betti computation. That
  makes the representation coarse by design.
- Plain persistent homology may already separate healthy and faulty windows.
  The `--method ph` baseline is here so that question can be checked plainly.
- Hochster's formula is a subset enumeration. Point-cloud sizes have to stay
  small, or the run will get silly fast.

## Reference

The main reference is:

Faisal Suwayyid and Guo-Wei Wei. "Persistent Stanley-Reisner Theory."
arXiv:2503.23482, 2025. <https://arxiv.org/abs/2503.23482>

Project notes and tests use the arXiv version:
<https://arxiv.org/pdf/2503.23482>.
