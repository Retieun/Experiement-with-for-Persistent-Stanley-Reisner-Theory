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

## Why I Made This

I made this as a small implementation exercise after reading the PSRT paper. My
goal was to see whether the persistent graded-Betti construction could be turned
into a reproducible signal-processing pipeline on real vibration data.

This is not an official implementation of the paper. I would be grateful for
corrections, especially around whether the persistent graded-Betti computation
matches the intended construction.

## Limitations

- This is prototype research code, not a polished package or benchmark suite.
- The point clouds are aggressively downsampled before Hochster enumeration.
- The reported CWRU numbers below are small-data checks, not publishable
  performance claims.
- The PH baseline here is a Betti-curve baseline from the same small Rips
  machinery, not a full persistence-diagram comparison with a mature TDA
  library.

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

By default the CLI uses `psrt-pairs`, an implementation attempt of the paper's
two-scale persistent graded-Betti construction. It computes ranks over all
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

## Small CWRU Runs

I tested the pipeline on two small official CWRU subsets. These runs are meant
to catch obvious pipeline mistakes and give a rough feel for cost. They are not
large enough to claim diagnostic performance.

The first smoke-test subset used two normal recordings and two 0.007 inch
inner-race fault recordings:

```text
Normal_0.mat
Normal_1.mat
IR007_0.mat
IR007_1.mat
```

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

I also ran a broader small subset with four normal files and twelve fault files
across inner-race, ball, and outer-race faults:

```text
Normal_0.mat ... Normal_3.mat
IR007_0.mat ... IR007_3.mat
B007_0.mat ... B007_3.mat
OR007@6_0.mat ... OR007@6_3.mat
```

Persistent-homology baseline:

```bash
python scripts/run_binary.py E:\datasets\cwru-share --windows-per-class 16 --max-points 8 --radius-count 3 --normalize --method ph --cv-folds 4 --metrics-json artifacts\cwru-share-ph.json
```

Result:

```text
feature_vectors: 32
accuracy: 0.857
f1: 0.857
confusion_matrix: [[3, 1], [0, 3]]
cv_accuracy_mean: 0.812
cv_f1_mean: 0.799
```

PSRT-pairs run:

```bash
python scripts/run_binary.py E:\datasets\cwru-share --windows-per-class 12 --max-points 10 --radius-count 3 --normalize --method psrt-pairs --cv-folds 4 --cache-dir artifacts\share-cache-psrt --metrics-json artifacts\cwru-share-psrt.json
```

Result:

```text
feature_vectors: 24
total_subsets_enumerated: 55440
accuracy: 0.833
f1: 0.800
confusion_matrix: [[3, 0], [1, 2]]
cv_accuracy_mean: 0.958
cv_f1_mean: 0.950
```

## Remaining Caveats

- The point clouds are downsampled before the graded Betti computation. That
  makes the representation coarse by design.
- Plain persistent homology may already separate healthy and faulty windows.
  The `--method ph` baseline is here so that question can be checked plainly.
- Hochster's formula is a subset enumeration. Point-cloud sizes have to stay
  small, or the run will get silly fast.

## Note For Sharing

If sharing this with the paper's authors, I would frame it as an independent
prototype rather than a completed implementation:

> I built this small demo after reading your Persistent Stanley-Reisner Theory
> paper. It applies an implementation attempt of the persistent graded-Betti
> construction to CWRU bearing vibration windows. I would be grateful for any
> corrections, especially if I have misunderstood the persistent graded-Betti
> or facet-persistence constructions.

## Reference

The main reference is:

Faisal Suwayyid and Guo-Wei Wei. "Persistent Stanley-Reisner Theory."
arXiv:2503.23482, 2025. <https://arxiv.org/abs/2503.23482>

Project notes and tests use the arXiv version:
<https://arxiv.org/pdf/2503.23482>.
