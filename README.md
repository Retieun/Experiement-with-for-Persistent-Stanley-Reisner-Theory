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
