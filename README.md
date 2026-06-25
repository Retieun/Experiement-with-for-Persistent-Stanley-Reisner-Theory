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
