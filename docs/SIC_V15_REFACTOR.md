# SICv15 code reduction

The valence-enhancement refactor keeps SICv15's model and experiment behavior.
It removes duplicated implementations by sharing the sequence/training core,
data preparation, reporting, grid rules, and calibration search with SICv11.
The original SICModel variant is unchanged. SICv11 keeps its independent-branch
reconstruction objective; v15 keeps its learned convex joint reconstruction.

## Size

Python source lines, including comments and blank lines:

| Scope | Before | After |
| --- | ---: | ---: |
| v15 `sic_model.py` | 3,149 | 1,286 |
| v15 training entry point | 1,627 | 833 |
| v15 CLI/configuration | 741 | 556 |
| v15 calibration-only entry point | 1,363 | 61 |
| Entire v15 package, including GPU support | 7,097 | 2,953 |
| All joint-architecture Python, including the new shared code | 18,251 | 14,165 |

The model file is 59% shorter and the v15 package is 58% shorter. Counting the
new shared modules as well, 4,086 lines of duplication are removed overall.

## Where the code lives

- `SICModelv15/sic_model.py`: v15 configuration and constructor, convex fusion,
  reconstruction objective, GPU hooks, and the public model builder.
- `sic_common/model.py`: shared sequence/head assembly, metrics, subject
  adversary, source/calibration steps, and serialization. It composes EEGProc's
  existing GCN/GRU encoders, recurrent classifier, VariationalClassifier and
  MLDG utilities.
- `sic_common/training.py`: shared array preparation and experiment reports.
- `sic_common/config.py`: TensorFlow-free Cartesian grids and selection rules.
- `sic_common/calibration.py`: checkpoint calibration through EEGProc's existing
  `subject_calibration_cv`; each entry point supplies its model version to
  spawned workers explicitly.

Both versions retain their module paths, builder arguments, CLI flags, defaults,
Keras model registration names, and compatibility aliases. The small change in
`domain_generalization/meta_learning.py` delays the legacy window sampler's CV
adapter imports until construction, so importing standalone MLDG before SIC no
longer triggers the CV package's circular import.

## Behavioral checks

A fixed-seed comparison against a snapshot of the pre-refactor implementation
covers full v15, graph-only, temporal-only, no decoder, window classification,
V-REx, MLDG, and calibration with unfrozen decoders. Predictions, losses,
gradients, initial/updated weights, configuration and metric names match
exactly on CPU. All eight pre-refactor `.keras` checkpoints load and reproduce
their predictions in the refactored model; every saved model weight is restored
exactly, including the fusion scalar and adversary.

Regression coverage also exercises source training/calibration and serialization
for all three SIC versions; fresh-process calibration-only workers for v11/v15;
MLDG with two logical devices, dropout, eager and graph execution; the shared
CV interfaces; and the valence sweep's preprocessing, checkpoints, budget and
oracle ranking. Real CUDA hardware and six-hour runs are not exercised locally.

The six-hour sweep and its configurations are unchanged. Its oracle still picks
one common epoch maximizing the unweighted mean held-out subject trial balanced
accuracy, with an earliest-epoch tie break.

## Checkpoint correction

The fresh-worker tests also exposed an existing v15 loading issue: the convex
fusion scalar was built only during reconstruction, after calibration had
already snapshotted the loaded model's weights. The loader now builds that
scalar and the shared subject-adversarial head before Keras restores the
checkpoint, preserving all their learned weights and preventing a later
weight-count mismatch. This does not change source training
or prediction; it repairs reconstruction/calibration after loading.
