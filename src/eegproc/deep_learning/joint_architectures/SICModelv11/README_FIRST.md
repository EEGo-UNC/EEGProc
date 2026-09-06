# SIC MTLFuseNet-style 3D-CNN update

SIC builder API version 15 replaces the within-window temporal BiLSTM with a
spatio-temporal 3D-CNN while retaining the trial-level BiGRU classifier:

```text
channel-major 3-band EEG, one-second windows (128, 42)
    -> parallel GCN-GRU and MTLFuseNet-style 3D-CNN encoders
    -> direct per-timestep feature concatenation
    -> ordered full-trial BiGRU sequence
    -> one VariationalClassifier logits head
```

The 3D-CNN first restores each timestep's 14 Emotiv electrodes to the same
9x9 scalp layout used by EEGProc's MTLFuseNet preprocessing. The input to
`Conv3D` is `(time, grid-row, grid-column, frequency-band)`. Convolutions span
time and both scalp axes; pooling acts only on the scalp axes. Spatial global
average pooling therefore produces `(128, cnn3d_filters[-1])` without removing
or reordering time.

The default branch is:

```json
{
  "cnn3d_filters": {"fixed": [32, 64, 128]},
  "cnn3d_temporal_kernel_size": 7,
  "cnn3d_spatial_kernel_size": 3,
  "cnn3d_spatial_pool_sizes": {"fixed": [2, 2, 1]},
  "cnn3d_dropout": 0.2,
  "cnn3d_grid_size": 9,
  "use_cnn3d_branch": true
}
```

With the standard GCN-GRU width of 384, direct fusion is 384 + 128 = 512
features per timestep. The classifier still uses the tapered two-layer BiGRU
`[128, 64]`, and the subject adversary still receives its final trial state.

When reconstruction is enabled, the 3D-CNN feature sequence has its own
independent decoder. It is never decoded through the graph branch or a fused
representation. Calibration freezes the 3D-CNN backbone; unfreeze level 2
continues to mean trial BiGRU plus VC head.

## Longleaf order

From the project root, first run one four-subject smoke test per label:

```bash
TARGET_DIMENSION=valence sbatch src/eegproc/deep_learning/joint_architectures/SICModelv11/SLURM_scripts/smoke_test_sic_mldg_brier_ablations.sh
TARGET_DIMENSION=arousal sbatch src/eegproc/deep_learning/joint_architectures/SICModelv11/SLURM_scripts/smoke_test_sic_mldg_brier_ablations.sh
```

After both complete without NaNs, shape failures, or class collapse in the
first diagnostic rows, submit the full comparison. Its array runs the fused
model, GCN-GRU-only, and 3D-CNN-only profiles sequentially:

```bash
TARGET_DIMENSION=valence sbatch src/eegproc/deep_learning/joint_architectures/SICModelv11/SLURM_scripts/full_run_sic_mldg_brier_ablations.sh
TARGET_DIMENSION=arousal sbatch src/eegproc/deep_learning/joint_architectures/SICModelv11/SLURM_scripts/full_run_sic_mldg_brier_ablations.sh
```

API-v14/BiLSTM checkpoints are intentionally incompatible with this update;
retrain the LOSO population models before calibration.
