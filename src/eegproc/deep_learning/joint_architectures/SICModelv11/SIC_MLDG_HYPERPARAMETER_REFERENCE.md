# SIC 3D-CNN + MLDG hyperparameter reference

This reference matches SIC builder API version 15 and the updated smoke/full
Longleaf workers. The population model is trained with complete, ordered EEG
trials and evaluated with LOSO zero-shot prediction followed by few-shot target
calibration.

## Encoder configuration

The full model concatenates a 384-wide source-only MI GCN-GRU sequence with a
128-wide MTLFuseNet-style 3D-CNN sequence at every sample. No learned fusion
projection or cross-window averaging is applied.

```json
{
  "gcn_units": {"fixed": [128, 64]},
  "spectral_gru_units": 384,
  "cnn3d_filters": {"fixed": [32, 64, 128]},
  "cnn3d_temporal_kernel_size": 7,
  "cnn3d_spatial_kernel_size": 3,
  "cnn3d_spatial_pool_sizes": {"fixed": [2, 2, 1]},
  "cnn3d_dropout": 0.2,
  "cnn3d_grid_size": 9,
  "classifier_rnn_units": {"fixed": [128, 64]}
}
```

| Hyperparameter | Baseline | Meaning |
| --- | ---: | --- |
| `cnn3d_filters` | `[32,64,128]` | Filters in successive Conv3D blocks; the last value is the branch width. |
| `cnn3d_temporal_kernel_size` | 7 | Kernel extent along the 128-sample time axis. |
| `cnn3d_spatial_kernel_size` | 3 | Kernel extent on each scalp-grid axis. |
| `cnn3d_spatial_pool_sizes` | `[2,2,1]` | Spatial-only pooling after each block; must match the filter-list length. |
| `cnn3d_dropout` | 0.20 | SpatialDropout3D rate after each block. |
| `cnn3d_grid_size` | 9 | Grid used by the fixed DREAMER electrode coordinates. |
| `use_cnn3d_branch` | true | Enables the replacement spatio-temporal branch. |

The encoder requires DREAMER's channel-major/band-minor order and 14 Emotiv
electrodes. The three bands are Conv3D input channels. It returns
`(batch, timesteps, cnn3d_filters[-1])`.

## MLDG baseline

| Hyperparameter | Smoke | Full |
| --- | ---: | ---: |
| `source_epochs` | 3 | 3 |
| `mldg_steps_per_epoch` | 15 | 20 |
| `mldg_meta_train_subjects` | 8 | 8 |
| `mldg_meta_test_subjects` | 4 | 4 |
| `mldg_trials_per_subject` | 3 | 3 |
| `mldg_inner_learning_rate` | 1e-4 | 1e-4 |
| `mldg_meta_test_weight` | 1.0 | 1.0 |
| `learning_rate` | 1e-4 | 1e-4 |
| `weight_decay` | 5e-5 | 5e-5 |

MLDG A/B subject groups are disjoint. The MI adjacency is estimated only from
the source side of each LOSO fold. `validation_subjects=0` is deliberate because
MLDG constructs virtual-unseen subjects inside each episode.

## Classifier and objectives

| Hyperparameter | Baseline |
| --- | ---: |
| `classifier_rnn_type` | `bigru` |
| `classifier_rnn_units` | `[128,64]` |
| `classifier_rnn_dropout` | 0.40 |
| `focal_gamma` | 0.5 |
| `vc_beta` | grid `[0.2,0.6]` |
| `vc_lambda` | 0.05 |
| `subject_adversarial_weight` | 0.6 |
| `reconstruction_loss_weight` | 0.10 |

The VC is the only emotion-logits head. The subject adversary receives the
final trial BiGRU state. Each active encoder is reconstructed independently;
the GCN-GRU and 3D-CNN representations are never mixed for reconstruction.

## Calibration and reporting

The scripts evaluate 3-, 6-, 9-, and 12-shot calibration. The smoke grid is
ranked at 6 shots and the full grid at 12 shots, both by minimized Brier score.
Discrimination metrics (especially balanced accuracy and macro F1), ECE, and
per-class prediction diagnostics must still be inspected before promoting a
configuration.

`calibration_unfreeze_layers=2` trains the trial BiGRU and VC head. The encoder
branches remain frozen. Old API-v14 BiLSTM checkpoints cannot be calibrated
with this architecture.

## First experiment matrix

The smoke worker runs the fused model on four subjects. The full worker has
three array profiles:

| Task | Profile | Purpose |
| ---: | --- | --- |
| 0 | `full` | GCN-GRU + 3D-CNN fusion |
| 1 | `no_gcn_gru` | 3D-CNN-only causal ablation |
| 2 | `no_cnn3d` | Original GCN-GRU-only causal ablation |

Run both valence and arousal. Only after those six full cells should filter
width (`[16,32,64]` vs `[32,64,128]`), temporal kernel (3/7/15), or decoder
weight be expanded into a grid; doing so earlier confounds architecture benefit
with capacity and multiplies the already expensive LOSO/calibration search.
