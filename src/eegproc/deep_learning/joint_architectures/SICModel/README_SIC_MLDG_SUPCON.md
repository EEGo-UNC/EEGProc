# Subject Invariant Calibration (SIC)

SIC is an EEG emotion classifier designed for cross-subject generalization and rapid subject calibration. This version trains on DREAMER valence with strict leave-one-subject-out (LOSO) evaluation, first-order MLDG, supervised contrastive regularization, and multi-shot calibration.

## Model in brief

Each EEG window is encoded by two parallel branches:

1. **GCN-GRU branch:** a mutual-information graph models relationships among EEG channels, a band-separated GCN extracts spatial-spectral features, and a GRU combines the frequency-band sequence.
2. **BiLSTM branch:** a bidirectional LSTM independently models the EEG feature sequence.

In `feature_fusion` mode, the two branch outputs are concatenated directly. A variational posterior maps the fused representation to latent `z`, and a dense softmax classifier predicts low/high valence. The source objective can also include:

- focal classification loss;
- variational-classifier regularization;
- VAE reconstruction and latent KL losses;
- a gradient-reversal subject adversary; and
- supervised contrastive (SupCon) loss on the existing classifier embedding. SupCon adds no prediction head.

Source training uses first-order **MLDG**. Every episode separates source subjects into meta-train group A and virtual-unseen meta-test group B, performs a temporary inner update on A, evaluates the adapted model on B, and applies the combined first-order gradient to the persistent model.

After source training, SIC evaluates the untouched model on every held-out subject (zero-shot). It then repeatedly restores that same source checkpoint, fine-tunes only the last two prediction layers on the target subject's calibration trials, and evaluates the remaining trials.

## Experiment represented by the full script

The supplied full-run script uses:

- DREAMER valence, 14 channels, theta/alpha/beta bands;
- 4-second non-overlapping windows with global-RMS normalization;
- strict LOSO across all available subjects;
- MLDG episodes with 18 meta-train subjects, 4 meta-test subjects, 1 trial per subject, and 6 steps per epoch;
- 100 source epochs and 30 calibration epochs;
- 3-, 6-, 9-, and 12-shot calibration with 6, 3, 2, and 3 folds, respectively;
- SupCon weights `0.0`, `0.05`, and `0.10` at temperature `0.10`; and
- Brier score as the main report and hyperparameter-selection metric.

The outer grid minimizes the aggregate **12-shot calibrated Brier score**. Zero-shot and every calibration level are still reported. A SupCon weight of `0.0` is the loss-off comparison.

## Ablations

`full_run_sic_mldg_supcon_brier_ablation_valence.sh` is a SLURM array with one task per one-factor-at-a-time profile:

| Array index | Profile | Difference from `full` |
| ---: | --- | --- |
| 0 | `full` | Both encoder branches, decoder, and median-label trials retained |
| 1 | `no_gcn_gru` | Disables the GCN-GRU branch |
| 2 | `no_bilstm` | Disables the BiLSTM branch |
| 3 | `no_decoder` | Removes the reconstruction decoder and reconstruction loss; latent KL remains |
| 4 | `remove_median` | Removes every trial whose original valence rating equals `3`, including all windows from that trial |

At least one encoder branch must remain enabled. Each profile searches all three SupCon weights, so one complete array submission evaluates 5 profiles × 3 weights = **15 configuration searches**, each with full nested LOSO and calibration.

## Required project layout

Place the full-run script at the EEGProc project root. The following project paths must already contain the current implementation:

```text
EEGProc/
├── requirements.txt
├── datasets/remove_gamma/dreamer_eeg.npy
├── datasets/remove_gamma/dreamer_labels.npy
├── full_run_sic_mldg_supcon_brier_ablation_valence.sh
└── src/eegproc/deep_learning/
    ├── cross_val.py
    ├── supervised/
    │   ├── supervised_contrastive_loss.py
    │   └── variational_classifier.py
    └── joint_architectures/sic/
        ├── sic_model.py
        └── sic_model_train.py
```

The script assumes the project is at `$HOME/EEGProc` and uses `$HOME/EEGProc/venv312`. Set `PROJECT_DIR` or `VENV_DIR` at submission if either path differs.

## Submit the full experiment

From the project root:

```bash
sbatch full_run_sic_mldg_supcon_brier_ablation_valence.sh
```

One command submits five array tasks. SLURM log files use the array job and task identifiers:

```text
sic_mldg_abl_val_<array-job-id>_<task-id>.out
sic_mldg_abl_val_<array-job-id>_<task-id>.err
```

Monitor or cancel the array with standard SLURM commands:

```bash
squeue -u "$USER"
scancel <array-job-id>
```

## Outputs

Each profile writes to its own directory:

```text
runs/full/sic_mldg_supcon_brier_ablation/DREAMER/valence/<profile>/
```

The training entry point writes per-configuration and aggregate JSON/CSV artifacts containing zero-shot metrics, per-shot calibration metrics, configuration rankings, the selected configuration, and the dataset-ablation summary. Brier score, ROC-AUC, ECE, balanced accuracy, accuracy, F1, precision, and recall are reported where defined.

Because the five profiles have disjoint output directories, their array tasks can run concurrently without overwriting one another.

## Common modifications

- To select configurations on zero-shot LOSO performance while still running and reporting calibration, change `HYPERPARAMETER_SELECTION_LEVEL` to `losocv` in the script.
- To select another supported metric, change `SELECTION_METRIC`. Brier score and ECE are minimized; accuracy, balanced accuracy, F1, and ROC-AUC are maximized.
- To alter the SupCon search, edit `supcon_loss_weight` in `make_model_grid`. Set `use_supcon` to `False` to remove SupCon computation entirely.
- To change calibration coverage, edit `CALIBRATION_LEVEL_ARGS` and ensure `CALIBRATION_SELECTION_SHOTS` matches one listed shot level.
- For ERM or V-REx, change `training_method` and remove or replace the `mldg_*` settings with the corresponding method's hyperparameters.

## Resource note

This is a substantially larger experiment than the smoke test: it runs all LOSO targets, five ablation profiles, three SupCon weights, and four calibration levels. The array design keeps each ablation independently restartable. If the cluster limits concurrent array tasks, add an array throttle such as `#SBATCH --array=0-4%2`.
