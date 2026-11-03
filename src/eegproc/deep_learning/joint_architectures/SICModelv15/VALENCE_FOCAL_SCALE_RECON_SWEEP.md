# Valence focal, logit-scale, reconstruction and episode sweep

This sweep starts from the model/data settings of array run `5168991_12`.
It contains **108 fixed configurations**:

| Setting | Values |
| --- | --- |
| `focal_gamma` | 0.5, 1.0 |
| `vc_logit_scale` | 8, 16, 32 |
| `reconstruction_loss_weight` | 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 |
| `mldg_steps_per_epoch` | 5, 10, 20 |
| Outer/source epochs | **24**, increased from 12 |
| `subject_loss_weight` | 0.3, fixed |

All remaining model/data hyperparameters match the saved run-12 configuration,
including MLDG 12 meta-train / 6 meta-test subjects, two trials per subject,
learning rates 0.0001, weight decay 0.00005, and seed 42. Each target model
uses all 22 other users; there is no validation holdout. Layer widths are
fixed architecture lists, with no implicit extra search dimensions.

The inner-count axis means complete **MLDG episodes per source epoch**. Each
episode still has one temporary inner adaptation step and one persistent outer
update. The three settings produce 120, 240, and 480 episodes over 24 epochs.
They intentionally vary the training budget. They do not introduce multiple
adaptation steps inside an episode.

Logit scale is applied during training. Increasing it lowers the effective
softmax temperature. A positive rescaling of an already trained checkpoint
does not change binary labels at threshold 0.5.

## Submit

Prepare `venv312` with the project's `deep-learning` dependencies, then run
the wrapper from the cluster checkout:

```bash
SIC_V15=src/eegproc/deep_learning/joint_architectures/SICModelv15

# Print both planned submissions without starting jobs:
DRY_RUN=1 bash "$SIC_V15/SLURM_scripts/full_run_v15_valence_focal_scale_recon_sweep.sh"

# Submit source jobs and their dependent collectors:
bash "$SIC_V15/SLURM_scripts/full_run_v15_valence_focal_scale_recon_sweep.sh"
```

The wrapper submits two arrays:

1. **324 source tasks**: three target-user shards per configuration, with held-out
   users 0-7, 8-15, and 16-22. Each source task requests four GPUs and runs two
   folds concurrently on separate GPU pairs. Every fold retains its full
   22-user source pool. Splitting targets keeps the longer training settings
   within smaller per-job workloads.
2. **108 collectors**, dependent on the entire source array finishing. Each
   collector requests two GPUs, combines its configuration's three shards,
   chooses one common epoch over all 23 users, and evaluates/calibrates only
   those selected models. It verifies matching settings and complete epoch
   coverage before publishing a model set. An incomplete or failed shard
   prevents that configuration from being selected or calibrated.

Both arrays default to two concurrent tasks. Source jobs can therefore use
eight GPUs at once, and collectors four. Both retain a 24-hour Slurm allocation.
Source workers have an 82,800-second soft budget with time to finish a current
batch/checkpoint; the allocation is a cap, not a measured runtime estimate.
The wrapper prints both array IDs. Collector output directories use the source
array ID so all results remain in one suite.

`MAX_CONCURRENT=1` limits source work to four GPUs at once. Other overrides are
`PROJECT_DIR`, `VENV_DIR`, `EEG_PATH`, `LABELS_PATH`, `OUTPUT_ROOT`, `SOURCE_EPOCHS`,
`SOURCE_BATCH_SIZE`, `TRAINING_SEED`, `TIME_BUDGET_SECONDS`, `GPU_PREFLIGHT`,
`CALIBRATION_EPOCHS`, and `CALIBRATION_BATCH_SIZE`. `SUBMIT_PYTHON` selects the
submission host's Python. `RUN_CALIBRATION=0` retains common-epoch selection
while skipping calibration. All configurations in a comparison should share
the same source epoch budget, seed and preprocessing.

## Selection and outputs

For each configuration, compute trial balanced accuracy for each held-out user
at every completed epoch, using threshold 0.5. Select the epoch maximizing the
unweighted mean over **all 23 users**; exact ties select the earliest epoch.
Shard-local winners do not determine the final model set. The collector copies
only small metric JSON files and references the saved models in their shards.

Results are under:

```text
runs/sweeps/sic_v15_valence_focal_scale_recon/suite_SOURCE_ARRAY_ID/
  configuration_N/
    shard_0/  shard_1/  shard_2/
    task.json
    execution.json
    subject_NN/epoch_EEE.json
    oracle_epoch_means.csv
    oracle_summary.json
    oracle_models.json
    final_metrics.json
    calibration/
```

Calibration defaults to one epoch at 3 shots / 6 folds. `final_metrics.json`
records the selected common epoch and the final zero-shot/3-shot means, with
the existing check that reloaded zero-shot balanced accuracy matches selection.
With calibration disabled, the selection score is in `oracle_summary.json`.
These scores are explicitly oracle estimates because held-out labels select
the epoch, while source gradients and adjacency fitting use source users only.

```bash
python "$SIC_V15/sic_v15_valence_sweep.py" rank \
  runs/sweeps/sic_v15_valence_focal_scale_recon/suite_SOURCE_ARRAY_ID
```

The manifest order is focal gamma, then logit scale, then reconstruction, then
episode count (fastest). **Configuration 19** has the original run-12 model
hyperparameters; **configuration 73** has the original run-21 model
hyperparameters. Their source epoch budget is now 24. Source array task IDs
are `3 * configuration + shard`; collector task IDs are configuration IDs.

All per-epoch checkpoints are retained: a full 108-configuration run with 24
epochs contains **59,616 source model files**, plus metric rows and shard data
caches. Use an output filesystem with suitable capacity. Jobs refuse to
overwrite an existing shard directory; use a new suite for a fresh run.
The fixed run-12 launcher and the separate earlier screening sweep remain
available independently.
