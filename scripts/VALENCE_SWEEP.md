# SICv15 valence oracle sweep

This experiment uses DREAMER and the SICv15 full-run valence winner as the
baseline. It submits 16 Slurm array tasks, one fixed configuration per task.
Each task requests four L40 GPUs, eight CPUs, 128 GB RAM, and **06:00:00**.
Two tasks may run concurrently by default (eight GPUs total). Inside each task,
two independent LOSO subjects run concurrently, each on its own pair of GPUs.

The default scope is all 23 target subjects. Each source model uses the other
22 subjects, six source epochs, and ten MLDG steps per epoch. Episodes retain
8 meta-train subjects, 4 meta-test subjects, and 3 trials per subject. The
threshold is 0.5, and valence ratings >= 3 are class 1, including median labels.
Preprocessing matches the full run: 128 Hz, one-second nonoverlapping windows,
global RMS normalization per window, and all windows preserved in trial order.
Use the same preconverted band-stacked `dreamer_eeg.npy` (23, trials, 42, samples)
and `dreamer_labels.npy` (23, trials, dimensions) as the SICv15 full run.
The 42 features are 14 channels × 3 bands; raw 14-channel recordings must first
be converted with the existing dataset preparation workflow.

## Submit

From the cluster checkout on `valence-enhancement`, prepare the Python 3.12
environment once before submitting. The launcher reuses `venv312` and does not
install packages from concurrent workers. The environment needs the project's
`deep-learning` extra; the cluster's existing CUDA/cuDNN modules and libdevice
setup are retained.

```bash
# Preview the 16-task submission without submitting anything:
DRY_RUN=1 bash scripts/full_run_v15_valence_sweep.sh

# Submit the complete array:
bash scripts/full_run_v15_valence_sweep.sh

# Or submit only the baseline to measure runtime first:
sbatch --array=0 scripts/full_run_v15_valence_sweep.sh
```

Overrides: `PROJECT_DIR`, `VENV_DIR`, `EEG_PATH`, `LABELS_PATH`, `OUTPUT_ROOT`,
`MAX_CONCURRENT`, `SOURCE_EPOCHS`, `TIME_BUDGET_SECONDS`, `CONFIGS`.
`SUBMIT_PYTHON` selects Python on the submission host (default `python3`).
`MAX_CONCURRENT` is applied by the `bash` submission wrapper; with direct
`sbatch`, use `--array=0-15%N`. The wrapper derives the range from the JSON
manifest, so extending the manifest does not require updating its task count.
Direct `sbatch` uses the default 0–15 range in the script unless overridden.

The default real-GPU preflight checks numerical agreement of the two-device
path, gradients, updates and checkpoint portability before starting the task.
`GPU_PREFLIGHT=0` skips it after the same cluster environment has been validated.
For a smaller screening run, set `TARGET_SUBJECTS="0 1 2 3"`; all 23 subjects
remain available to form each target's source pool. The ranking tool separates
different target sets and epoch budgets.

**Six hours is a wall-time cap, not a measured completion-time estimate.**
The controller stops starting folds after 21,000 seconds (5h50m, measured after
the environment/preflight setup). Active folds stop at batch boundaries and
finish any completed epoch's metrics/checkpoint. The controller allows another
four minutes before terminating remaining workers; Slurm enforces the six-hour
job limit, including setup. Expensive batches or checkpoint writes can still
hit that limit. Setup time and cluster load can reduce the available training
time. There is no padding if a configuration completes sooner.

No full-size GPU run has been timed here. Measure the baseline before scheduling
all configurations. If it cannot finish all 23 folds within the cap, reduce
`SOURCE_EPOCHS` consistently across the whole sweep or use a screening target
set, then rerun finalists over all 23 subjects. Completed epoch files survive
timeouts; incomplete configurations cannot win the leaderboard. This launcher
does not resume interrupted training. Use a new submission/output directory.

## Oracle model selection

For configuration c and epoch e, compute trial balanced accuracy separately
on each held-out subject, then take the unweighted mean over all target
subjects. Select **one common epoch** maximizing that mean. Exact ties select
the earliest epoch. The saved model set contains one LOSO model per subject,
all from that same selected epoch. This is not an average of independently
maximized per-subject scores, and no decision-threshold search is performed.

Every completed epoch saves an actual `.keras` model plus its target labels,
trial IDs and probabilities. Selection runs after the folds finish, ensuring
the chosen metric and the saved model correspond to the same epoch. Models
from all epochs are retained for auditing; plan disk space for 23 × 6 models
per configuration, plus the prepared array cache. `oracle_models.json` points
to the selected checkpoints using paths relative to the configuration folder.

This intentionally uses held-out labels to select checkpoints, so the resulting
score is an **oracle estimate**, not an unbiased test estimate. Target EEG/labels
are excluded from source gradients and MI-adjacency fitting. There is no target
calibration in this sweep; the six-hour budget is used for source training and
oracle evaluation. Calibration experiments can follow after selecting finalists.

## Configurations

All settings below change from the same baseline; there is no Cartesian grid
inside an array task. Layer-width lists represent one architecture.

| Task | Name | Change from baseline |
| --- | --- | --- |
| 0 | baseline | Existing full-run winner |
| 1 | lr_5e5 | Outer and inner learning rates 0.00005 |
| 2 | lr_2e4 | Outer and inner learning rates 0.0002 |
| 3 | scale_8 | VC logit scale 8 |
| 4 | scale_32 | VC logit scale 32 |
| 5 | scale_64 | VC logit scale 64 |
| 6 | vc_alpha_1 | VC alpha 1 |
| 7 | vc_beta_03 | VC beta 0.3 |
| 8 | vc_lambda_001 | VC lambda 0.01 |
| 9 | reconstruction_02 | Reconstruction loss weight 0.2 |
| 10 | reconstruction_04 | Reconstruction loss weight 0.4 |
| 11 | reconstruction_08 | Reconstruction loss weight 0.8 |
| 12 | subject_005 | Subject loss weight 0.05 |
| 13 | subject_04 | Subject loss weight 0.4 |
| 14 | dropout_02 | Classifier recurrent dropout 0.2 |
| 15 | focal_1 | Focal gamma 1 |

These are controlled comparisons around the current winner. Combinations of
promising changes should be evaluated in a subsequent sweep.

## Results

Array logs: `sicv15_valence_oracle_ARRAYID_TASKID.out/.err`.
Per-subject training logs and outputs:
`runs/sweeps/sic_v15_valence_oracle/suite_ARRAYID/configuration_TASKID/`.
Each directory contains `task.json` with the exact configuration, the prepared
data, subject logs, per-epoch models/metrics, `execution.json`,
`oracle_epoch_means.csv`, `oracle_summary.json`, and (for completed tasks)
`oracle_models.json`. The latter identifies the oracle model set.

```bash
python scripts/sic_v15_valence_sweep.py rank \
  runs/sweeps/sic_v15_valence_oracle/suite_ARRAYID

# Reconstruct a report from completed epoch files after an abrupt Slurm kill:
python scripts/sic_v15_valence_sweep.py summarize \
  runs/sweeps/sic_v15_valence_oracle/suite_ARRAYID/configuration_0
```

A missing subject/epoch, missing checkpoint, worker failure, or interrupted run
prevents a task from entering the ranking. Partial means are retained in the
summary for inspection. Training data and settings should be held fixed across
a suite; do not mix separate experiments in the same suite directory.

## Restored research code

The branch starts from EEGProc v2, which removed research pipelines. The
SIC model family and full training/calibration entry points now come from
`origin/SIC-model`, commit `393858533020e3b3853decec29fcf08a4f607682`.
The imports and cross-validation worker support have been adapted to v2's
modular library. See `docs/SIC_MODEL_IMPORT.md` for the import details.

This oracle runner keeps its original 16 configurations, source-only protocol,
and six-hour task limits. Its manifest baseline was taken from the earlier
`origin/fix/arousal-v11-calibration-level` commit
`e6201db95b6657608b346803cfce4415470643be`, specifically
`SICModelv15/SLURM_scripts/full_run_v15_valence.sh`. The imported full-run
launcher has a separate nine-configuration grid and does not replace this
oracle sweep.
