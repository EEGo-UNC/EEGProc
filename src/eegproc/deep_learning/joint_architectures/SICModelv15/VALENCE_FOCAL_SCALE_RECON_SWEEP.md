# Valence focal, logit-scale, reconstruction and episode sweep

This sweep starts from the model/data settings of array run `5168991_12`.
It contains **24 fixed configurations**:

| Setting | Values |
| --- | --- |
| `focal_gamma` | 0.5, 1.0 |
| `vc_logit_scale` | 8, 16, 32 |
| `reconstruction_loss_weight` | 0.0, 0.4 |
| `mldg_steps_per_epoch` | 8, 16 |
| Outer/source epochs | **48**, increased from 24 |
| `subject_loss_weight` | 0.3, fixed |

All remaining model/data hyperparameters match the saved run-12 configuration,
including MLDG 12 meta-train / 6 meta-test subjects, two trials per subject,
learning rates 0.0001, weight decay 0.00005, and seed 42. Each target model
uses all 22 other users; there is no validation holdout. Layer widths are
fixed architecture lists, with no implicit extra search dimensions.

The inner-count axis means complete **MLDG episodes per source epoch**. Each
episode still has one temporary inner adaptation step and one persistent outer
update. The two settings produce 384 and 768 episodes over 48 epochs.
They intentionally vary the training budget. They do not introduce multiple
adaptation steps inside an episode.

Logit scale is applied during training. Increasing it lowers the effective
softmax temperature. A positive rescaling of an already trained checkpoint
does not change binary labels at threshold 0.5.

## Submit

Prepare `venv312` with the project's `deep-learning` dependencies, then run
the wrapper with **bash** from the cluster checkout. The wrapper checks storage
before requesting GPUs and supplies absolute Slurm log paths for both arrays;
do not submit it directly with `sbatch`.

By default, outputs go under
`/work/users/<first>/<second>/<onyen>/EEGProc/runs/sweeps/sic_v15_valence_focal_scale_recon`.
The existing account directory must be available. There is no fallback to home.
Set `SWEEP_STORAGE_ROOT` to use another work/project directory. For example,
if your Longleaf scratch allocation is available:

```bash
export SWEEP_STORAGE_ROOT="/pine/scr/${USER:0:1}/${USER:1:1}/$USER/EEGProc"
```

Scratch is temporary storage subject to site retention policies. Copy final
results to retained storage. `OUTPUT_ROOT` overrides the results directory
directly; `LOG_DIR` defaults to `OUTPUT_ROOT/logs`. Both paths must resolve
outside home, including through symlinks.

The preflight creates both directories, writes and flushes a temporary file in
each to catch exhausted quota or write permissions, then checks filesystem free
space and inodes. It runs before submission and again in each array task.
`MIN_FREE_GIB=100` and `MIN_FREE_INODES=100000` are default minimum guards,
not estimates or reservations for a whole sweep. The wrapper displays `quota -s`
when available. Filesystem free space does **not** measure remaining personal or
group quota; verify sufficient allocation for the complete run. The dry run also
creates/probes the directories but submits no jobs.

```bash
SIC_V15=src/eegproc/deep_learning/joint_architectures/SICModelv15

# Print both planned submissions without starting jobs:
DRY_RUN=1 bash "$SIC_V15/SLURM_scripts/full_run_v15_valence_focal_scale_recon_sweep.sh"

# Submit source jobs and their dependent collectors:
bash "$SIC_V15/SLURM_scripts/full_run_v15_valence_focal_scale_recon_sweep.sh"
```

The wrapper submits two arrays:

1. **72 source tasks**: three target-user shards per configuration, with held-out
   users 0-7, 8-15, and 16-22. Each source task requests four GPUs and runs two
   folds concurrently on separate GPU pairs. Every fold retains its full
   22-user source pool. Splitting targets keeps the longer training settings
   within smaller per-job workloads.
2. **24 collectors**, dependent on the entire source array finishing. Each
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
Source logs are `LOG_DIR/v15_valence_focal_scale_recon_ARRAY_TASK.out/.err`;
collector logs are `LOG_DIR/v15_valence_collect_ARRAY_TASK.out/.err`.
Shell failures include the failed command and line number in the error log.

`MAX_CONCURRENT=1` limits source work to four GPUs at once. Other overrides are
`PROJECT_DIR`, `VENV_DIR`, `EEG_PATH`, `LABELS_PATH`, `SWEEP_STORAGE_ROOT`,
`OUTPUT_ROOT`, `LOG_DIR`, `MIN_FREE_GIB`, `MIN_FREE_INODES`, `SOURCE_EPOCHS`,
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
OUTPUT_ROOT/suite_SOURCE_ARRAY_ID/
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
  /absolute/output/root/suite_SOURCE_ARRAY_ID
```

The manifest order is focal gamma, then logit scale, then reconstruction, then
episode count (fastest). **Configuration 4** uses the original run-12 model
settings with 8 episodes per epoch; **configuration 16** uses the original
run-21 model settings with 8 episodes per epoch. Both use 48 source epochs.
Source array task IDs are `3 * configuration + shard`; collector task IDs are configuration IDs.

All per-epoch checkpoints are retained: a full 24-configuration run with 48
epochs contains **26,496 source model files**, plus metric rows and shard data
caches. Use an output filesystem with suitable capacity. Jobs refuse to
overwrite an existing shard directory; use a new suite for a fresh run.
The fixed run-12 launcher and the separate earlier screening sweep remain
available independently.

## Recovering suite 8061088 after quota exhaustion

Stop any remaining jobs writing to this suite before transferring it. The
wrapper does not cancel jobs, move results, delete files, recreate status files,
or resume existing shards. First inspect the current results without writing:

```bash
python "$SIC_V15/audit_valence_sweep.py" \
  runs/sweeps/sic_v15_valence_focal_scale_recon/suite_8061088
```

After selecting `SWEEP_STORAGE_ROOT` on storage with sufficient quota, copy the
suite and verify the transfer with a checksum dry run:

```bash
OLD_SUITE="$PWD/runs/sweeps/sic_v15_valence_focal_scale_recon/suite_8061088"
NEW_SUITE="$SWEEP_STORAGE_ROOT/runs/sweeps/sic_v15_valence_focal_scale_recon/suite_8061088"
mkdir -p "$NEW_SUITE"
rsync -a --info=progress2 "$OLD_SUITE/" "$NEW_SUITE/"
rsync -acn --itemize-changes "$OLD_SUITE/" "$NEW_SUITE/"
python "$SIC_V15/audit_valence_sweep.py" "$NEW_SUITE" --verify-checkpoints
```

The checksum dry run should finish successfully without reporting file changes.
The full audit reads every saved Keras archive and verifies its CRCs, which can
take substantial I/O. It checks saved epoch coverage, configuration/target
identity, checkpoints and execution status separately, and never edits the
suite. Archive integrity does not replace evaluating a reloaded model.

Only remove the home copy after the transfer is verified and the results you
need are preserved. This frees home quota; redirecting new outputs alone does
not free existing space. Complete epoch files with missing `execution.json`
need review before collection: do not invent a successful status. Incomplete
shards need retraining; automatic checkpoint resume is not implemented. A new
wrapper submission uses a new suite ID and refuses to overwrite old shards.
