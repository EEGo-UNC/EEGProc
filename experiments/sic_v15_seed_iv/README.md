# SICv15 on SEED-IV

## Current weighted-accuracy oracle protocol

The current entry point enables class weighting for source training and target
calibration. Weights are computed from each fitting partition's window labels.
Predictions use the four-class probability argmax; accuracy is the unweighted
fraction of correct windows. Class weights affect the training loss.

Source training, epoch selection, and calibration now run in three phases:

1. `train.py --phase source --data-dir ... --suite-dir NEW_SUITE --target 0`
   trains one participant's LOSO model. Use targets 0 through 14. The existing
   12 fitting participants/two source-validation participants split is retained.
   All 12 source epochs are run, with no validation-loss early stopping or
   restoration. Every completed epoch saves its actual `.keras` checkpoint,
   held-out probabilities, labels, movie IDs, and ordinary window accuracy.
2. `train.py --phase select --suite-dir NEW_SUITE` waits for complete epoch
   records for all 15 participants. It selects **one common epoch** maximizing
   the **unweighted mean participant window accuracy**, with the earliest epoch
   winning an exact tie. It does not average each participant's independent
   maximum. This follows DREAMER valence's common-epoch oracle rule, using
   accuracy and SEED-IV windows in place of trial balanced accuracy. Missing
   epochs, mismatched protocols, modified checkpoints, or inconsistent archived
   scores prevent selection. `oracle_models.json` identifies the selected set.
3. `train.py --phase calibrate --data-dir ... --suite-dir NEW_SUITE --target 0`
   loads that participant's checkpoint from the shared selected epoch. It then
   runs the existing independent 4/8/12-movie, three-split calibration protocol
   with fresh optimizers and class weights from each calibration subset. The
   selected source model is restored between splits. Calibration uses its
   existing fixed five-epoch budget; oracle epoch selection applies to source
   checkpoints. Results are saved under `calibration/subject_00/`.

Held-out labels intentionally select the source epoch. Selected zero-shot and
downstream calibration results are **oracle estimates**. Target EEG and labels
remain excluded from source gradients, source class weights, and MI fitting.
Balanced accuracy remains a secondary diagnostic; ordinary accuracy selects
the checkpoint. This protocol does not establish that weighting improves scores.

Use a fresh `RUN_ID`/suite. Existing fold directories are never overwritten.
The current weighted oracle outputs are separate from the historical baseline
below and from `results/2026-11-09_partial.json`. Plan storage for 15 times 12
source checkpoints plus small prediction archives. Source and calibration
SUCCESS records belong to their own new phase/fold directories.

After deploying the reviewed source to the experiment snapshot on Longleaf:

```bash
source experiments/sic_v15_seed_iv/environment.sh
export RUN_ID=weighted_accuracy_oracle_v1
export SOURCE_EPOCHS=12
source_job=$(sbatch --parsable --export=ALL,PHASE=source experiments/sic_v15_seed_iv/train.slurm)
select_job=$(sbatch --parsable --dependency=afterok:"$source_job" --export=ALL experiments/sic_v15_seed_iv/select.slurm)
sbatch --dependency=afterok:"$select_job" --export=ALL,PHASE=calibrate experiments/sic_v15_seed_iv/train.slurm
```

The source array's full success gates selection; completed cohort selection
gates calibration. This does not submit anything automatically. `SMOKE=1`
uses one source epoch, the original small smoke dataset and 256 MI observations.
Selection expects all 15 smoke participants by default. For an explicit smaller
wiring cohort, pass the matching `--targets` list to the selection CLI; it must
not be reported as a full-cohort result. `--source-epochs N` must match across
source and selection phases. The CPU selection phase does not require a GPU.

Run `python -m pytest experiments/sic_v15_seed_iv/test_oracle.py` in addition
to the existing signal and streaming checks.

## Historical baseline snapshot

The following sections record the initial protocol and jobs from commit
`1d376cb`. Its source-training settings and job instructions are historical.
The current model configuration and three-phase workflow above supersede them;
the published partial results retain their original interpretation.

### Review contents

`download_raw.py` is the raw-only Kaggle CLI downloader used on Longleaf, with
output-directory creation added so it can also start in a fresh directory.
`preprocess.py`, `build_cache.py`, `streaming.py`, `train.py`, the configuration,
and Slurm launchers at that archived commit reproduce the original experiment. Paths in these scripts
are specific to Pranav's Longleaf account. The baseline source code was
snapshotted from commit `df5c31ee1f2281a085e4be73d100e6b0682a8175`.
The no-op ablation memory fix is now included directly in `sic_common/training.py`;
the patch file records how that fix was applied to the old cluster snapshot.

`ACCURACY_AUDIT.md` separates verified data checks, training-setting issues, and
proposed accuracy experiments. Those proposed accuracy changes have **not** been
applied to the running baseline. `results/2026-11-09_partial.json` contains only
compact metrics, Slurm status, and provenance for the first five completed
subjects; it contains no raw EEG, model weights, or individual predictions.
See `results/README.md` for the preliminary summary.

Run checks from this experiment directory with EEGProc on `PYTHONPATH`:
`python test_adapter.py` and `python test_streaming.py`.

For the current gated submission order, source `environment.sh`, submit
`build_cache.py` in a 64 GB general-partition CPU job, submit `train.slurm` with
`--array=0 --mem=16G --time=02:00:00 --export=ALL,SMOKE=1`, then submit the
first full fold with `--array=0 --export=ALL,SMOKE=0` and `afterok` dependencies
on both jobs. Submit `--array=1-14%2 --export=ALL,SMOKE=0` only with an `afterok`
dependency on that first full fold. The cache builder must not run while an
existing experiment is using or building its input cache. Existing jobs have
already been submitted; these instructions are for a new experiment.

The original adapter used EEGProc's existing `bandpass_filter` and the unchanged
`SICModelv15.sic_model_train.train_sic` LOSO/calibration implementation.
It does not modify SICv16, its checkout, environment, jobs, or results.

### Data and preprocessing

Raw input on Longleaf: `/work/users/p/m/pmucharl/data/SEED_IV/eeg_raw_data`.
Work/output root: `/work/users/p/m/pmucharl/sic-v15-seed-iv`.

15 real subjects, three sessions each, 24 original movie trials/session.
Class IDs from the dataset ReadMe: 0 neutral, 1 sad, 2 fear, 3 happy.
The supplied MATLAB EEG is sampled at 200 Hz. Keep all 62 channels in original
order. Apply common-average reference, zero-phase fourth-order Butterworth
filters at 1–4, 4–8, 8–14, 14–31 and 31–50 Hz, and constant detrending using
EEGProc. No extra notch. Segment into four-second nonoverlapping windows;
discard only the final incomplete window of each movie. Normalize each window
by its own global RMS. No pooled target-subject statistics or fitted
cross-subject normalization. Features are `(windows,800,310)`, channel-major,
band-minor. Shard by participant and preserve session and original trial IDs.

Dataset protocol: https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html
This uses raw band-filtered waveforms, not the distributed DE/PSD features.

### Historical initial training protocol

Four-class window classification using SICv15's GCN-GRU, BiLSTM, recurrent
classifier, subject adversary, and convex joint reconstruction. Standard ERM
joint optimization, not the DREAMER full-trial MLDG recipe. Existing full-trial
grouping requires equal windows/trial and cannot represent SEED-IV's variable
durations without dropping data or adding a masking implementation.

One fixed baseline configuration; no target-score hyperparameter tuning.
15 LOSO folds, pooling all three sessions by true subject identity. Per fold:
12 source-training subjects, two source-only validation subjects, one untouched
target subject. Source-only MI graph and class weights. Up to 12 source epochs,
batch size 8, validation-loss early stopping with patience 3 and restored best
weights. Report zero-shot accuracy/balanced accuracy separately from calibration.
Then independently calibrate on 4/8/12 whole target movie trials, three seeded
splits per shot level, five epochs, with all windows of the calibration movie
excluded from its evaluation split. Window-level metrics are not interchangeable
with paper results using other within/cross-session protocols or DE features.

The GPU smoke uses three movie identities/class/subject and one window/movie,
one source epoch and a single 4-shot calibration split. It verifies the full
pipeline, checkpointing, and four-class reports; its scores are not results.

### Longleaf jobs

Source is an isolated snapshot of the remote tracked EEGProc checkout plus
this adapter. Existing `~/EEGProc/venv312` is read-only; no package installs.
Preprocessing: CPU array 1–15, at most three tasks. GPU smoke: one L40S.
Full training: array 0–14, at most two one-GPU folds at a time, 128 GB RAM,
48-hour limit/fold. Full training depends on successful preprocessing and smoke.
Use `submission.json` in the work root to find actual submitted job IDs.
Logs are in `logs/`; processed data in `processed/`; results in `runs/`.
Array indices 0–14 correspond to actual subjects 1–15.

Submitted on 2026-11-06: preprocessing array `8427400`, GPU smoke array
`8427401` (task 0), dependent full training array `8427402` (tasks 0–14).
Remote source snapshot commit: `df5c31ee1f2281a085e4be73d100e6b0682a8175`.
Signal contract checks and model-configuration/API checks passed before
submission. GPU execution is validated by the dependent smoke job.

The first GPU smoke timed out during model construction under the shortened
20-minute limit; full array `8427402` was cancelled by its failed dependency.
On 2026-11-07, resubmitted smoke `8476521` with a two-hour limit and 16 GB RAM,
and dependent full array `8476522`. The smoke alone limits MI estimation to 256
source observations; full folds retain 15,000. Graph-construction timing and
periodic stack traces distinguish CPU MI estimation from GPU execution stalls.

Smoke `8476521` passed (2m09s). Full array `8476522` started but exhausted
128 GB host RAM before its first epoch; it was stopped for repair. Replacement
streaming GPU smoke `8483426` gates replacement full array `8483427`.
`streaming.py` converts only batches to TensorFlow instead of whole-fold arrays.
The isolated source snapshot also includes `snapshot_memory_fix.patch`, avoiding
a complete dataset copy when median-label removal is disabled. These changes
preserve all windows, split membership, model widths, MI sample count, and
optimization settings. Weighted two-epoch training/validation was compared with
ordinary Keras array feeding and produced matching model weights.

Array `8483427` reached epoch 2 for target 0, but all tasks subsequently failed
with SIGSEGV. Failures coincided with the 600-second faulthandler timer; this
is a suspected diagnostic-triggered crash, not a proven model defect. Removed
periodic asynchronous stack dumps and replaced them with `/proc/self/status`
RSS/high-water logging. Peak host RAM was also close to 128 GiB in some folds.
The full input now uses a read-only memory-mapped concatenation, verified
element-for-element against every original preprocessed shard.

2026-11-08 repair jobs: cache `8563580`, GPU smoke `8563581`, first full fold
`8563582`, and remaining target array `8563583` (1–14). The remaining array
depends on completion of the first full fold, so a short smoke alone no longer
releases all 15 tasks. This preserves the scientific configuration and adds a
full-size stability gate. Memory stability is not established by the short smoke.
