# SIC v16 — Pranav's three-band CNN experiments

Branch: `SICv16-Pranav`, based on `origin/valence-enhancement` at `c0adc65`.
All eight synthetic cluster preflight tests passed on a Longleaf L40 in job
6610071. Real-data smoke results are still pending. No TensorFlow execution,
training or GPU tests were run on the local machine.

## Architecture

Input features remain channel-major, band-minor: `AF3_theta, AF3_alpha,
AF3_beta, F7_theta, ...`. Reshape the last axis to `(14, 3)` and select each
band; do not slice three contiguous groups of 14 features.

Each band has independent weights:

```
theta (T,14) -> theta Conv3D x2 -> [theta Transformer] --+
alpha (T,14) -> alpha Conv3D x2 -> [alpha Transformer] --+-> concat (T',3D)
beta  (T,14) -> beta  Conv3D x2 -> [beta  Transformer] --+
                                                       -> temporal average
                                                       -> Dense/ReLU/dropout
                                                       -> Dense logits
```

Each CNN places its 14 electrodes on the existing 9x9 scalp grid. No bands
are summed. Every branch uses the same temporal pooling schedule, so matching
tokens align at concatenation. The optional Transformer is within each window;
it does not attend across windows or across bands. No joint Transformer is
included in the initial four comparisons.

In trial mode, the ordered window/time axes are merged after concatenation,
then averaged for one trial prediction. In window mode, only that window's
time axis is averaged. The supplied experiment runner uses trial mode only.

Each optional decoder takes its band's **pre-Transformer CNN latent sequence**
and reconstructs its own normalized 14-channel input. All bands are equally
weighted: `loss = classification_loss + reconstruction_loss_weight *
mean(theta_MSE, alpha_MSE, beta_MSE)`. No grid zeros enter reconstruction MSE.
The default coefficient is 0.1, an initial setting to investigate rather than
a validated optimum. Per-band and combined reconstruction losses are logged.
Decoders are omitted entirely when `use_decoder=false`.

## Reuse of SIC

`ThreeBandSICModel` subclasses v15 and reuses the shared SIC ERM/V-REx/MLDG
optimizer steps, classification metrics, subject adversary and calibration.
The inherited `graph_encoder` and `gcn_gru_decoder` attributes are adapter slots
holding the three-band CNN container and its three decoders. There is no GCN,
MI adjacency estimation, spectral GRU, BiLSTM, convex decoder fusion, or encoder
VAE in v16. The shared code and v15 behavior are unchanged.

Initial configurations disable subject adversity, VC regularizers and MLDG.
`focal_gamma=0` makes classification ordinary cross-entropy. Later comparisons
can opt into `use_recurrent_classifier`, `use_variational_head`,
`use_subject_adversarial`, or `training_method=vrex/mldg`. MLDG requires explicit
episode settings, including `mldg_steps_per_epoch`. A separate VC discriminator
optimizer and v15's graph-specific multi-GPU sharding are not supported in v16.
Use one GPU per fold. Additional feature experiments should follow the initial
four comparisons; the cluster preflight exercises these integration paths on
small synthetic inputs.

`prepare_for_subject_calibration` retains SIC's freeze policy: level 1 adapts
the logits head; level 2 also adapts the sequence summarizer (pooling/dense by
default, recurrent when enabled). The CNN/Transformer backbone stays frozen.
Decoders can optionally remain trainable. A calibration launcher and repeated
seed experiment schedule should follow validated source models and smoke timing.

## Controlled experiment protocol

`configurations.json` defines four fixed configurations:

| Array index | Model | Reconstruction |
| --- | --- | --- |
| 0 | Three CNNs | off |
| 1 | Three CNNs | on |
| 2 | Three CNN + Transformer branches | off |
| 3 | Three CNN + Transformer branches | on |

Use the existing band-stacked DREAMER arrays: EEG `(23,trials,42,samples)` and
ratings `(23,trials,dimensions)`. Raw 14-channel arrays must be converted first.
Preprocessing reuses v15's `prepare_data`: 128-sample nonoverlapping windows,
global RMS per window, all complete windows retained in chronological order,
and valence rating >= 3 as class 1. This differs from the earlier standalone
per-feature z-score/window-prediction campaign, so old scores are not directly
matched controls. All four new configurations use this same preprocessing.

For each held-out target, two other subjects are reserved for validation using
`seed + target_subject`; the remaining 20 subjects supply gradients. No test
or validation subjects enter training. Target subsets limit test folds only;
all 23 subjects remain available to the splitter. The source-validation pool,
seeds, epochs and architecture dimensions are fixed across configurations.

Every completed epoch saves a `.keras` checkpoint and its validation and target
probabilities. No early stopping is used, so all completed configurations have
the same epoch budget. Two separate reports are produced:

* `validation_selected_summary.json`: each fold's checkpoint maximizes mean
  validation-subject trial balanced accuracy, earliest tie. Its selected test
  metrics are the ordinary validation-selected result.
* `oracle_summary.json` and `oracle_models.json`: reuse v15's `select_oracle`.
  Choose one common epoch maximizing unweighted mean target-subject trial
  balanced accuracy, earliest tie. This is test-selected oracle performance,
  not an unbiased test estimate. It is not the mean of each subject's best
  epoch. A missing subject/epoch prevents oracle selection.

Both use a fixed 0.5 threshold, with no threshold search. A screening subset
must never be compared as though it were full LOSO. Reports include accuracy,
balanced accuracy, macro F1, AUC and Brier score. `task.json` records configuration,
seed, Git revision/dirty status, input sizes/timestamps and manifest digest.
`subject_XX/split.json` makes subject separation reviewable.

## Submit the first required check

Run these on Longleaf, updating a clean checkout from the published branch:

```bash
cd "$HOME/EEGProc"
git fetch origin
git switch SICv16-Pranav
git pull --ff-only origin SICv16-Pranav
```

On first checkout, Git creates the local branch from `origin/SICv16-Pranav`.
No ZIP transfer is needed. The branch contains no datasets or prior run artifacts.

Prepare the environment once, outside array jobs:

```bash
cd "$HOME/EEGProc"
module load python/3.12.4
# Only if venv312 does not already exist:
python -m venv venv312
source venv312/bin/activate
python -m pip install -e '.[deep-learning]'
```

Submit **preflight only first**:

```bash
cd "$HOME/EEGProc"
SIC16=src/eegproc/deep_learning/joint_architectures/SICModelv16
sbatch "$SIC16/SLURM_scripts/preflight.sh"
```

The preflight requests one L40, 32 GB host memory and one hour. It tests band
isolation/order, reconstruction gradients to all CNNs, absence of reconstruction
gradients through Transformers, all four models' training/checkpoint reloads,
calibration, temporal pooling, window mode, optional SIC heads, subject adversity,
V-REx, MLDG and common-epoch oracle selection. It uses tiny synthetic arrays.
Send back `sicv16_preflight_JOBID.out` and `.err`, including failures.

Only after preflight passes, submit real-data smoke:

```bash
sbatch "$SIC16/SLURM_scripts/smoke.sh"
```

Smoke runs all four configurations, two epochs, targets 0 and 1, one GPU per
task, one task at a time. It retains full trials and all training subjects.
The two-hour task limit is a cap, not a measured runtime estimate. Send back
the Slurm logs and the complete `runs/sic_v16/smoke/suite_ARRAYID` directory.
If checkpoints are too large to transfer initially, send JSONs and subject
logs first, retaining all `.keras` files on the cluster for follow-up inspection.

## Later full runs

`full.sh` prepares all four configurations, all 23 target subjects, 50 epochs,
one GPU per task, and a 24-hour cap. Do not submit it until smoke results establish
memory use and timing; a single task runs its folds sequentially and may need
fold-level Slurm parallelization before a full campaign. The controller stops
starting folds at its soft deadline, and partial runs remain incomplete.
There is no resume support; each submission writes a new suite directory.
All epochs' checkpoints are retained, so budget storage for 4 x 23 x 50 models
for the full comparison. Use smoke checkpoint sizes to estimate actual space.

```bash
sbatch "$SIC16/SLURM_scripts/full.sh"
```

Override `PROJECT_DIR`, `VENV_DIR`, `EEG_PATH`, `LABELS_PATH`, `OUTPUT_ROOT`,
`SOURCE_EPOCHS`, `TARGET_SUBJECTS`, `BATCH_SIZE`, `VALIDATION_SUBJECTS`, `SEED`,
`TIME_BUDGET_SECONDS`, or `CONFIGS` as needed. The supplied array range is 0–3;
adjust `sbatch --array` for a different manifest. Keep budgets/settings matched
across the four comparisons. Batch size defaults to one complete trial to
limit activation memory. To rebuild reports from saved epochs:

```bash
python "$SIC16/run_experiments.py" summarize \
  runs/sic_v16/smoke/suite_ARRAYID/configuration_0
```

All launchers assume submission from the repository root and reuse the existing
Longleaf Python/CUDA/cuDNN module versions. They never submit or train locally.
