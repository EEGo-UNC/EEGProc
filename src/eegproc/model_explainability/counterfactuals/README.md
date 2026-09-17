# SIC counterfactuals

For the paired distributional-typicality study using Eq. (7), source-only
threshold calibration, complete optimization archives, and offline manuscript
tables/figures, see [typicality workflow](../typicality/README.md). Its entry point is
`python -m eegproc.model_explainability.typicality.runner`.

Counterfactual optimization and plotting utilities. No source training,
subject calibration, VAE, KL term, window-level optimization, or new model
construction.

| File | Owns |
| --- | --- |
| `arguments.py` | CLI arguments and their validation; no TensorFlow imports. |
| `loss.py` | `CounterfactualLoss`: target, latent, decoded, physiological, and central loss methods. |
| `vcsc_calibration_dreamer.npz` | Measured DREAMER calibration curves used by the VCSC term. |
| `optimizer.py` | Full-trial latent gradient descent with the saved SIC classifier and decoders. |
| `runner.py` | Model/data loading, trial selection, printed diagnostics, and result files. |
| `plotting.py` | Shared counterfactual NPZ loading and channel-label helpers. |
| `heatmap.py` | Channel-by-time decoded counterfactual difference heatmap. |
| `topography.py` | Side-by-side whole-trial scalp maps for each EEG band. |
| `training_monitor.py` | Static or live epoch × target probability × difference trajectory. |

## Install in EEGProc

Keep these `.py` files together in `src/eegproc/model_explainability/counterfactuals/`.
The new runner does not import the old `counterfactual_losses.py` or
`counterfactual_state.py`. Leave those old files in place if other code still
imports them; this rewrite is not a compatibility layer for their APIs.
No change to `cross_val.py`, SIC training, or the SIC model is required. The
runner supports SICModelv11's independent reconstructions and SICModelv15's
learned joint reconstruction.

Use the existing EEGProc environment. Tests here used Python 3.12,
TensorFlow CPU 2.20.0, Keras 3.15.1, and NumPy. `pytest` is needed only for tests.

## Local smoke test

From the EEGProc repository root with its virtual environment activated, run:

```bash
PYTHONPATH=src python -m eegproc.model_explainability.counterfactuals.runner --help
```

This verifies that Python can import TensorFlow and the counterfactual modules
and build the command-line interface. It does not load a checkpoint, process
data, create output files, or run optimization.

## Run from existing raw-data files

From the EEGProc repository root, replace the checkpoint/data paths below.
The window, sampling-rate, normalization, label, and filtering settings must
match the original training run. These example values are not auto-detected.

```bash
PYTHONPATH=src python -m eegproc.model_explainability.counterfactuals.runner \
  --model /path/to/loso_target_0_zero_shot.keras \
  --raw-eeg-npy /path/to/dreamer_eeg.npy \
  --raw-labels-npy /path/to/dreamer_labels.npy \
  --dataset dreamer \
  --label-dimension valence \
  --fs 128 --window-sec 1 --window-overlap 0 \
  --window-normalization global_rms \
  --label-threshold-mode global --median-label 3 \
  --subject-id 0 --trial-id 0 \
  --target-probability 0.8 \
  --learning-rate 0.01 --max-steps 20 \
  --target-weight 1 --latent-weight 0.1 --decoded-weight 0.1 \
  --physiological-weight 0 \
  --log-every 1 \
  --out-dir runs/counterfactuals/subject0_trial0_smoke
```

Add `--remove-median-label` if that trial exclusion was used in training.
Use `--trial-ids 0 1 2` for a subset, or omit trial selection to process all
trials of the specified subject. The default target is the opposite of the
original predicted class for a binary model. Set `--target-class 0` or `1`
explicitly if desired. Multiclass models require an explicit target.

For SICModelv15's single learned joint reconstruction, add:

```bash
  --model-module eegproc.deep_learning.joint_architectures.SICModelv15.sic_model \
  --decoder-mode joint
```

Joint mode requires a v15 checkpoint trained with the decoder and both encoder
branches enabled. It runs both branch decoders and applies the checkpoint's
learned convex fusion weight; it does not train or recalibrate that weight.

Raw mode reuses `load_sic_training_data` and `_group_windows_into_trials` from
the matching SIC training module. It does not call any training function or
estimate a new MI adjacency. The default model registration module is
`eegproc.deep_learning.joint_architectures.SICModelv11.sic_model`; override
`--model-module` if your package path differs. Raw mode expects its sibling
`sic_model_train` module.

The current training metadata does not contain every preprocessing setting,
so raw mode requires explicit `--label-dimension`, `--fs`, `--window-sec`, and
`--window-normalization`. The remaining preprocessing options are printed by
`--help` and saved in `settings.json`; check them against the training command.

## Run from prepared trials

An NPZ must contain:

| Key | Shape | Meaning |
| --- | --- | --- |
| `features` | `(N, W, T, F)` | Preprocessed, normalized, chronological trial windows. |
| `subject_ids` | `(N,)` | Integer subject ID for each trial. |
| `trial_ids` | `(N,)` | Integer trial ID, unique within each subject. |
| `labels` (optional) | `(N,)` | Integer true labels, recorded only as metadata. |

For example, after the existing SIC data loading/grouping steps:

```python
np.savez_compressed(
    "prepared_trials.npz",
    features=X,
    subject_ids=subjects,
    trial_ids=trials,
    labels=y,
)
```

```bash
PYTHONPATH=src python -m eegproc.model_explainability.counterfactuals.runner \
  --model /path/to/loso_target_0_zero_shot.keras \
  --trials-npz prepared_trials.npz \
  --subject-id 0 --trial-id 0 \
  --max-steps 200 --out-dir runs/counterfactuals/subject0_trial0
```

Prepared inputs are not normalized or filtered again. No padding, cropping,
mask inference, or averaging is introduced. The inspected SIC trainer groups
equal numbers of real windows per trial and its classifier has no padding
mask. An NPZ containing a `window_mask` with invalid windows is rejected.

## Slurm: all 18 trials for subject 0

Submit the 18-task array with the subject 0 SICModelv15 checkpoint:

```bash
sbatch \
  --export=ALL,MODEL_PATH=/absolute/path/to/subject_0_sic_v15.keras \
  src/eegproc/model_explainability/counterfactuals/slurm/run_joint_counterfactuals_user_0.sh
```

The array indices are exactly `0-17`; each task processes the matching DREAMER
trial for subject 0 in joint-decoder mode. Raw data defaults to
`datasets/dreamer_eeg.npy` and `datasets/dreamer_labels.npy`. To use a prepared
trial file instead, also export `TRIALS_NPZ=/absolute/path/prepared_trials.npz`.
The optimization settings can be overridden through the environment variables
listed near the top of the script. In particular, set
`PHYSIOLOGICAL_WEIGHT` to make VCSC part of the optimized objective; its
default is `0.0`, which computes and reports VCSC as a diagnostic only. The
VCSC distance and z-score settings are also exposed as environment variables.

Every task also creates the joint heatmap, signed topography, and optimization
trajectory. Outputs are grouped under:

```text
XAI_runs/subject_0_joint_counterfactuals_<array-job-id>/
├── all_metrics.json
├── all_metrics.json.lock
├── trial_00/subject_0_trial_0/
├── ...
└── trial_17/subject_0_trial_17/
```

On exit, every task runs the locked, atomic metrics collector. The last
successful task therefore leaves `all_metrics.json` containing each complete
trial's result metrics, complete optimization history, settings, artifact list,
and cross-trial numeric summaries. `complete` is true only after all 18 trials
are present. Failed or unfinished trials remain listed in `missing_trial_ids`.

To refresh or validate the combined file manually:

```bash
PYTHONPATH=src python -m \
  eegproc.model_explainability.counterfactuals.aggregate_metrics \
  XAI_runs/subject_0_joint_counterfactuals_<array-job-id> \
  --subject-id 0 --expected-trials 18 --require-complete
```

## Model-agnostic workflow

The adapter-based workflow now lives in [`../model_agnostic/`](../model_agnostic/README.md). Its APIs and commands are documented there.

## Objective and gradient path

For original trial `x`, encode once to `z`, then initialize `z_prime = z`. The
default `--decoder-mode branches` uses the objective:

\[
L = \lambda_t\max(0, \log p_{min}-\log p(y^*\mid z'))
  + \lambda_z\operatorname{MSE}(z',z)
  + \lambda_x\frac{1}{R}\sum_r\operatorname{MSE}(R_r(z'),R_r(z))
  + \lambda_{phys}\frac{1}{R}\sum_r\operatorname{VCSC}(R_r(z')).
\]

Here, `r` ranges over the reconstruction paths selected by `--decoder-mode`:
the two independent branch paths in branch mode or the single fused path in
joint mode. VCSC compares spatial coherence and debiased wPLI² across all 91
electrode pairs and three bands with the measured DREAMER calibration bundled
in `vcsc_calibration_dreamer.npz`.

`z_prime` has shape `(1,W,T,C)`, before the recurrent classifier. Every
timestep is retained: `(1,W,T,C)` becomes `(1,W*T,C)` for the saved BiGRU,
then its output goes through the saved VC head. The code also supports a
saved SIC GRU configuration without substituting a different classifier.

For decoding, split the final feature axis at the saved branch widths.
GCN-GRU features enter only the GCN-GRU decoder; BiLSTM features enter only
the BiLSTM decoder. Each decoder sees `(W,T,C_branch)` and reconstructs the
original EEG windows. The fixed original reconstructions are computed once
per trial; each candidate is compared to its matching reconstruction R_r(z).
In branch mode, their MSEs are averaged and each
reconstruction is saved separately. Single-branch ablations are supported.

With `--decoder-mode joint`, the sole decoded reconstruction and decoded loss
are:

\[
\hat{x}_{joint}=\alpha D_g(z'_g)+(1-\alpha)D_b(z'_b),\qquad
L_x=\operatorname{MSE}(\hat{x}_{joint}(z'),\hat{x}_{joint}(z)).
\]

`alpha` is the learned sigmoid fusion weight stored in the SICModelv15
checkpoint. Only `z_prime` is optimized, so gradients reach both latent branch
segments through the frozen fusion while neither `alpha` nor decoder weights
change. This intentionally uses the single joint MSE rather than averaging the
joint MSE with the two branch MSEs.

All distances are elementwise MSEs. Coefficients are starting settings, not
empirically tuned values. The loss contains detailed method docstrings.
The adapter-based optimizer uses the same reconstruction reference for its
`signal` term. For an identity/input-space adapter, the original reconstruction
equals the original input. Decoder reconstruction error relative to observed
input remains a separate reported diagnostic. New histories and summaries
record `decoded_distance_reference` (or `signal_distance_reference`) as
`original_reconstruction`; older archives retain their original semantics.

Only `z_prime` is watched by `GradientTape` and passed to Adam. Every model
call uses `training=False`. Encoder weights, BiGRU/VC weights, decoder weights,
and model trainability flags are unchanged. A new Adam instance is created
for each trial. Gradients are checked for finiteness and clipped by norm.

VCSC is evaluated and logged even when `--physiological-weight 0`; at that
default it is diagnostic only and has no gradient contribution. A positive
weight makes its weighted value part of the objective. The VCSC score is one
physiological-plausibility diagnostic, not a certificate that a generated EEG
trial is physiologically valid.

## Selection and interpretation

Success requires the target to be the argmax class AND its softmax
probability to reach `--target-probability`. This uses argmax consistently;
it does not inherit a separately selected binary decision threshold from CV.

By default all requested steps are evaluated. Among successful iterates,
return the one with the lowest weighted proximity penalty. If none succeeds,
return the finite iterate with lowest total loss and `success=false`.
`--stop-on-success` instead stops at the first successful iterate. An
already-satisfied target returns the original latent at step 0. A zero
gradient or numerical failure stops optimization and preserves the best
finite iterate. A non-finite original objective is an error.

`selected_step` identifies the returned arrays. `steps_completed` counts
actual updates and can be larger. The last history row is not necessarily
the returned candidate. `--max-steps 0` performs baseline diagnostics only.

Console diagnostics default to every evaluated step. Each line groups the
four raw objective terms, their weighted contributions, and each contribution's
percentage of the total loss. `--log-every N` reduces the display frequency;
`--log-every 0` suppresses console diagnostics without changing `history.csv`.
Joint mode also prints the checkpoint's frozen GCN-GRU `alpha` and BiLSTM
`1-alpha` weights once before optimization.

Decoded signals are never passed back through the encoder. Counterfactual
success is the frozen classifier's prediction directly from the optimized
latent state. Reconstructions are used only for decoded displacement,
reconstruction error, VCSC, and other signal-domain diagnostics. The final
report includes VCSC for the reconstructed original, the decoded
counterfactual, and their difference. These are optimization diagnostics, not
improved accuracy or causal effects.

## Outputs

The output directory must be new or empty; existing results are never
overwritten. Completed trials are saved individually.

| File | Contents |
| --- | --- |
| `settings.json` | Arguments, loss weights, model path, input shape, selected trials, environment version. |
| `subject_<id>_trial_<id>/history.csv` | Step 0 and each finite evaluated step: total/raw/weighted losses, selected reconstruction-path MSEs, probabilities, prediction, success, gradient norm. |
| `subject_<id>_trial_<id>/result.json` | Original/latent predictions, reconstruction distances, selected losses, VCSC metrics, selected step, update count, runtime, stop reason. |
| `subject_<id>_trial_<id>/counterfactual.npz` | `x`, `z`, `z_prime`, `x_reconstructed_<path>`, `x_prime_<path>`; joint mode uses `<path>=joint`. |
| `results.json` | Completed trial summaries, updated after each trial. |
| `summary.json` | Aggregate latent success, class-flip, distance, probability-change, and VCSC metrics after all trials finish. |

`x_prime_<path>` remains in the model's preprocessed input space. This
runner does not reconstruct missing raw EEG bands or undo normalization.
Missing/disabled decoders cause an explicit error. Verify that reconstruction
was actually trained in the selected checkpoint; decoder presence alone does
not establish reconstruction quality. Select the correct saved LOSO model
for the chosen subject: the runner cannot prove training-subject exclusion.

## Quick start: generate the graphs

Run these commands from the EEGProc repository root after activating the same
Python environment used for the counterfactual run:

```bash
cd /Users/tolas/Documents/coding/EEGProc
source venv/bin/activate
```

Each completed trial directory contains the two plotting inputs:

```text
runs/counterfactuals/YOUR_RUN/subject_0_trial_0/
├── counterfactual.npz   # difference heatmap and scalp topography
└── history.csv          # three-dimensional optimization trajectory
```

Replace `YOUR_RUN` and the subject/trial numbers in the commands below with an
actual completed result directory.

### 1. Counterfactual difference heatmap

```bash
PYTHONPATH=src python -m eegproc.model_explainability.counterfactuals.heatmap \
  runs/counterfactuals/YOUR_RUN/subject_0_trial_0/counterfactual.npz \
  --branch joint \
  --sampling-rate 128
```

This interprets the 42 features as 14 electrodes × 3 bands and creates three
stacked heatmap sections: Theta, Alpha, and Beta. Each section contains the
same 14 channel rows, its own peak-channel summary, and the same time axis. It
shows RMS decoded change in one-second bins by default, avoiding the dense
phase-driven striping produced by plotting every signed EEG sample. Each band
has its own labeled robust color scale so lower-amplitude band structure stays
visible. It opens the graph and saves
`counterfactual_joint_counterfactual_difference_heatmap.png` beside the NPZ.
The default difference is
`x_prime_joint - x_reconstructed_joint`, which isolates the intervention
from decoder reconstruction error. Use `--reference input` only to include
that reconstruction error. Use `--time-bin-seconds 0.5` for finer temporal
resolution, `--measure mean-absolute` for average magnitude, or
`--measure signed-mean` when the direction of a binned change is meaningful.
Add `--shared-scale` when direct cross-band color comparison is more important
than seeing lower-amplitude within-band structure.

### 2. Whole-trial scalp topographies by band

```bash
PYTHONPATH=src python -m eegproc.model_explainability.counterfactuals.topography \
  runs/counterfactuals/YOUR_RUN/subject_0_trial_0/counterfactual.npz \
  --branch joint
```

This interprets the 42 features as 14 electrodes × 3 bands, opens three
side-by-side scalp maps (Theta, Alpha, and Beta), prints the peak channel for
each band, and saves
`counterfactual_joint_difference_topography.png` beside the NPZ. The default
maps show the signed mean counterfactual difference over the complete trial on
a zero-centered red/blue scale. Positive and negative electrodes therefore
remain distinguishable. Each band has its own symmetric color range, preserving
spatial contrast within lower-amplitude bands. Colors should be compared within
a band, not between bands; the adjacent colorbars retain the actual values.
Add `--shared-scale` only when direct cross-band magnitude comparison is
desired. Use `--measure mean-absolute` to restore unsigned activity magnitude.

For 14-channel data, the DREAMER order is automatic: AF3, F7, F3, FC5, T7,
P7, O1, O2, P8, T8, FC6, F4, F8, AF4. The scalp map currently requires these
positioned DREAMER/Emotiv channel names. Mean absolute activity is used so
positive and negative EEG values do not cancel only when
`--measure mean-absolute` is selected; the default signed mean intentionally
retains cancellation and direction.

The default flattened feature order is channel-major: Theta/Alpha/Beta for
AF3, then Theta/Alpha/Beta for F7, and so on. If the saved features instead
contain all channels for Theta followed by all channels for Alpha and Beta, add
`--feature-order band-major`. Override the labels, if needed, with for example
`--band-names Delta Theta Alpha`.

### 3. Training/optimization trajectory

```bash
PYTHONPATH=src python -m eegproc.model_explainability.counterfactuals.training_monitor \
  runs/counterfactuals/YOUR_RUN/subject_0_trial_0/history.csv
```

This opens the three-dimensional graph and saves
`history_counterfactual_training_trajectory.png` beside the CSV. Its axes are:

- x: optimization step, displayed as epoch
- y: `decoded`, the selected reconstruction-path MSE relative to the original
  reconstruction R(z) in new runs (older runs used the observed input)
- z: `target_probability`, displayed as `target_p` and zoomed to its observed
  variation so small changes remain visible

The current runner writes `history.csv` after a trial finishes, so this command
plots the completed trajectory. The plotting code contains a TODO for coloring
the trajectory by validity later.

Add `--full-probability-range` to restore a fixed y-axis from 0 to 1.

### Common options

- Joint-mode outputs use `--branch joint`.
- Branch-mode outputs use `--branch gcn_gru` or `--branch bilstm`.
- When the NPZ contains multiple reconstruction paths, `--branch` is required.
- Use `--band-names NAME1 NAME2 NAME3` to change the three band labels.
- Use `--feature-order band-major` if features are grouped by band rather than channel.
- Add `--no-show` to save the PNG without opening a graph window.
- Add `--output figures/my_plot.png` to choose a different output path.
- Append `--help` to any plotting command to see all available options.

## Verification

```bash
PYTHONPATH=src python -m pytest \
  src/tests/test_counterfactuals.py \
  src/tests/test_counterfactual_metrics.py -q
```

The focused integration suite builds a tiny SICModelv15 and verifies joint-only
loss/output routing, fixed model weights and fusion alpha, backward-compatible
branch routing, CLI exposure, and joint plotting-file loading.
The metrics tests verify complete and incomplete 18-task-style collections,
including full per-trial metrics/history preservation and aggregate values.
These tests do not validate performance on a trained DREAMER checkpoint.
