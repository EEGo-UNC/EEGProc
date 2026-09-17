# Distributional typicality counterfactuals

This workflow implements the supplied distributional-typicality equation and
paired base CFO / typicality CFO evaluation. It optimizes the complete SIC
decoder-latent trial through the frozen classifier and decoders. It does not
train or select the emotion-recognition models.

## Equation and coordinate space

For each coordinate of the chosen ordered VC-space sequence, compute the
trial mean and population variance (`ddof=0`). The class-1 Gaussian comes
directly from the checkpoint:

```text
mu_p = model.vc_target.prior_mu[1]
var_p = exp(2 * model.vc_target.prior_log_sigma[1])
var_q = maximum(trial_variance, epsilon)
var_p = maximum(var_p, epsilon)
D = 0.5 * mean((var_q + (mu_q - mu_p)^2) / var_p - 1 + log(var_p) - log(var_q))
L_typ = lambda_typ * maximum(0, D - tau)^2
```

The default variance floor is `1e-6`. There is no Gaussian refitting, covariance
shrinkage, division by `tau`, or logit-temperature factor in this equation.
The threshold is the empirical 95th percentile (`method="higher"`) of source
class-1 trial discrepancies, configurable before the study. All source
class-1 trials contribute regardless of their predicted class. The held-out
subject never contributes to threshold or physiological calibration.

**The sequence definition needs an explicit choice.** In the current valence
configuration, decoder features have 510 coordinates and the learned VC
Gaussian has 128. Their coordinates cannot be compared directly. Both
supported mappings reuse the checkpoint's recurrent weights:

| `--typicality-sequence` | Definition of the ordered sequence used in Eq. (7) |
| --- | --- |
| `vc_window_embeddings` | One VC-space vector per original window, obtained by running the frozen BiGRU summarizer on each window; recurrent state resets per window. |
| `vc_hidden_sequence` | Every hidden timestep from the last frozen GRU/BiGRU, followed by its saved normalization; recurrent state traverses the entire trial. |

Neither option changes full-trial classification. These are distinct
operational definitions of the manuscript's `Z`; select the intended one
before collecting results. The VC priors were trained on terminal trial
embeddings, so applying them to either sequence is an explicit modeling
choice. The chosen definition, sequence length, learned priors, and source
scores are retained. `TypicalityRegion` also accepts an explicitly supplied
sequence transform through its Python API.

## Run a paired LOSO study

From the EEGProc repository root, use the saved LOSO manifest and prepared
trials from the same preprocessing configuration as training:

```bash
PYTHONPATH=src python -m eegproc.model_explainability.typicality.runner \
  --models-json /path/to/configuration/loso_zero_shot_models.json \
  --model-dir /path/to/configuration/loso_zero_shot_models \
  --model-module eegproc.deep_learning.joint_architectures.SICModelv15.sic_model \
  --task valence --trials-npz /path/to/valence_trials.npz \
  --typicality-sequence vc_window_embeddings \
  --decoder-mode joint --target-probability 0.8 \
  --typicality-weight 1 --typicality-quantile 0.95 \
  --learning-rate 0.01 --max-steps 200 \
  --min-gradient-norm 1e-6 --low-gradient-patience 5 --fs 128 \
  --out-dir runs/typicality/valence
```

The mapping shown above is an example, not an automatically chosen paper
protocol. Use the corresponding frozen arousal manifest, arousal labels, and
`--task arousal` for the other task. Set objective weights and other settings
using source-subject model selection before evaluating held-out subjects.

The manifest accepts the trainer's existing `models` entries with `path`,
`target_subject`, and `stage: "zero_shot_source_model"`. `--model-dir` relocates
checkpoint filenames when paths were written on a cluster. An optional
`source_subject_ids` in each entry specifies source membership; otherwise the
LOSO complement of the dataset is used and this inference is recorded. The
manifest's stage declaration does not independently prove training provenance.

The prepared NPZ must contain `features (N,W,T,42)`, integer `subject_ids`,
`trial_ids`, and binary true `labels`. The EEG features must be chronological,
unpadded DREAMER windows in channel-major theta/alpha/beta order. Optional
normalization offsets/scales and signal units are preserved for physical-unit
figures. Do not describe normalized amplitudes as microvolts.

The existing raw loader is also supported. Replace `--trials-npz` with:

```bash
--data-loader eegproc.model_explainability.model_agnostic.sic_adapter:load_sic_raw_trials \
--data-config /path/to/data_config.json
```

That JSON supplies the existing loader's `raw_eeg_npy`, `raw_labels_npy`,
`label_dimension`, `dataset`, `fs`, `window_sec`, `window_overlap`,
`window_normalization`, label-threshold settings, and optional `signal_unit`.
Use the same preprocessing as training. `--subjects 0 1` creates a fold shard;
reporting multiple disjoint shards preserves the actual number of subjects.
Concurrent jobs should use separate output directories.

## Outcome definitions

- Eligible trials have true class 0 and original argmax prediction 0. Every
  eligible trial is attempted with both objectives from the same original
  sequence, seed, budget, frozen model, and source calibration.
- The optimizer's target criterion is argmax class 1 and `p1 >= p_target`.
  The typicality arm additionally requires `D <= tau` for candidate selection
  and success stopping. Both arms are evaluated for typicality. Success stopping
  is enabled by default and can be disabled with `--no-stop-on-success`.
- Independently, optimization stops after `--low-gradient-patience` consecutive
  evaluated steps whose raw global gradient norm is at most
  `--min-gradient-norm`. The defaults are five steps and `1e-6`; set both to
  zero to disable this rule. The history and result record the counter and the
  final stop reason.
- Decoded signals are never passed back through the encoder. Counterfactual
  success is measured directly by the frozen classifier on the optimized
  latent state.
- The results table's typicality success requires latent target success AND
  `D <= tau`.
- `d_z` is RMSE between the original and counterfactual VC-space sequences.
  `decoder_latent_rmse` separately measures the actual optimized decoder
  features and corresponds to the square root of the CFO latent MSE term.
  `delta_dec` is RMSE between decoded counterfactual and decoded original;
  `e_rec` is RMSE between decoded original and observed input.
- The CFO decoded proximity term uses `MSE(R(z'), R(z))`, with the original
  reconstruction cached and fixed. It is zero at initialization; reconstruction
  error relative to observed input is reported separately. The history's
  `decoded` term equals `delta_dec^2` for the single joint output.
- Distances summarize all finite selected endpoints, including unsuccessful
  optimizations. Unrecoverable errors have missing distances and remain in
  success-rate denominators. Available distance counts are reported.
- Unfinished trials/folds are labeled pending and reports provisional.
  Subjects with zero eligible trials retain rows with undefined rates.

## What is saved

```text
study.json                       protocol, checkpoint/data/code hashes, fold membership
environment.json                 versions, command, source revision
subject_<id>/
  fold.json                      threshold, eligibility, recognition metrics, status
  observations.npz               all held-out predictions, moments, discrepancies
  calibration/
    region.json + region.npz      Eq. (7) definition, learned Gaussian, epsilon, tau
    source_trials.npz            source IDs, labels, moments, scores, learned priors
    sequence.json                mapping and coordinate-space definition
    vcsc.npz                    held-out R(Z0) VCSC calibration and measurements
    physiology.npz              source descriptors and empirical check bounds
  trial_<id>/
    observed.npz                original input, decoder latent, VC sequence, predictions
    base/attempt_0001/           same structure for typicality/attempt_0001/
      history.jsonl + history.csv  every finite evaluated step, starting at step 0
      trajectory/step_000000.npz first latent/VC/decoded/gradient/optimizer state
      trajectory/step_<final>.npz final latent/VC/decoded/gradient/optimizer state
      counterfactual.npz         original and selected endpoint arrays + metadata
      result.json               outcomes, selected step, losses, errors, timing
      physiology_*.npz          original/reconstruction/counterfactual PSDs and checks
      physiology.json           per-family checks and missing-check reasons
      complete.json             hashes of committed trial artifacts
report/                          CSV tables, distributions, example selection, tables.tex
```

Step `s` precedes update `s+1`; the Adam slots reflect `s` completed updates.
History includes all probabilities, loss components and weights, discrepancy,
threshold, gradient norm, learning rate, decoded outcomes, and displacements.
The selected best step is distinct from the last evaluated step.

Scalars are flushed every step to `history.jsonl` and collected into
`history.csv` when the attempt closes. Each row includes target probability,
every raw and weighted loss component, total loss, discrepancy, learning rate,
gradient norm, decoded probabilities, and displacement metrics. Full tensor
snapshots are saved only for step 0 and the last finite step. The selected
endpoint is saved separately in `counterfactual.npz`. Files remain readable
with `np.load(..., allow_pickle=False)`.

`--resume` verifies completed trial artifacts and reuses them. A fully completed
fold skips model loading. Interrupted/error trials receive new attempt
directories. Partial folds may repeat their calibration/observation inference;
there is no automatic mid-trial restart from Adam snapshots. Changes to data,
checkpoint, protocol, or recorded source hashes cause resume to refuse mixing
incompatible results.

## Offline class-1 awareness and subject-invariance audit

After a `typicality.runner` study has produced its counterfactual archives,
recompute the requested correct-class-1 reference and evaluate both held-out
real trials and counterfactual endpoints without loading TensorFlow or a model:

```bash
PYTHONPATH=src python -m eegproc.model_explainability.typicality.class_awareness \
  runs/typicality/valence \
  --samples-per-source-subject 3 \
  --samples-per-target-subject 3 \
  --typicality-quantile 0.95 --seed 42 \
  --out-dir runs/typicality/valence_class_awareness
```

Sampling is performed independently within every source subject after applying
`true_class == 1 AND predicted_class == 1`. The held-out subject's correct
class-1 trials are sampled separately and never influence the primary audit
threshold. The report also retains the generation threshold, calculates an
all-true-class-1 sensitivity result, and labels a source-plus-target pooled
threshold as descriptive only. Set either sample count to `0` to use every
available correct class-1 trial.

The audit writes fold, real-trial, counterfactual, and aggregate CSV files plus
the exact sampling manifest and input hashes. Counterfactual transitions are
reported as entered, preserved inside, exited, or stayed outside. Plain
`counterfactuals.runner` archives contain `z` and `z_prime` but not the mapped
VC sequences or source reference bank; those older runs require one
checkpoint-backed enrichment pass before this offline command can be used.

## Rebuild tables and figures without models

```bash
PYTHONPATH=src python -m eegproc.model_explainability.typicality.results \
  runs/typicality/valence runs/typicality/arousal \
  --out-dir runs/typicality/paper_report

PYTHONPATH=src python -m eegproc.model_explainability.typicality.plotting \
  runs/typicality/paper_report --out-dir runs/typicality/paper_figures
```

After the counterfactual artifacts have been produced, build the paired
channel-by-band spectral table without loading a model:

```bash
PYTHONPATH=src python -m eegproc.model_explainability.typicality.spectral_features \
  runs/typicality/valence runs/typicality/arousal \
  --output runs/typicality/paper_report/spectral_entropy_features.csv
```

The CSV contains trial-level decoded-original, decoded-counterfactual, and
paired-change values for peak frequency and spectral centroid. It also records
the median and IQR of normalized Shannon spectral entropy across the original
decoder windows. The analysis uses the decoded original reconstruction as its
reference, never concatenates independent windows, flags peaks on band edges,
and reuses saved physiology PSDs when they are available.

These commands need no TensorFlow or checkpoint inference. Outputs include
recognition fold means and sample SDs, recalls, AUROC, top-label ECE,
population percentages and medians/IQRs, per-subject rates, paired percentage
point changes, and all observed/counterfactual discrepancy distributions.
The example is the typicality-arm latent-target-and-typical success nearest its task's median
VC-sequence displacement, with deterministic subject/trial tie breaking.

Figures include probability/discrepancy/displacement trajectories, per-subject
rates, discrepancy relative to each fold's threshold, and electrode-band
power-change maps. Numerical inputs for aggregate scalp maps are also saved.
Existing counterfactual heatmap and topography commands can read the new
`counterfactual.npz` files directly.

## Physiology and subject-identification limits

VCSC is calibrated without labels on all initial reconstructions `R(Z0)` from
the held-out subject, using the same selected decoder output that the study
reports and optimizes. A `--trial-ids` smoke-test filter does not reduce this
calibration set. Only VCSC summary statistics, measurements, and trial IDs are
saved; labels are not used, and full calibration reconstructions are processed
one at a time rather than retained or archived.
Additional source-calibrated checks cover the 99th-percentile absolute
amplitude, band spectral power, coherence, and signed debiased wPLI squared.
Per-component central intervals and required in-range fractions are explicit
protocol settings. PSDs, frequencies, connectivity pairs, and all check values
are saved. The debiased estimator can legitimately be negative; it is not
clipped to zero. See [Vinck et al. (2011)](https://pubmed.ncbi.nlm.nih.gov/21276857/).

An aperiodic exponent is not identified from the current band-filtered decoder
outputs. That fifth check is `null` with a reason; the full physiological pass
rate is `NA`. The rate over available checks is separately named. Computing
a defensible fifth check needs an agreed estimator and appropriate signal
support; the archived signals allow subsequent offline work on this.

The subject probe also runs offline:

```bash
PYTHONPATH=src python -m eegproc.model_explainability.typicality.subject_probe \
  runs/typicality/valence --coordinate-policy fold_specific_descriptive \
  --out-dir runs/typicality/valence_probe
```

It uses the same disjoint whole-trial split for original, base, and constrained
representations. Scaling is fitted only on training trials. All paired finite
endpoints are included irrespective of class-flip success. Each included
subject needs at least two paired trials. Fits, scales, split IDs, probabilities,
confusion matrices, and balanced accuracies are saved; chance uses the actual
number of evaluated subjects.

Independent LOSO models can learn different latent coordinate bases, so their
pooled probe may recognize the fold itself. `fold_specific_descriptive` records
that limitation. These scores alone do not support the manuscript's subject
invariance conclusion. The lower-level probe API supports supplied common-space
representations; `--coordinate-policy shared` rejects differing checkpoint
spaces. The default report does not silently evaluate a confounded probe.

Add saved probe result files to `typicality.results --probe-results ...` to
include the subject-identification table. Unavailable values remain `--`.
