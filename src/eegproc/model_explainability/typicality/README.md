# Full-trial embedding typicality counterfactuals

This workflow runs paired base CFO / typicality CFO evaluation. It optimizes
all encoder features of one complete supplied trial through the frozen
classifier and decoders. It does not train or select emotion-recognition models.

## Score and representation

Let `H` have shape `(1, W, T, D_encoder)`. Preserve all windows and timesteps
in chronological order, reshape it to `(1, W*T, D_encoder)`, and run the saved
trial recurrent classifier once. Its terminal embedding `e(H)` is exactly the
vector passed to the frozen VC head for prediction. Typicality reuses that same
embedding during optimization, so its gradient traverses the complete trial
without a second recurrent pass or a reset between windows.

The score is **squared diagonal Mahalanobis distance divided by VC dimension**:

```text
e = frozen_trial_recurrent_classifier(reshape(H, (1, W*T, D_encoder)))
mu = model.vc_target.prior_mu[1]
var = maximum(exp(2 * model.vc_target.prior_log_sigma[1]), epsilon)
D = mean((e - mu)^2 / var)
evaluation: typical iff D <= tau
phase-2 loss: L_typ = lambda_typ * D / maximum(tau, epsilon)
```

The default variance floor is `1e-6`. There is no square root, factor of 1/2,
within-trial mean/variance aggregation, Gaussian KL, covariance refitting, or
logit-temperature factor in this score. Both the Gaussian and `e(H)` belong to
the full-trial terminal classification space used during training.

The cutoff is the empirical 95th percentile (`method="higher"`) of source
class-1 **trial embedding scores**, configurable before the study. All source
class-1 trials contribute regardless of their predicted class. The held-out
subject never contributes to this threshold. Division by `tau` scales the
optimization penalty across folds and does not change the acceptance cutoff.

### Assumptions and interpretation

- A full trial means every supplied window and timestep, with no cropping,
  subsampling, or temporal averaging in the counterfactual workflow. The current
  DREAMER preprocessing retains the middle 60 seconds of each stimulus; with
  the launch scripts' one-second non-overlapping windows, this is 60 windows
  of 128 timesteps. It does not restore the original uncropped recording.
- The checkpoint's learned class Gaussian and model weights remain frozen.
  Only the source threshold is recalibrated; no new emotion model is trained.
- `--typicality-representation vc_trial_embedding` is the default and sole
  supported representation. Classifier state spans the complete trial.
- The existing one-sided cutoff and two-stage optimizer are retained. This is
  an operational measure of class-conditional embedding compatibility, not
  proof of a statistical typical set, temporal plausibility, or physiological
  validity. Minimizing the score favors the class Gaussian's center. Separate
  physiological checks and counterfactual proximity terms still apply.

### Migration from window-moment KL

Use a **new output directory** and recalibrate each fold. Old KL scores,
thresholds, and tuned weights must not be assumed transferable. Study and
region metadata use schema version 2 and explicitly record the score and
representation. Legacy regions cannot be loaded for new optimization; resume
rejects changed code/protocol. Offline reports and class-awareness audits can
still read legacy archives separately, but reject mixtures of score or
representation definitions. The full-trial subject probe requires new archives.

The old `vc_window_embeddings` and `vc_hidden_sequence` modes are rejected.
`--typicality-sequence` remains an option-name alias but accepts only the new
`vc_trial_embedding` value. Launch scripts use `TYPICALITY_REPRESENTATION`;
an explicitly supplied legacy `TYPICALITY_SEQUENCE` value is forwarded and
rejected rather than silently changing its meaning.

## Run a paired LOSO study

From the EEGProc repository root, use the saved LOSO manifest and prepared
trials from the same preprocessing configuration as training:

```bash
PYTHONPATH=src python -m eegproc.model_explainability.typicality.runner \
  --models-json /path/to/configuration/loso_zero_shot_models.json \
  --model-dir /path/to/configuration/loso_zero_shot_models \
  --model-module eegproc.deep_learning.joint_architectures.SICModelv15.sic_model \
  --task valence --trials-npz /path/to/valence_trials.npz \
  --typicality-representation vc_trial_embedding \
  --decoder-mode joint --target-probability 0.8 \
  --typicality-weight 1 --typicality-quantile 0.95 \
  --learning-rate 0.01 --max-steps 200 \
  --min-gradient-norm 1e-6 --low-gradient-patience 5 \
  --typicality-improvement-patience 10 --typicality-min-delta 1e-6 \
  --physiological-weight 1 --fs 128 \
  --out-dir runs/typicality/valence
```

Use the corresponding frozen arousal manifest, arousal labels, and
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
  A positive physiological weight additionally requires the raw VCSC penalty
  to be at most `--physiological-tolerance` for feasibility. The base arm stops
  at its first feasible success by default.
- The typicality arm first reaches target and VCSC feasibility, then activates
  the normalized, always-on `D` penalty. It does not stop at the first success.
  Among feasible candidates it selects the lowest `D`, breaking ties by latent,
  decoded, and physiological proximity. Evaluation remains `D <= tau`.
- Independently, optimization stops after `--low-gradient-patience` consecutive
  evaluated steps whose raw global gradient norm is at most
  `--min-gradient-norm`. The defaults are five steps and `1e-6`; set both to
  zero to disable this rule. The history and result record the counter and the
  final stop reason.
- The typicality phase also stops after
  `--typicality-improvement-patience` feasible evaluations without a decrease
  of at least `--typicality-min-delta` in `D`.
- The encoder processes the original trial once during CFO. Baseline
  reconstructions and decoded counterfactuals are never re-encoded.
- The main table reports latent target success (target argmax AND confidence),
  typicality (`D <= tau`) on the optimized full-trial classification embedding,
  and their conjunction. The frozen Gaussian and source threshold are unchanged.
  `typicality_success` denotes this latent target-and-typicality decision.
- Decoded signals provide reconstruction error, displacement, feature analysis,
  and physiological diagnostics. Latent success does not establish a class flip
  for a decoded signal passed through the full model. All eligible attempts
  remain in the success denominators. Physiology outcomes remain separate.
- `d_z` is RMSE between the original and counterfactual full-trial
  classification embeddings.
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
  observations.npz               all held-out predictions, full-trial embeddings, discrepancies
  calibration/
    region.json + region.npz      Mahalanobis definition, learned Gaussian, epsilon, tau
    source_trials.npz            source IDs, labels, full-trial embeddings, scores, learned priors
    representation.json          full-trial mapping and coordinate-space definition
    vcsc.npz                    held-out R(Z0) VCSC calibration and measurements
    physiology.npz              source descriptors and empirical check bounds
  trial_<id>/
    observed.npz                original input, decoder latent, classification embedding, predictions
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
History includes latent probabilities, loss components and weights, discrepancy,
threshold, gradient norm, learning rate, decoded distances, and displacements.
Endpoint predictions and typicality use the optimized classification embedding.
`decoded_trials.<path>` stores signal-distance and physiology diagnostics only.
The selected best step is distinct from the last evaluated step. `d_z` now
measures full-trial embedding RMSE; `decoder_latent_rmse` continues to measure
all optimized encoder coordinates. `observations.npz` and `source_trials.npz`
store `(N, d)` arrays named `embeddings`. Each trial's `observed.npz` stores
`classification_embedding` with shape `(1, d)`. Counterfactual endpoints store
`classification_embedding` and `classification_embedding_prime`, alongside
baseline reconstructions and decoded counterfactual signals.
No artificial sequence axis or window moments are saved.

New studies declare `round_trip_evaluation=latent_only`. Resume rejects earlier
round-trip manifests because their evaluation protocol differs. Historical
reports remain readable under their recorded protocol and cannot be pooled
with new latent-only studies. Use a new output directory for new runs.
The class-awareness audit and subject probe inspect optimized latent representations.

Scalars are flushed every step to `history.jsonl` and collected into
`history.csv` when the attempt closes. Each row includes target probability,
every raw and weighted loss component, total loss, discrepancy, learning rate,
gradient norm, latent probabilities, and displacement metrics. Full tensor
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
classification embeddings or source reference bank; those older runs require one
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
The example is a typicality-arm latent-target-and-typical success
nearest the successful cohort's median full-trial embedding displacement,
with deterministic subject/trial tie breaking. Latent-only legacy reports
retain their original example rule.

Figures include latent probability/discrepancy/displacement trajectories with
original and optimized latent typicality, per-subject
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
full-trial classification embeddings (`--representation classification_embedding`). Scaling is fitted only on training trials. All paired finite
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
