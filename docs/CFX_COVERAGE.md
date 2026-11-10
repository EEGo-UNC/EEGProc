# Bidirectional counterfactual coverage

`CFX-update` adds the SIC study runtime to the valence-enhancement codebase.
The workflow uses frozen LOSO checkpoints. It does not train models or tune
settings on held-out subjects.

Each task runs both **0 → 1** and **1 → 0** by default. A trial is eligible
when its true label and original argmax prediction both equal the direction's
source class. Every eligible trial is attempted under all four objectives:
`target_latent`, `base`, `typicality`, and `typicality_no_physiology`.
Both directions use the same frozen model, preprocessing, target confidence,
seeds, optimization budgets, proximity terms, and physiological references.
All arms in a direction share one eligibility manifest. Each arm starts from
the original trial with fresh optimizer state. Existing stopping and two-phase
typicality behavior are preserved.

Typicality uses the frozen **target-class** VC Gaussian: class 1 for 0 → 1,
class 0 for 1 → 0. Its acceptance threshold is the **10th percentile** of
squared diagonal Mahalanobis discrepancies from all true target-class source
trials, including misclassified source trials. NumPy's `method="higher"`
quantile convention and `D <= tau` define the boundary; ties and finite-sample
rounding can admit more than exactly 10% of source trials. No held-out trial
contributes to calibration. New study commands accept only
`--typicality-quantile 0.10`. Physiological interval settings remain separate.

## Run one task

Use Python with the project's `deep-learning` dependencies, and run from the
repository root. Prepared trials must match the checkpoint's preprocessing:
full chronological `(N, W, T, 42)` DREAMER features, binary labels, subject IDs,
and trial IDs. Raw-loader examples are in the launcher below.

```bash
PYTHONPATH=src python -m eegproc.model_explainability.typicality.runner \
  --models-json /path/to/valence/loso_zero_shot_models.json \
  --model-dir /path/to/valence/loso_zero_shot_models \
  --task valence --trials-npz /path/to/valence_trials.npz \
  --target-probability "$VALENCE_TARGET_PROBABILITY" \
  --typicality-quantile 0.10 \
  --out-dir runs/cfx_coverage/valence
```

Supply the previously frozen task-specific target confidence and optimization
options. The CLI supports learning rate/decay, step budget, objective weights,
VCSC tolerance, and stopping settings; use `--help` to inspect them. The runner
requires a zero-shot checkpoint for every dataset subject unless `--subjects`
explicitly selects a shard. Use all 23 folds for the full DREAMER experiment.
The arousal command uses its matching labels, checkpoints, and confidence.
The runner has no dependency on another checkout or branch.

`--directions 1_to_0` selects only the reverse direction. `--trial-ids` is an
explicit development subset; its excluded trials are still recorded. The
optional `--no-include-target-latent` and
`--no-include-typicality-no-physiology` switches produce smaller development
studies; do not combine reports with different objective sets.

Outputs for a task are isolated by direction:

```text
TASK/
  0_to_1/
    study.json
    subject_SS/
      fold.json
      eligibility.json
      observations.json
      calibration/{region,source_trials,physiology,vcsc}.json
      trial_TT/OBJECTIVE/attempt_NNNN/{result,physiology,complete}.json
    report/
  1_to_0/...
  report/
```

Histories are streamed to JSONL and CSV. Checkpoint/data/code hashes and
calibration membership are archived. EEG/latent tensors are not saved by
default; an optional repair baseline retains its reconstruction for paired
comparisons and verified resume.
Every completed attempt includes physiology components for original EEG,
its reconstruction, and the counterfactual in `physiology.json`, compatible
with the existing binary/graded physiology tools. Unsuccessful finite
endpoints receive the same diagnostics and summaries as successful endpoints.

`--resume` verifies completed-attempt hashes and reuses them. Interrupted or
error attempts are preserved and retried in a new attempt directory. Protocol,
checkpoint, data, or code changes require a new output directory.

## Full 46-shard Slurm run

The launcher runs valence subjects 0–22 and arousal subjects 0–22; each shard
runs both directions and all four arms. Activate the appropriate environment
before submission and use a new immutable output path.

```bash
export VALENCE_MODELS_JSON=/path/to/valence/loso_zero_shot_models.json
export AROUSAL_MODELS_JSON=/path/to/arousal/loso_zero_shot_models.json
export VALENCE_TRIALS_NPZ=/path/to/valence_trials.npz
export AROUSAL_TRIALS_NPZ=/path/to/arousal_trials.npz
export CFX_OUT_DIR=/path/to/runs/cfx_coverage_v1
# Also set VALENCE_TARGET_PROBABILITY and AROUSAL_TARGET_PROBABILITY
# to the frozen values from the matching experiments.
sbatch src/eegproc/model_explainability/slurm/run_cfx_coverage.sh
```

Optional `VALENCE_MODEL_DIR` / `AROUSAL_MODEL_DIR` relocate checkpoint files.
For a raw-data run, set `VALENCE_DATA_CONFIG` / `AROUSAL_DATA_CONFIG` to JSON
files instead of `*_TRIALS_NPZ`. These provide the existing raw loader's EEG
and label paths, task label dimension, threshold settings, sampling rate,
windowing, and normalization, exactly matching training. Each raw data config
must also set `model_module` if its model training module differs from v15.
`*_MODEL_MODULE` selects the checkpoint class, and `*_FIXED_JOINT_ALPHA`
explicitly selects a fixed reconstruction mixer when required. `RESUME=1`
reuses verified completed attempts. Extra launcher arguments are forwarded
to the runner so source-selected settings can be supplied explicitly.

## Coverage and failure reports

Reports rebuild offline without TensorFlow or checkpoints:

```bash
PYTHONPATH=src python -m eegproc.model_explainability.typicality.results \
  runs/cfx_coverage_v1 --out-dir runs/cfx_coverage_v1_report \
  --expected-subjects 23 --expected-tasks valence arousal \
  --expected-directions 0_to_1 1_to_0 --require-complete
```

- `eligibility.csv` records every held-out trial's decision for each direction.
- `exclusions.csv` lists `outside_source_class`, `originally_misclassified`,
  or `trial_filter` with task, direction, subject, and trial IDs. These are
  mutually exclusive reasons in that order.
- `coverage.csv` reports observed, eligible, and excluded counts at subject
  and population levels. Subjects with zero eligible trials retain rows and
  undefined rates.
- `trial_metrics.csv` includes unsuccessful endpoints, errors, pending work,
  stop reasons, attempt counts, and error-attempt counts. Retries contribute
  only one trial to each objective's denominator.
- `population_counterfactuals.csv` / `subject_counterfactuals.csv` report
  separate flip, confidence, typicality, and joint rates, with exact
  numerators and eligible-trial denominators. Known optimization errors count
  as failures. Pending work makes results provisional; the LaTeX table shows
  `--` for provisional cells. Distance medians and quartiles include all finite
  selected endpoints and report their available sample counts.
- `single_trial_examples.json` selects successful and unsuccessful finite
  examples independently within each task/direction/objective using proximity
  to the group's median displacement, with deterministic subject/trial ties.
- `discrepancy_distributions.csv` retains direction and fold thresholds and
  normalizes each score by its own threshold before cross-fold comparison.

The full-suite command requires all 23 subject IDs in both tasks and both
directions. Missing shards are listed as pending, and `--require-complete`
returns an error after writing the provisional report. Omit these expected
coverage options only when deliberately reporting a smaller development shard.

Direction groups are never pooled. Reports reject duplicate shards, mixed
confidence/quantile/score definitions, changed committed artifacts, inconsistent
eligibility, and mismatched result identities.

Success remains a **latent-space** criterion: target argmax and target
confidence on the optimized full-trial classification embedding. Decoded EEG
is also re-encoded after optimization for separate target-confidence and
Gaussian-typicality checks for every objective, in both directions. These
post-hoc outcomes do not select or optimize the endpoint. Older archives
without these measurements leave them unavailable. The band-filtered decoder cannot estimate the required
aperiodic exponent, so passing the four available physiological checks never
silently implies passing all five.


## Independent encoded-neighbour evaluation

The runner also evaluates distance to the fifth-nearest real target-class
source embedding, with a separate source-calibrated 10th-percentile cutoff.
Source participants are split into a reference bank and disjoint cutoff
calibration group; the outer held-out participant enters neither group.
This score is evaluated after optimization and never affects its objective,
early stopping or selected iterate. Both directions use their own target class.

Defaults are `--knn-evaluation --knn-k 5 --knn-quantile 0.10
--knn-calibration-fraction 0.25`. A small source shard with fewer than five
bank trials is explicitly marked unavailable rather than assigned a score.
`--no-knn-evaluation` disables this diagnostic while retaining terminal vectors.

Source, held-out, original, optimized, and decoded-then-re-encoded terminal
vectors are saved for independent evaluators. Optimized and re-encoded
counterfactuals remain separate. Independent reports are written under
`knn_report/`; errors and pending attempts remain visible in their denominators.
Raw distances are reported per trial; cross-fold summaries use normalized
distance and calibration percentiles.

See [the protocol and standalone report command](../src/eegproc/model_explainability/typicality/ENCODED_NEIGHBORS.md).

## Integrated repair, smoothing, and decoded evaluation

The completed `vcsc-repair-baseline` work is integrated with this suite. The
existing four objectives, target classes, failure denominators, source-only
calibration, and independent neighbours are retained. Extra options are
forwarded by `run_cfx_coverage.sh` to both directions:

- `--include-repair-baseline`: a separate optimization with the target term
  removed, measuring the physiological repair of the original reconstruction.
  Its outcome is not a fifth class-targeting objective. Its one small signal
  archive, `repair_reconstruction.npz`, is retained and verified on resume.
  A failed repair does not discard the other objectives. Feasibility and
  prediction preservation are reported; a feasible repair is not guaranteed.
- `--physiological-tolerance-quantile Q`: source-real-trial VCSC tolerance.
  `--amplitude-spectral-weight W` adds source-calibrated amplitude/power checks
  to the active constraints. Both are optional and recorded in the manifest.
- `--temporal-sparsity-weight A`, `--temporal-tv-weight B`: independently
  controlled penalties on the source-standardized latent edit.
- `--decoded-distance envelope`: phase-insensitive log-envelope proximity.
  MSE and envelope distances are reported regardless of which is optimized.
  These options act during counterfactual generation, not AE training.

For example, after setting the frozen task inputs described above:

```bash
sbatch src/eegproc/model_explainability/slurm/run_cfx_coverage.sh \
  --include-repair-baseline --physiological-tolerance-quantile 0.5 \
  --amplitude-spectral-weight 1.0 --temporal-sparsity-weight 0.003 \
  --temporal-tv-weight 0.003 --decoded-distance envelope
```

These numbers illustrate source-development options; freeze their values
using source-only development before any held-out comparison. Use a fresh
output directory after changing the protocol or implementation.

Each completed objective now reports `decoded_counterfactual` and
`decoded_reconstruction`; a completed repair also supplies `decoded_repair`.
The target probability uses class 1 for `0_to_1` and class 0 for `1_to_0`.
The CSV `reencoded_*` fields and population rates keep these measurements
separate from latent success. Missing measurements remain unavailable rather
than being interpreted as known successes. `seams.json` describes window
boundaries, and `physiology.json` retains compact components and graded scores.

## Typicality occupancy report

```bash
PYTHONPATH=src python -m eegproc.model_explainability.report_iclr_typicality_vacuity \
  --suite runs/cfx_coverage_v1 --out runs/cfx_coverage_v1_occupancy
```

The report recursively finds task/fold/direction CSVs, deduplicates identical
aggregate/shard rows, and rejects conflicting duplicates. Directions are
reported separately. It counts original membership, membership without the
typicality penalty, and paired baseline-to-typicality changes. It summarizes
completed finite endpoints, so its occupancy denominator can differ from the
main report's all-eligible success denominator. Identical endpoint discrepancy
alone does not establish that the penalty never ran.

See [remaining Table 2 displacement work](TABLE2_DISPLACEMENT_AUDIT.md).
