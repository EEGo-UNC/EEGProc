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
calibration membership are archived. Large EEG/latent tensors are not saved.
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
is used for displacement and physiology; decoded/re-encoded classification
validity is unmeasured. The band-filtered decoder cannot estimate the required
aperiodic exponent, so passing the four available physiological checks never
silently implies passing all five.
