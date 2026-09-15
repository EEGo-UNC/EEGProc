# Counterfactual paper evaluation runbook

This runbook turns every placeholder in the paper's counterfactual-evaluation
section into a named, reproducible output. It covers DREAMER valence and
arousal, all 23 leave-one-subject-out (LOSO) folds, the base counterfactual
objective, and the class-1-typicality-constrained objective.

The two GPU stages have Slurm launchers:

- `SLURM_scripts/run_paper_cfo_references.sh` encodes the real trials, builds
  each fold's source-only class-1 reference, and evaluates held-out real
  class-1 trials for the subject-invariance analysis.
- `SLURM_scripts/run_paper_cfo_population.sh` optimizes every eligible held-out
  class-0 trial under both objectives.

Aggregation, representative-example selection, and plotting are deliberately
local commands. They do not belong in GPU jobs.

## Readiness gate

Do **not** submit a paper run yet. The repository currently has the base CFO
runner and the single-trial heatmap/topography/trajectory tools, but it does
not yet have the following paper-evaluation interfaces:

1. `eegproc.model_explainability.fit_class_typicality`
2. typicality and eligibility options in `run_counterfactuals`
3. `eegproc.model_explainability.summarize_counterfactual_paper`
4. `eegproc.model_explainability.select_counterfactual_example`
5. `eegproc.model_explainability.plot_counterfactual_population`
6. a specified physiological pass/fail check suite

The Slurm scripts check for these interfaces and stop before doing work if they
are absent. This is intentional: setting `--physiological-weight` in the
current runner is **not** a substitute for class-conditional typicality.

Before removing this gate, add focused tests that establish all of the
following:

- the held-out subject never contributes to a class-1 reference or threshold;
- the constrained loss has a nonzero gradient through the recurrent
  classifier embedding to `z_prime` when the discrepancy exceeds its
  threshold;
- a zero typicality weight exactly reproduces base CFO;
- base and constrained runs use an identical eligible-trial manifest;
- unsuccessful `p_target` runs are retained and aggregated;
- decoded validity is computed by decoding, re-encoding, and classifying the
  joint reconstruction;
- aggregation is invariant to Slurm task completion order and refuses partial
  or mixed-configuration runs.
- candidate selection prefers a class-1 iterate that reaches `p_target`, then
  a class-1 iterate that misses `p_target`, before falling back to a non-flip;
  within a tier it applies the paper's fixed proximity rule. This preserves
  genuine failed-`p_target` counterfactuals instead of silently selecting an
  easier non-flipped iterate.

## Freeze the estimands before running

### Data and model

- Dataset: DREAMER, 23 subjects, 18 trials per subject.
- Tasks: valence and arousal, analyzed separately.
- Labels: global threshold 3, with median labels retained, matching the v15
  training launchers.
- Input: 1 s windows, 128 Hz, zero overlap, `global_rms` normalization.
- Checkpoint: the uncalibrated `zero_shot_source_model` for the held-out
  subject from the selected SICModelv15 configuration. Never use a checkpoint
  calibrated on the held-out subject.
- Decoder: the learned joint v15 reconstruction.
- Target: class 0 to class 1 only.
- Eligible trial: true class 0 and original model prediction class 0. Create
  this manifest once per fold and reuse it for both objectives.

### Class-1 discrepancy and threshold

The implementation must match the equation in the paper. If that equation is
the diagonal-Gaussian discrepancy implied by the saved variational classifier,
use the classifier embedding `h` and define

```text
D_C1(h) = mean_j [((h_j - prior_mu[1,j]) / exp(prior_log_sigma[1,j]))^2].
```

Do not apply this formula directly to the pre-recurrent four-dimensional
`window_features`; the learned class prior lives in the recurrent classifier
embedding space. Compute the fold threshold as

```text
tau_C1 = empirical quantile_q {D_C1(h_i): source subject, true class_i = 1}
```

with `q = 0.95` unless the Methods section specifies another value. All true
source class-1 trials belong in the calibration set; do not condition the
threshold on correct classification. Save the exact quantile convention,
source subject IDs, source trial IDs, checkpoint hash, input-data hashes,
preprocessing settings, and discrepancy formula in the reference artifact.

The constrained term should be a one-sided hinge,

```text
L_typ = relu(D_C1(h_cf) - tau_C1),
L_constrained = L_base + lambda_typ * L_typ,
```

unless the paper's stated objective differs. Tune `lambda_typ` and the common
CFO hyperparameters using source subjects only, then freeze them for all 23
held-out folds. Do not choose them from the final held-out results.

### Trial-level outcomes

Store separate booleans rather than overloading the word "success":

- `latent_flip`: the optimized latent is classified as class 1;
- `p_target_met`: its class-1 probability reaches the fixed target;
- `decoded_valid`: the decoded joint signal, after re-encoding, is classified
  as class 1;
- `typical`: `D_C1(h_cf) <= tau_C1`;
- `joint`: `decoded_valid and typical`;
- `phys_all_pass`: every preregistered physiological check passes.

The table's `Valid (%)` is the percentage with `decoded_valid`; `Typ. (%)` is
the percentage with `typical`; and `Joint (%)` is the percentage with `joint`.
All percentages use **every eligible trial** as the denominator, including
runs that flip class but fail `p_target`. Report `p_target_met` separately and
map condition `cfo_condition` to an explicit Boolean expression in the
summarizer. The equation itself is not present in this checkout, so that
mapping is a required manuscript decision before the final run.

Use these distance fields consistently:

- `d_z`: elementwise MSE between `z_prime` and `z` (the current runner's
  `selected_losses.latent`);
- `Delta_dec`: elementwise MSE between the decoded counterfactual and the
  reconstruction of the original latent (`decoded_change_mse`);
- `E_rec`: elementwise MSE between the original input and its reconstruction
  (`original_reconstruction_mse`).

If the paper equations instead define Euclidean norms, change the code and this
runbook before running; do not relabel MSE as an L2 norm. For the table, report
the median trial distance for each cell and preserve its IQR in the generated
machine-readable results, even if the compact LaTeX table displays only the
median.

### Physiological checks

The current code emits one graded VCSC penalty. It does not produce "X of N
checks" or an all-checks pass rate. Before the paper run, either:

1. define a source-calibrated check suite with one Boolean per check and an
   explicit all-pass rule, then populate `phys_pass_count`, `phys_n_checks`,
   and `phys_all_pass`; or
2. revise the paper and table to report VCSC as a continuous diagnostic.

Do not count VCSC's internal electrode-pair/band components as independent
checks after seeing the results. Any thresholds must be estimated from source
trials only and saved per fold.

## Required output contract

The reference job must write one directory per task and fold:

```text
runs/paper_counterfactuals/EXPERIMENT/references/TASK/fold_SS/
├── class_1_reference.json
├── class_1_reference.npz
├── source_class_1_discrepancies.csv
└── heldout_real_discrepancies.csv
```

`class_1_reference.json` must contain at least `target_subject`,
`target_class`, `threshold`, `threshold_quantile`, `n_source_trials`, model and
data hashes, and the discrepancy definition. The NPZ stores the fixed tensors
needed by the differentiable loss. `heldout_real_discrepancies.csv` contains
every held-out trial and its true class; its class-1 rows are the input to the
subject-invariance analysis, while its class-0 rows support the observed-data
panel and negative control.

The optimization job must write:

```text
runs/paper_counterfactuals/EXPERIMENT/counterfactuals/TASK/fold_SS/
├── eligible_trials.csv
├── base/
│   ├── settings.json
│   └── subject_SS_trial_TT/{result.json,history.csv,counterfactual.npz}
└── typicality/
    ├── settings.json
    └── subject_SS_trial_TT/{result.json,history.csv,counterfactual.npz}
```

Every `result.json` additionally needs `D_C1_original`, `D_C1_counterfactual`,
`tau_C1`, the six booleans above, the three distances, original/final VCSC,
and the physiological check fields. Both objective directories must list the
same trial IDs and record the reference artifact's hash.

## 1. Prepare the cluster checkout and checkpoints

Run from the EEGProc repository root. The two configuration directories must
each contain `loso_zero_shot_models.json` and 23 v15 `.keras` checkpoints.

```bash
cd /path/to/EEGProc
source venv312/bin/activate

export CFO_EXPERIMENT_ID=paper_cfo_v1
export VALENCE_CONFIG_DIR=/absolute/path/to/valence/configuration_XXXX
export AROUSAL_CONFIG_DIR=/absolute/path/to/arousal/configuration_XXXX
export PROJECT_DIR="$PWD"
export VENV_DIR="$PWD/venv312"
export EEG_PATH="$PWD/datasets/dreamer_eeg.npy"
export LABELS_PATH="$PWD/datasets/dreamer_labels.npy"
```

Use one immutable experiment ID for both jobs. Never point the two objectives
at different checkpoint configurations. Record the git commit plus any dirty
diff alongside the results before submission.

If the complete v15 zero-shot checkpoint sets do not exist yet, produce them
with the existing GPU launchers before continuing:

```bash
sbatch \
  src/eegproc/deep_learning/joint_architectures/SICModelv15/SLURM_scripts/full_run_v15_valence.sh
sbatch \
  src/eegproc/deep_learning/joint_architectures/SICModelv15/SLURM_scripts/full_run_v15_arousal.sh
```

The valence launcher evaluates more than one configuration. Select the one
configuration by its prespecified mean zero-shot LOSO balanced-accuracy rule,
then set `VALENCE_CONFIG_DIR` to that configuration. Do not choose the model
configuration using the counterfactual outcomes. Confirm that each selected
manifest has exactly one `zero_shot_source_model` for each target subject and
that each checkpoint reports joint reconstruction enabled.

Run the focused tests and CLI preflight locally:

```bash
PYTHONPATH=src python -m pytest \
  src/tests/test_counterfactuals.py \
  src/tests/test_counterfactual_metrics.py \
  src/tests/test_counterfactual_typicality.py \
  src/tests/test_counterfactual_paper_summary.py -q

PYTHONPATH=src python -m eegproc.model_explainability.fit_class_typicality --help
PYTHONPATH=src python -m eegproc.model_explainability.run_counterfactuals --help
PYTHONPATH=src python -m eegproc.model_explainability.summarize_counterfactual_paper --help
```

## 2. Submit the source-only references

The array has 46 tasks: valence subjects 0--22 followed by arousal subjects
0--22. Each task loads one fold checkpoint and encodes the real data.

```bash
REFERENCE_JOB_ID=$(sbatch --parsable --export=ALL \
  src/eegproc/model_explainability/SLURM_scripts/run_paper_cfo_references.sh)
echo "$REFERENCE_JOB_ID"
```

The job also writes held-out real discrepancy values for both classes. Those
values are evaluation data and must never be used to revise the threshold.

## 3. Submit both CFO objectives

Set the frozen constrained-objective weight. The value below is an example,
not a justified paper setting.

```bash
export TYPICALITY_WEIGHT=1.0

CFO_JOB_ID=$(sbatch --parsable \
  --dependency="afterok:${REFERENCE_JOB_ID}" \
  --export=ALL \
  src/eegproc/model_explainability/SLURM_scripts/run_paper_cfo_population.sh)
echo "$CFO_JOB_ID"
```

Each of the 46 tasks runs base and constrained CFO sequentially for one fold,
so both conditions see the same checkpoint, GPU type, eligibility manifest,
and environment. The runner must evaluate the full step budget and retain
failed-`p_target` cases; do not pass `--stop-on-success`.

The launcher defaults mirror the current development runs:

```text
p_target=0.60, learning_rate=1.0, decay=0.95, max_steps=200,
target/latent/decoded/physiology weights=1.0/0.1/0.1/0.0.
```

These are operational defaults, not validated scientific choices. Override
them only with source-only tuning results, and keep the overrides identical
between base and constrained runs except for `TYPICALITY_WEIGHT`.

## 4. Validate completeness locally

First produce summaries in validation-only mode. This is intentionally not a
Slurm job.

```bash
RESULT_ROOT="$PROJECT_DIR/runs/paper_counterfactuals/$CFO_EXPERIMENT_ID"

PYTHONPATH=src python -m \
  eegproc.model_explainability.summarize_counterfactual_paper \
  "$RESULT_ROOT" \
  --validate-only \
  --expected-subjects 23 \
  --require-paired-objectives \
  --require-complete
```

The validator must reject missing folds, duplicate trials, empty eligible
folds, held-out IDs in source references, model/reference hash mismatches,
different eligible sets between objectives, nonfinite metrics, and mixed
settings. It should print the denominator for every reported rate.

## 5. Generate table values and manuscript text

```bash
mkdir -p "$RESULT_ROOT/paper"

PYTHONPATH=src python -m \
  eegproc.model_explainability.summarize_counterfactual_paper \
  "$RESULT_ROOT" \
  --expected-subjects 23 \
  --require-paired-objectives \
  --require-complete \
  --bootstrap-unit subject \
  --bootstrap-replicates 10000 \
  --seed 42 \
  --output-dir "$RESULT_ROOT/paper"
```

Required products are:

- `trial_metrics.csv`: one row per task, objective, subject, and eligible trial;
- `subject_metrics.csv`: per-subject denominators and rates;
- `table_counterfactual_results.tex`: the four table rows;
- `paper_values.json`: every scalar and its exact numerator/denominator;
- `paper_sentences.md`: generated replacements for the population,
  cross-subject-typicality, and subject-invariance placeholders;
- `subject_invariance.json`: held-out real class-1 coverage, discrepancy
  summaries, and clustered confidence intervals.

For the population table, use trial-level (micro) rates across all eligible
trials. For the cross-subject paragraph, compute the typicality rate within
each subject first and then report the median and IQR of the 23 rates. Count a
subject as improved only when the constrained rate is strictly greater than
the paired base rate; also report ties in `paper_values.json`.

For subject invariance, compare each held-out real class-1 discrepancy with
the source class-1 empirical CDF from the matching fold. Primary outputs should
include the held-out class-1 threshold-coverage rate per subject, the median
held-out empirical percentile per subject, their 23-subject median/IQR, and a
subject-clustered bootstrap interval. A value near the source expectation is
evidence consistent with transfer of class-1 typicality; it is not proof of
full subject invariance. Keep observed class 0 as a labeled negative control.

## 6. Select the single-trial example without cherry-picking

The manuscript currently does not say whether the single example is valence
or arousal. Select within one preregistered task; do not pool task distances,
whose scales may differ. The default candidate pool below is the constrained
objective with `latent_flip`, `decoded_valid`, and `typical` all true. If the
paper means another success rule, change it before inspecting the figures.

```bash
PYTHONPATH=src python -m \
  eegproc.model_explainability.select_counterfactual_example \
  "$RESULT_ROOT/paper/trial_metrics.csv" \
  --task valence \
  --objective typicality \
  --require latent_flip decoded_valid typical \
  --distance d_z \
  --rule nearest-median \
  --tie-break subject_id trial_id \
  --output "$RESULT_ROOT/paper/single_trial_example.json"
```

The selected JSON must provide the exact values for original/final class-1
probability, original/final `D_C1`, `tau_C1`, `d_z`, `Delta_dec`, `E_rec`,
original/final VCSC, and physiological passes. It should also point to the NPZ
and history CSV rather than copying or recomputing them.

## 7. Generate the figures locally

Resolve the chosen paths from `single_trial_example.json`, then run the
existing single-trial plotters:

```bash
TRIAL_DIR=/path/from/single_trial_example.json

PYTHONPATH=src python -m eegproc.model_explainability.counterfactual_heatmap \
  "$TRIAL_DIR/counterfactual.npz" \
  --branch joint --sampling-rate 128 --no-show \
  --output "$RESULT_ROOT/paper/single_trial_heatmap.png"

PYTHONPATH=src python -m eegproc.model_explainability.counterfactual_topography \
  "$TRIAL_DIR/counterfactual.npz" \
  --branch joint --no-show \
  --output "$RESULT_ROOT/paper/single_trial_topography.png"

PYTHONPATH=src python -m \
  eegproc.model_explainability.counterfactual_training_monitor \
  "$TRIAL_DIR/history.csv" \
  --no-show \
  --output "$RESULT_ROOT/paper/single_trial_trajectory.png"
```

Generate the population discrepancy and per-subject panels from the tidy CSVs:

```bash
PYTHONPATH=src python -m \
  eegproc.model_explainability.plot_counterfactual_population \
  --trial-metrics "$RESULT_ROOT/paper/trial_metrics.csv" \
  --subject-metrics "$RESULT_ROOT/paper/subject_metrics.csv" \
  --references-root "$RESULT_ROOT/references" \
  --output "$RESULT_ROOT/paper/population_cfo.png" \
  --no-show
```

The discrepancy panel must show four explicitly labeled distributions for
each task: observed held-out class 0, base counterfactual, constrained
counterfactual, and observed held-out class 1. Show fold-specific thresholds
in normalized form (for example `D_C1 / tau_C1`) before pooling folds; a raw
pooled vertical threshold is invalid because `tau_C1` differs by fold. The
observed class-0 curve should use the original trials from the same eligible
manifest as the counterfactual curves; observed class 1 uses every true
held-out class-1 trial. The per-subject panel should show paired
base/constrained rates and retain subjects with zero successes.

## 8. Final audit

Before replacing any `XX`, archive the following together:

- git commit and dirty patch;
- Slurm job IDs and logs;
- checkpoint manifests and SHA-256 hashes;
- data hashes and preprocessing configuration;
- source-only hyperparameter-selection record;
- reference artifacts, eligibility manifests, and all per-trial results;
- generated CSV, JSON, Markdown, TeX, and figures.

Check that the prose says "percentage points" for differences between rates,
that each rate has a visible numerator and denominator in `paper_values.json`,
and that class flipping, reaching `p_target`, decoded validity, typicality, and
physiological passage are never treated as interchangeable outcomes.
