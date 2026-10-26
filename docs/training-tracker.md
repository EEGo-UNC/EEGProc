# Training tracker

The tracker keeps a local, auditable record of training experiments. It suggests
the next configuration, creates a run bundle and launcher, imports downloaded
results, tracks status, ranks results, and generates figures. Training runs on
your chosen machine or cluster; tracking and plotting run on your laptop.

## Install and start

From the EEGProc checkout, with Python 3.10 or newer:

```sh
python -m pip install -e .
training-tracker --campaign experiments/dreamer_3dcnn/campaign.json status
```

`python -m training_tracker` is equivalent to `training-tracker`. From an
uninstalled checkout, set `PYTHONPATH=src`, or continue using the compatible
`python experiments/dreamer_3dcnn/tracker.py` entry point. The core commands use
only the standard library and do not import TensorFlow or plotting packages.
The normal EEGProc installation supplies the dependencies for plotting. Only
the machine doing training needs the model's training dependencies.

## Everyday workflow

Global options (`--campaign`, `--runs-dir`) go **before** the command. Each
campaign uses a `runs` directory beside its campaign file unless overridden.

```sh
# Inspect the proposal without creating a run.
training-tracker --campaign experiments/dreamer_3dcnn/campaign.json suggest --target valence

# Create config.json, manifest.json and submit.slurm.
training-tracker --campaign experiments/dreamer_3dcnn/campaign.json plan --target valence
```

Transfer the new bundle to the same experiment directory in your cluster
checkout. Run the printed `sbatch` command from the cluster's repository root.
The tracker **does not submit jobs or transfer files over SSH**. Download the
run directory after or during training, then import it:

```sh
training-tracker --campaign experiments/dreamer_3dcnn/campaign.json sync /path/to/downloaded/runs
training-tracker --campaign experiments/dreamer_3dcnn/campaign.json status
training-tracker --campaign experiments/dreamer_3dcnn/campaign.json rebuild
```

`sync` accepts one bundle or a directory containing multiple bundles. It checks
the campaign and configuration fingerprint, refuses conflicting configurations,
and copies top-level result, manifest, log and other files. Nested checkpoint
directories are not imported. Repeated imports update the local snapshot.
`index.csv` and `REPORT.md` are derived reports and can be rebuilt at any time.

The training script owns the manifest status: `planned`, `running`, `completed`
or `failed`. A valid `result.json` is authoritative evidence of completion;
unreadable or malformed results appear as `invalid` with an error in status and
the CSV. There is no live scheduler polling: sync a newer manifest for updated
job status. The existing DREAMER trainer already writes status and failure logs.

Suggestions follow the existing strategy: a cheap smoke run, then the full
baseline, then one-factor changes around the best completed full run. Planned
or running runs block another proposal for the same target. Ranking maximizes
`mean - penalty * std`, or `-mean - penalty * std` for minimization. This is a
transparent search heuristic, not Bayesian optimization or statistical evidence.
Failed configurations remain recorded and are not automatically resubmitted;
inspect the failure and explicitly rerun the existing bundle on the cluster.
A partially written plan's launcher is repaired by the next `plan` command.

For the transformer, use the same commands with:

```text
experiments/dreamer_3dcnn_transformer/campaign.json
```

Existing configurations, fingerprints, bundles and launcher behavior remain
compatible. Both DREAMER campaigns explicitly declare `valence` and `arousal`.

## Generate figures

```sh
training-tracker --campaign experiments/dreamer_3dcnn/campaign.json plot --chance 0.5

# Include both architectures, keeping targets and stages separate.
training-tracker --campaign experiments/dreamer_3dcnn/campaign.json plot \
  --compare-campaign experiments/dreamer_3dcnn_transformer/campaign.json \
  --output-dir comparison-figures --format svg --chance 0.5
```

Use `--run-id ID` repeatedly to select particular runs. A comparison campaign
reads the `runs` directory beside its campaign file. All campaigns must use the
same objective metric. `--runs-dir` applies to the primary campaign.

The output directory contains:

| File | Purpose |
| --- | --- |
| `*-subjects.png` (or SVG/PDF) | Actual subject IDs and scores, with the cohort mean |
| `*-summary.*` | Saved mean and standard deviation, with subject scores when available |
| `*-comparison.*` | Run comparison on the intersection of recorded subject IDs |
| `runs.csv`, `subjects.csv` | Exact source tables, full run and campaign identities |
| `plots.json` | Input result paths, objective, reference line, figures and skipped-panel reasons |
| `README.md` | Figure gallery and interpretation notes |

The panels reuse the repository's existing `result_figures.py` functions for
subject accuracy, metric summaries and model comparison. They were brought from
the sibling `sic-band-fix` checkout into this checkout's `eegproc.plotting`
package; prediction-only panels and their dependencies were not needed.

The current DREAMER trainers log `user_metrics` even with prediction logging
disabled, so subject plots work with their existing `result.json` outputs.
Trial and window metrics stay distinct. A summary-only result still produces
a summary plot; unavailable subject panels are explained in `plots.json`.
Repeated subject IDs must be aggregated explicitly before plotting.

`--chance` supplies a descriptive reference in the objective's units; it is
optional and never inferred from a model name. A value of 0.5 may be appropriate
for a binary balanced-accuracy reference, but raw accuracy depends on class
balance. Error bars show between-subject SD, **not confidence intervals**.
Comparison bars use only shared subjects, while individual summaries retain
their original cohort. Subject ID equality alone does not establish matching
datasets or evaluation protocols; ensure those match before comparing.
Visible patterns do not establish non-randomness. Formal inference needs an
appropriate null/permutation design, independent evaluation and correction for
multiple comparisons. Scores used for hyperparameter selection are not an
unbiased final test estimate.

## Try it without Longleaf or real run outputs

Run from the repository root:

```sh
python -m training_tracker.demo --output-dir tracker-demo
training-tracker --campaign tracker-demo/model-1/campaign.json status
training-tracker --campaign tracker-demo/model-1/campaign.json plot \
  --compare-campaign tracker-demo/model-2/campaign.json \
  --output-dir tracker-demo/figures --chance 0.5
```

This creates both model campaigns, both targets and 23 invented subject scores
per run using seed 42, in the real LOSO result format. Campaign and run IDs are
marked `SYNTHETIC`. The generator refuses an existing output directory, keeping
fake data separate from real run stores. These values are test data, never
scientific results. Installed wheels without the example campaigns can supply
source campaigns using repeated `--campaign PATH` arguments to the demo command.

## Add any other model

Copy `experiments/tracker_template` into your experiment directory and edit:

1. `campaign`: a unique campaign name; `targets`: your task names.
2. `defaults`: arbitrary JSON parameters accepted by your training script,
   including `target`. Model architecture is not interpreted by the tracker.
3. `objective`: metric name, `maximize`, and optional nonnegative
   `uncertainty_penalty` (default zero).
4. `smoke_overrides`: cheap settings for a preliminary run. Omit it to begin
   directly with the full baseline. Overrides should differ from defaults.
5. `search_space`: parameter names present in defaults and lists of values.
6. `experiment_dir`, `training_script`, and `launcher`.

The template launcher is a normal shell script under your control. Its only
substitutions are `{{RUN_ID}}`, `{{EXPERIMENT_DIR}}` and `{{TRAINING_SCRIPT}}`.
Relative template paths resolve beside the campaign file. Configure your own
resources, dependencies, data validation and training arguments there. Templates
are trusted executable project files; planning only writes them. The built-in
`{"type": "longleaf"}` adapter retains the existing DREAMER/CUDA setup and
requires `longleaf` resource settings. Use `type: template` for other models or
sites so those DREAMER-specific data checks do not apply.

Make your training script read the bundle's configuration, preserve its manifest
identity/fingerprint, write running/failed/completed status, and write a
`result.json` with this framework-independent schema:

```json
{
  "metrics": {
    "loss": {"mean": 0.25, "std": 0.05}
  },
  "aggregation_unit": "subject",
  "subject_metrics": [
    {"subject_id": "S01", "loss": 0.20},
    {"subject_id": "S02", "loss": 0.30}
  ]
}
```

Metric names must exactly match `objective.metric`. Values must be finite and
standard deviations nonnegative; use zero when reporting a single observation.
`subject_metrics` is optional and supports classification or regression metrics.
`aggregation_unit` is optional; use it to label summary SD across `subject`,
`seeds`, `folds` or another evaluation unit. Without subject scores or native
LOSO output, the default label is `evaluations`.
Report values in consistent units, and document the aggregation used for mean
and SD. No custom Python adapter is needed for a new model that writes this
schema. EEGProc's native `config_results[best_config_index]` and `user_metrics`
format is also supported directly.

Keep planned `config.json` immutable: changes invalidate its fingerprint.
Keep downloaded bundles as the reproducible record of what ran, including the
training code revision and environment where possible. The existing DREAMER
trainer records those in its manifest.

## Design and maintenance

| Module | Responsibility |
| --- | --- |
| `training_tracker.core` | Bundle persistence, fingerprints, scan/status, planning, suggestions, sync, CLI |
| `training_tracker.results` | Canonical and EEGProc result adapters and metric validation |
| `training_tracker.launchers` | Scheduler-specific Longleaf adapter and user templates |
| `training_tracker.plotting` | Lazy plotting imports, cohort grouping, report artifacts and provenance |
| `training_tracker.demo` | Deterministic synthetic fixtures for both architectures |
| `eegproc.plotting.result_figures` | Reusable DataFrame-to-figure panels |

Bundles are the source of truth; reports and figures are disposable views.
There is no database, service, cloud account or training-framework dependency
in the tracking core. The console entry point is installed through the existing
package. The old experiment-local script delegates to the package, avoiding two
implementations of planning/sync logic.

This is a single-writer file store. Do not plan or sync concurrently into the
same store. JSON replacement is atomic, but an entire bundle import is not a
transaction. Sync only complete download snapshots. Arbitrary nested training
artifacts are outside the bundle-sync contract.

To move this into another repository, preserve the `training_tracker` package,
its CLI entry point and tests; keep EEGProc available for the shared plotting
panels or move those panels together under the repository's license. The core
does not need EEGProc imports. A separate public release still needs normal
project review and license/packaging decisions; this change does not publish
anything or choose a new license.

Validate with:

```sh
python -m pytest src/tests/test_training_tracker.py src/tests/test_dreamer_3dcnn_tracker.py
```
