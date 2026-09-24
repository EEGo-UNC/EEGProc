# Final ICLR counterfactual report

Run this from the **original EEGProc repository** after placing the archived
four-arm results under `runs/counterfactuals/final-ICLR`:

```bash
PYTHONPATH=src python -m eegproc.model_explainability.report_iclr_ablations \
  runs/counterfactuals/final-ICLR \
  --out-dir runs/counterfactuals/final-ICLR-report
```

The command discovers `study.json` files in task directories and `fold_*`
shards. It expects all four objectives (`target_latent`, `base`, `typicality`,
and `typicality_no_physiology`) and user IDs 0–22 for each task. Set
`--expected-subjects N` only for a deliberately smaller study with IDs
`0` through `N-1`. It refuses duplicate task/user shards, mixed typicality
definitions, mixed validity protocols, different confidence thresholds within
a task, and missing objective arms.

Outputs are `trial_optimizations.csv`, `user_optimizations.csv`,
`population_ablations.csv`, `report.json`, `counterfactual_results.tex`,
`counterfactual_results_with_provisional.tex`, and
individual `users/<task>_user_<id>.md` reports. Each task has four rows in the
LaTeX table, including typicality without the physiological penalty. The table
contains displacement, VCSC pass rate, and the full physiological pass rate.
Flip, confidence acquisition, and typicality
remain in the CSV and per-user reports for the accompanying prose. The
`report.json` file records missing user shards and whether the report is
complete. An optimization error remains in the denominator for percentages;
distances use finite selected endpoints and show median [Q1, Q3].
If a task is missing a user shard or an optimization is pending, its LaTeX
values stay `--` until the report is complete for that task. The CSV retains
provisional rates with an explicit flag. The second LaTeX file shows archived
partial-task values with a dagger and an explicit provisional caption.

**Metric space matters.** `Flip` means target-class argmax. `Conf.` means the
target-class probability reached the archived run threshold; the threshold is
recorded in `report.json` and per-trial CSV rows. `Flip+Typ.` requires a flip
and class-1 typicality; `Conf.+Typ.` additionally requires confidence
acquisition. Current ICLR `latent_only` archives evaluate these on the
optimized latent classification embedding. They do not re-encode decoded
counterfactual signals, so decoded validity cannot be inferred. Historical
round-trip archives, if all inputs use that protocol and preserve decoded
predictions, produce decoded/re-encoded metrics instead. The fifth
physiological check, aperiodic exponent, cannot be estimated from the
band-filtered decoder. An endpoint that fails any measured required check
therefore fails the full set; an endpoint passing all four measured checks has
an unknown full-set result. The four-check rate remains available separately
in the CSV. `VCSC` compares the
saved raw penalty against the saved tolerance even for objectives whose VCSC
optimization weight was zero. These are separate diagnostics.

## Real class-1 subject-invariance analysis

From the same EEGProc root, use the archived real-trial embeddings and each
fold's source-trained class-1 Gaussian:

```bash
python src/eegproc/model_explainability/report_iclr_subject_invariance.py \
  runs/counterfactuals/final-ICLR \
  --samples-per-subject 3 --seed 42 \
  --out-dir runs/counterfactuals/final-ICLR-subject-invariance
```

The input can contain task directories and `fold_*` shards. The command samples
all **true class-1** real trials, including misclassified trials, independently
of their discrepancy. It writes trial scores, per-fold source versus held-out
summaries, task-level summaries, a sampling manifest, and a ready-to-review
`subject_invariance_paragraph.tex`. The trial table includes each held-out
score's empirical percentile among all available true class-1 source trials
in the matching fold. Task-level medians, quartiles, and subject-bootstrap
intervals summarize coverage and
percentiles across folds. Held-out true class-0 trials are included as a
separately labeled negative control. Each held-out subject is scored only
against its own fold's frozen class-1 Gaussian. Fold contrasts
are normalized by the source-calibrated threshold before task-level summary;
raw scores from different LOSO models are not pooled. Set
`--samples-per-subject 0` to use every available true class-1 trial.

This requires `calibration/region.json`, `calibration/source_trials.json`,
and `observations.json` inside each `subject_*` fold directory. Historical
NPZ summaries also work. Compact archives omit embeddings, so individual
scores cannot be independently recomputed offline; the report validates
the saved scores and the source-calibrated threshold. Counterfactual result
files alone contain no real-trial source reference bank.
