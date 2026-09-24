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
`population_ablations.csv`, `report.json`, `counterfactual_results.tex`, and
individual `users/<task>_user_<id>.md` reports. Each task has four rows in the
LaTeX table, including typicality without the physiological penalty. The
`report.json` file records missing user shards and whether the report is
complete. An optimization error remains in the denominator for percentages;
distances use finite selected endpoints and show median [Q1, Q3].
If a task is missing a user shard or an optimization is pending, its LaTeX
values stay `--` until the report is complete for that task. The CSV retains
provisional rates with an explicit flag.

**Metric space matters.** `Flip` means target-class argmax. `Conf.` means the
target-class probability reached the archived run threshold; the threshold is
recorded in `report.json` and per-trial CSV rows. `Flip+Typ.` requires a flip
and class-1 typicality; `Conf.+Typ.` additionally requires confidence
acquisition. Current ICLR `latent_only` archives evaluate these on the
optimized latent classification embedding. They do not re-encode decoded
counterfactual signals, so decoded validity cannot be inferred. Historical
round-trip archives, if all inputs use that protocol and preserve decoded
predictions, produce decoded/re-encoded metrics instead. `Phys.` is `--` when a
required physiological check was unavailable; available-check rates remain in
the CSV.
