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
definitions, mixed validity protocols, and missing objective arms.

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

**Validity space matters.** Current ICLR `latent_only` archives evaluate the
target class and typicality on the optimized latent classification embedding.
They do not re-encode decoded counterfactual signals, so decoded validity and
decoded Joint cannot be inferred. The generated caption labels these as latent
metrics. Historical round-trip archives, if all inputs use that protocol,
produce decoded/re-encoded metrics instead. `Phys.` is `--` when a required
physiological check was unavailable; available-check rates remain in the CSV.
