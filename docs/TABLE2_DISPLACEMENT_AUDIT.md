# Table 2 displacement audit

The saved-result displacement audit is implemented and completed for the
original DREAMER suites (2026-11-10). It rebuilds Table 2, separates two latent
coordinate systems, separates reconstruction error from the decoded edit, and
exports the paired comparisons, subject summaries, cohort counts and figures.
No training, counterfactual regeneration, or decoded-validity evaluation is
needed for this audit.

## Reproduce the original Table 2 audit

Run from the repository root with the existing analysis dependencies:

```bash
PYTHONPATH=src python -m eegproc.model_explainability.typicality.displacement_audit \
  runs/counterfactuals/arousal_suite_1599318_ablations_2355492 \
  runs/counterfactuals/valence_suite_65452590_ablations_2356028 \
  --expected-subjects 23 --require-complete \
  --reference-population runs/counterfactuals/final-ICLR-report/population_ablations.csv \
  --out-dir runs/table2_displacement_audit
```

The command accepts task/fold/direction study roots and recursively finds their
`study.json` files. Run different historical or revised protocols in separate
output directories. It rejects duplicate shards, mixed source/data/protocol
fingerprints, inconsistent frozen checkpoint identities and changed committed
endpoint artifacts. Current 10th-percentile settings are not applied to the
historical 95th-percentile generation study.

`--no-figures` skips PNG/PDF generation. `--require-complete` returns an error
for missing folds, pending endpoints or missing latent/signal measurements,
after saving the provisional audit. Invalid numeric inputs fail explicitly;
triangle-bound and reference-summary mismatches are exported and cause a
nonzero command exit status.

## Distances and their references

- `d_z`: RMSE between the original and optimized full-trial classification
  embeddings. Its coordinates are the frozen classifier's representation.
- `decoder_latent_rmse`: RMSE in the latent coordinates optimized by the
  decoder objective. It is not interchangeable with `d_z`.
- `e_rec = sqrt(original_reconstruction_mse) = RMSE(R(z), x)`.
- `delta_dec = sqrt(decoded_change_mse) = RMSE(R(z_cf), R(z))`.
- `e_cf = sqrt(counterfactual_to_original_mse) = RMSE(R(z_cf), x)`.

The normal study report now also exports `decoder_latent_rmse` and `e_cf`,
including finite sample counts, medians and quartiles. Missing historical
measurements remain unavailable rather than zero.

Signal distances use the archived model-input normalization. The audit tests
`abs(e_rec-delta_dec) <= e_cf <= e_rec+delta_dec` with numerical tolerance,
and checks the same original reconstruction is used across paired arms.
These distances are not additive and do not establish what caused an edit.

## Outputs

- `report.md`: completed findings, interpretation, scope and the case-study
  MSE/RMSE correction; `paper_text.md` / `paper_text.tex`: reusable paragraphs.
- `table2.csv` / `table2.tex` / `table2.png` / `table2.pdf`: the rebuilt main
  table, preserving small nonzero distances instead of rounding them to zero.
- `table2_signal_errors.csv` / `table2_signal_errors.tex`: the companion table
  of decoder-latent displacement and the three signal errors.
- `trial_displacements.csv`, `population_displacements.csv`, and
  `subject_displacements.csv`: complete available distances and their
  distributions, including tails and exact-zero versus rounded-zero counts.
- `paired_displacements.csv` / `paired_summary.csv`: within-trial changes
  relative to CFO; medians of paired changes are distinguished from changes
  between marginal medians.
- `latent_decoded_correlations.csv`: within-objective descriptive Spearman
  associations; no significance claim across clustered trials.
- Per-task distribution, latent/decoded scatter and signal-error figures in
  PNG and vector PDF formats.
- `cohort_summary.csv`, `cohort_folds.csv`, and `exclusions.csv`: the source
  cohort and each held-out fold, retaining zero-eligible participants.
- `reference_checks.csv` and `audit.json`: archived-summary comparisons,
  verification results, input hashes, generator metadata and audit-source
  fingerprints.

The LaTeX tables use `graphicx`'s `resizebox` to fit the available text width.

All finite completed endpoints are included, even if their latent target
attempt was unsuccessful. Errors and pending attempts retain their eligible
cohort membership and have unavailable distances. Each metric has its own
finite count; VCSC is the archived penalty-tolerance check among assessed
endpoints, not a complete physiological acceptance claim.

## Scope of the completed original-data audit

The original suites contain 596 objective endpoints: 57 arousal and 92 valence
trial identities, each evaluated under four objectives, over 23 folds per task.
All 596 signal-distance triplets satisfy the triangle bounds. Their archived
MSE-to-RMSE conversions, paired starting reconstructions and original Table 2
summaries can be verified from the saved files without loading EEG arrays.

The classification-embedding and signal metrics rank some objectives
oppositely: removing physiology produces larger classification-embedding
changes but much smaller decoded edits. Report both latent metrics with their
coordinate definitions. For the archived arousal subject-0/trial-16 typicality
case, the correct RMSE values are approximately 0.726503, 0.470274 and 0.872273;
the previously printed 0.528, 0.221 and 0.761 match the stored MSEs.

This verifies saved scalars and archive consistency, not independent
recomputation of distances from the original signal arrays. Keep these
historical results separate from experiments using revised reconstruction.

## Separately deferred or optional work

Decoded-validity evaluation is deferred until reconstruction incorporates
subject information, as requested. This audit does not re-encode signals,
evaluate decoded target predictions, or change reconstruction/training code.

A target-free physiological control is optional follow-up for interpreting
why the decoded signal moved. It is not a missing prerequisite for Table 2's
displacement audit, and no new control was generated here.
