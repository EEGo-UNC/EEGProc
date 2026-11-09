# Merge of vcsc-repair-baseline into CFX-update

Base: `4ffd6877` (CFX-update). Incoming: `fe9083fa`
(vcsc-repair-baseline, including the target-weight-zero control).

The student's `counterfactuals/loss.py` and `counterfactuals/optimizer.py`
are retained without changes. The repair baseline, real-trial VCSC tolerance,
amplitude/spectral objective, temporal smoothing, envelope distance, graded
physiology, re-encoded EEG evaluation, and target-weight-zero option are kept.

## Conflicts and resolutions

Git reported conflicts in nine files:

| Files | Resolution |
| --- | --- |
| `counterfactuals/arguments.py` | Keep the current shared argument validators. CFX-update had removed the standalone runner and parser. The active `typicality.runner` parser accepts `--target-weight 0` and all incoming options. |
| `typicality/runner.py` | Combine the incoming baseline, normalization, temporal, and re-encoding paths with bidirectional eligibility, per-direction calibration, and independent encoded-neighbor evaluation. Replace incoming class-1 constants with the selected target class. Keep compact JSON components and their axis metadata. |
| `typicality/grade_physiology_table.py` | Keep JSON, embedded-component, and historical NPZ grading with the bounds-validation gate. Add the incoming default `--which all`, real-EEG/reconstruction rows, and available-family pass rate. Deduplicate baseline copies within each trial; explicit single-signal reports retain objective grouping. |
| `typicality/README.md` | Keep JSON grading instructions and add the incoming repair and temporal/phase documentation. |
| `pyproject.toml` | Keep the current package version, supported Python versions, and base/optional dependency layout. Carry the incoming joblib cap into the deep-learning extra. |
| Both `run_cfo_ablations_*` launchers and `test_cfo_ablation_launcher.py` | Restore the incoming scripts and tests. Adapt the launchers to the available SICModelv15 registration module and the current 0.10 typicality quantile. Keep both directions. |

Two additional compatibility adjustments were needed after Git's automatic
merge: result tables must read the selected target-class probability rather
than always class 1; the incoming typicality-vacuity audit must discover nested
reports and keep directions separate. Legacy audit inputs retain their
previous per-task output keys. Explicit-direction inputs use keys such as
`valence/0_to_1`, and the CSV includes a `direction` column.

## Running after the merge

- Use a new output root. Source/protocol fingerprints change, so existing runs
  intentionally fail resume validation rather than mixing protocols.
- The CFX-update defaults remain both directions and the source target-class
  10th-percentile typicality cutoff. The student's earlier 0.95 setting is not
  restored; explicitly passing it still fails the current parser's validation.
- The repair, amplitude/spectral, temporal, and envelope options remain opt-in.
  `--target-weight 0` removes the target-loss contribution; target success is
  still measured and reported.
- Enabling `--include-repair-baseline` saves one `repair_reconstruction.npz`
  per eligible trial per direction for comparisons and resume. Other arms
  retain the current compact JSON/CSV archive format.
- The grading table now includes real EEG and reconstruction by default.
  Use `--which counterfactual` for an objective-only table.

Validation uses the existing test suite plus merge-specific checks for both
directions with repair, temporal/envelope terms, target weight zero, retries,
resume, neighbor evaluation, baseline deduplication, and nested audit reports.
Synthetic saved models exercise loading and optimization. Actual cluster
checkpoints, production EEG, and Slurm execution require a separate experiment
run; this merge does not establish their scientific outcomes.

### Validation results

- All 103 unique merge-related checks passed, including the final audit
  regression checks.
- Full suite: 430 passed, 7 subtests passed, 6 skipped, 5 failed.
- The five failures exactly match failures reproduced on untouched
  CFX-update at `4ffd6877`. No additional failures were introduced.
- With the incoming joblib cap satisfied by joblib 1.5.3, the cross-validation
  checks passed: 36 passed, 6 skipped. The existing environment was not modified.
- Both restored launchers pass Bash syntax validation; the merge passes
  Git's whitespace/conflict-marker check.

The pre-existing failures are:

| Check | Cause on the untouched base |
| --- | --- |
| `test_imported_launchers_parse_and_calibration_help_works` | References the removed SICModelv11 calibration launcher. |
| `test_valence_array_task_has_a_valid_calibration_selection` | Expects an obsolete inline Python configuration block in the valence launcher. |
| `test_source_training_calibration_and_checkpoint[SICModel]` | Imports the removed SICModel package. |
| `test_source_training_calibration_and_checkpoint[SICModelv11]` | Imports the removed SICModelv11 package. |
| `test_calibration_only_loads_version_in_fresh_workers[SICModelv11]` | Imports the removed SICModelv11 package. |
