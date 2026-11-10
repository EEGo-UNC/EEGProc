# Separately regularized trial identity and six-segment EEG

```bash
sbatch experiments/sic_v16/compare_block_regularization.slurm
```

Uses the newest completed verified centred_full_filter rebuild. Override with
REBUILD_DIR=/absolute/path/to/crop-rebuild-comparison/job-JOBID if desired.

Arms are trial ID only, and trial ID plus six ordered segment EEG features
with independent L2 penalties. Both use the same 23 outer subject folds,
22 inner subject validation folds, preprocessing, scaling and threshold grid.
Trial C spans 0.001, 0.01, 0.1, 1, 10, 100. EEG C spans zero plus those values.
Zero EEG drops every EEG feature, exactly recovering the trial-only candidate.
After training-only StandardScaler, EEG columns are multiplied by
sqrt(EEG_C/trial_C). With logistic regression C=trial_C this implements separate
block penalties in standardized feature coordinates. No fitted scaling uses
the inner validation or outer test subjects.

Select the highest subject-mean inner OOF balanced accuracy. Exact ties prefer
zero EEG, smaller trial C, smaller EEG C, a threshold nearer 0.5, then a lower
threshold. Refit on all 22 non-test subjects, and evaluate the held-out subject.
The outer test never selects penalties, threshold or zero/nonzero EEG. A zero
candidate permits validation to suppress EEG but does not guarantee that the
selected procedure cannot worsen held-out performance.

The adaptive arm fits 42 penalty pairs rather than six; the search is larger
than the previous comparison. The job requests 24 hours. The report includes
per-subject selected penalties, zero-EEG selection, fixed/selected thresholds,
inner predictions and differences versus the trial-only reference.

Send block-regularization-comparison-diagnostics.tar.gz from the printed path.
