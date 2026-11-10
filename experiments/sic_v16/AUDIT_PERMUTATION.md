# Alignment audit and nested permutation control

From the repository root on Longleaf:

```bash
bash experiments/sic_v16/submit_audit_permutation.sh
```

This submits an audit/preparation job, a dependent array (original labels plus
100 within-subject permutations, at most four tasks at once), and a dependent
summary job. Each task repeats all 23 outer folds and full inner C/threshold
selection. The summary requires every task to complete and reports upper-tail
Monte Carlo p-values using (1 + exceedances)/(1 + permutations), for selected
and fixed-0.5 thresholds. Trial labels are shuffled once per subject per
permutation, consistently across every outer fold, including test labels.
This assumes within-subject trial exchangeability under the null; temporal
or shared-stimulus dependence can limit the interpretation.

For a small pipeline check (coarse p-values only):

```bash
PERMUTATIONS=5 bash experiments/sic_v16/submit_audit_permutation.sh
```

The printed work directory contains audit.json immediately after preparation,
run-0 for the original-label result, and run-1 onward for null results. The
summary prints the path to audit-permutation-diagnostics.tar.gz. Logs are
sicv16-audit-permutation-JOBID.out/.err in the submission directory. If a task
fails, inspect these logs; the summary will not run on incomplete results.

The audit verifies array dimensions, prepared subject/trial identity, label
conversion from dimension 0 using >=3, finite values, trial duplicates and
constant signals. It records source array hashes and per-trial ratings.
It cannot prove original stimulus alignment, the semantic meaning of label
dimension 0, sampling frequency, or retained segment location without the
original export records. These are explicitly recorded as unresolved.
