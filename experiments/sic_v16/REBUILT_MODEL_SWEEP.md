# Four-model sweep on the verified rebuild

From the Longleaf repository root:

```bash
bash experiments/sic_v16/submit_rebuilt_model_sweep.sh
```

Defaults to the newest completed crop rebuild. Pin the verified rebuild with:

```bash
REBUILD_DIR=/work/users/p/m/pmucharl/sic-v16/crop-rebuild-comparison/job-8059671 bash experiments/sic_v16/submit_rebuilt_model_sweep.sh
```

Preparation verifies rebuilt EEG/label hashes and prepares shared normalized
arrays once. The source is centred_full_filter; original data is preserved.

Four learning checks use the same deterministic four trials per class from
source subjects 2/3, up to 1000 epochs, learning rate 0.001, and no dropout or
weight decay. These are training-only memorization checks, not validation
estimates. Each architecture must classify all eight correctly with inference
cross-entropy <=0.1. The reconstruction/Transformer switches define the four
variants; reconstruction weight is 0.1. Full training is dependent on all four
checks passing and checks gate records again inside each task. If a gate
fails, the full array remains blocked; inspect its logs and partial archive
rather than bypassing the gate. Partial diagnostics are collected after checks.

Full training has 92 tasks (4 models x 23 test subjects), at most four active,
24 hours per task. Each fold uses 20 training subjects, two whole validation
subjects and one test subject, fixed seed 42 plus test subject for model RNG.
All variants share folds and source data. Default training budget is 500 epochs,
learning rate 0.001, dropout 0.1 for encoder/classifier, zero weight decay and
ordinary cross-entropy. This is a matched architecture comparison, not a
hyperparameter sweep. Regularization is deliberately lighter than old runs.

Select the highest mean validation-subject balanced accuracy; break ties by
lower validation cross-entropy, then earliest epoch. After at least 200 epochs,
stop if selection has not improved for 100 epochs. Save only best.keras.
Training inference metrics and validation probabilities are recorded every
epoch. Test predictions are evaluated once from the selected checkpoint;
test labels never select epochs or stopping. Threshold is fixed at 0.5.

The final summary requires all 92 folds. It emits
rebuilt-four-model-diagnostics.tar.gz without checkpoint/data bytes.
If a task fails or times out there is no automatic training resume in this
new runner. Keep the sweep directory, inspect logs and collect partial JSONs:

```bash
python experiments/sic_v16/rebuilt_model_sweep.py collect --root /absolute/sweep/directory
```

Use the established cluster environment when invoking Python directly.
The helper prints all job IDs and the sweep directory. Logs are
sicv16-rebuilt-four-model-JOBID.out/.err. Array tasks have unique Slurm job IDs.
A failed prerequisite may leave dependent jobs pending; cancel them if ending
that sweep. To set budgets before preparation, export SWEEP_EPOCHS or
CHECK_EPOCHS when invoking the submission helper. The preparation provenance
fixes those budgets for the entire sweep.
