# Four controlled classifier-collapse diagnostics

From the repository root:

```bash
REBUILD_DIR=/work/users/p/m/pmucharl/sic-v16/crop-rebuild-comparison/job-8059671 bash experiments/sic_v16/submit_classifier_collapse.sh
```

Runs the plain CNN on one fixed full-training fold: test subject 0 is excluded,
two validation subjects and twenty training subjects, exactly the original
fold_subjects rule and seed 42. No test predictions are evaluated in these
runs. The four arms change one factor relative to the original reference:

0. Reference: dropout 0.1, ReLU, unweighted loss.
1. No dropout: encoder/classifier dropout zero.
2. LeakyReLU only in the 128-unit classifier, negative slope 0.01.
3. Inverse-frequency class weights from training labels only. Weight reduction
   uses mean(weight*loss) so weights remain effective for batch size one;
   the existing normalized weighted-mean path would cancel those weights.

Learning rate 0.001, zero weight decay, no attention or reconstruction, source
batch size one and all other settings are matched. Each arm runs 200 full
epochs without early stopping so trajectories have the same budget. Set
SWEEP_EPOCHS when submitting to override. The shared preparation fixes the
budget. Up to four tasks run at once; each requests 24 hours.

Initial and per-epoch inference metrics cover all training and validation
trials: probability spread and single-class flags, training/validation BA,
cross-entropy, positive preactivation count, units positive on any trial,
nonzero classifier embedding count, and mean between-trial embedding standard
deviation. LeakyReLU can produce many nonzero units without useful separation;
interpret activity together with prediction spread and performance.

Best checkpoints use mean validation-subject BA, then validation loss. No
held-out test scores guide diagnosis or selection. The archive includes
configs, weights, histories and provenance as JSON (not model checkpoint
bytes). The separately saved best.keras files remain on work storage.

Send classifier-collapse-diagnostics.tar.gz from the printed path. This
comparison diagnoses training, not cross-subject test performance. Existing
completed sweep outputs and default model activation/loss behavior remain
compatible. No automatic resume is implemented for these diagnostic tasks.
