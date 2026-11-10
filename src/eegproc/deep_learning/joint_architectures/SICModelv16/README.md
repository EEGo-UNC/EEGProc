# SIC v16 with the v15 training protocol

The only model replacement in the default experiment is the EEG encoder/decoder
pair: independent theta/alpha/beta CNNs, optional within-window Transformers,
and their matching band decoders. Band order, CNN topology, and the decoder's
pre-Transformer reconstruction inputs are preserved. The old GCN-GRU/BiLSTM
encoders and their convex decoder fusion are not reinstated alongside this pair.

All non-encoder model settings are copied from the fixed `baseline` in
`../SICModelv15/sic_v15_valence_configs.json`. This is an explicit reproducible
reference, rather than one arbitrary member of the separate v15 full-run grid:

- Full ordered-trial BiGRU classifier: widths [128, 64], dropout 0.4.
- Original VC head/objective: focal gamma 0.3, alpha 2, beta 0.6, lambda 0.05,
  logit scale 16; subject GRL strength 0.6 and subject loss weight 0.2.
- Original first-order MLDG: 8 meta-train / 4 meta-test subjects, 3 distinct
  trials per subject, 10 episodes per epoch, inner learning rate 0.0001.
- AdamW learning rate 0.0001, weight decay 0.00005. No class weighting.
- Reconstruction coefficient 0.6 for decoder variants. The replacement pair
  uses equal-band MSE, preserving its encoder/decoder design.

The four comparisons remain CNN, CNN+decoder, CNN+Transformer, and
CNN+Transformer+decoder. `configurations.json` and all new campaigns use the same
non-encoder settings. Pooling/dense heads remain explicit legacy ablations for
old checkpoint compatibility; neither is a restored default.

## Source fitting, selection and calibration

Each of all 23 LOSO targets trains on all 22 other subjects. There is no
source-validation holdout, no four-subject restriction, and no rebuilt/cropped
alternative dataset in this default workflow. A target subset only limits the
evaluated folds; it does not remove available source subjects.

Preprocessing is the existing v15 pipeline: complete ordered trials,
128-sample nonoverlapping windows, global RMS normalization per window,
valence rating >=3 as class 1, prediction threshold 0.5. Raw 14-channel inputs
are filtered by the established EEGProc filter before the same preparation;
42-feature arrays pass directly to v15 preparation. Inputs are never modified.

Two folds run in parallel, each on an isolated pair of GPUs. They reuse v15's
shared-variable device sharding, global episode VC statistics, single inner
adaptation and outer update, and deterministic device execution. Seeds and
GPU deterministic settings follow v15. The new pair supplies only encoder and
reconstruction adapters to these routines.

Each completed epoch saves its checkpoint and untouched target predictions.
One common epoch maximizes mean target-subject trial balanced accuracy, with
the earliest exact tie, using the original v15 `select_oracle`. This is explicitly
test-selected oracle performance. Partial folds/epochs cannot enter selection.
The full default is 12 source epochs; smoke uses 2 epochs and targets 0/1.
These MLDG epochs are not equivalent to the old v16 full-dataset ERM epochs.

After every requested source fold finishes, reload that common-epoch model set
and run the original shared v15 subject-calibration evaluator: 3 shots, 6 folds,
1 calibration epoch, batch size 64, AdamW at 0.0001. Every calibration fold restores
the same source weights and starts a fresh optimizer; only the configured
classifier suffix is unfrozen. Reloaded zero-shot scores must match selection.
`final_metrics.json` is published only after all source and calibration work
finishes. It contains selected epoch, zero-shot metrics and calibrated metrics.

## Run on Longleaf

Prepare the existing `venv312` environment with the project deep-learning extra.
From the repository root, submit the two-GPU synthetic preflight first:

```bash
SIC16=src/eegproc/deep_learning/joint_architectures/SICModelv16
sbatch "$SIC16/SLURM_scripts/preflight.sh"
```

It covers the replacement pair, restored heads/objectives, single/two-device
loss and gradient parity, an MLDG update, optimizer checkpoint restoration,
common-epoch selection, and the shared calibration path. Local tests use two
logical CPUs; CUDA memory use and full-size runtime still need cluster checks.

Plan the four tracked smoke runs:

```bash
python experiments/sic_v16/tracker.py --variant all plan --target valence
```

Submit the four exact commands printed by the tracker. Each requests four GPUs
(two folds x two GPUs), 8 CPUs and 128 GB RAM. The default scope is now `v15`;
new campaigns/results live under `experiments/sic_v16/v15_protocol/VARIANT/` and
matching work storage. After smoke, the tracker proposes the 12-epoch baselines.
There is no automatic hyperparameter search in this fixed architecture control.
The tracker ranks the common-epoch zero-shot oracle score and labels it as such;
calibration results are retained alongside it.

The alternative `smoke.sh` and `full.sh` array launchers use the same restored
protocol and manifest. Do not submit both tracked and array copies of a run.
One array task runs a fixed configuration. Source epochs, source batch size,
calibration epochs, seed, and output paths can be overridden as documented in
the scripts. Budgets are allocation caps, not measured completion estimates.

`run_experiments.py run` defaults to `--protocol v15`. Saved v15-protocol runs
can use `--resume`; settings and dataset metadata must match. Checkpoint weights
and optimizer moments are restored, but post-resume random draws need not match
an uninterrupted run. Preserve every source checkpoint needed for selection.

Old ERM/validation-selected campaigns and independent diagnostic/rebuilt-data
scripts are historical experiments, not the defaults. Their stored results are
preserved. Access the old tracker campaigns explicitly with `--scope all23` or
`--scope first4`; the old runner is explicit `--protocol legacy_validation`.
Old checkpoints cannot be resumed into the restored protocol.
