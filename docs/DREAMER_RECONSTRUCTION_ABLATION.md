# DREAMER reconstruction ablation

Train SICv15 separately for DREAMER valence and arousal with the same
[fixed hyperparameters](../src/eegproc/deep_learning/joint_architectures/SICModelv15/sic_v15_dreamer_reconstruction_configs.json).
The reconstruction weight is 0.4 in all four configurations, including the
MSE baseline. Seam and derivative coefficients are 1.0 when enabled. Each
loss uses the same coefficient in its single-loss and combined arms.

| Configuration | Seam weight | Derivative weight |
|---|---:|---:|
| 0 · MSE | 0 | 0 |
| 1 · MSE + seam | 1 | 0 |
| 2 · MSE + derivative | 0 | 1 |
| 3 · MSE + both | 1 | 1 |

Each fold uses the other 22 subjects for training, chronological 1-second
nonoverlapping windows at 128 Hz, global RMS normalization per window, and
a positive label for a rating of at least 3. The seed is 42 plus the held-out
subject ID for model initialization. Gradient reversal stays enabled.

## Run on Longleaf

Use a clean `reviewed-experiments` checkout with the usual `venv312` and DREAMER
arrays. Run from the login node with Bash.

```bash
launcher=src/eegproc/deep_learning/joint_architectures/SICModelv15/SLURM_scripts/run_v15_dreamer_reconstruction_ablation.sh

# Preview the eight pilot tasks. No jobs are submitted.
DRY_RUN=1 bash "$launcher" pilot

# Train all four arms for both targets on held-out subject 0 for two epochs.
bash "$launcher" pilot

# Once the pilot fits and checkpoints have been checked, train all 184 fits.
bash "$launcher" full
```

The pilot uses two GPUs per fit. The full run uses the existing three shards
(0–7, 8–15, 16–22) for every target and arm. Its 24 training tasks use two
concurrent folds on four GPUs per task. Eight CPU collectors validate the
three shards for each target and arm before aggregating results. The full
run evaluates the fixed final epoch (48 by default). Held-out metrics do not
select an epoch. Evaluation is zero-shot at a 0.5 classifier threshold.

Set `PILOT_SUBJECT`, `SOURCE_EPOCHS`, `MAX_CONCURRENT`, `EEG_PATH`, `LABELS_PATH`,
`VENV_DIR`, or `OUTPUT_ROOT` as needed. By default results are written under
`$SWEEP_STORAGE_ROOT/runs/sweeps/sic_v15_dreamer_reconstruction`, where
`SWEEP_STORAGE_ROOT` defaults to the account's `/work` EEGProc directory.
Processed EEG is temporary and each fit saves one final checkpoint.

## Check the results

In `suite_JOB_ID/{valence,arousal}/configuration_{0,1,2,3}/`, look for
`final_metrics.json`, `loso_zero_shot_models.json`, and per-subject predictions.
Check `execution.json` for unfinished or failed folds. The full run also
preserves each shard's results under that configuration.

Use the existing held-out reconstruction comparison to score all four arms.
Set `suite` to the output path printed by the launcher and choose the target.

```bash
target=valence
PYTHONPATH=src venv312/bin/python -m eegproc.model_explainability.typicality.compare_reconstructions \
  --label-dimension "$target" \
  --raw-eeg-npy datasets/dreamer_eeg.npy \
  --raw-labels-npy datasets/dreamer_labels.npy \
  --run "mse=$suite/$target/configuration_0/loso_zero_shot_models.json" \
  --run "seam=$suite/$target/configuration_1/loso_zero_shot_models.json" \
  --run "derivative=$suite/$target/configuration_2/loso_zero_shot_models.json" \
  --run "both=$suite/$target/configuration_3/loso_zero_shot_models.json" \
  --out "$suite/$target/reconstruction_comparison"
```

Compare held-out balanced accuracy, reconstruction MSE, seam error, and
within-window derivative error across the matched arms.
