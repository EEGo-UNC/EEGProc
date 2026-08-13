# SIC MLDG Hyperparameter Reference

This reference matches the deterministic SIC implementation and these worker scripts:

- `smoke_test_sic_direct_concat_v10.sh`
- `full_run_sic_direct_concat_v10.sh`
- `submit_sic_direct_concat_v10_suite.sh`

The GCN-GRU and BiLSTM outputs are not projected into a learned \(z\) space. Their complete feature vectors are concatenated and passed to the final classifier and subject-adversarial heads.

## 1. Experiment summary

| Setting | Smoke | Full |
| --- | --- | --- |
| Array profiles | `full`, `no_bilstm` | `full`, `remove_median`, `no_gcn_gru`, `no_bilstm` |
| Source epochs | 3 | 100 |
| Calibration epochs | 3 | 30 |
| Target-subject limit | 2 | All |
| MLDG steps per epoch | 2 | 20 |
| Calibration levels | 3-shot/2-fold, 6-shot/2-fold | 3-shot/6-fold, 6-shot/3-fold, 9-shot/2-fold, 12-shot/3-fold |
| Selection level | 6-shot calibrated | 12-shot calibrated |
| Selection metric | Brier score | Brier score |
| Diagnostic metric | Brier score | Brier score |
| Diagnostic subset | 128 samples/split | 256 samples/split |

Brier score is minimized. Most discrimination metrics are maximized.

## 2. JSON fixed values and grids

The parser distinguishes a fixed sequence from a grid:

```json
{
  "gcn_units": {"fixed": [128, 64]},
  "classification_hidden_units": {"fixed": [128, 64]},
  "bilstm_units": {"grid": [42, 63, 96]}
}
```

- `{"fixed": [128, 64]}` means one two-layer architecture.
- `{"grid": [42, 63, 96]}` creates three independent scalar candidates.
- An ordinary scalar such as `0.2` is fixed.

The full script has three grid axes when the BiLSTM is active:

| Axis | Candidates |
| --- | --- |
| `bilstm_units` | 42, 63, 96 |
| `focal_gamma` | 0, 1, 2 |
| `vc_beta` | 0.5, 1.5, 2.5 |

That is \(3\times3\times3=27\) candidates for each BiLSTM-enabled profile. The `no_bilstm` profile has \(3\times3=9\) candidates.

## 3. Source optimizer and training method

| Setting | Value | Meaning |
| --- | ---: | --- |
| `TRAINING_METHOD` / `--training-method` | `mldg` by default | Selects `erm`, `vrex`, or `mldg`. |
| `optimizer_name` | `adamw` | Main source optimizer. |
| `learning_rate` | `1e-4` | Persistent source-optimizer learning rate. |
| `weight_decay` | `5e-5` | Decoupled AdamW weight decay. |
| `VREX_PENALTY_WEIGHT` / `vrex_penalty_weight` | `1.0` | Multiplier on cross-subject focal-risk variance when V-REx is selected. |

`TRAINING_METHOD` is forwarded through the dedicated CLI flag. It is not hard-coded inside the JSON grid.

## 4. First-order MLDG

| Hyperparameter | Value | Meaning |
| --- | ---: | --- |
| `mldg_meta_train_subjects` | 18 | Number of A/meta-train source subjects in each episode. |
| `mldg_meta_test_subjects` | 4 | Number of disjoint B/virtual-unseen source subjects. |
| `mldg_trials_per_subject` | 1 | Complete trials selected per episode subject. |
| `mldg_steps_per_epoch` | Smoke 2; full 20 | Number of complete MLDG episodes and persistent outer updates per source epoch. |
| `mldg_inner_learning_rate` | `1e-4` | Temporary A-step learning rate. |
| `mldg_meta_test_weight` | `1.0` | Weight applied to the B gradient in the outer update. |
| `mldg_seed` | 42 | Subject/trial episode reproducibility seed. |

A and B together use all 22 non-target DREAMER subjects. Therefore the supplied MLDG configuration requires `validation_subjects=0`.

`mldg_steps_per_epoch` is not a conventional pass through every source window. One step is one complete subject-disjoint meta-learning episode.

MLDG implementation details live in `MetaLearning.py` rather than `sic_model.py`.

## 5. GCN-GRU branch

| Hyperparameter | Value | Meaning |
| --- | ---: | --- |
| `t_down` | 2 | Temporal downsampling factor. |
| `temporal_pool_sizes` | Fixed `[2]` | One temporal pooling stage whose total factor matches `t_down`. |
| `gcn_units` | Fixed `[128, 64]` | Successive graph-convolution widths. |
| `gcn_dropout` | 0.20 | Graph-branch dropout rate. |
| `gcn_activation` | `relu` | GCN activation. |
| `gcn_use_batch_norm` | `False` | Disables GCN batch normalization. |
| `spectral_gru_units` | 384 | Complete output width retained from the GCN-GRU branch. |
| `spectral_gru_dropout` | 0.20 | Spectral-GRU dropout. |
| `mi_n_neighbors` | 3 | Neighbor count for mutual-information estimation. |
| `mi_random_state` | 42 | MI estimation/subsampling seed. |
| `mi_zero_diagonal` | `False` | Retains diagonal/self-connectivity behavior. |
| `mi_band_reduction` | `mean` | Averages per-band MI information as expected by the graph encoder. |
| `mi_max_observations` | 15000 | Caps the observations used for MI estimation. |

The MI graph is estimated independently within each LOSO source fold. Target-subject features are excluded.

## 6. BiLSTM branch and direct feature width

| Hyperparameter | Smoke | Full | Meaning |
| --- | ---: | ---: | --- |
| `bilstm_units` | 63 | Grid 42, 63, 96 | Units per direction. |
| `n_bilstm_layers` | 1 | 1 | Stacked bidirectional LSTM layers. |
| `bilstm_dropout` | 0.30 | 0.30 | BiLSTM-output dropout. |

The BiLSTM output width is

$$
d_b=2\,\texttt{bilstm\_units}.
$$

The joint width is

$$
d_{\text{joint}}
=
384\cdot\mathbb{1}_{\text{GCN-GRU}}
+
2\,\texttt{bilstm\_units}\cdot\mathbb{1}_{\text{BiLSTM}}.
$$

| Active branches | Width |
| --- | ---: |
| Both, 42 units/direction | 468 |
| Both, 63 units/direction | 510 |
| Both, 96 units/direction | 576 |
| GCN-GRU only | 384 |
| BiLSTM only | 84, 126, or 192 |

Temporal pooling aligns sequence lengths. It does not project feature dimensions.

## 7. Emotion classifier and classifier-head regularizers

| Hyperparameter | Smoke | Full | Meaning |
| --- | ---: | ---: | --- |
| `focal_gamma` | 1.5 | Grid 0, 1, 2 | Focal focusing exponent; zero removes focal focusing. |
| `focal_alpha` | `None` | `None` | No separate scalar focal class weighting. |
| `classification_hidden_units` | Fixed `[128, 64]` | Fixed `[128, 64]` | Dense classifier hidden widths. |
| `classification_dropout` | 0.20 | 0.20 | Classifier-head dropout. |
| `vc_loss_weight` | 1.0 | 1.0 | Overall classifier/VC objective weight. |
| `vc_alpha` | 1.0 | 1.0 | Weight on the focal classification term inside the combined objective. |
| `vc_beta` | Grid 0.5, 1.5, 2.5 | Grid 0.5, 1.5, 2.5 | VariationalClassifier latent regularizer weight. |
| `vc_gamma` | 0.0 | 0.0 | Disables that VC auxiliary term. |
| `vc_lambda` | 0.05 | 0.05 | VC discriminator/divergence regularizer weight. |
| `update_vc_discriminator` | `False` | `False` | Does not run a separate VC-discriminator optimizer. |

The `VariationalClassifier` regularizes the final classification embedding. It does not turn the encoder outputs into a VAE latent, and it does not change the direct encoder concatenation width.

## 8. Subject-adversarial invariance

| Hyperparameter | Value | Meaning |
| --- | ---: | --- |
| `use_subject_adversarial` | `True` | Enables subject prediction with gradient reversal. |
| `subject_adversarial_weight` | 0.60 | Gradient-reversal strength. |
| `subject_loss_weight` | 1.0 | Subject-loss multiplier in the source objective. |
| `subject_hidden_units` | 64 | Subject-head hidden width. |
| `subject_dropout` | 0.0 | Disables subject-head dropout. |

The subject head receives the pooled concatenated deterministic encoder vector.

## 9. Architecture, data, and calibration switches

| Setting | Value/profile | Meaning |
| --- | --- | --- |
| `use_gcn_gru_branch` | False only in `no_gcn_gru` | Removes the complete graph/spectral branch. |
| `use_bilstm_branch` | False only in `no_bilstm` | Removes the complete temporal branch. |
| `remove_median_label` | True only in `remove_median` | Removes complete trials whose raw target rating equals 3. |
| `calibration_unfreeze_layers` | 2 | Unfreezes the final classifier hidden block and logits layer. |
| `calibration_use_vc_target` | `True` | Retains the configured frozen VC-target regularization during calibration. |
| `use_class_weight` | `False` | Disables model-side class weighting. |
| `--source-use-class-weight` | Not passed | Disables source class weights in the evaluator. |
| `--calibration-use-class-weight` | Not passed | Disables calibration class weights. |

There is no decoder switch because deterministic SIC has no decoder.

## 10. Selection and calibration CLI settings

| Setting | Smoke | Full | Meaning |
| --- | ---: | ---: | --- |
| `--selection-metric` | `brier_score` | `brier_score` | Metric used to rank configurations; lower is better. |
| `--hyperparameter-selection-level` | `calibration` | `calibration` | Selects from post-calibration rather than zero-shot results. |
| `--calibration-selection-shots` | 6 | 12 | Shot level used for ranking. Other levels remain reported. |
| `--calibration-epochs` | 3 | 30 | Maximum calibration epochs per continuation. |
| `--calibration-batch-size` | 32 | 32 | Calibration batch size. |
| `--calibration-learning-rate` | `1e-4` | `1e-4` | Fresh calibration-optimizer learning rate. |
| `--calibration-optimizer` | `adamw` | `adamw` | Calibration-only optimizer. |
| `--calibration-weight-decay` | `5e-5` | `5e-5` | Calibration-only weight decay. |
| `--calibration-seed` | 42 | 42 | Calibration split seed. |
| `--decision-threshold` | 0.5 | 0.5 | Binary probability threshold for label-based metrics. |
| `--ece-bins` | 15 | 15 | ECE confidence-bin count. |

## 11. Prediction diagnostics

| Setting | Smoke | Full | Meaning |
| --- | ---: | ---: | --- |
| `--prediction-diagnostics` | Enabled | Enabled | Records source-training diagnostic rows. |
| `PREDICTION_DIAGNOSTICS_METRIC` | `brier_score` | `brier_score` | Value passed to `--prediction-diagnostics-metric`. |
| `PREDICTION_DIAGNOSTICS_EVERY_N_EPOCHS` | 1 | 1 | Diagnostic frequency. |
| `PREDICTION_DIAGNOSTICS_MAX_SAMPLES` | 128 | 256 | Maximum approximately class-balanced samples per split. |
| `--prediction-diagnostics-threshold-tolerance` | 0.01 | 0.01 | Tolerance retained for probability-collapse diagnostics. |
| `--prediction-diagnostics-seed` | 42 | 42 | Diagnostic subset seed. |

Supported reported metrics are accuracy, F1, precision, recall, their macro variants, balanced accuracy, ROC-AUC, Brier score, and ECE.

Every row contains `reported_metric` and `reported_metric_value`. Rows are saved in `sic_prediction_diagnostics.csv`.

Because the supplied scripts reserve no source-validation subjects, only `train` diagnostic rows are produced.

## 12. LOSO, data, and compute settings

| Setting | Smoke | Full | Meaning |
| --- | ---: | ---: | --- |
| `--training-protocol` | `loso_validation` | `loso_validation` | Runs the grid-capable strict LOSO plus calibration pipeline. |
| `--classification-level` | `window` | `window` | Optimizes window predictions while retaining trial aggregation metadata. |
| `--n-channels` | 14 | 14 | DREAMER channel count. |
| `--n-bands` | 3 | 3 | Theta, alpha, and beta features. |
| `--source-batch-size` | 512 | 512 | Ordinary source batch setting; MLDG episode construction controls MLDG step contents. |
| `--validation-subjects` | 0 | 0 | Uses all 22 non-target subjects for MLDG A+B. |
| `--no-early-stopping` | Enabled | Enabled | Runs every configured source epoch. |
| `--max-subjects` | 2 | Omitted | Smoke-only target-subject cap. |
| `--label-threshold-mode` | `global` | `global` | Uses one shared DREAMER rating threshold. |
| `--median-label` | 3 | 3 | Boundary rating and remove-median target. |
| `--window-sec` | 1.0 | 1.0 | Window length supplied to the data builder. |
| `--window-overlap` | 0.0 | 0.0 | Non-overlapping windows. |
| `--window-normalization` | `global_rms` | `global_rms` | Window normalization mode. |
| `--n-jobs` | 2 | 2 | Two fold workers. |
| `--gpu-ids` | 0 1 | 0 1 | One local GPU ID per worker. |
| `--cpus-per-worker` | 2 | 2 | CPU threads per worker. |
| `--seed` | 42 | 42 | Global reproducibility seed. |

The scripts default to:

```text
datasets/remove_gamma/dreamer_eeg.npy
datasets/remove_gamma/dreamer_labels.npy
```

Override `EEG_PATH` and `LABELS_PATH` when the prepared arrays live elsewhere.

## 13. Shell controls

| Variable | Smoke default | Full default |
| --- | ---: | ---: |
| `PROJECT_DIR` | `$HOME/EEGProc` | `$HOME/EEGProc` |
| `VENV_DIR` | `$PROJECT_DIR/venv312` | `$PROJECT_DIR/venv312` |
| `TARGET_DIMENSION` | `valence` | `valence` |
| `TRAINING_METHOD` | `mldg` | `mldg` |
| `VREX_PENALTY_WEIGHT` | 1.0 | 1.0 |
| `SOURCE_EPOCHS` | 3 | 100 |
| `CALIBRATION_EPOCHS` | 3 | 30 |
| `MAX_SUBJECTS` | 2 | — |
| `MLDG_STEPS_PER_EPOCH` | 2 | 20 |
| `INSTALL_REQUIREMENTS` | 0 | 0 |

Example:

```bash
TARGET_DIMENSION=arousal \
TRAINING_METHOD=mldg \
PREDICTION_DIAGNOSTICS_METRIC=balanced_accuracy \
SOURCE_EPOCHS=10 \
  ./submit_sic_direct_concat_v10_suite.sh smoke 2
```

## 14. Removed or easily confused settings

Do not use these obsolete encoder-VAE parameters:

- `use_decoder`
- `z_log_var_clip_min`
- `z_log_var_clip_max`
- `vae_loss_weight`
- `vae_beta`
- `decoder_dropout`

SIC always uses parallel branches and direct feature concatenation.

| Parameters | Difference |
| --- | --- |
| `learning_rate` vs. `mldg_inner_learning_rate` vs. `calibration-learning-rate` | Persistent source optimizer vs. temporary MLDG inner step vs. target-calibration optimizer. |
| `mldg_steps_per_epoch` vs. `source_epochs` | MLDG episodes inside one epoch vs. number of reported source epochs. |
| `mldg_trials_per_subject` vs. `calibration-level` | Source trials per MLDG episode subject vs. target trials used for calibration. |
| `selection-metric` vs. `prediction-diagnostics-metric` | Final grid-ranking metric vs. per-epoch diagnostic metric. They may be the same or different. |
| `calibration-selection-shots` vs. `calibration-level` | One shot level used for ranking vs. every shot/fold pair that is evaluated. |
| `decision-threshold` vs. `median-label` | Probability-to-class threshold vs. raw DREAMER rating boundary. |
| `vc_beta` vs. removed `vae_beta` | Classifier-head regularizer weight vs. the former encoder-VAE KL weight. |
