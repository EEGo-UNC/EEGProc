# SIC MLDG Brier Experiment: Hyperparameter Reference

This document explains every configurable experiment setting in:

- `smoke_test_sic_mldg_brier(1).sh`
- `full_run_sic_mldg_brier_ablation_valence-2.sh`

The shell scripts themselves are unchanged. The smoke script is a short integration test; the full script runs the complete valence LOSO experiment and ablations.

## 1. Main differences between the two scripts

| Setting | Smoke test | Full run | Meaning |
|---|---:|---:|---|
| Array profiles | `full`, `no_bilstm` | `full`, `no_gcn_gru`, `no_bilstm`, `no_decoder`, `remove_median` | Architecture or data condition run by each Slurm array task. |
| Source epochs | 3 | 100 | Maximum source-model training epochs in each LOSO fold. |
| Calibration epochs | 3 | 30 | Fine-tuning epochs for each subject-calibration fold. |
| Maximum LOSO targets | 2 | All | Smoke limits the number of held-out subjects; the full run does not. |
| MLDG steps per epoch | 2 | 10 | Number of MLDG meta-updates performed per reported source epoch. |
| Calibration levels | 3-shot/2-fold, 6-shot/2-fold | 3-shot/6-fold, 6-shot/3-fold, 9-shot/2-fold, 12-shot/3-fold | Number of calibration trials and repeated folds at each shot level. |
| HP-selection shot level | 6-shot | 12-shot | Calibrated level whose mean Brier score ranks configurations. |
| BiLSTM width | Fixed at 63 | Grid over 42, 63, and 96 | Units per direction. The full run skips this grid when the BiLSTM is disabled. |
| Prediction latent samples | 5 | 20 | Monte Carlo latent draws averaged for probabilistic evaluation. |
| Wall-time limit | 4 hours | 48 hours | Maximum Slurm runtime per array task. |

## 2. How the JSON grid syntax works

| Form | Meaning |
|---|---|
| `"parameter": value` | Use one value directly. |
| `"parameter": {"fixed": value}` | Treat the value as one fixed grid item. This is useful when the value is itself a list or `None`. |
| `"parameter": {"grid": [a, b, c]}` | Run one configuration for every listed value as part of the Cartesian hyperparameter search. |

For example, `{"fixed": [128, 64]}` means one two-layer classifier with widths 128 and 64. It does **not** search 128 versus 64. In the full script, `{"grid": [42, 63, 96]}` for `bilstm_units` creates three configurations when that branch is active.

## 3. Model and optimization hyperparameters

### 3.1 Source optimizer

| Hyperparameter | Current value | What it does |
|---|---:|---|
| `training_method` | `mldg` | Selects Model-Agnostic Learning of Domain Generalization as the source-training procedure. Subjects act as domains. |
| `optimizer_name` | `adamw` | Uses AdamW for source optimization: Adam-style adaptive updates with decoupled weight decay. |
| `learning_rate` | `1e-4` | Step size for the main source-model optimizer. Larger values learn faster but can make training unstable. |
| `weight_decay` | `5e-5` | AdamW regularization strength applied to eligible weights. Larger values discourage large weights more strongly. |

### 3.2 MLDG

Within each LOSO fold, DREAMER leaves 22 source subjects after holding out one target. These settings divide those 22 subjects into 18 temporary meta-train subjects (set A) and 4 temporary meta-test subjects (set B).

| Hyperparameter | Current value | What it does |
|---|---:|---|
| `mldg_meta_train_subjects` | 18 | Number of source subjects sampled for the MLDG meta-train/domain set A. |
| `mldg_meta_test_subjects` | 4 | Number of different source subjects used as the temporary meta-test/domain set B. |
| `mldg_trials_per_subject` | 1 | Number of trials sampled from each selected subject for one MLDG step. It controls per-step data volume, not the number of LOSO folds. |
| `mldg_steps_per_epoch` | Smoke: 2; full: 10 | Number of meta-train → inner adaptation → meta-test updates in each reported source epoch. This is not automatically a full pass over all windows. |
| `mldg_inner_learning_rate` | `1e-4` | Learning rate for the temporary inner update on meta-train subjects before evaluating meta-test generalization. |
| `mldg_meta_test_weight` | `1.0` | Multiplier on the meta-test/generalization loss in the outer MLDG objective. Larger values emphasize performance after the temporary adaptation. |
| `mldg_seed` | 42 | Reproducibility seed for MLDG subject and trial sampling. |

### 3.3 GCN–GRU spatial/spectral branch

| Hyperparameter | Current value | What it does |
|---|---:|---|
| `t_down` | 2 | Temporal downsampling factor used before or within the spatial/spectral encoder. A larger factor reduces sequence length and memory use but removes temporal detail. |
| `temporal_pool_sizes` | Fixed `[2]` | Pooling window for each configured temporal-pooling stage. `[2]` halves the applicable temporal resolution once. |
| `gcn_units` | Fixed `[128, 64]` | Output widths of the successive graph-convolution layers. Each EEG channel aggregates information from connected channels. |
| `gcn_dropout` | 0.20 | Fraction of GCN activations randomly dropped during training. It regularizes the graph branch. |
| `gcn_activation` | `relu` | Nonlinearity used by the GCN layers. ReLU keeps positive activations and sets negative activations to zero. |
| `gcn_use_batch_norm` | `False` | Whether to apply batch normalization in the GCN. It is disabled to avoid batch-statistic dependence across subject domains. |
| `spectral_gru_units` | 384 | Hidden width of the GRU that combines the band-specific graph representations. |
| `spectral_gru_dropout` | 0.20 | Fraction of spectral-GRU inputs/activations dropped during training for regularization. |
| `mi_n_neighbors` | 3 | Number of nearest neighbors used by the k-nearest-neighbor mutual-information estimator that builds channel adjacency. |
| `mi_random_state` | 42 | Seed for reproducible mutual-information estimation or subsampling. |
| `mi_zero_diagonal` | `False` | If `True`, removes self-connections from the MI adjacency diagonal. `False` retains the computed/self-loop diagonal behavior. |
| `mi_band_reduction` | `mean` | Averages the per-band mutual-information adjacencies into the adjacency representation expected by the graph encoder. |
| `mi_max_observations` | 15000 | Maximum observations used to estimate MI. It caps adjacency-computation cost and memory. |

### 3.4 Independent BiLSTM temporal branch

| Hyperparameter | Smoke value | Full-run value | What it does |
|---|---:|---:|---|
| `bilstm_units` | 63 | Grid `[42, 63, 96]` | Hidden units **per direction**. The resulting bidirectional widths are 126 in smoke and 84/126/192 in the full grid. |
| `n_bilstm_layers` | 1 | 1 | Number of stacked bidirectional LSTM layers. More layers increase temporal modeling capacity and compute. |
| `bilstm_dropout` | 0.30 | 0.30 | Fraction of BiLSTM representations dropped during training. |

### 3.5 Variational latent representation and decoder

| Hyperparameter | Current value | What it does |
|---|---:|---|
| `z_dim` | 64 | Latent dimension produced by **each active branch**. With both branches enabled, their independent 64-D latents are concatenated into 128 dimensions. |
| `z_log_var_clip_min` | -20.0 | Lower clipping bound for posterior log-variance. It prevents variance from becoming numerically indistinguishable from zero. |
| `z_log_var_clip_max` | 20.0 | Upper clipping bound for posterior log-variance. It prevents exploding variance and unstable latent samples. |
| `vae_loss_weight` | 0.10 | Overall multiplier for the VAE/decoder objective relative to the other training losses. Setting it to zero effectively removes its optimization contribution. |
| `vae_beta` | 0.05 | Weight on the VAE KL-divergence term relative to reconstruction. Larger values force the posterior closer to its prior but can reduce task information. |
| `decoder_dropout` | 0.10 | Fraction of decoder activations dropped during source training. It regularizes reconstruction. |

### 3.6 Emotion classifier and variational-classifier objective

| Hyperparameter | Current value | What it does |
|---|---:|---|
| `focal_gamma` | 1.5 | Focal-loss focusing exponent. Larger values down-weight easy examples more strongly; `0` reduces the focusing behavior to ordinary cross-entropy. |
| `focal_alpha` | Fixed `None` | Optional class-specific focal weighting. `None` applies no focal-alpha class weighting. |
| `classification_hidden_units` | Fixed `[128, 64]` | Widths of the two hidden dense layers in the emotion-classification head. |
| `classification_dropout` | 0.20 | Fraction of classification-head activations dropped during training. |
| `vc_loss_weight` | 1.0 | Overall multiplier on the complete variational-classifier objective when it is combined with the other SIC losses. |
| `vc_alpha` | 1.0 | Coefficient on the VC objective's primary supervised classification term. |
| `vc_beta` | 0.5 | Coefficient on the VC latent KL regularization term; larger values impose a stronger variational bottleneck. |
| `vc_gamma` | 0.0 | Coefficient on the VC gamma auxiliary term. A value of zero disables that term. |
| `vc_lambda` | 0.20 | Weight assigned to the VC discriminator/divergence regularizer used by the implementation. |
| `update_vc_discriminator` | `False` | Controls whether the VC discriminator receives its own parameter updates. `False` leaves that discriminator frozen while the rest of the configured VC objective is computed. |

`vc_loss_weight` scales the VC objective as a whole; `vc_alpha`, `vc_beta`, `vc_gamma`, and `vc_lambda` control components inside it.

### 3.7 Subject-adversarial invariance

| Hyperparameter | Current value | What it does |
|---|---:|---|
| `use_subject_adversarial` | `True` | Enables the subject-prediction adversary. Through gradient reversal, the encoder is encouraged to make subject identity difficult to recover. |
| `subject_adversarial_weight` | 0.60 | Strength of the adversarial/reversed signal reaching the shared representation. Larger values push harder toward subject invariance. |
| `subject_loss_weight` | 1.0 | Overall multiplier on the subject-classification loss. |
| `subject_hidden_units` | 64 | Width of the hidden layer in the subject discriminator. |
| `subject_dropout` | 0.0 | Dropout rate in the subject discriminator. Zero disables discriminator dropout. |

### 3.8 Architecture, data, and calibration switches

| Hyperparameter | Value by profile | What it does |
|---|---|---|
| `use_gcn_gru_branch` | `False` only for `no_gcn_gru` | Includes or removes the spatial/spectral GCN–GRU encoder branch. |
| `use_bilstm_branch` | `False` only for `no_bilstm` | Includes or removes the independent temporal BiLSTM branch. |
| `use_decoder` | `False` only for `no_decoder` | Includes or removes the decoder and its reconstruction objective. |
| `remove_median_label` | `True` only for `remove_median` | Removes samples/trials whose raw rating equals the median label instead of assigning them to a binary class. |
| `calibration_unfreeze_layers` | 2 | Number of final classifier layers allowed to update during subject-specific calibration; the rest of the source model stays frozen. |
| `calibration_use_vc_target` | `True` | Uses the variational-classifier target/objective during calibration rather than calibrating only a plain deterministic softmax path. |
| `use_class_weight` | `False` | Disables inverse-frequency class weighting. Focal loss still operates, but no separate class-weight multiplier is applied. |

## 4. Selection, calibration, and reporting settings

| Setting | Smoke value | Full-run value | What it does |
|---|---:|---:|---|
| `SELECTION_METRIC` / `--selection-metric` | `brier_score` | `brier_score` | Metric used to rank hyperparameter configurations. Brier score measures squared probability error, so **lower is better**. |
| `HYPERPARAMETER_SELECTION_LEVEL` / `--hyperparameter-selection-level` | `calibration` | `calibration` | Selects configurations using post-calibration results rather than zero-shot LOSO results. Zero-shot metrics are still reported. |
| `CALIBRATION_SELECTION_SHOTS` / `--calibration-selection-shots` | 6 | 12 | Chooses which calibrated shot level supplies the selection metric. It does not suppress reporting at the other configured levels. |
| `--calibration-level SHOTS FOLDS` | `3 2`, `6 2` | `3 6`, `6 3`, `9 2`, `12 3` | Runs `FOLDS` subject-calibration repeats, each using `SHOTS` labeled trials from the held-out subject; evaluation uses that subject's remaining trials. |
| `--calibration-epochs` | 3 | 30 | Maximum classifier fine-tuning epochs in each calibration repeat. |
| `--calibration-batch-size` | 32 | 32 | Number of calibrated samples/windows used in one fine-tuning update. |
| `--calibration-learning-rate` | 0.0001 | 0.0001 | Learning rate for subject-specific fine-tuning. It is separate from both source and MLDG inner learning rates. |
| `--calibration-optimizer` | `adamw` | `adamw` | Optimizer used only during subject calibration. |
| `--calibration-weight-decay` | 0.00005 | 0.00005 | AdamW weight decay used only during calibration. |
| `--calibration-seed` | 42 | 42 | Seed controlling calibration trial/fold selection and calibration randomness. |
| `--decision-threshold` | 0.5 | 0.5 | Converts predicted positive-class probabilities into binary labels for threshold-based metrics. It does not affect Brier score or ROC-AUC. |
| `--prediction-latent-samples` | 5 | 20 | Number of stochastic latent predictions averaged at evaluation. More samples give a more stable probability estimate but cost more time. |
| `--latent-sampling-seed` | 42 | 42 | Makes evaluation-time latent sampling reproducible. |
| `--ece-bins` | 15 | 15 | Number of confidence bins used to compute Expected Calibration Error. More bins give finer calibration resolution but can be noisy with few examples. |

## 5. LOSO training and data settings

| Setting | Current value | What it does |
|---|---:|---|
| `--training-protocol` | `loso_validation` | Runs leave-one-subject-out training: one subject is held out as the target while the remaining subjects supply source training data. |
| `--raw-eeg-npy` | `datasets/remove_gamma/dreamer_eeg.npy` | Path to the prepared DREAMER EEG array. |
| `--raw-labels-npy` | `datasets/remove_gamma/dreamer_labels.npy` | Path to the aligned DREAMER rating/label array. |
| `--label-dimension` | `valence` | Uses valence as the binary prediction target. Change to `arousal` for the arousal experiment if supported by the trainer. |
| `--classification-level` | `window` | Trains the classifier on EEG windows. Trial-level metrics can still aggregate the window predictions belonging to each trial. |
| `--n-channels` | 14 | Number of EEG electrodes expected in each input. This matches DREAMER's 14-channel Emotiv montage. |
| `--n-bands` | 3 | Number of frequency bands expected in the prepared data. The `remove_gamma` dataset uses theta, alpha, and beta. |
| `--source-epochs` | Smoke: 3; full: 100 | Number of source-training epochs. Because early stopping is disabled, the trainer attempts all of them. |
| `--source-batch-size` | 512 | Number of source examples/windows processed per ordinary source batch where the selected training path uses batches. |
| `--validation-subjects` | 0 | Reserves no source subjects as a separate validation set within each LOSO fold. |
| `--no-early-stopping` | Enabled | Disables early stopping, so source training is controlled by `--source-epochs`. |
| `--max-subjects` | Smoke: 2; full: omitted | Limits the number of LOSO target subjects for a quick smoke test. Omitting it runs all available targets. |
| `--label-threshold-mode` | `global` | Applies one shared rating threshold across subjects instead of computing subject-specific thresholds. |
| `--median-label` | 3 | Sets rating 3 as the dataset's median/decision-boundary label. The `remove_median` ablation discards exactly this rating. |
| `--window-sec` | 4.0 | Length of each EEG window in seconds. |
| `--window-overlap` | 0.0 | Fractional overlap between adjacent windows. Zero produces non-overlapping windows. |
| `--window-normalization` | `global_rms` | Normalizes amplitudes using the configured global root-mean-square scaling procedure. |

## 6. Output and parallel-execution settings

| Setting | Smoke value | Full-run value | What it does |
|---|---|---|---|
| `--out-dir` | `runs/smoke/...` | `runs/full/...` | Root directory for metrics, configuration records, checkpoints, and reports. It is separated by suite and ablation profile. |
| `--run-name` | Ends in `_smoke` | Ends in `_full` | Human-readable identifier embedded in run outputs. |
| `--n-jobs` | 2 | 2 | Number of local worker processes/jobs used by the trainer. |
| `--gpu-ids` | `0 1` | `0 1` | Makes both GPUs allocated to the Slurm task available to the training workers. These are local IDs inside the job. |
| `--cpus-per-worker` | 2 | 2 | CPU threads assigned to each worker. With two workers, this matches the four requested CPUs. |
| `--verbose` | 2 | 2 | Logging verbosity level. Level 2 requests detailed progress without changing training. |
| `--seed` | 42 | 42 | Global experiment seed for reproducible splits, initialization, shuffling, and other trainer-controlled randomness. |
| `--hyperparameters-json` | `MODEL_CONFIG` | `MODEL_GRID` | Passes the generated model configuration or Cartesian grid into the training program. |

## 7. Shell-level run controls

The `${NAME:-default}` form means the script uses `default` unless `NAME` is supplied in the environment. For example:

```bash
SOURCE_EPOCHS=10 MLDG_STEPS_PER_EPOCH=4 sbatch smoke_test_sic_mldg_brier\(1\).sh
```

| Variable | Smoke default | Full default | What it does |
|---|---:|---:|---|
| `PROJECT_DIR` | `$HOME/EEGProc` | `$HOME/EEGProc` | Repository directory entered before training. |
| `VENV_DIR` | `$PROJECT_DIR/venv312` | `$PROJECT_DIR/venv312` | Python virtual environment created or reused by the job. |
| `SOURCE_EPOCHS` | 3 | 100 | Override forwarded to `--source-epochs`. |
| `CALIBRATION_EPOCHS` | 3 | 30 | Override forwarded to `--calibration-epochs`. |
| `MAX_SUBJECTS` | 2 | Not defined | Smoke-only cap forwarded to `--max-subjects`. |
| `MLDG_STEPS_PER_EPOCH` | 2 | 10 | Override inserted into the model JSON. |
| `INSTALL_REQUIREMENTS` | 0 | 0 | If `1`, reinstalls/updates `requirements.txt` even when the virtual environment already exists. |
| `SUITE_ID` | Slurm array/job ID | Slurm array/job ID | Groups every task in the same submitted array under one output suite. It falls back to `manual` outside Slurm. |
| `PROFILE_INDEX` | Array task ID | Array task ID | Chooses the ablation profile corresponding to the current Slurm array task. |
| `ABLATION_PROFILE` | Derived | Derived | Name of the selected profile and output subdirectory; normally do not set it directly. |

The Python, CUDA, and cuDNN module versions are environment dependencies rather than learning hyperparameters. `PYTHONNOUSERSITE=1` prevents user-site packages from leaking into the environment, and `PYTHONUNBUFFERED=1` makes logs appear immediately.

## 8. Slurm resource settings

These settings control cluster scheduling and resources; they do not change the model mathematics.

| Slurm option | Smoke value | Full-run value | What it does |
|---|---|---|---|
| `--job-name` | `smoke_sic_v8` | `sic_v8_abl_val` | Name shown in the Slurm queue. |
| `--output` | `smoke_sic_v8_%A_%a.out` | `sic_v8_abl_val_%A_%a.out` | Standard-output filename. `%A` is the array job ID and `%a` is the array task ID. |
| `--error` | `smoke_sic_v8_%A_%a.err` | `sic_v8_abl_val_%A_%a.err` | Standard-error filename using the same array placeholders. |
| `--partition` | `l40-gpu` | `l40-gpu` | Requests the L40 GPU partition. |
| `--qos` | `gpu_access` | `gpu_access` | Selects the cluster's GPU quality-of-service policy. |
| `--gres` | `gpu:2` | `gpu:2` | Allocates two GPUs to each array task. |
| `--cpus-per-task` | 4 | 4 | Allocates four CPU cores to each array task. |
| `--mem` | 128G | 128G | Allocates 128 GiB of host RAM to each array task. |
| `--time` | `4:00:00` | `48:00:00` | Hard wall-clock limit for each array task. |
| `--array` | `0-1%2` | `0-4%2` | Creates two or five tasks and permits at most two to run concurrently. Each running task requests two GPUs. |

## 9. Ablation profiles

| Profile | Changed setting | Scientific question |
|---|---|---|
| `full` | No component removed | How does the complete SIC model perform? |
| `no_gcn_gru` | `use_gcn_gru_branch=False` | How much does the spatial/spectral graph branch contribute? |
| `no_bilstm` | `use_bilstm_branch=False` | How much does the independent temporal branch contribute? |
| `no_decoder` | `use_decoder=False` | Does reconstruction regularization help generalization and calibration? |
| `remove_median` | `remove_median_label=True` | How do results change when ambiguous rating-3 examples are removed entirely? |

The full run is a one-factor-at-a-time ablation: each non-full profile removes or changes only one component relative to `full`.

## 10. Parameters that are easy to confuse

| Parameters | Difference |
|---|---|
| `learning_rate` vs. `mldg_inner_learning_rate` vs. `calibration-learning-rate` | Main source optimizer step size vs. temporary MLDG adaptation step size vs. held-out-subject fine-tuning step size. |
| `weight_decay` vs. `calibration-weight-decay` | Source-training AdamW regularization vs. calibration-only AdamW regularization. |
| `mldg_steps_per_epoch` vs. `source_epochs` | Meta-updates inside one epoch vs. number of outer epochs. |
| `mldg_trials_per_subject` vs. `calibration-level` | Source trials sampled per subject in an MLDG step vs. labeled target-subject trials used for calibration. |
| `vae_loss_weight` vs. `vae_beta` | Weight of the entire VAE objective vs. weight of KL divergence inside that objective. |
| `vc_loss_weight` vs. `vc_alpha/beta/gamma/lambda` | Weight of the complete VC objective vs. weights of its internal terms. |
| `subject_adversarial_weight` vs. `subject_loss_weight` | Strength of the invariance pressure on the representation vs. scale of the subject-classification loss. |
| `calibration-selection-shots` vs. all `calibration-level` entries | Shot level used to rank HPs vs. all shot levels evaluated and reported. |
| `decision-threshold` vs. `label-threshold-mode`/`median-label` | Threshold applied to model probabilities at evaluation vs. rule used to turn raw DREAMER ratings into training targets. |
| `prediction-latent-samples` vs. calibration folds | Monte Carlo samples of the variational latent per prediction vs. different labeled-trial subsets used for calibration evaluation. |
