# Subject Invariant Calibrator (SIC)

SIC is an EEG emotion-classification system for studying cross-subject generalization and rapid subject calibration on DREAMER. The experiment measures how well a source-trained population model predicts a completely unseen user at zero shot, then how its predictions change after calibrating its classifier with a small number of that user's trials.

The workflow supports both binary **valence** and binary **arousal**. It reports Brier score as the primary calibration-sensitive metric, alongside accuracy, balanced accuracy, F1, precision, recall, ROC-AUC, expected calibration error (ECE), and related macro metrics. Reports are generated at zero shot and at every requested calibration level, such as 3, 6, 9, and 12 shots.

This README describes the corrected SIC (model version 8) encoder. In this version, the GCN-GRU and BiLSTM branches have independent variational posteriors. Their latent samples are concatenated directly; there is no learned fusion projection between them and the classifier.

## Research questions

The supplied experiments are designed to answer:

1. How well does SIC generalize to a user who contributed no training or calibration data?
2. How much do 3-, 6-, 9-, and 12-shot user calibration change predictive discrimination and probability calibration?
3. Does calibration improve Brier score consistently across users, or only improve thresholded metrics such as accuracy?
4. Do ERM, V-REx, and MLDG produce different zero-shot and post-calibration behavior?
5. How much does each major component contribute: GCN-GRU, BiLSTM, decoder/reconstruction, and median-label trials?
6. Do conclusions hold for both DREAMER valence and arousal?

## Input and preprocessing

The supplied scripts expect DREAMER features with:

- 23 subjects and 18 trials per subject;
- 14 EEG channels;
- theta, alpha, and beta bands, giving 42 channel-band features per timestep;
- 4-second non-overlapping windows; and
- global-RMS window normalization.

The MI adjacency used by the graph branch is estimated from source-subject data only. The held-out target subject is not used to construct the source graph.

## Model architecture

For each EEG window, SIC runs two independent encoders in parallel.

```text
                                   ┌─ mean_g, log_var_g ─► z_g ─┐
EEG ─► MI graph ─► GCN ─► GRU ─────┤                            │
                                   │                            ├─► concatenate
                                   │                            │       z_joint
EEG ─► BiLSTM ─► LayerNorm/dropout ├─ mean_b, log_var_b ─► z_b  ┘          │
      └─ temporal pooling ─────────┘                                      │
                                                                          ├─► classifier
                                                                          ├─► decoder
                                                                          └─► subject adversary
```

### GCN-GRU branch

The spatial/spectral branch uses a source-only mutual-information graph over EEG channels. Band-separated graph convolutions extract channel relationships, and the spectral GRU integrates information across frequency bands. In the supplied configuration, `spectral_gru_units=384` creates a 384-dimensional deterministic spectral representation before the branch posterior.

The branch then produces:

$
q_g(z_g\mid x)=\mathcal N\left(\mu_g,\operatorname{diag}(\sigma_g^2)\right).
$

### BiLSTM branch

The BiLSTM receives the original 42-dimensional channel-band sequence independently of the graph branch. A forward and backward LSTM model temporal context in both directions. `bilstm_units` is the width of each direction, so 63 units produce a 126-dimensional deterministic output.

Each BiLSTM block is followed by LayerNorm and dropout. Average pooling then aligns its time axis with the downsampled GCN-GRU sequence. The aligned representation produces its own posterior:

$
q_b(z_b\mid x)=\mathcal N\left(\mu_b,\operatorname{diag}(\sigma_b^2)\right).
$

LayerNorm remains enabled in the supplied experiments and is not currently an ablation.

### Independent variational latents

During source training, each branch uses reparameterized sampling:

$
z_g=\mu_g+\sigma_g\odot\epsilon_g,
\qquad
z_b=\mu_b+\sigma_b\odot\epsilon_b.
$

Their latent vectors are concatenated directly:

$
z_{joint}=[z_g;z_b].
$

`z_dim` is the width per active branch. With `z_dim=64` and both branches enabled, the classifier, decoder, and subject adversary receive 128 features. A single-branch ablation receives 64 features. No `fusion_units`, `fusion_dropout`, or `architecture_mode=serial` setting should be used.

The VAE KL term averages the KL divergence of the active branch posteriors:

$
L_{KL}=\frac{KL_g+KL_b}{2}
$

when both branches are enabled. `gcn_kl_loss`, `bilstm_kl_loss`, and their combined `kl_loss` are logged separately.

### Classifier, decoder, and subject adversary

The concatenated latent sequence is averaged over time. In window mode, the resulting window vector enters the dense classifier. In trial mode, window vectors are additionally averaged within each trial before classification.

The dense classifier uses focal classification loss and the existing VariationalClassifier regularizers. The graph-aware joint decoder reconstructs the EEG window from `z_joint`. A gradient-reversal subject head receives the pooled concatenated posterior means and encourages the representation to discard source-subject identity.

For ERM, the source objective is approximately:

$
L_{ERM}=
w_{VC}L_{focal+VC}
+w_{VAE}(L_{reconstruction}+\beta L_{KL})
+w_{subject}L_{subject}.
$

V-REx adds its subject-risk variance penalty to this objective. MLDG is not another ordinary additive loss: it uses the complete objective for the meta-train gradient, evaluates focal classification after a temporary update on virtual-unseen meta-test subjects, and combines those gradients for the persistent optimizer step.

The current v8 model does **not** implement supervised contrastive loss. Do not add `use_supcon` or `supcon_*` settings to its grid; unsupported builder arguments could otherwise be misleading.

## Source-training methods

### ERM

Empirical risk minimization trains on ordinary source batches and minimizes the pooled source objective. It does not explicitly penalize differences among users or simulate unseen users.

Use:

```bash
sbatch --export=ALL,TRAINING_METHOD=erm full_run_sic_mldg_brier_ablation_valence.sh
```

### V-REx

V-REx treats each source subject represented in a batch as an environment. It calculates focal risk for each subject and adds the variance of those risks:

$
L_{V\text{-}REx}=L_{source}+\lambda_{vrex}\operatorname{Var}(R_1,\ldots,R_S).
$

The goal is to discourage a solution that performs well for some source users and poorly for others. `VREX_PENALTY_WEIGHT` controls \(\lambda_{vrex}\).

Use:

```bash
sbatch --export=ALL,TRAINING_METHOD=vrex,VREX_PENALTY_WEIGHT=1.0 \
  full_run_sic_mldg_brier_ablation_valence.sh
```

### First-order MLDG

MLDG creates subject-disjoint episodes. Group A contains meta-train subjects and group B contains rotating virtual-unseen subjects.

For each episode:

1. Calculate the complete SIC objective on A, including focal/VC, VAE, and subject-adversarial losses.
2. Take a temporary inner step using `mldg_inner_learning_rate`.
3. Evaluate focal emotion classification on B using the temporarily adapted parameters.
4. Restore the original parameters.
5. Apply one persistent outer update combining the A gradient and the weighted B gradient.

The supplied configuration uses 18 A subjects, 4 B subjects, one complete trial per selected subject, and a configurable number of episodes per epoch. MLDG is the default:

```bash
sbatch full_run_sic_mldg_brier_ablation_valence.sh
```

or explicitly:

```bash
sbatch --export=ALL,TRAINING_METHOD=mldg,MLDG_STEPS_PER_EPOCH=10 \
  full_run_sic_mldg_brier_ablation_valence.sh
```

## Strict LOSO cross-validation

For every hyperparameter configuration, the outer evaluation iterates over target subjects:

1. Select one target subject and exclude all of that subject's data from source training.
2. Train a fresh source model on the other 22 DREAMER subjects.
3. Evaluate the untouched source model on all trials of the target subject. This is the strict zero-shot LOSO result.
4. Save the same source checkpoint for all calibration comparisons for that target.
5. Move to the next target subject and repeat.

The supplied Slurm workers set `validation_subjects=0` and disable early stopping. Every source fold therefore uses a fixed source-epoch budget rather than selecting an epoch using reserved source-validation subjects.

Hyperparameter selection can use either:

- `losocv`: aggregate zero-shot performance; or
- `calibration`: aggregate post-calibration performance at `CALIBRATION_SELECTION_SHOTS`.

The supplied full run selects the lowest 12-shot calibrated Brier score. Calibration is still executed and reported regardless of which selection level is chosen.

## User calibration

After training the strict LOSO source model, SIC evaluates several target-user shot levels. A shot is one complete target-user trial, not one EEG window.

For every target subject and requested `(shots, folds)` pair:

1. Create a seeded target-only calibration/evaluation split using complete trials.
2. Restore the unchanged source checkpoint.
3. Record paired zero-shot predictions on that fold's evaluation trials.
4. Freeze both encoders, both posterior heads, the decoder, the VC target, and the subject adversary.
5. Fine-tune only the requested suffix of the dense classifier on the calibration trials.
6. Re-evaluate the remaining target trials.
7. Report zero-shot, calibrated, and calibrated-minus-zero-shot metrics.

`calibration_unfreeze_layers=2` unfreezes the final classifier hidden block plus the logits layer. Each calibration fold starts from the same source checkpoint and a fresh optimizer; calibration folds do not inherit weights or optimizer state from one another.

The full script evaluates:

| Calibration level | Folds |
| ---: | ---: |
| 3 shots | 6 |
| 6 shots | 3 |
| 9 shots | 2 |
| 12 shots | 3 |

## Why Brier score is primary

For binary classification, Brier score is the mean squared error of the predicted probability:

$
\operatorname{Brier}=\frac{1}{N}\sum_{i=1}^{N}(p_i-y_i)^2.
$

Lower is better. Unlike accuracy, Brier score penalizes poorly calibrated confidence. This is important for user calibration: a model may preserve its predicted class while changing a probability from 0.55 to 0.90, which accuracy cannot distinguish.

The reports also include discrimination and thresholded metrics where defined:

- accuracy and balanced accuracy;
- F1, macro F1, precision, recall, macro precision, and macro recall;
- ROC-AUC;
- Brier score; and
- ECE.

Results are calculated at window and/or trial aggregation levels by the cross-validation pipeline. The supplied worker invokes `classification_level=window`, while prediction reports also retain trial identifiers and trial-level aggregation where produced.

## Code organization

### `sic_model.py`

Defines the neural architecture and optimization behavior:

- GCN-GRU and BiLSTM encoder branches;
- branch-specific mean/log-variance posterior heads;
- latent sampling and direct concatenation;
- dense classifier, joint decoder, and subject adversary;
- focal/VC, VAE, KL, V-REx, and MLDG loss logic;
- source, calibration, test, and Monte Carlo prediction steps; and
- calibration freezing/unfreezing behavior.

### `sic_model_train.py`

Runs the experiment:

- loads and normalizes DREAMER data;
- optionally removes median-rating trials;
- creates window- or trial-level inputs;
- executes one fixed configuration or a Cartesian hyperparameter grid;
- calls strict LOSO plus target-user calibration;
- ranks configurations using the requested metric and selection level; and
- writes JSON, CSV, predictions, and logs.

### `sic_model_args.py`

Owns configuration and parsing so the training module stays readable:

- CLI definitions;
- JSON parsing;
- fixed-versus-grid decoding;
- full Cartesian-grid expansion;
- data/model hyperparameter separation;
- argument and configuration validation; and
- construction of `SICTrainingConfig`.

### Slurm scripts

- `smoke_test_sic_mldg_brier.sh`: two-task smoke array (`full`, `no_bilstm`).
- `full_run_sic_mldg_brier_ablation_valence.sh`: five-task full ablation array.
- `submit_sic_slurm_suite.sh`: optional launcher that submits smoke, full, or a full array dependent on successful smoke completion.

## Required project layout

```text
EEGProc/
├── requirements.txt
├── datasets/remove_gamma/
│   ├── dreamer_eeg.npy
│   └── dreamer_labels.npy
├── smoke_test_sic_mldg_brier.sh
├── full_run_sic_mldg_brier_ablation_valence.sh
├── submit_sic_slurm_suite.sh
└── src/eegproc/deep_learning/
    ├── cross_val.py
    ├── supervised/variational_classifier.py
    ├── unsupervised/GNN/GCNMTL.py
    └── joint_architectures/sic/
        ├── sic_model.py
        ├── sic_model_train.py
        └── sic_model_args.py
```

The workers default to `$HOME/EEGProc` and `$HOME/EEGProc/venv312`. Override `PROJECT_DIR` or `VENV_DIR` at submission if needed.

## Running the Slurm scripts

The workers already contain `#SBATCH --array` directives, so a normal `sbatch` command creates multiple jobs automatically.

### Smoke test

```bash
sbatch smoke_test_sic_mldg_brier.sh
```

This submits two array tasks:

| Task | Profile |
| ---: | --- |
| 0 | `full` |
| 1 | `no_bilstm` |

The smoke defaults to two target subjects, three source epochs, three calibration epochs, and two MLDG episodes per epoch. Override them without editing the script:

```bash
sbatch --export=ALL,MAX_SUBJECTS=4,SOURCE_EPOCHS=10,CALIBRATION_EPOCHS=10,MLDG_STEPS_PER_EPOCH=10 \
  smoke_test_sic_mldg_brier.sh
```

### Full ablation suite

```bash
sbatch full_run_sic_mldg_brier_ablation_valence.sh
```

This submits five independent array tasks:

| Task | Profile | Change from full |
| ---: | --- | --- |
| 0 | `full` | Both branches and decoder; retain median-label trials |
| 1 | `no_gcn_gru` | Disable the spatial/spectral branch |
| 2 | `no_bilstm` | Disable the temporal branch |
| 3 | `no_decoder` | Disable reconstruction; retain branch KL losses |
| 4 | `remove_median` | Remove complete trials whose original target rating is 3 |

The header uses `#SBATCH --array=0-4%2`. Slurm creates five jobs but runs at most two simultaneously. Since each task requests two GPUs, the default concurrency requests at most four GPUs at a time.

Every profile with an active BiLSTM searches `bilstm_units` over 42, 63, and 96 units per direction. `no_bilstm` runs only once, avoiding duplicate width configurations.

### Valence and arousal

Valence is the default:

```bash
sbatch full_run_sic_mldg_brier_ablation_valence.sh
```

Run the same experiment for arousal with:

```bash
sbatch --export=ALL,TARGET_DIMENSION=arousal \
  full_run_sic_mldg_brier_ablation_valence.sh
```

The target dimension is incorporated into the run name and output directory, so valence and arousal results do not overwrite one another.

### Optional launcher

The launcher makes array throttling and dependencies explicit:

```bash
./submit_sic_slurm_suite.sh smoke 2
./submit_sic_slurm_suite.sh full 2
./submit_sic_slurm_suite.sh full-after-smoke 2
```

`full-after-smoke` submits both arrays immediately, but Slurm holds the full array until every smoke task exits successfully.

For arousal:

```bash
TARGET_DIMENSION=arousal ./submit_sic_slurm_suite.sh full-after-smoke 2
```

For V-REx:

```bash
TRAINING_METHOD=vrex VREX_PENALTY_WEIGHT=1.0 \
  ./submit_sic_slurm_suite.sh full 2
```

### Monitoring and restarting array tasks

Logs contain the array job ID (`%A`) and task ID (`%a`):

```text
smoke_sic_v8_<array-job-id>_<task-id>.out
smoke_sic_v8_<array-job-id>_<task-id>.err
sic_v8_abl_val_<array-job-id>_<task-id>.out
sic_v8_abl_val_<array-job-id>_<task-id>.err
```

Monitor jobs:

```bash
squeue -j <array-job-id>
sacct -j <array-job-id> --format=JobID,JobName,State,Elapsed,ExitCode
```

If only task 2 fails, resubmit only that profile:

```bash
sbatch --array=2 full_run_sic_mldg_brier_ablation_valence.sh
```

## Output directories and reports

Each array task writes to a disjoint directory containing the training method, DREAMER target, suite ID, and ablation profile:

```text
runs/full/sic_v8_<method>_brier_ablation/
└── DREAMER/<valence-or-arousal>/suite_<array-job-id>/<profile>/
```

The run root contains:

- `training.log`: chronological training and selection log;
- `training_config.json`: resolved CLI/runtime configuration;
- `model_config.json`: submitted fixed/grid model configuration;
- `hyperparameter_grid.json`: every expanded Cartesian configuration;
- `hyperparameter_search_progress.json`: completed, failed, and remaining configurations;
- `hyperparameter_search_results.json`: ranked configurations and best score;
- `best_hyperparameters.json`: selected configuration and its directory; and
- `hyperparameter_search_summary.csv`: one row per ranked configuration, including zero-shot and calibration aggregates.

Each `configuration_####/` directory contains:

- `model_config.json`: exact candidate hyperparameters;
- `dataset_ablation.json`: retained/removed trials and samples;
- `sic_calibration_results.json`: complete nested subject, shot, and fold results;
- `sic_overall_metrics.json`: aggregate zero-shot and calibration metrics;
- `sic_subject_summary.csv`: per-target-subject summaries;
- `sic_calibration_folds.csv`: one row per target/shot/fold, including paired zero-shot, calibrated, and delta metrics;
- `sic_window_predictions.csv` and `sic_trial_predictions.csv` when prediction logs are produced; and
- `sic_window_uncertainty.csv` and `sic_trial_uncertainty.csv` when variational interval logging is enabled.

Important keys in `sic_overall_metrics.json` include:

- `zero_shot_all_trials_mean_scores`: strict LOSO zero-shot mean across target users;
- `zero_shot_all_trials_std_scores`: cross-user zero-shot variation;
- `calibration_levels.<shots>.paired_zero_shot_mean_scores`: zero-shot scores on the exact evaluation trials used by that shot level;
- `calibration_levels.<shots>.calibrated_mean_scores`: post-calibration scores;
- `calibration_levels.<shots>.delta_mean_scores`: calibrated minus paired zero-shot scores; and
- corresponding standard-deviation groups.

Use `sic_calibration_folds.csv` to inspect whether calibration helped consistently across folds and users. Use the aggregate JSON or search-summary CSV to compare training methods, target dimensions, hyperparameters, and ablations.

## Main configurable hyperparameters

The Slurm scripts comment each important setting in place. The most consequential groups are:

| Group | Hyperparameters | Meaning |
| --- | --- | --- |
| GCN-GRU | `gcn_units`, `spectral_gru_units`, `gcn_dropout`, `t_down` | Spatial/spectral capacity and temporal reduction |
| BiLSTM | `bilstm_units`, `n_bilstm_layers`, `bilstm_dropout` | Independent temporal capacity and regularization |
| Branch posterior | `z_dim`, `vae_beta`, `vae_loss_weight` | Per-branch bottleneck and VAE strength |
| Classifier/VC | `classification_hidden_units`, `focal_gamma`, `vc_alpha/beta/gamma/lambda` | Emotion prediction and VC regularizers |
| Subject adversary | `subject_adversarial_weight`, `subject_loss_weight` | Gradient reversal and adversarial-loss scale |
| MLDG | `mldg_meta_*`, `mldg_trials_per_subject`, `mldg_steps_per_epoch` | Episode composition and outer generalization gradient |
| V-REx | `vrex_penalty_weight` | Strength of cross-subject risk-variance penalty |
| Calibration | shot/fold pairs, epochs, learning rate, `calibration_unfreeze_layers` | Amount and scope of target-user adaptation |

## Experiment-size warning

The full experiment is intentionally large. Four ablation profiles search three BiLSTM widths and `no_bilstm` runs once, for 13 source configuration searches per target dimension and training method. Every configuration performs strict LOSO and all requested calibration folds. Running valence and arousal under ERM, V-REx, and MLDG multiplies that workload further. Use the smoke array first and control concurrency with the array throttle or launcher.
