#!/bin/bash

MODEL_CONFIG="$(python - <<PY
import json
print(json.dumps({
    "optimizer_name": "adamw",
    "learning_rate": 1e-4,
    "weight_decay": 5e-5,
    "t_down": 2,
    "temporal_pool_sizes": [2],
    "gcn_units": [32],
    "gcn_dropout": 0.20,
    "gcn_activation": "relu",
    "gcn_use_batch_norm": False,
    "spectral_gru_units": 384,
    "spectral_gru_dropout": 0.2,
    "mi_n_neighbors": 3,
    "mi_random_state": 42,
    "mi_zero_diagonal": False,
    "mi_band_reduction": "mean",
    "mi_max_observations": 15000,
    "bilstm_units": 128,
    "n_bilstm_layers": 1,
    "bilstm_dropout": 0.3,
    "architecture_mode": "feature_fusion",
    "fusion_units": 128,
    "fusion_dropout": 0.2,
    "focal_gamma": 1.0,
    "focal_alpha": None,
    "z_dim": 128,
    "classification_hidden_units": [128, 64],
    "classification_dropout": 0.20,
    "vc_loss_weight": 1.0,
    "vc_alpha": 1.0,
    "vc_beta": 0.5,
    "vc_gamma": 0.0,
    "vc_lambda": 0.0,
    "update_vc_discriminator": False,
    "vae_loss_weight": 0.10,
    "vae_beta": 0.05,
    "decoder_dropout": 0.10,
    "use_vrex": False,
    "vrex_penalty_weight": 0.0,
    "use_subject_adversarial": True,
    "subject_adversarial_weight": 0.6,
    "subject_loss_weight": 1.0,
    "subject_hidden_units": 64,
    "subject_dropout": 0.0,
    "calibration_unfreeze_layers": 2,
    "calibration_use_vc_target": True,
    "use_class_weight": False
}))
PY
)"

python -m src.eegproc.deep_learning.joint_architectures.sic.sic_model_train \
    --training-protocol loso_validation \
    --raw-eeg-npy datasets/remove_gamma/dreamer_eeg.npy \
    --raw-labels-npy datasets/remove_gamma/dreamer_labels.npy \
    --label-dimension valence \
    --classification-level window \
    --n-channels 14 \
    --n-bands 3 \
    --out-dir runs/smoke/sic_normal_validation/DREAMER/valence \
    --run-name dreamer_valence_sic_normal_validation_smoke \
    --source-epochs 100 \
    --source-batch-size 64 \
    --validation-subjects 6 \
    --validation-seed 42 \
    --early-stopping-patience 20 \
    --early-stopping-min-delta 0.001 \
    --early-stopping-monitor val_balanced_accuracy \
    --early-stopping-mode min \
    --decision-threshold 0.5 \
    --prediction-latent-samples 15 \
    --latent-sampling-seed 42 \
    --max-subjects 4 \
    --n-jobs 2 \
    --gpu-ids 0 1 \
    --cpus-per-worker 2 \
    --verbose 2 \
    --seed 42 \
    --label-threshold-mode global \
    --median-label 3 \
    --window-sec 1.0 \
    --window-overlap 0.0 \
    --window-normalization global_rms \
    --hyperparameters-json "$MODEL_CONFIG"
