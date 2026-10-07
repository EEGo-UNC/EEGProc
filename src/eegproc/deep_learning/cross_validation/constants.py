"""Shared constants for the cross-validation package."""

from __future__ import annotations


_FIT_RESERVED_KEYS = frozenset({"epochs", "batch_size"})


_CLASSIFICATION_METRICS = frozenset(
    {
        "accuracy",
        "f1",
        "precision",
        "recall",
        # Class-balanced alternatives retained for diagnostics.
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "balanced_accuracy",
        "roc_auc",
        "brier_score",
        "ece",
    }
)


_DEFAULT_ECE_BINS = 15


_DECODER_SCORE_NAMES = (
    "reconstruction_loss",
    "decoder_r2",
    "gcn_gru_reconstruction_loss",
    "gcn_gru_decoder_r2",
    "bilstm_reconstruction_loss",
    "bilstm_decoder_r2",
    # Retained for older joint models that used this R2 alias.
    "decoder_accuracy",
)


_DEFAULT_SEQUENCE_HYPERPARAMETER_DEPTHS = {
    "conv_filters": 1,
    "kernel_sizes": 1,
    "pool_after_layers": 1,
    "pool_sizes": 1,
    "gcn_units": 1,
    "temporal_pool_sizes": 1,
    # Dense classifier layer widths are one architecture value, e.g.
    # [128, 64] means Dense(128) -> Dense(64), not two grid candidates.
    # Multiple classifier architectures can still be supplied with one
    # additional nesting level, e.g. [[128, 64], [256, 128]].
    "classification_hidden_units": 1,
    "spatial_pool_sizes": 2,
}


_EMPTY_SEQUENCE_ALLOWED_KEYS = frozenset(
    {
        "pool_after_layers",
        "pool_sizes",
        "spatial_pool_sizes",
    }
)


_JOINT_LOSS_WEIGHT_KEYS = frozenset(
    {
        "ae_loss_weight",
        "vc_loss_weight",
        "vae_beta",
        "vc_alpha",
        "vc_beta",
        "vc_gamma",
        "vc_lambda",
        "subject_loss_weight",
        "mldg_meta_test_weight",
        "use_subject_adversarial",
        # "label_smoothing",
    }
)
