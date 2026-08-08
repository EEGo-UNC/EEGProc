"""CLI parsing and JSON/CSV serialization helpers for the STS training entry point."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def _add_bool_pair(
    parser: argparse.ArgumentParser,
    positive_flag: str,
    negative_flag: str,
    destination: str,
    default: bool,
    positive_help: str,
    negative_help: str,
) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        positive_flag,
        dest=destination,
        action="store_true",
        help=positive_help,
    )
    group.add_argument(
        negative_flag,
        dest=destination,
        action="store_false",
        help=negative_help,
    )
    parser.set_defaults(**{destination: default})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the alternating spatiotemporal BiLSTM + spatiospectral GCN "
            "fused VAE classifier with a dual-path decoder."
        )
    )
    parser.add_argument("--out-dir", default="runs/joint_sts")
    parser.add_argument("--run-name", default="joint_sts")
    parser.add_argument(
        "--dataset",
        choices=("dreamer", "amigos", "eegemotions_27"),
        default="dreamer",
    )
    parser.add_argument("--n-channels", type=int, default=14)
    parser.add_argument(
        "--n-bands",
        type=int,
        default=3,
        help="Features per electrode; DREAMER theta/alpha/beta preprocessing uses 3.",
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--optimizer",
        choices=("adam", "adamw"),
        default="adamw",
    )
    parser.add_argument("--classification-learning-rate", type=float, default=1e-4)
    parser.add_argument("--vae-learning-rate", type=float, default=5e-5)
    parser.add_argument("--discriminator-learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--classification-steps-per-batch", type=int, default=1)
    parser.add_argument("--vae-steps-per-batch", type=int, default=1)

    parser.add_argument(
        "--cv-strategy",
        choices=("loso", "lnskto"),
        default="loso",
    )
    parser.add_argument("--lnskto-subjects", type=int, default=3)
    parser.add_argument("--lnskto-trials", type=int, default=3)
    parser.add_argument("--lnskto-split-seed", type=int, default=42)
    _add_bool_pair(
        parser,
        "--lnskto-require-all-classes",
        "--lnskto-allow-single-class-folds",
        "lnskto_require_all_classes",
        True,
        "Require every LNSKTO test fold to contain all classes.",
        "Permit single-class LNSKTO test folds.",
    )
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--final-epochs", type=int, default=None)
    parser.add_argument(
        "--final-epoch-strategy",
        choices=("median", "mean", "max"),
        default="median",
    )
    _add_bool_pair(
        parser,
        "--run-no-validation-loso-before-final",
        "--skip-no-validation-loso-before-final",
        "run_no_validation_loso_before_final",
        True,
        (
            "Run a fixed-config LOSOCV diagnostic using the selected "
            "hyperparameters, derived epoch count, all 22 non-test subjects, "
            "and no validation set before fitting the final all-subject model."
        ),
        "Skip the fixed-config no-validation LOSOCV diagnostic.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selection-metric",
        choices=(
            "loss",
            "joint_loss",
            "accuracy",
            "f1",
            "precision",
            "recall",
            "macro_f1",
            "macro_precision",
            "macro_recall",
            "balanced_accuracy",
        ),
        default="f1",
    )
    parser.add_argument(
        "--selection-level",
        choices=("window", "trial"),
        default="trial",
    )

    parser.add_argument("--prediction-latent-samples", type=int, default=0)
    parser.add_argument("--latent-sampling-seed", type=int, default=None)
    parser.add_argument(
        "--decision-thresholds",
        type=float,
        nargs="+",
        default=[0.5],
    )
    parser.add_argument(
        "--threshold-selection-metric",
        choices=("accuracy", "f1", "balanced_accuracy", "binary_f1"),
        default="f1",
    )
    parser.add_argument(
        "--threshold-selection-level",
        choices=("window", "trial"),
        default="trial",
    )
    parser.add_argument("--no-prediction-diagnostics", action="store_true")
    parser.add_argument("--prediction-diagnostics-every", type=int, default=1)
    parser.add_argument("--prediction-diagnostics-samples", type=int, default=256)
    parser.add_argument("--prediction-threshold-tolerance", type=float, default=0.01)
    parser.add_argument("--prediction-diagnostics-seed", type=int, default=42)

    parser.add_argument("--validation-subjects", type=int, default=2)
    parser.add_argument("--validation-seed", type=int, default=None)
    _add_bool_pair(
        parser,
        "--alternate-subject-sets",
        "--no-alternate-subject-sets",
        "alternate_subject_sets",
        False,
        (
            "Split each LOSO training pool into two disjoint subject sets and "
            "alternate optimizer batches between them. Requires zero validation subjects."
        ),
        "Use ordinary shuffled minibatches across all training subjects.",
    )
    parser.add_argument("--alternating-subject-seed", type=int, default=42)
    _add_bool_pair(
        parser,
        "--use-mldg",
        "--no-mldg",
        "use_mldg",
        False,
        (
            "Enable first-order MLDG episodes over disjoint meta-train A and "
            "pseudo-unseen meta-test B subjects."
        ),
        "Disable first-order MLDG and use the selected ordinary training mode.",
    )
    parser.add_argument("--mldg-inner-learning-rate", type=float, default=1e-4)
    parser.add_argument("--mldg-meta-test-weight", type=float, default=1.0)
    parser.add_argument("--mldg-meta-train-subjects", type=int, default=6)
    parser.add_argument("--mldg-meta-test-subjects", type=int, default=2)
    parser.add_argument("--mldg-samples-per-subject", type=int, default=4)
    parser.add_argument("--mldg-seed", type=int, default=42)
    parser.add_argument("--no-early-stopping", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    parser.add_argument("--early-stopping-monitor", default="val_accuracy")
    parser.add_argument(
        "--early-stopping-mode",
        choices=("auto", "min", "max"),
        default="max",
    )
    parser.add_argument("--outer-verbose", type=int, default=0)
    parser.add_argument("--final-verbose", type=int, default=1)
    parser.add_argument("--no-save-full-model", action="store_true")
    parser.add_argument("--no-save-weights", action="store_true")
    parser.add_argument("--no-save-final-history-csv", action="store_true")
    parser.add_argument("--no-save-adjacency-matrices", action="store_true")

    parser.add_argument("--t-down", type=int, default=2)
    parser.add_argument("--temporal-pool-sizes", type=int, nargs="+", default=[2])
    parser.add_argument("--bilstm-units", type=int, default=64)
    parser.add_argument("--bilstm-layers", type=int, default=1)
    parser.add_argument("--bilstm-dropout", type=float, default=0.30)
    parser.add_argument("--temporal-emb-dim", type=int, default=32)
    parser.add_argument("--gcn-units", type=int, nargs="+", default=[64, 32])
    parser.add_argument("--spectral-emb-dim", type=int, default=32)
    parser.add_argument("--gcn-dropout", type=float, default=0.20)
    parser.add_argument("--gcn-activation", default="relu")
    parser.add_argument("--gcn-use-batch-norm", action="store_true")
    parser.add_argument("--graph-self-loop-bias", type=float, default=2.0)
    parser.add_argument("--graph-identity-mix", type=float, default=0.0)
    parser.add_argument("--graph-adjacency-reg-weight", type=float, default=1e-4)

    parser.add_argument("--fusion-dim", type=int, default=64)
    parser.add_argument("--latent-features", type=int, default=32)
    parser.add_argument("--fusion-dropout", type=float, default=0.20)
    parser.add_argument("--activation", default="relu")

    parser.add_argument("--decoder-temporal-units", type=int, default=64)
    parser.add_argument("--decoder-bilstm-layers", type=int, default=1)
    parser.add_argument("--decoder-graph-output-units", type=int, default=16)
    parser.add_argument("--decoder-branch-feature-dim", type=int, default=64)
    parser.add_argument("--decoder-fusion-units", type=int, default=64)
    parser.add_argument("--decoder-dropout", type=float, default=0.20)
    parser.add_argument(
        "--reconstruction-loss",
        choices=("mse", "mae", "huber"),
        default="mse",
    )

    parser.add_argument("--classification-hidden-units", type=int, default=64)
    parser.add_argument("--classification-dropout", type=float, default=0.30)
    parser.add_argument(
        "--classifier-head",
        choices=("dense", "hybrid", "variational"),
        default="dense",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--label-smoothing-levels", type=float, nargs="+", default=None)
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=1.0,
        help="Focal focusing parameter; 0 exactly recovers cross-entropy.",
    )
    parser.add_argument(
        "--focal-alpha",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional focal class weights. Pass one value per class; a single "
            "value is applied uniformly. Keras class_weight still multiplies "
            "the resulting per-sample focal loss when enabled."
        ),
    )
    parser.add_argument("--classification-loss-weight", type=float, default=1.0)
    parser.add_argument("--vae-loss-weight", type=float, default=1.0)
    parser.add_argument("--vae-beta", type=float, default=0.30)
    parser.add_argument("--vc-alpha", type=float, default=1.0)
    parser.add_argument("--vc-beta", type=float, default=0.0)
    parser.add_argument("--vc-gamma", type=float, default=0.0)
    parser.add_argument("--vc-lambda", type=float, default=0.0)
    parser.add_argument("--update-discriminator", action="store_true")

    _add_bool_pair(
        parser,
        "--use-subject-adversarial",
        "--no-subject-adversarial",
        "use_subject_adversarial",
        False,
        "Enable fold-local subject adversarial training.",
        "Disable subject adversarial training.",
    )
    parser.add_argument("--subject-adversarial-weight", type=float, default=0.05)
    parser.add_argument("--subject-loss-weight", type=float, default=1.0)
    parser.add_argument("--subject-hidden-units", type=int, default=64)
    parser.add_argument("--subject-dropout", type=float, default=0.0)
    parser.add_argument(
        "--subject-latent-mode",
        choices=("mean", "mc"),
        default="mean",
    )
    parser.add_argument("--subject-mc-samples", type=int, default=5)

    _add_bool_pair(
        parser,
        "--use-supcon",
        "--no-supcon",
        "use_supcon",
        False,
        "Enable supervised contrastive regularization.",
        "Disable supervised contrastive regularization.",
    )
    parser.add_argument("--supcon-weight", type=float, default=0.03)
    parser.add_argument("--supcon-temperature", type=float, default=0.10)
    _add_bool_pair(
        parser,
        "--supcon-cross-subject-only",
        "--supcon-all-same-class-positives",
        "supcon_cross_subject_only",
        True,
        "Require same-label SupCon positives to come from different subjects.",
        "Use every same-label non-self sample as a SupCon positive.",
    )

    parser.add_argument("--hyperparameters-json", default=None)
    parser.add_argument("--features-npy", default=None)
    parser.add_argument("--labels-npy", default=None)
    parser.add_argument("--subjects-npy", default=None)
    parser.add_argument("--trials-npy", default=None)
    parser.add_argument("--raw-eeg-npy", default=None)
    parser.add_argument("--raw-labels-npy", default=None)
    parser.add_argument(
        "--label-dimension",
        choices=("valence", "arousal"),
        default="valence",
    )
    parser.add_argument("--window-sec", type=float, default=4.0)
    parser.add_argument("--window-overlap", type=float, default=0.5)
    parser.add_argument("--fs", type=float, default=30.0)
    parser.add_argument("--median-label", type=float, default=0.0)
    parser.add_argument(
        "--window-normalization",
        choices=("none", "global_rms", "feature_zscore"),
        default="global_rms",
    )
    parser.add_argument("--no-zscore", action="store_true")
    parser.add_argument(
        "--label-threshold-mode",
        choices=("global", "subject_median"),
        default="global",
    )
    _add_bool_pair(
        parser,
        "--use-class-weight",
        "--no-class-weight",
        "use_class_weight",
        True,
        "Enable fold-local inverse-frequency class weighting.",
        "Disable class weighting.",
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--cpus-per-worker", type=int, default=None)
    return parser.parse_args(argv)


def _positive_int_tuple(
    name: str,
    value,
    *,
    allow_empty: bool = False,
) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple, got {value!r}.")
    if not value and not allow_empty:
        raise ValueError(f"{name} must be non-empty.")
    normalized = tuple(int(item) for item in value)
    if any(item < 1 for item in normalized):
        raise ValueError(f"Every {name} value must be >= 1.")
    return normalized


def _validate_temporal_pooling(t_down: int, temporal_pool_sizes) -> tuple[int, ...]:
    pools = _positive_int_tuple(
        "temporal_pool_sizes",
        temporal_pool_sizes,
        allow_empty=True,
    )
    effective = int(np.prod(pools, dtype=np.int64)) if pools else 1
    if int(t_down) != effective:
        raise ValueError(
            f"t_down={t_down}, but temporal_pool_sizes={pools} produces {effective}."
        )
    return pools


def _validate_args(args: argparse.Namespace, hyperparameters: dict) -> None:
    positive_fields = {
        "classification_learning_rate": args.classification_learning_rate,
        "vae_learning_rate": args.vae_learning_rate,
        "mldg_inner_learning_rate": args.mldg_inner_learning_rate,
        "mldg_meta_train_subjects": args.mldg_meta_train_subjects,
        "mldg_meta_test_subjects": args.mldg_meta_test_subjects,
        "mldg_samples_per_subject": args.mldg_samples_per_subject,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "classification_steps_per_batch": args.classification_steps_per_batch,
        "vae_steps_per_batch": args.vae_steps_per_batch,
        "n_channels": args.n_channels,
        "t_down": args.t_down,
        "bilstm_units": args.bilstm_units,
        "bilstm_layers": args.bilstm_layers,
        "temporal_emb_dim": args.temporal_emb_dim,
        "spectral_emb_dim": args.spectral_emb_dim,
        "fusion_dim": args.fusion_dim,
        "latent_features": args.latent_features,
        "decoder_temporal_units": args.decoder_temporal_units,
        "decoder_bilstm_layers": args.decoder_bilstm_layers,
        "decoder_graph_output_units": args.decoder_graph_output_units,
        "decoder_branch_feature_dim": args.decoder_branch_feature_dim,
        "decoder_fusion_units": args.decoder_fusion_units,
        "classification_hidden_units": args.classification_hidden_units,
        "subject_hidden_units": args.subject_hidden_units,
        "subject_mc_samples": args.subject_mc_samples,
        "prediction_diagnostics_every": args.prediction_diagnostics_every,
        "prediction_diagnostics_samples": args.prediction_diagnostics_samples,
    }
    for name, value in positive_fields.items():
        if value < 1 if isinstance(value, int) else value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.discriminator_learning_rate is not None and args.discriminator_learning_rate <= 0:
        raise ValueError("--discriminator-learning-rate must be positive.")
    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay must be non-negative.")
    if args.n_bands is not None and args.n_bands < 1:
        raise ValueError("--n-bands must be positive.")
    for name, value in (
        ("bilstm_dropout", args.bilstm_dropout),
        ("gcn_dropout", args.gcn_dropout),
        ("fusion_dropout", args.fusion_dropout),
        ("decoder_dropout", args.decoder_dropout),
        ("classification_dropout", args.classification_dropout),
        ("subject_dropout", args.subject_dropout),
        ("label_smoothing", args.label_smoothing),
    ):
        if not 0.0 <= value < 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1).")
    if args.label_smoothing_levels is not None and any(
        not 0.0 <= value < 1.0 for value in args.label_smoothing_levels
    ):
        raise ValueError("All label-smoothing levels must be in [0, 1).")
    if args.focal_alpha is not None:
        if any(not np.isfinite(value) or value < 0.0 for value in args.focal_alpha):
            raise ValueError("All --focal-alpha values must be finite and non-negative.")
        if not any(value > 0.0 for value in args.focal_alpha):
            raise ValueError("At least one --focal-alpha value must be positive.")
    nonnegative_fields = {
        "focal_gamma": args.focal_gamma,
        "classification_loss_weight": args.classification_loss_weight,
        "vae_loss_weight": args.vae_loss_weight,
        "vae_beta": args.vae_beta,
        "vc_alpha": args.vc_alpha,
        "vc_beta": args.vc_beta,
        "vc_gamma": args.vc_gamma,
        "vc_lambda": args.vc_lambda,
        "subject_adversarial_weight": args.subject_adversarial_weight,
        "subject_loss_weight": args.subject_loss_weight,
        "supcon_weight": args.supcon_weight,
        "mldg_meta_test_weight": args.mldg_meta_test_weight,
        "graph_self_loop_bias": args.graph_self_loop_bias,
        "graph_adjacency_reg_weight": args.graph_adjacency_reg_weight,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "prediction_threshold_tolerance": args.prediction_threshold_tolerance,
    }
    for name, value in nonnegative_fields.items():
        if value < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative.")
    if args.classification_loss_weight == 0.0:
        raise ValueError("--classification-loss-weight must be greater than zero.")
    if args.vae_loss_weight == 0.0:
        raise ValueError("--vae-loss-weight must be greater than zero.")
    if args.supcon_temperature <= 0.0:
        raise ValueError("--supcon-temperature must be positive.")
    if not 0.0 <= args.graph_identity_mix <= 1.0:
        raise ValueError("--graph-identity-mix must be in [0, 1].")
    _validate_temporal_pooling(args.t_down, args.temporal_pool_sizes)
    _positive_int_tuple("gcn_units", args.gcn_units)
    if args.prediction_latent_samples < 0:
        raise ValueError("--prediction-latent-samples must be non-negative.")
    if args.validation_subjects < 0:
        raise ValueError("--validation-subjects must be non-negative.")
    if args.alternate_subject_sets and args.use_mldg:
        raise ValueError(
            "--alternate-subject-sets and --use-mldg are mutually exclusive."
        )
    if args.use_mldg and args.cv_strategy != "loso":
        raise ValueError("--use-mldg currently requires --cv-strategy loso.")
    if args.mldg_seed is not None and args.mldg_seed < 0:
        raise ValueError("--mldg-seed must be non-negative or omitted.")
    if args.alternate_subject_sets and args.validation_subjects != 0:
        raise ValueError(
            "--alternate-subject-sets requires --validation-subjects 0."
        )
    if args.alternate_subject_sets and not args.no_early_stopping:
        raise ValueError(
            "--alternate-subject-sets requires --no-early-stopping because no "
            "validation split is used."
        )
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience must be non-negative.")
    if args.lnskto_subjects < 1 or args.lnskto_trials < 1:
        raise ValueError("LNSKTO subject and trial counts must be positive.")
    if not args.decision_thresholds:
        raise ValueError("--decision-thresholds must not be empty.")
    thresholds = [float(value) for value in args.decision_thresholds]
    if any(not 0.0 < value < 1.0 for value in thresholds):
        raise ValueError("Decision thresholds must lie strictly between 0 and 1.")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("Decision thresholds must not contain duplicates.")
    if not isinstance(hyperparameters, dict):
        raise ValueError("--hyperparameters-json must decode to an object.")


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if tf.is_tensor(value):
        return value.numpy().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
