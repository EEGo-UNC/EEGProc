"""CLI helpers for joint_v4_sts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def _bool_pair(
    parser,
    positive_flag,
    negative_flag,
    destination,
    default,
    positive_help,
    negative_help,
):
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Train joint_v4_sts: band-separated GCN -> BiLSTM -> classifier "
            "with optional VAE, subject adversity, and run-level MLDG."
        )
    )

    parser.add_argument("--out-dir", default="runs/joint_v4_sts")
    parser.add_argument("--run-name", default="dreamer_valence_joint_v4_sts")
    parser.add_argument(
        "--dataset",
        choices=("dreamer", "amigos", "eegemotions_27"),
        default="dreamer",
    )
    parser.add_argument("--raw-eeg-npy", default=None)
    parser.add_argument("--raw-labels-npy", default=None)
    parser.add_argument(
        "--label-dimension",
        choices=("valence", "arousal"),
        default="valence",
    )
    parser.add_argument(
        "--label-threshold-mode",
        choices=("global", "subject_median"),
        default="global",
    )
    parser.add_argument("--median-label", type=float, default=3.0)
    parser.add_argument("--window-sec", type=float, default=4.0)
    parser.add_argument("--window-overlap", type=float, default=0.0)
    parser.add_argument("--fs", type=float, default=30.0)
    parser.add_argument(
        "--window-normalization",
        choices=("none", "global_rms", "feature_zscore"),
        default="global_rms",
    )

    parser.add_argument("--n-channels", type=int, default=14)
    parser.add_argument("--n-bands", type=int, default=3)
    parser.add_argument(
        "--classification-level",
        choices=("window", "trial"),
        default="trial",
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--optimizer",
        choices=("adam", "adamw"),
        default="adamw",
    )
    parser.add_argument(
        "--classification-learning-rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--t-down", type=int, default=2)
    parser.add_argument(
        "--temporal-pool-sizes",
        type=int,
        nargs="+",
        default=[2],
    )
    parser.add_argument(
        "--gcn-units",
        type=int,
        nargs="+",
        default=[128, 64],
    )
    parser.add_argument("--spectral-emb-dim", type=int, default=128)
    parser.add_argument("--gcn-dropout", type=float, default=0.20)
    parser.add_argument("--gcn-activation", default="relu")
    parser.add_argument("--gcn-use-batch-norm", action="store_true")
    parser.add_argument("--graph-self-loop-bias", type=float, default=2.0)
    parser.add_argument("--graph-identity-mix", type=float, default=0.0)
    parser.add_argument(
        "--graph-adjacency-reg-weight",
        type=float,
        default=1e-4,
    )

    parser.add_argument("--bilstm-units", type=int, default=128)
    parser.add_argument("--bilstm-layers", type=int, default=1)
    parser.add_argument("--bilstm-dropout", type=float, default=0.30)
    parser.add_argument("--bilstm-emb-dim", type=int, default=64)
    parser.add_argument(
        "--classification-hidden-units",
        type=int,
        default=64,
    )
    parser.add_argument("--classification-dropout", type=float, default=0.30)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument(
        "--focal-alpha",
        type=float,
        nargs="+",
        default=None,
    )

    _bool_pair(
        parser,
        "--use-vae",
        "--no-vae",
        "use_vae",
        False,
        "Enable the auxiliary variational autoencoder.",
        "Disable the auxiliary variational autoencoder.",
    )
    parser.add_argument("--vae-loss-weight", type=float, default=0.10)
    parser.add_argument("--vae-beta", type=float, default=0.05)
    parser.add_argument("--vae-learning-rate", type=float, default=5e-5)

    _bool_pair(
        parser,
        "--use-subject-adversarial",
        "--no-subject-adversarial",
        "use_subject_adversarial",
        False,
        "Enable the subject-adversarial head.",
        "Disable the subject-adversarial head.",
    )
    parser.add_argument(
        "--subject-adversarial-weight",
        type=float,
        default=0.30,
    )
    parser.add_argument("--subject-loss-weight", type=float, default=0.30)
    parser.add_argument("--subject-hidden-units", type=int, default=64)
    parser.add_argument("--subject-dropout", type=float, default=0.0)

    _bool_pair(
        parser,
        "--use-mldg",
        "--no-mldg",
        "use_mldg",
        False,
        "Enable first-order subject-domain MLDG.",
        "Disable MLDG.",
    )
    parser.add_argument(
        "--mldg-inner-learning-rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--mldg-meta-test-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--mldg-meta-train-subjects",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--mldg-meta-test-subjects",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--mldg-samples-per-subject",
        type=int,
        default=4,
    )
    parser.add_argument("--mldg-seed", type=int, default=42)

    parser.add_argument("--cv-strategy", choices=("loso",), default="loso")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--validation-subjects", type=int, default=2)
    parser.add_argument("--validation-seed", type=int, default=None)
    parser.add_argument("--no-early-stopping", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--early-stopping-monitor",
        default="val_accuracy",
    )
    parser.add_argument(
        "--early-stopping-mode",
        choices=("auto", "min", "max"),
        default="max",
    )
    parser.add_argument("--final-epochs", type=int, default=None)
    parser.add_argument(
        "--final-epoch-strategy",
        choices=("median", "mean", "max"),
        default="median",
    )
    parser.add_argument(
        "--skip-no-validation-loso-before-final",
        action="store_true",
    )

    parser.add_argument(
        "--selection-level",
        choices=("window", "trial"),
        default="trial",
    )
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
        default="accuracy",
    )
    parser.add_argument(
        "--decision-thresholds",
        type=float,
        nargs="+",
        default=[0.5],
    )
    parser.add_argument(
        "--threshold-selection-level",
        choices=("window", "trial"),
        default="trial",
    )
    parser.add_argument(
        "--threshold-selection-metric",
        choices=("accuracy", "f1", "balanced_accuracy", "binary_f1"),
        default="accuracy",
    )

    parser.add_argument("--no-prediction-diagnostics", action="store_true")
    parser.add_argument(
        "--prediction-diagnostics-every",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--prediction-diagnostics-samples",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--prediction-threshold-tolerance",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--prediction-diagnostics-seed",
        type=int,
        default=42,
    )

    _bool_pair(
        parser,
        "--use-class-weight",
        "--no-class-weight",
        "use_class_weight",
        False,
        "Enable inverse-frequency class weighting.",
        "Disable class weighting.",
    )

    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--cpus-per-worker", type=int, default=None)
    parser.add_argument("--outer-verbose", type=int, default=0)
    parser.add_argument("--final-verbose", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--no-save-full-model", action="store_true")
    parser.add_argument("--no-save-weights", action="store_true")
    parser.add_argument(
        "--no-save-final-history-csv",
        action="store_true",
    )
    parser.add_argument(
        "--no-save-adjacency-matrices",
        action="store_true",
    )

    parser.add_argument("--hyperparameters-json", default=None)
    return parser.parse_args(argv)


def validate_args(args, hyperparameters):
    if args.n_channels < 1 or args.n_bands < 1:
        raise ValueError("n_channels and n_bands must be positive.")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive.")
    if args.classification_learning_rate <= 0.0:
        raise ValueError("classification-learning-rate must be positive.")
    if args.vae_learning_rate <= 0.0:
        raise ValueError("vae-learning-rate must be positive.")
    if args.weight_decay < 0.0:
        raise ValueError("weight-decay must be non-negative.")
    if args.t_down < 1:
        raise ValueError("t-down must be positive.")
    if not args.temporal_pool_sizes:
        effective_t_down = 1
    else:
        effective_t_down = int(
            np.prod(args.temporal_pool_sizes, dtype=np.int64)
        )
    if effective_t_down != args.t_down:
        raise ValueError(
            "t_down must equal product(temporal_pool_sizes)."
        )
    if any(value < 1 for value in args.gcn_units):
        raise ValueError("All GCN units must be positive.")
    for name, value in (
        ("gcn_dropout", args.gcn_dropout),
        ("bilstm_dropout", args.bilstm_dropout),
        ("classification_dropout", args.classification_dropout),
        ("subject_dropout", args.subject_dropout),
    ):
        if not 0.0 <= value < 1.0:
            raise ValueError(
                f"--{name.replace('_', '-')} must be in [0, 1)."
            )
    if args.bilstm_units < 1 or args.bilstm_layers < 1:
        raise ValueError("BiLSTM dimensions must be positive.")
    if args.bilstm_emb_dim < 1:
        raise ValueError("bilstm-emb-dim must be positive.")
    if args.classification_hidden_units < 1:
        raise ValueError("classification-hidden-units must be positive.")
    if args.subject_hidden_units < 1:
        raise ValueError("subject-hidden-units must be positive.")
    if args.focal_gamma < 0.0:
        raise ValueError("focal-gamma must be non-negative.")
    if args.vae_loss_weight < 0.0 or args.vae_beta < 0.0:
        raise ValueError("VAE weights must be non-negative.")
    if args.subject_adversarial_weight < 0.0:
        raise ValueError("subject-adversarial-weight must be non-negative.")
    if args.subject_loss_weight < 0.0:
        raise ValueError("subject-loss-weight must be non-negative.")
    if args.mldg_inner_learning_rate <= 0.0:
        raise ValueError("mldg-inner-learning-rate must be positive.")
    if args.mldg_meta_test_weight < 0.0:
        raise ValueError("mldg-meta-test-weight must be non-negative.")
    if args.mldg_meta_train_subjects < 1:
        raise ValueError("mldg-meta-train-subjects must be positive.")
    if args.mldg_meta_test_subjects < 1:
        raise ValueError("mldg-meta-test-subjects must be positive.")
    if args.mldg_samples_per_subject < 1:
        raise ValueError("mldg-samples-per-subject must be positive.")
    if args.validation_subjects < 0:
        raise ValueError("validation-subjects must be non-negative.")
    if not args.decision_thresholds:
        raise ValueError("decision-thresholds must not be empty.")
    if any(
        not 0.0 < float(value) < 1.0
        for value in args.decision_thresholds
    ):
        raise ValueError("Decision thresholds must lie in (0, 1).")
    if not isinstance(hyperparameters, dict):
        raise ValueError("--hyperparameters-json must decode to an object.")

    # MLDG changes the required batch structure. It is intentionally a
    # run-level switch and must not vary inside one hyperparameter grid.
    forbidden_grid_keys = {
        "use_mldg",
        "mldg_inner_learning_rate",
        "mldg_meta_test_weight",
        "mldg_meta_train_subjects",
        "mldg_meta_test_subjects",
        "mldg_samples_per_subject",
        "mldg_seed",
    }
    present = sorted(forbidden_grid_keys.intersection(hyperparameters))
    if present:
        raise ValueError(
            "MLDG settings are run-level CLI flags, not grid "
            f"hyperparameters. Remove: {present}"
        )


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if tf.is_tensor(value):
        return value.numpy().tolist()
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            default=json_default,
        )


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(
        dict.fromkeys(key for row in rows for key in row)
    )
    with Path(path).open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
