"""CLI for joint_v4_sts."""

from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import tensorflow as tf

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Band-separated GCN -> BiLSTM -> classifier.")
    p.add_argument("--out-dir", default="runs/joint_v4_sts")
    p.add_argument("--run-name", default="dreamer_arousal_joint_v4_sts")
    p.add_argument("--dataset", choices=("dreamer","amigos","eegemotions_27"), default="dreamer")
    p.add_argument("--raw-eeg-npy", default=None)
    p.add_argument("--raw-labels-npy", default=None)
    p.add_argument("--label-dimension", choices=("valence","arousal"), default="valence")
    p.add_argument("--label-threshold-mode", choices=("global","subject_median"), default="global")
    p.add_argument("--median-label", type=float, default=3.0)
    p.add_argument("--window-sec", type=float, default=4.0)
    p.add_argument("--window-overlap", type=float, default=0.0)
    p.add_argument("--fs", type=float, default=30.0)
    p.add_argument("--window-normalization", choices=("none","global_rms","feature_zscore"), default="global_rms")
    p.add_argument("--n-channels", type=int, default=14)
    p.add_argument("--n-bands", type=int, default=3)
    p.add_argument("--classification-level", choices=("window","trial"), default="trial",
                   help="window: one prediction per EEG window; trial: GCN per window then BiLSTM across windows.")

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--optimizer", choices=("adam","adamw"), default="adamw")
    p.add_argument("--classification-learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    p.add_argument("--t-down", type=int, default=2)
    p.add_argument("--temporal-pool-sizes", type=int, nargs="+", default=[2])
    p.add_argument("--gcn-units", type=int, nargs="+", default=[128,64])
    p.add_argument("--spectral-emb-dim", type=int, default=128)
    p.add_argument("--gcn-dropout", type=float, default=0.2)
    p.add_argument("--gcn-activation", default="relu")
    p.add_argument("--gcn-use-batch-norm", action="store_true")
    p.add_argument("--graph-self-loop-bias", type=float, default=2.0)
    p.add_argument("--graph-identity-mix", type=float, default=0.0)
    p.add_argument("--graph-adjacency-reg-weight", type=float, default=1e-4)

    p.add_argument("--bilstm-units", type=int, default=256)
    p.add_argument("--bilstm-layers", type=int, default=1)
    p.add_argument("--bilstm-dropout", type=float, default=0.3)
    p.add_argument("--bilstm-emb-dim", type=int, default=64)
    p.add_argument("--classification-hidden-units", type=int, default=128)
    p.add_argument("--classification-dropout", type=float, default=0.3)
    p.add_argument("--activation", default="relu")
    p.add_argument("--focal-gamma", type=float, default=0.0)
    p.add_argument("--focal-alpha", type=float, nargs="+", default=None)

    # Auxiliary heads. These defaults may also be overridden per CV grid via
    # use_vae/use_subject_adversarial in --hyperparameters-json.
    p.add_argument("--use-vae", dest="use_vae", action="store_true")
    p.add_argument("--no-vae", dest="use_vae", action="store_false")
    p.set_defaults(use_vae=False)
    p.add_argument("--vae-loss-weight", type=float, default=0.1)
    p.add_argument("--vae-beta", type=float, default=0.05)
    p.add_argument("--vae-learning-rate", type=float, default=5e-5)

    p.add_argument("--use-subject-adversarial", dest="use_subject_adversarial", action="store_true")
    p.add_argument("--no-subject-adversarial", dest="use_subject_adversarial", action="store_false")
    p.set_defaults(use_subject_adversarial=False)
    p.add_argument("--subject-adversarial-weight", type=float, default=0.3)
    p.add_argument("--subject-loss-weight", type=float, default=0.3)
    p.add_argument("--subject-hidden-units", type=int, default=64)
    p.add_argument("--subject-dropout", type=float, default=0.0)

    # MLDG is deliberately run-level because it changes batch construction.
    p.add_argument("--use-mldg", dest="use_mldg", action="store_true")
    p.add_argument("--no-mldg", dest="use_mldg", action="store_false")
    p.set_defaults(use_mldg=False)
    p.add_argument("--mldg-inner-learning-rate", type=float, default=1e-4)
    p.add_argument("--mldg-meta-test-weight", type=float, default=1.0)
    p.add_argument("--mldg-meta-train-subjects", type=int, default=6)
    p.add_argument("--mldg-meta-test-subjects", type=int, default=2)
    p.add_argument("--mldg-samples-per-subject", type=int, default=4)
    p.add_argument("--mldg-seed", type=int, default=42)

    p.add_argument("--cv-strategy", choices=("loso",), default="loso")
    p.add_argument("--max-folds", type=int, default=None)
    p.add_argument("--validation-subjects", type=int, default=2)
    p.add_argument("--validation-seed", type=int, default=None)
    p.add_argument("--no-early-stopping", action="store_true")
    p.add_argument("--early-stopping-patience", type=int, default=20)
    p.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    p.add_argument("--early-stopping-monitor", default="val_accuracy")
    p.add_argument("--early-stopping-mode", choices=("auto","min","max"), default="max")
    p.add_argument("--final-epochs", type=int, default=None)
    p.add_argument("--final-epoch-strategy", choices=("median","mean","max"), default="median")
    p.add_argument("--skip-no-validation-loso-before-final", action="store_true")

    p.add_argument("--selection-level", choices=("window","trial"), default="trial")
    p.add_argument("--selection-metric",
        choices=("loss","joint_loss","accuracy","f1","precision","recall",
                 "macro_f1","macro_precision","macro_recall","balanced_accuracy"),
        default="accuracy")
    p.add_argument("--decision-thresholds", type=float, nargs="+", default=[0.5])
    p.add_argument("--threshold-selection-level", choices=("window","trial"), default="trial")
    p.add_argument("--threshold-selection-metric",
        choices=("accuracy","f1","balanced_accuracy","binary_f1"), default="accuracy")

    p.add_argument("--use-class-weight", dest="use_class_weight", action="store_true")
    p.add_argument("--no-class-weight", dest="use_class_weight", action="store_false")
    p.set_defaults(use_class_weight=False)

    p.add_argument("--prediction-latent-samples", type=int, default=0)
    p.add_argument("--latent-sampling-seed", type=int, default=None)
    p.add_argument("--no-prediction-diagnostics", action="store_true")
    p.add_argument("--prediction-diagnostics-every", type=int, default=1)
    p.add_argument("--prediction-diagnostics-samples", type=int, default=256)
    p.add_argument("--prediction-threshold-tolerance", type=float, default=0.01)
    p.add_argument("--prediction-diagnostics-seed", type=int, default=42)

    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--cpus-per-worker", type=int, default=None)
    p.add_argument("--outer-verbose", type=int, default=0)
    p.add_argument("--final-verbose", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-save-full-model", action="store_true")
    p.add_argument("--no-save-weights", action="store_true")
    p.add_argument("--no-save-final-history-csv", action="store_true")
    p.add_argument("--no-save-adjacency-matrices", action="store_true")
    p.add_argument("--hyperparameters-json", default=None)
    return p.parse_args(argv)

def validate_args(args, hparams):
    if args.n_channels < 1 or args.n_bands < 1:
        raise ValueError("n_channels and n_bands must be positive.")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive.")
    if args.classification_learning_rate <= 0:
        raise ValueError("classification learning rate must be positive.")
    if args.weight_decay < 0 or args.focal_gamma < 0:
        raise ValueError("weight_decay and focal_gamma must be non-negative.")
    for name, value in (
        ("gcn_dropout", args.gcn_dropout),
        ("bilstm_dropout", args.bilstm_dropout),
        ("classification_dropout", args.classification_dropout),
        ("subject_dropout", args.subject_dropout),
    ):
        if not 0 <= value < 1:
            raise ValueError(f"{name} must be in [0,1).")
    if args.bilstm_emb_dim < 1 or args.subject_hidden_units < 1:
        raise ValueError("bilstm_emb_dim and subject_hidden_units must be positive.")
    for name, value in (
        ("vae_loss_weight", args.vae_loss_weight),
        ("vae_beta", args.vae_beta),
        ("subject_adversarial_weight", args.subject_adversarial_weight),
        ("subject_loss_weight", args.subject_loss_weight),
        ("mldg_meta_test_weight", args.mldg_meta_test_weight),
    ):
        if value < 0:
            raise ValueError(f"{name} must be non-negative.")
    if args.vae_learning_rate <= 0 or args.mldg_inner_learning_rate <= 0:
        raise ValueError("VAE and MLDG learning rates must be positive.")
    if args.mldg_meta_train_subjects < 1 or args.mldg_meta_test_subjects < 1 or args.mldg_samples_per_subject < 1:
        raise ValueError("MLDG subject/sample counts must be positive.")
    if "use_mldg" in hparams:
        raise ValueError("use_mldg is run-level; use --use-mldg/--no-mldg, not hyperparameters JSON.")
    if int(np.prod(args.temporal_pool_sizes, dtype=np.int64)) != args.t_down:
        raise ValueError("t_down must equal product(temporal_pool_sizes).")
    if args.prediction_latent_samples != 0:
        raise ValueError("v4 classification is deterministic; prediction-latent-samples must be 0.")
    if not isinstance(hparams, dict):
        raise ValueError("hyperparameters-json must decode to an object.")

def json_default(value):
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if tf.is_tensor(value): return value.numpy().tolist()
    raise TypeError(type(value).__name__)

def write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=json_default)

def write_csv(path, rows):
    if not rows: return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
