"""CLI for joint_v5_sts."""
from __future__ import annotations

import argparse


def _bool_pair(parser, positive, negative, dest, default):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(positive, dest=dest, action="store_true")
    group.add_argument(negative, dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train joint v5: MTLFuseNet-style GCN -> spectral GRU -> classifier."
    )
    p.add_argument("--out-dir", default="runs/joint_v5_sts")
    p.add_argument("--run-name", default="dreamer_valence_joint_v5_sts")
    p.add_argument("--dataset", default="dreamer", choices=("dreamer","amigos","eegemotions_27"))
    p.add_argument("--classification-level", choices=("window","trial"), default="window")
    p.add_argument("--n-channels", type=int, default=14)
    p.add_argument("--n-bands", type=int, default=3)

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--optimizer", choices=("adam","adamw"), default="adamw")
    p.add_argument("--classification-learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=5e-5)

    p.add_argument("--t-down", type=int, default=1)
    p.add_argument("--temporal-pool-sizes", type=int, nargs="*", default=[])
    p.add_argument("--gcn-units", type=int, nargs="+", default=[32])
    p.add_argument("--gcn-dropout", type=float, default=0.10)
    p.add_argument("--gcn-activation", default="relu")
    _bool_pair(p, "--gcn-use-batch-norm", "--no-gcn-use-batch-norm", "gcn_use_batch_norm", False)
    p.add_argument("--spectral-gru-units", type=int, default=384)
    p.add_argument("--spectral-gru-dropout", type=float, default=0.0)
    p.add_argument("--mi-n-neighbors", type=int, default=3)
    p.add_argument("--mi-random-state", type=int, default=42)
    _bool_pair(p, "--mi-zero-diagonal", "--mi-keep-diagonal", "mi_zero_diagonal", False)
    p.add_argument("--mi-band-reduction", choices=("mean","max","median"), default="mean")
    p.add_argument("--mi-max-observations", type=int, default=50000)

    p.add_argument("--classification-hidden-units", type=int, default=128)
    p.add_argument("--classification-dropout", type=float, default=0.30)
    p.add_argument("--activation", default="relu")
    p.add_argument("--focal-gamma", type=float, default=0.0)
    p.add_argument("--focal-alpha", type=float, nargs="+", default=None)
    _bool_pair(p, "--use-class-weight", "--no-class-weight", "use_class_weight", False)

    p.add_argument("--validation-subjects", type=int, default=4)
    p.add_argument("--validation-seed", type=int, default=None)
    p.add_argument("--early-stopping-patience", type=int, default=20)
    p.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    p.add_argument("--early-stopping-monitor", default="val_accuracy")
    p.add_argument("--early-stopping-mode", choices=("min","max","auto"), default="max")
    p.add_argument("--no-early-stopping", action="store_true")
    p.add_argument("--max-folds", type=int, default=None)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--cpus-per-worker", type=int, default=None)
    p.add_argument("--outer-verbose", type=int, default=0)
    p.add_argument("--final-verbose", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--selection-level", choices=("window","trial"), default="trial")
    p.add_argument("--selection-metric", choices=("accuracy","f1","precision","recall","macro_f1","macro_precision","macro_recall","balanced_accuracy","loss"), default="accuracy")
    p.add_argument("--decision-thresholds", type=float, nargs="+", default=[0.5])
    p.add_argument("--threshold-selection-level", choices=("window","trial"), default="trial")
    p.add_argument("--threshold-selection-metric", choices=("accuracy","f1","balanced_accuracy","binary_f1"), default="accuracy")
    p.add_argument("--final-epochs", type=int, default=None)
    p.add_argument("--final-epoch-strategy", choices=("median","mean","max"), default="median")
    p.add_argument("--skip-no-validation-loso-before-final", action="store_true")
    p.add_argument("--hyperparameters-json", default=None)

    p.add_argument("--raw-eeg-npy", default=None)
    p.add_argument("--raw-labels-npy", default=None)
    p.add_argument("--label-dimension", choices=("valence","arousal"), default="valence")
    p.add_argument("--window-sec", type=float, default=1.0)
    p.add_argument("--window-overlap", type=float, default=0.0)
    p.add_argument("--fs", type=float, default=128.0)
    p.add_argument("--median-label", type=float, default=3.0)
    p.add_argument("--window-normalization", choices=("none","global_rms","feature_zscore"), default="global_rms")
    p.add_argument("--label-threshold-mode", choices=("global","subject_median"), default="global")

    p.add_argument("--no-prediction-diagnostics", action="store_true")
    p.add_argument("--prediction-diagnostics-every", type=int, default=1)
    p.add_argument("--prediction-diagnostics-samples", type=int, default=256)
    p.add_argument("--prediction-threshold-tolerance", type=float, default=0.01)
    p.add_argument("--prediction-diagnostics-seed", type=int, default=42)

    p.add_argument("--no-save-full-model", action="store_true")
    p.add_argument("--no-save-weights", action="store_true")
    p.add_argument("--no-save-adjacency-matrices", action="store_true")
    return p.parse_args(argv)
