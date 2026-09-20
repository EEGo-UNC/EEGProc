"""CLI definitions only: no TensorFlow, training, data loading, or saving."""

import argparse
import math
from pathlib import Path


def _positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def _nonnegative_float(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return number


def _decay_float(value):
    number = float(value)
    if not math.isfinite(number) or not 0 < number <= 1:
        raise argparse.ArgumentTypeError("must be finite and in (0, 1]")
    return number


def _nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def build_parser():
    """Build the standalone counterfactual CLI without importing the model.

    Use a prepared trial NPZ or the existing SIC raw-data loader. Raw-data
    mode requires explicit window length, sampling rate, normalization, and
    label dimension: a .keras archive does not encode all preprocessing.
    A single model is used for the requested subject; the caller must choose
    its correct LOSO checkpoint. No source model is rebuilt or retrained.
    """
    parser = argparse.ArgumentParser(
        description="Optimize full-trial SIC latent counterfactuals; model weights stay fixed."
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Saved SIC .keras model with trained branch decoders.",
    )
    parser.add_argument(
        "--model-module",
        default="eegproc.deep_learning.joint_architectures.SICModelv11.sic_model",
        help="Import this module to register the saved model's Keras classes.",
    )
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument(
        "--trials-npz",
        type=Path,
        help="Prepared features (N,W,T,F), subject_ids, trial_ids, optional labels.",
    )
    data.add_argument("--raw-eeg-npy", type=Path)
    parser.add_argument("--raw-labels-npy", type=Path)
    parser.add_argument(
        "--dataset", default="dreamer", choices=("dreamer", "amigos", "eegemotions_27")
    )
    parser.add_argument("--label-dimension", choices=("valence", "arousal"))
    parser.add_argument("--window-sec", type=_positive_float)
    parser.add_argument("--fs", type=_positive_float)
    parser.add_argument(
        "--window-normalization", choices=("none", "global_rms", "feature_zscore")
    )
    parser.add_argument("--window-overlap", type=_nonnegative_float, default=0.0)
    parser.add_argument("--median-label", type=float, default=3.0)
    parser.add_argument(
        "--label-threshold-mode", choices=("global", "subject_median"), default="global"
    )
    parser.add_argument("--remove-median-label", action="store_true")
    parser.add_argument("--subject-id", type=int, required=True)
    parser.add_argument(
        "--trial-ids",
        "--trial-id",
        type=int,
        nargs="+",
        help="Default: all trials of the selected subject, in source order.",
    )
    parser.add_argument(
        "--target-class",
        type=_nonnegative_int,
        help="Default: opposite original predicted class; binary models only.",
    )
    parser.add_argument("--target-probability", type=_positive_float, default=0.8)
    parser.add_argument(
        "--target-loss-component",
        choices=("confidence", "focal", "vc", "focal_vc"),
        default="confidence",
        help=(
            "Classification-side objective differentiated through the frozen "
            "SIC head: confidence keeps the existing probability-threshold "
            "hinge; focal selects only the checkpoint-weighted focal term; "
            "vc selects only the checkpoint-weighted auxiliary VC terms; "
            "focal_vc selects their sum (default: confidence)."
        ),
    )
    parser.add_argument("--learning-rate", type=_positive_float, default=0.01)
    parser.add_argument(
        "--learning-rate-decay",
        type=_decay_float,
        default=1.0,
        help=(
            "Per-update exponential learning-rate multiplier in (0, 1]; "
            "1 keeps the learning rate constant (default: 1)."
        ),
    )
    parser.add_argument("--max-steps", type=_nonnegative_int, default=200)
    parser.add_argument("--gradient-clip-norm", type=_positive_float, default=5.0)
    parser.add_argument("--target-weight", type=_positive_float, default=1.0)
    parser.add_argument("--latent-weight", type=_nonnegative_float, default=0.1)
    parser.add_argument("--decoded-weight", type=_nonnegative_float, default=0.1)
    parser.add_argument(
        "--decoder-mode",
        choices=("branches", "joint"),
        default="branches",
        help=(
            "Decoded reconstruction used by the counterfactual objective. "
            "'branches' preserves the independent-decoder behavior; 'joint' "
            "uses only the saved learned fusion of both branch reconstructions."
        ),
    )
    parser.add_argument(
        "--physiological-weight",
        type=_nonnegative_float,
        default=0.0,
        help=(
            "Weight for the VCSC physiological-validity penalty applied to "
            "each selected decoded reconstruction (default: 0, diagnostic only)."
        ),
    )
    parser.add_argument(
        "--vcsc-distance-cm",
        type=_nonnegative_float,
        default=12.0,
        help="Electrode distance below which VCSC deviations receive extra weight.",
    )
    parser.add_argument(
        "--vcsc-tau-cm",
        type=_positive_float,
        default=4.0,
        help="Distance-decay scale for the VCSC proximity weighting.",
    )
    parser.add_argument(
        "--vcsc-z0",
        type=_nonnegative_float,
        default=2.0,
        help="VCSC z-deviation threshold before a pair is penalized.",
    )
    parser.add_argument(
        "--vcsc-z-max",
        type=_positive_float,
        default=20.0,
        help="Maximum VCSC pair deviation before exponentiation.",
    )
    parser.add_argument(
        "--stop-on-success",
        action="store_true",
        help="Stop at first target success instead of retaining the best over all steps.",
    )
    parser.add_argument(
        "--log-every",
        type=_nonnegative_int,
        default=1,
        help=(
            "Print objective diagnostics every N evaluated steps (default: 1); "
            "0 suppresses console logs. History is still saved every step."
        ),
    )
    parser.add_argument("--seed", type=_nonnegative_int, default=42)
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="New or empty output directory; existing results are never overwritten.",
    )
    return parser


def parse_args(argv=None):
    """Parse and validate CLI settings; runtime also validates its Python API."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.target_probability >= 1:
        parser.error("--target-probability must be strictly between 0 and 1")
    if args.vcsc_z_max <= args.vcsc_z0:
        parser.error("--vcsc-z-max must exceed --vcsc-z0")
    if args.vcsc_z_max - args.vcsc_z0 > 80:
        parser.error("--vcsc-z-max minus --vcsc-z0 must not exceed 80")
    if args.window_overlap >= 1 or not math.isfinite(args.median_label):
        parser.error("--window-overlap must be in [0,1); --median-label must be finite")
    raw_options = (
        args.raw_labels_npy,
        args.label_dimension,
        args.window_sec,
        args.fs,
        args.window_normalization,
    )
    if args.raw_eeg_npy and any(value is None for value in raw_options):
        parser.error(
            "raw mode requires --raw-labels-npy, --label-dimension, --window-sec, --fs, and --window-normalization matching training"
        )
    if args.trials_npz and (
        any(value is not None for value in raw_options) or args.remove_median_label
    ):
        parser.error(
            "prepared NPZ inputs must already be normalized and filtered; do not pass raw-data options"
        )
    return args
