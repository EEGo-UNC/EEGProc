"""Train CMHFE on raw EEG with leave-one-subject-out cross-validation.

The script reuses the repo's existing raw-signal helpers and the built-in LOSO
cross-validation runner. It trains the shared CMHFE feature extractor with the
two emotion heads in one run, using separate valence and arousal labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from ..joint_architectures.joint_v2_data import (
    load_raw_eeg_and_labels,
    window_trial_signal,
    zscore_subject_eeg,
)
from ..archival_cv import loso_cv
from .cmhfe_dan import CMHFEConfig, build_cmhfe_model


DATASET_PRESETS: dict[str, dict[str, float | int]] = {
    "deap": {
        "n_channels": 32,
        "sampling_frequency": 128.0,
        "valence_threshold": 5.0,
        "arousal_threshold": 5.0,
        "window_length_sec": 4.0,
        "overlap": 0.5,
    },
    "dreamer": {
        "n_channels": 14,
        "sampling_frequency": 128.0,
        "valence_threshold": 2.5,
        "arousal_threshold": 2.5,
        "window_length_sec": 4.0,
        "overlap": 0.5,
    },
    "amigos": {
        "n_channels": 14,
        "sampling_frequency": 128.0,
        "valence_threshold": 5.0,
        "arousal_threshold": 5.0,
        "window_length_sec": 4.0,
        "overlap": 0.5,
    },
    "custom": {
        "n_channels": 0,
        "sampling_frequency": 128.0,
        "valence_threshold": 5.0,
        "arousal_threshold": 5.0,
        "window_length_sec": 4.0,
        "overlap": 0.5,
    },
}


def _parse_int_list(text: str) -> tuple[int, ...]:
    values = [chunk.strip() for chunk in text.split(",") if chunk.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of integers.")
    return tuple(int(value) for value in values)


def _resolve_float(cli_value: float | None, preset_value: float) -> float:
    return float(cli_value) if cli_value is not None else float(preset_value)


def build_multi_output_dataset(
    eeg_path: str | Path,
    labels_path: str | Path,
    n_channels: int,
    sampling_frequency: float,
    window_length_sec: float,
    overlap: float,
    valence_threshold: float,
    arousal_threshold: float,
    zscore: bool = True,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Convert raw subject/trial arrays into LOSO-ready windows and labels."""
    eeg, raw_labels = load_raw_eeg_and_labels(eeg_path, labels_path)

    if raw_labels.ndim != 3 or raw_labels.shape[-1] < 2:
        raise ValueError(
            "labels array must have at least two dimensions for valence and arousal."
        )

    if eeg.shape[2] < n_channels:
        raise ValueError(
            f"Requested n_channels={n_channels}, but eeg array only has {eeg.shape[2]}."
        )

    eeg = eeg[:, :, :n_channels, :]

    valence_trials = (raw_labels[:, :, 0] > valence_threshold).astype(np.int32)
    arousal_trials = (raw_labels[:, :, 1] > arousal_threshold).astype(np.int32)

    window_size = int(round(window_length_sec * sampling_frequency))
    feature_chunks: list[np.ndarray] = []
    valence_chunks: list[np.ndarray] = []
    arousal_chunks: list[np.ndarray] = []
    subject_chunks: list[np.ndarray] = []

    for subject_idx in range(eeg.shape[0]):
        subject_eeg = eeg[subject_idx]
        if zscore:
            subject_eeg = zscore_subject_eeg(subject_eeg)

        for trial_idx in range(eeg.shape[1]):
            trial_windows = window_trial_signal(
                subject_eeg[trial_idx],
                window_size=window_size,
                overlap=overlap,
            )

            trial_windows = np.transpose(trial_windows, (0, 2, 1)).astype(np.float32)
            n_windows = trial_windows.shape[0]

            feature_chunks.append(trial_windows)
            valence_chunks.append(
                np.full(n_windows, valence_trials[subject_idx, trial_idx], dtype=np.int32)
            )
            arousal_chunks.append(
                np.full(n_windows, arousal_trials[subject_idx, trial_idx], dtype=np.int32)
            )
            subject_chunks.append(np.full(n_windows, subject_idx, dtype=np.int64))

    feature_array = np.concatenate(feature_chunks, axis=0)
    label_array = {
        "valence": np.concatenate(valence_chunks, axis=0),
        "arousal": np.concatenate(arousal_chunks, axis=0),
    }
    subject_id_array = np.concatenate(subject_chunks, axis=0)

    return feature_array, label_array, subject_id_array


def build_model(config: CMHFEConfig) -> tf.keras.Model:
    return build_cmhfe_model(config)


def run_training(args: argparse.Namespace) -> dict:
    preset = DATASET_PRESETS[args.dataset]

    n_channels = args.n_channels if args.n_channels is not None else int(preset["n_channels"])
    if n_channels < 1:
        raise ValueError("n_channels must be provided for custom runs and be at least 1.")
    sampling_frequency = _resolve_float(args.sampling_frequency, float(preset["sampling_frequency"]))
    window_length_sec = _resolve_float(args.window_length_sec, float(preset["window_length_sec"]))
    overlap = _resolve_float(args.overlap, float(preset["overlap"]))
    valence_threshold = _resolve_float(args.valence_threshold, float(preset["valence_threshold"]))
    arousal_threshold = _resolve_float(args.arousal_threshold, float(preset["arousal_threshold"]))

    feature_array, label_array, subject_id_array = build_multi_output_dataset(
        eeg_path=args.eeg_path,
        labels_path=args.labels_path,
        n_channels=n_channels,
        sampling_frequency=sampling_frequency,
        window_length_sec=window_length_sec,
        overlap=overlap,
        valence_threshold=valence_threshold,
        arousal_threshold=arousal_threshold,
        zscore=not args.no_zscore,
    )

    config = CMHFEConfig(
        n_channels=n_channels,
        window_length=window_length_sec,
        sampling_frequency=sampling_frequency,
        cnn_filters=args.cnn_filters,
        conv_kernel_size=args.kernel_size,
        conv_strides=args.strides,
        conv_padding=args.padding,
        dropout_rate=args.dropout_rate,
        l2_regularization=args.l2_regularization,
        transformer_heads=args.transformer_heads,
        transformer_embedding_dim=args.transformer_embedding_dim,
        transformer_ffn_dim=args.transformer_ffn_dim,
        enable_maxpool=args.enable_maxpool,
        maxpool_size=args.maxpool_size,
        domain_loss_weight=args.domain_loss_weight,
        grl_lambda=args.grl_lambda,
        valence_threshold=valence_threshold,
        arousal_threshold=arousal_threshold,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
    )

    def model_builder(**_hparams):
        return build_model(config)

    hyperparameters = {"epochs": args.epochs, "batch_size": args.batch_size}
    results = loso_cv(
        model_builder_function=model_builder,
        feature_array=feature_array,
        label_array=label_array,
        subject_id_array=subject_id_array,
        hyperparameters=hyperparameters,
        validation_n_users=args.validation_n_users,
        verbose=args.verbose,
        extra_fit_kwargs={"shuffle": True},
    )

    results["dataset"] = args.dataset
    results["config"] = {
        "n_channels": n_channels,
        "sampling_frequency": sampling_frequency,
        "window_length_sec": window_length_sec,
        "overlap": overlap,
        "valence_threshold": valence_threshold,
        "arousal_threshold": arousal_threshold,
        "cnn_filters": list(config.cnn_filters),
        "kernel_size": list(config.conv_kernel_size),
        "strides": list(config.conv_strides),
        "padding": list(config.conv_padding),
        "dropout_rate": config.dropout_rate,
        "l2_regularization": config.l2_regularization,
        "transformer_heads": config.transformer_heads,
        "transformer_embedding_dim": config.transformer_embedding_dim,
        "transformer_ffn_dim": config.transformer_ffn_dim,
        "enable_maxpool": config.enable_maxpool,
        "maxpool_size": config.maxpool_size,
        "domain_loss_weight": config.domain_loss_weight,
        "grl_lambda": config.grl_lambda,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "epochs": args.epochs,
        "validation_n_users": args.validation_n_users,
    }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.results_dir / f"cmhfe_loso_{args.dataset}.json"
    results_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    print("\nLOSO training complete")
    print(f"Saved results to {results_path}")
    print(f"Mean scores: {results['mean_scores']}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CMHFE with LOSO CV")
    parser.add_argument("--dataset", choices=sorted(DATASET_PRESETS), default="dreamer")
    parser.add_argument("--eeg_path", type=Path, required=True)
    parser.add_argument("--labels_path", type=Path, required=True)
    parser.add_argument("--results_dir", type=Path, default=Path("runs/cmhfe_loso"))
    parser.add_argument("--n_channels", type=int, default=None)
    parser.add_argument("--sampling_frequency", type=float, default=None)
    parser.add_argument("--window_length_sec", type=float, default=None)
    parser.add_argument("--overlap", type=float, default=None)
    parser.add_argument("--valence_threshold", type=float, default=None)
    parser.add_argument("--arousal_threshold", type=float, default=None)
    parser.add_argument("--cnn_filters", type=_parse_int_list, default=(64, 128, 256, 128))
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--strides", type=int, default=1)
    parser.add_argument("--padding", choices=["same", "valid"], default="same")
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--l2_regularization", type=float, default=0.001)
    parser.add_argument("--transformer_heads", type=int, default=4)
    parser.add_argument("--transformer_embedding_dim", type=int, default=128)
    parser.add_argument("--transformer_ffn_dim", type=int, default=512)
    parser.add_argument("--enable_maxpool", action="store_true")
    parser.add_argument("--maxpool_size", type=int, default=2)
    parser.add_argument("--domain_loss_weight", type=float, default=1.0)
    parser.add_argument("--grl_lambda", type=float, default=0.5)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--validation_n_users", type=int, default=None)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_zscore", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)
    run_training(args)


if __name__ == "__main__":
    main()