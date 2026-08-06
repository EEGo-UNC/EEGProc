"""LOSO training for the reconstruction-only subject-adversarial STS VAE.

The emotional labels produced by the shared DREAMER data builder are discarded.
They never enter model.fit, model selection, or evaluation. The raw labels file
is accepted only because EEGProc's existing raw-data builder uses it to recover
its established subject/trial-aligned window representation.

For each outer LOSO fold:

1. Hold out one complete subject for reconstruction testing.
2. Hold out complete subjects from the remaining pool for validation.
3. Select hyperparameters and epoch count by validation ``decoder_accuracy``
   (dataset-level reconstruction R^2, maximized).
4. Rebuild the selected model and train it on every non-test subject for the
   selected epoch count.
5. Evaluate deterministic posterior-mean reconstruction on the unseen subject.

The saved fold models are the subject-independent decoders intended for later
counterfactual faithfulness experiments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import argparse
import csv
import gc
import itertools
import json
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np
import tensorflow as tf

try:
    from .inverse_subject_vae_model import (
        InverseSubjectSTSVAE,
        build_inverse_subject_sts_vae,
    )
    from ..joint_v3_sts.joint_sts_model_train import load_joint_sts_training_data
    from ..joint_v2_data import (
        DEFAULT_DREAMER_EEG_PATH,
        DEFAULT_DREAMER_LABELS_PATH,
        DREAMER_FS,
        DREAMER_MEDIAN_LABEL,
    )
except ImportError:
    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    from inverse_subject_vae_model import (
        InverseSubjectSTSVAE,
        build_inverse_subject_sts_vae,
    )
    from eegproc.deep_learning.joint_architectures.joint_v3_sts.joint_sts_model_train import load_joint_sts_training_data
    from eegproc.deep_learning.joint_architectures.joint_v2_data import (
        DEFAULT_DREAMER_EEG_PATH,
        DEFAULT_DREAMER_LABELS_PATH,
        DREAMER_FS,
        DREAMER_MEDIAN_LABEL,
    )


_SEQUENCE_HPARAMETERS = {"gcn_units", "temporal_pool_sizes"}
_FIT_HPARAMETERS = {"epochs", "batch_size"}


@dataclass(slots=True)
class InverseSubjectVAETrainingConfig:
    output_dir: Path = Path("runs") / "inverse_subject_vae"
    run_name: str = "dreamer_inverse_subject_sts_vae"
    dataset: str = "dreamer"
    n_channels: int = 14
    n_bands: int | None = 3

    # LOSO model-selection protocol.
    epochs: int = 100
    batch_size: int = 32
    validation_subjects_per_fold: int = 3
    validation_seed: int | None = 42
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 0.001
    max_folds: int | None = None
    seed: int | None = 42
    verbose: int = 2
    candidate_verbose: int = 0

    # Saved outputs.
    save_fold_models: bool = True
    save_final_model: bool = True
    save_fold_reconstructions: bool = False
    save_adjacency_matrices: bool = True
    run_final_full_data_fit: bool = True

    # Optimizers and alternating update counts.
    optimizer_name: str = "adamw"
    vae_learning_rate: float = 5e-5
    subject_learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    subject_steps_per_batch: int = 1
    vae_steps_per_batch: int = 1

    # Shared temporal downsampling.
    t_down: int = 2
    temporal_pool_sizes: tuple[int, ...] = (2,)

    # Spatiotemporal BiLSTM encoder.
    bilstm_units: int = 64
    n_bilstm_layers: int = 1
    bilstm_dropout: float = 0.30
    temporal_emb_dim: int = 32

    # Spatiospectral GCN encoder.
    gcn_units: tuple[int, ...] = (64, 32)
    spectral_emb_dim: int = 32
    gcn_dropout: float = 0.20
    gcn_activation: str = "relu"
    gcn_use_batch_norm: bool = False
    graph_self_loop_bias: float = 2.0
    graph_identity_mix: float = 0.0
    graph_adjacency_reg_weight: float = 1e-4

    # Fused posterior and dual-path decoder.
    fusion_dim: int = 64
    latent_features: int = 32
    fusion_dropout: float = 0.20
    activation: str = "relu"
    decoder_temporal_units: int = 64
    decoder_bilstm_layers: int = 1
    decoder_graph_output_units: int = 16
    decoder_branch_feature_dim: int = 64
    decoder_fusion_units: int = 64
    decoder_dropout: float = 0.20

    # VAE and inverse-subject objectives.
    reconstruction_loss: str = "mse"
    vae_loss_weight: float = 1.0
    vae_beta: float = 0.30
    subject_adversarial_weight: float = 1.0
    subject_loss_weight: float = 1.0
    subject_hidden_units: int = 64
    subject_dropout: float = 0.0

    # Raw-data preprocessing compatibility.
    window_size_sec: float = 4.0
    fs: float = DREAMER_FS
    overlap: float = 0.0
    median_label: float = DREAMER_MEDIAN_LABEL
    window_normalization: str = "global_rms"

    # Optional grid. Emotion/classifier hyperparameters are intentionally absent.
    hyperparameters: dict = field(default_factory=dict)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if tf.is_tensor(value):
        return value.numpy().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: Any) -> None:
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


def _configure_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"eegproc.inverse_subject_vae.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(run_dir / "training.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _normalize_windows(
    features: np.ndarray,
    mode: str,
    epsilon: float = 1e-6,
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 3:
        raise ValueError(
            "Window normalization expects (windows, timesteps, features); "
            f"got {features.shape}."
        )
    mode = str(mode).lower()
    if mode == "none":
        return features
    if mode == "global_rms":
        rms = np.sqrt(
            np.mean(np.square(features, dtype=np.float64), axis=(1, 2), keepdims=True)
        )
        output = features.astype(np.float64) / np.maximum(rms, epsilon)
    elif mode == "feature_zscore":
        mean = np.mean(features, axis=1, keepdims=True, dtype=np.float64)
        std = np.std(features, axis=1, keepdims=True, dtype=np.float64)
        output = (features.astype(np.float64) - mean) / np.maximum(std, epsilon)
    else:
        raise ValueError(
            "window_normalization must be none, global_rms, or feature_zscore."
        )
    if not np.isfinite(output).all():
        raise ValueError("Window normalization produced NaN or Inf values.")
    return output.astype(np.float32)


def _flatten_optional_trial_tensor(
    features: np.ndarray,
    subjects: np.ndarray,
    trials: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float32)
    subjects = np.asarray(subjects).reshape(-1)
    if trials is None:
        trials = np.arange(len(subjects), dtype=np.int64)
    trials = np.asarray(trials).reshape(-1)
    if features.ndim == 3:
        if not (len(features) == len(subjects) == len(trials)):
            raise ValueError("Feature, subject, and trial arrays must align.")
        return features, subjects, trials
    if features.ndim != 4:
        raise ValueError(
            "Features must be rank 3 (windows) or rank 4 (trials x windows); "
            f"got {features.shape}."
        )
    if not (len(features) == len(subjects) == len(trials)):
        raise ValueError("Grouped feature, subject, and trial arrays must align.")
    n_groups, n_windows, timesteps, n_features = features.shape
    return (
        features.reshape(n_groups * n_windows, timesteps, n_features),
        np.repeat(subjects, n_windows),
        np.repeat(trials, n_windows),
    )


def load_inverse_subject_training_data(
    *,
    features_path: str | Path | None,
    subjects_path: str | Path | None,
    trials_path: str | Path | None,
    raw_eeg_path: str | Path | None,
    raw_labels_path: str | Path | None,
    config: InverseSubjectVAETrainingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load EEG windows and IDs without creating an emotional target."""
    if features_path is not None or subjects_path is not None:
        if features_path is None or subjects_path is None:
            raise ValueError("--features-npy and --subjects-npy must be supplied together.")
        features = np.load(Path(features_path), allow_pickle=False)
        subjects = np.load(Path(subjects_path), allow_pickle=False)
        trials = (
            None
            if trials_path is None
            else np.load(Path(trials_path), allow_pickle=False)
        )
        features, subjects, trials = _flatten_optional_trial_tensor(
            features,
            subjects,
            trials,
        )
        features = _normalize_windows(features, config.window_normalization)
        return features, subjects, trials

    eeg_path = DEFAULT_DREAMER_EEG_PATH if raw_eeg_path is None else raw_eeg_path
    labels_path = (
        DEFAULT_DREAMER_LABELS_PATH if raw_labels_path is None else raw_labels_path
    )
    features, _discarded_emotion_labels, subjects, trials = (
        load_joint_sts_training_data(
            eeg_path=eeg_path,
            labels_path=labels_path,
            # This value affects only the discarded compatibility labels. The
            # EEG features, subject IDs, and trial IDs are independent of it.
            label_dimension="valence",
            window_size_sec=config.window_size_sec,
            fs=config.fs,
            overlap=config.overlap,
            median_label=config.median_label,
            window_normalization=config.window_normalization,
            label_threshold_mode="global",
            dataset=config.dataset,
        )
    )
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(subjects).reshape(-1),
        np.asarray(trials).reshape(-1),
    )


def _sequence_candidates(key: str, value) -> list:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{key} must be represented by a list or tuple.")
    if not value:
        raise ValueError(f"{key} cannot be empty.")
    if all(not isinstance(item, (list, tuple)) for item in value):
        return [list(value)]
    return [list(item) for item in value]


def _expand_hyperparameter_grid(hyperparameters: dict | None) -> list[dict]:
    if not hyperparameters:
        return [{}]
    keys = list(hyperparameters)
    candidates: list[list] = []
    for key in keys:
        value = hyperparameters[key]
        if key in _SEQUENCE_HPARAMETERS:
            candidates.append(_sequence_candidates(key, value))
        elif isinstance(value, (list, tuple)):
            if not value:
                raise ValueError(f"Hyperparameter {key!r} has no candidates.")
            candidates.append(list(value))
        else:
            candidates.append([value])
    return [
        dict(zip(keys, combination))
        for combination in itertools.product(*candidates)
    ]


def _allowed_model_hyperparameters() -> set[str]:
    return {
        "optimizer_name",
        "vae_learning_rate",
        "subject_learning_rate",
        "weight_decay",
        "subject_steps_per_batch",
        "vae_steps_per_batch",
        "t_down",
        "temporal_pool_sizes",
        "bilstm_units",
        "n_bilstm_layers",
        "bilstm_dropout",
        "temporal_emb_dim",
        "gcn_units",
        "spectral_emb_dim",
        "gcn_dropout",
        "gcn_activation",
        "gcn_use_batch_norm",
        "graph_self_loop_bias",
        "graph_identity_mix",
        "graph_adjacency_reg_weight",
        "fusion_dim",
        "latent_features",
        "fusion_dropout",
        "activation",
        "decoder_temporal_units",
        "decoder_bilstm_layers",
        "decoder_graph_output_units",
        "decoder_branch_feature_dim",
        "decoder_fusion_units",
        "decoder_dropout",
        "reconstruction_loss",
        "vae_loss_weight",
        "vae_beta",
        "subject_adversarial_weight",
        "subject_loss_weight",
        "subject_hidden_units",
        "subject_dropout",
    }


def _base_model_kwargs(config: InverseSubjectVAETrainingConfig) -> dict:
    return {
        "n_channels": config.n_channels,
        "n_bands": config.n_bands,
        "t_down": config.t_down,
        "temporal_pool_sizes": config.temporal_pool_sizes,
        "bilstm_units": config.bilstm_units,
        "n_bilstm_layers": config.n_bilstm_layers,
        "bilstm_dropout": config.bilstm_dropout,
        "temporal_emb_dim": config.temporal_emb_dim,
        "gcn_units": config.gcn_units,
        "spectral_emb_dim": config.spectral_emb_dim,
        "gcn_dropout": config.gcn_dropout,
        "gcn_activation": config.gcn_activation,
        "gcn_use_batch_norm": config.gcn_use_batch_norm,
        "graph_self_loop_bias": config.graph_self_loop_bias,
        "graph_identity_mix": config.graph_identity_mix,
        "graph_adjacency_reg_weight": config.graph_adjacency_reg_weight,
        "fusion_dim": config.fusion_dim,
        "latent_features": config.latent_features,
        "fusion_dropout": config.fusion_dropout,
        "activation": config.activation,
        "decoder_temporal_units": config.decoder_temporal_units,
        "decoder_bilstm_layers": config.decoder_bilstm_layers,
        "decoder_graph_output_units": config.decoder_graph_output_units,
        "decoder_branch_feature_dim": config.decoder_branch_feature_dim,
        "decoder_fusion_units": config.decoder_fusion_units,
        "decoder_dropout": config.decoder_dropout,
        "reconstruction_loss": config.reconstruction_loss,
        "vae_loss_weight": config.vae_loss_weight,
        "vae_beta": config.vae_beta,
        "subject_adversarial_weight": config.subject_adversarial_weight,
        "subject_loss_weight": config.subject_loss_weight,
        "subject_hidden_units": config.subject_hidden_units,
        "subject_dropout": config.subject_dropout,
        "subject_steps_per_batch": config.subject_steps_per_batch,
        "vae_steps_per_batch": config.vae_steps_per_batch,
        "optimizer_name": config.optimizer_name,
        "vae_learning_rate": config.vae_learning_rate,
        "subject_learning_rate": config.subject_learning_rate,
        "weight_decay": config.weight_decay,
    }


def _build_model(
    input_shape: tuple[int, int],
    config: InverseSubjectVAETrainingConfig,
    candidate: dict,
    model_name: str,
) -> InverseSubjectSTSVAE:
    unknown = set(candidate) - _allowed_model_hyperparameters() - _FIT_HPARAMETERS
    if unknown:
        raise ValueError(
            "Unsupported inverse-subject VAE hyperparameter(s): "
            f"{sorted(unknown)}"
        )
    model_kwargs = _base_model_kwargs(config)
    model_kwargs.update(
        {key: value for key, value in candidate.items() if key not in _FIT_HPARAMETERS}
    )
    if model_kwargs["n_bands"] is None:
        raise ValueError("n_bands must be resolved before model construction.")
    return build_inverse_subject_sts_vae(
        input_shape=input_shape,
        model_name=model_name,
        **model_kwargs,
    )


def _resolve_channel_band_shape(
    n_features: int,
    n_channels: int,
    n_bands: int | None,
) -> tuple[int, int]:
    n_features = int(n_features)
    n_channels = int(n_channels)
    if n_bands is None:
        if n_features % n_channels != 0:
            raise ValueError(
                f"Cannot infer n_bands because {n_features} is not divisible by "
                f"n_channels={n_channels}."
            )
        n_bands = n_features // n_channels
    n_bands = int(n_bands)
    if n_channels * n_bands != n_features:
        raise ValueError(
            "Input must satisfy n_features = n_channels * n_bands; got "
            f"{n_features} != {n_channels} * {n_bands}."
        )
    return n_channels, n_bands


def _choose_validation_subjects(
    training_subjects: np.ndarray,
    *,
    count: int,
    seed: int | None,
    fold_index: int,
) -> np.ndarray:
    subjects = np.sort(np.unique(training_subjects))
    count = int(count)
    if count < 1:
        raise ValueError("validation_subjects_per_fold must be at least 1.")
    if count >= len(subjects):
        raise ValueError(
            "Each LOSO fold needs at least two gradient-training subjects after "
            f"validation; requested {count} validation subjects from {len(subjects)}."
        )
    base_seed = 0 if seed is None else int(seed)
    rng = np.random.default_rng(np.random.SeedSequence([base_seed, fold_index]))
    return np.sort(rng.choice(subjects, size=count, replace=False))


def _best_epoch_from_history(history: dict[str, list], fallback: int) -> int:
    values = np.asarray(history.get("val_decoder_accuracy", []), dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).any():
        return max(1, int(fallback))
    finite_values = np.where(np.isfinite(values), values, -np.inf)
    return int(np.argmax(finite_values)) + 1


def _deterministic_reconstruction_statistics(
    model: InverseSubjectSTSVAE,
    features: np.ndarray,
    batch_size: int,
) -> tuple[dict[str, float | int], np.ndarray]:
    reconstruction = model.reconstruct(features, batch_size=batch_size)
    target = np.asarray(features, dtype=np.float64)
    prediction = np.asarray(reconstruction, dtype=np.float64)
    error = target - prediction
    sse = float(np.sum(np.square(error), dtype=np.float64))
    target_sum = float(np.sum(target, dtype=np.float64))
    target_squared_sum = float(np.sum(np.square(target), dtype=np.float64))
    count = int(target.size)
    tss = target_squared_sum - (target_sum * target_sum / max(count, 1))
    r2 = 1.0 - sse / tss if tss > np.finfo(np.float64).eps else float(sse == 0.0)
    return (
        {
            "r2": float(r2),
            "mse": float(np.mean(np.square(error), dtype=np.float64)),
            "mae": float(np.mean(np.abs(error), dtype=np.float64)),
            "sse": sse,
            "target_sum": target_sum,
            "target_squared_sum": target_squared_sum,
            "target_count": count,
        },
        reconstruction,
    )


def _save_adjacency_matrices(model: InverseSubjectSTSVAE, path: Path) -> None:
    nested = model.get_adjacency_matrices()
    flattened: dict[str, np.ndarray] = {}
    for component_name, matrices in nested.items():
        for layer_name, matrix in matrices.items():
            flattened[f"{component_name}__{layer_name}"] = np.asarray(matrix.numpy())
    if flattened:
        np.savez_compressed(path, **flattened)


def _clear_model(model=None) -> None:
    del model
    tf.keras.backend.clear_session()
    gc.collect()


def train_inverse_subject_vae_loso(
    *,
    features: np.ndarray,
    subject_ids: np.ndarray,
    trial_ids: np.ndarray,
    config: InverseSubjectVAETrainingConfig,
) -> dict:
    """Run validation-selected outer LOSOCV using held-out-subject R^2."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.output_dir / f"{config.run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    folds_dir = run_dir / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)
    logger = _configure_logger(run_dir)

    if config.seed is not None:
        tf.keras.utils.set_random_seed(config.seed)
        np.random.seed(config.seed)

    features = np.asarray(features, dtype=np.float32)
    subject_ids = np.asarray(subject_ids).reshape(-1)
    trial_ids = np.asarray(trial_ids).reshape(-1)
    if features.ndim != 3:
        raise ValueError("features must have shape (windows, timesteps, features).")
    if not (len(features) == len(subject_ids) == len(trial_ids)):
        raise ValueError("Features, subject IDs, and trial IDs must align.")
    config.n_channels, config.n_bands = _resolve_channel_band_shape(
        features.shape[-1],
        config.n_channels,
        config.n_bands,
    )

    unique_subjects = np.sort(np.unique(subject_ids))
    if len(unique_subjects) < 4:
        raise ValueError(
            "Subject-adversarial LOSO requires enough subjects for test, "
            "validation, and at least two training identities."
        )
    outer_subjects = unique_subjects
    if config.max_folds is not None:
        outer_subjects = outer_subjects[: max(1, int(config.max_folds))]

    candidates = _expand_hyperparameter_grid(config.hyperparameters)
    _write_json(run_dir / "training_config.json", asdict(config))
    _write_json(run_dir / "hyperparameter_candidates.json", candidates)

    logger.info("Starting inverse-subject STS VAE LOSOCV in %s", run_dir)
    logger.info("Feature tensor: %s", features.shape)
    logger.info("Subjects: %d; outer folds: %d", len(unique_subjects), len(outer_subjects))
    logger.info("Hyperparameter candidates: %d", len(candidates))
    logger.info(
        "Selection/evaluation metric: validation/test decoder_accuracy (R^2), maximized"
    )
    logger.info(
        "No emotional labels, classifier, thresholds, class weights, SupCon, or MLDG are used."
    )

    fold_results: list[dict] = []
    candidate_records: list[dict] = []

    for fold_index, test_subject in enumerate(outer_subjects, start=1):
        test_mask = subject_ids == test_subject
        outer_train_mask = ~test_mask
        outer_train_subjects = subject_ids[outer_train_mask]
        validation_subjects = _choose_validation_subjects(
            outer_train_subjects,
            count=config.validation_subjects_per_fold,
            seed=config.validation_seed,
            fold_index=fold_index,
        )
        validation_mask = outer_train_mask & np.isin(subject_ids, validation_subjects)
        fit_mask = outer_train_mask & ~np.isin(subject_ids, validation_subjects)

        X_fit = features[fit_mask]
        s_fit = subject_ids[fit_mask]
        X_val = features[validation_mask]
        X_outer_train = features[outer_train_mask]
        s_outer_train = subject_ids[outer_train_mask]
        X_test = features[test_mask]
        test_trials = trial_ids[test_mask]

        logger.info(
            "Fold %d/%d | test_subject=%s | fit_subjects=%d | val_subjects=%s | "
            "fit/val/test windows=%d/%d/%d",
            fold_index,
            len(outer_subjects),
            test_subject,
            len(np.unique(s_fit)),
            validation_subjects.tolist(),
            len(X_fit),
            len(X_val),
            len(X_test),
        )

        best_candidate_index: int | None = None
        best_validation_r2 = -np.inf
        best_epoch = config.epochs
        fold_candidate_rows: list[dict] = []

        for candidate_index, candidate in enumerate(candidates):
            candidate_epochs = int(candidate.get("epochs", config.epochs))
            candidate_batch_size = int(candidate.get("batch_size", config.batch_size))
            if candidate_epochs < 1 or candidate_batch_size < 1:
                raise ValueError("Candidate epochs and batch_size must be positive.")

            tf.keras.utils.set_random_seed(
                (0 if config.seed is None else int(config.seed))
                + fold_index * 1000
                + candidate_index
            )
            model = _build_model(
                input_shape=tuple(features.shape[1:]),
                config=config,
                candidate=candidate,
                model_name=f"inverse_subject_fold_{fold_index}_candidate_{candidate_index}",
            )
            fit_inputs = model.prepare_fit_inputs(X_fit, s_fit)
            dummy_fit = np.zeros(len(X_fit), dtype=np.float32)
            dummy_val = np.zeros(len(X_val), dtype=np.float32)
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_decoder_accuracy",
                    mode="max",
                    patience=config.early_stopping_patience,
                    min_delta=config.early_stopping_min_delta,
                    restore_best_weights=True,
                    verbose=0,
                ),
                tf.keras.callbacks.TerminateOnNaN(),
            ]
            history = model.fit(
                fit_inputs,
                dummy_fit,
                validation_data=(X_val, dummy_val),
                epochs=candidate_epochs,
                batch_size=candidate_batch_size,
                callbacks=callbacks,
                verbose=config.candidate_verbose,
                shuffle=True,
            )
            selected_epoch = _best_epoch_from_history(
                history.history,
                fallback=candidate_epochs,
            )
            validation_eval = model.evaluate(
                X_val,
                dummy_val,
                batch_size=candidate_batch_size,
                verbose=0,
                return_dict=True,
            )
            validation_r2 = float(validation_eval["decoder_accuracy"])
            row = {
                "fold": fold_index,
                "test_subject": _json_default(np.asarray(test_subject)),
                "candidate_index": candidate_index,
                "validation_subjects": validation_subjects.tolist(),
                "validation_r2": validation_r2,
                "best_epoch": selected_epoch,
                "epochs_cap": candidate_epochs,
                "batch_size": candidate_batch_size,
                "candidate": candidate,
            }
            fold_candidate_rows.append(row)
            candidate_records.append(row)
            logger.info(
                "Fold %d candidate %d | val_R2=%.6f | best_epoch=%d | %s",
                fold_index,
                candidate_index,
                validation_r2,
                selected_epoch,
                candidate,
            )
            if validation_r2 > best_validation_r2:
                best_validation_r2 = validation_r2
                best_candidate_index = candidate_index
                best_epoch = selected_epoch
            _clear_model(model)

        if best_candidate_index is None:
            raise RuntimeError(f"No valid candidate was selected for fold {fold_index}.")
        selected_candidate = candidates[best_candidate_index]
        selected_batch_size = int(
            selected_candidate.get("batch_size", config.batch_size)
        )

        # Retrain after selection on every non-test subject. The outer test
        # subject remains completely absent from training and model selection.
        tf.keras.utils.set_random_seed(
            (0 if config.seed is None else int(config.seed)) + fold_index * 1000 + 999
        )
        fold_model = _build_model(
            input_shape=tuple(features.shape[1:]),
            config=config,
            candidate=selected_candidate,
            model_name=f"inverse_subject_loso_fold_{fold_index}",
        )
        outer_fit_inputs = fold_model.prepare_fit_inputs(
            X_outer_train,
            s_outer_train,
        )
        fold_model.fit(
            outer_fit_inputs,
            np.zeros(len(X_outer_train), dtype=np.float32),
            epochs=best_epoch,
            batch_size=selected_batch_size,
            callbacks=[tf.keras.callbacks.TerminateOnNaN()],
            verbose=config.verbose,
            shuffle=True,
        )
        test_eval = fold_model.evaluate(
            X_test,
            np.zeros(len(X_test), dtype=np.float32),
            batch_size=selected_batch_size,
            verbose=0,
            return_dict=True,
        )
        reconstruction_stats, test_reconstruction = (
            _deterministic_reconstruction_statistics(
                fold_model,
                X_test,
                selected_batch_size,
            )
        )
        if not np.isclose(
            float(test_eval["decoder_accuracy"]),
            reconstruction_stats["r2"],
            rtol=1e-5,
            atol=1e-5,
        ):
            logger.warning(
                "Fold %d Keras R2 %.8f differs from direct R2 %.8f.",
                fold_index,
                float(test_eval["decoder_accuracy"]),
                reconstruction_stats["r2"],
            )

        fold_dir = folds_dir / f"fold_{fold_index:02d}_subject_{test_subject}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        if config.save_fold_models:
            fold_model.save(fold_dir / "model.keras")
            fold_model.save_weights(fold_dir / "model.weights.h5")
        if config.save_adjacency_matrices:
            _save_adjacency_matrices(
                fold_model,
                fold_dir / "adjacency_matrices.npz",
            )
        if config.save_fold_reconstructions:
            np.save(fold_dir / "test_reconstruction.npy", test_reconstruction)
            np.save(fold_dir / "test_target.npy", X_test)
            np.save(fold_dir / "test_trial_ids.npy", test_trials)

        fold_result = {
            "fold": fold_index,
            "test_subject": _json_default(np.asarray(test_subject)),
            "validation_subjects": validation_subjects.tolist(),
            "selected_candidate_index": best_candidate_index,
            "selected_candidate": selected_candidate,
            "selected_epoch": best_epoch,
            "selected_batch_size": selected_batch_size,
            "validation_r2": best_validation_r2,
            "test_r2": reconstruction_stats["r2"],
            "test_mse": reconstruction_stats["mse"],
            "test_mae": reconstruction_stats["mae"],
            "test_sse": reconstruction_stats["sse"],
            "test_target_sum": reconstruction_stats["target_sum"],
            "test_target_squared_sum": reconstruction_stats[
                "target_squared_sum"
            ],
            "test_target_count": reconstruction_stats["target_count"],
            "keras_test_decoder_accuracy": float(test_eval["decoder_accuracy"]),
            "keras_test_reconstruction_loss": float(
                test_eval["reconstruction_loss"]
            ),
            "n_fit_windows_during_selection": len(X_fit),
            "n_validation_windows": len(X_val),
            "n_outer_training_windows": len(X_outer_train),
            "n_test_windows": len(X_test),
            "fold_model_dir": str(fold_dir),
        }
        fold_results.append(fold_result)
        _write_json(fold_dir / "fold_result.json", fold_result)
        _write_json(fold_dir / "candidate_results.json", fold_candidate_rows)
        logger.info(
            "Fold %d complete | test_subject=%s | test_R2=%.6f | MSE=%.8f | MAE=%.8f",
            fold_index,
            test_subject,
            reconstruction_stats["r2"],
            reconstruction_stats["mse"],
            reconstruction_stats["mae"],
        )
        _clear_model(fold_model)

    r2_values = np.asarray([row["test_r2"] for row in fold_results], dtype=np.float64)
    sse_total = float(sum(row["test_sse"] for row in fold_results))
    target_sum_total = float(sum(row["test_target_sum"] for row in fold_results))
    target_squared_sum_total = float(
        sum(row["test_target_squared_sum"] for row in fold_results)
    )
    target_count_total = int(sum(row["test_target_count"] for row in fold_results))
    pooled_tss = target_squared_sum_total - (
        target_sum_total * target_sum_total / max(target_count_total, 1)
    )
    pooled_r2 = (
        1.0 - sse_total / pooled_tss
        if pooled_tss > np.finfo(np.float64).eps
        else float(sse_total == 0.0)
    )

    candidate_summary: list[dict] = []
    for candidate_index, candidate in enumerate(candidates):
        rows = [
            row for row in candidate_records if row["candidate_index"] == candidate_index
        ]
        scores = np.asarray([row["validation_r2"] for row in rows], dtype=np.float64)
        epochs = np.asarray([row["best_epoch"] for row in rows], dtype=np.int64)
        candidate_summary.append(
            {
                "candidate_index": candidate_index,
                "candidate": candidate,
                "mean_validation_r2": float(np.mean(scores)),
                "std_validation_r2": float(np.std(scores)),
                "median_best_epoch": int(np.rint(np.median(epochs))),
                "n_folds": len(rows),
            }
        )
    best_global_candidate_row = max(
        candidate_summary,
        key=lambda row: row["mean_validation_r2"],
    )

    final_model_path = None
    if config.run_final_full_data_fit:
        final_candidate = dict(best_global_candidate_row["candidate"])
        final_batch_size = int(final_candidate.get("batch_size", config.batch_size))
        final_epochs = int(best_global_candidate_row["median_best_epoch"])
        tf.keras.utils.set_random_seed(
            0 if config.seed is None else int(config.seed) + 999999
        )
        final_model = _build_model(
            input_shape=tuple(features.shape[1:]),
            config=config,
            candidate=final_candidate,
            model_name="inverse_subject_sts_vae_final",
        )
        final_inputs = final_model.prepare_fit_inputs(features, subject_ids)
        final_model.fit(
            final_inputs,
            np.zeros(len(features), dtype=np.float32),
            epochs=final_epochs,
            batch_size=final_batch_size,
            callbacks=[tf.keras.callbacks.TerminateOnNaN()],
            verbose=config.verbose,
            shuffle=True,
        )
        final_dir = run_dir / "final_model"
        final_dir.mkdir(parents=True, exist_ok=True)
        if config.save_final_model:
            final_model_path = final_dir / "model.keras"
            final_model.save(final_model_path)
            final_model.save_weights(final_dir / "model.weights.h5")
        if config.save_adjacency_matrices:
            _save_adjacency_matrices(
                final_model,
                final_dir / "adjacency_matrices.npz",
            )
        _write_json(
            final_dir / "final_fit_config.json",
            {
                "candidate_index": best_global_candidate_row["candidate_index"],
                "candidate": final_candidate,
                "epochs": final_epochs,
                "batch_size": final_batch_size,
                "training_subjects": unique_subjects.tolist(),
                "note": (
                    "This model is trained on all subjects. Use the saved outer-fold "
                    "model for strictly unseen-subject faithfulness evaluation."
                ),
            },
        )
        _clear_model(final_model)

    summary = {
        "run_dir": str(run_dir),
        "architecture": "subject_adversarial_parallel_bilstm_gcn_fused_vae",
        "objective": (
            "reconstruction + beta*KL + gradient-reversed fold-local subject CE"
        ),
        "selection_metric": "validation_decoder_accuracy_r2",
        "outer_evaluation": "LOSO deterministic posterior-mean reconstruction",
        "n_subjects": len(unique_subjects),
        "n_completed_folds": len(fold_results),
        "subject_macro_r2_mean": float(np.mean(r2_values)),
        "subject_macro_r2_std": float(np.std(r2_values)),
        "subject_macro_r2_median": float(np.median(r2_values)),
        "subject_macro_r2_min": float(np.min(r2_values)),
        "subject_macro_r2_max": float(np.max(r2_values)),
        "pooled_window_element_r2": float(pooled_r2),
        "best_global_candidate": best_global_candidate_row,
        "final_model_path": None if final_model_path is None else str(final_model_path),
        "fold_results": fold_results,
        "candidate_summary": candidate_summary,
    }
    _write_json(run_dir / "loso_results.json", summary)
    _write_csv(run_dir / "loso_folds.csv", fold_results)
    _write_csv(
        run_dir / "candidate_summary.csv",
        [
            {
                **{key: value for key, value in row.items() if key != "candidate"},
                "candidate": json.dumps(row["candidate"], sort_keys=True),
            }
            for row in candidate_summary
        ],
    )
    _write_csv(
        run_dir / "candidate_fold_scores.csv",
        [
            {
                **{key: value for key, value in row.items() if key != "candidate"},
                "validation_subjects": json.dumps(row["validation_subjects"]),
                "candidate": json.dumps(row["candidate"], sort_keys=True),
            }
            for row in candidate_records
        ],
    )
    logger.info(
        "LOSO complete | subject-macro R2=%.6f +/- %.6f | median=%.6f | pooled R2=%.6f",
        summary["subject_macro_r2_mean"],
        summary["subject_macro_r2_std"],
        summary["subject_macro_r2_median"],
        summary["pooled_window_element_r2"],
    )
    logger.info("Saved artifacts to %s", run_dir)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a reconstruction-only subject-adversarial STS VAE and "
            "evaluate unseen-subject reconstruction with LOSOCV R^2."
        )
    )
    parser.add_argument("--out-dir", default="runs/inverse_subject_vae")
    parser.add_argument("--run-name", default="dreamer_inverse_subject_sts_vae")
    parser.add_argument(
        "--dataset",
        choices=("dreamer", "amigos", "eegemotions_27"),
        default="dreamer",
    )
    parser.add_argument("--n-channels", type=int, default=14)
    parser.add_argument("--n-bands", type=int, default=3)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-subjects", type=int, default=3)
    parser.add_argument("--validation-seed", type=int, default=42)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--candidate-verbose", type=int, choices=(0, 1, 2), default=0)

    parser.add_argument("--no-save-fold-models", action="store_true")
    parser.add_argument("--no-save-final-model", action="store_true")
    parser.add_argument("--save-fold-reconstructions", action="store_true")
    parser.add_argument("--no-save-adjacency-matrices", action="store_true")
    parser.add_argument("--skip-final-full-data-fit", action="store_true")

    parser.add_argument("--optimizer", choices=("adam", "adamw"), default="adamw")
    parser.add_argument("--vae-learning-rate", type=float, default=5e-5)
    parser.add_argument("--subject-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--subject-steps-per-batch", type=int, default=1)
    parser.add_argument("--vae-steps-per-batch", type=int, default=1)

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
    parser.add_argument("--vae-loss-weight", type=float, default=1.0)
    parser.add_argument("--vae-beta", type=float, default=0.30)
    parser.add_argument("--subject-adversarial-weight", type=float, default=1.0)
    parser.add_argument("--subject-loss-weight", type=float, default=1.0)
    parser.add_argument("--subject-hidden-units", type=int, default=64)
    parser.add_argument("--subject-dropout", type=float, default=0.0)

    parser.add_argument("--features-npy", default=None)
    parser.add_argument("--subjects-npy", default=None)
    parser.add_argument("--trials-npy", default=None)
    parser.add_argument("--raw-eeg-npy", default=None)
    parser.add_argument(
        "--raw-labels-npy",
        default=None,
        help=(
            "Used only by the shared raw-data window builder. Emotional labels "
            "are discarded and never used as model targets."
        ),
    )
    parser.add_argument("--window-sec", type=float, default=4.0)
    parser.add_argument("--window-overlap", type=float, default=0.0)
    parser.add_argument("--fs", type=float, default=DREAMER_FS)
    parser.add_argument("--median-label", type=float, default=DREAMER_MEDIAN_LABEL)
    parser.add_argument(
        "--window-normalization",
        choices=("none", "global_rms", "feature_zscore"),
        default="global_rms",
    )
    parser.add_argument("--hyperparameters-json", default=None)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace, hyperparameters: dict) -> None:
    positive_values = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "validation_subjects": args.validation_subjects,
        "early_stopping_patience": args.early_stopping_patience,
        "vae_learning_rate": args.vae_learning_rate,
        "subject_learning_rate": args.subject_learning_rate,
        "subject_steps_per_batch": args.subject_steps_per_batch,
        "vae_steps_per_batch": args.vae_steps_per_batch,
        "n_channels": args.n_channels,
        "n_bands": args.n_bands,
        "window_sec": args.window_sec,
        "fs": args.fs,
    }
    invalid = {key: value for key, value in positive_values.items() if value <= 0}
    if invalid:
        raise ValueError(f"These arguments must be positive: {invalid}")
    if args.weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative.")
    if args.vae_beta < 0.0:
        raise ValueError("vae_beta must be non-negative.")
    if args.subject_adversarial_weight < 0.0 or args.subject_loss_weight < 0.0:
        raise ValueError("Subject adversarial weights must be non-negative.")
    if not 0.0 <= args.window_overlap < 1.0:
        raise ValueError("window_overlap must be in [0, 1).")
    allowed = _allowed_model_hyperparameters() | _FIT_HPARAMETERS
    unknown = set(hyperparameters) - allowed
    if unknown:
        raise ValueError(f"Unknown hyperparameter grid keys: {sorted(unknown)}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    hyperparameters = {}
    if args.hyperparameters_json is not None:
        with Path(args.hyperparameters_json).open("r", encoding="utf-8") as handle:
            hyperparameters = json.load(handle)
        if not isinstance(hyperparameters, dict):
            raise ValueError("hyperparameters JSON must contain an object.")
    _validate_args(args, hyperparameters)

    config = InverseSubjectVAETrainingConfig(
        output_dir=Path(args.out_dir),
        run_name=args.run_name,
        dataset=args.dataset,
        n_channels=args.n_channels,
        n_bands=args.n_bands,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_subjects_per_fold=args.validation_subjects,
        validation_seed=args.validation_seed,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        max_folds=args.max_folds,
        seed=args.seed,
        verbose=args.verbose,
        candidate_verbose=args.candidate_verbose,
        save_fold_models=not args.no_save_fold_models,
        save_final_model=not args.no_save_final_model,
        save_fold_reconstructions=args.save_fold_reconstructions,
        save_adjacency_matrices=not args.no_save_adjacency_matrices,
        run_final_full_data_fit=not args.skip_final_full_data_fit,
        optimizer_name=args.optimizer,
        vae_learning_rate=args.vae_learning_rate,
        subject_learning_rate=args.subject_learning_rate,
        weight_decay=args.weight_decay,
        subject_steps_per_batch=args.subject_steps_per_batch,
        vae_steps_per_batch=args.vae_steps_per_batch,
        t_down=args.t_down,
        temporal_pool_sizes=tuple(args.temporal_pool_sizes),
        bilstm_units=args.bilstm_units,
        n_bilstm_layers=args.bilstm_layers,
        bilstm_dropout=args.bilstm_dropout,
        temporal_emb_dim=args.temporal_emb_dim,
        gcn_units=tuple(args.gcn_units),
        spectral_emb_dim=args.spectral_emb_dim,
        gcn_dropout=args.gcn_dropout,
        gcn_activation=args.gcn_activation,
        gcn_use_batch_norm=args.gcn_use_batch_norm,
        graph_self_loop_bias=args.graph_self_loop_bias,
        graph_identity_mix=args.graph_identity_mix,
        graph_adjacency_reg_weight=args.graph_adjacency_reg_weight,
        fusion_dim=args.fusion_dim,
        latent_features=args.latent_features,
        fusion_dropout=args.fusion_dropout,
        activation=args.activation,
        decoder_temporal_units=args.decoder_temporal_units,
        decoder_bilstm_layers=args.decoder_bilstm_layers,
        decoder_graph_output_units=args.decoder_graph_output_units,
        decoder_branch_feature_dim=args.decoder_branch_feature_dim,
        decoder_fusion_units=args.decoder_fusion_units,
        decoder_dropout=args.decoder_dropout,
        reconstruction_loss=args.reconstruction_loss,
        vae_loss_weight=args.vae_loss_weight,
        vae_beta=args.vae_beta,
        subject_adversarial_weight=args.subject_adversarial_weight,
        subject_loss_weight=args.subject_loss_weight,
        subject_hidden_units=args.subject_hidden_units,
        subject_dropout=args.subject_dropout,
        window_size_sec=args.window_sec,
        fs=args.fs,
        overlap=args.window_overlap,
        median_label=args.median_label,
        window_normalization=args.window_normalization,
        hyperparameters=hyperparameters,
    )
    features, subjects, trials = load_inverse_subject_training_data(
        features_path=args.features_npy,
        subjects_path=args.subjects_npy,
        trials_path=args.trials_npy,
        raw_eeg_path=args.raw_eeg_npy,
        raw_labels_path=args.raw_labels_npy,
        config=config,
    )
    train_inverse_subject_vae_loso(
        features=features,
        subject_ids=subjects,
        trial_ids=trials,
        config=config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
