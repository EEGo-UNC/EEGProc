"""Training entry point for the fused spatiotemporal-spatiospectral model.

The STS model classifies and reconstructs one EEG window at a time. Its
parallel BiLSTM and GCN encoders are fused into a variational latent sequence,
which feeds a dense/variational classifier and a dual-path BiLSTM-GCN decoder.
Classification supports ordinary updates or first-order subject-domain MLDG;
the VAE objective remains a separate alternating phase inside ``train_step``.

Cross-validation supports strict leave-one-subject-out (LOSO) and the existing
leave-N-subjects-and-K-trials-out (LNSKTO) protocol. Although the neural model
is window-level, configuration selection, decision-threshold fitting, and final
reporting may aggregate window probabilities by ``(subject_id, trial_id)``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf

try:
    from .joint_sts_cli import (
        _json_default,
        _validate_args,
        _write_csv,
        _write_json,
        parse_args,
    )
except ImportError:
    # Preserve the direct-execution behavior used by the rest of this module.
    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    from joint_sts_cli import (
        _json_default,
        _validate_args,
        _write_csv,
        _write_json,
        parse_args,
    )

try:
    from .joint_sts_model import JointSTSModel, build_joint_sts_model
    from ..joint_v2_data import (
        DatasetConfig,
        DEFAULT_DREAMER_EEG_PATH,
        DEFAULT_DREAMER_LABELS_PATH,
        DREAMER_FS,
        DREAMER_MEDIAN_LABEL,
        build_joint_v2_dataset,
        get_dataset_config,
    )
except ImportError:
    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    from joint_sts_model import JointSTSModel, build_joint_sts_model
    from joint_v2_data import (
        DatasetConfig,
        DEFAULT_DREAMER_EEG_PATH,
        DEFAULT_DREAMER_LABELS_PATH,
        DREAMER_FS,
        DREAMER_MEDIAN_LABEL,
        build_joint_v2_dataset,
        get_dataset_config,
    )

try:
    try:
        from ...cross_val import (
            MetaLearningSubjectSequence,
            PredictionDiagnostics,
            fixed_loso_cv,
            loso_cv,
        )
    except ImportError:
        from ...cross_val import (
            MetaLearningSubjectSequence,
            PredictionDiagnostics,
            fixed_loso_cv,
            loso_cv,
        )
except ImportError:
    SRC_ROOT = Path(__file__).resolve().parents[3]
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    try:
        from eegproc.deep_learning.cross_val import (
            MetaLearningSubjectSequence,
            PredictionDiagnostics,
            fixed_loso_cv,
            loso_cv,
        )
    except ImportError:
        from eegproc.deep_learning.cross_val import (
            MetaLearningSubjectSequence,
            PredictionDiagnostics,
            fixed_loso_cv,
            loso_cv,
        )


@dataclass(slots=True)
class JointSTSTrainingConfig:
    """Complete training, architecture, and CV configuration for STS."""

    output_dir: Path = Path("runs") / "joint_sts"
    run_name: str = "joint_sts"
    dataset: str = "dreamer"
    n_channels: int = 14
    n_bands: int | None = 3

    # Alternating optimizer configuration.
    optimizer_name: str = "adamw"
    classification_learning_rate: float = 1e-4
    vae_learning_rate: float = 5e-5
    discriminator_learning_rate: float | None = None
    weight_decay: float = 1e-4
    classification_steps_per_batch: int = 1
    vae_steps_per_batch: int = 1

    # Fit and cross-validation configuration.
    batch_size: int = 32
    cv_max_epochs: int = 100
    cv_strategy: str = "loso"
    lnskto_subjects: int = 3
    lnskto_trials: int = 3
    lnskto_split_seed: int | None = 42
    lnskto_require_all_classes_in_test: bool = True
    final_epoch_strategy: str = "median"
    final_epochs: int | None = None
    run_no_validation_loso_before_final: bool = True
    selection_metric: str = "f1"
    selection_level: str = "trial"
    maximize_metric: bool | None = None
    max_folds: int | None = None
    n_jobs: int = 1
    cpus_per_worker: int | None = None
    alternate_subject_sets: bool = False
    alternating_subject_seed: int | None = 42

    # First-order MLDG subject-domain generalization.
    use_mldg: bool = False
    mldg_inner_learning_rate: float = 1e-4
    mldg_meta_test_weight: float = 1.0
    mldg_meta_train_subjects: int = 6
    mldg_meta_test_subjects: int = 2
    mldg_samples_per_subject: int = 4
    mldg_seed: int | None = 42

    # Prediction, thresholds, and diagnostics.
    prediction_latent_samples: int = 0
    latent_sampling_seed: int | None = None
    decision_thresholds: tuple[float, ...] = (0.5,)
    threshold_selection_metric: str = "f1"
    threshold_selection_level: str = "trial"
    prediction_diagnostics: bool = True
    prediction_diagnostics_every_n_epochs: int = 1
    prediction_diagnostics_max_samples: int = 256
    prediction_diagnostics_threshold_tolerance: float = 0.01
    prediction_diagnostics_seed: int | None = 42

    # Validation and checkpointing.
    validation_subjects_per_fold: int = 2
    validation_seed: int | None = 42
    outer_verbose: int = 0
    final_verbose: int = 1
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 0.001
    early_stopping_monitor: str = "val_accuracy"
    early_stopping_mode: str = "max"
    use_early_stopping: bool = True
    save_full_model: bool = True
    save_weights: bool = True
    save_final_history_csv: bool = True
    save_adjacency_matrices: bool = True
    seed: int | None = 42

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

    # Fused variational posterior.
    fusion_dim: int = 64
    latent_features: int = 32
    fusion_dropout: float = 0.20
    activation: str = "relu"

    # Dual-path fused decoder.
    decoder_temporal_units: int = 64
    decoder_bilstm_layers: int = 1
    decoder_graph_output_units: int = 16
    decoder_branch_feature_dim: int = 64
    decoder_fusion_units: int = 64
    decoder_dropout: float = 0.20
    reconstruction_loss: str = "mse"

    # Classification pathway and optional variational head terms.
    classification_hidden_units: int = 64
    classification_dropout: float = 0.30
    classifier_head: str = "dense"
    classifier_kwargs: dict = field(default_factory=dict)
    label_smoothing: float = 0.0
    focal_gamma: float = 1.0
    focal_alpha: tuple[float, ...] | None = None
    classification_loss_weight: float = 1.0
    vae_loss_weight: float = 1.0
    vae_beta: float = 0.30
    vc_alpha: float = 1.0
    vc_beta: float = 0.0
    vc_gamma: float = 0.0
    vc_lambda: float = 0.0
    update_discriminator: bool = False
    use_class_weight: bool = True

    # Subject-adversarial objective.
    use_subject_adversarial: bool = False
    subject_adversarial_weight: float = 0.05
    subject_loss_weight: float = 1.0
    subject_hidden_units: int = 64
    subject_dropout: float = 0.0
    subject_latent_mode: str = "mean"
    subject_mc_samples: int = 5

    # Supervised contrastive objective.
    use_supcon: bool = False
    supcon_weight: float = 0.03
    supcon_temperature: float = 0.10
    supcon_cross_subject_only: bool = True

    # Dataset protocol.
    label_threshold_mode: str = "global"
    window_normalization: str = "global_rms"

    # Cross-validation grid. Values follow cross_val.py's existing convention.
    hyperparameters: dict = field(default_factory=dict)


def _flatten_grouped_trials_to_windows(
    feature_array: np.ndarray,
    label_array: np.ndarray,
    subject_id_array: np.ndarray,
    trial_id_array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert a legacy rank-4 trial tensor into aligned rank-3 windows."""
    features = np.asarray(feature_array, dtype=np.float32)
    labels = np.asarray(label_array)
    subjects = np.asarray(subject_id_array).reshape(-1)
    trials = np.asarray(trial_id_array).reshape(-1)
    if features.ndim != 4:
        return features, labels, subjects, trials
    if not (len(features) == len(labels) == len(subjects) == len(trials)):
        raise ValueError("Grouped trial arrays must align on axis 0.")
    n_trials, n_windows, timesteps, n_features = features.shape
    return (
        features.reshape(n_trials * n_windows, timesteps, n_features),
        np.repeat(labels, n_windows, axis=0),
        np.repeat(subjects, n_windows),
        np.repeat(trials, n_windows),
    )


def _normalize_each_window(
    feature_array: np.ndarray,
    mode: str = "global_rms",
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Normalize each window while optionally preserving channel-band ratios."""
    features = np.asarray(feature_array, dtype=np.float32)
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
        normalized = features.astype(np.float64) / np.maximum(rms, epsilon)
    elif mode == "feature_zscore":
        mean = np.mean(features, axis=1, keepdims=True, dtype=np.float64)
        std = np.std(features, axis=1, keepdims=True, dtype=np.float64)
        normalized = (features.astype(np.float64) - mean) / np.maximum(std, epsilon)
    else:
        raise ValueError(
            "window_normalization must be none, global_rms, or feature_zscore."
        )
    if not np.isfinite(normalized).all():
        raise ValueError("Window normalization produced NaN or Inf values.")
    return normalized.astype(np.float32)


def _subject_median_window_labels(
    labels_path: str | Path,
    label_dimension: str,
    subject_id_array: np.ndarray,
    trial_id_array: np.ndarray,
) -> np.ndarray:
    """Create subject-relative binary labels and repeat them over windows."""
    raw_labels = np.load(Path(labels_path), allow_pickle=False)
    if raw_labels.ndim != 3 or raw_labels.shape[-1] < 2:
        raise ValueError(
            "subject_median requires raw labels shaped (subjects, trials, >=2)."
        )
    dimension_index = {"valence": 0, "arousal": 1}[label_dimension]
    subjects = np.asarray(subject_id_array).reshape(-1)
    trials = np.asarray(trial_id_array).reshape(-1)
    output = np.empty(len(subjects), dtype=np.int64)
    unique_subjects = np.unique(subjects)
    if len(unique_subjects) != raw_labels.shape[0]:
        raise ValueError(
            "Raw-label subject count does not match the window subject IDs."
        )

    for subject_row, subject_id in enumerate(sorted(unique_subjects.tolist())):
        subject_mask = subjects == subject_id
        unique_trials = sorted(np.unique(trials[subject_mask]).tolist())
        if len(unique_trials) != raw_labels.shape[1]:
            raise ValueError(
                f"Subject {subject_id!r} has {len(unique_trials)} trial IDs, "
                f"but the label array has {raw_labels.shape[1]} trials."
            )
        ratings = raw_labels[subject_row, :, dimension_index].astype(np.float64)
        threshold = float(np.median(ratings))
        binary_by_trial = (ratings >= threshold).astype(np.int64)
        trial_to_label = {
            trial_id: int(binary_by_trial[index])
            for index, trial_id in enumerate(unique_trials)
        }
        output[subject_mask] = np.asarray(
            [trial_to_label[trial_id] for trial_id in trials[subject_mask]],
            dtype=np.int64,
        )
    return output


def load_joint_sts_training_data(
    eeg_path: str | Path = DEFAULT_DREAMER_EEG_PATH,
    labels_path: str | Path = DEFAULT_DREAMER_LABELS_PATH,
    label_dimension: str = "valence",
    window_size_sec: float = 4.0,
    fs: float = DREAMER_FS,
    overlap: float = 0.5,
    median_label: float = DREAMER_MEDIAN_LABEL,
    window_normalization: str = "global_rms",
    label_threshold_mode: str = "global",
    dataset: str | DatasetConfig = "dreamer",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load flat EEG windows for STS classification and reconstruction."""
    window_size = int(round(window_size_sec * fs))
    if window_size < 1:
        raise ValueError("window_size_sec * fs must be positive.")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1).")
    if window_normalization not in {"none", "global_rms", "feature_zscore"}:
        raise ValueError("Unknown window_normalization mode.")
    if label_threshold_mode not in {"global", "subject_median"}:
        raise ValueError("label_threshold_mode must be global or subject_median.")

    dataset_arrays = build_joint_v2_dataset(
        eeg_path=eeg_path,
        labels_path=labels_path,
        label_dimension=label_dimension,
        window_size_sec=window_size_sec,
        fs=fs,
        overlap=overlap,
        median_label=median_label,
        zscore=False,
        dataset=dataset,
    )
    if len(dataset_arrays) == 4:
        features, labels, subjects, trials = dataset_arrays
    elif len(dataset_arrays) == 3:
        features, labels, subjects = dataset_arrays
        raw_eeg = np.load(Path(eeg_path), mmap_mode="r", allow_pickle=False)
        if raw_eeg.ndim != 4:
            raise ValueError(
                "Raw EEG must have shape (subjects, trials, channels, samples)."
            )
        n_subjects, n_trials, _n_channels, n_samples = raw_eeg.shape
        hop = max(1, int(round(window_size * (1.0 - overlap))))
        n_windows_per_trial = 1 + (n_samples - window_size) // hop
        trials = np.tile(
            np.repeat(np.arange(n_trials, dtype=np.int64), n_windows_per_trial),
            n_subjects,
        )
    else:
        raise ValueError("build_joint_v2_dataset must return 3 or 4 arrays.")

    features, labels, subjects, trials = _flatten_grouped_trials_to_windows(
        features,
        labels,
        subjects,
        trials,
    )
    if features.ndim != 3:
        raise ValueError(
            "STS requires rank-3 windows shaped (windows, timesteps, features); "
            f"got {features.shape}."
        )
    lengths = tuple(len(array) for array in (features, labels, subjects, trials))
    if len(set(lengths)) != 1:
        raise ValueError(f"Window arrays are not aligned: {lengths}.")

    if label_threshold_mode == "subject_median":
        labels = _subject_median_window_labels(
            labels_path=labels_path,
            label_dimension=label_dimension,
            subject_id_array=subjects,
            trial_id_array=trials,
        )
    features = _normalize_each_window(features, mode=window_normalization)
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels),
        np.asarray(subjects),
        np.asarray(trials),
    )


def _load_numpy_array(path: str | Path) -> np.ndarray:
    return np.load(Path(path), allow_pickle=False)


def _ensure_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _as_class_ids(label_array: np.ndarray) -> np.ndarray:
    labels = np.asarray(label_array)
    if labels.ndim == 1:
        return labels.astype(np.int64, copy=False)
    if labels.ndim == 2 and labels.shape[1] == 1:
        return labels[:, 0].astype(np.int64, copy=False)
    if labels.ndim == 2 and labels.shape[1] > 1:
        return np.argmax(labels, axis=1).astype(np.int64, copy=False)
    raise ValueError(
        "Labels must have shape (n,), (n, 1), or (n, n_classes); "
        f"got {labels.shape}."
    )


def _infer_n_classes(label_array: np.ndarray) -> int:
    labels = np.asarray(label_array)
    if labels.ndim == 2 and labels.shape[1] > 1:
        return int(labels.shape[1])
    flattened = labels.reshape(-1)
    if flattened.size == 0:
        raise ValueError("label_array must not be empty.")
    return int(np.max(flattened)) + 1


def _resolve_channel_band_shape(
    n_features: int,
    n_channels: int,
    n_bands: int | None,
) -> tuple[int, int]:
    n_features = int(n_features)
    n_channels = int(n_channels)
    if n_features < 1 or n_channels < 1:
        raise ValueError("n_features and n_channels must be positive.")
    if n_bands is None:
        if n_features % n_channels != 0:
            raise ValueError(
                f"Cannot infer n_bands because {n_features} is not divisible "
                f"by n_channels={n_channels}."
            )
        n_bands = n_features // n_channels
    n_bands = int(n_bands)
    if n_bands < 1 or n_channels * n_bands != n_features:
        raise ValueError(
            "STS input must satisfy n_features = n_channels * n_bands; got "
            f"{n_features} != {n_channels} * {n_bands}."
        )
    return n_channels, n_bands


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


def _configure_run_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"eegproc.joint_sts.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(run_dir / "training.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def _grid_summary_rows(cv_results: dict) -> list[dict]:
    rows: list[dict] = []
    for result in cv_results.get("config_results", []):
        row = {
            "config_index": int(result["config_index"]),
            "is_selected": int(
                result["config_index"] == cv_results.get("best_config_index")
            ),
            "selection_level": result.get("selection_level"),
            "selection_metric": result.get("selection_metric"),
            "selection_score": result.get("selection_score"),
            "selection_score_std": result.get("selection_score_std"),
            "n_folds": result.get("n_folds"),
            "config": json.dumps(
                result.get("config", {}),
                sort_keys=True,
                default=_json_default,
            ),
        }
        for prefix in ("window", "trial"):
            for metric, value in result.get(f"{prefix}_mean_scores", {}).items():
                row[f"{prefix}_{metric}_mean"] = value
            for metric, value in result.get(f"{prefix}_std_scores", {}).items():
                row[f"{prefix}_{metric}_std"] = value
        rows.append(row)
    return rows


def _cv_fold_records(cv_results: dict) -> list[dict]:
    """Return canonical per-fold records across current and legacy schemas."""
    for key in ("fold_results", "outer_fold_results"):
        rows = cv_results.get(key)
        if rows:
            return list(rows)
    rows = cv_results.get("best_config_result", {}).get("fold_results", [])
    return list(rows)


def _select_final_epochs_from_cv(
    cv_results: dict,
    strategy: str,
    fallback_epochs: int,
) -> tuple[int, list[int]]:
    fold_rows = _cv_fold_records(cv_results)
    best_epochs = [
        int(row["best_epoch"])
        for row in fold_rows
        if row.get("best_epoch") is not None and int(row["best_epoch"]) >= 1
    ]
    if not best_epochs:
        return max(1, int(fallback_epochs)), []
    if strategy == "median":
        selected = int(np.rint(np.median(best_epochs)))
    elif strategy == "mean":
        selected = int(np.rint(np.mean(best_epochs)))
    elif strategy == "max":
        selected = int(np.max(best_epochs))
    else:
        raise ValueError("final_epoch_strategy must be median, mean, or max.")
    return max(1, selected), best_epochs


def _save_adjacency_matrices(model: JointSTSModel, run_dir: Path) -> None:
    """Save learned encoder/decoder graph matrices into one compressed archive."""
    if not hasattr(model, "get_adjacency_matrices"):
        return
    nested = model.get_adjacency_matrices()
    flattened: dict[str, np.ndarray] = {}
    for component_name, matrices in nested.items():
        for layer_name, matrix in matrices.items():
            flattened[f"{component_name}__{layer_name}"] = np.asarray(matrix.numpy())
    if flattened:
        np.savez_compressed(run_dir / "adjacency_matrices.npz", **flattened)


def _model_hparameter_keys() -> set[str]:
    return {
        "learning_rate",  # Alias for classification_learning_rate.
        "classification_learning_rate",
        "vae_learning_rate",
        "discriminator_learning_rate",
        "optimizer",
        "optimizer_name",
        "weight_decay",
        "classification_steps_per_batch",
        "vae_steps_per_batch",
        "mldg_inner_learning_rate",
        "mldg_meta_test_weight",
        "t_down",
        "temporal_pool_sizes",
        "bilstm_units",
        "bilstm_layers",
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
        "classification_hidden_units",
        "classification_dropout",
        "classifier_head",
        "classifier_kwargs",
        "label_smoothing",
        "focal_gamma",
        "focal_alpha",
        "classification_loss_weight",
        "vae_loss_weight",
        "vae_beta",
        "vc_alpha",
        "vc_beta",
        "vc_gamma",
        "vc_lambda",
        "update_discriminator",
        "use_subject_adversarial",
        "subject_adversarial_weight",
        "subject_loss_weight",
        "subject_hidden_units",
        "subject_dropout",
        "subject_latent_mode",
        "subject_mc_samples",
        "use_supcon",
        "supcon_weight",
        "supcon_temperature",
        "supcon_cross_subject_only",
    }


def _load_lnskto_cv():
    """Load the optional LNSKTO implementation only when it is requested.

    The current cross_val.py used by the LOSO pipeline may be LOSO-only.
    Keeping this import lazy prevents a missing legacy LNSKTO helper from
    breaking ordinary LOSO training at module-import time.
    """
    try:
        from ...cross_val import lnskto_cv as function
    except ImportError:
        try:
            from eegproc.deep_learning.cross_val import lnskto_cv as function
        except ImportError as exc:
            raise RuntimeError(
                "cv_strategy='lnskto' was requested, but this cross_val.py "
                "does not define lnskto_cv. Restore the LNSKTO implementation "
                "or use --cv-strategy loso."
            ) from exc
    return function


def train_joint_sts_model(
    feature_array: np.ndarray | None = None,
    label_array: np.ndarray | None = None,
    subject_id_array: np.ndarray | None = None,
    trial_id_array: np.ndarray | None = None,
    data_loader: Callable[
        [], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] | None = None,
    training_config: JointSTSTrainingConfig | None = None,
    model_builder_function: Callable[..., tf.keras.Model] | None = None,
) -> dict:
    """Train the alternating fused STS model with LOSO or LNSKTO CV."""
    config = training_config or JointSTSTrainingConfig()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = config.run_name if config.run_name.endswith("_sts") else f"{config.run_name}_sts"
    run_dir = _ensure_path(config.output_dir / f"{run_name}_{timestamp}")
    logger = _configure_run_logger(run_dir)

    if config.seed is not None:
        tf.keras.utils.set_random_seed(config.seed)
        np.random.seed(config.seed)

    if data_loader is not None:
        feature_array, label_array, subject_id_array, trial_id_array = data_loader()
    elif any(
        value is None
        for value in (feature_array, label_array, subject_id_array, trial_id_array)
    ):
        feature_array, label_array, subject_id_array, trial_id_array = (
            load_joint_sts_training_data(dataset=config.dataset)
        )

    feature_array, label_array, subject_id_array, trial_id_array = (
        _flatten_grouped_trials_to_windows(
            feature_array,
            label_array,
            subject_id_array,
            trial_id_array,
        )
    )
    feature_array = np.asarray(feature_array, dtype=np.float32)
    label_array = np.asarray(label_array)
    subject_id_array = np.asarray(subject_id_array).reshape(-1)
    trial_id_array = np.asarray(trial_id_array).reshape(-1)
    if feature_array.ndim != 3:
        raise ValueError(
            "JointSTSModel supports window tensors shaped "
            f"(windows, timesteps, features); got {feature_array.shape}."
        )
    lengths = tuple(
        len(value)
        for value in (
            feature_array,
            label_array,
            subject_id_array,
            trial_id_array,
        )
    )
    if len(set(lengths)) != 1:
        raise ValueError(f"Input arrays do not align: {lengths}.")

    config.n_channels, config.n_bands = _resolve_channel_band_shape(
        n_features=feature_array.shape[-1],
        n_channels=config.n_channels,
        n_bands=config.n_bands,
    )
    config.temporal_pool_sizes = _validate_temporal_pooling(
        config.t_down,
        config.temporal_pool_sizes,
    )

    if model_builder_function is None:
        allowed_hparams = _model_hparameter_keys()

        def model_builder_function(**hparams) -> tf.keras.Model:
            unknown = set(hparams) - allowed_hparams
            if unknown:
                raise ValueError(f"Unknown STS hyperparameter(s): {sorted(unknown)}")

            classifier_kwargs = dict(config.classifier_kwargs)
            classifier_kwargs.update(hparams.get("classifier_kwargs", {}))
            temporal_pool_sizes = hparams.get(
                "temporal_pool_sizes",
                config.temporal_pool_sizes,
            )
            t_down = int(hparams.get("t_down", config.t_down))
            temporal_pool_sizes = _validate_temporal_pooling(
                t_down,
                temporal_pool_sizes,
            )
            return build_joint_sts_model(
                input_shape=tuple(feature_array.shape[1:]),
                n_classes=_infer_n_classes(label_array),
                n_channels=config.n_channels,
                n_bands=config.n_bands,
                t_down=t_down,
                temporal_pool_sizes=temporal_pool_sizes,
                bilstm_units=int(hparams.get("bilstm_units", config.bilstm_units)),
                n_bilstm_layers=int(
                    hparams.get(
                        "bilstm_layers",
                        hparams.get("n_bilstm_layers", config.n_bilstm_layers),
                    )
                ),
                bilstm_dropout=float(
                    hparams.get("bilstm_dropout", config.bilstm_dropout)
                ),
                temporal_emb_dim=int(
                    hparams.get("temporal_emb_dim", config.temporal_emb_dim)
                ),
                gcn_units=tuple(
                    int(value)
                    for value in hparams.get("gcn_units", config.gcn_units)
                ),
                spectral_emb_dim=int(
                    hparams.get("spectral_emb_dim", config.spectral_emb_dim)
                ),
                gcn_dropout=float(
                    hparams.get("gcn_dropout", config.gcn_dropout)
                ),
                gcn_activation=str(
                    hparams.get("gcn_activation", config.gcn_activation)
                ),
                gcn_use_batch_norm=bool(
                    hparams.get("gcn_use_batch_norm", config.gcn_use_batch_norm)
                ),
                graph_self_loop_bias=float(
                    hparams.get(
                        "graph_self_loop_bias",
                        config.graph_self_loop_bias,
                    )
                ),
                graph_identity_mix=float(
                    hparams.get("graph_identity_mix", config.graph_identity_mix)
                ),
                graph_adjacency_reg_weight=float(
                    hparams.get(
                        "graph_adjacency_reg_weight",
                        config.graph_adjacency_reg_weight,
                    )
                ),
                fusion_dim=int(hparams.get("fusion_dim", config.fusion_dim)),
                latent_features=int(
                    hparams.get("latent_features", config.latent_features)
                ),
                fusion_dropout=float(
                    hparams.get("fusion_dropout", config.fusion_dropout)
                ),
                activation=str(hparams.get("activation", config.activation)),
                decoder_temporal_units=int(
                    hparams.get(
                        "decoder_temporal_units",
                        config.decoder_temporal_units,
                    )
                ),
                decoder_bilstm_layers=int(
                    hparams.get(
                        "decoder_bilstm_layers",
                        config.decoder_bilstm_layers,
                    )
                ),
                decoder_graph_output_units=int(
                    hparams.get(
                        "decoder_graph_output_units",
                        config.decoder_graph_output_units,
                    )
                ),
                decoder_branch_feature_dim=int(
                    hparams.get(
                        "decoder_branch_feature_dim",
                        config.decoder_branch_feature_dim,
                    )
                ),
                decoder_fusion_units=int(
                    hparams.get(
                        "decoder_fusion_units",
                        config.decoder_fusion_units,
                    )
                ),
                decoder_dropout=float(
                    hparams.get("decoder_dropout", config.decoder_dropout)
                ),
                reconstruction_loss=str(
                    hparams.get("reconstruction_loss", config.reconstruction_loss)
                ),
                classification_hidden_units=int(
                    hparams.get(
                        "classification_hidden_units",
                        config.classification_hidden_units,
                    )
                ),
                classification_dropout=float(
                    hparams.get(
                        "classification_dropout",
                        config.classification_dropout,
                    )
                ),
                classifier_head=str(
                    hparams.get("classifier_head", config.classifier_head)
                ),
                classifier_kwargs=classifier_kwargs,
                label_smoothing=float(
                    hparams.get("label_smoothing", config.label_smoothing)
                ),
                focal_gamma=float(
                    hparams.get("focal_gamma", config.focal_gamma)
                ),
                focal_alpha=hparams.get("focal_alpha", config.focal_alpha),
                classification_loss_weight=float(
                    hparams.get(
                        "classification_loss_weight",
                        config.classification_loss_weight,
                    )
                ),
                vae_loss_weight=float(
                    hparams.get("vae_loss_weight", config.vae_loss_weight)
                ),
                vae_beta=float(hparams.get("vae_beta", config.vae_beta)),
                vc_alpha=float(hparams.get("vc_alpha", config.vc_alpha)),
                vc_beta=float(hparams.get("vc_beta", config.vc_beta)),
                vc_gamma=float(hparams.get("vc_gamma", config.vc_gamma)),
                vc_lambda=float(hparams.get("vc_lambda", config.vc_lambda)),
                update_discriminator=bool(
                    hparams.get("update_discriminator", config.update_discriminator)
                ),
                # Class weighting is a run-level switch. In particular,
                # --no-class-weight must override every CV configuration and
                # cannot be re-enabled through the hyperparameter grid.
                use_class_weight=bool(config.use_class_weight),
                use_subject_adversarial=bool(
                    hparams.get(
                        "use_subject_adversarial",
                        config.use_subject_adversarial,
                    )
                ),
                subject_adversarial_weight=float(
                    hparams.get(
                        "subject_adversarial_weight",
                        config.subject_adversarial_weight,
                    )
                ),
                subject_loss_weight=float(
                    hparams.get("subject_loss_weight", config.subject_loss_weight)
                ),
                subject_hidden_units=int(
                    hparams.get("subject_hidden_units", config.subject_hidden_units)
                ),
                subject_dropout=float(
                    hparams.get("subject_dropout", config.subject_dropout)
                ),
                subject_latent_mode=str(
                    hparams.get("subject_latent_mode", config.subject_latent_mode)
                ),
                subject_mc_samples=int(
                    hparams.get("subject_mc_samples", config.subject_mc_samples)
                ),
                use_supcon=bool(hparams.get("use_supcon", config.use_supcon)),
                supcon_weight=float(
                    hparams.get("supcon_weight", config.supcon_weight)
                ),
                supcon_temperature=float(
                    hparams.get("supcon_temperature", config.supcon_temperature)
                ),
                supcon_cross_subject_only=bool(
                    hparams.get(
                        "supcon_cross_subject_only",
                        config.supcon_cross_subject_only,
                    )
                ),
                classification_steps_per_batch=int(
                    hparams.get(
                        "classification_steps_per_batch",
                        config.classification_steps_per_batch,
                    )
                ),
                vae_steps_per_batch=int(
                    hparams.get("vae_steps_per_batch", config.vae_steps_per_batch)
                ),
                # MLDG changes the required batch structure and is therefore
                # a run-level switch, not a per-grid model hyperparameter.
                use_mldg=bool(config.use_mldg),
                mldg_inner_learning_rate=float(
                    hparams.get(
                        "mldg_inner_learning_rate",
                        config.mldg_inner_learning_rate,
                    )
                ),
                mldg_meta_test_weight=float(
                    hparams.get(
                        "mldg_meta_test_weight",
                        config.mldg_meta_test_weight,
                    )
                ),
                optimizer_name=str(
                    hparams.get(
                        "optimizer",
                        hparams.get("optimizer_name", config.optimizer_name),
                    )
                ),
                classification_learning_rate=float(
                    hparams.get(
                        "learning_rate",
                        hparams.get(
                            "classification_learning_rate",
                            config.classification_learning_rate,
                        ),
                    )
                ),
                vae_learning_rate=float(
                    hparams.get("vae_learning_rate", config.vae_learning_rate)
                ),
                discriminator_learning_rate=(
                    config.discriminator_learning_rate
                    if hparams.get(
                        "discriminator_learning_rate",
                        config.discriminator_learning_rate,
                    )
                    is None
                    else float(
                        hparams.get(
                            "discriminator_learning_rate",
                            config.discriminator_learning_rate,
                        )
                    )
                ),
                weight_decay=float(
                    hparams.get("weight_decay", config.weight_decay)
                ),
                model_name="joint_sts_model",
            )

    logger.info("Starting joint STS training run in %s", run_dir)
    logger.info(
        "Architecture: parallel BiLSTM + GCN encoders -> fused VAE posterior -> "
        "dense/VC classifier + dual-path BiLSTM-GCN decoder"
    )
    logger.info("Feature tensor shape: %s", feature_array.shape)
    logger.info(
        "Channel-band grid: %d channels x %d bands",
        config.n_channels,
        config.n_bands,
    )
    logger.info(
        "Alternating updates: classification=%d, VAE=%d per batch",
        config.classification_steps_per_batch,
        config.vae_steps_per_batch,
    )
    logger.info(
        "Optimizers: %s, classification_lr=%.8g, vae_lr=%.8g, weight_decay=%.8g",
        config.optimizer_name,
        config.classification_learning_rate,
        config.vae_learning_rate,
        config.weight_decay,
    )
    logger.info("Classifier head: %s", config.classifier_head)
    logger.info(
        "Classification loss: focal (gamma=%s, alpha=%s)",
        config.focal_gamma,
        config.focal_alpha,
    )
    logger.info(
        "Keras class weighting enabled: %s",
        config.use_class_weight,
    )
    logger.info("Cross-validation strategy: %s", config.cv_strategy)
    logger.info(
        "Alternating two-subject-set optimization: %s (seed=%s)",
        config.alternate_subject_sets,
        config.alternating_subject_seed,
    )
    logger.info(
        "First-order MLDG: %s (sampling=natural_within_subject, "
        "inner_lr=%.8g, meta_test_weight=%.6g, A_subjects=%d, "
        "B_subjects=%d, samples_per_subject=%d, seed=%s)",
        config.use_mldg,
        config.mldg_inner_learning_rate,
        config.mldg_meta_test_weight,
        config.mldg_meta_train_subjects,
        config.mldg_meta_test_subjects,
        config.mldg_samples_per_subject,
        config.mldg_seed,
    )
    logger.info(
        "Selection: %s_%s; thresholds selected by %s_%s",
        config.selection_level,
        config.selection_metric,
        config.threshold_selection_level,
        config.threshold_selection_metric,
    )
    logger.info(
        "Metric convention: MTLFuseNet-compatible trial aggregation; "
        "f1/precision/recall treat class 1 as positive"
    )
    logger.info("Subject adversarial enabled: %s", config.use_subject_adversarial)
    logger.info("SupCon enabled: %s", config.use_supcon)
    logger.info("Window normalization: %s", config.window_normalization)
    logger.info("Label threshold mode: %s", config.label_threshold_mode)
    logger.info("Unique subjects: %d", len(np.unique(subject_id_array)))
    logger.info("Windows: %d", len(feature_array))
    logger.info(
        "Unique subject-trial groups: %d",
        len(set(zip(subject_id_array.tolist(), trial_id_array.tolist()))),
    )

    class_ids = _as_class_ids(label_array)
    class_values, class_counts = np.unique(class_ids, return_counts=True)
    class_distribution = {
        int(value): int(count)
        for value, count in zip(class_values, class_counts)
    }
    majority_baseline = float(np.max(class_counts) / np.sum(class_counts))
    logger.info("Global class counts: %s", class_distribution)
    logger.info("Majority-class accuracy baseline: %.6f", majority_baseline)
    if majority_baseline >= 0.60:
        logger.warning(
            "Class imbalance is substantial. Inspect predicted_class_*_fraction "
            "and trial-level F1 in addition to accuracy."
        )
    if config.use_supcon and config.batch_size < 8:
        logger.warning(
            "SupCon is enabled with batch_size=%d; valid positive pairs may be rare.",
            config.batch_size,
        )

    _write_json(run_dir / "training_config.json", asdict(config))

    # cross_val.py uses this metadata to distinguish list-valued layer settings
    # from grid axes. Each setting below is one flat sequence of integers.
    model_builder_function._sequence_hyperparameter_depths = {
        "gcn_units": 1,
        "temporal_pool_sizes": 1,
        "focal_alpha": 1,
    }

    common_cv_kwargs = {
        "model_builder_function": model_builder_function,
        "feature_array": feature_array,
        "label_array": label_array,
        "subject_id_array": subject_id_array,
        "trial_id_array": trial_id_array,
        "n_epochs": config.cv_max_epochs,
        "batch_size": config.batch_size,
        "hyperparameters": config.hyperparameters,
        # MTLFuseNet reports classification after averaging window probabilities
        # within each trial, so trial-level metrics are the primary fold metrics.
        "evaluation_level": "trial",
        "selection_level": config.selection_level,
        "selection_metric": config.selection_metric,
        "maximize_metric": config.maximize_metric,
        "metrics": (
            "accuracy",
            "f1",
            "precision",
            "recall",
            "macro_f1",
            "macro_precision",
            "macro_recall",
            "balanced_accuracy",
        ),
        "log_predictions": True,
        "n_prediction_latent_samples": config.prediction_latent_samples,
        "latent_sampling_seed": config.latent_sampling_seed,
        "decision_thresholds": config.decision_thresholds,
        "threshold_selection_metric": config.threshold_selection_metric,
        "threshold_selection_level": config.threshold_selection_level,
        "prediction_diagnostics": config.prediction_diagnostics,
        "prediction_diagnostics_every_n_epochs": (
            config.prediction_diagnostics_every_n_epochs
        ),
        "prediction_diagnostics_max_samples": (
            config.prediction_diagnostics_max_samples
        ),
        "prediction_diagnostics_threshold_tolerance": (
            config.prediction_diagnostics_threshold_tolerance
        ),
        "prediction_diagnostics_seed": config.prediction_diagnostics_seed,
        "validation_subjects_per_fold": (
            config.validation_subjects_per_fold if config.use_early_stopping else 0
        ),
        "validation_seed": config.validation_seed,
        "early_stopping_patience": (
            config.early_stopping_patience if config.use_early_stopping else None
        ),
        "early_stopping_min_delta": config.early_stopping_min_delta,
        "early_stopping_monitor": config.early_stopping_monitor,
        "early_stopping_mode": config.early_stopping_mode,
        "restore_best_weights": True,
        "verbose": config.outer_verbose,
        "extra_fit_kwargs": {
            "callbacks": [tf.keras.callbacks.TerminateOnNaN()]
        },
        "n_jobs": config.n_jobs,
        "cpus_per_worker": config.cpus_per_worker,
    }

    if config.use_mldg and config.cv_strategy != "loso":
        raise ValueError("First-order MLDG is currently implemented for LOSO only.")

    if config.cv_strategy == "loso":
        cv_results = loso_cv(
            **common_cv_kwargs,
            max_folds=config.max_folds,
            alternate_subject_sets=config.alternate_subject_sets,
            alternating_subject_seed=config.alternating_subject_seed,
            use_mldg=config.use_mldg,
            mldg_meta_train_subjects=config.mldg_meta_train_subjects,
            mldg_meta_test_subjects=config.mldg_meta_test_subjects,
            mldg_samples_per_subject=config.mldg_samples_per_subject,
            mldg_seed=config.mldg_seed,
        )
    elif config.cv_strategy == "lnskto":
        lnskto_cv_function = _load_lnskto_cv()
        cv_results = lnskto_cv_function(
            **common_cv_kwargs,
            n_subjects=config.lnskto_subjects,
            k_trials=config.lnskto_trials,
            n_folds=config.max_folds,
            split_seed=config.lnskto_split_seed,
            require_all_classes_in_test=config.lnskto_require_all_classes_in_test,
        )
    else:
        raise ValueError("cv_strategy must be loso or lnskto.")

    fold_rows: list[dict] = []
    for fold_result in _cv_fold_records(cv_results):
        row = dict(fold_result)
        for log_key in (
            "prediction_log",
            "window_prediction_log",
            "trial_prediction_log",
            "variational_interval_log",
            "window_variational_interval_log",
            "trial_variational_interval_log",
            "prediction_diagnostics_log",
        ):
            row.pop(log_key, None)
        test_subjects = row.pop(
            "outer_test_subjects",
            row.pop("left_out_subjects", []),
        )
        row["outer_test_subjects"] = ",".join(map(str, test_subjects))
        if "validation_subjects" in row:
            row["validation_subjects"] = ",".join(
                map(str, row["validation_subjects"])
            )
        if "held_out_trials" in row:
            row["held_out_trials"] = json.dumps(
                row["held_out_trials"],
                sort_keys=True,
                default=_json_default,
            )
        row["inner_fold_results"] = json.dumps(
            row.get("inner_fold_results", []),
            default=_json_default,
        )
        fold_rows.append(row)

    prefix = "loso" if config.cv_strategy == "loso" else "lnskto"
    _write_json(run_dir / "cv_results.json", cv_results)
    _write_csv(run_dir / "cv_folds.csv", fold_rows)
    _write_json(run_dir / f"{prefix}_cv_results.json", cv_results)
    _write_csv(run_dir / f"{prefix}_cv_folds.csv", fold_rows)
    _write_csv(run_dir / "grid_search_summary.csv", _grid_summary_rows(cv_results))
    _write_csv(
        run_dir / "prediction_diagnostics.csv",
        cv_results.get("prediction_diagnostics_log", []),
    )
    _write_csv(
        run_dir / "window_predictions.csv",
        cv_results.get("window_prediction_log", []),
    )
    _write_csv(
        run_dir / "trial_predictions.csv",
        cv_results.get("trial_prediction_log", []),
    )

    if "best_config" not in cv_results:
        raise RuntimeError("Cross-validation did not return best_config.")
    selected_config = dict(cv_results["best_config"])
    _write_json(run_dir / "selected_config.json", selected_config)

    fold_thresholds = [
        float(row["decision_threshold"])
        for row in _cv_fold_records(cv_results)
        if row.get("decision_threshold") is not None
    ]
    final_threshold = float(
        np.median(fold_thresholds)
        if fold_thresholds
        else config.decision_thresholds[0]
    )
    _write_json(
        run_dir / "decision_threshold_summary.json",
        {
            "fold_selected_thresholds": fold_thresholds,
            "final_median_threshold": final_threshold,
            "selection_metric": config.threshold_selection_metric,
            "selection_level": config.threshold_selection_level,
            "candidate_thresholds": list(config.decision_thresholds),
        },
    )
    logger.info("Final inference threshold from fold median: %.6f", final_threshold)

    configured_epoch_cap = int(
        selected_config.get("epochs", config.cv_max_epochs)
    )
    if config.final_epochs is not None:
        final_epochs = max(1, int(config.final_epochs))
        cv_best_epochs: list[int] = []
    elif config.use_early_stopping:
        final_epochs, cv_best_epochs = _select_final_epochs_from_cv(
            cv_results,
            config.final_epoch_strategy,
            configured_epoch_cap,
        )
    else:
        final_epochs = max(1, configured_epoch_cap)
        cv_best_epochs = []
    final_batch_size = int(selected_config.get("batch_size", config.batch_size))
    final_hparams = {
        key: value
        for key, value in selected_config.items()
        if key not in {"epochs", "batch_size"}
    }
    logger.info("Selected final config: %s", selected_config)
    logger.info("Selected final epochs: %d", final_epochs)

    no_validation_loso_results: dict | None = None
    if config.run_no_validation_loso_before_final:
        logger.info(
            "Running fixed-config LOSOCV with no validation: epochs=%d, "
            "batch_size=%d, threshold=%.6f",
            final_epochs,
            final_batch_size,
            final_threshold,
        )
        no_validation_loso_results = fixed_loso_cv(
            model_builder_function=model_builder_function,
            feature_array=feature_array,
            label_array=label_array,
            subject_id_array=subject_id_array,
            trial_id_array=trial_id_array,
            fixed_config=final_hparams,
            n_epochs=final_epochs,
            batch_size=final_batch_size,
            evaluation_level="trial",
            selection_level=config.selection_level,
            # This stage evaluates one already-selected configuration, so the
            # ranking field is informational only. Balanced accuracy is always
            # available and keeps the diagnostic focused on class balance.
            selection_metric="balanced_accuracy",
            maximize_metric=True,
            metrics=(
                "accuracy",
                "f1",
                "precision",
                "recall",
                "macro_f1",
                "macro_precision",
                "macro_recall",
                "balanced_accuracy",
            ),
            log_predictions=True,
            n_prediction_latent_samples=config.prediction_latent_samples,
            latent_sampling_seed=config.latent_sampling_seed,
            decision_threshold=final_threshold,
            prediction_diagnostics=config.prediction_diagnostics,
            prediction_diagnostics_every_n_epochs=(
                config.prediction_diagnostics_every_n_epochs
            ),
            prediction_diagnostics_max_samples=(
                config.prediction_diagnostics_max_samples
            ),
            prediction_diagnostics_threshold_tolerance=(
                config.prediction_diagnostics_threshold_tolerance
            ),
            prediction_diagnostics_seed=config.prediction_diagnostics_seed,
            verbose=config.outer_verbose,
            extra_fit_kwargs={
                "callbacks": [tf.keras.callbacks.TerminateOnNaN()]
            },
            n_jobs=config.n_jobs,
            cpus_per_worker=config.cpus_per_worker,
            max_folds=config.max_folds,
            alternate_subject_sets=config.alternate_subject_sets,
            alternating_subject_seed=config.alternating_subject_seed,
            use_mldg=config.use_mldg,
            mldg_meta_train_subjects=config.mldg_meta_train_subjects,
            mldg_meta_test_subjects=config.mldg_meta_test_subjects,
            mldg_samples_per_subject=config.mldg_samples_per_subject,
            mldg_seed=config.mldg_seed,
        )

        no_validation_fold_rows: list[dict] = []
        for fold_result in _cv_fold_records(no_validation_loso_results):
            row = dict(fold_result)
            test_subjects = row.pop("left_out_subjects", [])
            row["outer_test_subjects"] = ",".join(map(str, test_subjects))
            row["validation_subjects"] = ",".join(
                map(str, row.get("validation_subjects", []))
            )
            row["held_out_trials"] = json.dumps(
                row.get("held_out_trials", []),
                sort_keys=True,
                default=_json_default,
            )
            no_validation_fold_rows.append(row)

        _write_json(
            run_dir / "no_validation_loso_results.json",
            no_validation_loso_results,
        )
        _write_csv(
            run_dir / "no_validation_loso_folds.csv",
            no_validation_fold_rows,
        )
        _write_csv(
            run_dir / "no_validation_loso_grid_summary.csv",
            _grid_summary_rows(no_validation_loso_results),
        )
        _write_csv(
            run_dir / "no_validation_loso_prediction_diagnostics.csv",
            no_validation_loso_results.get("prediction_diagnostics_log", []),
        )
        _write_csv(
            run_dir / "no_validation_loso_window_predictions.csv",
            no_validation_loso_results.get("window_prediction_log", []),
        )
        _write_csv(
            run_dir / "no_validation_loso_trial_predictions.csv",
            no_validation_loso_results.get("trial_prediction_log", []),
        )

    final_model = model_builder_function(**final_hparams)
    final_callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.TerminateOnNaN()
    ]
    diagnostics_callback: PredictionDiagnostics | None = None
    if config.prediction_diagnostics:
        diagnostics_callback = PredictionDiagnostics(
            X_train=feature_array,
            y_train=label_array,
            fold_number=None,
            batch_size=final_batch_size,
            every_n_epochs=config.prediction_diagnostics_every_n_epochs,
            max_samples=config.prediction_diagnostics_max_samples,
            threshold_tolerance=config.prediction_diagnostics_threshold_tolerance,
            seed=config.prediction_diagnostics_seed,
        )
        final_callbacks.append(diagnostics_callback)
    if config.save_final_history_csv:
        final_callbacks.insert(
            0,
            tf.keras.callbacks.CSVLogger(
                str(run_dir / "final_training_history.csv")
            ),
        )

    final_class_weight = None
    if config.use_class_weight:
        classes, counts = np.unique(_as_class_ids(label_array), return_counts=True)
        final_class_weight = {
            int(class_id): len(label_array) / (len(classes) * count)
            for class_id, count in zip(classes, counts)
        }
        logger.info("Final-fit class weights: %s", final_class_weight)
    else:
        logger.info(
            "Final-fit class weighting disabled by --no-class-weight."
        )

    if config.use_mldg:
        final_mldg_sequence = MetaLearningSubjectSequence(
            X=feature_array,
            y=label_array,
            subject_ids=subject_id_array,
            model=final_model,
            meta_train_subjects=config.mldg_meta_train_subjects,
            meta_test_subjects=config.mldg_meta_test_subjects,
            samples_per_subject=config.mldg_samples_per_subject,
            class_weight=final_class_weight,
            seed=config.mldg_seed,
        )
        logger.info(
            "Final MLDG fit uses %d episodes per epoch.",
            len(final_mldg_sequence),
        )
        final_history = final_model.fit(
            final_mldg_sequence,
            epochs=final_epochs,
            verbose=config.final_verbose,
            callbacks=final_callbacks,
        )
    else:
        final_fit_inputs = (
            final_model.prepare_fit_inputs(feature_array, subject_id_array)
            if getattr(final_model, "requires_subject_ids", False)
            else feature_array
        )
        final_fit_kwargs = {
            "epochs": final_epochs,
            "batch_size": final_batch_size,
            "verbose": config.final_verbose,
            "callbacks": final_callbacks,
        }
        # Do not even pass class_weight when weighting is disabled.
        if config.use_class_weight:
            final_fit_kwargs["class_weight"] = final_class_weight
        final_history = final_model.fit(
            final_fit_inputs,
            label_array,
            **final_fit_kwargs,
        )

    if diagnostics_callback is not None:
        _write_csv(
            run_dir / "final_prediction_diagnostics.csv",
            diagnostics_callback.history,
        )

    final_eval = final_model.evaluate(
        feature_array,
        label_array,
        verbose=0,
        return_dict=True,
    )
    if config.save_weights:
        final_model.save_weights(run_dir / "final_model.weights.h5")
    if config.save_full_model:
        final_model.save(run_dir / "final_model.keras")
    if config.save_adjacency_matrices:
        _save_adjacency_matrices(final_model, run_dir)

    final_summary = {
        "run_dir": str(run_dir),
        "architecture": "parallel_bilstm_gcn_fused_vae_dual_path_decoder",
        "n_channels": config.n_channels,
        "n_bands": config.n_bands,
        "selected_final_config": selected_config,
        "selected_final_epochs": final_epochs,
        "selected_final_batch_size": final_batch_size,
        "cv_best_epochs": cv_best_epochs,
        "final_epoch_strategy": config.final_epoch_strategy,
        "classification_level": "window",
        "primary_evaluation_level": "trial",
        "metric_convention": "mtlfusenet_binary_class_1_positive",
        "selection_level": config.selection_level,
        "default_classifier_head": config.classifier_head,
        "classification_steps_per_batch": config.classification_steps_per_batch,
        "vae_steps_per_batch": config.vae_steps_per_batch,
        "classification_learning_rate": config.classification_learning_rate,
        "vae_learning_rate": config.vae_learning_rate,
        "use_mldg": config.use_mldg,
        "mldg_inner_learning_rate": config.mldg_inner_learning_rate,
        "mldg_meta_test_weight": config.mldg_meta_test_weight,
        "mldg_meta_train_subjects": config.mldg_meta_train_subjects,
        "mldg_meta_test_subjects": config.mldg_meta_test_subjects,
        "mldg_samples_per_subject": config.mldg_samples_per_subject,
        "mldg_sampling": "natural_within_subject",
        "mldg_seed": config.mldg_seed,
        "use_class_weight": config.use_class_weight,
        "final_class_weight": final_class_weight,
        "focal_gamma": config.focal_gamma,
        "focal_alpha": config.focal_alpha,
        "use_subject_adversarial": config.use_subject_adversarial,
        "use_supcon": config.use_supcon,
        "label_threshold_mode": config.label_threshold_mode,
        "decision_threshold_candidates": list(config.decision_thresholds),
        "cv_selected_decision_thresholds": fold_thresholds,
        "final_decision_threshold": final_threshold,
        "threshold_selection_metric": config.threshold_selection_metric,
        "threshold_selection_level": config.threshold_selection_level,
        "cv_strategy": config.cv_strategy,
        "cv_results": cv_results,
        "no_validation_loso_results": no_validation_loso_results,
        "final_fit_history": final_history.history,
        "final_full_dataset_metrics": final_eval,
    }
    _write_json(run_dir / "training_summary.json", final_summary)
    logger.info("Final full-data metrics: %s", final_eval)
    logger.info("Saved STS artifacts to %s", run_dir)
    return final_summary



def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    hyperparameters = (
        json.loads(args.hyperparameters_json)
        if args.hyperparameters_json
        else {}
    )
    if args.label_smoothing_levels is not None:
        if "label_smoothing" in hyperparameters:
            raise ValueError(
                "Use either --label-smoothing-levels or label_smoothing in the "
                "hyperparameter JSON, not both."
            )
        hyperparameters["label_smoothing"] = list(args.label_smoothing_levels)
    _validate_args(args, hyperparameters)

    validation_seed = (
        args.validation_seed
        if args.validation_seed is not None
        else (args.seed if args.seed is not None else 42)
    )
    config = JointSTSTrainingConfig(
        output_dir=Path(args.out_dir),
        run_name=args.run_name,
        dataset=args.dataset,
        n_channels=args.n_channels,
        n_bands=args.n_bands,
        optimizer_name=args.optimizer,
        classification_learning_rate=args.classification_learning_rate,
        vae_learning_rate=args.vae_learning_rate,
        discriminator_learning_rate=args.discriminator_learning_rate,
        weight_decay=args.weight_decay,
        classification_steps_per_batch=args.classification_steps_per_batch,
        vae_steps_per_batch=args.vae_steps_per_batch,
        batch_size=args.batch_size,
        cv_max_epochs=args.epochs,
        cv_strategy=args.cv_strategy,
        lnskto_subjects=args.lnskto_subjects,
        lnskto_trials=args.lnskto_trials,
        lnskto_split_seed=args.lnskto_split_seed,
        lnskto_require_all_classes_in_test=args.lnskto_require_all_classes,
        final_epoch_strategy=args.final_epoch_strategy,
        final_epochs=args.final_epochs,
        run_no_validation_loso_before_final=(
            args.run_no_validation_loso_before_final
        ),
        selection_metric=args.selection_metric,
        selection_level=args.selection_level,
        max_folds=args.max_folds,
        n_jobs=args.n_jobs,
        cpus_per_worker=args.cpus_per_worker,
        alternate_subject_sets=args.alternate_subject_sets,
        alternating_subject_seed=args.alternating_subject_seed,
        use_mldg=args.use_mldg,
        mldg_inner_learning_rate=args.mldg_inner_learning_rate,
        mldg_meta_test_weight=args.mldg_meta_test_weight,
        mldg_meta_train_subjects=args.mldg_meta_train_subjects,
        mldg_meta_test_subjects=args.mldg_meta_test_subjects,
        mldg_samples_per_subject=args.mldg_samples_per_subject,
        mldg_seed=args.mldg_seed,
        prediction_latent_samples=args.prediction_latent_samples,
        latent_sampling_seed=args.latent_sampling_seed,
        decision_thresholds=tuple(sorted(map(float, args.decision_thresholds))),
        threshold_selection_metric=args.threshold_selection_metric,
        threshold_selection_level=args.threshold_selection_level,
        prediction_diagnostics=not args.no_prediction_diagnostics,
        prediction_diagnostics_every_n_epochs=args.prediction_diagnostics_every,
        prediction_diagnostics_max_samples=args.prediction_diagnostics_samples,
        prediction_diagnostics_threshold_tolerance=args.prediction_threshold_tolerance,
        prediction_diagnostics_seed=args.prediction_diagnostics_seed,
        validation_subjects_per_fold=args.validation_subjects,
        validation_seed=validation_seed,
        outer_verbose=args.outer_verbose,
        final_verbose=args.final_verbose,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        early_stopping_monitor=args.early_stopping_monitor,
        early_stopping_mode=args.early_stopping_mode,
        use_early_stopping=not args.no_early_stopping,
        save_full_model=not args.no_save_full_model,
        save_weights=not args.no_save_weights,
        save_final_history_csv=not args.no_save_final_history_csv,
        save_adjacency_matrices=not args.no_save_adjacency_matrices,
        seed=args.seed,
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
        classification_hidden_units=args.classification_hidden_units,
        classification_dropout=args.classification_dropout,
        classifier_head=args.classifier_head,
        label_smoothing=args.label_smoothing,
        focal_gamma=args.focal_gamma,
        focal_alpha=(
            None
            if args.focal_alpha is None
            else tuple(float(value) for value in args.focal_alpha)
        ),
        classification_loss_weight=args.classification_loss_weight,
        vae_loss_weight=args.vae_loss_weight,
        vae_beta=args.vae_beta,
        vc_alpha=args.vc_alpha,
        vc_beta=args.vc_beta,
        vc_gamma=args.vc_gamma,
        vc_lambda=args.vc_lambda,
        update_discriminator=args.update_discriminator,
        use_class_weight=args.use_class_weight,
        use_subject_adversarial=args.use_subject_adversarial,
        subject_adversarial_weight=args.subject_adversarial_weight,
        subject_loss_weight=args.subject_loss_weight,
        subject_hidden_units=args.subject_hidden_units,
        subject_dropout=args.subject_dropout,
        subject_latent_mode=args.subject_latent_mode,
        subject_mc_samples=args.subject_mc_samples,
        use_supcon=args.use_supcon,
        supcon_weight=args.supcon_weight,
        supcon_temperature=args.supcon_temperature,
        supcon_cross_subject_only=args.supcon_cross_subject_only,
        label_threshold_mode=args.label_threshold_mode,
        window_normalization=(
            "none" if args.no_zscore else args.window_normalization
        ),
        hyperparameters=hyperparameters,
    )

    feature_array = label_array = subject_id_array = trial_id_array = None
    windowed_paths = (
        args.features_npy,
        args.labels_npy,
        args.subjects_npy,
        args.trials_npy,
    )
    if any(path is not None for path in windowed_paths) and not all(
        path is not None for path in windowed_paths
    ):
        raise ValueError(
            "Pre-windowed input requires --features-npy, --labels-npy, "
            "--subjects-npy, and --trials-npy together."
        )

    if all(path is not None for path in windowed_paths):
        feature_array = _load_numpy_array(args.features_npy)
        label_array = _load_numpy_array(args.labels_npy)
        subject_id_array = _load_numpy_array(args.subjects_npy)
        trial_id_array = _load_numpy_array(args.trials_npy)
        data_loader = None
    else:
        dataset_config = get_dataset_config(args.dataset)
        eeg_path = args.raw_eeg_npy or dataset_config.eeg_path
        labels_path = args.raw_labels_npy or dataset_config.labels_path
        data_loader = lambda: load_joint_sts_training_data(
            eeg_path=eeg_path,
            labels_path=labels_path,
            label_dimension=args.label_dimension,
            window_size_sec=args.window_sec,
            fs=args.fs,
            overlap=args.window_overlap,
            median_label=args.median_label,
            window_normalization=(
                "none" if args.no_zscore else args.window_normalization
            ),
            label_threshold_mode=args.label_threshold_mode,
            dataset=dataset_config,
        )

    train_joint_sts_model(
        feature_array=feature_array,
        label_array=label_array,
        subject_id_array=subject_id_array,
        trial_id_array=trial_id_array,
        data_loader=data_loader,
        training_config=config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
