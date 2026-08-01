"""Training entry point for joint VAE classification with selectable CV.

Window mode preserves one prediction per EEG window. Trial mode groups ordered
windows by ``(subject_id, trial_id)``, pads sessions only after all available
windows have been retained, reconstructs valid windows independently, and
trains a BiLSTM across the complete session window sequence to emit one label.

The loader supports window-local channel-band z-scoring and an explicit
subject-median label mode. Cross-validation can use strict LOSO or hold out K
complete trials from each of N selected subjects without splitting a trial
across train and test.
"""

from __future__ import annotations

# Joint-loss variant: defaults to early stopping on the complete validation
# VAE+VC objective and selects the global flat-LOSO configuration by mean
# held-out joint loss.

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
import argparse
import csv
import json
import logging
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf

try:
    from .joint_v2_autoencoder_vc import JointAutoencoderVariationalClassifierV2
    from .joint_v2_data import (
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

    from joint_v2_autoencoder_vc import JointAutoencoderVariationalClassifierV2
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
        from ..cross_val import PredictionDiagnostics, lnskto_cv, loso_cv
    except ImportError:
        # Supports replacing the repository's ordinary cross_val.py with this
        # variant while retaining the standard module name.
        from ..cross_val import PredictionDiagnostics, lnskto_cv, loso_cv
    from ..supervised.rnn_architectures import BiLSTMClassifier
    from ..supervised.variational_classifier import (
        DenseClassifier,
        HybridClassifier,
        VariationalClassifier,
    )
    from ..unsupervised.Convolutions.CNN1D import CNN1DDecoder, CNN1DEncoder
    from ..unsupervised.Convolutions.CNN2D import CNN2DDecoder, CNN2DEncoder
    from ..unsupervised.Convolutions.GCN import GCNDecoder, GCNEncoder
except ImportError:
    SRC_ROOT = Path(__file__).resolve().parents[3]
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

    try:
        from eegproc.deep_learning.cross_val import PredictionDiagnostics, lnskto_cv, loso_cv
    except ImportError:
        from eegproc.deep_learning.cross_val import PredictionDiagnostics, lnskto_cv, loso_cv
    from eegproc.deep_learning.supervised.rnn_architectures import (
        BiLSTMClassifier,
    )
    from eegproc.deep_learning.supervised.variational_classifier import (
        DenseClassifier,
        HybridClassifier,
        VariationalClassifier,
    )
    from eegproc.deep_learning.unsupervised.Convolutions.CNN1D import (
        CNN1DDecoder,
        CNN1DEncoder,
    )
    from eegproc.deep_learning.unsupervised.Convolutions.CNN2D import (
        CNN2DDecoder,
        CNN2DEncoder,
    )
    from eegproc.deep_learning.unsupervised.Convolutions.GCN import (
        GCNDecoder,
        GCNEncoder,
    )


@dataclass(slots=True)
class JointV2TrainingConfig:
    """Tunable settings for the joint-model training pipeline."""

    output_dir: Path = Path("runs") / "joint_autoencoder_vc_v2"
    run_name: str = "joint_autoencoder_vc_v2"
    dataset: str = "dreamer"
    encoder_type: str = "cnn1d"
    n_channels: int = 14
    n_bands: int | None = None
    learning_rate: float = 1e-4
    optimizer_name: str = "adamw"
    weight_decay: float = 1e-4
    batch_size: int = 64
    cv_max_epochs: int = 50
    cv_strategy: str = "loso"
    lnskto_subjects: int = 3
    lnskto_trials: int = 3
    lnskto_split_seed: int | None = 42
    lnskto_require_all_classes_in_test: bool = True
    final_epoch_strategy: str = "median"
    final_epochs: int | None = None
    classification_level: str = "window"
    selection_metric: str = "f1"
    selection_level: str = "window"
    trial_max_windows: int | None = None
    trial_crop: str = "center"
    maximize_metric: bool | None = None
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
    validation_subjects_per_fold: int = 2
    validation_seed: int | None = 42
    outer_verbose: int = 0
    final_verbose: int = 1
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0
    early_stopping_monitor: str = "val_accuracy"
    early_stopping_mode: str = "max"
    use_early_stopping: bool = True
    save_full_model: bool = True
    save_weights: bool = True
    save_final_history_csv: bool = True
    seed: int | None = None
    # In window mode the BiLSTM runs over latent timesteps. In trial mode it
    # runs over the ordered sequence of pooled window embeddings.
    bilstm_units: int = 64
    n_bilstm_layers: int = 1
    bilstm_dropout: float = 0.30
    # Backwards-compatible aliases. Leave as None so bilstm_* remains canonical.
    trial_bilstm_units: int | None = None
    n_trial_bilstm_layers: int | None = None
    trial_bilstm_dropout: float | None = None
    bilstm_kwargs: dict = field(default_factory=dict)
    trial_bilstm_kwargs: dict = field(default_factory=dict)
    encoder_kwargs: dict = field(default_factory=dict)
    decoder_kwargs: dict = field(default_factory=dict)
    classifier_kwargs: dict = field(default_factory=dict)
    classifier_head: str = "variational"
    label_smoothing: float = 0.0
    model_kwargs: dict = field(default_factory=dict)
    hyperparameters: dict = field(default_factory=dict)
    n_jobs: int = 4
    cpus_per_worker: int = 2
    max_folds: int | None = None
    use_class_weight: bool = True
    use_subject_adversarial: bool = False
    subject_adversarial_weight: float = 0.05
    subject_loss_weight: float = 1.0
    subject_hidden_units: int = 64
    subject_dropout: float = 0.0
    subject_latent_mode: str = "mean"
    subject_mc_samples: int = 5
    use_supcon: bool = False
    supcon_weight: float = 0.03
    supcon_temperature: float = 0.1
    supcon_cross_subject_only: bool = True
    label_threshold_mode: str = "global"
    window_normalization: str = "global_rms"


def _flatten_grouped_trials_to_windows(
    feature_array: np.ndarray,
    label_array: np.ndarray,
    subject_id_array: np.ndarray,
    trial_id_array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert legacy rank-4 trial tensors into aligned flat windows."""
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


def _group_windows_by_subject_trial(
    feature_array: np.ndarray,
    label_array: np.ndarray,
    subject_id_array: np.ndarray,
    trial_id_array: np.ndarray,
    max_windows: int | None = None,
    crop: str = "center",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Group ordered flat windows into zero-padded trial tensors.

    The first occurrence order of each ``(subject_id, trial_id)`` key is
    preserved, as is the original within-trial window order. When
    ``max_windows`` is ``None``, every available window is retained and trials
    are padded to the longest session. A positive cap crops longer trials and
    is useful when GPU memory cannot accommodate the complete session.

    Returns ``(X_trial, y_trial, subject_trial, trial_trial, lengths)``.
    """
    features = np.asarray(feature_array, dtype=np.float32)
    labels = np.asarray(label_array)
    subjects = np.asarray(subject_id_array).reshape(-1)
    trials = np.asarray(trial_id_array).reshape(-1)
    crop = str(crop).lower()
    if crop not in {"start", "center", "end"}:
        raise ValueError("trial crop must be start, center, or end.")
    if max_windows is not None and int(max_windows) < 1:
        raise ValueError("trial_max_windows must be >= 1 when supplied.")

    if features.ndim == 4:
        if not (len(features) == len(labels) == len(subjects) == len(trials)):
            raise ValueError("Rank-4 trial arrays must align on axis 0.")
        lengths = np.sum(
            np.any(features != 0.0, axis=(2, 3)),
            axis=1,
            dtype=np.int64,
        )
        if np.any(lengths < 1):
            raise ValueError("Every grouped trial must contain a valid window.")
        if max_windows is None or features.shape[1] <= int(max_windows):
            return features, labels, subjects, trials, lengths

        target = int(max_windows)
        cropped = np.zeros(
            (len(features), target, features.shape[2], features.shape[3]),
            dtype=np.float32,
        )
        output_lengths = np.minimum(lengths, target)
        for index, length in enumerate(lengths.tolist()):
            if crop == "start":
                start = 0
            elif crop == "end":
                start = max(0, length - target)
            else:
                start = max(0, (length - target) // 2)
            cropped[index, : output_lengths[index]] = features[
                index,
                start : start + output_lengths[index],
            ]
        return cropped, labels, subjects, trials, output_lengths

    if features.ndim != 3:
        raise ValueError(
            "Trial grouping expects rank-3 windows or rank-4 trials; "
            f"got {features.shape}."
        )
    if not (len(features) == len(labels) == len(subjects) == len(trials)):
        raise ValueError("Window arrays must align before trial grouping.")

    class_ids = _as_class_ids(labels)
    grouped: dict[tuple, list[int]] = {}
    for index, (subject_id, trial_id) in enumerate(zip(subjects, trials)):
        key = (
            subject_id.item() if isinstance(subject_id, np.generic) else subject_id,
            trial_id.item() if isinstance(trial_id, np.generic) else trial_id,
        )
        grouped.setdefault(key, []).append(index)

    if not grouped:
        raise ValueError("No subject-trial groups were found.")

    raw_lengths = np.asarray(
        [len(indices) for indices in grouped.values()],
        dtype=np.int64,
    )
    target_windows = (
        int(np.max(raw_lengths))
        if max_windows is None
        else int(max_windows)
    )
    timesteps, n_features = features.shape[1:]
    grouped_features = np.zeros(
        (len(grouped), target_windows, timesteps, n_features),
        dtype=np.float32,
    )
    grouped_labels = np.empty(len(grouped), dtype=np.int64)
    grouped_subjects = np.empty(len(grouped), dtype=subjects.dtype)
    grouped_trials = np.empty(len(grouped), dtype=trials.dtype)
    grouped_lengths = np.empty(len(grouped), dtype=np.int64)

    for output_index, ((subject_id, trial_id), indices) in enumerate(grouped.items()):
        indices_array = np.asarray(indices, dtype=np.int64)
        unique_labels = np.unique(class_ids[indices_array])
        if len(unique_labels) != 1:
            raise ValueError(
                "All windows in one subject-trial session must share one label; "
                f"subject={subject_id!r}, trial={trial_id!r}, "
                f"labels={unique_labels.tolist()}."
            )
        length = len(indices_array)
        kept = min(length, target_windows)
        if crop == "start":
            start = 0
        elif crop == "end":
            start = max(0, length - target_windows)
        else:
            start = max(0, (length - target_windows) // 2)
        selected = indices_array[start : start + kept]
        grouped_features[output_index, :kept] = features[selected]
        grouped_labels[output_index] = int(unique_labels[0])
        grouped_subjects[output_index] = subject_id
        grouped_trials[output_index] = trial_id
        grouped_lengths[output_index] = kept

    return (
        grouped_features,
        grouped_labels,
        grouped_subjects,
        grouped_trials,
        grouped_lengths,
    )


def _normalize_each_window(
    feature_array: np.ndarray,
    mode: str = "global_rms",
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Normalize windows without destroying channel-band power structure.

    ``global_rms`` divides each complete window by one scalar RMS value. This
    removes gross recording-gain differences while preserving relative power
    across electrodes and frequency bands. ``feature_zscore`` is retained only
    as an explicit ablation because it forces every channel-band stream to unit
    variance and can erase the amplitude information used for EEG emotion
    recognition. ``none`` leaves the filtered waveforms unchanged.
    """
    features = np.asarray(feature_array, dtype=np.float32)
    if features.ndim != 3:
        raise ValueError(
            "Window normalization expects (n_windows, timesteps, n_features); "
            f"got {features.shape}."
        )
    mode = str(mode).lower()
    if mode == "none":
        return features
    if mode == "global_rms":
        rms = np.sqrt(
            np.mean(np.square(features, dtype=np.float64), axis=(1, 2), keepdims=True)
        )
        normalized = features.astype(np.float64) / np.maximum(rms, float(epsilon))
    elif mode == "feature_zscore":
        mean = np.mean(features, axis=1, keepdims=True, dtype=np.float64)
        std = np.std(features, axis=1, keepdims=True, dtype=np.float64)
        normalized = (features.astype(np.float64) - mean) / np.maximum(
            std, float(epsilon)
        )
    else:
        raise ValueError(
            "window normalization must be one of: none, global_rms, "
            f"feature_zscore; got {mode!r}."
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
    """Create subject-relative binary labels and repeat them over windows.

    This changes the target from an absolute DREAMER threshold to whether a
    trial is at or above that subject's own median rating. It mitigates rating
    scale usage differences but must be reported as a separate label protocol.
    """
    raw_labels = np.load(Path(labels_path), allow_pickle=False)
    if raw_labels.ndim != 3 or raw_labels.shape[-1] < 2:
        raise ValueError(
            "subject_median label mode requires raw labels shaped "
            "(subjects, trials, >=2) with [valence, arousal]."
        )
    dimension_index = {"valence": 0, "arousal": 1}[label_dimension]
    subjects = np.asarray(subject_id_array).reshape(-1)
    trials = np.asarray(trial_id_array).reshape(-1)
    output = np.empty(len(subjects), dtype=np.int64)
    unique_subjects = np.unique(subjects)
    if len(unique_subjects) != raw_labels.shape[0]:
        raise ValueError(
            "Raw-label subject count does not match the windowed subject IDs: "
            f"{raw_labels.shape[0]} versus {len(unique_subjects)}."
        )

    for subject_row, subject_id in enumerate(sorted(unique_subjects.tolist())):
        subject_mask = subjects == subject_id
        unique_trials = sorted(np.unique(trials[subject_mask]).tolist())
        if len(unique_trials) != raw_labels.shape[1]:
            raise ValueError(
                f"Subject {subject_id!r} has {len(unique_trials)} trial IDs, but "
                f"the raw label file has {raw_labels.shape[1]} trials."
            )
        ratings = raw_labels[subject_row, :, dimension_index].astype(np.float64)
        threshold = float(np.median(ratings))
        binary_by_trial = (ratings >= threshold).astype(np.int64)
        trial_to_label = {
            trial_id: int(binary_by_trial[trial_row])
            for trial_row, trial_id in enumerate(unique_trials)
        }
        output[subject_mask] = np.asarray(
            [trial_to_label[trial_id] for trial_id in trials[subject_mask]],
            dtype=np.int64,
        )
    return output


def load_joint_v2_training_data(
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
    """Load flat window samples for subject-disjoint LOSO classification.

    Returns ``(features, labels, subject_ids, trial_ids)`` with features shaped
    ``(n_windows, window_timesteps, n_features)``. Upstream normalization in
    ``build_joint_v2_dataset`` is disabled. Optional final-window normalization
    uses one global RMS scale per window by default, preserving channel-band
    power ratios.
    """
    window_size = int(round(window_size_sec * fs))
    if window_size <= 0:
        raise ValueError("window_size_sec * fs must produce a positive size.")
    if not (0.0 <= overlap < 1.0):
        raise ValueError(f"overlap must be in [0, 1), got {overlap}.")
    if window_normalization not in {"none", "global_rms", "feature_zscore"}:
        raise ValueError(
            "window_normalization must be none, global_rms, or feature_zscore."
        )
    if label_threshold_mode not in {"global", "subject_median"}:
        raise ValueError(
            "label_threshold_mode must be 'global' or 'subject_median'."
        )

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
        feature_array, label_array, subject_id_array, trial_id_array = dataset_arrays
    elif len(dataset_arrays) == 3:
        feature_array, label_array, subject_id_array = dataset_arrays
        raw_eeg = np.load(Path(eeg_path), mmap_mode="r", allow_pickle=False)
        if raw_eeg.ndim != 4:
            raise ValueError(
                "Expected raw EEG shaped (subjects, trials, channels, samples); "
                f"got {raw_eeg.shape}."
            )
        n_subjects, n_trials, _n_channels, n_samples = raw_eeg.shape
        hop = max(1, int(round(window_size * (1.0 - overlap))))
        n_windows_per_trial = 1 + (n_samples - window_size) // hop
        trial_id_array = np.tile(
            np.repeat(np.arange(n_trials, dtype=np.int64), n_windows_per_trial),
            n_subjects,
        )
    else:
        raise ValueError(
            "build_joint_v2_dataset must return three or four aligned arrays."
        )

    feature_array, label_array, subject_id_array, trial_id_array = (
        _flatten_grouped_trials_to_windows(
            feature_array,
            label_array,
            subject_id_array,
            trial_id_array,
        )
    )
    if feature_array.ndim != 3:
        raise ValueError(
            "Window-level training requires features shaped "
            f"(n_windows, timesteps, features); got {feature_array.shape}."
        )
    lengths = tuple(
        len(array)
        for array in (
            feature_array,
            label_array,
            subject_id_array,
            trial_id_array,
        )
    )
    if len(set(lengths)) != 1:
        raise ValueError(f"Window arrays are not aligned: {lengths}.")

    if label_threshold_mode == "subject_median":
        label_array = _subject_median_window_labels(
            labels_path=labels_path,
            label_dimension=label_dimension,
            subject_id_array=subject_id_array,
            trial_id_array=trial_id_array,
        )
    feature_array = _normalize_each_window(
        feature_array, mode=window_normalization
    )

    return (
        np.asarray(feature_array, dtype=np.float32),
        np.asarray(label_array),
        np.asarray(subject_id_array),
        np.asarray(trial_id_array),
    )

def _load_numpy_array(path: str | Path) -> np.ndarray:
    return np.load(Path(path), allow_pickle=False)


def _ensure_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _infer_n_classes(label_array: np.ndarray) -> int:
    labels = np.asarray(label_array)

    if labels.ndim == 2 and labels.shape[1] > 1:
        return int(labels.shape[1])

    flattened = labels.reshape(-1)
    if flattened.size == 0:
        raise ValueError("label_array must not be empty.")

    return int(np.max(flattened)) + 1


def _as_class_ids(label_array: np.ndarray) -> np.ndarray:
    """Convert sparse, column-vector, or one-hot labels to integer IDs."""
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


def _configure_run_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"eegproc.joint_v2.{run_dir.name}")
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


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    fieldnames = list(
        dict.fromkeys(
            key
            for row in rows
            for key in row
        )
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



def _grid_summary_rows(cv_results: dict) -> list[dict]:
    """Flatten configuration-level LOSO results for a compact CSV summary."""
    rows: list[dict] = []
    for config_result in cv_results.get("config_results", []):
        row = {
            "config_index": int(config_result["config_index"]),
            "is_selected": int(
                config_result["config_index"] == cv_results.get("best_config_index")
            ),
            "selection_level": config_result.get("selection_level"),
            "selection_metric": config_result.get("selection_metric"),
            "selection_score": config_result.get("selection_score"),
            "selection_score_std": config_result.get("selection_score_std"),
            "n_folds": config_result.get("n_folds"),
            "config": json.dumps(
                config_result.get("config", {}),
                sort_keys=True,
                default=_json_default,
            ),
        }

        for prefix in ("window", "trial"):
            mean_scores = config_result.get(f"{prefix}_mean_scores", {})
            std_scores = config_result.get(f"{prefix}_std_scores", {})
            for metric_name, metric_value in mean_scores.items():
                row[f"{prefix}_{metric_name}_mean"] = metric_value
            for metric_name, metric_value in std_scores.items():
                row[f"{prefix}_{metric_name}_std"] = metric_value

        rows.append(row)
    return rows


def _select_final_epochs_from_cv(
    cv_results: dict,
    strategy: str,
    fallback_epochs: int,
) -> tuple[int, list[int]]:
    """Choose a full-data epoch count from fold-local best validation epochs."""
    fold_rows = cv_results.get("best_config_result", {}).get("fold_results", [])
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
        raise ValueError(
            f"Unknown final_epoch_strategy={strategy!r}; expected median, mean, or max."
        )

    return max(1, selected), best_epochs


def _positive_int_tuple(
    name: str,
    value,
    *,
    allow_empty: bool = False,
) -> tuple[int, ...]:
    """Normalize a sequence of positive integer layer settings."""
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple, got {value!r}.")
    if not value and not allow_empty:
        raise ValueError(f"{name} must be non-empty, got {value!r}.")
    normalized = tuple(int(item) for item in value)
    if any(item < 1 for item in normalized):
        raise ValueError(f"{name} values must all be >= 1, got {normalized!r}.")
    return normalized


def _nonnegative_int_tuple(name: str, value) -> tuple[int, ...]:
    """Normalize a sequence of non-negative integer layer indices."""
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple, got {value!r}.")
    normalized = tuple(int(item) for item in value)
    if any(item < 0 for item in normalized):
        raise ValueError(f"{name} values must all be >= 0, got {normalized!r}.")
    return normalized


def _normalize_common_encoder_configuration(encoder_config: dict) -> dict:
    """Normalize settings shared by all supported encoder families."""
    config = dict(encoder_config)
    config["t_down"] = int(config["t_down"])
    config["emb_dim"] = int(config["emb_dim"])
    config["dropout"] = float(config["dropout"])
    config["use_batch_norm"] = bool(config["use_batch_norm"])
    config["activation"] = str(config.get("activation", "relu"))

    if config["t_down"] < 1:
        raise ValueError(f"t_down must be >= 1, got {config['t_down']}.")
    if config["emb_dim"] < 1:
        raise ValueError(f"emb_dim must be >= 1, got {config['emb_dim']}.")
    if not 0.0 <= config["dropout"] < 1.0:
        raise ValueError(
            f"Encoder dropout must be in [0, 1), got {config['dropout']}."
        )

    return config


def _normalize_cnn1d_configuration(encoder_config: dict) -> dict:
    """Validate and normalize one CNN1D encoder configuration."""
    config = _normalize_common_encoder_configuration(encoder_config)
    config.pop("activation", None)  # CNN1DEncoder currently owns its activations.
    config["conv_filters"] = _positive_int_tuple(
        "conv_filters", config["conv_filters"]
    )
    config["kernel_sizes"] = _positive_int_tuple(
        "kernel_sizes", config["kernel_sizes"]
    )
    config["pool_after_layers"] = _nonnegative_int_tuple(
        "pool_after_layers", config["pool_after_layers"]
    )
    config["pool_sizes"] = _positive_int_tuple(
        "pool_sizes", config["pool_sizes"], allow_empty=True
    )

    n_conv_layers = len(config["conv_filters"])
    if len(config["kernel_sizes"]) != n_conv_layers:
        raise ValueError(
            "conv_filters and kernel_sizes must describe the same number of "
            f"CNN1D layers. Got {config['conv_filters']!r} and "
            f"{config['kernel_sizes']!r}."
        )
    if len(config["pool_after_layers"]) != len(config["pool_sizes"]):
        raise ValueError(
            "pool_after_layers and pool_sizes must have the same length. Got "
            f"{config['pool_after_layers']!r} and {config['pool_sizes']!r}."
        )
    if len(set(config["pool_after_layers"])) != len(config["pool_after_layers"]):
        raise ValueError(
            "pool_after_layers cannot contain duplicate layer indices: "
            f"{config['pool_after_layers']!r}."
        )
    if any(index >= n_conv_layers for index in config["pool_after_layers"]):
        raise ValueError(
            "pool_after_layers contains an index outside the CNN1D stack of "
            f"length {n_conv_layers}: {config['pool_after_layers']!r}."
        )

    return config


def _normalize_2d_kernel_sizes(value, n_layers: int) -> tuple[tuple[int, int], ...]:
    """Normalize one 2D kernel pair or one pair per Conv2D layer."""
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"kernel_sizes must be a list or tuple, got {value!r}.")

    if len(value) == 2 and all(isinstance(item, (int, np.integer)) for item in value):
        pair = tuple(int(item) for item in value)
        if any(item < 1 for item in pair):
            raise ValueError(f"2D kernel dimensions must be >= 1, got {pair!r}.")
        return tuple(pair for _ in range(n_layers))

    kernels: list[tuple[int, int]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(
                "Each CNN2D kernel must contain exactly two integers. "
                f"Got {item!r} in {value!r}."
            )
        pair = tuple(int(dimension) for dimension in item)
        if any(dimension < 1 for dimension in pair):
            raise ValueError(f"2D kernel dimensions must be >= 1, got {pair!r}.")
        kernels.append(pair)

    if len(kernels) != n_layers:
        raise ValueError(
            f"CNN2D kernel_sizes must contain one pair or {n_layers} pairs; "
            f"got {len(kernels)} pairs."
        )
    return tuple(kernels)


def _normalize_2d_pool_sizes(
    value,
    n_layers: int,
) -> tuple[tuple[int, int] | None, ...]:
    """Normalize optional spatial pooling to one setting per Conv2D layer."""
    if value is None:
        return tuple(None for _ in range(n_layers))
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"spatial_pool_sizes must be a list, tuple, or None; got {value!r}."
        )
    if not value:
        return tuple(None for _ in range(n_layers))

    if len(value) == 2 and all(
        isinstance(item, (int, np.integer)) for item in value
    ):
        pair = tuple(int(item) for item in value)
        if any(item < 1 for item in pair):
            raise ValueError(
                f"Spatial pool dimensions must be >= 1, got {pair!r}."
            )
        return tuple(pair for _ in range(n_layers))

    if len(value) != n_layers:
        raise ValueError(
            "spatial_pool_sizes must contain one pair or one entry per Conv2D "
            f"layer ({n_layers}); got {value!r}."
        )

    normalized: list[tuple[int, int] | None] = []
    for item in value:
        if item is None:
            normalized.append(None)
            continue
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(
                "Each spatial_pool_sizes entry must be None or two integers; "
                f"got {item!r}."
            )
        pair = tuple(int(dimension) for dimension in item)
        if any(dimension < 1 for dimension in pair):
            raise ValueError(
                f"Spatial pool dimensions must be >= 1, got {pair!r}."
            )
        normalized.append(pair)

    return tuple(normalized)


def _validate_temporal_pooling(config: dict) -> dict:
    """Validate temporal pooling shared by CNN2D and GCN encoders."""
    config["temporal_pool_sizes"] = _positive_int_tuple(
        "temporal_pool_sizes", config["temporal_pool_sizes"], allow_empty=True
    )
    effective_t_down = int(np.prod(config["temporal_pool_sizes"], dtype=np.int64))
    if effective_t_down != config["t_down"]:
        raise ValueError(
            f"t_down={config['t_down']}, but temporal_pool_sizes="
            f"{config['temporal_pool_sizes']!r} produces {effective_t_down}."
        )
    return config


def _normalize_cnn2d_configuration(encoder_config: dict) -> dict:
    """Validate and normalize one CNN2D encoder configuration."""
    config = _normalize_common_encoder_configuration(encoder_config)
    config["n_channels"] = int(config["n_channels"])
    config["n_bands"] = int(config["n_bands"])
    config["conv_filters"] = _positive_int_tuple(
        "conv_filters", config["conv_filters"]
    )
    n_conv_layers = len(config["conv_filters"])
    config["kernel_sizes"] = _normalize_2d_kernel_sizes(
        config["kernel_sizes"], n_conv_layers
    )
    config["spatial_pool_sizes"] = _normalize_2d_pool_sizes(
        config.get("spatial_pool_sizes", (2, 2)),
        n_conv_layers,
    )
    if config["n_channels"] < 1 or config["n_bands"] < 1:
        raise ValueError(
            "n_channels and n_bands must both be positive; got "
            f"{config['n_channels']} and {config['n_bands']}."
        )
    return _validate_temporal_pooling(config)


def _normalize_gcn_configuration(encoder_config: dict) -> dict:
    """Validate and normalize one GCN encoder configuration."""
    config = _normalize_common_encoder_configuration(encoder_config)
    config["n_channels"] = int(config["n_channels"])
    config["n_bands"] = int(config["n_bands"])
    config["gcn_units"] = _positive_int_tuple("gcn_units", config["gcn_units"])
    if config["n_channels"] < 1 or config["n_bands"] < 1:
        raise ValueError(
            "n_channels and n_bands must both be positive; got "
            f"{config['n_channels']} and {config['n_bands']}."
        )
    return _validate_temporal_pooling(config)


def _resolve_channel_band_shape(
    n_features: int,
    n_channels: int,
    n_bands: int | None,
) -> tuple[int, int]:
    """Resolve the flattened feature dimension into channels x bands."""
    n_features = int(n_features)
    n_channels = int(n_channels)
    if n_features < 1 or n_channels < 1:
        raise ValueError(
            f"n_features and n_channels must be positive; got {n_features}, "
            f"{n_channels}."
        )

    if n_bands is None:
        if n_features % n_channels != 0:
            raise ValueError(
                f"Cannot infer n_bands: input has {n_features} features, which "
                f"is not divisible by n_channels={n_channels}. Pass --n-bands "
                "and --n-channels explicitly or reshape the input."
            )
        n_bands = n_features // n_channels
    else:
        n_bands = int(n_bands)

    if n_bands < 1 or n_channels * n_bands != n_features:
        raise ValueError(
            "CNN2D/GCN input must satisfy n_features = n_channels * n_bands. "
            f"Got {n_features} != {n_channels} * {n_bands}."
        )
    return n_channels, n_bands



def _build_encoder_decoder(
    encoder_type: str,
    timesteps: int,
    n_features: int,
    n_channels: int,
    n_bands: int | None,
    encoder_kwargs: dict | None,
    decoder_kwargs: dict | None,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Build a matched encoder-decoder pair for the selected architecture."""
    encoder_type = encoder_type.lower()
    supplied = dict(encoder_kwargs or {})
    decoder_kwargs = dict(decoder_kwargs or {})

    if encoder_type == "cnn1d":
        defaults = {
            "timesteps": timesteps,
            "n_features": n_features,
            "t_down": 2,
            "conv_filters": (16, 32),
            "kernel_sizes": (5, 3),
            "pool_after_layers": (0,),
            "pool_sizes": (2,),
            "emb_dim": 16,
            "dropout": 0.1,
            "use_batch_norm": False,
            "activation": "relu",
        }
        defaults.update(supplied)
        encoder = CNN1DEncoder(**_normalize_cnn1d_configuration(defaults))
        decoder = CNN1DDecoder.from_encoder(encoder, **decoder_kwargs)
        return encoder, decoder

    unsupported_decoder_keys = set(decoder_kwargs) - {"name"}
    if unsupported_decoder_keys:
        raise ValueError(
            "CNN2D/GCN decoder_kwargs supports only an optional model name; got "
            f"{sorted(unsupported_decoder_keys)}."
        )

    requested_channels = int(supplied.get("n_channels", n_channels))
    requested_bands = supplied.get("n_bands", n_bands)
    resolved_channels, resolved_bands = _resolve_channel_band_shape(
        n_features=n_features,
        n_channels=requested_channels,
        n_bands=requested_bands,
    )

    if encoder_type == "cnn2d":
        band_kernel = min(3, resolved_bands)
        defaults = {
            "timesteps": timesteps,
            "t_down": 2,
            "n_channels": resolved_channels,
            "n_bands": resolved_bands,
            "conv_filters": (16, 32),
            "kernel_sizes": ((3, band_kernel), (3, band_kernel)),
            # Pool the channel-band grid after each Conv2D block so the second
            # convolution does not retain a full 14 x 4 activation map.
            "spatial_pool_sizes": (2, 2),
            "temporal_pool_sizes": (2,),
            "emb_dim": 16,
            "dropout": 0.1,
            "activation": "relu",
            "use_batch_norm": False,
        }
        defaults.update(supplied)
        encoder = CNN2DEncoder(**_normalize_cnn2d_configuration(defaults))
        decoder = CNN2DDecoder.from_encoder(encoder, **decoder_kwargs)
        return encoder, decoder

    if encoder_type == "gcn":
        defaults = {
            "timesteps": timesteps,
            "t_down": 2,
            "n_channels": resolved_channels,
            "n_bands": resolved_bands,
            "gcn_units": (16, 32),
            "temporal_pool_sizes": (2,),
            "emb_dim": 16,
            "dropout": 0.1,
            "activation": "relu",
            "use_batch_norm": False,
        }
        defaults.update(supplied)
        encoder = GCNEncoder(**_normalize_gcn_configuration(defaults))
        decoder = GCNDecoder.from_encoder(encoder, **decoder_kwargs)
        return encoder, decoder

    raise ValueError(
        f"Unknown encoder_type={encoder_type!r}; expected cnn1d, cnn2d, or gcn."
    )



def _build_optimizer(
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
) -> tf.keras.optimizers.Optimizer:
    """Build the selected Adam-family optimizer.

    AdamW applies decoupled weight decay to trainable parameters. Plain Adam is
    retained as an explicit ablation and ignores ``weight_decay``.
    """
    optimizer_name = str(optimizer_name).lower()
    learning_rate = float(learning_rate)
    weight_decay = float(weight_decay)

    if learning_rate <= 0.0:
        raise ValueError(
            f"learning_rate must be positive, got {learning_rate}."
        )
    if weight_decay < 0.0:
        raise ValueError(
            f"weight_decay must be non-negative, got {weight_decay}."
        )

    if optimizer_name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if optimizer_name == "adamw":
        return tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

    raise ValueError(
        "optimizer_name must be 'adam' or 'adamw', "
        f"got {optimizer_name!r}."
    )


def build_joint_autoencoder_variational_classifier_v2(
    input_shape: tuple[int, int],
    n_classes: int = 2,
    classification_level: str = "window",
    n_windows_per_trial: int | None = None,
    encoder_type: str = "cnn1d",
    n_channels: int = 14,
    n_bands: int | None = None,
    learning_rate: float = 1e-3,
    optimizer_name: str = "adamw",
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.0,
    ae_loss_weight: float = 0.5,
    vc_loss_weight: float = 0.5,
    vae_beta: float = 1.0,
    vc_alpha: float = 1.0,
    vc_beta: float = 1.0,
    vc_gamma: float = 0.0,
    vc_lambda: float = 1.0,
    update_discriminator: bool = False,
    use_class_weight: bool = True,
    use_subject_adversarial: bool = False,
    n_subject_classes: int | None = None,
    subject_adversarial_weight: float = 0.05,
    subject_loss_weight: float = 1.0,
    subject_hidden_units: int = 64,
    subject_dropout: float = 0.0,
    subject_latent_mode: str = "mean",
    subject_mc_samples: int = 5,
    use_supcon: bool = False,
    supcon_weight: float = 0.03,
    supcon_temperature: float = 0.1,
    supcon_cross_subject_only: bool = True,
    bilstm_units: int = 64,
    n_bilstm_layers: int = 1,
    bilstm_dropout: float = 0.30,
    trial_bilstm_units: int | None = None,
    n_trial_bilstm_layers: int | None = None,
    trial_bilstm_dropout: float | None = None,
    bilstm_kwargs: dict | None = None,
    trial_bilstm_kwargs: dict | None = None,
    encoder_kwargs: dict | None = None,
    decoder_kwargs: dict | None = None,
    classifier_kwargs: dict | None = None,
    classifier_head: str = "variational",
    model_name: str | None = None,
) -> JointAutoencoderVariationalClassifierV2:
    """Build the VAE plus a window- or trial-level recurrent classifier."""
    classification_level = str(classification_level).lower()
    if classification_level not in {"window", "trial"}:
        raise ValueError("classification_level must be window or trial.")
    if classification_level == "trial" and (
        n_windows_per_trial is None or int(n_windows_per_trial) < 1
    ):
        raise ValueError(
            "Trial classification requires n_windows_per_trial >= 1."
        )
    timesteps, n_features = map(int, input_shape)
    encoder_type = encoder_type.lower()
    encoder, decoder = _build_encoder_decoder(
        encoder_type=encoder_type,
        timesteps=timesteps,
        n_features=n_features,
        n_channels=n_channels,
        n_bands=n_bands,
        encoder_kwargs=encoder_kwargs,
        decoder_kwargs=decoder_kwargs,
    )

    label_smoothing = float(label_smoothing)
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1).")
    classifier_defaults = {
        "n_classes": n_classes,
        "label_smoothing": label_smoothing,
    }
    if classifier_kwargs:
        classifier_defaults.update(classifier_kwargs)

    dummy_window = tf.zeros((1, timesteps, n_features), dtype=tf.float32)
    latent_sequence = encoder(dummy_window, training=False)
    if latent_sequence.shape.rank != 3:
        raise ValueError(
            f"{type(encoder).__name__} must return rank-3 latent sequences; "
            f"got {latent_sequence.shape}."
        )
    latent_timesteps = latent_sequence.shape[1]
    latent_features = latent_sequence.shape[2]
    if latent_timesteps is None or latent_features is None:
        raise ValueError("The encoder must expose static latent dimensions.")

    resolved_bilstm_units = int(
        bilstm_units if trial_bilstm_units is None else trial_bilstm_units
    )
    resolved_bilstm_layers = int(
        n_bilstm_layers
        if n_trial_bilstm_layers is None
        else n_trial_bilstm_layers
    )
    resolved_bilstm_dropout = float(
        bilstm_dropout
        if trial_bilstm_dropout is None
        else trial_bilstm_dropout
    )
    if resolved_bilstm_units < 1 or resolved_bilstm_layers < 1:
        raise ValueError("BiLSTM units and layers must both be >= 1.")
    if not 0.0 <= resolved_bilstm_dropout < 1.0:
        raise ValueError("bilstm_dropout must be in [0, 1).")

    recurrent_timesteps = (
        int(latent_timesteps)
        if classification_level == "window"
        else int(n_windows_per_trial)
    )
    recurrent_features = (
        int(latent_features)
        if classification_level == "window"
        else int(latent_features)
    )
    recurrent_defaults = {
        "lstm_units": resolved_bilstm_units,
        "n_bilstm_layers": resolved_bilstm_layers,
        "dropout": resolved_bilstm_dropout,
        "name": f"joint_{encoder_type}_{classification_level}_bilstm",
    }
    reserved = {"timesteps", "n_features", "n_classes"}
    for kwargs_name, supplied_kwargs in (
        ("bilstm_kwargs", bilstm_kwargs),
        ("trial_bilstm_kwargs", trial_bilstm_kwargs),
    ):
        if not supplied_kwargs:
            continue
        conflicting = reserved.intersection(supplied_kwargs)
        if conflicting:
            raise ValueError(
                f"{kwargs_name} cannot override dimensions: "
                f"{sorted(conflicting)}"
            )
        recurrent_defaults.update(supplied_kwargs)

    classification_model = BiLSTMClassifier(
        timesteps=recurrent_timesteps,
        n_features=recurrent_features,
        n_classes=n_classes,
        **recurrent_defaults,
    ).build_feature_extractor()

    classifier_head = str(classifier_head).lower()
    if classifier_head == "dense":
        variational_classifier = DenseClassifier(**classifier_defaults)
    elif classifier_head == "hybrid":
        variational_classifier = HybridClassifier(**classifier_defaults)
    elif classifier_head == "variational":
        variational_classifier = VariationalClassifier(**classifier_defaults)
    else:
        raise ValueError(
            "classifier_head must be 'dense', 'hybrid', or 'variational'; "
            f"got {classifier_head!r}."
        )

    model = JointAutoencoderVariationalClassifierV2(
        encoder=encoder,
        decoder=decoder,
        classification_model=classification_model,
        variational_classifier=variational_classifier,
        latent_features=int(latent_features),
        classification_level=classification_level,
        ae_loss_weight=ae_loss_weight,
        vc_loss_weight=vc_loss_weight,
        vae_beta=vae_beta,
        vc_alpha=vc_alpha,
        vc_beta=vc_beta,
        vc_gamma=vc_gamma,
        vc_lambda=vc_lambda,
        update_discriminator=update_discriminator,
        use_class_weight=use_class_weight,
        use_subject_adversarial=use_subject_adversarial,
        n_subject_classes=n_subject_classes,
        subject_adversarial_weight=subject_adversarial_weight,
        subject_loss_weight=subject_loss_weight,
        subject_hidden_units=subject_hidden_units,
        subject_dropout=subject_dropout,
        subject_latent_mode=subject_latent_mode,
        subject_mc_samples=subject_mc_samples,
        use_supcon=use_supcon,
        supcon_weight=supcon_weight,
        supcon_temperature=supcon_temperature,
        supcon_cross_subject_only=supcon_cross_subject_only,
        name=(
            model_name
            or f"joint_{encoder_type}_{classification_level}_vae_bilstm_"
            f"{classifier_head}_v2"
        ),
    )
    optimizer = _build_optimizer(
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    model.compile(optimizer=optimizer)
    return model


def train_joint_autoencoder_variational_classifier_v2(
    feature_array: np.ndarray | None = None,
    label_array: np.ndarray | None = None,
    subject_id_array: np.ndarray | None = None,
    trial_id_array: np.ndarray | None = None,
    data_loader: Callable[
        [], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] | None = None,
    training_config: JointV2TrainingConfig | None = None,
    model_builder_function: Callable[[], tf.keras.Model] | None = None,
) -> dict:
    """Train window or grouped-trial samples with selectable CV."""

    training_config = training_config or JointV2TrainingConfig()
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    encoder_type = training_config.encoder_type.lower()
    if encoder_type not in {"cnn1d", "cnn2d", "gcn"}:
        raise ValueError(
            f"Unknown encoder_type={training_config.encoder_type!r}; expected "
            "cnn1d, cnn2d, or gcn."
        )
    run_name = training_config.run_name
    if not run_name.lower().endswith(f"_{encoder_type}"):
        run_name = f"{run_name}_{encoder_type}"
    run_dir = _ensure_path(
        training_config.output_dir / f"{run_name}_{run_timestamp}"
    )
    logger = _configure_run_logger(run_dir)

    if training_config.seed is not None:
        tf.keras.utils.set_random_seed(training_config.seed)
        np.random.seed(training_config.seed)

    if data_loader is not None:
        feature_array, label_array, subject_id_array, trial_id_array = data_loader()
    elif any(
        array is None
        for array in (feature_array, label_array, subject_id_array, trial_id_array)
    ):
        (
            feature_array,
            label_array,
            subject_id_array,
            trial_id_array,
        ) = load_joint_v2_training_data(dataset=training_config.dataset)

    feature_array = np.asarray(feature_array, dtype=np.float32)
    label_array = np.asarray(label_array)
    subject_id_array = np.asarray(subject_id_array)
    trial_id_array = np.asarray(trial_id_array)

    classification_level = str(training_config.classification_level).lower()
    if classification_level not in {"window", "trial"}:
        raise ValueError("classification_level must be window or trial.")

    trial_lengths = None
    if classification_level == "window":
        (
            feature_array,
            label_array,
            subject_id_array,
            trial_id_array,
        ) = _flatten_grouped_trials_to_windows(
            feature_array,
            label_array,
            subject_id_array,
            trial_id_array,
        )
        if feature_array.ndim != 3:
            raise ValueError(
                "Window mode expects (n_windows, timesteps, features); got "
                f"{feature_array.shape}."
            )
    else:
        (
            feature_array,
            label_array,
            subject_id_array,
            trial_id_array,
            trial_lengths,
        ) = _group_windows_by_subject_trial(
            feature_array=feature_array,
            label_array=label_array,
            subject_id_array=subject_id_array,
            trial_id_array=trial_id_array,
            max_windows=training_config.trial_max_windows,
            crop=training_config.trial_crop,
        )
        if feature_array.ndim != 4:
            raise ValueError(
                "Trial mode expects (n_trials, windows, timesteps, features); "
                f"got {feature_array.shape}."
            )

    if encoder_type in {"cnn2d", "gcn"}:
        resolved_channels, resolved_bands = _resolve_channel_band_shape(
            n_features=feature_array.shape[-1],
            n_channels=training_config.n_channels,
            n_bands=training_config.n_bands,
        )
        training_config.n_channels = resolved_channels
        training_config.n_bands = resolved_bands

    input_lengths = (
        len(feature_array),
        len(label_array),
        len(subject_id_array),
        len(trial_id_array),
    )
    if len(set(input_lengths)) != 1:
        raise ValueError(
            "feature_array, label_array, subject_id_array, and trial_id_array "
            f"must align. Got lengths {input_lengths}."
        )

    if model_builder_function is None:
        common_encoder_hparam_keys = {
            "t_down",
            "emb_dim",
            "dropout",
            "use_batch_norm",
        }
        architecture_hparam_keys = {
            "cnn1d": {
                "conv_filters",
                "kernel_sizes",
                "pool_after_layers",
                "pool_sizes",
            },
            "cnn2d": {
                "activation",
                "conv_filters",
                "kernel_sizes",
                "spatial_pool_sizes",
                "temporal_pool_sizes",
            },
            "gcn": {
                "activation",
                "gcn_units",
                "temporal_pool_sizes",
            },
        }
        encoder_hparam_keys = (
            common_encoder_hparam_keys | architecture_hparam_keys[encoder_type]
        )
        bilstm_hparam_keys = {
            "bilstm_units",
            "bilstm_layers",
            "bilstm_dropout",
            "trial_bilstm_units",
            "trial_bilstm_layers",
            "trial_bilstm_dropout",
        }
        model_hparam_keys = {
            "learning_rate",
            "optimizer",
            "optimizer_name",
            "weight_decay",
            "label_smoothing",
            "ae_loss_weight",
            "vc_loss_weight",
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
            "classifier_head",
            *encoder_hparam_keys,
            *bilstm_hparam_keys,
            "bilstm_kwargs",
            "trial_bilstm_kwargs",
            "encoder_kwargs",
            "decoder_kwargs",
            "classifier_kwargs",
        }

        def model_builder_function(**hparams) -> tf.keras.Model:
            unknown_hparams = set(hparams) - model_hparam_keys
            if unknown_hparams:
                raise ValueError(
                    f"Unknown {encoder_type} hyperparameter(s): "
                    f"{sorted(unknown_hparams)}"
                )

            bilstm_kwargs = dict(training_config.bilstm_kwargs)
            bilstm_kwargs.update(hparams.get("bilstm_kwargs", {}))

            trial_bilstm_kwargs = dict(training_config.trial_bilstm_kwargs)
            trial_bilstm_kwargs.update(hparams.get("trial_bilstm_kwargs", {}))

            encoder_kwargs = dict(training_config.encoder_kwargs)
            encoder_kwargs.update(
                {key: hparams[key] for key in encoder_hparam_keys if key in hparams}
            )
            encoder_kwargs.update(hparams.get("encoder_kwargs", {}))

            decoder_kwargs = dict(training_config.decoder_kwargs)
            decoder_kwargs.update(hparams.get("decoder_kwargs", {}))

            classifier_kwargs = dict(training_config.classifier_kwargs)
            classifier_kwargs.update(hparams.get("classifier_kwargs", {}))

            return build_joint_autoencoder_variational_classifier_v2(
                input_shape=tuple(feature_array.shape[-2:]),
                classification_level=classification_level,
                n_windows_per_trial=(
                    None if classification_level == "window"
                    else int(feature_array.shape[1])
                ),
                n_classes=_infer_n_classes(label_array),
                encoder_type=encoder_type,
                n_channels=training_config.n_channels,
                n_bands=training_config.n_bands,
                learning_rate=float(
                    hparams.get("learning_rate", training_config.learning_rate)
                ),
                optimizer_name=str(
                    hparams.get(
                        "optimizer",
                        hparams.get(
                            "optimizer_name",
                            training_config.optimizer_name,
                        ),
                    )
                ),
                weight_decay=float(
                    hparams.get("weight_decay", training_config.weight_decay)
                ),
                label_smoothing=float(
                    hparams.get(
                        "label_smoothing",
                        training_config.label_smoothing,
                    )
                ),
                ae_loss_weight=float(
                    hparams.get(
                        "ae_loss_weight",
                        training_config.model_kwargs.get("ae_loss_weight", 0.5),
                    )
                ),
                vc_loss_weight=float(
                    hparams.get(
                        "vc_loss_weight",
                        training_config.model_kwargs.get("vc_loss_weight", 0.5),
                    )
                ),
                vae_beta=float(
                    hparams.get(
                        "vae_beta",
                        training_config.model_kwargs.get("vae_beta", 1.0),
                    )
                ),
                vc_alpha=float(
                    hparams.get(
                        "vc_alpha",
                        training_config.model_kwargs.get("vc_alpha", 1.0),
                    )
                ),
                vc_beta=float(
                    hparams.get(
                        "vc_beta",
                        training_config.model_kwargs.get("vc_beta", 1.0),
                    )
                ),
                vc_gamma=float(
                    hparams.get(
                        "vc_gamma",
                        training_config.model_kwargs.get("vc_gamma", 0.0),
                    )
                ),
                vc_lambda=float(
                    hparams.get(
                        "vc_lambda",
                        training_config.model_kwargs.get("vc_lambda", 1.0),
                    )
                ),
                update_discriminator=bool(
                    hparams.get(
                        "update_discriminator",
                        training_config.model_kwargs.get("update_discriminator", False),
                    )
                ),
                use_class_weight=training_config.use_class_weight,
                use_subject_adversarial=bool(
                    hparams.get(
                        "use_subject_adversarial",
                        training_config.use_subject_adversarial,
                    )
                ),
                subject_adversarial_weight=float(
                    hparams.get(
                        "subject_adversarial_weight",
                        training_config.subject_adversarial_weight,
                    )
                ),
                subject_loss_weight=float(
                    hparams.get(
                        "subject_loss_weight",
                        training_config.subject_loss_weight,
                    )
                ),
                subject_hidden_units=int(
                    hparams.get(
                        "subject_hidden_units",
                        training_config.subject_hidden_units,
                    )
                ),
                subject_dropout=float(
                    hparams.get(
                        "subject_dropout",
                        training_config.subject_dropout,
                    )
                ),
                subject_latent_mode=str(
                    hparams.get(
                        "subject_latent_mode",
                        training_config.subject_latent_mode,
                    )
                ),
                subject_mc_samples=int(
                    hparams.get(
                        "subject_mc_samples",
                        training_config.subject_mc_samples,
                    )
                ),
                use_supcon=bool(
                    hparams.get("use_supcon", training_config.use_supcon)
                ),
                supcon_weight=float(
                    hparams.get("supcon_weight", training_config.supcon_weight)
                ),
                supcon_temperature=float(
                    hparams.get(
                        "supcon_temperature",
                        training_config.supcon_temperature,
                    )
                ),
                supcon_cross_subject_only=bool(
                    hparams.get(
                        "supcon_cross_subject_only",
                        training_config.supcon_cross_subject_only,
                    )
                ),
                bilstm_units=int(
                    hparams.get("bilstm_units", training_config.bilstm_units)
                ),
                n_bilstm_layers=int(
                    hparams.get("bilstm_layers", training_config.n_bilstm_layers)
                ),
                bilstm_dropout=float(
                    hparams.get("bilstm_dropout", training_config.bilstm_dropout)
                ),
                trial_bilstm_units=(
                    None
                    if hparams.get(
                        "trial_bilstm_units", training_config.trial_bilstm_units
                    )
                    is None
                    else int(
                        hparams.get(
                            "trial_bilstm_units",
                            training_config.trial_bilstm_units,
                        )
                    )
                ),
                n_trial_bilstm_layers=(
                    None
                    if hparams.get(
                        "trial_bilstm_layers",
                        training_config.n_trial_bilstm_layers,
                    )
                    is None
                    else int(
                        hparams.get(
                            "trial_bilstm_layers",
                            training_config.n_trial_bilstm_layers,
                        )
                    )
                ),
                trial_bilstm_dropout=(
                    None
                    if hparams.get(
                        "trial_bilstm_dropout",
                        training_config.trial_bilstm_dropout,
                    )
                    is None
                    else float(
                        hparams.get(
                            "trial_bilstm_dropout",
                            training_config.trial_bilstm_dropout,
                        )
                    )
                ),
                bilstm_kwargs=bilstm_kwargs,
                trial_bilstm_kwargs=trial_bilstm_kwargs,
                encoder_kwargs=encoder_kwargs,
                decoder_kwargs=decoder_kwargs,
                classifier_kwargs=classifier_kwargs,
                classifier_head=str(
                    hparams.get(
                        "classifier_head",
                        training_config.classifier_head,
                    )
                ),
            )

    logger.info("Starting joint-model v2 training run in %s", run_dir)
    logger.info("Encoder type: %s", encoder_type)
    logger.info("Classification/evaluation level: %s", classification_level)
    logger.info("Cross-validation strategy: %s", training_config.cv_strategy)
    if training_config.cv_strategy == "lnskto":
        logger.info(
            "LNSKTO split: %d subjects x %d held-out trials, split_seed=%s, "
            "require_all_classes_in_test=%s, folds=%s; selected subjects keep "
            "their non-test trials in training and test trial keys never repeat",
            training_config.lnskto_subjects,
            training_config.lnskto_trials,
            training_config.lnskto_split_seed,
            training_config.lnskto_require_all_classes_in_test,
            training_config.max_folds,
        )
    logger.info("Default classifier head: %s", training_config.classifier_head)
    logger.info(
        "Optimizer: %s, learning_rate=%.8g, weight_decay=%.8g",
        training_config.optimizer_name,
        training_config.learning_rate,
        training_config.weight_decay,
    )
    logger.info("Class weighting enabled: %s", training_config.use_class_weight)
    logger.info("Default label smoothing: %.6f", training_config.label_smoothing)
    logger.info(
        "Decision thresholds: %s; validation selection=%s_%s",
        list(training_config.decision_thresholds),
        training_config.threshold_selection_level,
        training_config.threshold_selection_metric,
    )
    logger.info(
        "Subject-adversarial branch enabled: %s",
        training_config.use_subject_adversarial,
    )
    if training_config.use_subject_adversarial:
        logger.info(
            "Subject branch: GRL weight=%.6f, loss weight=%.6f, "
            "latent_mode=%s, mc_samples=%d",
            training_config.subject_adversarial_weight,
            training_config.subject_loss_weight,
            training_config.subject_latent_mode,
            training_config.subject_mc_samples,
        )
    logger.info("Supervised contrastive loss enabled: %s", training_config.use_supcon)
    if training_config.use_supcon:
        logger.info(
            "SupCon: weight=%.6f, temperature=%.6f, cross_subject_only=%s",
            training_config.supcon_weight,
            training_config.supcon_temperature,
            training_config.supcon_cross_subject_only,
        )
        if training_config.batch_size < 8:
            logger.warning(
                "SupCon is using batch_size=%d. Small batches may contain few "
                "valid positive pairs; inspect supcon_valid_anchor_fraction.",
                training_config.batch_size,
            )
    logger.info("Window normalization: %s", training_config.window_normalization)
    logger.info("Label threshold mode: %s", training_config.label_threshold_mode)
    logger.info("Feature tensor shape: %s", feature_array.shape)
    if trial_lengths is not None:
        logger.info(
            "Trial windows: min=%d median=%.1f max=%d padded_to=%d",
            int(np.min(trial_lengths)),
            float(np.median(trial_lengths)),
            int(np.max(trial_lengths)),
            int(feature_array.shape[1]),
        )
    if encoder_type in {"cnn2d", "gcn"}:
        logger.info(
            "Channel-band grid: %d channels x %d bands",
            training_config.n_channels,
            training_config.n_bands,
        )
        if encoder_type == "cnn2d" and training_config.n_bands == 1:
            logger.warning(
                "CNN2D is running on a channels x 1 raw-signal grid. This is "
                "valid, but a channels x frequency-bands representation gives "
                "the second spatial dimension more meaning."
            )
    logger.info("Unique subjects: %d", len(np.unique(subject_id_array)))
    logger.info(
        "%s samples: %d",
        "Window" if classification_level == "window" else "Trial",
        len(feature_array),
    )
    logger.info(
        "Unique subject-trial labels represented: %d",
        len(set(zip(subject_id_array.tolist(), trial_id_array.tolist()))),
    )
    if classification_level == "window":
        logger.info(
            "Classifier path: window encoder -> latent-time BiLSTM -> %s head",
            training_config.classifier_head,
        )
    else:
        logger.info(
            "Classifier path: per-window encoder -> posterior pooling -> "
            "session-window BiLSTM -> %s head",
            training_config.classifier_head,
        )
    logger.info("Encoder window shape: %s", tuple(feature_array.shape[-2:]))

    class_ids = _as_class_ids(label_array)
    class_values, class_counts = np.unique(class_ids, return_counts=True)
    class_distribution = {
        int(class_value): int(class_count)
        for class_value, class_count in zip(class_values, class_counts)
    }
    majority_baseline = float(np.max(class_counts) / np.sum(class_counts))
    logger.info("Global class counts: %s", class_distribution)
    logger.info("Global majority-class accuracy baseline: %.6f", majority_baseline)
    if majority_baseline >= 0.60:
        logger.warning(
            "The dataset is imbalanced (majority baseline %.4f). Compare each "
            "epoch's predicted_class_*_fraction metrics with the corresponding "
            "true_class_*_fraction metrics; matching a single majority class "
            "indicates classifier collapse.",
            majority_baseline,
        )

    _write_json(run_dir / "training_config.json", asdict(training_config))

    # Attach architecture-specific sequence metadata to the builder instead of
    # changing loso_cv's public API. cross_val reads this attribute before
    # expanding the hyperparameter grid; older compatible loso_cv signatures
    # therefore do not receive an unexpected keyword argument.
    model_builder_function._sequence_hyperparameter_depths = {
        "cnn1d": {
            "conv_filters": 1,
            "kernel_sizes": 1,
            "pool_after_layers": 1,
            "pool_sizes": 1,
        },
        "cnn2d": {
            "conv_filters": 1,
            "kernel_sizes": 2,
            "spatial_pool_sizes": 2,
            "temporal_pool_sizes": 1,
        },
        "gcn": {
            "gcn_units": 1,
            "temporal_pool_sizes": 1,
        },
    }[encoder_type]

    common_cv_kwargs = {
        "model_builder_function": model_builder_function,
        "feature_array": feature_array,
        "label_array": label_array,
        "subject_id_array": subject_id_array,
        "trial_id_array": trial_id_array,
        "n_epochs": training_config.cv_max_epochs,
        "batch_size": training_config.batch_size,
        "hyperparameters": training_config.hyperparameters,
        "evaluation_level": classification_level,
        "selection_level": (
            "trial"
            if classification_level == "trial"
            else training_config.selection_level
        ),
        "selection_metric": training_config.selection_metric,
        "maximize_metric": training_config.maximize_metric,
        "metrics": ("accuracy", "f1", "precision", "recall"),
        "log_predictions": True,
        "n_prediction_latent_samples": (
            training_config.prediction_latent_samples
        ),
        "latent_sampling_seed": training_config.latent_sampling_seed,
        "decision_thresholds": training_config.decision_thresholds,
        "threshold_selection_metric": (
            training_config.threshold_selection_metric
        ),
        "threshold_selection_level": (
            training_config.threshold_selection_level
        ),
        "prediction_diagnostics": training_config.prediction_diagnostics,
        "prediction_diagnostics_every_n_epochs": (
            training_config.prediction_diagnostics_every_n_epochs
        ),
        "prediction_diagnostics_max_samples": (
            training_config.prediction_diagnostics_max_samples
        ),
        "prediction_diagnostics_threshold_tolerance": (
            training_config.prediction_diagnostics_threshold_tolerance
        ),
        "prediction_diagnostics_seed": (
            training_config.prediction_diagnostics_seed
        ),
        "validation_subjects_per_fold": (
            training_config.validation_subjects_per_fold
            if training_config.use_early_stopping
            else 0
        ),
        "validation_seed": training_config.validation_seed,
        "early_stopping_patience": (
            training_config.early_stopping_patience
            if training_config.use_early_stopping
            else None
        ),
        "early_stopping_min_delta": (
            training_config.early_stopping_min_delta
        ),
        "early_stopping_monitor": training_config.early_stopping_monitor,
        "early_stopping_mode": training_config.early_stopping_mode,
        "restore_best_weights": True,
        "verbose": training_config.outer_verbose,
        "extra_fit_kwargs": {
            "callbacks": [tf.keras.callbacks.TerminateOnNaN()]
        },
        "n_jobs": training_config.n_jobs,
        "cpus_per_worker": training_config.cpus_per_worker,
    }

    if training_config.cv_strategy == "loso":
        cv_results = loso_cv(
            **common_cv_kwargs,
            max_folds=training_config.max_folds,
        )
    elif training_config.cv_strategy == "lnskto":
        cv_results = lnskto_cv(
            **common_cv_kwargs,
            n_subjects=training_config.lnskto_subjects,
            k_trials=training_config.lnskto_trials,
            n_folds=training_config.max_folds,
            split_seed=training_config.lnskto_split_seed,
            require_all_classes_in_test=(
                training_config.lnskto_require_all_classes_in_test
            ),
        )
    else:
        raise ValueError(
            "cv_strategy must be 'loso' or 'lnskto'; got "
            f"{training_config.cv_strategy!r}."
        )

    fold_rows: list[dict] = []
    for fold_result in cv_results["outer_fold_results"]:
        row = dict(fold_result)
        # Large per-example and per-epoch logs are exported to dedicated CSVs
        # below rather than embedded as Python-list strings in the fold table.
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
        test_subjects = row.pop("outer_test_subjects", row.pop("left_out_subjects", []))
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
            row["inner_fold_results"], default=_json_default
        )
        fold_rows.append(row)

    cv_artifact_prefix = (
        "loso" if training_config.cv_strategy == "loso" else "lnskto"
    )
    _write_json(run_dir / "cv_results.json", cv_results)
    _write_csv(run_dir / "cv_folds.csv", fold_rows)
    _write_json(
        run_dir / f"{cv_artifact_prefix}_cv_results.json",
        cv_results,
    )
    _write_csv(
        run_dir / f"{cv_artifact_prefix}_cv_folds.csv",
        fold_rows,
    )
    _write_csv(
        run_dir / "grid_search_summary.csv",
        _grid_summary_rows(cv_results),
    )
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
        raise RuntimeError(
            "The selected CV function did not return best_config; "
            "the final model cannot be built."
        )
    selected_final_config = dict(cv_results["best_config"])
    _write_json(run_dir / "selected_config.json", selected_final_config)
    cv_selected_thresholds = [
        float(row["decision_threshold"])
        for row in cv_results.get("outer_fold_results", [])
        if row.get("decision_threshold") is not None
    ]
    final_decision_threshold = float(
        np.median(cv_selected_thresholds)
        if cv_selected_thresholds
        else training_config.decision_thresholds[0]
    )
    _write_json(
        run_dir / "decision_threshold_summary.json",
        {
            "fold_selected_thresholds": cv_selected_thresholds,
            "final_median_threshold": final_decision_threshold,
            "selection_metric": training_config.threshold_selection_metric,
            "selection_level": training_config.threshold_selection_level,
            "candidate_thresholds": list(training_config.decision_thresholds),
        },
    )
    logger.info(
        "Final inference threshold from fold median: %.6f",
        final_decision_threshold,
    )
    configured_epoch_cap = int(
        selected_final_config.get("epochs", training_config.cv_max_epochs)
    )
    cv_best_epochs: list[int] = []
    if training_config.final_epochs is not None:
        selected_final_epochs = max(1, int(training_config.final_epochs))
    elif training_config.use_early_stopping:
        selected_final_epochs, cv_best_epochs = _select_final_epochs_from_cv(
            cv_results=cv_results,
            strategy=training_config.final_epoch_strategy,
            fallback_epochs=configured_epoch_cap,
        )
    else:
        selected_final_epochs = max(1, configured_epoch_cap)
    selected_final_batch_size = int(
        selected_final_config.get("batch_size", training_config.batch_size)
    )
    selected_final_model_hparams = {
        key: value
        for key, value in selected_final_config.items()
        if key not in {"epochs", "batch_size"}
    }

    logger.info("Selected final config: %s", selected_final_config)
    if cv_best_epochs:
        logger.info(
            "Fold-local best epochs for selected config: %s",
            cv_best_epochs,
        )
        logger.info(
            "Selected %d final epochs using the %s strategy.",
            selected_final_epochs,
            training_config.final_epoch_strategy,
        )
    else:
        logger.info(
            "Selected %d epochs for the final full-data fit.",
            selected_final_epochs,
        )

    final_model = model_builder_function(**selected_final_model_hparams)
    final_callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    final_prediction_diagnostics_callback: PredictionDiagnostics | None = None
    if training_config.prediction_diagnostics:
        final_prediction_diagnostics_callback = PredictionDiagnostics(
            X_train=feature_array,
            y_train=label_array,
            fold_number=None,
            batch_size=selected_final_batch_size,
            every_n_epochs=training_config.prediction_diagnostics_every_n_epochs,
            max_samples=training_config.prediction_diagnostics_max_samples,
            threshold_tolerance=(
                training_config.prediction_diagnostics_threshold_tolerance
            ),
            seed=training_config.prediction_diagnostics_seed,
        )
        final_callbacks.append(final_prediction_diagnostics_callback)
    if training_config.save_final_history_csv:
        final_callbacks.insert(
            0,
            tf.keras.callbacks.CSVLogger(str(run_dir / "final_training_history.csv")),
        )

    final_class_weight = None
    if training_config.use_class_weight:
        final_class_ids = _as_class_ids(label_array)
        final_classes, final_counts = np.unique(final_class_ids, return_counts=True)
        final_class_weight = {
            int(class_id): len(final_class_ids) / (len(final_classes) * count)
            for class_id, count in zip(final_classes, final_counts)
        }
        logger.info("Final-fit window class weights: %s", final_class_weight)
    else:
        logger.info("Final-fit class weighting is disabled.")

    final_fit_inputs = (
        final_model.prepare_fit_inputs(feature_array, subject_id_array)
        if getattr(final_model, "requires_subject_ids", False)
        else feature_array
    )
    final_history = final_model.fit(
        final_fit_inputs,
        label_array,
        class_weight=final_class_weight,
        epochs=selected_final_epochs,
        batch_size=selected_final_batch_size,
        verbose=training_config.final_verbose,
        callbacks=final_callbacks,
    )

    if final_prediction_diagnostics_callback is not None:
        _write_csv(
            run_dir / "final_prediction_diagnostics.csv",
            final_prediction_diagnostics_callback.history,
        )

    final_eval = final_model.evaluate(
        feature_array,
        label_array,
        verbose=0,
        return_dict=True,
    )

    if training_config.save_weights:
        final_model.save_weights(run_dir / "final_model.weights.h5")

    if training_config.save_full_model:
        final_model.save(run_dir / "final_model.keras")

    final_summary = {
        "run_dir": str(run_dir),
        "encoder_type": encoder_type,
        "n_channels": training_config.n_channels,
        "n_bands": training_config.n_bands,
        "selected_final_config": selected_final_config,
        "selected_final_epochs": selected_final_epochs,
        "selected_final_batch_size": selected_final_batch_size,
        "cv_best_epochs": cv_best_epochs,
        "final_epoch_strategy": training_config.final_epoch_strategy,
        "classification_level": classification_level,
        "default_classifier_head": training_config.classifier_head,
        "use_class_weight": training_config.use_class_weight,
        "use_supcon": bool(
            selected_final_config.get("use_supcon", training_config.use_supcon)
        ),
        "supcon_weight": float(
            selected_final_config.get("supcon_weight", training_config.supcon_weight)
        ),
        "supcon_temperature": float(
            selected_final_config.get(
                "supcon_temperature",
                training_config.supcon_temperature,
            )
        ),
        "supcon_cross_subject_only": bool(
            selected_final_config.get(
                "supcon_cross_subject_only",
                training_config.supcon_cross_subject_only,
            )
        ),
        "label_threshold_mode": training_config.label_threshold_mode,
        "prediction_diagnostics": training_config.prediction_diagnostics,
        "label_smoothing": float(
            selected_final_config.get(
                "label_smoothing",
                training_config.label_smoothing,
            )
        ),
        "decision_threshold_candidates": list(training_config.decision_thresholds),
        "cv_selected_decision_thresholds": cv_selected_thresholds,
        "final_decision_threshold": final_decision_threshold,
        "threshold_selection_metric": training_config.threshold_selection_metric,
        "threshold_selection_level": training_config.threshold_selection_level,
        "prediction_diagnostics_every_n_epochs": (
            training_config.prediction_diagnostics_every_n_epochs
        ),
        "prediction_diagnostics_max_samples": (
            training_config.prediction_diagnostics_max_samples
        ),
        "prediction_diagnostics_threshold_tolerance": (
            training_config.prediction_diagnostics_threshold_tolerance
        ),
        "cv_strategy": training_config.cv_strategy,
        "lnskto_subjects": (
            training_config.lnskto_subjects
            if training_config.cv_strategy == "lnskto"
            else None
        ),
        "lnskto_trials": (
            training_config.lnskto_trials
            if training_config.cv_strategy == "lnskto"
            else None
        ),
        "lnskto_split_seed": (
            training_config.lnskto_split_seed
            if training_config.cv_strategy == "lnskto"
            else None
        ),
        "lnskto_selected_subjects_remain_in_training": (
            True if training_config.cv_strategy == "lnskto" else None
        ),
        "lnskto_test_trial_keys_are_globally_unique": (
            True if training_config.cv_strategy == "lnskto" else None
        ),
        "cv_results": cv_results,
        "loso_cv": (
            cv_results if training_config.cv_strategy == "loso" else None
        ),
        "lnskto_cv": (
            cv_results if training_config.cv_strategy == "lnskto" else None
        ),
        "final_fit_history": final_history.history,
        "final_full_dataset_metrics": final_eval,
    }

    _write_json(run_dir / "training_summary.json", final_summary)
    logger.info("Final full-data metrics: %s", final_eval)
    logger.info("Saved run artifacts to %s", run_dir)

    return final_summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train JointAutoencoderVariationalClassifierV2 with a CNN1D, "
            "CNN2D, or GCN autoencoder, seeded validation, selectable "
            "window/trial classification, and LOSO or LNSKTO CV."
        )
    )
    parser.add_argument("--out-dir", default="runs/joint_autoencoder_vc_v2")
    parser.add_argument("--run-name", default="joint_autoencoder_vc_v2")
    parser.add_argument(
        "--encoder-type",
        choices=("cnn1d", "cnn2d", "gcn"),
        default="cnn1d",
        help="Autoencoder family to use for this complete run (default: cnn1d).",
    )
    parser.add_argument(
        "--dataset",
        choices=("dreamer", "amigos", "eegemotions_27"),
        default="dreamer",
        help=(
            "Dataset config to use. eegemotions_27 preserves the raw 27-way "
            "Cowen labels instead of mapping them to valence/arousal."
        ),
    )
    parser.add_argument(
        "--n-channels",
        type=int,
        default=14,
        help=(
            "Number of electrode channels for CNN2D/GCN reshaping "
            "(default: 14)."
        ),
    )
    parser.add_argument(
        "--n-bands",
        type=int,
        default=None,
        help=(
            "Features per channel for CNN2D/GCN. If omitted, infer as "
            "input_features / n_channels; raw DREAMER therefore becomes 14x1."
        ),
    )
    parser.add_argument(
        "--classification-level",
        choices=("window", "trial"),
        default="window",
        help=(
            "window preserves one prediction per EEG window; trial groups "
            "ordered windows by subject/session and emits one prediction for "
            "the complete session (default: window)."
        ),
    )
    parser.add_argument(
        "--cv-strategy",
        choices=("loso", "lnskto"),
        default="loso",
        help=(
            "Cross-validation protocol. loso holds out one complete subject; "
            "lnskto holds out K complete trials from each of N selected "
            "subjects while retaining their other trials in training. Every "
            "held-out (subject, trial) key is used as test data at most once "
            "across all generated folds (default: loso)."
        ),
    )
    parser.add_argument(
        "--lnskto-subjects",
        type=int,
        default=3,
        help=(
            "Number of subjects contributing held-out trials in each LNSKTO "
            "fold (default: 3)."
        ),
    )
    parser.add_argument(
        "--lnskto-trials",
        type=int,
        default=3,
        help=(
            "Number of complete trials held out from each selected subject in "
            "every LNSKTO fold (default: 3)."
        ),
    )
    parser.add_argument(
        "--lnskto-split-seed",
        type=int,
        default=42,
        help=(
            "Seed for deterministic, globally trial-disjoint LNSKTO fold "
            "generation (default: 42)."
        ),
    )
    lnskto_class_group = parser.add_mutually_exclusive_group()
    lnskto_class_group.add_argument(
        "--lnskto-require-all-classes",
        dest="lnskto_require_all_classes",
        action="store_true",
        help="Require every LNSKTO test fold to contain all target classes.",
    )
    lnskto_class_group.add_argument(
        "--lnskto-allow-single-class-folds",
        dest="lnskto_require_all_classes",
        action="store_false",
        help="Allow an LNSKTO test fold containing only one target class.",
    )
    parser.set_defaults(lnskto_require_all_classes=True)
    parser.add_argument(
        "--trial-max-windows",
        type=int,
        default=None,
        help=(
            "Optional cap on windows retained per trial. Omit to keep the "
            "entire session and pad to the longest trial."
        ),
    )
    parser.add_argument(
        "--trial-crop",
        choices=("start", "center", "end"),
        default="center",
        help=(
            "Which contiguous windows to retain when --trial-max-windows "
            "crops a longer session (default: center)."
        ),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--optimizer",
        choices=("adam", "adamw"),
        default="adamw",
        help=(
            "Optimizer used for every CV and final fit. AdamW applies "
            "decoupled weight decay (default: adamw)."
        ),
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help=(
            "Decoupled weight decay used by AdamW. Ignored by plain Adam "
            "(default: 1e-4)."
        ),
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help=(
            "Default classifier label-smoothing level in [0, 1). "
            "Use 0 to disable it (default: 0)."
        ),
    )
    parser.add_argument(
        "--label-smoothing-levels",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Convenience grid for testing several smoothing levels, for "
            "example: --label-smoothing-levels 0 0.05 0.1. This populates "
            "the label_smoothing hyperparameter grid."
        ),
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help=(
            "Limit the current CV strategy to N folds. For LOSO this uses "
            "the first N sorted subjects; for LNSKTO this generates exactly N "
            "balanced folds. Use 1 for an end-to-end smoke test."
        ),
    )
    parser.add_argument("--final-epochs", type=int, default=None)
    parser.add_argument(
        "--final-epoch-strategy",
        choices=("median", "mean", "max"),
        default="median",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--selection-metric",
        choices=("loss", "joint_loss", "accuracy", "f1", "precision", "recall"),
        default="f1",
        help=(
            "Metric used to rank complete CV configurations. 'loss' is "
            "classification probability log loss; 'joint_loss' is the complete "
            "weighted Keras VAE+VC objective (default: f1)."
        ),
    )
    parser.add_argument(
        "--selection-level",
        choices=("window", "trial"),
        default="window",
        help=(
            "Metric aggregation level used for configuration selection. "
            "Trial classification requires trial; window classification "
            "may select by window or trial aggregation."
        ),
    )
    parser.add_argument(
        "--prediction-latent-samples",
        type=int,
        default=0,
        help=(
            "Number of q(z|x) samples averaged for every CV prediction. "
            "Use 0 for deterministic z_mean inference, 1 for one random draw, "
            "or values such as 5/30 for Monte Carlo averaging (default: 0)."
        ),
    )
    parser.add_argument(
        "--latent-sampling-seed",
        type=int,
        default=None,
        help="Optional seed for reproducible Monte Carlo latent prediction.",
    )
    parser.add_argument(
        "--decision-thresholds",
        type=float,
        nargs="+",
        default=[0.5],
        help=(
            "Candidate binary class-1 thresholds selected independently in "
            "each CV fold using validation subjects, for example 0.35 0.40 "
            "0.45 0.50 0.55 (default: 0.5)."
        ),
    )
    parser.add_argument(
        "--threshold-selection-metric",
        choices=("accuracy", "f1", "balanced_accuracy", "binary_f1"),
        default="f1",
        help="Validation metric maximized when choosing the fold threshold.",
    )
    parser.add_argument(
        "--threshold-selection-level",
        choices=("window", "trial"),
        default="trial",
        help=(
            "Choose thresholds from validation windows or trial-averaged "
            "probabilities (default: trial)."
        ),
    )
    parser.add_argument(
        "--no-prediction-diagnostics",
        action="store_true",
        help=(
            "Disable per-epoch deterministic probability, logit, and latent "
            "spread diagnostics."
        ),
    )
    parser.add_argument(
        "--prediction-diagnostics-every",
        type=int,
        default=1,
        help="Run prediction diagnostics every N epochs (default: 1).",
    )
    parser.add_argument(
        "--prediction-diagnostics-samples",
        type=int,
        default=256,
        help=(
            "Maximum approximately class-balanced samples inspected from each "
            "training/validation split (default: 256)."
        ),
    )
    parser.add_argument(
        "--prediction-threshold-tolerance",
        type=float,
        default=0.01,
        help=(
            "Probability distance used to flag near-uniform predictions. For "
            "binary models, 0.01 marks p(class 1) in (0.49, 0.51)."
        ),
    )
    parser.add_argument(
        "--prediction-diagnostics-seed",
        type=int,
        default=42,
        help="Seed used to select fixed diagnostic subsets (default: 42).",
    )
    parser.add_argument("--outer-verbose", type=int, default=0)
    parser.add_argument("--final-verbose", type=int, default=1)
    parser.add_argument(
        "--validation-subjects",
        type=int,
        default=2,
        help=(
            "Number of seeded subjects reserved for validation inside every "
            "CV fold (default: 2). In LNSKTO, subjects contributing test "
            "trials are excluded from validation selection."
        ),
    )
    parser.add_argument(
        "--validation-seed",
        type=int,
        default=None,
        help="Validation split seed; defaults to --seed or 42.",
    )
    parser.add_argument(
        "--no-early-stopping",
        "--no-inner-early-stopping",
        action="store_true",
        help="Disable seeded validation and early stopping in LOSO fits.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        "--inner-patience",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        "--inner-min-delta",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--early-stopping-monitor",
        default="val_accuracy",
        help=(
            "Validation metric monitored by early stopping. val_loss is the "
            "complete weighted joint VAE+VC objective and should use "
            "--early-stopping-mode min. Other available metrics include "
            "val_accuracy, val_loss, val_decoder_accuracy, "
            "val_vc_cross_entropy, and val_accuracy "
            "(default monitor: val_accuracy)."
        ),
    )
    parser.add_argument(
        "--early-stopping-mode",
        choices=("auto", "min", "max"),
        default="max",
        help=(
            "Whether the monitored metric should decrease or increase "
            "(default: max for val_accuracy)."
        ),
    )
    parser.add_argument("--no-save-full-model", action="store_true")
    parser.add_argument("--no-save-weights", action="store_true")
    parser.add_argument("--no-save-final-history-csv", action="store_true")
    parser.add_argument("--ae-loss-weight", type=float, default=0.5)
    parser.add_argument("--vc-loss-weight", type=float, default=0.5)
    parser.add_argument(
        "--vae-beta",
        type=float,
        default=1.0,
        help=(
            "Positive KL weight for the autoencoder posterior "
            "q(z_ae|x) against N(0, I) (default: 1.0)."
        ),
    )
    parser.add_argument("--vc-alpha", type=float, default=1.0)
    parser.add_argument("--vc-beta", type=float, default=1.0)
    parser.add_argument("--vc-gamma", type=float, default=0.0)
    parser.add_argument("--vc-lambda", type=float, default=1.0)
    parser.add_argument("--update-discriminator", action="store_true")
    subject_group = parser.add_mutually_exclusive_group()
    subject_group.add_argument(
        "--use-subject-adversarial",
        dest="use_subject_adversarial",
        action="store_true",
        help=(
            "Enable a fold-local subject classifier behind gradient reversal "
            "so the encoder learns subject-invariant latent features."
        ),
    )
    subject_group.add_argument(
        "--no-subject-adversarial",
        dest="use_subject_adversarial",
        action="store_false",
        help="Disable the subject-adversarial branch (default).",
    )
    parser.set_defaults(use_subject_adversarial=False)
    parser.add_argument(
        "--subject-adversarial-weight",
        type=float,
        default=0.05,
        help=(
            "Gradient-reversal strength applied only to the encoder "
            "(default: 0.05)."
        ),
    )
    parser.add_argument(
        "--subject-loss-weight",
        type=float,
        default=1.0,
        help=(
            "Positive subject cross-entropy weight for the subject head "
            "(default: 1.0)."
        ),
    )
    parser.add_argument("--subject-hidden-units", type=int, default=64)
    parser.add_argument("--subject-dropout", type=float, default=0.0)
    parser.add_argument(
        "--subject-latent-mode",
        choices=("mean", "mc"),
        default="mean",
        help=(
            "Use the posterior mean or Monte Carlo posterior samples for the "
            "subject adversary (default: mean)."
        ),
    )
    parser.add_argument(
        "--subject-mc-samples",
        type=int,
        default=5,
        help=(
            "Number of posterior draws averaged when "
            "--subject-latent-mode mc is selected (default: 5)."
        ),
    )
    supcon_group = parser.add_mutually_exclusive_group()
    supcon_group.add_argument(
        "--use-supcon",
        dest="use_supcon",
        action="store_true",
        help=(
            "Enable supervised contrastive regularization on the post-BiLSTM, "
            "pre-classifier embedding."
        ),
    )
    supcon_group.add_argument(
        "--no-supcon",
        dest="use_supcon",
        action="store_false",
        help="Disable supervised contrastive regularization (default).",
    )
    parser.set_defaults(use_supcon=False)
    parser.add_argument(
        "--supcon-weight",
        type=float,
        default=0.03,
        help="Weight applied to the supervised contrastive loss (default: 0.03).",
    )
    parser.add_argument(
        "--supcon-temperature",
        type=float,
        default=0.1,
        help="Positive SupCon similarity temperature (default: 0.1).",
    )
    supcon_positive_group = parser.add_mutually_exclusive_group()
    supcon_positive_group.add_argument(
        "--supcon-cross-subject-only",
        dest="supcon_cross_subject_only",
        action="store_true",
        help=(
            "Use only same-label examples from different subjects as positives "
            "(default)."
        ),
    )
    supcon_positive_group.add_argument(
        "--supcon-all-same-class-positives",
        dest="supcon_cross_subject_only",
        action="store_false",
        help="Treat every same-label non-self example as a SupCon positive.",
    )
    parser.set_defaults(supcon_cross_subject_only=True)
    parser.add_argument(
        "--bilstm-units",
        type=int,
        default=64,
        help=(
            "Hidden units in the active classification BiLSTM (default: 64)."
        ),
    )
    parser.add_argument(
        "--bilstm-layers",
        type=int,
        default=1,
        help=(
            "Number of stacked layers in the active classification BiLSTM "
            "(default: 1)."
        ),
    )
    parser.add_argument(
        "--bilstm-dropout",
        type=float,
        default=0.30,
        help="Dropout in the active classification BiLSTM (default: 0.30).",
    )
    parser.add_argument(
        "--trial-bilstm-units",
        type=int,
        default=None,
        help=(
            "Deprecated alias for --bilstm-units. When supplied, it overrides "
            "the canonical value."
        ),
    )
    parser.add_argument(
        "--trial-bilstm-layers",
        type=int,
        default=None,
        help=(
            "Deprecated alias for --bilstm-layers. When supplied, it overrides "
            "the canonical value."
        ),
    )
    parser.add_argument(
        "--trial-bilstm-dropout",
        type=float,
        default=None,
        help=(
            "Deprecated alias for --bilstm-dropout. When supplied, it overrides "
            "the canonical value."
        ),
    )
    parser.add_argument(
        "--classifier-head",
        choices=("dense", "hybrid", "variational"),
        default="variational",
        help=(
            "Default classification head when classifier_head is absent from "
            "--hyperparameters-json. The JSON grid can contain "
            "classifier_head: [\"dense\", \"variational\"] "
            "to compare both heads (default: variational)."
        ),
    )
    parser.add_argument(
        "--hyperparameters-json",
        default=None,
        help=(
            "Cartesian grid passed to the selected cross-validation function. "
            "Use only keys valid "
            "for the selected --encoder-type. CNN1D uses conv_filters, "
            "kernel_sizes, pool_after_layers, and pool_sizes; CNN2D uses "
            "conv_filters, 2D kernel_sizes, spatial_pool_sizes, and "
            "temporal_pool_sizes; GCN uses "
            "gcn_units and temporal_pool_sizes. Common keys include t_down, "
            "emb_dim, dropout, use_batch_norm, classifier_head "
            "('dense', 'hybrid', or 'variational'), SupCon settings, and the single "
            "window-level "
            "BiLSTM/loss settings."
        ),
    )
    parser.add_argument("--features-npy", default=None)
    parser.add_argument("--labels-npy", default=None)
    parser.add_argument("--subjects-npy", default=None)
    parser.add_argument(
        "--trials-npy",
        default=None,
        help=(
            "Trial ID array aligned with --features-npy. Rank-3 window "
            "arrays are used directly; legacy rank-4 trial arrays are flattened."
        ),
    )
    parser.add_argument(
        "--raw-eeg-npy",
        default=None,
        help=(
            "Path to a pre-converted *_eeg.npy file, shape "
            "(n_subjects, n_trials, n_channels, n_samples) (see "
            "STSNet/prepare_datasets.py). Defaults to the bundled DREAMER "
            "array if neither this nor --features-npy/--labels-npy/"
            "--subjects-npy are given."
        ),
    )
    parser.add_argument(
        "--raw-labels-npy",
        default=None,
        help=(
            "Path to the matching *_labels.npy file, shape "
            "(n_subjects, n_trials, n_label_dims)."
        ),
    )
    parser.add_argument(
        "--label-dimension",
        choices=("valence", "arousal"),
        default="valence",
        help="Which label dimension to classify (default: valence).",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=4.0,
        help="Window length in seconds for raw-signal segmentation (default: 4.0).",
    )
    parser.add_argument(
        "--window-overlap",
        type=float,
        default=0.5,
        help="Fractional overlap in [0, 1) between consecutive windows (default: 0.5).",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=DREAMER_FS,
        help="Sampling frequency in Hz (default: 128, matching STSNet's DREAMER config).",
    )
    parser.add_argument(
        "--median-label",
        type=float,
        default=DREAMER_MEDIAN_LABEL,
        help="Median-split threshold for label binarization (default: 3).",
    )
    parser.add_argument(
        "--window-normalization",
        choices=("none", "global_rms", "feature_zscore"),
        default="global_rms",
        help=(
            "Window normalization mode. global_rms removes overall gain while "
            "preserving relative channel-band power; feature_zscore is an "
            "aggressive ablation; none disables normalization."
        ),
    )
    parser.add_argument(
        "--no-zscore",
        action="store_true",
        help=(
            "Deprecated compatibility alias for --window-normalization none."
        ),
    )
    parser.add_argument(
        "--label-threshold-mode",
        choices=("global", "subject_median"),
        default="global",
        help=(
            "global preserves the dataset-wide threshold; subject_median "
            "binarizes each subject relative to their own rating median."
        ),
    )
    class_weight_group = parser.add_mutually_exclusive_group()
    class_weight_group.add_argument(
        "--use-class-weight",
        dest="use_class_weight",
        action="store_true",
        help="Enable fold-local inverse-frequency class weighting (default).",
    )
    class_weight_group.add_argument(
        "--no-class-weight",
        dest="use_class_weight",
        action="store_false",
        help="Disable class weighting for CV and final training.",
    )
    parser.set_defaults(use_class_weight=True)

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help=(
            "Number of concurrent outer-fold worker processes. "
            "Use 1 for sequential execution."
        ),
    )

    parser.add_argument(
        "--cpus-per-worker",
        type=int,
        default=None,
        help=(
            "Maximum TensorFlow CPU threads assigned to each worker. "
            "For example, 4 jobs with 2 CPUs each requires approximately 8 CPUs."
        ),
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    parsed_hyperparameters = (
        json.loads(args.hyperparameters_json)
        if args.hyperparameters_json
        else {}
    )
    if not isinstance(parsed_hyperparameters, dict):
        raise ValueError("--hyperparameters-json must decode to a JSON object.")
    if args.label_smoothing_levels is not None:
        if "label_smoothing" in parsed_hyperparameters:
            raise ValueError(
                "Specify label smoothing through either "
                "--label-smoothing-levels or label_smoothing in "
                "--hyperparameters-json, not both."
            )
        parsed_hyperparameters["label_smoothing"] = list(
            args.label_smoothing_levels
        )
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive.")
    smoothing_values = (
        args.label_smoothing_levels
        if args.label_smoothing_levels is not None
        else [args.label_smoothing]
    )
    if any(not 0.0 <= float(value) < 1.0 for value in smoothing_values):
        raise ValueError("All label-smoothing levels must be in [0, 1).")
    if not args.decision_thresholds:
        raise ValueError("--decision-thresholds must contain at least one value.")
    if any(
        not 0.0 < float(value) < 1.0
        for value in args.decision_thresholds
    ):
        raise ValueError(
            "Every --decision-thresholds value must be strictly between 0 and 1."
        )
    if len(set(map(float, args.decision_thresholds))) != len(
        args.decision_thresholds
    ):
        raise ValueError("--decision-thresholds must not contain duplicates.")
    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay must be >= 0.")
    if args.prediction_latent_samples < 0:
        raise ValueError("--prediction-latent-samples must be >= 0.")
    if args.prediction_diagnostics_every < 1:
        raise ValueError("--prediction-diagnostics-every must be >= 1.")
    if args.prediction_diagnostics_samples < 1:
        raise ValueError("--prediction-diagnostics-samples must be >= 1.")
    if args.prediction_threshold_tolerance < 0.0:
        raise ValueError("--prediction-threshold-tolerance must be >= 0.")
    if args.trial_max_windows is not None and args.trial_max_windows < 1:
        raise ValueError("--trial-max-windows must be >= 1.")
    if args.lnskto_subjects < 1:
        raise ValueError("--lnskto-subjects must be >= 1.")
    if args.lnskto_trials < 1:
        raise ValueError("--lnskto-trials must be >= 1.")
    if args.lnskto_split_seed is not None and args.lnskto_split_seed < 0:
        raise ValueError("--lnskto-split-seed must be >= 0 or omitted.")
    if args.n_channels < 1:
        raise ValueError("--n-channels must be >= 1.")
    if args.n_bands is not None and args.n_bands < 1:
        raise ValueError("--n-bands must be >= 1 when supplied.")
    if args.validation_subjects < 0:
        raise ValueError("--validation-subjects must be >= 0.")
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience must be >= 0.")
    if args.early_stopping_min_delta < 0.0:
        raise ValueError("--early-stopping-min-delta must be >= 0.")
    if args.subject_adversarial_weight < 0.0:
        raise ValueError("--subject-adversarial-weight must be >= 0.")
    if args.subject_loss_weight < 0.0:
        raise ValueError("--subject-loss-weight must be >= 0.")
    if args.subject_hidden_units < 1:
        raise ValueError("--subject-hidden-units must be >= 1.")
    if not 0.0 <= args.subject_dropout < 1.0:
        raise ValueError("--subject-dropout must be in [0, 1).")
    if args.subject_mc_samples < 1:
        raise ValueError("--subject-mc-samples must be >= 1.")
    if args.supcon_weight < 0.0:
        raise ValueError("--supcon-weight must be >= 0.")
    if args.supcon_temperature <= 0.0:
        raise ValueError("--supcon-temperature must be positive.")
    validation_seed = (
        args.validation_seed
        if args.validation_seed is not None
        else (args.seed if args.seed is not None else 42)
    )
    if (
        args.classification_level == "trial"
        and args.selection_level != "trial"
    ):
        print(
            "Trial classification requires trial-level selection; overriding "
            "--selection-level to trial.",
            flush=True,
        )
    resolved_selection_level = (
        "trial"
        if args.classification_level == "trial"
        else args.selection_level
    )
    early_stopping_monitor = args.early_stopping_monitor

    config = JointV2TrainingConfig(
        output_dir=Path(args.out_dir),
        run_name=args.run_name,
        dataset=args.dataset,
        encoder_type=args.encoder_type,
        n_channels=args.n_channels,
        n_bands=args.n_bands,
        learning_rate=args.learning_rate,
        optimizer_name=args.optimizer,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        cv_max_epochs=args.epochs,
        cv_strategy=args.cv_strategy,
        lnskto_subjects=args.lnskto_subjects,
        lnskto_trials=args.lnskto_trials,
        lnskto_split_seed=args.lnskto_split_seed,
        lnskto_require_all_classes_in_test=(
            args.lnskto_require_all_classes
        ),
        final_epoch_strategy=args.final_epoch_strategy,
        final_epochs=args.final_epochs,
        classification_level=args.classification_level,
        selection_metric=args.selection_metric,
        selection_level=resolved_selection_level,
        trial_max_windows=args.trial_max_windows,
        trial_crop=args.trial_crop,
        prediction_latent_samples=args.prediction_latent_samples,
        latent_sampling_seed=args.latent_sampling_seed,
        decision_thresholds=tuple(sorted(map(float, args.decision_thresholds))),
        threshold_selection_metric=args.threshold_selection_metric,
        threshold_selection_level=args.threshold_selection_level,
        prediction_diagnostics=not args.no_prediction_diagnostics,
        prediction_diagnostics_every_n_epochs=(
            args.prediction_diagnostics_every
        ),
        prediction_diagnostics_max_samples=args.prediction_diagnostics_samples,
        prediction_diagnostics_threshold_tolerance=(
            args.prediction_threshold_tolerance
        ),
        prediction_diagnostics_seed=args.prediction_diagnostics_seed,
        validation_subjects_per_fold=args.validation_subjects,
        validation_seed=validation_seed,
        outer_verbose=args.outer_verbose,
        final_verbose=args.final_verbose,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        early_stopping_monitor=early_stopping_monitor,
        early_stopping_mode=args.early_stopping_mode,
        use_early_stopping=not args.no_early_stopping,
        save_full_model=not args.no_save_full_model,
        save_weights=not args.no_save_weights,
        save_final_history_csv=not args.no_save_final_history_csv,
        seed=args.seed,
        bilstm_units=args.bilstm_units,
        n_bilstm_layers=args.bilstm_layers,
        bilstm_dropout=args.bilstm_dropout,
        trial_bilstm_units=args.trial_bilstm_units,
        n_trial_bilstm_layers=args.trial_bilstm_layers,
        trial_bilstm_dropout=args.trial_bilstm_dropout,
        classifier_head=args.classifier_head,
        label_smoothing=args.label_smoothing,
        model_kwargs={
            "ae_loss_weight": args.ae_loss_weight,
            "vc_loss_weight": args.vc_loss_weight,
            "vae_beta": args.vae_beta,
            "vc_alpha": args.vc_alpha,
            "vc_beta": args.vc_beta,
            "vc_gamma": args.vc_gamma,
            "vc_lambda": args.vc_lambda,
            "update_discriminator": args.update_discriminator,
        },
        hyperparameters=parsed_hyperparameters,
        n_jobs=args.n_jobs,
        cpus_per_worker=args.cpus_per_worker,
        max_folds=args.max_folds,
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
        data_loader = lambda: load_joint_v2_training_data(
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

    train_joint_autoencoder_variational_classifier_v2(
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
