"""Training entry point for SIC: Subject Invariant Calibrator.

Protocol
--------
For every hyperparameter configuration and target subject:
  1. pretrain one source model on all other subjects;
  2. evaluate the untouched source model on all target trials (0 calibration);
  3. for every requested (shots, folds) calibration pair, construct seeded
     target-only calibration/evaluation splits;
  4. for each split, restore the same source weights, evaluate paired zero-shot,
     fine-tune only the configured dense/softmax suffix, and re-evaluate;
  5. aggregate folds within each subject and shot level, then across subjects.

The expensive source model is fit once per target subject. Every calibration
fit uses a fresh optimizer and restored source weights. Grid ranking
can use either the zero-shot LOSO aggregate or the post-calibration aggregate;
calibration is always run and reported regardless of the selection level.

``remove_median_label`` is a data hyperparameter. When true, every trial whose
original target rating equals ``median_label`` is removed in full before SIC
splitting, including every window from that trial. It may be fixed or searched
through the same JSON Cartesian grid and is never forwarded to the model.

Source optimization is selected with the model hyperparameter
``training_method``: ``"erm"`` for ordinary joint training, ``"vrex"`` for
subject-risk variance regularization, or ``"mldg"`` for first-order
subject-episodic meta-learning.  MLDG defaults to four rotating virtual-unseen
subjects and every remaining source subject in meta-train.  Each episode samples
complete trials and applies one persistent outer update.

For ``architecture_mode="feature_fusion"``, SIC directly concatenates the
GCN-GRU sequence with an independently encoded raw-EEG BiLSTM sequence. Use
``bilstm_output_dim`` to choose the BiLSTM's total bidirectional output width
(for example 84 or 126). ``alternating_branch_optimization=true`` alternates
GCN-GRU and BiLSTM encoder updates 1:1 while updating shared heads every batch.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from itertools import product
import json
import logging
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf

try:
    from .sic_model import SIC_BUILDER_API_VERSION, build_sic_model
except ImportError:
    HERE = Path(__file__).resolve().parent
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from sic_model import SIC_BUILDER_API_VERSION, build_sic_model

try:
    from ...cross_val import subject_calibration_cv
except ImportError:
    from eegproc.deep_learning.cross_val import subject_calibration_cv

try:
    from ..joint_v2_data import (
        build_joint_v2_dataset,
        get_dataset_config,
    )
except ImportError:
    from eegproc.deep_learning.joint_architectures.joint_v2_data import (
        build_joint_v2_dataset,
        get_dataset_config,
    )


@dataclass(slots=True)
class SICTrainingConfig:
    output_dir: Path = Path("runs/sic")
    run_name: str = "dreamer_valence_sic"
    dataset: str = "dreamer"
    n_channels: int = 14
    n_bands: int = 3
    classification_level: str = "window"

    training_protocol: str = "subject_calibration"
    source_epochs: int = 100
    source_batch_size: int = 8

    validation_subjects: int = 4
    validation_seed: int | None = 42
    early_stopping_patience: int | None = 10
    early_stopping_min_delta: float = 0.001
    early_stopping_monitor: str = "val_loss"
    early_stopping_mode: str = "auto"
    best_epoch_metric: str | None = None
    selection_metric: str = "balanced_accuracy"
    hyperparameter_selection_level: str = "calibration"
    restore_best_weights: bool = True
    calibration_epochs: int = 30
    calibration_batch_size: int = 6
    calibration_trials: int = 6
    calibration_folds: int = 3
    calibration_levels: tuple[tuple[int, int], ...] = ()
    calibration_selection_shots: int | None = None
    calibration_learning_rate: float = 1e-4
    calibration_optimizer: str = "adamw"
    calibration_weight_decay: float = 0.0
    calibration_seed: int | None = 42
    stratify_calibration: bool = True

    decision_threshold: float = 0.5
    prediction_latent_samples: int = 0
    latent_sampling_seed: int | None = 42
    log_variational_intervals: bool = False
    uncertainty_samples: int = 30
    ece_bins: int = 15

    source_use_class_weight: bool = False
    calibration_use_class_weight: bool = False
    n_jobs: int = 1
    gpu_ids: tuple[int, ...] | None = None
    cpus_per_worker: int | None = None
    max_subjects: int | None = None
    verbose: int = 1
    seed: int | None = 42

    label_threshold_mode: str = "global"
    median_label: float = 3.0
    window_normalization: str = "global_rms"
    model_config: dict = field(default_factory=dict)


# These model arguments are sequences themselves. Grid decoding uses this
# metadata to distinguish one fixed sequence (for example, gcn_units=[32])
# from sequence-valued candidates (for example, gcn_units=[[32], [64, 32]]).
SIC_SEQUENCE_HYPERPARAMETER_DEPTHS = {
    "gcn_units": 1,
    "temporal_pool_sizes": 1,
    "classification_hidden_units": 1,
    "focal_alpha": 1,
}

# These settings participate in the same JSON/Cartesian search as model
# hyperparameters, but are consumed by the data pipeline and must never be
# forwarded to ``build_sic_model``.
SIC_DATA_HYPERPARAMETERS = frozenset({"remove_median_label"})

SIC_SELECTION_OVERALL_KEYS = {
    "losocv": "zero_shot_all_trials_mean_scores",
    "calibration": "calibrated_mean_scores",
}

SIC_CLASSIFICATION_METRICS = (
    "accuracy",
    "f1",
    "precision",
    "recall",
    "macro_f1",
    "macro_precision",
    "macro_recall",
    "balanced_accuracy",
    "roc_auc",
    "brier_score",
    "ece",
)

SIC_MINIMIZE_METRICS = frozenset({"loss", "joint_loss", "brier_score", "ece"})

SIC_TRAINING_METHOD_ALIASES = {
    "normal": "erm",
    "standard": "erm",
    "joint": "erm",
    "erm": "erm",
    "vrex": "vrex",
    "v-rex": "vrex",
    "v_rex": "vrex",
    "mldg": "mldg",
    "fo_mldg": "mldg",
    "first_order_mldg": "mldg",
}


def _configuration_training_method(configuration: dict) -> str:
    """Resolve one expanded configuration, including legacy use_vrex."""
    raw_method = configuration.get("training_method")
    legacy_vrex = bool(configuration.get("use_vrex", False))
    if raw_method is None:
        return "vrex" if legacy_vrex else "erm"
    normalized = str(raw_method).strip().lower().replace("-", "_")
    normalized = SIC_TRAINING_METHOD_ALIASES.get(normalized, normalized)
    if normalized not in {"erm", "vrex", "mldg"}:
        raise ValueError(
            "training_method must be one of 'erm', 'vrex', or 'mldg'; "
            f"got {raw_method!r}."
        )
    if legacy_vrex and normalized != "vrex":
        raise ValueError(
            "use_vrex=true conflicts with training_method="
            f"{raw_method!r}. Use training_method='vrex' or remove use_vrex."
        )
    return normalized


def _metric_mode(metric_name: str) -> str:
    """Return the mathematically correct optimization direction for a metric."""
    metric_name = str(metric_name).lower()
    return "min" if metric_name in SIC_MINIMIZE_METRICS else "max"


def _calibration_plan(config: SICTrainingConfig) -> tuple[tuple[int, int], ...]:
    return config.calibration_levels or (
        (int(config.calibration_trials), int(config.calibration_folds)),
    )


def _selected_calibration_shots(config: SICTrainingConfig) -> int:
    levels = _calibration_plan(config)
    return (
        max(shots for shots, _ in levels)
        if config.calibration_selection_shots is None
        else int(config.calibration_selection_shots)
    )


def _selection_score_source(config: SICTrainingConfig) -> str:
    if config.hyperparameter_selection_level == "calibration":
        return (
            "overall.calibration_levels."
            f"{_selected_calibration_shots(config)}.calibrated_mean_scores"
        )
    return "overall.zero_shot_all_trials_mean_scores"


def _json_fingerprint(value) -> str:
    """Return a stable, human-readable representation for logs/errors."""
    return json.dumps(value, sort_keys=True, default=_json_default)


def _decode_grid_value(name: str, value):
    """Return (is_grid_axis, value_or_candidates) for one JSON entry.

    Unambiguous wrappers are supported for every parameter::

        {"grid": [candidate_1, candidate_2]}
        {"fixed": any_json_value}

    The legacy shorthand remains supported.  A list is a grid for scalar
    parameters.  For parameters whose value is itself a sequence, one list is
    fixed and a nested list denotes a grid (for example,
    ``gcn_units=[[32], [64, 32]]``).
    """
    if isinstance(value, dict) and set(value) in ({"grid"}, {"fixed"}):
        if "fixed" in value:
            return False, value["fixed"]
        candidates = value["grid"]
        if not isinstance(candidates, list):
            raise ValueError(
                f"Grid wrapper for {name!r} must contain a JSON list; "
                f"got {type(candidates).__name__}."
            )
        if not candidates:
            raise ValueError(f"Grid axis {name!r} has no candidates.")
        return True, candidates

    if not isinstance(value, list):
        return False, value
    if not value:
        if name in SIC_SEQUENCE_HYPERPARAMETER_DEPTHS:
            return False, value
        raise ValueError(f"Grid axis {name!r} has no candidates.")

    if name in SIC_SEQUENCE_HYPERPARAMETER_DEPTHS:
        # Sequence-valued model settings need one extra nesting level to be a
        # legacy grid.  The explicit {"grid": [...]} form is preferred because
        # it also handles None and other mixed candidate types without ambiguity.
        is_grid = any(isinstance(item, (list, tuple)) for item in value)
        return (True, value) if is_grid else (False, value)
    return True, value


def expand_cartesian_grid(model_config: dict) -> tuple[list[dict], dict[str, int]]:
    """Expand every requested axis into the complete Cartesian product.

    Returns the fully materialized configurations and an ordered mapping of
    grid-axis names to candidate counts.  Fixed values are copied into every
    configuration.  No random sampling or one-factor-at-a-time reduction is
    performed.
    """
    if not isinstance(model_config, dict):
        raise ValueError("Hyperparameter configuration must be a JSON object.")

    fixed: dict = {}
    axis_names: list[str] = []
    axis_candidates: list[list] = []
    for name, raw_value in model_config.items():
        is_axis, decoded = _decode_grid_value(name, raw_value)
        if is_axis:
            axis_names.append(name)
            axis_candidates.append(list(decoded))
        else:
            fixed[name] = decoded

    combinations = product(*axis_candidates) if axis_candidates else [()]
    configurations: list[dict] = []
    for values in combinations:
        configuration = dict(fixed)
        configuration.update(dict(zip(axis_names, values)))
        configurations.append(configuration)

    dimensions = {
        name: len(candidates)
        for name, candidates in zip(axis_names, axis_candidates)
    }
    return configurations, dimensions


def _split_data_hyperparameters(configuration: dict) -> tuple[dict, dict]:
    """Separate dataset controls from arguments accepted by the SIC builder."""
    model_configuration = dict(configuration)
    data_configuration = {
        name: model_configuration.pop(name)
        for name in SIC_DATA_HYPERPARAMETERS
        if name in model_configuration
    }
    remove_median_label = data_configuration.get("remove_median_label", False)
    if not isinstance(remove_median_label, (bool, np.bool_)):
        raise ValueError(
            "remove_median_label must be true or false; got "
            f"{remove_median_label!r}."
        )
    data_configuration["remove_median_label"] = bool(remove_median_label)
    return model_configuration, data_configuration


def _ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _configure_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"eegproc.sic.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(run_dir / "training.log", encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if tf.is_tensor(value):
        return value.numpy().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _class_ids(y) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 1:
        return y.astype(np.int64)
    if y.ndim == 2 and y.shape[1] == 1:
        return y[:, 0].astype(np.int64)
    if y.ndim == 2:
        return np.argmax(y, axis=1).astype(np.int64)
    raise ValueError(f"Unsupported label shape: {y.shape}")


def _n_classes(y) -> int:
    ids = _class_ids(y)
    return int(np.max(ids)) + 1


def _flatten_grouped_trials_to_windows(features, labels, subjects, trials):
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels)
    subjects = np.asarray(subjects).reshape(-1)
    trials = np.asarray(trials).reshape(-1)
    if features.ndim != 4:
        return features, labels, subjects, trials
    n_trials, n_windows, timesteps, n_features = features.shape
    return (
        features.reshape(n_trials * n_windows, timesteps, n_features),
        np.repeat(labels, n_windows, axis=0),
        np.repeat(subjects, n_windows),
        np.repeat(trials, n_windows),
    )


def _group_windows_into_trials(features, labels, subjects, trials):
    """Return (trials, windows, time, features), one label per trial."""
    features = np.asarray(features, dtype=np.float32)
    labels = _class_ids(labels)
    subjects = np.asarray(subjects).reshape(-1)
    trials = np.asarray(trials).reshape(-1)

    keys: list[tuple] = []
    seen = set()
    for subject_id, trial_id in zip(subjects.tolist(), trials.tolist()):
        key = (subject_id, trial_id)
        if key not in seen:
            seen.add(key)
            keys.append(key)

    grouped_x = []
    grouped_y = []
    grouped_subjects = []
    grouped_trials = []
    counts = []
    for subject_id, trial_id in keys:
        indices = np.flatnonzero((subjects == subject_id) & (trials == trial_id))
        trial_labels = labels[indices]
        if len(indices) == 0:
            continue
        if np.any(trial_labels != trial_labels[0]):
            raise ValueError(
                f"Inconsistent labels within subject={subject_id}, trial={trial_id}."
            )
        grouped_x.append(features[indices])
        grouped_y.append(int(trial_labels[0]))
        grouped_subjects.append(subject_id)
        grouped_trials.append(trial_id)
        counts.append(int(len(indices)))

    if not grouped_x:
        raise ValueError("No trial groups were created.")
    if len(set(counts)) != 1:
        raise ValueError(
            "SIC trial mode requires equal windows per trial; observed "
            f"counts={sorted(set(counts))}."
        )
    return (
        np.stack(grouped_x, axis=0).astype(np.float32),
        np.asarray(grouped_y, dtype=np.int64),
        np.asarray(grouped_subjects),
        np.asarray(grouped_trials),
    )


def _normalize_each_window(features, mode="global_rms", epsilon=1e-6):
    x = np.asarray(features, dtype=np.float32)
    if mode == "none":
        return x
    if mode == "global_rms":
        rms = np.sqrt(
            np.mean(np.square(x, dtype=np.float64), axis=(1, 2), keepdims=True)
        )
        return (x.astype(np.float64) / np.maximum(rms, epsilon)).astype(np.float32)
    if mode == "feature_zscore":
        mean = np.mean(x, axis=1, keepdims=True, dtype=np.float64)
        std = np.std(x, axis=1, keepdims=True, dtype=np.float64)
        return ((x.astype(np.float64) - mean) / np.maximum(std, epsilon)).astype(
            np.float32
        )
    raise ValueError(f"Unknown window normalization: {mode}")


def _original_target_window_ratings(
    labels_path,
    label_dimension,
    subjects,
    trials,
):
    """Map each sample back to its original, continuous trial rating."""
    raw = np.load(Path(labels_path), allow_pickle=False)
    dim = {"valence": 0, "arousal": 1}[label_dimension]
    subjects = np.asarray(subjects).reshape(-1)
    trials = np.asarray(trials).reshape(-1)
    if raw.ndim != 3 or raw.shape[-1] <= dim:
        raise ValueError(
            "Median-label controls require labels shaped "
            "(subjects, trials, dimensions); got "
            f"{raw.shape}."
        )
    out = np.empty(len(subjects), dtype=np.float64)
    for subject_row, subject_id in enumerate(sorted(np.unique(subjects).tolist())):
        mask = subjects == subject_id
        unique_trials = sorted(np.unique(trials[mask]).tolist())
        ratings = raw[subject_row, :, dim].astype(np.float64)
        if len(unique_trials) > len(ratings):
            raise ValueError(
                f"Subject {subject_id!r} has {len(unique_trials)} trial IDs but "
                f"only {len(ratings)} raw ratings."
            )
        mapping = {
            trial_id: float(ratings[index])
            for index, trial_id in enumerate(unique_trials)
        }
        out[mask] = np.asarray([mapping[trial_id] for trial_id in trials[mask]])
    return out


def _subject_median_window_labels(labels_path, label_dimension, subjects, trials):
    ratings = _original_target_window_ratings(
        labels_path,
        label_dimension,
        subjects,
        trials,
    )
    subjects = np.asarray(subjects).reshape(-1)
    out = np.empty(len(subjects), dtype=np.int64)
    for subject_id in sorted(np.unique(subjects).tolist()):
        mask = subjects == subject_id
        out[mask] = (ratings[mask] >= np.median(ratings[mask])).astype(np.int64)
    return out


def _group_consistent_trial_values(values, subjects, trials, *, value_name):
    """Collapse a window-aligned value to one value per subject/trial pair."""
    values = np.asarray(values).reshape(-1)
    subjects = np.asarray(subjects).reshape(-1)
    trials = np.asarray(trials).reshape(-1)
    grouped = []
    seen = set()
    for subject_id, trial_id in zip(subjects.tolist(), trials.tolist()):
        key = (subject_id, trial_id)
        if key in seen:
            continue
        seen.add(key)
        indices = np.flatnonzero((subjects == subject_id) & (trials == trial_id))
        trial_values = values[indices]
        if not np.allclose(trial_values, trial_values[0], rtol=0.0, atol=1e-8):
            raise ValueError(
                f"Inconsistent {value_name} within subject={subject_id}, "
                f"trial={trial_id}."
            )
        grouped.append(trial_values[0])
    return np.asarray(grouped, dtype=values.dtype)


def _apply_median_label_ablation(
    features,
    labels,
    subjects,
    trials,
    original_ratings,
    *,
    remove_median_label,
    median_label,
):
    """Remove every sample belonging to a trial with the selected raw rating."""
    arrays = (
        np.asarray(features),
        np.asarray(labels),
        np.asarray(subjects).reshape(-1),
        np.asarray(trials).reshape(-1),
    )
    if not remove_median_label:
        return (
            *arrays,
            {
                "remove_median_label": False,
                "removed_samples": 0,
                "removed_trials": 0,
            },
        )
    if original_ratings is None:
        raise ValueError(
            "remove_median_label=true requires original target ratings from "
            "load_sic_training_data(return_original_ratings=True)."
        )
    ratings = np.asarray(original_ratings, dtype=np.float64).reshape(-1)
    if any(len(array) != len(ratings) for array in arrays):
        raise ValueError(
            "Original ratings must align one-to-one with features, labels, "
            "subjects, and trials."
        )
    remove_mask = np.isclose(
        ratings,
        float(median_label),
        rtol=0.0,
        atol=1e-8,
    )
    keep_mask = ~remove_mask
    if not np.any(keep_mask):
        raise ValueError(
            f"Removing original rating {median_label:g} would empty the dataset."
        )
    removed_trials = len(
        {
            (subject_id, trial_id)
            for subject_id, trial_id in zip(
                arrays[2][remove_mask].tolist(),
                arrays[3][remove_mask].tolist(),
            )
        }
    )
    filtered = tuple(array[keep_mask] for array in arrays)
    return (
        *filtered,
        {
            "remove_median_label": True,
            "median_label": float(median_label),
            "removed_samples": int(np.sum(remove_mask)),
            "removed_trials": int(removed_trials),
            "retained_samples": int(np.sum(keep_mask)),
        },
    )


def load_sic_training_data(
    *,
    eeg_path,
    labels_path,
    label_dimension="valence",
    window_size_sec=4.0,
    fs=30.0,
    overlap=0.0,
    median_label=3.0,
    window_normalization="global_rms",
    label_threshold_mode="global",
    dataset="dreamer",
    return_original_ratings=False,
):
    arrays = build_joint_v2_dataset(
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
    if len(arrays) == 4:
        features, labels, subjects, trials = arrays
    elif len(arrays) == 3:
        features, labels, subjects = arrays
        raw_eeg = np.load(Path(eeg_path), mmap_mode="r", allow_pickle=False)
        n_subjects, n_trials, _, n_samples = raw_eeg.shape
        window_size = int(round(window_size_sec * fs))
        hop = max(1, int(round(window_size * (1.0 - overlap))))
        n_windows = 1 + (n_samples - window_size) // hop
        trials = np.tile(
            np.repeat(np.arange(n_trials, dtype=np.int64), n_windows),
            n_subjects,
        )
    else:
        raise ValueError("build_joint_v2_dataset must return 3 or 4 arrays.")

    features, labels, subjects, trials = _flatten_grouped_trials_to_windows(
        features, labels, subjects, trials
    )
    original_ratings = None
    if return_original_ratings or label_threshold_mode == "subject_median":
        original_ratings = _original_target_window_ratings(
            labels_path, label_dimension, subjects, trials
        )
    if label_threshold_mode == "subject_median":
        labels = np.empty(len(subjects), dtype=np.int64)
        for subject_id in sorted(np.unique(subjects).tolist()):
            mask = subjects == subject_id
            labels[mask] = (
                original_ratings[mask] >= np.median(original_ratings[mask])
            ).astype(np.int64)
    features = _normalize_each_window(features, window_normalization)
    output = (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels),
        np.asarray(subjects),
        np.asarray(trials),
    )
    return (*output, original_ratings) if return_original_ratings else output


def _subject_summary_rows(results: dict) -> list[dict]:
    return list(results.get("subject_summary_rows", []))


def _calibration_fold_rows(results: dict) -> list[dict]:
    rows: list[dict] = []
    for subject_result in results.get("subject_results", []):
        for level in subject_result.get("calibration_levels", []):
            for fold in level.get("calibration_runs", []):
                row = {
                    "target_subject": fold.get("target_subject"),
                    "calibration_shots": fold.get("calibration_shots"),
                    "calibration_fold": fold.get("calibration_fold"),
                    "calibration_trial_ids": json.dumps(
                        fold.get("calibration_trial_ids", [])
                    ),
                    "evaluation_trial_ids": json.dumps(
                        fold.get("evaluation_trial_ids", [])
                    ),
                    "n_calibration_samples": fold.get("n_calibration_samples"),
                    "n_evaluation_samples": fold.get("n_evaluation_samples"),
                    "calibration_epochs_ran": fold.get("calibration_epochs_ran"),
                }
                for prefix, scores in (
                    ("zero_shot", fold.get("zero_shot_scores", {})),
                    ("calibrated", fold.get("calibrated_scores", {})),
                    ("delta", fold.get("delta_scores", {})),
                ):
                    row.update(
                        {f"{prefix}_{key}": value for key, value in scores.items()}
                    )
                rows.append(row)
    return rows


def _selection_score_from_calibration_results(
    results: dict,
    *,
    selection_level: str,
    selection_metric: str,
    calibration_selection_shots: int | None = None,
) -> float:
    """Return the subject-aggregated score used to rank one configuration."""
    if selection_level not in SIC_SELECTION_OVERALL_KEYS:
        raise ValueError(
            f"Unknown hyperparameter selection level {selection_level!r}; "
            f"expected one of {sorted(SIC_SELECTION_OVERALL_KEYS)}."
        )
    overall = results.get("overall", {})
    if selection_level == "calibration":
        if calibration_selection_shots is None:
            calibration_selection_shots = overall.get(
                "calibration_selection_shots"
            )
        score_group = (
            f"calibration_levels.{calibration_selection_shots}."
            "calibrated_mean_scores"
        )
        scores = (
            overall.get("calibration_levels", {})
            .get(str(calibration_selection_shots), {})
            .get("calibrated_mean_scores", {})
        )
    else:
        score_group = SIC_SELECTION_OVERALL_KEYS[selection_level]
        scores = overall.get(score_group, {})
    if selection_metric not in scores:
        raise KeyError(
            f"Calibration results do not contain metric {selection_metric!r} "
            f"under overall.{score_group}. Available metrics: "
            f"{sorted(scores)}."
        )
    score = float(scores[selection_metric])
    if not np.isfinite(score):
        raise ValueError(
            f"Configuration produced a non-finite selection score: "
            f"level={selection_level!r}, metric={selection_metric!r}, "
            f"score={score!r}."
        )
    return score


def _configuration_summary_row(summary: dict) -> dict:
    """Flatten one configuration summary for the search-results CSV."""
    row = {
        "rank": summary.get("rank"),
        "configuration_id": summary["configuration_id"],
        "status": summary.get("status"),
        "selection_level": summary["selection_level"],
        "selection_metric": summary["selection_metric"],
        "selection_score": summary["selection_score"],
        "configuration_dir": summary.get("configuration_dir"),
        "hyperparameters_json": _json_fingerprint(summary["hyperparameters"]),
    }
    for prefix, score_group in (
        ("losocv", summary.get("zero_shot_all_trials_mean_scores", {})),
        ("paired_zero_shot", summary.get("paired_zero_shot_mean_scores", {})),
        ("calibration", summary.get("calibrated_mean_scores", {})),
        ("calibration_delta", summary.get("delta_mean_scores", {})),
    ):
        row.update({f"{prefix}_{name}": value for name, value in score_group.items()})
    for shots, level in summary.get("calibration_level_metrics", {}).items():
        for name, value in level.get("calibrated_mean_scores", {}).items():
            row[f"calibration_{shots}_shot_{name}"] = value
    return row


def _save_calibration_artifacts(
    configuration_dir: Path,
    *,
    model_config: dict,
    results: dict,
) -> None:
    """Persist all zero-shot and calibration artifacts for one grid candidate."""
    _write_json(configuration_dir / "model_config.json", model_config)
    _write_json(configuration_dir / "sic_calibration_results.json", results)
    _write_json(
        configuration_dir / "sic_overall_metrics.json",
        results.get("overall", {}),
    )
    _write_csv(
        configuration_dir / "sic_subject_summary.csv",
        _subject_summary_rows(results),
    )
    _write_csv(
        configuration_dir / "sic_calibration_folds.csv",
        _calibration_fold_rows(results),
    )
    for filename, result_key in (
        ("sic_window_predictions.csv", "window_prediction_log"),
        ("sic_trial_predictions.csv", "trial_prediction_log"),
        ("sic_window_uncertainty.csv", "window_variational_interval_log"),
        ("sic_trial_uncertainty.csv", "trial_variational_interval_log"),
    ):
        rows = list(results.get(result_key, []))
        if rows:
            _write_csv(configuration_dir / filename, rows)


def _source_checkpoint_settings(
    config: SICTrainingConfig,
) -> tuple[str, str]:
    """Resolve source-only checkpoint monitor aliases for nested LOSO."""
    if config.best_epoch_metric is not None:
        return (
            f"val_{config.classification_level}_{config.best_epoch_metric}",
            _metric_mode(config.best_epoch_metric),
        )

    monitor = config.early_stopping_monitor
    aliases = {
        "val_accuracy": f"val_{config.classification_level}_accuracy",
        "val_balanced_accuracy": (
            f"val_{config.classification_level}_balanced_accuracy"
        ),
        "accuracy": f"{config.classification_level}_accuracy",
        "balanced_accuracy": (
            f"{config.classification_level}_balanced_accuracy"
        ),
        "val_roc_auc": f"val_{config.classification_level}_roc_auc",
        "val_brier_score": f"val_{config.classification_level}_brier_score",
        "val_ece": f"val_{config.classification_level}_ece",
        "roc_auc": f"{config.classification_level}_roc_auc",
        "brier_score": f"{config.classification_level}_brier_score",
        "ece": f"{config.classification_level}_ece",
    }
    resolved_monitor = aliases.get(monitor, monitor)
    resolved_mode = config.early_stopping_mode
    if resolved_mode == "auto":
        metric_name = resolved_monitor.removeprefix("val_")
        metric_name = metric_name.removeprefix(
            f"{config.classification_level}_"
        )
        resolved_mode = _metric_mode(metric_name)
    return resolved_monitor, resolved_mode


def _run_subject_calibration_configuration(
    *,
    builder,
    X,
    y,
    subjects,
    trials,
    model_config: dict,
    config: SICTrainingConfig,
) -> dict:
    """Run the common LOSO + three-fold calibration evaluator once."""
    source_monitor, source_monitor_mode = _source_checkpoint_settings(config)
    return subject_calibration_cv(
        model_builder_function=builder,
        feature_array=X,
        label_array=y,
        subject_id_array=subjects,
        trial_id_array=trials,
        fixed_config=model_config,
        source_epochs=config.source_epochs,
        source_batch_size=config.source_batch_size,
        calibration_epochs=config.calibration_epochs,
        calibration_batch_size=config.calibration_batch_size,
        calibration_trials=config.calibration_trials,
        calibration_folds=config.calibration_folds,
        calibration_levels=_calibration_plan(config),
        calibration_selection_shots=_selected_calibration_shots(config),
        calibration_learning_rate=config.calibration_learning_rate,
        calibration_optimizer=config.calibration_optimizer,
        calibration_weight_decay=config.calibration_weight_decay,
        calibration_seed=config.calibration_seed,
        stratify_calibration=config.stratify_calibration,
        validation_subjects_per_fold=config.validation_subjects,
        validation_seed=config.validation_seed,
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_min_delta=config.early_stopping_min_delta,
        early_stopping_monitor=source_monitor,
        early_stopping_mode=source_monitor_mode,
        restore_best_weights=config.restore_best_weights,
        evaluation_level=config.classification_level,
        metrics=SIC_CLASSIFICATION_METRICS,
        ece_bins=config.ece_bins,
        decision_threshold=config.decision_threshold,
        n_prediction_latent_samples=config.prediction_latent_samples,
        latent_sampling_seed=config.latent_sampling_seed,
        log_predictions=True,
        log_variational_intervals=config.log_variational_intervals,
        n_uncertainty_samples=config.uncertainty_samples,
        source_use_class_weight=config.source_use_class_weight,
        calibration_use_class_weight=config.calibration_use_class_weight,
        source_fit_kwargs={"callbacks": [tf.keras.callbacks.TerminateOnNaN()]},
        calibration_fit_kwargs={
            "callbacks": [tf.keras.callbacks.TerminateOnNaN()]
        },
        verbose=config.verbose,
        n_jobs=config.n_jobs,
        gpu_ids=config.gpu_ids,
        cpus_per_worker=config.cpus_per_worker,
        max_subjects=config.max_subjects,
    )


def train_sic_loso_validation(
    feature_array,
    label_array,
    subject_id_array,
    trial_id_array,
    config: SICTrainingConfig,
    original_rating_array=None,
):
    """Full Cartesian SIC search with calibration nested inside every LOSO fold.

    Every configuration runs the same target-subject loop. The source model is
    evaluated zero-shot, then restored independently for each of the three
    calibration folds. ``hyperparameter_selection_level`` changes only which
    aggregate ranks configurations; it never disables calibration or its
    reports.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _ensure_dir(config.output_dir / f"{config.run_name}_{timestamp}")
    logger = _configure_logger(run_dir)

    if config.seed is not None:
        tf.keras.utils.set_random_seed(config.seed)
        np.random.seed(config.seed)

    X = np.asarray(feature_array, dtype=np.float32)
    y = np.asarray(label_array)
    subjects = np.asarray(subject_id_array).reshape(-1)
    trials = np.asarray(trial_id_array).reshape(-1)
    original_ratings = (
        None
        if original_rating_array is None
        else np.asarray(original_rating_array, dtype=np.float64).reshape(-1)
    )

    expected_rank = 4 if config.classification_level == "trial" else 3
    if X.ndim != expected_rank:
        raise ValueError(
            f"SIC {config.classification_level} mode expects rank {expected_rank}; "
            f"got {X.shape}."
        )
    if X.shape[-1] != config.n_channels * config.n_bands:
        raise ValueError(
            f"features={X.shape[-1]} but {config.n_channels}*{config.n_bands}="
            f"{config.n_channels * config.n_bands}."
        )

    model_config = dict(config.model_config)
    model_config.setdefault("classification_level", config.classification_level)
    model_config.setdefault("n_classes", _n_classes(y))
    model_config.setdefault("n_channels", config.n_channels)
    model_config.setdefault("n_bands", config.n_bands)

    # ``training_method`` is the primary selector. Preserve legacy JSON files
    # that still express V-REx with use_vrex=true.
    if "training_method" not in model_config and "use_vrex" not in model_config:
        model_config["training_method"] = "erm"

    grid_configurations, grid_dimensions = expand_cartesian_grid(model_config)
    if not grid_configurations:  # Defensive: product() always yields at least one.
        raise ValueError("Hyperparameter grid produced no configurations.")
    grid_records = [
        {"configuration_id": index, "hyperparameters": candidate}
        for index, candidate in enumerate(grid_configurations, start=1)
    ]

    _write_json(run_dir / "training_config.json", asdict(config))
    _write_json(run_dir / "model_config.json", model_config)
    _write_json(
        run_dir / "hyperparameter_grid.json",
        {
            "search_type": "full_cartesian_product",
            "hyperparameter_selection_level": (
                config.hyperparameter_selection_level
            ),
            "evaluation_level": config.classification_level,
            "selection_metric": config.selection_metric,
            "maximize_metric": _metric_mode(config.selection_metric) == "max",
            "selection_score_source": _selection_score_source(config),
            "calibration_always_runs": True,
            "calibration_plan": [
                {"shots": shots, "folds": folds}
                for shots, folds in _calibration_plan(config)
            ],
            "calibration_selection_shots": _selected_calibration_shots(config),
            "data_hyperparameters": sorted(SIC_DATA_HYPERPARAMETERS),
            "calibration_aggregation": {
                "within_subject": "mean of each shot level's calibration-fold metrics",
                "across_subjects": "mean of subject-level aggregates",
            },
            "dimensions": grid_dimensions,
            "n_configurations": len(grid_configurations),
            "configurations": grid_records,
        },
    )

    logger.info("Model: SIC (Subject Invariant Calibrator)")
    logger.info(
        "Training protocol: configuration -> LOSO subject -> calibration plan %s",
        _calibration_plan(config),
    )
    logger.info("Classification level: %s", config.classification_level)
    logger.info(
        "Each target: one source fit + independent fine-tunes for %s",
        _calibration_plan(config),
    )
    logger.info(
        "Hyperparameter-grid selection: level=%s metric=%s source=%s",
        config.hyperparameter_selection_level,
        config.selection_metric,
        _selection_score_source(config),
    )
    logger.info("Calibration runs and is saved for every configuration.")
    logger.info(
        "Source checkpoint selection: %d source-validation subjects, "
        "monitor=%s mode=%s patience=%s restore_best=%s",
        config.validation_subjects,
        *_source_checkpoint_settings(config),
        config.early_stopping_patience,
        config.restore_best_weights,
    )
    logger.info(
        "LOSO target data never enters source fitting, source checkpoint "
        "selection, or MI-adjacency estimation."
    )
    logger.info(
        "Full Cartesian grid: dimensions=%s total_configurations=%d",
        grid_dimensions or {"fixed_configuration": 1},
        len(grid_configurations),
    )
    for index, candidate in enumerate(grid_configurations, start=1):
        logger.info(
            "Grid configuration %d/%d: %s",
            index,
            len(grid_configurations),
            _json_fingerprint(candidate),
        )

    progress_path = run_dir / "hyperparameter_search_progress.json"
    completed_summaries: list[dict] = []
    failed_configurations: list[dict] = []

    for index, candidate in enumerate(grid_configurations, start=1):
        configuration_dir = _ensure_dir(run_dir / f"configuration_{index:04d}")
        logger.info(
            "Starting configuration %d/%d: %s",
            index,
            len(grid_configurations),
            _json_fingerprint(candidate),
        )
        _write_json(configuration_dir / "model_config.json", candidate)
        try:
            if config.seed is not None:
                tf.keras.utils.set_random_seed(config.seed)
                np.random.seed(config.seed)
            builder_config, data_config = _split_data_hyperparameters(candidate)
            (
                candidate_X,
                candidate_y,
                candidate_subjects,
                candidate_trials,
                data_summary,
            ) = _apply_median_label_ablation(
                X,
                y,
                subjects,
                trials,
                original_ratings,
                remove_median_label=data_config["remove_median_label"],
                median_label=config.median_label,
            )
            logger.info(
                "Configuration %d dataset: remove_median_label=%s "
                "median_label=%s removed_trials=%d removed_samples=%d "
                "retained_samples=%d",
                index,
                data_summary["remove_median_label"],
                config.median_label,
                data_summary["removed_trials"],
                data_summary["removed_samples"],
                data_summary.get("retained_samples", len(candidate_X)),
            )
            _write_json(configuration_dir / "dataset_ablation.json", data_summary)
            builder = partial(
                build_sic_model,
                input_shape=tuple(candidate_X.shape[1:]),
            )
            results = _run_subject_calibration_configuration(
                builder=builder,
                X=candidate_X,
                y=candidate_y,
                subjects=candidate_subjects,
                trials=candidate_trials,
                model_config=builder_config,
                config=config,
            )
            _save_calibration_artifacts(
                configuration_dir,
                model_config=candidate,
                results=results,
            )
            overall = dict(results.get("overall", {}))
            selection_score = _selection_score_from_calibration_results(
                results,
                selection_level=config.hyperparameter_selection_level,
                selection_metric=config.selection_metric,
                calibration_selection_shots=_selected_calibration_shots(config),
            )
            summary = {
                "configuration_id": index,
                "status": "completed",
                "configuration_dir": str(configuration_dir),
                "hyperparameters": candidate,
                "selection_level": config.hyperparameter_selection_level,
                "selection_metric": config.selection_metric,
                "selection_score": selection_score,
                "dataset_ablation": data_summary,
                "zero_shot_all_trials_mean_scores": overall.get(
                    "zero_shot_all_trials_mean_scores", {}
                ),
                "zero_shot_all_trials_std_scores": overall.get(
                    "zero_shot_all_trials_std_scores", {}
                ),
                "paired_zero_shot_mean_scores": overall.get(
                    "paired_zero_shot_mean_scores", {}
                ),
                "calibrated_mean_scores": overall.get(
                    "calibrated_mean_scores", {}
                ),
                "calibrated_std_scores": overall.get(
                    "calibrated_std_scores", {}
                ),
                "calibration_level_metrics": overall.get(
                    "calibration_levels", {}
                ),
                "delta_mean_scores": overall.get("delta_mean_scores", {}),
                "delta_std_scores": overall.get("delta_std_scores", {}),
            }
            completed_summaries.append(summary)
            logger.info(
                "Completed configuration %d/%d | zero-shot %s=%s | "
                "calibration %s=%s | selected score=%s",
                index,
                len(grid_configurations),
                config.selection_metric,
                summary["zero_shot_all_trials_mean_scores"].get(
                    config.selection_metric
                ),
                config.selection_metric,
                summary["calibrated_mean_scores"].get(config.selection_metric),
                selection_score,
            )
        except Exception as exc:
            failed = {
                "configuration_id": index,
                "status": "failed",
                "configuration_dir": str(configuration_dir),
                "hyperparameters": candidate,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failed_configurations.append(failed)
            _write_json(configuration_dir / "failure.json", failed)
            _write_json(
                progress_path,
                {
                    "completed": completed_summaries,
                    "failed": failed_configurations,
                    "remaining_configuration_ids": list(
                        range(index + 1, len(grid_configurations) + 1)
                    ),
                },
            )
            logger.exception("Configuration %d failed; progress was saved.", index)
            raise
        finally:
            tf.keras.backend.clear_session()

        _write_json(
            progress_path,
            {
                "completed": completed_summaries,
                "failed": failed_configurations,
                "remaining_configuration_ids": list(
                    range(index + 1, len(grid_configurations) + 1)
                ),
            },
        )

    maximize_metric = _metric_mode(config.selection_metric) == "max"
    ranked_summaries = sorted(
        completed_summaries,
        key=lambda row: (
            -float(row["selection_score"])
            if maximize_metric
            else float(row["selection_score"]),
            row["configuration_id"],
        ),
    )
    for rank, summary in enumerate(ranked_summaries, start=1):
        summary["rank"] = rank
    best = ranked_summaries[0]
    search_results = {
        "search_type": "full_cartesian_product_with_nested_subject_calibration",
        "hyperparameter_selection_level": config.hyperparameter_selection_level,
        "selection_metric": config.selection_metric,
        "maximize_metric": maximize_metric,
        "selection_score_source": _selection_score_source(config),
        "calibration_always_runs": True,
        "calibration_plan": [
            {"shots": shots, "folds": folds}
            for shots, folds in _calibration_plan(config)
        ],
        "calibration_selection_shots": _selected_calibration_shots(config),
        "data_hyperparameters": sorted(SIC_DATA_HYPERPARAMETERS),
        "calibration_aggregation": {
            "within_subject": "mean of each shot level's calibration-fold metrics",
            "across_subjects": "mean of subject-level aggregates",
        },
        "n_configurations": len(grid_configurations),
        "best_configuration_id": best["configuration_id"],
        "best_score": best["selection_score"],
        "best_hyperparameters": best["hyperparameters"],
        "ranked_configurations": ranked_summaries,
    }
    _write_json(run_dir / "hyperparameter_search_results.json", search_results)
    _write_json(
        run_dir / "best_hyperparameters.json",
        {
            "configuration_id": best["configuration_id"],
            "selection_level": config.hyperparameter_selection_level,
            "selection_metric": config.selection_metric,
            "selection_score": best["selection_score"],
            "hyperparameters": best["hyperparameters"],
            "configuration_dir": best["configuration_dir"],
        },
    )
    _write_csv(
        run_dir / "hyperparameter_search_summary.csv",
        [_configuration_summary_row(row) for row in ranked_summaries],
    )
    logger.info(
        "Best configuration: id=%d %s/%s=%s hyperparameters=%s",
        best["configuration_id"],
        config.hyperparameter_selection_level,
        config.selection_metric,
        best["selection_score"],
        _json_fingerprint(best["hyperparameters"]),
    )
    logger.info("Saved nested LOSO/calibration grid artifacts to %s", run_dir)
    return {"run_dir": str(run_dir), "results": search_results}


def train_sic(
    feature_array,
    label_array,
    subject_id_array,
    trial_id_array,
    config: SICTrainingConfig,
    original_rating_array=None,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _ensure_dir(config.output_dir / f"{config.run_name}_{timestamp}")
    logger = _configure_logger(run_dir)

    if config.seed is not None:
        tf.keras.utils.set_random_seed(config.seed)
        np.random.seed(config.seed)

    X = np.asarray(feature_array, dtype=np.float32)
    y = np.asarray(label_array)
    subjects = np.asarray(subject_id_array).reshape(-1)
    trials = np.asarray(trial_id_array).reshape(-1)
    original_ratings = (
        None
        if original_rating_array is None
        else np.asarray(original_rating_array, dtype=np.float64).reshape(-1)
    )
    expected_rank = 4 if config.classification_level == "trial" else 3
    if X.ndim != expected_rank:
        raise ValueError(
            f"SIC {config.classification_level} mode expects rank {expected_rank}; "
            f"got {X.shape}."
        )
    if X.shape[-1] != config.n_channels * config.n_bands:
        raise ValueError(
            f"features={X.shape[-1]} but {config.n_channels}*{config.n_bands}="
            f"{config.n_channels * config.n_bands}."
        )

    hyperparameters = dict(config.model_config)
    hyperparameters.setdefault("classification_level", config.classification_level)
    hyperparameters.setdefault("n_classes", _n_classes(y))
    hyperparameters.setdefault("n_channels", config.n_channels)
    hyperparameters.setdefault("n_bands", config.n_bands)
    if (
        "training_method" not in hyperparameters
        and "use_vrex" not in hyperparameters
    ):
        hyperparameters["training_method"] = "erm"

    fixed_configurations, grid_dimensions = expand_cartesian_grid(hyperparameters)
    if grid_dimensions:
        raise ValueError(
            "subject_calibration accepts one fixed configuration. Use "
            "--training-protocol loso_validation for the nested LOSO/calibration "
            "Cartesian search; grid axes were "
            f"{sorted(grid_dimensions)}."
        )
    hyperparameters = fixed_configurations[0]
    model_config, data_config = _split_data_hyperparameters(hyperparameters)
    X, y, subjects, trials, data_summary = _apply_median_label_ablation(
        X,
        y,
        subjects,
        trials,
        original_ratings,
        remove_median_label=data_config["remove_median_label"],
        median_label=config.median_label,
    )

    _write_json(run_dir / "training_config.json", asdict(config))
    _write_json(run_dir / "model_config.json", hyperparameters)
    _write_json(run_dir / "dataset_ablation.json", data_summary)

    logger.info("Model: SIC (Subject Invariant Calibrator)")
    logger.info("SIC builder API version: %s", SIC_BUILDER_API_VERSION)
    logger.info("Input shape: %s", X.shape)
    logger.info(
        "Dataset: remove_median_label=%s median_label=%s removed_trials=%d "
        "removed_samples=%d retained_samples=%d",
        data_summary["remove_median_label"],
        config.median_label,
        data_summary["removed_trials"],
        data_summary["removed_samples"],
        data_summary.get("retained_samples", len(X)),
    )
    logger.info(
        "Protocol: strict LOSO followed by calibration plan %s",
        _calibration_plan(config),
    )
    logger.info(
        "Calibration head depth: %s",
        model_config.get("calibration_unfreeze_layers", 1),
    )
    logger.info(
        "Source optimization: method=%s; V-REx penalty_weight=%s; "
        "MLDG A=%s B=%s trials_per_subject=%s inner_lr=%s beta=%s; "
        "subject_adversarial=%s",
        _configuration_training_method(model_config),
        model_config.get("vrex_penalty_weight", 1.0),
        model_config.get("mldg_meta_train_subjects", "all remaining"),
        model_config.get("mldg_meta_test_subjects", 4),
        model_config.get("mldg_trials_per_subject", 1),
        model_config.get("mldg_inner_learning_rate", 1e-4),
        model_config.get("mldg_meta_test_weight", 1.0),
        model_config.get("use_subject_adversarial", True),
    )

    builder = partial(build_sic_model, input_shape=tuple(X.shape[1:]))

    results = _run_subject_calibration_configuration(
        builder=builder,
        X=X,
        y=y,
        subjects=subjects,
        trials=trials,
        model_config=model_config,
        config=config,
    )

    _save_calibration_artifacts(
        run_dir,
        model_config=hyperparameters,
        results=results,
    )

    overall = results.get("overall", {})
    zero_accuracy = overall.get("zero_shot_all_trials_mean_scores", {}).get(
        "accuracy"
    )
    logger.info("Final 0-shot mean accuracy: %s", zero_accuracy)
    for shots, level in overall.get("calibration_levels", {}).items():
        logger.info(
            "Final %s-shot mean accuracy: %s",
            shots,
            level.get("calibrated_mean_scores", {}).get("accuracy"),
        )
    logger.info("Saved SIC artifacts to %s", run_dir)
    return {"run_dir": str(run_dir), "results": results}


def _positive_int(value: str) -> int:
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Train SIC with serial or direct-concatenation GCN-GRU/BiLSTM "
            "feature-fusion encoder, "
            "with VAE reconstruction, selectable ERM/V-REx/first-order MLDG "
            "source optimization, optional subject adversity, VC target, and "
            "configurable multi-level subject calibration."
        )
    )
    parser.add_argument("--out-dir", default="runs/sic")
    parser.add_argument("--run-name", default="dreamer_valence_sic")
    parser.add_argument(
        "--dataset", default="dreamer", choices=("dreamer", "amigos", "eegemotions_27")
    )
    parser.add_argument("--raw-eeg-npy", default=None)
    parser.add_argument("--raw-labels-npy", default=None)
    parser.add_argument(
        "--label-dimension", choices=("valence", "arousal"), default="valence"
    )
    parser.add_argument(
        "--label-threshold-mode", choices=("global", "subject_median"), default="global"
    )
    parser.add_argument("--median-label", type=float, default=3.0)
    parser.add_argument(
        "--remove-median-label",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Fixed CLI override for the remove_median_label data "
            "hyperparameter. When enabled, discard every trial whose original "
            "target rating equals --median-label, including all of that "
            "trial's windows. The default is false."
        ),
    )
    parser.add_argument("--window-sec", type=float, default=4.0)
    parser.add_argument("--window-overlap", type=float, default=0.0)
    parser.add_argument("--fs", type=float, default=30.0)
    parser.add_argument(
        "--window-normalization",
        choices=("none", "global_rms", "feature_zscore"),
        default="global_rms",
    )
    parser.add_argument("--n-channels", type=_positive_int, default=14)
    parser.add_argument("--n-bands", type=_positive_int, default=3)
    parser.add_argument(
        "--classification-level", choices=("trial", "window"), default="window"
    )

    parser.add_argument(
        "--training-protocol",
        choices=("subject_calibration", "loso_validation"),
        default="subject_calibration",
        help=(
            "subject_calibration runs one fixed SIC configuration. "
            "loso_validation runs the full Cartesian configuration -> LOSO "
            "subject -> calibration-fold search. Both report zero-shot and "
            "post-calibration metrics."
        ),
    )
    parser.add_argument("--source-epochs", type=_positive_int, default=100)
    parser.add_argument(
        "--source-batch-size",
        type=_positive_int,
        default=8,
        help=(
            "Ordinary ERM/V-REx source batch size. For MLDG, one train step is "
            "one complete subject episode; its size is controlled by the "
            "mldg_* hyperparameters instead."
        ),
    )
    parser.add_argument(
        "--training-method",
        choices=("erm", "vrex", "mldg"),
        default=None,
        help=(
            "Optional fixed override for the source-optimization hyperparameter. "
            "The same setting can be fixed or grid-searched as training_method "
            "inside --hyperparameters-json."
        ),
    )
    parser.add_argument(
        "--validation-subjects",
        type=int,
        default=4,
        help=(
            "Source-only validation subjects per outer fold. Set 0 for a fixed-"
            "budget MLDG source fit with no reserved source subjects."
        ),
    )
    parser.add_argument("--validation-seed", type=int, default=42)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help=(
            "Disable source early stopping and use the full --source-epochs "
            "budget. Recommended with --validation-subjects 0."
        ),
    )
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    parser.add_argument("--early-stopping-monitor", default="val_loss")
    parser.add_argument(
        "--best-epoch-metric",
        choices=(
            "accuracy",
            "balanced_accuracy",
            "roc_auc",
            "brier_score",
            "ece",
        ),
        default=None,
        help=(
            "Select and restore the best source-model epoch using source-only "
            "validation subjects at --classification-level. Probability "
            "metrics monitor the matching val_window_* or val_trial_* key. "
            "Brier/ECE are "
            "minimized; the others are maximized. This "
            "is separate from outer hyperparameter selection."
        ),
    )
    parser.add_argument(
        "--selection-metric",
        choices=(
            "accuracy",
            "balanced_accuracy",
            "f1",
            "macro_f1",
            "precision",
            "macro_precision",
            "recall",
            "macro_recall",
            "roc_auc",
            "brier_score",
            "ece",
        ),
        default="balanced_accuracy",
        help=(
            "Metric used by the outer hyperparameter search. Its score source "
            "is chosen by --hyperparameter-selection-level."
        ),
    )
    parser.add_argument(
        "--hyperparameter-selection-level",
        choices=("losocv", "calibration"),
        default="calibration",
        help=(
            "Choose which subject-aggregated result ranks configurations. "
            "'losocv' uses the zero-shot score on all held-out target trials; "
            "'calibration' uses the per-subject aggregate of the three "
            "post-calibration evaluations. Calibration always runs and is "
            "reported for both choices."
        ),
    )
    parser.add_argument(
        "--early-stopping-mode",
        choices=("auto", "min", "max"),
        default="auto",
    )
    parser.add_argument("--no-restore-best-weights", action="store_true")
    parser.add_argument("--calibration-epochs", type=_positive_int, default=30)
    parser.add_argument("--calibration-batch-size", type=_positive_int, default=6)
    parser.add_argument("--calibration-trials", type=_positive_int, default=6)
    parser.add_argument("--calibration-folds", type=_positive_int, default=3)
    parser.add_argument(
        "--calibration-level",
        type=_positive_int,
        nargs=2,
        action="append",
        metavar=("SHOTS", "FOLDS"),
        default=None,
        help=(
            "Repeat this flag to configure multi-shot calibration pairs, for "
            "example: --calibration-level 3 6 --calibration-level 6 3 "
            "--calibration-level 9 2 --calibration-level 12 3. The strict "
            "LOSO source model is trained once per target, then independently "
            "restored and calibrated for every requested pair. If omitted, "
            "the legacy --calibration-trials/--calibration-folds pair is used."
        ),
    )
    parser.add_argument(
        "--calibration-selection-shots",
        type=_positive_int,
        default=None,
        help=(
            "Shot level used when --hyperparameter-selection-level calibration. "
            "Defaults to the largest configured shot level."
        ),
    )
    parser.add_argument("--calibration-learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--calibration-optimizer", choices=("adam", "adamw"), default="adamw"
    )
    parser.add_argument("--calibration-weight-decay", type=float, default=0.0)
    parser.add_argument("--calibration-seed", type=int, default=42)
    parser.add_argument("--no-stratify-calibration", action="store_true")

    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--prediction-latent-samples", type=int, default=0)
    parser.add_argument("--latent-sampling-seed", type=int, default=42)
    parser.add_argument("--log-variational-intervals", action="store_true")
    parser.add_argument("--uncertainty-samples", type=_positive_int, default=30)
    parser.add_argument(
        "--ece-bins",
        type=_positive_int,
        default=15,
        help="Number of equal-width confidence bins used for ECE reporting.",
    )
    parser.add_argument("--source-use-class-weight", action="store_true")
    parser.add_argument("--calibration-use-class-weight", action="store_true")

    parser.add_argument("--n-jobs", type=_positive_int, default=1)
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=None)
    parser.add_argument("--cpus-per-worker", type=_positive_int, default=None)
    parser.add_argument("--max-subjects", type=_positive_int, default=None)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-gcn-gru-branch",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or ablate the GCN-GRU encoder branch.",
    )
    parser.add_argument(
        "--use-bilstm-branch",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or ablate the BiLSTM encoder branch.",
    )
    parser.add_argument(
        "--use-decoder",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or ablate the reconstruction decoder.",
    )
    parser.add_argument(
        "--hyperparameters-json",
        default=None,
        help=(
            "SIC model configuration or full Cartesian LOSO-validation grid as "
            "JSON. Every candidate list is crossed with every other candidate "
            "list. Use focal_gamma=[0.5,1.0,2.0] for scalar parameters. For "
            "sequence-valued parameters, prefer an explicit wrapper such as "
            "gcn_units={\"grid\":[[32],[64,32]]}; the legacy nested-list form "
            "gcn_units=[[32],[64,32]] also works. Use {\"fixed\":value} to "
            "force any JSON value to remain fixed. loso_validation searches "
            "the full grid with nested calibration; subject_calibration accepts "
            "one fixed configuration. Select source optimization with "
            "training_method=\"erm\", \"vrex\", or \"mldg\"; use "
            "training_method=[\"erm\",\"vrex\",\"mldg\"] to compare all three "
            "in a Cartesian search. MLDG settings are "
            "mldg_meta_train_subjects (null means every non-B subject), "
            "mldg_meta_test_subjects (default 4), mldg_trials_per_subject, "
            "mldg_steps_per_epoch, mldg_inner_learning_rate, "
            "mldg_meta_test_weight, and mldg_seed. For feature fusion, set "
            "bilstm_output_dim to the total output after both directions (for "
            "example 84 or 126), and optionally set "
            "alternating_branch_optimization=true for 1:1 branch updates. "
            "Ablations can set use_gcn_gru_branch=false, "
            "use_bilstm_branch=false, or use_decoder=false; at least one "
            "encoder branch must remain enabled. Equivalent fixed CLI switches "
            "are --[no-]use-gcn-gru-branch, --[no-]use-bilstm-branch, and "
            "--[no-]use-decoder. Dataset ablation remove_median_label=true "
            "removes every trial whose original target rating equals "
            "--median-label; use remove_median_label=[false,true] to compare "
            "both datasets in loso_validation, or the fixed CLI override "
            "--[no-]remove-median-label. This data setting is recorded with "
            "the hyperparameters but is not passed to the model builder. "
            "fusion_units and fusion_dropout are legacy no-op settings."
        ),
    )
    parser.add_argument(
        "--print-grid-only",
        action="store_true",
        help=(
            "Expand and print the exact Cartesian configurations, then exit "
            "before loading data or creating a model."
        ),
    )
    return parser.parse_args(argv)


def _validate_args(args, model_config):
    if not isinstance(model_config, dict):
        raise ValueError("--hyperparameters-json must decode to a JSON object.")
    if args.validation_subjects < 0:
        raise ValueError("--validation-subjects must be >= 0.")
    if args.validation_subjects == 0 and args.best_epoch_metric is not None:
        raise ValueError(
            "--best-epoch-metric requires source validation subjects. With "
            "--validation-subjects 0, use the fixed --source-epochs budget."
        )
    if args.early_stopping_patience is not None and args.early_stopping_patience < 1:
        raise ValueError("--early-stopping-patience must be >= 1.")
    if args.early_stopping_min_delta < 0.0:
        raise ValueError("--early-stopping-min-delta must be non-negative.")
    calibration_levels = tuple(
        tuple(pair)
        for pair in (
            args.calibration_level
            or [(args.calibration_trials, args.calibration_folds)]
        )
    )
    shot_levels = [shots for shots, _ in calibration_levels]
    if len(set(shot_levels)) != len(shot_levels):
        raise ValueError("Each --calibration-level SHOTS value must be unique.")
    if args.dataset == "dreamer" and any(shots >= 18 for shots in shot_levels):
        raise ValueError(
            "DREAMER has 18 trials per subject, so every calibration shot "
            "level must be < 18 to leave evaluation trials."
        )
    if (
        args.calibration_selection_shots is not None
        and args.calibration_selection_shots not in shot_levels
    ):
        raise ValueError(
            "--calibration-selection-shots must match one configured "
            f"--calibration-level; available levels are {shot_levels}."
        )
    if args.calibration_learning_rate <= 0.0:
        raise ValueError("calibration-learning-rate must be positive.")
    if args.calibration_weight_decay < 0.0:
        raise ValueError("calibration-weight-decay must be non-negative.")
    if not 0.0 < args.decision_threshold < 1.0:
        raise ValueError("decision-threshold must be in (0, 1).")
    if args.prediction_latent_samples < 0:
        raise ValueError("prediction-latent-samples must be >= 0.")
    if args.ece_bins < 2:
        raise ValueError("--ece-bins must be >= 2.")
    if "epochs" in model_config or "batch_size" in model_config:
        raise ValueError(
            "Do not put epochs/batch_size inside --hyperparameters-json; SIC has "
            "separate source and calibration epoch/batch arguments."
        )
    configurations, grid_dimensions = expand_cartesian_grid(model_config)
    for index, configuration in enumerate(configurations, start=1):
        builder_config, _ = _split_data_hyperparameters(configuration)
        training_method = _configuration_training_method(builder_config)
        if (
            builder_config.get("use_gcn_gru_branch") is False
            and builder_config.get("use_bilstm_branch") is False
        ):
            raise ValueError(
                "At least one of use_gcn_gru_branch/use_bilstm_branch must be "
                f"true; grid configuration {index} disables both."
            )
        if training_method == "mldg":
            meta_train_subjects = builder_config.get(
                "mldg_meta_train_subjects"
            )
            meta_test_subjects = int(
                builder_config.get("mldg_meta_test_subjects", 4)
            )
            trials_per_subject = int(
                builder_config.get("mldg_trials_per_subject", 1)
            )
            steps_per_epoch = builder_config.get("mldg_steps_per_epoch")
            inner_learning_rate = float(
                builder_config.get("mldg_inner_learning_rate", 1e-4)
            )
            meta_test_weight = float(
                builder_config.get("mldg_meta_test_weight", 1.0)
            )
            if meta_train_subjects is not None and int(meta_train_subjects) < 1:
                raise ValueError(
                    "mldg_meta_train_subjects must be >= 1 or null in grid "
                    f"configuration {index}."
                )
            if meta_test_subjects < 1 or trials_per_subject < 1:
                raise ValueError(
                    "mldg_meta_test_subjects and mldg_trials_per_subject must "
                    f"be >= 1 in grid configuration {index}."
                )
            if steps_per_epoch is not None and int(steps_per_epoch) < 1:
                raise ValueError(
                    "mldg_steps_per_epoch must be >= 1 or null in grid "
                    f"configuration {index}."
                )
            if inner_learning_rate <= 0.0 or meta_test_weight < 0.0:
                raise ValueError(
                    "mldg_inner_learning_rate must be positive and "
                    "mldg_meta_test_weight non-negative in grid configuration "
                    f"{index}."
                )
            if args.dataset == "dreamer" and meta_train_subjects is not None:
                available_source_subjects = 22 - args.validation_subjects
                if (
                    int(meta_train_subjects) + meta_test_subjects
                    > available_source_subjects
                ):
                    raise ValueError(
                        "DREAMER MLDG has at most 22 outer-source subjects minus "
                        f"{args.validation_subjects} reserved validation subjects, "
                        f"but configuration {index} requests "
                        f"{int(meta_train_subjects)} A + {meta_test_subjects} B. "
                        "Use mldg_meta_train_subjects=null to consume every "
                        "remaining source subject, or reduce validation subjects."
                    )
    if args.training_protocol == "subject_calibration":
        if grid_dimensions:
            raise ValueError(
                "Hyperparameter grids require --training-protocol "
                "loso_validation. subject_calibration accepts one fixed model "
                "configuration; grid-valued keys were: "
                f"{sorted(grid_dimensions)}."
            )


def main(argv=None):
    args = parse_args(argv)
    model_config = (
        json.loads(args.hyperparameters_json) if args.hyperparameters_json else {}
    )
    if args.training_method is not None:
        model_config["training_method"] = args.training_method
        # The explicit CLI selector supersedes the legacy boolean so it cannot
        # leave a contradictory pair in the builder configuration.
        model_config.pop("use_vrex", None)
    if "training_method" not in model_config and "use_vrex" not in model_config:
        model_config["training_method"] = "erm"
    for key in (
        "use_gcn_gru_branch",
        "use_bilstm_branch",
        "use_decoder",
        "remove_median_label",
    ):
        cli_value = getattr(args, key)
        if cli_value is not None:
            model_config[key] = bool(cli_value)
    _validate_args(args, model_config)
    calibration_levels = tuple(
        tuple(pair)
        for pair in (
            args.calibration_level
            or [(args.calibration_trials, args.calibration_folds)]
        )
    )
    calibration_selection_shots = (
        max(shots for shots, _ in calibration_levels)
        if args.calibration_selection_shots is None
        else int(args.calibration_selection_shots)
    )

    if args.print_grid_only:
        configurations, dimensions = expand_cartesian_grid(model_config)
        print(
            json.dumps(
                {
                    "search_type": "full_cartesian_product",
                    "hyperparameter_selection_level": (
                        args.hyperparameter_selection_level
                    ),
                    "evaluation_level": args.classification_level,
                    "selection_metric": args.selection_metric,
                    "maximize_metric": _metric_mode(args.selection_metric) == "max",
                    "selection_score_source": (
                        "overall.zero_shot_all_trials_mean_scores"
                        if args.hyperparameter_selection_level == "losocv"
                        else (
                            "overall.calibration_levels."
                            f"{calibration_selection_shots}.calibrated_mean_scores"
                        )
                    ),
                    "calibration_always_runs": True,
                    "calibration_plan": [
                        {"shots": shots, "folds": folds}
                        for shots, folds in calibration_levels
                    ],
                    "calibration_selection_shots": calibration_selection_shots,
                    "data_hyperparameters": sorted(SIC_DATA_HYPERPARAMETERS),
                    "dimensions": dimensions,
                    "n_configurations": len(configurations),
                    "configurations": [
                        {"configuration_id": index, "hyperparameters": candidate}
                        for index, candidate in enumerate(configurations, start=1)
                    ],
                },
                indent=2,
                default=_json_default,
            )
        )
        return 0

    dataset_config = get_dataset_config(args.dataset)
    eeg_path = args.raw_eeg_npy or dataset_config.eeg_path
    labels_path = args.raw_labels_npy or dataset_config.labels_path
    requested_configurations, _ = expand_cartesian_grid(model_config)
    needs_original_ratings = any(
        _split_data_hyperparameters(candidate)[1]["remove_median_label"]
        for candidate in requested_configurations
    )

    loaded = load_sic_training_data(
        eeg_path=eeg_path,
        labels_path=labels_path,
        label_dimension=args.label_dimension,
        window_size_sec=args.window_sec,
        fs=args.fs,
        overlap=args.window_overlap,
        median_label=args.median_label,
        window_normalization=args.window_normalization,
        label_threshold_mode=args.label_threshold_mode,
        dataset=dataset_config,
        return_original_ratings=needs_original_ratings,
    )
    if needs_original_ratings:
        X, y, subjects, trials, original_ratings = loaded
    else:
        X, y, subjects, trials = loaded
        original_ratings = None

    if args.classification_level == "trial":
        if original_ratings is not None:
            original_ratings = _group_consistent_trial_values(
                original_ratings,
                subjects,
                trials,
                value_name="original target rating",
            )
        X, y, subjects, trials = _group_windows_into_trials(X, y, subjects, trials)

    config = SICTrainingConfig(
        output_dir=Path(args.out_dir),
        run_name=args.run_name,
        dataset=args.dataset,
        n_channels=args.n_channels,
        n_bands=args.n_bands,
        classification_level=args.classification_level,
        training_protocol=args.training_protocol,
        source_epochs=args.source_epochs,
        source_batch_size=args.source_batch_size,
        validation_subjects=args.validation_subjects,
        validation_seed=args.validation_seed,
        early_stopping_patience=(
            None
            if args.no_early_stopping or args.validation_subjects == 0
            else args.early_stopping_patience
        ),
        early_stopping_min_delta=args.early_stopping_min_delta,
        early_stopping_monitor=args.early_stopping_monitor,
        early_stopping_mode=args.early_stopping_mode,
        best_epoch_metric=args.best_epoch_metric,
        selection_metric=args.selection_metric,
        hyperparameter_selection_level=args.hyperparameter_selection_level,
        restore_best_weights=(
            not args.no_restore_best_weights and args.validation_subjects > 0
        ),
        calibration_epochs=args.calibration_epochs,
        calibration_batch_size=args.calibration_batch_size,
        calibration_trials=args.calibration_trials,
        calibration_folds=args.calibration_folds,
        calibration_levels=calibration_levels,
        calibration_selection_shots=calibration_selection_shots,
        calibration_learning_rate=args.calibration_learning_rate,
        calibration_optimizer=args.calibration_optimizer,
        calibration_weight_decay=args.calibration_weight_decay,
        calibration_seed=args.calibration_seed,
        stratify_calibration=not args.no_stratify_calibration,
        decision_threshold=args.decision_threshold,
        prediction_latent_samples=args.prediction_latent_samples,
        latent_sampling_seed=args.latent_sampling_seed,
        log_variational_intervals=args.log_variational_intervals,
        uncertainty_samples=args.uncertainty_samples,
        ece_bins=args.ece_bins,
        source_use_class_weight=args.source_use_class_weight,
        calibration_use_class_weight=args.calibration_use_class_weight,
        n_jobs=args.n_jobs,
        gpu_ids=None if args.gpu_ids is None else tuple(args.gpu_ids),
        cpus_per_worker=args.cpus_per_worker,
        max_subjects=args.max_subjects,
        verbose=args.verbose,
        seed=args.seed,
        label_threshold_mode=args.label_threshold_mode,
        median_label=args.median_label,
        window_normalization=args.window_normalization,
        model_config=model_config,
    )
    if args.training_protocol == "loso_validation":
        train_sic_loso_validation(
            X,
            y,
            subjects,
            trials,
            config,
            original_rating_array=original_ratings,
        )
    else:
        train_sic(
            X,
            y,
            subjects,
            trials,
            config,
            original_rating_array=original_ratings,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
