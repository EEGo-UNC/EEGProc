"""Training entry point for SIC: Subject Invariant Calibrator.

Protocol
--------
For every target subject:
  1. pretrain one source model on all other subjects;
  2. evaluate the untouched source model on all target trials (0 calibration);
  3. partition the target's 18 trials into three disjoint six-trial sets;
  4. for each set, restore source weights, evaluate zero-shot on the other
     12 trials, fine-tune only the configured dense/softmax suffix on the six
     calibration trials, and evaluate those same 12 trials again.

The expensive source model is fit once per target subject.  The three
calibration fits use fresh optimizers and restored source weights.
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
    from ...cross_val import loso_cv, subject_calibration_cv
except ImportError:
    from eegproc.deep_learning.cross_val import loso_cv, subject_calibration_cv

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
    early_stopping_mode: str = "min"
    best_epoch_metric: str | None = None
    selection_metric: str = "balanced_accuracy"
    restore_best_weights: bool = True
    calibration_epochs: int = 30
    calibration_batch_size: int = 6
    calibration_trials: int = 6
    calibration_folds: int = 3
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

    source_use_class_weight: bool = False
    calibration_use_class_weight: bool = False
    n_jobs: int = 1
    gpu_ids: tuple[int, ...] | None = None
    cpus_per_worker: int | None = None
    max_subjects: int | None = None
    verbose: int = 1
    seed: int | None = 42

    label_threshold_mode: str = "global"
    window_normalization: str = "global_rms"
    model_config: dict = field(default_factory=dict)


# These model arguments are sequences themselves.  loso_cv uses this metadata
# to distinguish one fixed sequence (for example, gcn_units=[32]) from a grid
# of sequence-valued candidates (for example, gcn_units=[[32], [64, 32]]).
SIC_SEQUENCE_HYPERPARAMETER_DEPTHS = {
    "gcn_units": 1,
    "temporal_pool_sizes": 1,
    "classification_hidden_units": 1,
    "focal_alpha": 1,
}

# The grid is expanded here, rather than leaving the Cartesian-product behavior
# implicit in cross_val.py.  loso_cv therefore receives one explicit search
# dimension whose candidates are complete SIC model configurations.
SIC_EXPLICIT_GRID_KEY = "sic_grid_configuration"


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


def _build_sic_from_explicit_grid(
    *args,
    sic_grid_configuration=None,
    training_features=None,
    training_labels=None,
    training_subject_ids=None,
    training_trial_ids=None,
    adjacency=None,
    **kwargs,
):
    """Expose one complete grid candidate without hiding fold context.

    ``cross_val._build_model_with_fold_training_context`` discovers supported
    fold-local inputs from the builder signature.  Keep the ``training_*``
    arguments explicit here so LOSO passes source-training data through to
    ``build_sic_model`` for leakage-free MI-adjacency estimation.
    """
    if sic_grid_configuration is None:
        raise ValueError(
            f"Missing required explicit-grid key {SIC_EXPLICIT_GRID_KEY!r}."
        )
    if not isinstance(sic_grid_configuration, dict):
        raise TypeError(
            f"{SIC_EXPLICIT_GRID_KEY!r} must be a dictionary; got "
            f"{type(sic_grid_configuration).__name__}."
        )
    if "hyperparameters" in sic_grid_configuration:
        model_hyperparameters = sic_grid_configuration["hyperparameters"]
        if not isinstance(model_hyperparameters, dict):
            raise TypeError("Explicit-grid 'hyperparameters' must be a dictionary.")
    else:
        # Backward-compatible with a raw complete configuration.
        model_hyperparameters = sic_grid_configuration

    fold_context = {
        "training_features": training_features,
        "training_labels": training_labels,
        "training_subject_ids": training_subject_ids,
        "training_trial_ids": training_trial_ids,
        "adjacency": adjacency,
    }
    overlap = (set(kwargs) | set(fold_context)).intersection(model_hyperparameters)
    if overlap:
        raise ValueError(
            "Grid configuration collides with direct or fold-context builder "
            "arguments: "
            f"{sorted(overlap)}."
        )
    return build_sic_model(
        *args,
        **fold_context,
        **kwargs,
        **model_hyperparameters,
    )


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


def _subject_median_window_labels(labels_path, label_dimension, subjects, trials):
    raw = np.load(Path(labels_path), allow_pickle=False)
    dim = {"valence": 0, "arousal": 1}[label_dimension]
    subjects = np.asarray(subjects).reshape(-1)
    trials = np.asarray(trials).reshape(-1)
    out = np.empty(len(subjects), dtype=np.int64)
    for subject_row, subject_id in enumerate(sorted(np.unique(subjects).tolist())):
        mask = subjects == subject_id
        unique_trials = sorted(np.unique(trials[mask]).tolist())
        ratings = raw[subject_row, :, dim].astype(np.float64)
        binary = (ratings >= np.median(ratings)).astype(np.int64)
        mapping = {
            trial_id: int(binary[index]) for index, trial_id in enumerate(unique_trials)
        }
        out[mask] = np.asarray([mapping[trial_id] for trial_id in trials[mask]])
    return out


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
    if label_threshold_mode == "subject_median":
        labels = _subject_median_window_labels(
            labels_path, label_dimension, subjects, trials
        )
    features = _normalize_each_window(features, window_normalization)
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels),
        np.asarray(subjects),
        np.asarray(trials),
    )


def _subject_summary_rows(results: dict) -> list[dict]:
    return list(results.get("subject_summary_rows", []))


def _calibration_fold_rows(results: dict) -> list[dict]:
    rows: list[dict] = []
    for subject_result in results.get("subject_results", []):
        for fold in subject_result.get("calibration_folds", []):
            row = {
                "target_subject": fold.get("target_subject"),
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
                row.update({f"{prefix}_{key}": value for key, value in scores.items()})
            rows.append(row)
    return rows


def train_sic_loso_validation(
    feature_array,
    label_array,
    subject_id_array,
    trial_id_array,
    config: SICTrainingConfig,
):
    """Ordinary SIC LOSO training with subject-disjoint validation.

    This mode intentionally bypasses subject calibration. It exists to verify
    that the SIC architecture learns under conventional ERM before introducing
    V-REx. One LOSO subject is held out for test; a seeded subset of the
    remaining source subjects is held out for validation in each fold.
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

    expected_rank = 4 if config.classification_level == "trial" else 3
    if X.ndim != expected_rank:
        raise ValueError(
            f"SIC {config.classification_level} mode expects rank {expected_rank}; "
            f"got {X.shape}."
        )

    model_config = dict(config.model_config)
    model_config.setdefault("classification_level", config.classification_level)
    model_config.setdefault("n_classes", _n_classes(y))
    model_config.setdefault("n_channels", config.n_channels)
    model_config.setdefault("n_bands", config.n_bands)

    # Normal-validation mode is conventional ERM unless the caller explicitly
    # requests otherwise. The dedicated smoke script sets this False.
    model_config.setdefault("use_vrex", False)

    grid_configurations, grid_dimensions = expand_cartesian_grid(model_config)
    if not grid_configurations:  # Defensive: product() always yields at least one.
        raise ValueError("Hyperparameter grid produced no configurations.")
    grid_records = [
        {"configuration_id": index, "hyperparameters": candidate}
        for index, candidate in enumerate(grid_configurations, start=1)
    ]

    if config.best_epoch_metric is not None:
        if config.classification_level != "window":
            raise ValueError(
                "--best-epoch-metric currently selects window-level metrics and "
                "therefore requires --classification-level window."
            )
        effective_early_stopping_monitor = (
            f"val_window_{config.best_epoch_metric}"
        )
        effective_early_stopping_mode = "max"
    else:
        effective_early_stopping_monitor = config.early_stopping_monitor
        effective_early_stopping_mode = config.early_stopping_mode
        if config.classification_level == "window":
            legacy_window_monitor_aliases = {
                "val_accuracy": "val_window_accuracy",
                "val_balanced_accuracy": "val_window_balanced_accuracy",
                "accuracy": "window_accuracy",
                "balanced_accuracy": "window_balanced_accuracy",
            }
            effective_early_stopping_monitor = legacy_window_monitor_aliases.get(
                effective_early_stopping_monitor,
                effective_early_stopping_monitor,
            )

    _write_json(run_dir / "training_config.json", asdict(config))
    _write_json(run_dir / "model_config.json", model_config)
    _write_json(
        run_dir / "hyperparameter_grid.json",
        {
            "search_type": "full_cartesian_product",
            "selection_level": config.classification_level,
            "selection_metric": config.selection_metric,
            "maximize_metric": True,
            "dimensions": grid_dimensions,
            "n_configurations": len(grid_configurations),
            "configurations": grid_records,
        },
    )

    logger.info("Model: SIC (Subject Invariant Calibrator)")
    logger.info("Training protocol: ordinary LOSO + subject-disjoint validation")
    logger.info("Classification level: %s", config.classification_level)
    logger.info("Validation subjects per fold: %d", config.validation_subjects)
    logger.info(
        "Early stopping: monitor=%s mode=%s patience=%s min_delta=%s restore_best=%s",
        effective_early_stopping_monitor,
        effective_early_stopping_mode,
        config.early_stopping_patience,
        config.early_stopping_min_delta,
        config.restore_best_weights,
    )
    logger.info(
        "Hyperparameter-grid selection: level=%s metric=%s",
        config.classification_level,
        config.selection_metric,
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
    logger.info(
        "V-REx: enabled=%s weight=%s",
        model_config.get("use_vrex", False),
        model_config.get("vrex_penalty_weight", 0.0),
    )

    # The only search dimension handed to loso_cv is a list of already-expanded
    # complete configurations.  This guarantees that all and only the full
    # Cartesian combinations above are evaluated in every LOSO fold.
    builder = partial(
        _build_sic_from_explicit_grid,
        input_shape=tuple(X.shape[1:]),
    )
    builder._sequence_hyperparameter_depths = {}
    explicit_grid = {SIC_EXPLICIT_GRID_KEY: grid_records}

    results = loso_cv(
        model_builder_function=builder,
        feature_array=X,
        label_array=y,
        subject_id_array=subjects,
        trial_id_array=trials,
        n_epochs=config.source_epochs,
        batch_size=config.source_batch_size,
        hyperparameters=explicit_grid,
        evaluation_level=config.classification_level,
        selection_metric=config.selection_metric,
        selection_level=config.classification_level,
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
        log_variational_intervals=config.log_variational_intervals,
        n_prediction_latent_samples=config.prediction_latent_samples,
        latent_sampling_seed=config.latent_sampling_seed,
        n_uncertainty_samples=config.uncertainty_samples,
        validation_subjects_per_fold=config.validation_subjects,
        validation_seed=config.validation_seed,
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_min_delta=config.early_stopping_min_delta,
        early_stopping_monitor=effective_early_stopping_monitor,
        early_stopping_mode=effective_early_stopping_mode,
        restore_best_weights=config.restore_best_weights,
        decision_thresholds=(config.decision_threshold,),
        threshold_selection_metric=config.selection_metric,
        threshold_selection_level=config.classification_level,
        verbose=config.verbose,
        extra_fit_kwargs={
            "callbacks": [
                tf.keras.callbacks.TerminateOnNaN(),
            ]
        },
        n_jobs=config.n_jobs,
        gpu_ids=config.gpu_ids,
        cpus_per_worker=config.cpus_per_worker,
        max_folds=config.max_subjects,
        alternate_subject_sets=False,
        use_mldg=False,
    )

    _write_json(run_dir / "sic_loso_validation_results.json", results)
    logger.info("Saved normal-validation SIC artifacts to %s", run_dir)
    return {"run_dir": str(run_dir), "results": results}


def train_sic(
    feature_array,
    label_array,
    subject_id_array,
    trial_id_array,
    config: SICTrainingConfig,
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

    fixed_configurations, grid_dimensions = expand_cartesian_grid(model_config)
    if grid_dimensions:
        raise ValueError(
            "subject_calibration cannot select hyperparameters without a "
            "subject-disjoint validation set. Run the grid with "
            "--training-protocol loso_validation first; grid axes were "
            f"{sorted(grid_dimensions)}."
        )
    model_config = fixed_configurations[0]

    _write_json(run_dir / "training_config.json", asdict(config))
    _write_json(run_dir / "model_config.json", model_config)

    logger.info("Model: SIC (Subject Invariant Calibrator)")
    logger.info("SIC builder API version: %s", SIC_BUILDER_API_VERSION)
    logger.info("Input shape: %s", X.shape)
    logger.info(
        "Protocol: %d-fold, %d-trial calibration; zero-shot + post-calibration metrics",
        config.calibration_folds,
        config.calibration_trials,
    )
    logger.info(
        "Calibration head depth: %s",
        model_config.get("calibration_unfreeze_layers", 1),
    )
    logger.info(
        "Source generalization: V-REx enabled=%s penalty_weight=%s; subject_adversarial=%s",
        model_config.get("use_vrex", False),
        model_config.get("vrex_penalty_weight", 0.0),
        model_config.get("use_subject_adversarial", True),
    )

    builder = partial(build_sic_model, input_shape=tuple(X.shape[1:]))

    results = subject_calibration_cv(
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
        calibration_learning_rate=config.calibration_learning_rate,
        calibration_optimizer=config.calibration_optimizer,
        calibration_weight_decay=config.calibration_weight_decay,
        calibration_seed=config.calibration_seed,
        stratify_calibration=config.stratify_calibration,
        evaluation_level=config.classification_level,
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
        decision_threshold=config.decision_threshold,
        n_prediction_latent_samples=config.prediction_latent_samples,
        latent_sampling_seed=config.latent_sampling_seed,
        log_predictions=True,
        log_variational_intervals=config.log_variational_intervals,
        n_uncertainty_samples=config.uncertainty_samples,
        source_use_class_weight=config.source_use_class_weight,
        calibration_use_class_weight=config.calibration_use_class_weight,
        source_fit_kwargs={"callbacks": [tf.keras.callbacks.TerminateOnNaN()]},
        calibration_fit_kwargs={"callbacks": [tf.keras.callbacks.TerminateOnNaN()]},
        verbose=config.verbose,
        n_jobs=config.n_jobs,
        gpu_ids=config.gpu_ids,
        cpus_per_worker=config.cpus_per_worker,
        max_subjects=config.max_subjects,
    )

    _write_json(run_dir / "sic_calibration_results.json", results)
    _write_json(run_dir / "sic_overall_metrics.json", results.get("overall", {}))
    _write_csv(run_dir / "sic_subject_summary.csv", _subject_summary_rows(results))
    _write_csv(run_dir / "sic_calibration_folds.csv", _calibration_fold_rows(results))
    _write_csv(
        run_dir / "sic_trial_predictions.csv",
        list(results.get("trial_prediction_log", [])),
    )
    if results.get("trial_variational_interval_log"):
        _write_csv(
            run_dir / "sic_trial_uncertainty.csv",
            list(results["trial_variational_interval_log"]),
        )

    overall = results.get("overall", {})
    logger.info(
        "Zero-shot all-target mean: %s", overall.get("zero_shot_all_trials_mean_scores")
    )
    logger.info(
        "Paired zero-shot mean: %s", overall.get("paired_zero_shot_mean_scores")
    )
    logger.info("Post-calibration mean: %s", overall.get("calibrated_mean_scores"))
    logger.info("Calibration delta mean: %s", overall.get("delta_mean_scores"))
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
            "Train SIC with serial or GCN-GRU/BiLSTM feature-fusion encoder, "
            "with VAE reconstruction, optional V-REx and subject adversity, "
            "VC target, and three-fold six-trial subject calibration."
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
            "subject_calibration runs SIC pretraining + six-trial calibration; "
            "loso_validation runs ordinary ERM/V-REx LOSO with subject-disjoint validation."
        ),
    )
    parser.add_argument("--source-epochs", type=_positive_int, default=100)
    parser.add_argument("--source-batch-size", type=_positive_int, default=8)
    parser.add_argument("--validation-subjects", type=int, default=4)
    parser.add_argument("--validation-seed", type=int, default=42)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    parser.add_argument("--early-stopping-monitor", default="val_loss")
    parser.add_argument(
        "--best-epoch-metric",
        choices=("accuracy", "balanced_accuracy"),
        default=None,
        help=(
            "Select/restores the best LOSO-validation epoch using the requested "
            "window metric. 'accuracy' monitors val_window_accuracy; "
            "'balanced_accuracy' monitors val_window_balanced_accuracy. "
            "When omitted, --early-stopping-monitor/--early-stopping-mode are used."
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
        ),
        default="balanced_accuracy",
        help=(
            "Validation metric used to select the best hyperparameter-grid "
            "configuration. The selection level is set by "
            "--classification-level. This is separate from "
            "--best-epoch-metric, which controls checkpoint/epoch selection."
        ),
    )
    parser.add_argument(
        "--early-stopping-mode",
        choices=("auto", "min", "max"),
        default="min",
    )
    parser.add_argument("--no-restore-best-weights", action="store_true")
    parser.add_argument("--calibration-epochs", type=_positive_int, default=30)
    parser.add_argument("--calibration-batch-size", type=_positive_int, default=6)
    parser.add_argument("--calibration-trials", type=_positive_int, default=6)
    parser.add_argument("--calibration-folds", type=_positive_int, default=3)
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
    parser.add_argument("--source-use-class-weight", action="store_true")
    parser.add_argument("--calibration-use-class-weight", action="store_true")

    parser.add_argument("--n-jobs", type=_positive_int, default=1)
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=None)
    parser.add_argument("--cpus-per-worker", type=_positive_int, default=None)
    parser.add_argument("--max-subjects", type=_positive_int, default=None)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
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
            "force any JSON value to remain fixed. subject_calibration accepts "
            "only one fixed configuration."
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
    if args.validation_subjects < 0:
        raise ValueError("--validation-subjects must be >= 0.")
    if args.training_protocol == "loso_validation" and args.validation_subjects < 1:
        raise ValueError(
            "--training-protocol loso_validation requires at least one "
            "subject-disjoint validation subject."
        )
    if args.early_stopping_patience is not None and args.early_stopping_patience < 1:
        raise ValueError("--early-stopping-patience must be >= 1.")
    if args.early_stopping_min_delta < 0.0:
        raise ValueError("--early-stopping-min-delta must be non-negative.")
    if args.best_epoch_metric is not None and args.classification_level != "window":
        raise ValueError(
            "--best-epoch-metric requires --classification-level window."
        )
    if (
        args.calibration_trials * args.calibration_folds != 18
        and args.dataset == "dreamer"
    ):
        raise ValueError(
            "DREAMER SIC currently uses the complete 18-trial protocol: "
            "calibration_trials * calibration_folds must equal 18."
        )
    if args.calibration_learning_rate <= 0.0:
        raise ValueError("calibration-learning-rate must be positive.")
    if args.calibration_weight_decay < 0.0:
        raise ValueError("calibration-weight-decay must be non-negative.")
    if not 0.0 < args.decision_threshold < 1.0:
        raise ValueError("decision-threshold must be in (0, 1).")
    if args.prediction_latent_samples < 0:
        raise ValueError("prediction-latent-samples must be >= 0.")
    if not isinstance(model_config, dict):
        raise ValueError("--hyperparameters-json must decode to a JSON object.")
    if SIC_EXPLICIT_GRID_KEY in model_config:
        raise ValueError(
            f"{SIC_EXPLICIT_GRID_KEY!r} is reserved for the internal explicit-grid "
            "adapter."
        )
    if "epochs" in model_config or "batch_size" in model_config:
        raise ValueError(
            "Do not put epochs/batch_size inside --hyperparameters-json; SIC has "
            "separate source and calibration epoch/batch arguments."
        )
    _, grid_dimensions = expand_cartesian_grid(model_config)
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
    _validate_args(args, model_config)

    if args.print_grid_only:
        configurations, dimensions = expand_cartesian_grid(model_config)
        print(
            json.dumps(
                {
                    "search_type": "full_cartesian_product",
                    "selection_level": args.classification_level,
                    "selection_metric": args.selection_metric,
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

    X, y, subjects, trials = load_sic_training_data(
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
    )

    if args.classification_level == "trial":
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
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        early_stopping_monitor=args.early_stopping_monitor,
        early_stopping_mode=args.early_stopping_mode,
        best_epoch_metric=args.best_epoch_metric,
        selection_metric=args.selection_metric,
        restore_best_weights=not args.no_restore_best_weights,
        calibration_epochs=args.calibration_epochs,
        calibration_batch_size=args.calibration_batch_size,
        calibration_trials=args.calibration_trials,
        calibration_folds=args.calibration_folds,
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
        source_use_class_weight=args.source_use_class_weight,
        calibration_use_class_weight=args.calibration_use_class_weight,
        n_jobs=args.n_jobs,
        gpu_ids=None if args.gpu_ids is None else tuple(args.gpu_ids),
        cpus_per_worker=args.cpus_per_worker,
        max_subjects=args.max_subjects,
        verbose=args.verbose,
        seed=args.seed,
        label_threshold_mode=args.label_threshold_mode,
        window_normalization=args.window_normalization,
        model_config=model_config,
    )
    if args.training_protocol == "loso_validation":
        train_sic_loso_validation(X, y, subjects, trials, config)
    else:
        train_sic(X, y, subjects, trials, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
