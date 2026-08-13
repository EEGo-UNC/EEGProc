"""Training and reporting entry point for SIC: Subject Invariant Calibrator.

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

SIC uses independent parallel deterministic branches. The complete GCN-GRU and
raw-EEG BiLSTM feature vectors are concatenated without an encoder projection or
bottleneck. Pooling changes only the temporal/window axes; fusion occurs in the
final classifier and subject-adversarial heads. There is no VAE, KL loss,
reconstruction loss, or decoder.

CLI construction, JSON/grid decoding, validation, and conversion to the runtime
configuration live in ``sic_model_args.py`` so this module stays focused on
data preparation, training, evaluation, and reporting.
"""

from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime
from functools import partial
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
    from .sic_model_args import (
        SIC_CLASSIFICATION_METRICS,
        SIC_DATA_HYPERPARAMETERS,
        SIC_SELECTION_OVERALL_KEYS,
        SICTrainingConfig,
        calibration_levels_from_args,
        calibration_plan as _calibration_plan,
        configuration_training_method as _configuration_training_method,
        expand_cartesian_grid,
        metric_mode as _metric_mode,
        parse_cli,
        selected_calibration_shots as _selected_calibration_shots,
        selection_score_source as _selection_score_source,
        split_data_hyperparameters as _split_data_hyperparameters,
        training_config_from_args,
    )
except ImportError:
    from sic_model_args import (
        SIC_CLASSIFICATION_METRICS,
        SIC_DATA_HYPERPARAMETERS,
        SIC_SELECTION_OVERALL_KEYS,
        SICTrainingConfig,
        calibration_levels_from_args,
        calibration_plan as _calibration_plan,
        configuration_training_method as _configuration_training_method,
        expand_cartesian_grid,
        metric_mode as _metric_mode,
        parse_cli,
        selected_calibration_shots as _selected_calibration_shots,
        selection_score_source as _selection_score_source,
        split_data_hyperparameters as _split_data_hyperparameters,
        training_config_from_args,
    )

try:
    from ...cross_val import subject_calibration_cv
except ImportError:
    from eegproc.deep_learning.cross_val import subject_calibration_cv

try:
    from ..joint_models_data import (
        build_joint_v2_dataset,
        get_dataset_config,
    )
except ImportError:
    from eegproc.deep_learning.joint_architectures.joint_models_data import (
        build_joint_v2_dataset,
        get_dataset_config,
    )


SIC_REMOVED_VAE_HYPERPARAMETERS = frozenset(
    {
        "use_decoder",
        "z_dim",
        "z_log_var_clip_min",
        "z_log_var_clip_max",
        "vae_loss_weight",
        "vae_beta",
        "decoder_dropout",
    }
)


def _deterministic_model_config(model_config: dict) -> tuple[dict, list[str]]:
    """Remove obsolete encoder-VAE axes before Cartesian expansion."""
    cleaned = dict(model_config)
    removed = sorted(SIC_REMOVED_VAE_HYPERPARAMETERS.intersection(cleaned))
    for key in removed:
        cleaned.pop(key)
    return cleaned, removed


def _json_fingerprint(value) -> str:
    """Return a stable, human-readable representation for logs/errors."""
    return json.dumps(value, sort_keys=True, default=_json_default)


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
    _write_json(
        configuration_dir / "loso_zero_shot_models.json",
        results.get("zero_shot_source_models", {}),
    )
    _write_csv(
        configuration_dir / "sic_subject_summary.csv",
        _subject_summary_rows(results),
    )
    _write_csv(
        configuration_dir / "sic_calibration_folds.csv",
        _calibration_fold_rows(results),
    )
    diagnostic_rows = list(results.get("prediction_diagnostics_log", []))
    if diagnostic_rows:
        _write_csv(
            configuration_dir / "sic_prediction_diagnostics.csv",
            diagnostic_rows,
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
    results_dir: Path,
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
        prediction_diagnostics=config.prediction_diagnostics,
        prediction_diagnostics_metric=config.prediction_diagnostics_metric,
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
        n_prediction_latent_samples=0,
        latent_sampling_seed=config.latent_sampling_seed,
        log_predictions=True,
        log_variational_intervals=False,
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
        source_model_output_dir=results_dir / "loso_zero_shot_models",
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

    model_config, removed_vae_parameters = _deterministic_model_config(
        config.model_config
    )
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
    if removed_vae_parameters:
        logger.info(
            "Removed obsolete deterministic-SIC VAE parameters: %s",
            removed_vae_parameters,
        )
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
                results_dir=configuration_dir,
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

    hyperparameters, removed_vae_parameters = _deterministic_model_config(
        config.model_config
    )
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
    if removed_vae_parameters:
        logger.info(
            "Removed obsolete deterministic-SIC VAE parameters: %s",
            removed_vae_parameters,
        )
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
        results_dir=run_dir,
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


def main(argv=None):
    args, model_config = parse_cli(argv)
    model_config, _ = _deterministic_model_config(model_config)
    calibration_levels = calibration_levels_from_args(args)
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

    config = training_config_from_args(
        args,
        model_config,
        calibration_levels,
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
