"""SIC array preparation and reporting shared by the training entry points."""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
import sys
import numpy as np
import tensorflow as tf

from ..joint_models_data import build_joint_v2_dataset, get_dataset_config
from .config import (
    SIC_CLASSIFICATION_METRICS, SIC_DATA_HYPERPARAMETERS, SIC_SELECTION_OVERALL_KEYS,
    calibration_plan as _calibration_plan,
    configuration_training_method as _configuration_training_method,
    expand_cartesian_grid, metric_mode as _metric_mode,
    selected_calibration_shots as _selected_calibration_shots,
    selection_score_source as _selection_score_source,
    split_data_hyperparameters as _split_data_hyperparameters,
)


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
    """Group windows without reordering them, returning one label per trial.

    The dataset builder emits each trial's windows chronologically.
    ``np.flatnonzero`` below returns matching positions in ascending source
    order, so the second output axis is the original EEG-window sequence that
    the trial GRU/BiGRU must learn from.
    """
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


def _reporting_rating_values(original_ratings, median_label):
    """Return stable raw-rating values to display, including absent 1--5 ratings."""
    if original_ratings is None:
        return []
    ratings = np.asarray(original_ratings, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(ratings)):
        raise ValueError("Original target ratings must all be finite.")
    observed = set(ratings.tolist())
    observed.add(float(median_label))
    if observed and all(
        float(value).is_integer() and 1.0 <= float(value) <= 5.0
        for value in observed
    ):
        return [float(value) for value in range(1, 6)]
    return sorted(float(value) for value in observed)


def _rating_count_key(value) -> str:
    value = float(value)
    return str(int(value)) if value.is_integer() else f"{value:g}"


def _trial_distribution_summary(
    labels,
    subjects,
    trials,
    original_ratings=None,
    *,
    rating_values=None,
):
    """Summarize labels and raw ratings once per unique subject/trial pair."""
    class_ids = _class_ids(labels).reshape(-1)
    subjects = np.asarray(subjects).reshape(-1)
    trials = np.asarray(trials).reshape(-1)
    ratings = (
        None
        if original_ratings is None
        else np.asarray(original_ratings, dtype=np.float64).reshape(-1)
    )
    lengths = [len(class_ids), len(subjects), len(trials)]
    if ratings is not None:
        lengths.append(len(ratings))
    if len(set(lengths)) != 1:
        raise ValueError(
            "Labels, subjects, trials, and original ratings must align before "
            f"trial-distribution reporting; got lengths={lengths}."
        )

    grouped_indices: dict[tuple, list[int]] = {}
    for index, (subject_id, trial_id) in enumerate(
        zip(subjects.tolist(), trials.tolist())
    ):
        grouped_indices.setdefault((subject_id, trial_id), []).append(index)

    records: list[dict] = []
    subject_order = []
    seen_subjects = set()
    for (subject_id, trial_id), indices in grouped_indices.items():
        indices_array = np.asarray(indices, dtype=np.int64)
        trial_classes = class_ids[indices_array]
        if np.any(trial_classes != trial_classes[0]):
            raise ValueError(
                f"Inconsistent class labels within subject={subject_id}, "
                f"trial={trial_id}."
            )
        rating = None
        if ratings is not None:
            trial_ratings = ratings[indices_array]
            if not np.allclose(
                trial_ratings,
                trial_ratings[0],
                rtol=0.0,
                atol=1e-8,
            ):
                raise ValueError(
                    f"Inconsistent original ratings within subject={subject_id}, "
                    f"trial={trial_id}."
                )
            rating = float(trial_ratings[0])
        records.append(
            {
                "subject": subject_id,
                "trial": trial_id,
                "class_id": int(trial_classes[0]),
                "original_rating": rating,
                "sample_count": int(len(indices)),
            }
        )
        if subject_id not in seen_subjects:
            seen_subjects.add(subject_id)
            subject_order.append(subject_id)

    observed_classes = sorted({record["class_id"] for record in records})
    if observed_classes and set(observed_classes).issubset({0, 1}):
        class_values = [0, 1]
    else:
        class_values = observed_classes
    if rating_values is None:
        rating_values = sorted(
            {
                record["original_rating"]
                for record in records
                if record["original_rating"] is not None
            }
        )
    rating_values = [float(value) for value in rating_values]

    def aggregate(selected_records):
        trial_count = int(len(selected_records))
        class_counts = {
            str(class_id): int(
                sum(record["class_id"] == class_id for record in selected_records)
            )
            for class_id in class_values
        }
        rating_counts = {
            _rating_count_key(rating): int(
                sum(
                    record["original_rating"] is not None
                    and np.isclose(
                        record["original_rating"],
                        rating,
                        rtol=0.0,
                        atol=1e-8,
                    )
                    for record in selected_records
                )
            )
            for rating in rating_values
        }
        if trial_count:
            majority_class, majority_count = min(
                class_counts.items(),
                key=lambda item: (-item[1], int(item[0])),
            )
            majority_baseline = float(majority_count / trial_count)
            majority_class = int(majority_class)
        else:
            majority_class = None
            majority_baseline = None
        return {
            "sample_count": int(
                sum(record["sample_count"] for record in selected_records)
            ),
            "trial_count": trial_count,
            "class_counts": class_counts,
            "original_rating_counts": rating_counts,
            "majority_class": majority_class,
            "majority_class_baseline_accuracy": majority_baseline,
        }

    summary = aggregate(records)
    per_subject = []
    for subject_id in subject_order:
        subject_records = [
            record for record in records if record["subject"] == subject_id
        ]
        subject_summary = aggregate(subject_records)
        subject_summary["subject"] = subject_id
        per_subject.append(subject_summary)
    summary["subject_count"] = int(len(subject_order))
    summary["per_subject"] = per_subject
    return summary


def _log_trial_distribution(logger, data_summary, *, context):
    """Print the overall and per-user trial distributions for one run/config."""
    before = data_summary["before_ablation"]
    retained = data_summary["after_ablation"]
    if data_summary["remove_median_label"]:
        logger.info(
            "%s before median removal: subjects=%d trials=%d samples=%d "
            "class_counts=%s original_rating_counts=%s "
            "majority_class=%s majority_class_baseline=%.4f",
            context,
            before["subject_count"],
            before["trial_count"],
            before["sample_count"],
            _json_fingerprint(before["class_counts"]),
            _json_fingerprint(before["original_rating_counts"]),
            before["majority_class"],
            before["majority_class_baseline_accuracy"],
        )
    logger.info(
        "%s retained trial data: subjects=%d trials=%d samples=%d "
        "class_counts=%s original_rating_counts=%s "
        "majority_class=%s majority_class_baseline=%.4f",
        context,
        retained["subject_count"],
        retained["trial_count"],
        retained["sample_count"],
        _json_fingerprint(retained["class_counts"]),
        _json_fingerprint(retained["original_rating_counts"]),
        retained["majority_class"],
        retained["majority_class_baseline_accuracy"],
    )
    for subject_summary in retained["per_subject"]:
        logger.info(
            "%s user=%s retained_trials=%d samples=%d class_counts=%s "
            "original_rating_counts=%s majority_class=%s "
            "majority_class_baseline=%.4f",
            context,
            subject_summary["subject"],
            subject_summary["trial_count"],
            subject_summary["sample_count"],
            _json_fingerprint(subject_summary["class_counts"]),
            _json_fingerprint(subject_summary["original_rating_counts"]),
            subject_summary["majority_class"],
            subject_summary["majority_class_baseline_accuracy"],
        )


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
    ratings = (
        None
        if original_ratings is None
        else np.asarray(original_ratings, dtype=np.float64).reshape(-1)
    )
    if remove_median_label and ratings is None:
        raise ValueError(
            "remove_median_label=true requires original target ratings from "
            "load_sic_training_data(return_original_ratings=True)."
        )
    if ratings is not None and any(len(array) != len(ratings) for array in arrays):
        raise ValueError(
            "Original ratings must align one-to-one with features, labels, "
            "subjects, and trials."
        )
    rating_values = _reporting_rating_values(ratings, median_label)
    before_summary = _trial_distribution_summary(
        arrays[1],
        arrays[2],
        arrays[3],
        ratings,
        rating_values=rating_values,
    )
    remove_mask = (
        np.isclose(
            ratings,
            float(median_label),
            rtol=0.0,
            atol=1e-8,
        )
        if remove_median_label
        else np.zeros(len(arrays[0]), dtype=bool)
    )
    keep_mask = ~remove_mask
    if not np.any(keep_mask):
        raise ValueError(
            f"Removing original rating {median_label:g} would empty the dataset."
        )
    filtered = tuple(array[keep_mask] for array in arrays)
    filtered_ratings = None if ratings is None else ratings[keep_mask]
    after_summary = _trial_distribution_summary(
        filtered[1],
        filtered[2],
        filtered[3],
        filtered_ratings,
        rating_values=rating_values,
    )
    return (
        *filtered,
        {
            "remove_median_label": bool(remove_median_label),
            "median_label": float(median_label),
            "removed_samples": int(np.sum(remove_mask)),
            "removed_trials": int(
                before_summary["trial_count"] - after_summary["trial_count"]
            ),
            "retained_samples": int(np.sum(keep_mask)),
            "retained_trials": int(after_summary["trial_count"]),
            "before_ablation": before_summary,
            "after_ablation": after_summary,
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
    config,
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
