"""Aggregating window-level predictions up to trial level."""

from __future__ import annotations

import numpy as np

from .arrays import _as_numpy_1d, _python_scalar
from .probabilities import _predict_labels

def _direct_trial_aggregation(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    subject_ids: np.ndarray,
    trial_ids: np.ndarray,
    n_windows_per_trial: int,
    decision_threshold: float = 0.5,
) -> dict:
    """Build the trial-log structure when the model already predicts trials."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    y_true = _as_numpy_1d(y_true).astype(np.int64)
    subject_ids = np.asarray(subject_ids)
    trial_ids = np.asarray(trial_ids)
    lengths = (len(probabilities), len(y_true), len(subject_ids), len(trial_ids))
    if len(set(lengths)) != 1:
        raise ValueError(
            "Trial probabilities, labels, subject IDs, and trial IDs must "
            f"align; got lengths {lengths}."
        )
    return {
        "probabilities": probabilities,
        "y_true": y_true,
        "y_pred": _predict_labels(
            probabilities,
            decision_threshold=decision_threshold,
        ),
        "subject_ids": subject_ids,
        "trial_ids": trial_ids,
        "n_windows": np.full(len(y_true), int(n_windows_per_trial), dtype=np.int64),
        "window_indices": [],
    }


def _aggregate_window_probabilities_by_trial(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    subject_ids: np.ndarray,
    trial_ids: np.ndarray,
    decision_threshold: float = 0.5,
) -> dict:
    """Aggregate window probabilities into one prediction per subject/trial.

    The model is still trained and run at the window level. For evaluation, all
    window probabilities belonging to the same ``(subject_id, trial_id)`` are
    averaged, and the class with the highest mean probability becomes the trial
    prediction.

    Every window in one trial must have the same ground-truth label. Subject ID
    is included in the grouping key because trial numbers commonly repeat across
    subjects.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    y_true = _as_numpy_1d(y_true).astype(np.int64)
    subject_ids = np.asarray(subject_ids)
    trial_ids = np.asarray(trial_ids)

    n_windows = len(y_true)
    if probabilities.ndim != 2 or len(probabilities) != n_windows:
        raise ValueError(
            "probabilities must have shape (n_windows, n_classes) and align "
            f"with y_true. Got probabilities={probabilities.shape}, "
            f"n_windows={n_windows}."
        )
    if len(subject_ids) != n_windows or len(trial_ids) != n_windows:
        raise ValueError(
            "subject_ids and trial_ids must contain one value per window. "
            f"Got {len(subject_ids)} subjects, {len(trial_ids)} trials, "
            f"and {n_windows} labels."
        )

    grouped_indices: dict[tuple, list[int]] = {}
    for index, (subject_id, trial_id) in enumerate(zip(subject_ids, trial_ids)):
        key = (_python_scalar(subject_id), _python_scalar(trial_id))
        grouped_indices.setdefault(key, []).append(index)

    trial_probabilities: list[np.ndarray] = []
    trial_y_true: list[int] = []
    trial_subject_ids: list = []
    output_trial_ids: list = []
    trial_window_counts: list[int] = []
    trial_window_indices: list[np.ndarray] = []

    for (subject_id, trial_id), indices_list in grouped_indices.items():
        indices = np.asarray(indices_list, dtype=np.int64)
        labels = np.unique(y_true[indices])

        if len(labels) != 1:
            raise ValueError(
                "All windows in one trial must share one ground-truth label. "
                f"Subject {subject_id!r}, trial {trial_id!r} contains labels "
                f"{labels.tolist()}."
            )

        trial_probabilities.append(probabilities[indices].mean(axis=0))
        trial_y_true.append(int(labels[0]))
        trial_subject_ids.append(subject_id)
        output_trial_ids.append(trial_id)
        trial_window_counts.append(int(len(indices)))
        trial_window_indices.append(indices)

    trial_probabilities_array = np.stack(trial_probabilities, axis=0)

    return {
        "probabilities": trial_probabilities_array,
        "y_true": np.asarray(trial_y_true, dtype=np.int64),
        "y_pred": _predict_labels(
            trial_probabilities_array,
            decision_threshold=decision_threshold,
        ),
        "subject_ids": np.asarray(trial_subject_ids),
        "trial_ids": np.asarray(output_trial_ids),
        "n_windows": np.asarray(trial_window_counts, dtype=np.int64),
        "window_indices": trial_window_indices,
    }
