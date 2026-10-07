"""Subject-set construction for LOSO folds and subject-calibration splits."""

from __future__ import annotations

from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import warnings

from .arrays import _as_numpy_1d, _python_scalar

def _balanced_two_subject_sets(
    subject_ids: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split fold-local subjects into two nearly equal, label-balanced sets."""
    subjects = np.asarray(subject_ids).reshape(-1)
    y_ids = _as_numpy_1d(labels).astype(np.int64)
    unique_subjects = np.sort(np.unique(subjects))
    if len(unique_subjects) < 2:
        raise ValueError("Alternating optimization requires at least two subjects.")

    rows = []
    for subject in unique_subjects:
        mask = subjects == subject
        subject_labels = y_ids[mask]
        rows.append(
            (
                subject,
                int(np.sum(mask)),
                float(np.mean(subject_labels == 1)) if len(subject_labels) else 0.0,
            )
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    rows.sort(key=lambda row: (row[2], row[1]), reverse=True)

    sets = [[], []]
    counts = [0, 0]
    positives = [0.0, 0.0]
    target_sizes = [
        len(unique_subjects) // 2,
        len(unique_subjects) - len(unique_subjects) // 2,
    ]
    for subject, count, positive_fraction in rows:
        candidates = [idx for idx in (0, 1) if len(sets[idx]) < target_sizes[idx]]
        choice = min(
            candidates,
            key=lambda idx: (
                positives[idx] / max(counts[idx], 1),
                counts[idx],
                len(sets[idx]),
            ),
        )
        sets[choice].append(subject)
        counts[choice] += count
        positives[choice] += positive_fraction * count

    return np.sort(np.asarray(sets[0])), np.sort(np.asarray(sets[1]))


def _class_weight_from_labels(labels: np.ndarray) -> dict[int, float] | None:
    """Return inverse-frequency class weights for one fitting partition."""
    y_ids = _as_numpy_1d(labels).astype(np.int64)
    classes, counts = np.unique(y_ids, return_counts=True)
    if len(classes) < 2:
        return None
    return {
        int(class_id): float(len(y_ids) / (len(classes) * count))
        for class_id, count in zip(classes, counts)
    }


def _subject_trial_labels(
    labels: np.ndarray,
    trial_ids: np.ndarray,
) -> dict:
    """Return one ground-truth class per trial, validating trial consistency."""
    y_ids = _as_numpy_1d(labels).astype(np.int64)
    trial_ids = np.asarray(trial_ids).reshape(-1)
    if len(y_ids) != len(trial_ids):
        raise ValueError(
            "labels and trial_ids must align when constructing calibration splits."
        )

    output: dict = {}
    for trial_id in np.unique(trial_ids):
        mask = trial_ids == trial_id
        unique_labels = np.unique(y_ids[mask])
        if len(unique_labels) != 1:
            raise ValueError(
                "Every target-subject trial must have exactly one class label. "
                f"Trial {trial_id!r} contains labels {unique_labels.tolist()}."
            )
        output[_python_scalar(trial_id)] = int(unique_labels[0])
    return output


def _make_subject_calibration_splits(
    labels: np.ndarray,
    trial_ids: np.ndarray,
    *,
    calibration_trials: int = 6,
    calibration_folds: int = 3,
    seed: int | None = 42,
    stratify: bool = True,
    allow_overlapping_folds: bool = False,
) -> list[dict]:
    """Create disjoint fixed-size calibration sets for one target subject.

    The v6 protocol is intentionally the reverse of ordinary K-fold CV: one
    fold (six DREAMER trials by default) is used for calibration and every
    remaining target-subject trial is used for evaluation. Across folds, each
    trial appears exactly once in calibration. When ``allow_overlapping_folds``
    is true and the subject does not retain exactly that many trials, each fold
    instead draws a fixed-size calibration subset and evaluates on every other
    retained trial. Calibration subsets may then overlap across folds.
    """
    if calibration_trials < 1:
        raise ValueError("calibration_trials must be at least 1.")
    if calibration_folds < 1:
        raise ValueError("calibration_folds must be at least 1.")
    if seed is not None and int(seed) < 0:
        raise ValueError("calibration seed must be >= 0 or None.")

    trial_ids = np.asarray(trial_ids).reshape(-1)
    unique_trials = np.asarray(sorted(np.unique(trial_ids).tolist()))
    required_trials = int(calibration_trials) * int(calibration_folds)
    exact_partition = len(unique_trials) == required_trials
    if not exact_partition and not allow_overlapping_folds:
        raise ValueError(
            "The complete calibration-partition protocol requires exactly "
            "calibration_trials * calibration_folds unique trials for every "
            f"target subject. Got {len(unique_trials)} trials, but "
            f"{calibration_trials} * {calibration_folds} = {required_trials}."
        )
    if not exact_partition and len(unique_trials) <= calibration_trials:
        raise ValueError(
            "Repeated calibration holdout requires more retained trials than "
            "calibration trials so that every fold has an evaluation set. Got "
            f"{len(unique_trials)} retained trials and "
            f"calibration_trials={calibration_trials}."
        )

    trial_label_map = _subject_trial_labels(labels, trial_ids)
    rng = np.random.default_rng(seed)
    fold_trials: list[list] = [[] for _ in range(calibration_folds)]

    if not exact_partition:
        trial_labels = np.asarray(
            [trial_label_map[_python_scalar(value)] for value in unique_trials]
        )
        split_indices: list[np.ndarray] = []
        if stratify:
            try:
                splitter = StratifiedShuffleSplit(
                    n_splits=calibration_folds,
                    train_size=calibration_trials,
                    random_state=seed,
                )
                split_indices = [
                    np.asarray(calibration_indices, dtype=np.int64)
                    for calibration_indices, _ in splitter.split(
                        unique_trials,
                        trial_labels,
                    )
                ]
            except ValueError as exc:
                warnings.warn(
                    "Could not stratify the retained trials for repeated "
                    f"calibration holdouts ({exc}); falling back to seeded "
                    "unstratified sampling.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if not split_indices:
            split_indices = [
                np.asarray(
                    rng.choice(
                        len(unique_trials),
                        size=calibration_trials,
                        replace=False,
                    ),
                    dtype=np.int64,
                )
                for _ in range(calibration_folds)
            ]
        fold_trials = [unique_trials[indices].tolist() for indices in split_indices]
    elif not stratify:
        shuffled = unique_trials.copy()
        rng.shuffle(shuffled)
        for fold_index in range(calibration_folds):
            start = fold_index * calibration_trials
            stop = start + calibration_trials
            fold_trials[fold_index] = shuffled[start:stop].tolist()
    else:
        labels_present = sorted(set(trial_label_map.values()))
        class_counts = [
            {label: 0 for label in labels_present}
            for _ in range(calibration_folds)
        ]
        # Fixed random tie-breakers make equal-cost assignments reproducible
        # without systematically favoring fold 0.
        tie_breakers = rng.random(calibration_folds)

        # Place rarer classes first; this maximizes the chance that every
        # calibration set contains minority-class examples when the subject's
        # label distribution permits it.
        grouped_trials: list[tuple[int, np.ndarray]] = []
        for class_label in labels_present:
            class_trial_ids = np.asarray(
                [
                    trial_id
                    for trial_id in unique_trials.tolist()
                    if trial_label_map[_python_scalar(trial_id)] == class_label
                ]
            )
            rng.shuffle(class_trial_ids)
            grouped_trials.append((class_label, class_trial_ids))
        grouped_trials.sort(key=lambda item: len(item[1]))

        for class_label, class_trial_ids in grouped_trials:
            for trial_id in class_trial_ids.tolist():
                candidates = [
                    fold_index
                    for fold_index in range(calibration_folds)
                    if len(fold_trials[fold_index]) < calibration_trials
                ]
                if not candidates:
                    raise RuntimeError(
                        "Calibration split construction exhausted fold capacity."
                    )
                chosen_fold = min(
                    candidates,
                    key=lambda fold_index: (
                        class_counts[fold_index][class_label],
                        len(fold_trials[fold_index]),
                        tie_breakers[fold_index],
                        fold_index,
                    ),
                )
                fold_trials[chosen_fold].append(trial_id)
                class_counts[chosen_fold][class_label] += 1

    output: list[dict] = []
    all_trials_set = set(_python_scalar(value) for value in unique_trials.tolist())
    for fold_index, calibration_trial_list in enumerate(fold_trials, start=1):
        if len(calibration_trial_list) != calibration_trials:
            raise RuntimeError(
                f"Calibration fold {fold_index} contains "
                f"{len(calibration_trial_list)} trials, expected "
                f"{calibration_trials}."
            )
        calibration_trial_list = sorted(
            _python_scalar(value) for value in calibration_trial_list
        )
        calibration_set = set(calibration_trial_list)
        evaluation_trial_list = sorted(all_trials_set - calibration_set)
        calibration_class_counts = {
            int(class_label): int(
                sum(trial_label_map[trial_id] == class_label for trial_id in calibration_trial_list)
            )
            for class_label in sorted(set(trial_label_map.values()))
        }
        evaluation_class_counts = {
            int(class_label): int(
                sum(trial_label_map[trial_id] == class_label for trial_id in evaluation_trial_list)
            )
            for class_label in sorted(set(trial_label_map.values()))
        }
        output.append(
            {
                "calibration_fold": int(fold_index),
                "partition_mode": (
                    "disjoint_complete_partition"
                    if exact_partition
                    else "repeated_holdout"
                ),
                "calibration_trial_ids": calibration_trial_list,
                "evaluation_trial_ids": evaluation_trial_list,
                "calibration_class_counts": calibration_class_counts,
                "evaluation_class_counts": evaluation_class_counts,
            }
        )

    if exact_partition:
        calibration_occurrences = [
            trial_id
            for row in output
            for trial_id in row["calibration_trial_ids"]
        ]
        if sorted(calibration_occurrences) != sorted(all_trials_set):
            raise RuntimeError(
                "Calibration folds must partition the target trials exactly once."
            )
    return output
