from __future__ import annotations

import gc
import itertools
import multiprocessing as mp
import os
import queue
import traceback
from itertools import combinations
from pprint import pformat
from typing import Callable, Literal, Mapping

import numpy as np
import tensorflow as tf
from joblib.externals import cloudpickle
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score

_FIT_RESERVED_KEYS = frozenset({"epochs", "batch_size"})
_CLASSIFICATION_METRICS = frozenset({"accuracy", "f1", "precision", "recall"})

def _expand_hyperparameter_grid(hp: dict | None) -> list[dict]:
    """Expand a hyperparameter dict into a Cartesian-product grid."""
    if not hp:
        return [{}]

    keys = list(hp.keys())
    values = [v if isinstance(v, (list, tuple)) else [v] for v in hp.values()]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _normalize_fixed_hyperparameters(hp: dict | None) -> dict:
    """Normalize one fixed hyperparameter configuration.

    Plain LOSO evaluates a configuration that has already been selected; it
    does not perform an inner hyperparameter search. Scalar values are accepted
    directly. Singleton lists/tuples are also accepted for compatibility with
    JSON grids used by ``nested_lnso_cv``. Any parameter containing multiple
    candidate values is rejected to prevent accidental tuning on the LOSO test
    folds.
    """
    if not hp:
        return {}

    fixed: dict = {}

    for key, value in hp.items():
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError(
                    "loso_cv evaluates one fixed hyperparameter configuration. "
                    f"Parameter {key!r} contains {len(value)} candidates: "
                    f"{value!r}. Tune the configuration separately, then pass "
                    "one value per parameter."
                )
            fixed[key] = value[0]
        else:
            fixed[key] = value

    return fixed


def _split_config(config: dict) -> tuple[dict, dict]:
    """Split a flat config into model-builder kwargs and model.fit kwargs."""
    model_hp = {k: v for k, v in config.items() if k not in _FIT_RESERVED_KEYS}
    fit_hp = {k: v for k, v in config.items() if k in _FIT_RESERVED_KEYS}
    return model_hp, fit_hp


def _choose_best_config_index(
    mean_scores: list[dict],
    selection_metric: str,
    maximize_metric: bool,
) -> int:
    """Choose the best hyperparameter config from inner-CV mean scores."""
    metric_values = [scores[selection_metric] for scores in mean_scores]

    if maximize_metric:
        return int(np.argmax(metric_values))

    return int(np.argmin(metric_values))


def _apply_preprocessing_strategy(
    preprocessing_strategy: Callable | None,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
):
    """Apply optional preprocessing inside a CV fold.

    The preprocessing strategy is deliberately fold-local so that anything
    fit inside the strategy is fit only on the training partition.
    """
    if preprocessing_strategy is None:
        return X_train, y_train, X_eval, y_eval

    result = preprocessing_strategy(
        X_train,
        y_train,
        X_eval,
        y_eval,
        train_indices,
        eval_indices,
    )

    if not isinstance(result, tuple):
        raise ValueError(
            "preprocessing_strategy must return a tuple with either 2 or 4 values."
        )

    if len(result) == 2:
        X_train_processed, X_eval_processed = result
        return X_train_processed, y_train, X_eval_processed, y_eval

    if len(result) == 4:
        return result

    raise ValueError(
        "preprocessing_strategy must return either "
        "(X_train, X_eval) or (X_train, y_train, X_eval, y_eval)."
    )


def _as_numpy_1d(values: np.ndarray) -> np.ndarray:
    """Return labels as a 1D numpy array.

    Supports integer labels shaped (n,), binary labels shaped (n, 1), and
    one-hot labels shaped (n, n_classes).
    """
    values = np.asarray(values)

    if values.ndim == 1:
        return values

    if values.ndim == 2 and values.shape[1] == 1:
        return values[:, 0]

    if values.ndim == 2 and values.shape[1] > 1:
        return np.argmax(values, axis=1)

    raise ValueError(
        f"Expected labels with shape (n,), (n, 1), or (n, c). Got {values.shape}."
    )


def _to_probabilities(model_output: np.ndarray) -> np.ndarray:
    """Convert model output to class probabilities.

    Handles:
        - binary sigmoid probabilities/logits with shape (n,) or (n, 1)
        - multiclass softmax probabilities with shape (n, c)
        - multiclass logits with shape (n, c)
    """
    output = np.asarray(model_output)

    if output.ndim == 1:
        output = output.reshape(-1, 1)

    if output.ndim != 2:
        raise ValueError(
            f"Expected model output with shape (n,), (n, 1), or (n, c). Got {output.shape}."
        )

    if output.shape[1] == 1:
        p1 = output[:, 0].astype(np.float64)

        # If values are outside [0, 1], assume logits and sigmoid them.
        if np.any(p1 < 0.0) or np.any(p1 > 1.0):
            p1 = 1.0 / (1.0 + np.exp(-p1))

        p1 = np.clip(p1, 0.0, 1.0)
        p0 = 1.0 - p1
        return np.stack([p0, p1], axis=1)

    row_sums = output.sum(axis=1)

    # Already probabilities.
    if (
        np.all(output >= 0.0)
        and np.all(output <= 1.0)
        and np.allclose(row_sums, 1.0, atol=1e-4)
    ):
        return output.astype(np.float64)

    # Otherwise assume logits and softmax.
    shifted = output - np.max(output, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _predict_probabilities(model, X, batch_size=None):
    """Return class probabilities for Keras and scikit-learn models."""

    if hasattr(model, "predict_proba"):
        raw_pred = model.predict_proba(X)
    else:
        predict_kwargs = {"verbose": 0}

        if batch_size is not None:
            predict_kwargs["batch_size"] = batch_size

        raw_pred = model.predict(X, **predict_kwargs)

    # Joint and multi-output Keras models may return a dictionary.
    # For classification evaluation, extract the classifier output.
    if isinstance(raw_pred, Mapping):
        if "probabilities" in raw_pred:
            raw_pred = raw_pred["probabilities"]
        elif "logits" in raw_pred:
            raw_pred = raw_pred["logits"]
        else:
            raise ValueError(
                "Model.predict() returned a dictionary, but it did not contain "
                "'logits' or 'probabilities'. "
                f"Available outputs: {list(raw_pred.keys())}"
            )

    return _to_probabilities(raw_pred)


def _predict_labels(probabilities: np.ndarray) -> np.ndarray:
    """Convert probabilities to integer class predictions."""
    return np.argmax(probabilities, axis=1).astype(np.int64)


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: list[str] | tuple[str, ...],
    n_classes: int,
) -> dict:
    """Compute selected classification metrics across all expected classes.

    F1, precision, and recall use macro averaging so every class contributes
    equally. ``n_classes`` must come from the model output width rather than
    from the labels observed in one fold, because a validation/test fold may
    contain no examples of one of the expected classes.
    """
    y_true = _as_numpy_1d(y_true).astype(np.int64)
    y_pred = _as_numpy_1d(y_pred).astype(np.int64)

    if n_classes < 2:
        raise ValueError(f"n_classes must be >= 2, got {n_classes}.")

    expected_labels = list(range(n_classes))

    if np.any(y_true < 0) or np.any(y_true >= n_classes):
        raise ValueError(
            f"y_true contains labels outside the expected range "
            f"[0, {n_classes - 1}]."
        )
    if np.any(y_pred < 0) or np.any(y_pred >= n_classes):
        raise ValueError(
            f"y_pred contains labels outside the expected range "
            f"[0, {n_classes - 1}]."
        )

    scores: dict[str, float] = {}

    for metric in metrics:
        if metric not in _CLASSIFICATION_METRICS:
            raise ValueError(
                f"Unsupported classification metric: {metric}. "
                f"Supported metrics: {sorted(_CLASSIFICATION_METRICS)}"
            )

        if metric == "accuracy":
            scores["accuracy"] = float(accuracy_score(y_true, y_pred))

        elif metric == "f1":
            scores["f1"] = float(
                f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    labels=expected_labels,
                    zero_division=0,
                )
            )

        elif metric == "precision":
            scores["precision"] = float(
                precision_score(
                    y_true,
                    y_pred,
                    average="macro",
                    labels=expected_labels,
                    zero_division=0,
                )
            )

        elif metric == "recall":
            scores["recall"] = float(
                recall_score(
                    y_true,
                    y_pred,
                    average="macro",
                    labels=expected_labels,
                    zero_division=0,
                )
            )

    return scores



def _aggregate_window_probabilities_by_trial(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    subject_ids: np.ndarray,
    trial_ids: np.ndarray,
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
        "y_pred": _predict_labels(trial_probabilities_array),
        "subject_ids": np.asarray(trial_subject_ids),
        "trial_ids": np.asarray(output_trial_ids),
        "n_windows": np.asarray(trial_window_counts, dtype=np.int64),
        "window_indices": trial_window_indices,
    }


def _probability_log_loss(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """Return multiclass log loss for a probability matrix."""
    y_true = _as_numpy_1d(y_true).astype(np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)

    if probabilities.ndim != 2:
        raise ValueError(
            f"Expected probabilities with shape (n, c), got {probabilities.shape}."
        )

    return float(
        log_loss(
            y_true,
            probabilities,
            labels=list(range(probabilities.shape[1])),
        )
    )


def _level_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    metrics: list[str] | tuple[str, ...],
) -> dict:
    """Compute loss and requested metrics for one evaluation level."""
    scores = {
        "loss": _probability_log_loss(y_true, probabilities),
    }
    scores.update(
        _classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            metrics=metrics,
            n_classes=probabilities.shape[1],
        )
    )
    return scores


def _prefix_scores(scores: dict, prefix: str) -> dict:
    """Prefix metric names, for example ``accuracy`` -> ``trial_accuracy``."""
    return {f"{prefix}_{key}": value for key, value in scores.items()}


def _validate_evaluation_level(level: str, parameter_name: str) -> None:
    """Validate a window/trial evaluation-level parameter."""
    if level not in {"window", "trial"}:
        raise ValueError(
            f"{parameter_name} must be 'window' or 'trial', got {level!r}."
        )


def _validate_processed_alignment(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    trial_ids: np.ndarray,
    partition_name: str,
) -> None:
    """Ensure fold-local preprocessing preserved window order and count."""
    lengths = (len(X), len(y), len(subject_ids), len(trial_ids))
    if len(set(lengths)) != 1:
        raise ValueError(
            f"Preprocessing changed the number of {partition_name} windows or "
            "misaligned labels/IDs. Window creation, removal, reordering, and "
            "resampling must occur before nested_lnso_cv. Got lengths "
            f"X/y/subject/trial={lengths}."
        )


def _keras_loss_value(
    model: tf.keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int | None = None,
) -> float:
    """Evaluate and return the Keras loss value."""
    eval_output = model.evaluate(
        X,
        y,
        batch_size=batch_size,
        verbose=0,
        return_dict=True,
    )

    if "loss" not in eval_output:
        raise ValueError(
            f"model.evaluate(..., return_dict=True) did not return 'loss': {eval_output}"
        )

    return float(eval_output["loss"])


def _make_prediction_log(
    fold_index: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    subject_ids: np.ndarray,
    trial_ids: np.ndarray,
) -> list[dict]:
    """Create one prediction-log row per evaluated window."""
    y_true = _as_numpy_1d(y_true).astype(np.int64)
    y_pred = _as_numpy_1d(y_pred).astype(np.int64)

    rows: list[dict] = []

    for i in range(len(y_true)):
        pred_class = int(y_pred[i])
        row = {
            "fold": int(fold_index),
            "window_index": int(i),
            # Backwards-compatible alias retained for existing exports.
            "sample_index": int(i),
            "subject_id": _python_scalar(subject_ids[i]),
            "trial_id": _python_scalar(trial_ids[i]),
            "y_true": int(y_true[i]),
            "y_pred": pred_class,
            "p_pred": float(probabilities[i, pred_class]),
        }

        for class_idx in range(probabilities.shape[1]):
            row[f"p_class_{class_idx}"] = float(probabilities[i, class_idx])

        rows.append(row)

    return rows


def _make_trial_prediction_log(
    fold_index: int,
    trial_aggregation: dict,
) -> list[dict]:
    """Create one prediction-log row per evaluated subject/trial."""
    rows: list[dict] = []
    probabilities = trial_aggregation["probabilities"]
    y_true = trial_aggregation["y_true"]
    y_pred = trial_aggregation["y_pred"]

    for i in range(len(y_true)):
        pred_class = int(y_pred[i])
        row = {
            "fold": int(fold_index),
            "trial_index": int(i),
            "subject_id": _python_scalar(trial_aggregation["subject_ids"][i]),
            "trial_id": _python_scalar(trial_aggregation["trial_ids"][i]),
            "n_windows": int(trial_aggregation["n_windows"][i]),
            "y_true": int(y_true[i]),
            "y_pred": pred_class,
            "p_pred": float(probabilities[i, pred_class]),
        }

        for class_idx in range(probabilities.shape[1]):
            row[f"p_class_{class_idx}"] = float(probabilities[i, class_idx])

        rows.append(row)

    return rows


def _extract_classifier_output(raw_output):
    """Extract classifier logits/probabilities from a model call or prediction."""
    if isinstance(raw_output, Mapping):
        if "probabilities" in raw_output:
            return raw_output["probabilities"]
        if "logits" in raw_output:
            return raw_output["logits"]
        raise ValueError(
            "Model output dictionary did not contain 'logits' or "
            f"'probabilities'. Available outputs: {list(raw_output.keys())}"
        )

    if isinstance(raw_output, (tuple, list)):
        return raw_output[0]

    return raw_output


def _make_variational_interval_logs(
    model: tf.keras.Model,
    X: np.ndarray,
    y_true: np.ndarray,
    subject_ids: np.ndarray,
    trial_ids: np.ndarray,
    fold_index: int,
    n_uncertainty_samples: int = 30,
    ci_level: float = 0.95,
) -> tuple[list[dict], list[dict]]:
    """Estimate stochastic probability intervals for windows and trials.

    Trial intervals are calculated correctly by first averaging window
    probabilities within each trial for every stochastic forward pass, then
    taking quantiles across those trial-level samples.
    """
    if n_uncertainty_samples < 2:
        raise ValueError(
            "n_uncertainty_samples must be >= 2 when interval logging is enabled."
        )

    y_true = _as_numpy_1d(y_true).astype(np.int64)
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)

    probability_samples: list[np.ndarray] = []

    for _ in range(n_uncertainty_samples):
        raw_output = _extract_classifier_output(model(X_tensor, training=True))
        if hasattr(raw_output, "numpy"):
            raw_output = raw_output.numpy()
        probability_samples.append(_to_probabilities(raw_output))

    window_samples = np.stack(probability_samples, axis=0)
    window_mean = window_samples.mean(axis=0)

    alpha = 1.0 - ci_level
    window_low = np.quantile(window_samples, alpha / 2.0, axis=0)
    window_high = np.quantile(window_samples, 1.0 - alpha / 2.0, axis=0)
    window_pred = _predict_labels(window_mean)

    window_rows: list[dict] = []
    for i in range(len(y_true)):
        pred_class = int(window_pred[i])
        row = {
            "fold": int(fold_index),
            "window_index": int(i),
            "sample_index": int(i),
            "subject_id": _python_scalar(subject_ids[i]),
            "trial_id": _python_scalar(trial_ids[i]),
            "y_true": int(y_true[i]),
            "y_pred": pred_class,
            "p_pred_mean": float(window_mean[i, pred_class]),
            "p_pred_ci_low": float(window_low[i, pred_class]),
            "p_pred_ci_high": float(window_high[i, pred_class]),
            "ci_level": float(ci_level),
            "n_uncertainty_samples": int(n_uncertainty_samples),
        }
        for class_idx in range(window_mean.shape[1]):
            row[f"p_class_{class_idx}_mean"] = float(window_mean[i, class_idx])
            row[f"p_class_{class_idx}_ci_low"] = float(window_low[i, class_idx])
            row[f"p_class_{class_idx}_ci_high"] = float(window_high[i, class_idx])
        window_rows.append(row)

    reference_aggregation = _aggregate_window_probabilities_by_trial(
        probabilities=window_mean,
        y_true=y_true,
        subject_ids=subject_ids,
        trial_ids=trial_ids,
    )

    trial_sample_list: list[np.ndarray] = []
    for sample_index in range(window_samples.shape[0]):
        trial_sample_list.append(
            np.stack(
                [
                    window_samples[sample_index, indices].mean(axis=0)
                    for indices in reference_aggregation["window_indices"]
                ],
                axis=0,
            )
        )

    trial_samples = np.stack(trial_sample_list, axis=0)
    trial_mean = trial_samples.mean(axis=0)
    trial_low = np.quantile(trial_samples, alpha / 2.0, axis=0)
    trial_high = np.quantile(trial_samples, 1.0 - alpha / 2.0, axis=0)
    trial_pred = _predict_labels(trial_mean)

    trial_rows: list[dict] = []
    for i in range(len(reference_aggregation["y_true"])):
        pred_class = int(trial_pred[i])
        row = {
            "fold": int(fold_index),
            "trial_index": int(i),
            "subject_id": _python_scalar(reference_aggregation["subject_ids"][i]),
            "trial_id": _python_scalar(reference_aggregation["trial_ids"][i]),
            "n_windows": int(reference_aggregation["n_windows"][i]),
            "y_true": int(reference_aggregation["y_true"][i]),
            "y_pred": pred_class,
            "p_pred_mean": float(trial_mean[i, pred_class]),
            "p_pred_ci_low": float(trial_low[i, pred_class]),
            "p_pred_ci_high": float(trial_high[i, pred_class]),
            "ci_level": float(ci_level),
            "n_uncertainty_samples": int(n_uncertainty_samples),
        }
        for class_idx in range(trial_mean.shape[1]):
            row[f"p_class_{class_idx}_mean"] = float(trial_mean[i, class_idx])
            row[f"p_class_{class_idx}_ci_low"] = float(trial_low[i, class_idx])
            row[f"p_class_{class_idx}_ci_high"] = float(trial_high[i, class_idx])
        trial_rows.append(row)

    return window_rows, trial_rows

def _python_scalar(value):
    """Convert numpy scalars to plain Python scalars for logs/JSON."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _print_fold_header(fold_number: int, total_folds: int, description: str) -> None:
    """Print a readable progress line for the current fold."""
    print(f"\n[Fold {fold_number:>3} / {total_folds}] {description}")


def _print_config(title: str, config: dict) -> None:
    """Pretty-print a config dict without terminal truncation."""
    print(title)
    print(pformat(config, indent=4, width=120, sort_dicts=False))


def _print_metric_row(title: str, row: dict) -> None:
    """Pretty-print a metric row."""
    print("\n" + title)
    print("-" * len(title))

    for key, value in row.items():
        if isinstance(value, float):
            print(f"{key:>24}: {value:.6f}")
        else:
            print(f"{key:>24}: {value}")


def _print_user_metrics(user_metric_rows: list[dict]) -> None:
    """Print compact per-user metrics."""
    print("\nPer-user metrics")
    print("-" * 100)

    for row in user_metric_rows:
        parts = [
            f"fold={row['fold']}",
            f"subject={row['subject_id']}",
            f"n={row['n_samples']}",
        ]

        for key, value in row.items():
            if key in {"fold", "subject_id", "n_samples"}:
                continue
            if isinstance(value, float):
                parts.append(f"{key}={value:.6f}")
            else:
                parts.append(f"{key}={value}")

        print("  " + "  ".join(parts))


def _mean_std_rows(rows: list[dict], metric_names: list[str]) -> tuple[dict, dict]:
    """Compute mean/std for selected metric fields across row dicts."""
    mean_scores: dict[str, float] = {}
    std_scores: dict[str, float] = {}

    for metric_name in metric_names:
        values = [row[metric_name] for row in rows if metric_name in row]

        if not values:
            continue

        mean_scores[metric_name] = float(np.mean(values))
        std_scores[metric_name] = float(np.std(values))

    return mean_scores, std_scores


# ---------------------------------------------------------------------
# Fold evaluation
# ---------------------------------------------------------------------


def _evaluate_classification_fold(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    subject_ids_test: np.ndarray,
    trial_ids_test: np.ndarray,
    fold_index: int,
    metrics: list[str] | tuple[str, ...],
    evaluation_level: Literal["window", "trial"] = "trial",
    batch_size: int | None = None,
    log_predictions: bool = True,
    log_variational_intervals: bool = False,
    n_uncertainty_samples: int = 30,
    ci_level: float = 0.95,
) -> dict:
    """Evaluate one outer fold at both window and trial levels."""
    _validate_evaluation_level(evaluation_level, "evaluation_level")
    y_true_window = _as_numpy_1d(y_test).astype(np.int64)

    probabilities_window = _predict_probabilities(
        model=model,
        X=X_test,
        batch_size=batch_size,
    )
    y_pred_window = _predict_labels(probabilities_window)

    # model.evaluate() is retained as a diagnostic because joint Keras models
    # may include reconstruction/regularization terms beyond classification.
    keras_model_loss = _keras_loss_value(
        model=model,
        X=X_test,
        y=y_test,
        batch_size=batch_size,
    )

    window_scores = _level_scores(
        y_true=y_true_window,
        y_pred=y_pred_window,
        probabilities=probabilities_window,
        metrics=metrics,
    )

    trial_aggregation = _aggregate_window_probabilities_by_trial(
        probabilities=probabilities_window,
        y_true=y_true_window,
        subject_ids=subject_ids_test,
        trial_ids=trial_ids_test,
    )
    trial_scores = _level_scores(
        y_true=trial_aggregation["y_true"],
        y_pred=trial_aggregation["y_pred"],
        probabilities=trial_aggregation["probabilities"],
        metrics=metrics,
    )

    primary_scores = trial_scores if evaluation_level == "trial" else window_scores
    fold_scores = {
        "fold": int(fold_index),
        "evaluation_level": evaluation_level,
        "n_samples": int(
            len(trial_aggregation["y_true"])
            if evaluation_level == "trial"
            else len(y_true_window)
        ),
        "n_windows": int(len(y_true_window)),
        "n_trials": int(len(trial_aggregation["y_true"])),
        "keras_model_loss": float(keras_model_loss),
        **primary_scores,
        **_prefix_scores(window_scores, "window"),
        **_prefix_scores(trial_scores, "trial"),
    }

    window_fold_metrics = {
        "fold": int(fold_index),
        "n_windows": int(len(y_true_window)),
        "keras_model_loss": float(keras_model_loss),
        **window_scores,
    }
    trial_fold_metrics = {
        "fold": int(fold_index),
        "n_trials": int(len(trial_aggregation["y_true"])),
        **trial_scores,
    }

    user_rows: list[dict] = []
    for subject_id in np.unique(subject_ids_test):
        window_mask = subject_ids_test == subject_id
        trial_mask = trial_aggregation["subject_ids"] == subject_id

        user_window_scores = _level_scores(
            y_true=y_true_window[window_mask],
            y_pred=y_pred_window[window_mask],
            probabilities=probabilities_window[window_mask],
            metrics=metrics,
        )
        user_trial_scores = _level_scores(
            y_true=trial_aggregation["y_true"][trial_mask],
            y_pred=trial_aggregation["y_pred"][trial_mask],
            probabilities=trial_aggregation["probabilities"][trial_mask],
            metrics=metrics,
        )
        user_primary_scores = (
            user_trial_scores if evaluation_level == "trial" else user_window_scores
        )

        user_rows.append(
            {
                "fold": int(fold_index),
                "subject_id": _python_scalar(subject_id),
                "evaluation_level": evaluation_level,
                "n_samples": int(trial_mask.sum() if evaluation_level == "trial" else window_mask.sum()),
                "n_windows": int(window_mask.sum()),
                "n_trials": int(trial_mask.sum()),
                **user_primary_scores,
                **_prefix_scores(user_window_scores, "window"),
                **_prefix_scores(user_trial_scores, "trial"),
            }
        )

    window_prediction_rows: list[dict] = []
    trial_prediction_rows: list[dict] = []
    if log_predictions:
        window_prediction_rows = _make_prediction_log(
            fold_index=fold_index,
            y_true=y_true_window,
            y_pred=y_pred_window,
            probabilities=probabilities_window,
            subject_ids=subject_ids_test,
            trial_ids=trial_ids_test,
        )
        trial_prediction_rows = _make_trial_prediction_log(
            fold_index=fold_index,
            trial_aggregation=trial_aggregation,
        )

    window_interval_rows: list[dict] = []
    trial_interval_rows: list[dict] = []
    if log_variational_intervals:
        window_interval_rows, trial_interval_rows = _make_variational_interval_logs(
            model=model,
            X=X_test,
            y_true=y_true_window,
            subject_ids=subject_ids_test,
            trial_ids=trial_ids_test,
            fold_index=fold_index,
            n_uncertainty_samples=n_uncertainty_samples,
            ci_level=ci_level,
        )

    _print_metric_row(
        title=f"Fold {fold_index} metrics ({evaluation_level} primary)",
        row=fold_scores,
    )
    _print_user_metrics(user_rows)

    return {
        "fold_metrics": fold_scores,
        "window_fold_metrics": window_fold_metrics,
        "trial_fold_metrics": trial_fold_metrics,
        "user_metrics": user_rows,
        # Backwards-compatible name: prediction_log remains window-level.
        "prediction_log": window_prediction_rows,
        "window_prediction_log": window_prediction_rows,
        "trial_prediction_log": trial_prediction_rows,
        # Backwards-compatible name: variational_interval_log remains window-level.
        "variational_interval_log": window_interval_rows,
        "window_variational_interval_log": window_interval_rows,
        "trial_variational_interval_log": trial_interval_rows,
    }


def _evaluate_inner_config(
    model: tf.keras.Model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    subject_ids_val: np.ndarray,
    trial_ids_val: np.ndarray,
    metrics: list[str] | tuple[str, ...],
    selection_level: Literal["window", "trial"] = "trial",
    batch_size: int | None = None,
) -> dict:
    """Evaluate an inner fold at both levels and expose selection-level scores."""
    _validate_evaluation_level(selection_level, "selection_level")
    y_true_window = _as_numpy_1d(y_val).astype(np.int64)
    probabilities_window = _predict_probabilities(model, X_val, batch_size=batch_size)
    y_pred_window = _predict_labels(probabilities_window)

    window_scores = _level_scores(
        y_true=y_true_window,
        y_pred=y_pred_window,
        probabilities=probabilities_window,
        metrics=metrics,
    )
    window_scores["keras_model_loss"] = _keras_loss_value(
        model, X_val, y_val, batch_size=batch_size
    )

    trial_aggregation = _aggregate_window_probabilities_by_trial(
        probabilities=probabilities_window,
        y_true=y_true_window,
        subject_ids=subject_ids_val,
        trial_ids=trial_ids_val,
    )
    trial_scores = _level_scores(
        y_true=trial_aggregation["y_true"],
        y_pred=trial_aggregation["y_pred"],
        probabilities=trial_aggregation["probabilities"],
        metrics=metrics,
    )

    primary_scores = trial_scores if selection_level == "trial" else window_scores
    return {
        **{key: primary_scores[key] for key in ["loss", *metrics]},
        **_prefix_scores(window_scores, "window"),
        **_prefix_scores(trial_scores, "trial"),
        "selection_level": selection_level,
        "n_val_windows": int(len(y_true_window)),
        "n_val_trials": int(len(trial_aggregation["y_true"])),
    }


# ---------------------------------------------------------------------
# Concurrent outer-fold execution
# ---------------------------------------------------------------------


def _resolve_cuda_device_token(gpu_id: int) -> str:
    """Resolve a local GPU index to the token inherited by a child process.

    Slurm commonly sets ``CUDA_VISIBLE_DEVICES`` to physical ordinals or GPU
    UUIDs. Public ``gpu_ids`` are interpreted as local indices into that visible
    list, so ``gpu_ids=(0, 1)`` always means the first and second GPUs allocated
    to the job rather than physical devices 0 and 1 on the node.
    """
    gpu_id = int(gpu_id)
    if gpu_id < 0:
        raise ValueError(f"GPU indices must be non-negative, got {gpu_id}.")

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is None:
        return str(gpu_id)

    tokens = [token.strip() for token in visible_devices.split(",") if token.strip()]
    if not tokens or tokens == ["-1"]:
        raise ValueError(
            "gpu_ids were supplied, but CUDA_VISIBLE_DEVICES disables all GPUs."
        )
    if gpu_id >= len(tokens):
        raise ValueError(
            f"Requested local GPU index {gpu_id}, but CUDA_VISIBLE_DEVICES="
            f"{visible_devices!r} exposes only {len(tokens)} device(s)."
        )

    return tokens[gpu_id]


def _count_visible_gpus() -> int:
    """Return the number of GPUs visible to the current Slurm/job process."""
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None:
        tokens = [
            token.strip()
            for token in visible_devices.split(",")
            if token.strip()
        ]
        if not tokens or tokens == ["-1"]:
            return 0
        return len(tokens)

    # Outside Slurm, fall back to TensorFlow's physical-device discovery.
    return len(tf.config.list_physical_devices("GPU"))


def _auto_assign_gpu_ids(n_workers: int) -> tuple[int, ...] | None:
    """Assign one local visible GPU to each worker when GPUs are available."""
    visible_gpu_count = _count_visible_gpus()
    if visible_gpu_count == 0:
        return None

    if n_workers > visible_gpu_count:
        print(
            f"Requested {n_workers} workers, but only {visible_gpu_count} GPU(s) "
            "are visible. Reducing the worker count to one worker per GPU.",
            flush=True,
        )

    return tuple(range(min(n_workers, visible_gpu_count)))


def _start_device_bound_process(
    context,
    target: Callable,
    target_args_prefix: tuple,
    requested_gpu_id: int | None,
    cpus_per_worker: int | None,
    name: str,
) -> mp.Process:
    """Start one spawned process with its GPU mask set before TensorFlow import.

    ``spawn`` launches a fresh interpreter that imports this module. Temporarily
    changing the parent's environment around ``Process.start`` ensures the child
    sees only its assigned GPU before importing TensorFlow. Inside a GPU-bound
    worker that device is therefore always worker-local GPU 0.
    """
    previous_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

    if requested_gpu_id is None:
        child_cuda_visible_devices = "-1"
        worker_local_gpu_id = None
        assigned_device_label = "CPU"
    else:
        cuda_token = _resolve_cuda_device_token(requested_gpu_id)
        child_cuda_visible_devices = cuda_token
        worker_local_gpu_id = 0
        assigned_device_label = (
            f"GPU {int(requested_gpu_id)} "
            f"(CUDA_VISIBLE_DEVICES={cuda_token})"
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = child_cuda_visible_devices
    try:
        process = context.Process(
            target=target,
            args=(
                *target_args_prefix,
                worker_local_gpu_id,
                cpus_per_worker,
                assigned_device_label,
            ),
            name=name,
        )
        process.start()
    finally:
        if previous_cuda_visible_devices is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous_cuda_visible_devices

    return process


def _configure_tensorflow_worker(
    gpu_id: int | None,
    cpus_per_worker: int | None,
    assigned_device_label: str | None = None,
) -> None:
    """Configure TensorFlow before a worker constructs any model.

    A GPU worker is started with a one-device ``CUDA_VISIBLE_DEVICES`` mask, so
    ``gpu_id`` is normally 0 inside that process. A CPU worker is started with
    ``CUDA_VISIBLE_DEVICES=-1``. This prevents every process from probing or
    allocating memory on every GPU in a multi-GPU Slurm allocation.
    """
    if cpus_per_worker is not None:
        if cpus_per_worker < 1:
            raise ValueError("cpus_per_worker must be >= 1 when provided.")

        tf.config.threading.set_intra_op_parallelism_threads(cpus_per_worker)
        tf.config.threading.set_inter_op_parallelism_threads(1)

    physical_gpus = tf.config.list_physical_devices("GPU")

    if gpu_id is None:
        tf.config.set_visible_devices([], "GPU")
        logical_gpus = tf.config.list_logical_devices("GPU")
        if logical_gpus:
            raise RuntimeError(
                "CPU-only worker still has visible logical GPUs after TensorFlow "
                "configuration."
            )
        device_description = assigned_device_label or "CPU"
    else:
        gpu_id = int(gpu_id)

        if gpu_id < 0 or gpu_id >= len(physical_gpus):
            raise ValueError(
                f"Worker requested local GPU index {gpu_id}, but TensorFlow sees "
                f"{len(physical_gpus)} GPU(s). The child process should have been "
                "started with exactly one assigned CUDA device."
            )

        selected_gpu = physical_gpus[gpu_id]
        tf.config.set_visible_devices(selected_gpu, "GPU")
        tf.config.experimental.set_memory_growth(selected_gpu, True)

        logical_gpus = tf.config.list_logical_devices("GPU")
        if len(logical_gpus) != 1:
            raise RuntimeError(
                "A GPU worker must see exactly one logical GPU after isolation; "
                f"TensorFlow sees {len(logical_gpus)}."
            )

        device_description = assigned_device_label or f"GPU {gpu_id}"

    print(
        f"[{mp.current_process().name}] initialized on {device_description}",
        flush=True,
    )


def _collect_spawned_results(
    result_queue,
    processes: list[mp.Process],
    expected_results: int,
    worker_description: str,
) -> list[dict]:
    """Collect fold results without hanging forever after a worker crash."""
    outputs_by_fold: dict[int, dict] = {}

    while len(outputs_by_fold) < expected_results:
        try:
            status, fold_number, payload = result_queue.get(timeout=1.0)
        except queue.Empty:
            failed_processes = [
                process
                for process in processes
                if process.exitcode not in (None, 0)
            ]
            if failed_processes:
                failures = ", ".join(
                    f"{process.name} exitcode={process.exitcode}"
                    for process in failed_processes
                )
                raise RuntimeError(
                    f"A spawned {worker_description} process exited without "
                    f"returning a Python traceback: {failures}. This commonly "
                    "indicates an OS-level kill, CUDA failure, or out-of-memory "
                    "condition."
                )

            if all(process.exitcode is not None for process in processes):
                missing = expected_results - len(outputs_by_fold)
                raise RuntimeError(
                    f"All spawned {worker_description} processes exited, but "
                    f"{missing} fold result(s) were never returned."
                )
            continue

        if status == "error":
            location = (
                f" while running fold {fold_number}"
                if fold_number >= 0
                else " during TensorFlow worker initialization"
            )
            raise RuntimeError(
                f"A spawned {worker_description} worker failed{location}.\n\n"
                f"{payload}"
            )

        if status != "ok":
            raise RuntimeError(
                f"Unknown worker status {status!r} from fold {fold_number}."
            )

        if fold_number in outputs_by_fold:
            raise RuntimeError(
                f"Received duplicate result for fold {fold_number}."
            )

        outputs_by_fold[int(fold_number)] = payload

    return [outputs_by_fold[index] for index in sorted(outputs_by_fold)]


def _run_spawned_fold_pool(
    worker_target: Callable,
    worker_state: dict,
    tasks: list[tuple],
    n_workers: int,
    gpu_ids: tuple[int, ...] | None,
    cpus_per_worker: int | None,
    worker_name_prefix: str,
    worker_description: str,
) -> list[dict]:
    """Run fold tasks using persistent, device-isolated spawned workers."""
    context = mp.get_context("spawn")
    task_queue = context.Queue()
    result_queue = context.Queue()
    processes: list[mp.Process] = []
    completed_successfully = False

    try:
        try:
            worker_state_payload = cloudpickle.dumps(worker_state)
        except BaseException as exc:
            raise RuntimeError(
                "Could not serialize the cross-validation worker state. "
                "The model builder, preprocessing strategy, callbacks, and "
                "captured configuration must be cloudpickle-serializable."
            ) from exc

        payload_size_mb = len(worker_state_payload) / (1024 ** 2)
        if payload_size_mb >= 256.0:
            print(
                f"Warning: serialized worker state is {payload_size_mb:.1f} MiB. "
                "Each spawned worker will hold its own host-memory copy.",
                flush=True,
            )

        for worker_index in range(n_workers):
            requested_gpu_id = (
                gpu_ids[worker_index] if gpu_ids is not None else None
            )
            process = _start_device_bound_process(
                context=context,
                target=worker_target,
                target_args_prefix=(worker_state_payload, task_queue, result_queue),
                requested_gpu_id=requested_gpu_id,
                cpus_per_worker=cpus_per_worker,
                name=f"{worker_name_prefix}-{worker_index + 1}",
            )
            processes.append(process)

        for task in tasks:
            task_queue.put(task)

        for _ in processes:
            task_queue.put(None)

        outputs = _collect_spawned_results(
            result_queue=result_queue,
            processes=processes,
            expected_results=len(tasks),
            worker_description=worker_description,
        )
        completed_successfully = True
        return outputs
    finally:
        if not completed_successfully:
            for process in processes:
                if process.is_alive():
                    process.terminate()

        for process in processes:
            process.join(timeout=10.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)

        for multiprocessing_queue in (task_queue, result_queue):
            try:
                if not completed_successfully:
                    multiprocessing_queue.cancel_join_thread()
                multiprocessing_queue.close()
            except (AttributeError, OSError, ValueError):
                pass


def _outer_fold_process_main(
    worker_state_payload: bytes,
    task_queue,
    result_queue,
    gpu_id: int | None,
    cpus_per_worker: int | None,
    assigned_device_label: str | None,
) -> None:
    """Run outer-fold tasks in one persistent spawned process."""
    try:
        _configure_tensorflow_worker(
            gpu_id=gpu_id,
            cpus_per_worker=cpus_per_worker,
            assigned_device_label=assigned_device_label,
        )
        worker_state = cloudpickle.loads(worker_state_payload)

        while True:
            task = task_queue.get()

            if task is None:
                return

            outer_fold_number, outer_test_subjects = task

            try:
                fold_output = _run_outer_fold(
                    outer_fold_number=outer_fold_number,
                    outer_test_subjects=np.asarray(outer_test_subjects),
                    **worker_state,
                )
                result_queue.put(("ok", int(outer_fold_number), fold_output))
            except BaseException:
                result_queue.put(
                    (
                        "error",
                        int(outer_fold_number),
                        traceback.format_exc(),
                    )
                )
                return

    except BaseException:
        result_queue.put(("error", -1, traceback.format_exc()))
    finally:
        tf.keras.backend.clear_session()
        gc.collect()


def _run_outer_fold(
    outer_fold_number: int,
    outer_test_subjects: np.ndarray,
    total_outer_folds: int,
    model_builder_function: Callable[..., tf.keras.Model],
    feature_array: np.ndarray,
    label_array: np.ndarray,
    subject_id_array: np.ndarray,
    trial_id_array: np.ndarray,
    n_inner_subjects_to_leave_out: int,
    grid_configs: list[dict],
    batch_size: int,
    preprocessing_strategy: Callable | None,
    selection_metric: str,
    selection_level: Literal["window", "trial"],
    evaluation_level: Literal["window", "trial"],
    maximize_metric: bool,
    metrics: tuple[str, ...],
    log_predictions: bool,
    log_variational_intervals: bool,
    n_uncertainty_samples: int,
    ci_level: float,
    verbose: int,
    extra_fit_kwargs: dict,
) -> dict:
    """Run one complete outer fold, including its inner grid search."""
    outer_test_mask = np.isin(subject_id_array, outer_test_subjects)
    outer_train_mask = ~outer_test_mask

    outer_train_indices = np.where(outer_train_mask)[0]
    outer_test_indices = np.where(outer_test_mask)[0]

    outer_train_subject_ids = subject_id_array[outer_train_indices]
    unique_outer_train_subjects = np.sort(np.unique(outer_train_subject_ids))

    inner_subject_splits = list(
        combinations(unique_outer_train_subjects, n_inner_subjects_to_leave_out)
    )

    _print_fold_header(
        outer_fold_number,
        total_outer_folds,
        f"outer test subjects={outer_test_subjects.tolist()} "
        f"(outer_train={len(outer_train_indices)}, "
        f"outer_test={len(outer_test_indices)} windows)",
    )

    inner_scores_by_config: list[list[dict]] = [[] for _ in grid_configs]
    inner_fold_results: list[dict] = []

    # -----------------------------------------------------------------
    # Inner CV: choose hyperparameters.
    # -----------------------------------------------------------------
    for inner_fold_number, inner_val_subjects in enumerate(
        inner_subject_splits,
        start=1,
    ):
        inner_val_subjects = np.asarray(inner_val_subjects)

        inner_val_mask_relative = np.isin(
            outer_train_subject_ids,
            inner_val_subjects,
        )
        inner_train_mask_relative = ~inner_val_mask_relative

        inner_train_indices = outer_train_indices[inner_train_mask_relative]
        inner_val_indices = outer_train_indices[inner_val_mask_relative]

        X_inner_train = feature_array[inner_train_indices]
        y_inner_train = label_array[inner_train_indices]
        X_inner_val = feature_array[inner_val_indices]
        y_inner_val = label_array[inner_val_indices]
        subject_ids_inner_train = subject_id_array[inner_train_indices]
        subject_ids_inner_val = subject_id_array[inner_val_indices]
        trial_ids_inner_train = trial_id_array[inner_train_indices]
        trial_ids_inner_val = trial_id_array[inner_val_indices]

        (
            X_inner_train,
            y_inner_train,
            X_inner_val,
            y_inner_val,
        ) = _apply_preprocessing_strategy(
            preprocessing_strategy=preprocessing_strategy,
            X_train=X_inner_train,
            y_train=y_inner_train,
            X_eval=X_inner_val,
            y_eval=y_inner_val,
            train_indices=inner_train_indices,
            eval_indices=inner_val_indices,
        )

        _validate_processed_alignment(
            X_inner_train, y_inner_train, subject_ids_inner_train,
            trial_ids_inner_train, "inner-training"
        )
        _validate_processed_alignment(
            X_inner_val, y_inner_val, subject_ids_inner_val,
            trial_ids_inner_val, "inner-validation"
        )

        config_results_this_inner_fold: list[dict] = []

        for config_index, config in enumerate(grid_configs):
            model_hp, fit_hp = _split_config(config)
            current_batch_size = fit_hp.get("batch_size", batch_size)

            tf.keras.backend.clear_session()
            model = model_builder_function(**model_hp)

            try:
                fit_kwargs = dict(fit_hp)
                fit_kwargs["validation_data"] = (X_inner_val, y_inner_val)

                model.fit(
                    X_inner_train,
                    y_inner_train,
                    verbose=verbose,
                    **fit_kwargs,
                    **extra_fit_kwargs,
                )

                val_scores = _evaluate_inner_config(
                    model=model,
                    X_val=X_inner_val,
                    y_val=y_inner_val,
                    subject_ids_val=subject_ids_inner_val,
                    trial_ids_val=trial_ids_inner_val,
                    metrics=metrics,
                    selection_level=selection_level,
                    batch_size=current_batch_size,
                )

                config_result = {
                    "outer_fold": int(outer_fold_number),
                    "inner_fold": int(inner_fold_number),
                    "left_out_subjects": inner_val_subjects.tolist(),
                    "config_index": int(config_index),
                    "config": dict(config),
                    **val_scores,
                }

                config_results_this_inner_fold.append(config_result)
                inner_scores_by_config[config_index].append(val_scores)

            finally:
                del model
                gc.collect()
                tf.keras.backend.clear_session()

        inner_fold_results.append(
            {
                "outer_fold": int(outer_fold_number),
                "inner_fold": int(inner_fold_number),
                "left_out_subjects": inner_val_subjects.tolist(),
                "n_train_windows": int(len(inner_train_indices)),
                "n_val_windows": int(len(inner_val_indices)),
                "n_train_trials": int(len(set(zip(
                    subject_ids_inner_train.tolist(), trial_ids_inner_train.tolist()
                )))),
                "n_val_trials": int(len(set(zip(
                    subject_ids_inner_val.tolist(), trial_ids_inner_val.tolist()
                )))),
                "configs": config_results_this_inner_fold,
            }
        )

    # -----------------------------------------------------------------
    # Aggregate inner-CV scores and choose the best configuration.
    # -----------------------------------------------------------------
    inner_mean_scores: list[dict] = []
    inner_std_scores: list[dict] = []
    score_metric_names = [
        "loss", *metrics, "window_loss", "window_keras_model_loss",
        *[f"window_{metric}" for metric in metrics],
        "trial_loss", *[f"trial_{metric}" for metric in metrics],
    ]

    for config_index, config in enumerate(grid_configs):
        mean_scores_for_config, std_scores_for_config = _mean_std_rows(
            inner_scores_by_config[config_index],
            score_metric_names,
        )

        inner_mean_scores.append(
            {
                "config_index": int(config_index),
                "config": dict(config),
                **mean_scores_for_config,
            }
        )
        inner_std_scores.append(
            {
                "config_index": int(config_index),
                "config": dict(config),
                **std_scores_for_config,
            }
        )

    best_config_index = _choose_best_config_index(
        mean_scores=inner_mean_scores,
        selection_metric=selection_metric,
        maximize_metric=maximize_metric,
    )
    best_config = grid_configs[best_config_index]

    print(
        f"\nBest config from inner CV for outer fold {outer_fold_number}: "
        f"{selection_metric}="
        f"{inner_mean_scores[best_config_index][selection_metric]:.6f}",
        flush=True,
    )
    _print_config("Best config:", best_config)

    best_config_result = {
        "outer_fold": int(outer_fold_number),
        "best_config_index": int(best_config_index),
        "best_config": dict(best_config),
        "selection_metric": selection_metric,
        "selection_level": selection_level,
        "selection_score": float(
            inner_mean_scores[best_config_index][selection_metric]
        ),
    }

    inner_cv_result = {
        "outer_fold": int(outer_fold_number),
        "inner_fold_results": inner_fold_results,
        "inner_mean_scores": inner_mean_scores,
        "inner_std_scores": inner_std_scores,
    }

    # -----------------------------------------------------------------
    # Final outer training and testing.
    # -----------------------------------------------------------------
    X_outer_train = feature_array[outer_train_indices]
    y_outer_train = label_array[outer_train_indices]
    X_outer_test = feature_array[outer_test_indices]
    y_outer_test = label_array[outer_test_indices]
    subject_ids_outer_train = subject_id_array[outer_train_indices]
    subject_ids_outer_test = subject_id_array[outer_test_indices]
    trial_ids_outer_train = trial_id_array[outer_train_indices]
    trial_ids_outer_test = trial_id_array[outer_test_indices]

    (
        X_outer_train,
        y_outer_train,
        X_outer_test,
        y_outer_test,
    ) = _apply_preprocessing_strategy(
        preprocessing_strategy=preprocessing_strategy,
        X_train=X_outer_train,
        y_train=y_outer_train,
        X_eval=X_outer_test,
        y_eval=y_outer_test,
        train_indices=outer_train_indices,
        eval_indices=outer_test_indices,
    )

    _validate_processed_alignment(
        X_outer_train, y_outer_train, subject_ids_outer_train,
        trial_ids_outer_train, "outer-training"
    )
    _validate_processed_alignment(
        X_outer_test, y_outer_test, subject_ids_outer_test,
        trial_ids_outer_test, "outer-test"
    )

    model_hp, fit_hp = _split_config(best_config)
    current_batch_size = fit_hp.get("batch_size", batch_size)

    tf.keras.backend.clear_session()
    final_model = model_builder_function(**model_hp)

    try:
        final_model.fit(
            X_outer_train,
            y_outer_train,
            verbose=verbose,
            **fit_hp,
            **extra_fit_kwargs,
        )

        fold_result = _evaluate_classification_fold(
            model=final_model,
            X_test=X_outer_test,
            y_test=y_outer_test,
            subject_ids_test=subject_ids_outer_test,
            trial_ids_test=trial_ids_outer_test,
            fold_index=outer_fold_number,
            metrics=metrics,
            evaluation_level=evaluation_level,
            batch_size=current_batch_size,
            log_predictions=log_predictions,
            log_variational_intervals=log_variational_intervals,
            n_uncertainty_samples=n_uncertainty_samples,
            ci_level=ci_level,
        )

    finally:
        del final_model
        gc.collect()
        tf.keras.backend.clear_session()

    outer_fold_result = {
        "outer_fold_number": int(outer_fold_number),
        "left_out_subjects": outer_test_subjects.tolist(),
        "n_outer_train_windows": int(len(outer_train_indices)),
        "n_outer_test_windows": int(len(outer_test_indices)),
        "n_outer_train_trials": int(len(set(zip(
            subject_ids_outer_train.tolist(), trial_ids_outer_train.tolist()
        )))),
        "n_outer_test_trials": int(len(set(zip(
            subject_ids_outer_test.tolist(), trial_ids_outer_test.tolist()
        )))),
        "selection_level": selection_level,
        "evaluation_level": evaluation_level,
        "best_config": dict(best_config),
        "inner_fold_results": inner_fold_results,
        "inner_mean_scores": inner_mean_scores,
        "inner_std_scores": inner_std_scores,
        "fold_metrics": fold_result["fold_metrics"],
        "window_fold_metrics": fold_result["window_fold_metrics"],
        "trial_fold_metrics": fold_result["trial_fold_metrics"],
        "user_metrics": fold_result["user_metrics"],
        "prediction_log": fold_result["prediction_log"],
        "window_prediction_log": fold_result["window_prediction_log"],
        "trial_prediction_log": fold_result["trial_prediction_log"],
        "variational_interval_log": fold_result["variational_interval_log"],
        "window_variational_interval_log": fold_result["window_variational_interval_log"],
        "trial_variational_interval_log": fold_result["trial_variational_interval_log"],
    }

    return {
        "outer_fold_number": int(outer_fold_number),
        "best_config_result": best_config_result,
        "inner_cv_result": inner_cv_result,
        "outer_fold_result": outer_fold_result,
        **fold_result,
    }


# ---------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------


def nested_lnso_cv(
    model_builder_function: Callable[..., tf.keras.Model],
    feature_array: np.ndarray,
    label_array: np.ndarray,
    subject_id_array: np.ndarray,
    trial_id_array: np.ndarray | None = None,
    n_outer_subjects_to_leave_out: int = 1,
    n_inner_subjects_to_leave_out: int = 1,
    n_epochs: int = 50,
    batch_size: int = 32,
    hyperparameters: dict | None = None,
    preprocessing_strategy: Callable | None = None,
    selection_metric: str = "loss",
    selection_level: Literal["window", "trial"] = "trial",
    evaluation_level: Literal["window", "trial"] = "trial",
    maximize_metric: bool | None = None,
    metrics: list[str] | tuple[str, ...] = ("accuracy", "f1", "precision", "recall"),
    log_predictions: bool = True,
    log_variational_intervals: bool = False,
    n_uncertainty_samples: int = 30,
    ci_level: float = 0.95,
    verbose: int = 0,
    extra_fit_kwargs: dict | None = None,
    n_jobs: int = 1,
    gpu_ids: list[int] | tuple[int, ...] | None = None,
    cpus_per_worker: int | None = None,
) -> dict:
    """Run nested Leave-N-Subjects-Out CV.

    Outer folds can be executed concurrently using multiprocessing with the
    ``spawn`` start method. Inner folds and hyperparameter configurations remain
    sequential inside each worker, preventing nested process oversubscription.

    Trial-level evaluation
    ----------------------
    ``trial_id_array`` must contain one trial ID per input window. Predictions
    are made per window, then probabilities are averaged within each
    ``(subject_id, trial_id)`` group. ``selection_level`` controls inner-CV
    hyperparameter selection and ``evaluation_level`` controls the unprefixed
    primary metrics. Both window- and trial-level metrics/logs are always
    returned.

    Parameters added for concurrency
    --------------------------------
    n_jobs:
        Number of persistent outer-fold worker processes. ``1`` preserves the
        original sequential behavior unless ``gpu_ids`` is supplied.
    gpu_ids:
        Local GPU indices assigned one-per-worker. For example,
        ``gpu_ids=(0, 1, 2, 3)`` with ``n_jobs=4``. When this is ``None`` and
        multiple workers are requested, visible Slurm/TensorFlow GPUs are
        assigned automatically, one per worker.
    cpus_per_worker:
        TensorFlow intra-op CPU threads available to each worker. Keep
        ``n_jobs * cpus_per_worker`` within the CPUs allocated by Slurm.

    Notes
    -----
    Worker state is serialized with cloudpickle, so locally defined model
    builders and preprocessing callables are supported. The training entry
    point must still be protected by ``if __name__ == "__main__":``.
    """
    extra_fit_kwargs = extra_fit_kwargs or {}

    if "validation_data" in extra_fit_kwargs:
        raise ValueError(
            "Do not pass validation_data in extra_fit_kwargs. "
            "nested_lnso_cv creates validation_data from the inner folds."
        )

    if subject_id_array is None:
        raise ValueError("subject_id_array is required for nested LNSO CV.")
    if trial_id_array is None:
        raise ValueError(
            "trial_id_array is required for trial-level prediction and metrics. "
            "Pass one trial ID per window, aligned with feature_array."
        )

    _validate_evaluation_level(selection_level, "selection_level")
    _validate_evaluation_level(evaluation_level, "evaluation_level")

    feature_array = np.asarray(feature_array)
    label_array = np.asarray(label_array)
    subject_id_array = np.asarray(subject_id_array)
    trial_id_array = np.asarray(trial_id_array)

    input_lengths = (
        len(feature_array), len(label_array), len(subject_id_array), len(trial_id_array)
    )
    if len(set(input_lengths)) != 1:
        raise ValueError(
            "feature_array, label_array, subject_id_array, and trial_id_array "
            "must have the same first dimension. Got lengths "
            f"{input_lengths}."
        )

    metrics = tuple(metrics)

    for metric in metrics:
        if metric not in _CLASSIFICATION_METRICS:
            raise ValueError(
                f"Unsupported metric: {metric}. Supported metrics: "
                f"{sorted(_CLASSIFICATION_METRICS)}"
            )

    allowed_selection_metrics = {"loss", *metrics}

    if selection_metric not in allowed_selection_metrics:
        raise ValueError(
            f"selection_metric='{selection_metric}' is not available. "
            f"Use 'loss' or one of metrics={list(metrics)}."
        )

    if maximize_metric is None:
        maximize_metric = selection_metric != "loss"

    if not (0.0 < ci_level < 1.0):
        raise ValueError("ci_level must be between 0 and 1.")

    if n_jobs < 1:
        raise ValueError("n_jobs must be >= 1.")

    if cpus_per_worker is not None and cpus_per_worker < 1:
        raise ValueError("cpus_per_worker must be >= 1 when provided.")

    unique_subjects = np.sort(np.unique(subject_id_array))

    if n_outer_subjects_to_leave_out < 1:
        raise ValueError("n_outer_subjects_to_leave_out must be >= 1.")

    if n_inner_subjects_to_leave_out < 1:
        raise ValueError("n_inner_subjects_to_leave_out must be >= 1.")

    if n_outer_subjects_to_leave_out >= len(unique_subjects):
        raise ValueError(
            "n_outer_subjects_to_leave_out must be smaller than the number "
            f"of unique subjects. Got {n_outer_subjects_to_leave_out} for "
            f"{len(unique_subjects)} subjects."
        )

    n_outer_train_subjects = len(unique_subjects) - n_outer_subjects_to_leave_out

    if n_inner_subjects_to_leave_out >= n_outer_train_subjects:
        raise ValueError(
            "n_inner_subjects_to_leave_out must be smaller than the number "
            "of subjects available in each outer-training pool. Got "
            f"{n_inner_subjects_to_leave_out} for {n_outer_train_subjects} "
            "outer-training subjects."
        )

    if hyperparameters is None:
        hyperparameters = {}

    effective_hyperparameters = {
        "epochs": n_epochs,
        "batch_size": batch_size,
        **hyperparameters,
    }

    grid_configs = _expand_hyperparameter_grid(effective_hyperparameters)
    outer_subject_splits = list(
        combinations(unique_subjects, n_outer_subjects_to_leave_out)
    )
    total_outer_folds = len(outer_subject_splits)
    effective_n_jobs = min(n_jobs, total_outer_folds)

    normalized_gpu_ids: tuple[int, ...] | None = None

    if gpu_ids is None and effective_n_jobs > 1:
        normalized_gpu_ids = _auto_assign_gpu_ids(effective_n_jobs)
        if normalized_gpu_ids is not None:
            effective_n_jobs = len(normalized_gpu_ids)
    elif gpu_ids is not None:
        normalized_gpu_ids = tuple(int(gpu_id) for gpu_id in gpu_ids)

        if not normalized_gpu_ids:
            raise ValueError("gpu_ids must contain at least one GPU index.")
        if len(set(normalized_gpu_ids)) != len(normalized_gpu_ids):
            raise ValueError("gpu_ids must not contain duplicate GPU indices.")
        if effective_n_jobs > len(normalized_gpu_ids):
            raise ValueError(
                f"n_jobs={effective_n_jobs} requires at least that many GPU IDs, "
                f"but gpu_ids={normalized_gpu_ids}. Use one GPU per worker."
            )

        normalized_gpu_ids = normalized_gpu_ids[:effective_n_jobs]

    results = {
        "fold_metrics": [],
        "window_fold_metrics": [],
        "trial_fold_metrics": [],
        "user_metrics": [],
        "prediction_log": [],
        "window_prediction_log": [],
        "trial_prediction_log": [],
        "variational_interval_log": [],
        "window_variational_interval_log": [],
        "trial_variational_interval_log": [],
        "best_configs": [],
        "inner_cv_results": [],
        "outer_fold_results": [],
        "mean_scores": {},
        "std_scores": {},
        "window_mean_scores": {},
        "window_std_scores": {},
        "trial_mean_scores": {},
        "trial_std_scores": {},
    }

    print(
        f"\nNested LNSO CV — {total_outer_folds} outer folds, "
        f"{len(grid_configs)} hyperparameter config"
        f"{'s' if len(grid_configs) != 1 else ''}"
    )
    print(f"Requested metrics: {list(metrics)}")
    print(
        f"Selection metric: {selection_level}-level {selection_metric} "
        f"({'maximize' if maximize_metric else 'minimize'})"
    )
    print(f"Primary reported metrics: {evaluation_level}-level")
    print(f"Prediction logging: {log_predictions}")
    print(f"Variational interval logging: {log_variational_intervals}")
    print(f"Outer-fold workers: {effective_n_jobs}")

    if effective_n_jobs > 1 and normalized_gpu_ids is None:
        print("Worker devices: CPU-only")
    elif normalized_gpu_ids is not None:
        print(f"Worker devices: GPUs {list(normalized_gpu_ids)}")
    else:
        print("Worker device: current TensorFlow default")

    tasks = [
        (fold_number, tuple(outer_test_subjects))
        for fold_number, outer_test_subjects in enumerate(
            outer_subject_splits,
            start=1,
        )
    ]

    worker_state = {
        "total_outer_folds": total_outer_folds,
        "model_builder_function": model_builder_function,
        "feature_array": feature_array,
        "label_array": label_array,
        "subject_id_array": subject_id_array,
        "trial_id_array": trial_id_array,
        "n_inner_subjects_to_leave_out": n_inner_subjects_to_leave_out,
        "grid_configs": grid_configs,
        "batch_size": batch_size,
        "preprocessing_strategy": preprocessing_strategy,
        "selection_metric": selection_metric,
        "selection_level": selection_level,
        "evaluation_level": evaluation_level,
        "maximize_metric": bool(maximize_metric),
        "metrics": metrics,
        "log_predictions": log_predictions,
        "log_variational_intervals": log_variational_intervals,
        "n_uncertainty_samples": n_uncertainty_samples,
        "ci_level": ci_level,
        "verbose": verbose,
        "extra_fit_kwargs": extra_fit_kwargs,
    }

    # Preserve the old in-process behavior for n_jobs=1 with no explicit GPU
    # assignment. This avoids spawn/pickling overhead for ordinary runs.
    if effective_n_jobs == 1 and normalized_gpu_ids is None:
        fold_outputs = [
            _run_outer_fold(
                outer_fold_number=fold_number,
                outer_test_subjects=np.asarray(outer_test_subjects),
                **worker_state,
            )
            for fold_number, outer_test_subjects in tasks
        ]
    else:
        fold_outputs = _run_spawned_fold_pool(
            worker_target=_outer_fold_process_main,
            worker_state=worker_state,
            tasks=tasks,
            n_workers=effective_n_jobs,
            gpu_ids=normalized_gpu_ids,
            cpus_per_worker=cpus_per_worker,
            worker_name_prefix="OuterFoldWorker",
            worker_description="outer-fold",
        )

    # Results arrive in completion order, so restore deterministic fold order.
    fold_outputs.sort(key=lambda row: row["outer_fold_number"])

    for fold_output in fold_outputs:
        results["fold_metrics"].append(fold_output["fold_metrics"])
        results["window_fold_metrics"].append(fold_output["window_fold_metrics"])
        results["trial_fold_metrics"].append(fold_output["trial_fold_metrics"])
        results["user_metrics"].extend(fold_output["user_metrics"])
        results["prediction_log"].extend(fold_output["prediction_log"])
        results["window_prediction_log"].extend(fold_output["window_prediction_log"])
        results["trial_prediction_log"].extend(fold_output["trial_prediction_log"])
        results["variational_interval_log"].extend(
            fold_output["variational_interval_log"]
        )
        results["window_variational_interval_log"].extend(
            fold_output["window_variational_interval_log"]
        )
        results["trial_variational_interval_log"].extend(
            fold_output["trial_variational_interval_log"]
        )
        results["best_configs"].append(fold_output["best_config_result"])
        results["inner_cv_results"].append(fold_output["inner_cv_result"])
        results["outer_fold_results"].append(fold_output["outer_fold_result"])

    mean_scores, std_scores = _mean_std_rows(
        results["fold_metrics"],
        ["loss", *metrics],
    )

    results["mean_scores"] = mean_scores
    results["std_scores"] = std_scores

    window_mean_scores, window_std_scores = _mean_std_rows(
        results["window_fold_metrics"], ["loss", *metrics]
    )
    trial_mean_scores, trial_std_scores = _mean_std_rows(
        results["trial_fold_metrics"], ["loss", *metrics]
    )
    results["window_mean_scores"] = window_mean_scores
    results["window_std_scores"] = window_std_scores
    results["trial_mean_scores"] = trial_mean_scores
    results["trial_std_scores"] = trial_std_scores

    print("\nNested LNSO CV complete")
    print("=" * 80)
    print("Mean outer scores:")
    print(pformat(mean_scores, indent=4, width=120, sort_dicts=False))
    print("Std outer scores:")
    print(pformat(std_scores, indent=4, width=120, sort_dicts=False))
    print("Window-level mean scores:")
    print(pformat(window_mean_scores, indent=4, width=120, sort_dicts=False))
    print("Trial-level mean scores:")
    print(pformat(trial_mean_scores, indent=4, width=120, sort_dicts=False))

    return results


# ---------------------------------------------------------------------
# Plain Leave-One-Subject-Out cross-validation
# ---------------------------------------------------------------------


def _run_loso_fold(
    fold_number: int,
    test_subject,
    total_folds: int,
    model_builder_function: Callable[..., tf.keras.Model],
    feature_array: np.ndarray,
    label_array: np.ndarray,
    subject_id_array: np.ndarray,
    trial_id_array: np.ndarray,
    fixed_config: dict,
    batch_size: int,
    preprocessing_strategy: Callable | None,
    evaluation_level: Literal["window", "trial"],
    metrics: tuple[str, ...],
    log_predictions: bool,
    log_variational_intervals: bool,
    n_uncertainty_samples: int,
    ci_level: float,
    verbose: int,
    extra_fit_kwargs: dict,
) -> dict:
    """Train and evaluate one ordinary LOSO fold.

    No inner folds or hyperparameter selection occur here. ``fixed_config`` is
    applied unchanged in every fold.
    """
    test_mask = subject_id_array == test_subject
    train_mask = ~test_mask

    train_indices = np.where(train_mask)[0]
    test_indices = np.where(test_mask)[0]

    if len(train_indices) == 0 or len(test_indices) == 0:
        raise ValueError(
            f"Invalid LOSO split for subject {test_subject!r}: "
            f"train={len(train_indices)}, test={len(test_indices)} windows."
        )

    _print_fold_header(
        fold_number,
        total_folds,
        f"LOSO test subject={_python_scalar(test_subject)!r} "
        f"(train={len(train_indices)}, test={len(test_indices)} windows)",
    )

    X_train = feature_array[train_indices]
    y_train = label_array[train_indices]
    X_test = feature_array[test_indices]
    y_test = label_array[test_indices]

    subject_ids_train = subject_id_array[train_indices]
    subject_ids_test = subject_id_array[test_indices]
    trial_ids_train = trial_id_array[train_indices]
    trial_ids_test = trial_id_array[test_indices]

    X_train, y_train, X_test, y_test = _apply_preprocessing_strategy(
        preprocessing_strategy=preprocessing_strategy,
        X_train=X_train,
        y_train=y_train,
        X_eval=X_test,
        y_eval=y_test,
        train_indices=train_indices,
        eval_indices=test_indices,
    )

    _validate_processed_alignment(
        X_train,
        y_train,
        subject_ids_train,
        trial_ids_train,
        "LOSO-training",
    )
    _validate_processed_alignment(
        X_test,
        y_test,
        subject_ids_test,
        trial_ids_test,
        "LOSO-test",
    )

    model_hp, fit_hp = _split_config(fixed_config)
    current_batch_size = int(fit_hp.get("batch_size", batch_size))

    duplicate_fit_keys = set(fit_hp).intersection(extra_fit_kwargs)
    if duplicate_fit_keys:
        raise ValueError(
            "The following model.fit arguments were supplied in both the fixed "
            f"configuration and extra_fit_kwargs: {sorted(duplicate_fit_keys)}"
        )

    tf.keras.backend.clear_session()
    model = model_builder_function(**model_hp)

    try:
        model.fit(
            X_train,
            y_train,
            verbose=verbose,
            **fit_hp,
            **extra_fit_kwargs,
        )

        evaluation = _evaluate_classification_fold(
            model=model,
            X_test=X_test,
            y_test=y_test,
            subject_ids_test=subject_ids_test,
            trial_ids_test=trial_ids_test,
            fold_index=fold_number,
            metrics=metrics,
            evaluation_level=evaluation_level,
            batch_size=current_batch_size,
            log_predictions=log_predictions,
            log_variational_intervals=log_variational_intervals,
            n_uncertainty_samples=n_uncertainty_samples,
            ci_level=ci_level,
        )
    finally:
        del model
        gc.collect()
        tf.keras.backend.clear_session()

    fold_record = {
        "fold_number": int(fold_number),
        "outer_fold_number": int(fold_number),
        "left_out_subject": _python_scalar(test_subject),
        "left_out_subjects": [_python_scalar(test_subject)],
        "outer_test_subjects": [_python_scalar(test_subject)],
        "n_train_windows": int(len(train_indices)),
        "n_test_windows": int(len(test_indices)),
        # Compatibility aliases for code that previously consumed nested CV.
        "n_outer_train_windows": int(len(train_indices)),
        "n_outer_test_windows": int(len(test_indices)),
        "n_train_trials": int(
            len(
                set(
                    zip(
                        subject_ids_train.tolist(),
                        trial_ids_train.tolist(),
                    )
                )
            )
        ),
        "n_test_trials": int(
            len(
                set(
                    zip(
                        subject_ids_test.tolist(),
                        trial_ids_test.tolist(),
                    )
                )
            )
        ),
        "n_outer_train_trials": int(
            len(
                set(
                    zip(
                        subject_ids_train.tolist(),
                        trial_ids_train.tolist(),
                    )
                )
            )
        ),
        "n_outer_test_trials": int(
            len(
                set(
                    zip(
                        subject_ids_test.tolist(),
                        trial_ids_test.tolist(),
                    )
                )
            )
        ),
        "evaluation_level": evaluation_level,
        "selection_level": None,
        "fixed_config": dict(fixed_config),
        # Compatibility aliases: there was no inner search in plain LOSO.
        "best_config": dict(fixed_config),
        "inner_fold_results": [],
        "inner_mean_scores": [],
        "inner_std_scores": [],
        "fold_metrics": evaluation["fold_metrics"],
        "window_fold_metrics": evaluation["window_fold_metrics"],
        "trial_fold_metrics": evaluation["trial_fold_metrics"],
        "user_metrics": evaluation["user_metrics"],
        "prediction_log": evaluation["prediction_log"],
        "window_prediction_log": evaluation["window_prediction_log"],
        "trial_prediction_log": evaluation["trial_prediction_log"],
        "variational_interval_log": evaluation["variational_interval_log"],
        "window_variational_interval_log": evaluation[
            "window_variational_interval_log"
        ],
        "trial_variational_interval_log": evaluation[
            "trial_variational_interval_log"
        ],
    }

    fixed_config_record = {
        "outer_fold": int(fold_number),
        "best_config_index": 0,
        "best_config": dict(fixed_config),
        "selection_metric": None,
        "selection_level": None,
        "selection_score": None,
        "configuration_source": "fixed",
    }

    empty_inner_cv_record = {
        "outer_fold": int(fold_number),
        "inner_fold_results": [],
        "inner_mean_scores": [],
        "inner_std_scores": [],
        "configuration_source": "not_applicable_for_loso",
    }

    return {
        "outer_fold_number": int(fold_number),
        "fold_record": fold_record,
        "outer_fold_result": fold_record,
        "best_config_result": fixed_config_record,
        "inner_cv_result": empty_inner_cv_record,
        **evaluation,
    }


def _loso_fold_process_main(
    worker_state_payload: bytes,
    task_queue,
    result_queue,
    gpu_id: int | None,
    cpus_per_worker: int | None,
    assigned_device_label: str | None,
) -> None:
    """Run ordinary LOSO folds in one persistent spawned process."""
    try:
        _configure_tensorflow_worker(
            gpu_id=gpu_id,
            cpus_per_worker=cpus_per_worker,
            assigned_device_label=assigned_device_label,
        )
        worker_state = cloudpickle.loads(worker_state_payload)

        while True:
            task = task_queue.get()

            if task is None:
                return

            fold_number, test_subject = task

            try:
                fold_output = _run_loso_fold(
                    fold_number=fold_number,
                    test_subject=test_subject,
                    **worker_state,
                )
                result_queue.put(("ok", int(fold_number), fold_output))
            except BaseException:
                result_queue.put(
                    (
                        "error",
                        int(fold_number),
                        traceback.format_exc(),
                    )
                )
                return

    except BaseException:
        result_queue.put(("error", -1, traceback.format_exc()))
    finally:
        tf.keras.backend.clear_session()
        gc.collect()

def loso_cv(
    model_builder_function: Callable[..., tf.keras.Model],
    feature_array: np.ndarray,
    label_array: np.ndarray,
    subject_id_array: np.ndarray,
    trial_id_array: np.ndarray | None = None,
    n_epochs: int = 50,
    batch_size: int = 32,
    hyperparameters: dict | None = None,
    preprocessing_strategy: Callable | None = None,
    evaluation_level: Literal["window", "trial"] = "trial",
    metrics: list[str] | tuple[str, ...] = (
        "accuracy",
        "f1",
        "precision",
        "recall",
    ),
    log_predictions: bool = True,
    log_variational_intervals: bool = False,
    n_uncertainty_samples: int = 30,
    ci_level: float = 0.95,
    verbose: int = 0,
    extra_fit_kwargs: dict | None = None,
    n_jobs: int = 1,
    gpu_ids: list[int] | tuple[int, ...] | None = None,
    cpus_per_worker: int | None = None,
    max_folds: int | None = None,
) -> dict:
    """Run ordinary Leave-One-Subject-Out cross-validation.

    Each unique subject is held out exactly once. The model is trained on all
    remaining subjects and evaluated on the held-out subject. There is no inner
    cross-validation and no hyperparameter selection.

    Hyperparameters
    ---------------
    ``hyperparameters`` must describe one fixed configuration. Scalar values
    are accepted directly, and singleton lists/tuples are accepted for
    compatibility with existing JSON grids. Multiple candidate values are
    rejected because selecting among them using LOSO test results would bias
    the reported performance.

    Trial-level evaluation
    ----------------------
    ``trial_id_array`` must contain one trial ID per window. Predictions are
    made per window and averaged within each ``(subject_id, trial_id)`` group.
    Both window- and trial-level metrics are always returned;
    ``evaluation_level`` controls the unprefixed primary metrics.

    Concurrency
    -----------
    Outer LOSO folds may run concurrently. Worker state is serialized with
    cloudpickle, so locally defined builders are supported. When ``n_jobs > 1``
    and ``gpu_ids`` is omitted, visible GPUs are assigned automatically, one per
    worker. The training entry point must still use an
    ``if __name__ == "__main__":`` guard.

    Smoke testing
    -------------
    ``max_folds`` deterministically limits execution to the first N sorted
    subjects. Leave it as ``None`` for a complete LOSO evaluation.
    """
    extra_fit_kwargs = extra_fit_kwargs or {}

    if "validation_data" in extra_fit_kwargs:
        raise ValueError(
            "Do not pass a fixed validation_data array to loso_cv. It would not "
            "be reconstructed fold-locally and could create leakage."
        )

    if subject_id_array is None:
        raise ValueError("subject_id_array is required for LOSO CV.")
    if trial_id_array is None:
        raise ValueError(
            "trial_id_array is required for trial-level prediction and metrics. "
            "Pass one trial ID per window, aligned with feature_array."
        )

    _validate_evaluation_level(evaluation_level, "evaluation_level")

    feature_array = np.asarray(feature_array)
    label_array = np.asarray(label_array)
    subject_id_array = np.asarray(subject_id_array)
    trial_id_array = np.asarray(trial_id_array)

    input_lengths = (
        len(feature_array),
        len(label_array),
        len(subject_id_array),
        len(trial_id_array),
    )
    if len(set(input_lengths)) != 1:
        raise ValueError(
            "feature_array, label_array, subject_id_array, and trial_id_array "
            "must have the same first dimension. Got lengths "
            f"{input_lengths}."
        )

    metrics = tuple(metrics)
    for metric in metrics:
        if metric not in _CLASSIFICATION_METRICS:
            raise ValueError(
                f"Unsupported metric: {metric}. Supported metrics: "
                f"{sorted(_CLASSIFICATION_METRICS)}"
            )

    if not (0.0 < ci_level < 1.0):
        raise ValueError("ci_level must be between 0 and 1.")

    if n_jobs < 1:
        raise ValueError("n_jobs must be >= 1.")

    if cpus_per_worker is not None and cpus_per_worker < 1:
        raise ValueError("cpus_per_worker must be >= 1 when provided.")

    unique_subjects = np.sort(np.unique(subject_id_array))
    if len(unique_subjects) < 2:
        raise ValueError(
            "LOSO CV requires at least two unique subjects. "
            f"Got {len(unique_subjects)}."
        )

    if max_folds is not None:
        if max_folds < 1:
            raise ValueError("max_folds must be >= 1 when provided.")
        test_subjects = unique_subjects[: min(max_folds, len(unique_subjects))]
    else:
        test_subjects = unique_subjects

    fixed_hyperparameters = _normalize_fixed_hyperparameters(hyperparameters)
    fixed_config = {
        "epochs": n_epochs,
        "batch_size": batch_size,
        **fixed_hyperparameters,
    }

    total_folds = len(test_subjects)
    effective_n_jobs = min(n_jobs, total_folds)

    normalized_gpu_ids: tuple[int, ...] | None = None
    if gpu_ids is None and effective_n_jobs > 1:
        normalized_gpu_ids = _auto_assign_gpu_ids(effective_n_jobs)
        if normalized_gpu_ids is not None:
            effective_n_jobs = len(normalized_gpu_ids)
    elif gpu_ids is not None:
        normalized_gpu_ids = tuple(int(gpu_id) for gpu_id in gpu_ids)

        if not normalized_gpu_ids:
            raise ValueError("gpu_ids must contain at least one GPU index.")
        if len(set(normalized_gpu_ids)) != len(normalized_gpu_ids):
            raise ValueError("gpu_ids must not contain duplicate GPU indices.")
        if effective_n_jobs > len(normalized_gpu_ids):
            raise ValueError(
                f"n_jobs={effective_n_jobs} requires at least that many GPU IDs, "
                f"but gpu_ids={normalized_gpu_ids}. Use one GPU per worker."
            )

        normalized_gpu_ids = normalized_gpu_ids[:effective_n_jobs]

    results = {
        "cv_strategy": "loso",
        "fixed_config": dict(fixed_config),
        "n_subjects": int(len(unique_subjects)),
        "n_evaluated_folds": int(total_folds),
        "max_folds": max_folds,
        "fold_metrics": [],
        "window_fold_metrics": [],
        "trial_fold_metrics": [],
        "user_metrics": [],
        "prediction_log": [],
        "window_prediction_log": [],
        "trial_prediction_log": [],
        "variational_interval_log": [],
        "window_variational_interval_log": [],
        "trial_variational_interval_log": [],
        "fold_results": [],
        # Compatibility fields used by the previous nested-CV training script.
        "best_configs": [],
        "inner_cv_results": [],
        "outer_fold_results": [],
        "mean_scores": {},
        "std_scores": {},
        "window_mean_scores": {},
        "window_std_scores": {},
        "trial_mean_scores": {},
        "trial_std_scores": {},
    }

    print(f"\nLOSO CV — {total_folds} fold{'s' if total_folds != 1 else ''}")
    print(f"Total available subjects: {len(unique_subjects)}")
    if max_folds is not None:
        print(
            f"Smoke-test fold limit: {total_folds} of "
            f"{len(unique_subjects)} subjects"
        )
    _print_config("Fixed configuration:", fixed_config)
    print(f"Requested metrics: {list(metrics)}")
    print(f"Primary reported metrics: {evaluation_level}-level")
    print(f"Prediction logging: {log_predictions}")
    print(f"Variational interval logging: {log_variational_intervals}")
    print(f"Fold workers: {effective_n_jobs}")

    if effective_n_jobs > 1 and normalized_gpu_ids is None:
        print("Worker devices: CPU-only")
    elif normalized_gpu_ids is not None:
        print(f"Worker devices: GPUs {list(normalized_gpu_ids)}")
    else:
        print("Worker device: current TensorFlow default")

    tasks = [
        (fold_number, _python_scalar(test_subject))
        for fold_number, test_subject in enumerate(test_subjects, start=1)
    ]

    worker_state = {
        "total_folds": total_folds,
        "model_builder_function": model_builder_function,
        "feature_array": feature_array,
        "label_array": label_array,
        "subject_id_array": subject_id_array,
        "trial_id_array": trial_id_array,
        "fixed_config": fixed_config,
        "batch_size": batch_size,
        "preprocessing_strategy": preprocessing_strategy,
        "evaluation_level": evaluation_level,
        "metrics": metrics,
        "log_predictions": log_predictions,
        "log_variational_intervals": log_variational_intervals,
        "n_uncertainty_samples": n_uncertainty_samples,
        "ci_level": ci_level,
        "verbose": verbose,
        "extra_fit_kwargs": extra_fit_kwargs,
    }

    if effective_n_jobs == 1 and normalized_gpu_ids is None:
        fold_outputs = [
            _run_loso_fold(
                fold_number=fold_number,
                test_subject=test_subject,
                **worker_state,
            )
            for fold_number, test_subject in tasks
        ]
    else:
        fold_outputs = _run_spawned_fold_pool(
            worker_target=_loso_fold_process_main,
            worker_state=worker_state,
            tasks=tasks,
            n_workers=effective_n_jobs,
            gpu_ids=normalized_gpu_ids,
            cpus_per_worker=cpus_per_worker,
            worker_name_prefix="LOSOFoldWorker",
            worker_description="LOSO-fold",
        )

    fold_outputs.sort(key=lambda row: row["outer_fold_number"])

    for fold_output in fold_outputs:
        results["fold_metrics"].append(fold_output["fold_metrics"])
        results["window_fold_metrics"].append(fold_output["window_fold_metrics"])
        results["trial_fold_metrics"].append(fold_output["trial_fold_metrics"])
        results["user_metrics"].extend(fold_output["user_metrics"])
        results["prediction_log"].extend(fold_output["prediction_log"])
        results["window_prediction_log"].extend(
            fold_output["window_prediction_log"]
        )
        results["trial_prediction_log"].extend(
            fold_output["trial_prediction_log"]
        )
        results["variational_interval_log"].extend(
            fold_output["variational_interval_log"]
        )
        results["window_variational_interval_log"].extend(
            fold_output["window_variational_interval_log"]
        )
        results["trial_variational_interval_log"].extend(
            fold_output["trial_variational_interval_log"]
        )
        results["fold_results"].append(fold_output["fold_record"])
        results["outer_fold_results"].append(fold_output["outer_fold_result"])
        results["best_configs"].append(fold_output["best_config_result"])
        results["inner_cv_results"].append(fold_output["inner_cv_result"])

    mean_scores, std_scores = _mean_std_rows(
        results["fold_metrics"],
        ["loss", *metrics],
    )
    window_mean_scores, window_std_scores = _mean_std_rows(
        results["window_fold_metrics"],
        ["loss", *metrics],
    )
    trial_mean_scores, trial_std_scores = _mean_std_rows(
        results["trial_fold_metrics"],
        ["loss", *metrics],
    )

    results["mean_scores"] = mean_scores
    results["std_scores"] = std_scores
    results["window_mean_scores"] = window_mean_scores
    results["window_std_scores"] = window_std_scores
    results["trial_mean_scores"] = trial_mean_scores
    results["trial_std_scores"] = trial_std_scores

    print("\nLOSO CV complete")
    print("=" * 80)
    print("Primary mean scores:")
    print(pformat(mean_scores, indent=4, width=120, sort_dicts=False))
    print("Primary score standard deviations:")
    print(pformat(std_scores, indent=4, width=120, sort_dicts=False))
    print("Window-level mean scores:")
    print(pformat(window_mean_scores, indent=4, width=120, sort_dicts=False))
    print("Trial-level mean scores:")
    print(pformat(trial_mean_scores, indent=4, width=120, sort_dicts=False))

    return results
