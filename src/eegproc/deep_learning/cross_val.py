from __future__ import annotations

import gc
import itertools
from itertools import combinations
from pprint import pformat
from typing import Callable, Literal

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


_FIT_RESERVED_KEYS = frozenset({"epochs", "batch_size"})
_CLASSIFICATION_METRICS = frozenset({"accuracy", "f1", "precision", "recall"})

def _expand_hyperparameter_grid(hp: dict | None) -> list[dict]:
    """Expand a hyperparameter dict into a Cartesian-product grid."""
    if not hp:
        return [{}]

    keys = list(hp.keys())
    values = [v if isinstance(v, (list, tuple)) else [v] for v in hp.values()]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


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

    raise ValueError(f"Expected labels with shape (n,), (n, 1), or (n, c). Got {values.shape}.")


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


def _predict_probabilities(
    model: tf.keras.Model,
    X: np.ndarray,
    batch_size: int | None = None,
) -> np.ndarray:
    """Run model.predict and convert output to probabilities."""
    raw_pred = model.predict(X, batch_size=batch_size, verbose=0)

    if isinstance(raw_pred, (tuple, list)):
        raw_pred = raw_pred[0]

    return _to_probabilities(raw_pred)


def _predict_labels(probabilities: np.ndarray) -> np.ndarray:
    """Convert probabilities to integer class predictions."""
    return np.argmax(probabilities, axis=1).astype(np.int64)


def _metric_average(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Use binary averaging for binary 0/1 labels, macro otherwise."""
    classes = np.unique(np.concatenate([y_true, y_pred]))

    if len(classes) <= 2 and set(classes.tolist()) <= {0, 1}:
        return "binary"

    return "macro"


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: list[str] | tuple[str, ...],
) -> dict:
    """Compute selected classification metrics."""
    y_true = _as_numpy_1d(y_true).astype(np.int64)
    y_pred = _as_numpy_1d(y_pred).astype(np.int64)

    average = _metric_average(y_true, y_pred)
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
                f1_score(y_true, y_pred, average=average, zero_division=0)
            )

        elif metric == "precision":
            scores["precision"] = float(
                precision_score(y_true, y_pred, average=average, zero_division=0)
            )

        elif metric == "recall":
            scores["recall"] = float(
                recall_score(y_true, y_pred, average=average, zero_division=0)
            )

    return scores


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
        raise ValueError(f"model.evaluate(..., return_dict=True) did not return 'loss': {eval_output}")

    return float(eval_output["loss"])


def _make_prediction_log(
    fold_index: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    subject_ids: np.ndarray,
    task_ids: np.ndarray | None = None,
) -> list[dict]:
    """Create one prediction-log row per evaluated sample/window.

    When ``task_ids`` is provided (e.g. a session/trial id per window), a
    ``task_id`` field is added next to ``subject_id`` so downstream reporting
    can break accuracy down per subject and per task.
    """
    y_true = _as_numpy_1d(y_true).astype(np.int64)
    y_pred = _as_numpy_1d(y_pred).astype(np.int64)

    rows: list[dict] = []

    for i in range(len(y_true)):
        pred_class = int(y_pred[i])
        row = {
            "fold": int(fold_index),
            "sample_index": int(i),
            "subject_id": _python_scalar(subject_ids[i]),
        }

        if task_ids is not None:
            row["task_id"] = _python_scalar(task_ids[i])

        row.update(
            {
                "y_true": int(y_true[i]),
                "y_pred": pred_class,
                "p_pred": float(probabilities[i, pred_class]),
            }
        )

        for class_idx in range(probabilities.shape[1]):
            row[f"p_class_{class_idx}"] = float(probabilities[i, class_idx])

        rows.append(row)

    return rows


def _make_variational_interval_log(
    model: tf.keras.Model,
    X: np.ndarray,
    y_true: np.ndarray,
    subject_ids: np.ndarray,
    fold_index: int,
    n_uncertainty_samples: int = 30,
    ci_level: float = 0.95,
    task_ids: np.ndarray | None = None,
) -> list[dict]:
    """Estimate predictive probability intervals with stochastic forward passes.

    This logs empirical probability intervals, not a closed-form Bayesian credible
    interval. It is useful when the model has stochastic inference behavior under
    training=True, such as dropout or latent sampling.
    """
    if n_uncertainty_samples < 2:
        raise ValueError("n_uncertainty_samples must be >= 2 when interval logging is enabled.")

    y_true = _as_numpy_1d(y_true).astype(np.int64)
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)

    probability_samples: list[np.ndarray] = []

    for _ in range(n_uncertainty_samples):
        raw_output = model(X_tensor, training=True)

        if isinstance(raw_output, (tuple, list)):
            raw_output = raw_output[0]

        probabilities = _to_probabilities(raw_output.numpy())
        probability_samples.append(probabilities)

    probability_samples_array = np.stack(probability_samples, axis=0)
    mean_probabilities = probability_samples_array.mean(axis=0)

    alpha = 1.0 - ci_level
    ci_low = np.quantile(probability_samples_array, alpha / 2.0, axis=0)
    ci_high = np.quantile(probability_samples_array, 1.0 - alpha / 2.0, axis=0)

    y_pred = np.argmax(mean_probabilities, axis=1).astype(np.int64)

    rows: list[dict] = []

    for i in range(len(y_true)):
        pred_class = int(y_pred[i])

        row = {
            "fold": int(fold_index),
            "sample_index": int(i),
            "subject_id": _python_scalar(subject_ids[i]),
        }

        if task_ids is not None:
            row["task_id"] = _python_scalar(task_ids[i])

        row.update(
            {
                "y_true": int(y_true[i]),
                "y_pred": pred_class,
                "p_pred_mean": float(mean_probabilities[i, pred_class]),
                "p_pred_ci_low": float(ci_low[i, pred_class]),
                "p_pred_ci_high": float(ci_high[i, pred_class]),
                "ci_level": float(ci_level),
                "n_uncertainty_samples": int(n_uncertainty_samples),
            }
        )

        for class_idx in range(mean_probabilities.shape[1]):
            row[f"p_class_{class_idx}_mean"] = float(mean_probabilities[i, class_idx])
            row[f"p_class_{class_idx}_ci_low"] = float(ci_low[i, class_idx])
            row[f"p_class_{class_idx}_ci_high"] = float(ci_high[i, class_idx])

        rows.append(row)

    return rows


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
    fold_index: int,
    metrics: list[str] | tuple[str, ...],
    batch_size: int | None = None,
    log_predictions: bool = True,
    log_variational_intervals: bool = False,
    n_uncertainty_samples: int = 30,
    ci_level: float = 0.95,
    task_ids_test: np.ndarray | None = None,
) -> dict:
    """Evaluate one outer fold and return all requested logs."""
    y_true = _as_numpy_1d(y_test).astype(np.int64)

    probabilities = _predict_probabilities(
        model=model,
        X=X_test,
        batch_size=batch_size,
    )
    y_pred = _predict_labels(probabilities)

    loss_value = _keras_loss_value(
        model=model,
        X=X_test,
        y=y_test,
        batch_size=batch_size,
    )

    fold_scores = {
        "fold": int(fold_index),
        "n_samples": int(len(y_true)),
        "loss": float(loss_value),
    }
    fold_scores.update(
        _classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            metrics=metrics,
        )
    )

    user_rows: list[dict] = []

    for subject_id in np.unique(subject_ids_test):
        mask = subject_ids_test == subject_id
        user_y_true = y_true[mask]
        user_y_pred = y_pred[mask]

        user_row = {
            "fold": int(fold_index),
            "subject_id": _python_scalar(subject_id),
            "n_samples": int(mask.sum()),
        }
        user_row.update(
            _classification_metrics(
                y_true=user_y_true,
                y_pred=user_y_pred,
                metrics=metrics,
            )
        )
        user_rows.append(user_row)

    prediction_rows: list[dict] = []

    if log_predictions:
        prediction_rows = _make_prediction_log(
            fold_index=fold_index,
            y_true=y_true,
            y_pred=y_pred,
            probabilities=probabilities,
            subject_ids=subject_ids_test,
            task_ids=task_ids_test,
        )

    interval_rows: list[dict] = []

    if log_variational_intervals:
        interval_rows = _make_variational_interval_log(
            model=model,
            X=X_test,
            y_true=y_true,
            subject_ids=subject_ids_test,
            fold_index=fold_index,
            n_uncertainty_samples=n_uncertainty_samples,
            ci_level=ci_level,
            task_ids=task_ids_test,
        )

    _print_metric_row(
        title=f"Fold {fold_index} metrics",
        row=fold_scores,
    )
    _print_user_metrics(user_rows)

    return {
        "fold_metrics": fold_scores,
        "user_metrics": user_rows,
        "prediction_log": prediction_rows,
        "variational_interval_log": interval_rows,
    }


def _evaluate_inner_config(
    model: tf.keras.Model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    metrics: list[str] | tuple[str, ...],
    batch_size: int | None = None,
) -> dict:
    """Evaluate an inner validation fold for hyperparameter selection."""
    y_true = _as_numpy_1d(y_val).astype(np.int64)
    probabilities = _predict_probabilities(model, X_val, batch_size=batch_size)
    y_pred = _predict_labels(probabilities)

    scores = {
        "loss": _keras_loss_value(model, X_val, y_val, batch_size=batch_size),
    }
    scores.update(_classification_metrics(y_true, y_pred, metrics=metrics))
    return scores


# ---------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------

def nested_lnso_cv(
    model_builder_function: Callable[..., tf.keras.Model],
    feature_array: np.ndarray,
    label_array: np.ndarray,
    subject_id_array: np.ndarray,
    task_id_array: np.ndarray | None = None,
    n_outer_subjects_to_leave_out: int = 1,
    n_inner_subjects_to_leave_out: int = 1,
    n_epochs: int = 50,
    batch_size: int = 32,
    hyperparameters: dict | None = None,
    preprocessing_strategy: Callable | None = None,
    selection_metric: str = "loss",
    maximize_metric: bool | None = None,
    metrics: list[str] | tuple[str, ...] = ("accuracy", "f1", "precision", "recall"),
    log_predictions: bool = True,
    log_variational_intervals: bool = False,
    n_uncertainty_samples: int = 30,
    ci_level: float = 0.95,
    verbose: int = 0,
    extra_fit_kwargs: dict | None = None,
) -> dict:
    """Run nested Leave-N-Subjects-Out CV with explicit classification logging.

    This function performs subject-level nested cross-validation:
        1. Outer LNSO estimates generalization to unseen subjects.
        2. Inner LNSO selects hyperparameters using only outer-training subjects.
        3. A fresh final model is trained on all outer-training data.
        4. The final model is evaluated on the outer-test subjects.

    New classification reporting behavior:
        - `metrics` controls which post-hoc classification metrics are computed.
        - `selection_metric` can be "loss" or one of the requested metrics.
        - `fold_metrics` stores one row per outer fold.
        - `user_metrics` stores one row per left-out subject per outer fold.
        - `prediction_log` stores one row per evaluated sample/window.
        - `variational_interval_log` optionally stores empirical predictive
          probability intervals from repeated stochastic forward passes.

    `task_id_array` is optional. When supplied (one task/session/trial id per
    window, aligned with `feature_array`), every prediction-log and
    variational-interval-log row gains a `task_id` field so reporting can break
    accuracy down per subject and per task.

    Returns
    -------
    dict
        Main keys:
            fold_metrics : list[dict]
            user_metrics : list[dict]
            prediction_log : list[dict]
            variational_interval_log : list[dict]
            best_configs : list[dict]
            inner_cv_results : list[dict]
            outer_fold_results : list[dict]
            mean_scores : dict
            std_scores : dict
    """
    extra_fit_kwargs = extra_fit_kwargs or {}

    if "validation_data" in extra_fit_kwargs:
        raise ValueError(
            "Do not pass validation_data in extra_fit_kwargs. "
            "nested_lnso_cv creates validation_data from the inner folds."
        )

    if subject_id_array is None:
        raise ValueError("subject_id_array is required for nested LNSO CV.")

    feature_array = np.asarray(feature_array)
    label_array = np.asarray(label_array)
    subject_id_array = np.asarray(subject_id_array)

    if len(feature_array) != len(label_array) or len(feature_array) != len(subject_id_array):
        raise ValueError(
            "feature_array, label_array, and subject_id_array must have the same "
            f"first dimension. Got {len(feature_array)}, {len(label_array)}, "
            f"and {len(subject_id_array)}."
        )

    if task_id_array is not None:
        task_id_array = np.asarray(task_id_array)

        if len(task_id_array) != len(feature_array):
            raise ValueError(
                "task_id_array must have the same first dimension as "
                f"feature_array. Got {len(task_id_array)} and {len(feature_array)}."
            )

    metrics = tuple(metrics)

    for metric in metrics:
        if metric not in _CLASSIFICATION_METRICS:
            raise ValueError(
                f"Unsupported metric: {metric}. Supported metrics: {sorted(_CLASSIFICATION_METRICS)}"
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

    results = {
        "fold_metrics": [],
        "user_metrics": [],
        "prediction_log": [],
        "variational_interval_log": [],
        "best_configs": [],
        "inner_cv_results": [],
        "outer_fold_results": [],
        "mean_scores": {},
        "std_scores": {},
    }

    print(
        f"\nNested LNSO CV — {len(outer_subject_splits)} outer folds, "
        f"{len(grid_configs)} hyperparameter config"
        f"{'s' if len(grid_configs) != 1 else ''}"
    )
    print(f"Requested metrics: {list(metrics)}")
    print(f"Selection metric: {selection_metric} ({'maximize' if maximize_metric else 'minimize'})")
    print(f"Prediction logging: {log_predictions}")
    print(f"Variational interval logging: {log_variational_intervals}\n")

    for outer_fold_number, outer_test_subjects in enumerate(
        outer_subject_splits,
        start=1,
    ):
        outer_test_subjects = np.array(outer_test_subjects)

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
            len(outer_subject_splits),
            f"outer test subjects={outer_test_subjects.tolist()} "
            f"(outer_train={len(outer_train_indices)}, "
            f"outer_test={len(outer_test_indices)} windows)",
        )

        inner_scores_by_config: list[list[dict]] = [[] for _ in grid_configs]
        inner_fold_results: list[dict] = []

        # -------------------------------------------------------------
        # Inner CV: choose hyperparameters.
        # -------------------------------------------------------------
        for inner_fold_number, inner_val_subjects in enumerate(
            inner_subject_splits,
            start=1,
        ):
            inner_val_subjects = np.array(inner_val_subjects)

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
                        metrics=metrics,
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
                    "configs": config_results_this_inner_fold,
                }
            )

        # -------------------------------------------------------------
        # Aggregate inner-CV scores and choose best config.
        # -------------------------------------------------------------
        inner_mean_scores: list[dict] = []
        inner_std_scores: list[dict] = []
        score_metric_names = ["loss", *metrics]

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
            f"{selection_metric}={inner_mean_scores[best_config_index][selection_metric]:.6f}"
        )
        _print_config("Best config:", best_config)

        results["best_configs"].append(
            {
                "outer_fold": int(outer_fold_number),
                "best_config_index": int(best_config_index),
                "best_config": dict(best_config),
                "selection_metric": selection_metric,
                "selection_score": float(inner_mean_scores[best_config_index][selection_metric]),
            }
        )

        results["inner_cv_results"].append(
            {
                "outer_fold": int(outer_fold_number),
                "inner_fold_results": inner_fold_results,
                "inner_mean_scores": inner_mean_scores,
                "inner_std_scores": inner_std_scores,
            }
        )

        # -------------------------------------------------------------
        # Final outer training and testing.
        # -------------------------------------------------------------
        X_outer_train = feature_array[outer_train_indices]
        y_outer_train = label_array[outer_train_indices]
        X_outer_test = feature_array[outer_test_indices]
        y_outer_test = label_array[outer_test_indices]
        subject_ids_outer_test = subject_id_array[outer_test_indices]
        task_ids_outer_test = (
            task_id_array[outer_test_indices] if task_id_array is not None else None
        )

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
                fold_index=outer_fold_number,
                metrics=metrics,
                batch_size=current_batch_size,
                log_predictions=log_predictions,
                log_variational_intervals=log_variational_intervals,
                n_uncertainty_samples=n_uncertainty_samples,
                ci_level=ci_level,
                task_ids_test=task_ids_outer_test,
            )

        finally:
            del final_model
            gc.collect()
            tf.keras.backend.clear_session()

        results["fold_metrics"].append(fold_result["fold_metrics"])
        results["user_metrics"].extend(fold_result["user_metrics"])
        results["prediction_log"].extend(fold_result["prediction_log"])
        results["variational_interval_log"].extend(
            fold_result["variational_interval_log"]
        )

        results["outer_fold_results"].append(
            {
                "outer_fold_number": int(outer_fold_number),
                "left_out_subjects": outer_test_subjects.tolist(),
                "n_outer_train_windows": int(len(outer_train_indices)),
                "n_outer_test_windows": int(len(outer_test_indices)),
                "best_config": dict(best_config),
                "inner_fold_results": inner_fold_results,
                "inner_mean_scores": inner_mean_scores,
                "inner_std_scores": inner_std_scores,
                "fold_metrics": fold_result["fold_metrics"],
                "user_metrics": fold_result["user_metrics"],
                "prediction_log": fold_result["prediction_log"],
                "variational_interval_log": fold_result["variational_interval_log"],
            }
        )

    mean_scores, std_scores = _mean_std_rows(
        results["fold_metrics"],
        ["loss", *metrics],
    )

    results["mean_scores"] = mean_scores
    results["std_scores"] = std_scores

    print("\nNested LNSO CV complete")
    print("=" * 80)
    print("Mean outer scores:")
    print(pformat(mean_scores, indent=4, width=120, sort_dicts=False))
    print("Std outer scores:")
    print(pformat(std_scores, indent=4, width=120, sort_dicts=False))

    return results