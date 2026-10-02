"""Shaping fold results, prediction logs and console output."""

from __future__ import annotations

import numpy as np
from pprint import pformat

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotations only
    from typing import Mapping
    import tensorflow as tf

from .arrays import _as_numpy_1d, _is_trial_tensor, _python_scalar
from .constants import _DECODER_SCORE_NAMES, _DEFAULT_ECE_BINS
from .probabilities import _predict_labels, _predict_mc_probability_samples
from .aggregation import _aggregate_window_probabilities_by_trial
from .metrics import _classification_metrics, _probability_log_loss

def _level_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    metrics: list[str] | tuple[str, ...],
    ece_bins: int = _DEFAULT_ECE_BINS,
) -> dict:
    """Compute loss and requested metrics for one evaluation level."""
    scores = {
        "loss": _probability_log_loss(y_true, probabilities),
    }
    scores.update(
        _classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            probabilities=probabilities,
            metrics=metrics,
            n_classes=probabilities.shape[1],
            ece_bins=ece_bins,
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


def _keras_evaluation_results(
    model: tf.keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int | None = None,
) -> dict[str, float]:
    """Evaluate once and return all scalar Keras metrics as Python floats."""
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

    scalar_results: dict[str, float] = {}
    for metric_name, metric_value in eval_output.items():
        value_array = np.asarray(metric_value)
        if value_array.ndim != 0:
            raise ValueError(
                "Keras evaluation metrics must be scalar. "
                f"Metric {metric_name!r} returned shape {value_array.shape}."
            )
        scalar_results[str(metric_name)] = float(value_array)

    return scalar_results


def _keras_decoder_scores(keras_evaluation: Mapping[str, float]) -> dict[str, float]:
    """Extract every aggregate or branch-specific decoder metric present."""
    return {
        name: float(keras_evaluation[name])
        for name in _DECODER_SCORE_NAMES
        if name in keras_evaluation
    }


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
            "correct": int(pred_class == int(y_true[i])),
            "p_pred": float(probabilities[i, pred_class]),
            "confidence": float(np.max(probabilities[i])),
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
            "correct": int(pred_class == int(y_true[i])),
            "p_pred": float(probabilities[i, pred_class]),
            "confidence": float(np.max(probabilities[i])),
        }
        for class_idx in range(probabilities.shape[1]):
            row[f"p_class_{class_idx}"] = float(probabilities[i, class_idx])

        rows.append(row)

    return rows


def _make_variational_interval_logs(
    model: tf.keras.Model,
    X: np.ndarray,
    y_true: np.ndarray,
    subject_ids: np.ndarray,
    trial_ids: np.ndarray,
    fold_index: int,
    n_uncertainty_samples: int = 30,
    ci_level: float = 0.95,
    decision_threshold: float = 0.5,
) -> tuple[list[dict], list[dict]]:
    """Log Monte Carlo mean probabilities for windows and trials.

    Trial means are calculated by averaging windows within each stochastic
    forward pass before averaging across posterior samples. ``ci_level`` is
    retained for call compatibility but is no longer serialized or used.
    """
    if n_uncertainty_samples < 2:
        raise ValueError(
            "n_uncertainty_samples must be >= 2 when interval logging is enabled."
        )

    y_true = _as_numpy_1d(y_true).astype(np.int64)

    if _is_trial_tensor(X):
        trial_samples = _predict_mc_probability_samples(
            model=model,
            X=X,
            n_samples=n_uncertainty_samples,
            batch_size=None,
            seed=None,
        )
        trial_mean = trial_samples.mean(axis=0)
        trial_pred = _predict_labels(trial_mean, decision_threshold=decision_threshold)

        trial_rows: list[dict] = []
        for i in range(len(y_true)):
            pred_class = int(trial_pred[i])
            row = {
                "fold": int(fold_index),
                "trial_index": int(i),
                "subject_id": _python_scalar(subject_ids[i]),
                "trial_id": _python_scalar(trial_ids[i]),
                "n_windows": int(X.shape[1]),
                "y_true": int(y_true[i]),
                "y_pred": pred_class,
                "p_pred_mean": float(trial_mean[i, pred_class]),
            }
            for class_idx in range(trial_mean.shape[1]):
                row[f"p_class_{class_idx}_mean"] = float(trial_mean[i, class_idx])
            trial_rows.append(row)
        return [], trial_rows

    window_samples = _predict_mc_probability_samples(
        model=model,
        X=X,
        n_samples=n_uncertainty_samples,
        batch_size=None,
        seed=None,
    )
    window_mean = window_samples.mean(axis=0)

    window_pred = _predict_labels(window_mean, decision_threshold=decision_threshold)

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
        }
        for class_idx in range(window_mean.shape[1]):
            row[f"p_class_{class_idx}_mean"] = float(window_mean[i, class_idx])
        window_rows.append(row)

    reference_aggregation = _aggregate_window_probabilities_by_trial(
        probabilities=window_mean,
        y_true=y_true,
        subject_ids=subject_ids,
        trial_ids=trial_ids,
        decision_threshold=decision_threshold,
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
    trial_pred = _predict_labels(trial_mean, decision_threshold=decision_threshold)

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
        }
        for class_idx in range(trial_mean.shape[1]):
            row[f"p_class_{class_idx}_mean"] = float(trial_mean[i, class_idx])
        trial_rows.append(row)

    return window_rows, trial_rows


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

        numeric_values = np.asarray(values, dtype=np.float64)
        finite_values = numeric_values[np.isfinite(numeric_values)]
        if not len(finite_values):
            mean_scores[metric_name] = float("nan")
            std_scores[metric_name] = float("nan")
            continue

        mean_scores[metric_name] = float(np.mean(finite_values))
        std_scores[metric_name] = float(np.std(finite_values))

    return mean_scores, std_scores


def _calibration_score_dict(
    evaluation: dict,
    metric_names: tuple[str, ...],
) -> dict[str, float]:
    """Extract comparable classifier metrics from one evaluation result."""
    row = evaluation["fold_metrics"]
    output: dict[str, float] = {}
    for metric_name in metric_names:
        if metric_name in row:
            output[metric_name] = float(row[metric_name])
    return output


def _final_history_values(history) -> dict[str, float]:
    """Return the final finite scalar recorded for every Keras history key."""
    output: dict[str, float] = {}
    for key, values in history.history.items():
        if not values:
            continue
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        finite = array[np.isfinite(array)]
        if len(finite):
            output[str(key)] = float(finite[-1])
    return output


def _safe_model_filename_token(value) -> str:
    """Return a stable filesystem-safe token for a LOSO target identifier."""
    raw = str(_python_scalar(value))
    token = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in raw
    ).strip("_")
    return token or "subject"
