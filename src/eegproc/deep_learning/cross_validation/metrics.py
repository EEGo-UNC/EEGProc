"""Classification metrics and diagnostic summaries."""

from __future__ import annotations

from typing import Mapping
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
import tensorflow as tf

from .arrays import _as_numpy_1d
from .constants import _CLASSIFICATION_METRICS, _DEFAULT_ECE_BINS
from .probabilities import _predict_labels

def _prediction_diagnostic_summary(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    threshold_tolerance: float = 0.01,
    internal_outputs: Mapping[str, np.ndarray] | None = None,
    reported_metric: str = "accuracy",
    decision_threshold: float = 0.5,
    ece_bins: int = _DEFAULT_ECE_BINS,
) -> dict[str, float | int | str]:
    """Summarize confidence, threshold collapse, and internal feature spread."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    y_ids = _as_numpy_1d(y_true).astype(np.int64)
    if probabilities.ndim != 2 or len(probabilities) != len(y_ids):
        raise ValueError(
            "Diagnostic probabilities must have shape (n, c) and align with "
            f"labels; got {probabilities.shape} and {len(y_ids)} labels."
        )
    if threshold_tolerance < 0.0:
        raise ValueError("threshold_tolerance must be non-negative.")
    reported_metric = str(reported_metric).strip().lower()
    if reported_metric not in _CLASSIFICATION_METRICS:
        raise ValueError(
            f"Unsupported prediction diagnostic metric {reported_metric!r}. "
            f"Supported metrics: {sorted(_CLASSIFICATION_METRICS)}"
        )

    y_pred = _predict_labels(
        probabilities,
        decision_threshold=decision_threshold,
    )
    confidence = np.max(probabilities, axis=1)
    diagnostic_metric_names = tuple(
        dict.fromkeys((reported_metric, "roc_auc"))
    )
    diagnostic_scores = _classification_metrics(
        y_true=y_ids,
        y_pred=y_pred,
        probabilities=probabilities,
        metrics=diagnostic_metric_names,
        n_classes=int(probabilities.shape[1]),
        ece_bins=int(ece_bins),
    )
    reported_value = diagnostic_scores[reported_metric]

    summary: dict[str, float | int | str] = {
        "n_samples": int(len(y_ids)),
        "accuracy": float(np.mean(y_pred == y_ids)),
        "roc_auc": float(diagnostic_scores["roc_auc"]),
        "reported_metric": reported_metric,
        "reported_metric_value": float(reported_value),
        "confidence_mean": float(np.mean(confidence)),
        "confidence_std": float(np.std(confidence)),
    }
    summary[reported_metric] = float(reported_value)

    for class_index in range(probabilities.shape[1]):
        class_probabilities = probabilities[:, class_index]
        summary[f"true_class_{class_index}_fraction"] = float(
            np.mean(y_ids == class_index)
        )
        summary[f"predicted_class_{class_index}_fraction"] = float(
            np.mean(y_pred == class_index)
        )
    return summary


def _print_probability_diagnostics(
    label: str,
    probabilities: np.ndarray,
    y_true: np.ndarray,
    threshold_tolerance: float = 0.01,
) -> dict[str, float | int | str]:
    """Print a compact probability-distribution diagnostic line."""
    summary = _prediction_diagnostic_summary(
        probabilities=probabilities,
        y_true=y_true,
        threshold_tolerance=threshold_tolerance,
    )
    parts = [
        f"n={summary['n_samples']}",
        f"accuracy={summary['accuracy']:.4f}",
        f"roc_auc={summary['roc_auc']:.4f}",
        f"confidence={summary['confidence_mean']:.4f}",
    ]
    if probabilities.shape[1] == 2:
        parts.extend(
            [
                f"pred1={summary['predicted_class_1_fraction']:.4f}",
                f"true1={summary['true_class_1_fraction']:.4f}",
            ]
        )
    print(f"\nPrediction diagnostics [{label}]: " + "  ".join(parts), flush=True)
    return summary


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    metrics: list[str] | tuple[str, ...],
    n_classes: int,
    ece_bins: int = _DEFAULT_ECE_BINS,
) -> dict:
    """Compute selected classification metrics.

    For binary tasks, ``f1``, ``precision``, and ``recall`` follow the
    Binary classification convention: class 1 is the positive class and no macro averaging
    is applied. ``binary_f1``, ``binary_precision``, and ``binary_recall`` are
    retained as backward-compatible aliases. Explicit ``macro_*`` metrics and
    ``balanced_accuracy`` remain available for class-balanced diagnostics.

    For multiclass tasks, the canonical metrics fall back to macro averaging
    because binary positive-class metrics are undefined. ``roc_auc`` uses the
    class-1 probability for binary tasks and macro one-vs-rest AUC for
    multiclass tasks; it is reported as NaN when an evaluation partition is
    missing a required ground-truth class. ``brier_score`` is the conventional
    binary Brier score for two-class tasks and the mean summed one-hot squared
    error for multiclass tasks. ``ece`` is equal-width, top-label expected
    calibration error over ``ece_bins`` confidence bins.
    """
    y_true = _as_numpy_1d(y_true).astype(np.int64)
    y_pred = _as_numpy_1d(y_pred).astype(np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)

    if n_classes < 2:
        raise ValueError(f"n_classes must be >= 2, got {n_classes}.")
    if int(ece_bins) < 2:
        raise ValueError(f"ece_bins must be >= 2, got {ece_bins}.")
    if probabilities.ndim != 2 or probabilities.shape != (len(y_true), n_classes):
        raise ValueError(
            "probabilities must have shape (n_samples, n_classes); got "
            f"{probabilities.shape} for {len(y_true)} labels and "
            f"{n_classes} classes."
        )

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

    binary_metric_names = {
        "binary_f1",
        "binary_precision",
        "binary_recall",
    }
    if n_classes != 2 and any(metric in binary_metric_names for metric in metrics):
        raise ValueError(
            "binary_f1, binary_precision, and binary_recall require "
            f"exactly two classes; got n_classes={n_classes}."
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
            if n_classes == 2:
                value = f1_score(
                    y_true,
                    y_pred,
                    average="binary",
                    pos_label=1,
                    zero_division=0,
                )
            else:
                value = f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    labels=expected_labels,
                    zero_division=0,
                )
            scores["f1"] = float(value)

        elif metric == "precision":
            if n_classes == 2:
                value = precision_score(
                    y_true,
                    y_pred,
                    average="binary",
                    pos_label=1,
                    zero_division=0,
                )
            else:
                value = precision_score(
                    y_true,
                    y_pred,
                    average="macro",
                    labels=expected_labels,
                    zero_division=0,
                )
            scores["precision"] = float(value)

        elif metric == "recall":
            if n_classes == 2:
                value = recall_score(
                    y_true,
                    y_pred,
                    average="binary",
                    pos_label=1,
                    zero_division=0,
                )
            else:
                value = recall_score(
                    y_true,
                    y_pred,
                    average="macro",
                    labels=expected_labels,
                    zero_division=0,
                )
            scores["recall"] = float(value)

        elif metric == "macro_f1":
            scores["macro_f1"] = float(
                f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    labels=expected_labels,
                    zero_division=0,
                )
            )

        elif metric == "macro_precision":
            scores["macro_precision"] = float(
                precision_score(
                    y_true,
                    y_pred,
                    average="macro",
                    labels=expected_labels,
                    zero_division=0,
                )
            )

        elif metric == "macro_recall":
            scores["macro_recall"] = float(
                recall_score(
                    y_true,
                    y_pred,
                    average="macro",
                    labels=expected_labels,
                    zero_division=0,
                )
            )

        elif metric == "balanced_accuracy":
            # For binary/multiclass classification this is macro recall over
            # the complete expected label set, including an absent class as 0.
            scores["balanced_accuracy"] = float(
                recall_score(
                    y_true,
                    y_pred,
                    average="macro",
                    labels=expected_labels,
                    zero_division=0,
                )
            )

        elif metric == "binary_f1":
            scores["binary_f1"] = float(
                f1_score(
                    y_true,
                    y_pred,
                    average="binary",
                    pos_label=1,
                    zero_division=0,
                )
            )

        elif metric == "binary_precision":
            scores["binary_precision"] = float(
                precision_score(
                    y_true,
                    y_pred,
                    average="binary",
                    pos_label=1,
                    zero_division=0,
                )
            )

        elif metric == "binary_recall":
            scores["binary_recall"] = float(
                recall_score(
                    y_true,
                    y_pred,
                    average="binary",
                    pos_label=1,
                    zero_division=0,
                )
            )

        elif metric == "roc_auc":
            if len(np.unique(y_true)) < n_classes:
                scores["roc_auc"] = float("nan")
            elif n_classes == 2:
                scores["roc_auc"] = float(
                    roc_auc_score(y_true, probabilities[:, 1])
                )
            else:
                scores["roc_auc"] = float(
                    roc_auc_score(
                        y_true,
                        probabilities,
                        labels=expected_labels,
                        multi_class="ovr",
                        average="macro",
                    )
                )

        elif metric == "brier_score":
            if n_classes == 2:
                scores["brier_score"] = float(
                    np.mean(np.square(probabilities[:, 1] - y_true))
                )
            else:
                one_hot = np.eye(n_classes, dtype=np.float64)[y_true]
                scores["brier_score"] = float(
                    np.mean(np.sum(np.square(probabilities - one_hot), axis=1))
                )

        elif metric == "ece":
            confidences = np.max(probabilities, axis=1)
            correct = (y_pred == y_true).astype(np.float64)
            bin_ids = np.minimum(
                (confidences * int(ece_bins)).astype(np.int64),
                int(ece_bins) - 1,
            )
            ece = 0.0
            for bin_index in range(int(ece_bins)):
                mask = bin_ids == bin_index
                if np.any(mask):
                    ece += float(np.mean(mask)) * abs(
                        float(np.mean(correct[mask]))
                        - float(np.mean(confidences[mask]))
                    )
            scores["ece"] = float(ece)

    return scores


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


def _decoder_reconstruction_scores(
    model: tf.keras.Model,
    X: np.ndarray,
    batch_size: int | None,
) -> dict[str, float]:
    """Compute memory-bounded branch and aggregate reconstruction diagnostics."""
    if not bool(getattr(model, "use_decoder", False)):
        return {}
    reconstruct_branches = getattr(model, "reconstruct_branches", None)
    if reconstruct_branches is None:
        return {}

    X_array = np.asarray(X)
    if len(X_array) == 0:
        return {}
    requested_batch_size = len(X_array) if batch_size is None else int(batch_size)
    if requested_batch_size < 1:
        raise ValueError("Decoder diagnostic batch_size must be at least 1.")
    # A single rank-4 trial expands to all of its flattened windows inside the model.
    effective_batch_size = 1 if X_array.ndim == 4 else requested_batch_size

    accumulators: dict[str, dict[str, float]] = {}
    for start in range(0, len(X_array), effective_batch_size):
        stop = min(start + effective_batch_size, len(X_array))
        target = np.asarray(X_array[start:stop], dtype=np.float64)
        raw_reconstructions = reconstruct_branches(
            tf.convert_to_tensor(X_array[start:stop], dtype=tf.float32)
        )
        if not isinstance(raw_reconstructions, Mapping):
            raise TypeError("reconstruct_branches() must return a branch mapping.")

        branch_names = set(str(name) for name in raw_reconstructions)
        if accumulators and branch_names != set(accumulators):
            raise ValueError(
                "Decoder branches changed between oracle batches: "
                f"expected={sorted(accumulators)}, got={sorted(branch_names)}."
            )
        for raw_name, raw_reconstruction in raw_reconstructions.items():
            branch = str(raw_name)
            reconstruction = raw_reconstruction
            if hasattr(reconstruction, "numpy"):
                reconstruction = reconstruction.numpy()
            reconstruction = np.asarray(reconstruction, dtype=np.float64)
            if reconstruction.shape != target.shape:
                raise ValueError(
                    f"{branch} reconstruction shape {reconstruction.shape} does "
                    f"not match target shape {target.shape}."
                )
            values = accumulators.setdefault(
                branch,
                {"sse": 0.0, "target_sum": 0.0, "target_sq_sum": 0.0, "count": 0.0},
            )
            residual = target - reconstruction
            values["sse"] += float(np.sum(np.square(residual), dtype=np.float64))
            values["target_sum"] += float(np.sum(target, dtype=np.float64))
            values["target_sq_sum"] += float(
                np.sum(np.square(target), dtype=np.float64)
            )
            values["count"] += float(target.size)

    if not accumulators:
        return {}

    def finish(values: Mapping[str, float]) -> tuple[float, float]:
        count = float(values["count"])
        mse = float(values["sse"]) / count
        ss_total = float(values["target_sq_sum"]) - (
            float(values["target_sum"]) ** 2 / count
        )
        epsilon = np.finfo(np.float64).eps
        if ss_total > epsilon:
            r2 = 1.0 - float(values["sse"]) / ss_total
        else:
            r2 = 1.0 if float(values["sse"]) <= epsilon else 0.0
        return float(mse), float(r2)

    scores: dict[str, float] = {}
    aggregate = {"sse": 0.0, "target_sum": 0.0, "target_sq_sum": 0.0, "count": 0.0}
    for branch, values in accumulators.items():
        mse, r2 = finish(values)
        scores[f"{branch}_reconstruction_loss"] = mse
        scores[f"{branch}_decoder_r2"] = r2
        for key in aggregate:
            aggregate[key] += float(values[key])

    scores["reconstruction_loss"], scores["decoder_r2"] = finish(aggregate)
    return scores
