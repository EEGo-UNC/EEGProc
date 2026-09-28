"""Turning raw model output into probabilities, labels and decision thresholds."""

from __future__ import annotations

from typing import Mapping
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
from sklearn.metrics import recall_score
import tensorflow as tf

from .arrays import _as_numpy_1d

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


def _predict_mc_probability_samples(
    model,
    X: np.ndarray,
    n_samples: int,
    batch_size: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Return repeated or posterior-sampled probabilities shaped ``(S, N, C)``.

    Joint VAE models can expose ``predict_mc_probabilities`` to encode each
    input batch once and vectorize the recurrent/classifier work across latent
    samples. Deterministic models declare ``supports_latent_sampling=False``;
    their single prediction is repeated so shared reporting code remains valid
    and correctly produces zero latent-sampling dispersion.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")

    X = np.asarray(X)
    effective_batch_size = len(X) if batch_size is None else int(batch_size)
    if effective_batch_size < 1:
        raise ValueError("batch_size must be at least 1 when provided.")

    sample_batches: list[np.ndarray] = []
    for batch_index, start in enumerate(range(0, len(X), effective_batch_size)):
        X_batch = X[start : start + effective_batch_size]
        batch_seed = None if seed is None else (int(seed), int(batch_index))

        if getattr(model, "supports_latent_sampling", True) is False:
            raw_output = model(
                tf.convert_to_tensor(X_batch, dtype=tf.float32),
                training=False,
            )
            raw_output = _extract_classifier_output(raw_output)
            if hasattr(raw_output, "numpy"):
                raw_output = raw_output.numpy()
            deterministic_probabilities = _to_probabilities(raw_output)
            probability_samples = np.repeat(
                deterministic_probabilities[np.newaxis, ...],
                int(n_samples),
                axis=0,
            )
        elif hasattr(model, "predict_mc_probabilities"):
            mc_output = model.predict_mc_probabilities(
                X_batch,
                n_samples=n_samples,
                seed=batch_seed,
            )
            probability_samples = mc_output["probability_samples"]
            if hasattr(probability_samples, "numpy"):
                probability_samples = probability_samples.numpy()
            probability_samples = np.asarray(probability_samples, dtype=np.float64)
        else:
            probability_draws: list[np.ndarray] = []
            for sample_index in range(n_samples):
                if seed is not None:
                    tf.random.set_seed(
                        int(seed) + batch_index * n_samples + sample_index
                    )
                try:
                    raw_output = model(
                        tf.convert_to_tensor(X_batch, dtype=tf.float32),
                        training=False,
                        sample_latent=True,
                    )
                except TypeError as exc:
                    raise TypeError(
                        "Monte Carlo latent prediction requires the model to "
                        "implement predict_mc_probabilities(...) or accept "
                        "sample_latent=True in call(...)."
                    ) from exc
                raw_output = _extract_classifier_output(raw_output)
                if hasattr(raw_output, "numpy"):
                    raw_output = raw_output.numpy()
                probability_draws.append(_to_probabilities(raw_output))
            probability_samples = np.stack(probability_draws, axis=0)

        if probability_samples.ndim != 3:
            raise ValueError(
                "Monte Carlo probabilities must have shape "
                f"(n_samples, batch, n_classes); got {probability_samples.shape}."
            )
        sample_batches.append(probability_samples)

    return np.concatenate(sample_batches, axis=1)


def _predict_probabilities(
    model,
    X,
    batch_size=None,
    n_prediction_latent_samples: int = 0,
    latent_sampling_seed: int | None = None,
):
    """Return deterministic probabilities or averaged MC latent draws.

    For variational models, positive ``n_prediction_latent_samples`` values
    average that many samples from ``q(z|x)``. Deterministic models ignore the
    sampling count and return their single deterministic prediction.
    """
    if n_prediction_latent_samples < 0:
        raise ValueError("n_prediction_latent_samples must be >= 0.")

    if n_prediction_latent_samples > 0:
        probability_samples = _predict_mc_probability_samples(
            model=model,
            X=X,
            n_samples=n_prediction_latent_samples,
            batch_size=batch_size,
            seed=latent_sampling_seed,
        )
        return probability_samples.mean(axis=0)

    if hasattr(model, "predict_proba"):
        raw_pred = model.predict_proba(X)
    else:
        predict_kwargs = {"verbose": 0}

        if batch_size is not None:
            predict_kwargs["batch_size"] = batch_size

        raw_pred = model.predict(X, **predict_kwargs)

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


def _normalize_decision_thresholds(
    thresholds: list[float] | tuple[float, ...] | np.ndarray,
) -> tuple[float, ...]:
    """Validate, deduplicate, and sort binary class-1 thresholds."""
    values = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("decision_thresholds must contain at least one value.")
    if not np.isfinite(values).all():
        raise ValueError("decision_thresholds must contain only finite values.")
    if np.any(values <= 0.0) or np.any(values >= 1.0):
        raise ValueError("Every decision threshold must be strictly between 0 and 1.")
    return tuple(float(value) for value in np.unique(values))


def _predict_labels(
    probabilities: np.ndarray,
    decision_threshold: float = 0.5,
) -> np.ndarray:
    """Convert probabilities to labels using a binary class-1 threshold."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError(
            "probabilities must have shape (n_samples, n_classes); got "
            f"{probabilities.shape}."
        )
    if probabilities.shape[1] == 2:
        threshold = float(decision_threshold)
        if not 0.0 < threshold < 1.0:
            raise ValueError("decision_threshold must be strictly between 0 and 1.")
        return (probabilities[:, 1] >= threshold).astype(np.int64)
    if not np.isclose(float(decision_threshold), 0.5):
        raise ValueError(
            "Custom decision thresholds are supported only for binary models."
        )
    return np.argmax(probabilities, axis=1).astype(np.int64)


def _threshold_metric_value(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str,
) -> float:
    """Score one validation threshold without using test labels."""
    y_true = _as_numpy_1d(y_true).astype(np.int64)
    y_pred = _as_numpy_1d(y_pred).astype(np.int64)
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric == "f1":
        # Binary F1 uses class 1 as the positive class.
        return float(
            f1_score(
                y_true,
                y_pred,
                average="binary",
                pos_label=1,
                zero_division=0,
            )
        )
    if metric == "balanced_accuracy":
        return float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                labels=[0, 1],
                zero_division=0,
            )
        )
    if metric == "binary_f1":
        return float(
            f1_score(
                y_true,
                y_pred,
                average="binary",
                pos_label=1,
                zero_division=0,
            )
        )
    raise ValueError(
        "threshold_selection_metric must be accuracy, f1, "
        "balanced_accuracy, or binary_f1. Here f1 follows the "
        "binary class-1 convention."
    )


def _select_binary_decision_threshold(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    thresholds: tuple[float, ...],
    metric: str,
) -> tuple[float, float, list[dict]]:
    """Select a threshold on validation data with deterministic tie-breaking."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        if len(thresholds) > 1 or not np.isclose(thresholds[0], 0.5):
            raise ValueError(
                "Threshold search requires a binary two-probability output."
            )
        return 0.5, float("nan"), []

    rows: list[dict] = []
    for threshold in thresholds:
        y_pred = _predict_labels(
            probabilities,
            decision_threshold=threshold,
        )
        score = _threshold_metric_value(y_true, y_pred, metric)
        rows.append(
            {
                "threshold": float(threshold),
                "score": float(score),
                "predicted_class_1_fraction": float(np.mean(y_pred == 1)),
            }
        )

    # Maximize score; ties prefer the threshold closest to the conventional 0.5,
    # then the lower threshold for a stable deterministic result.
    best = min(
        rows,
        key=lambda row: (
            -row["score"],
            abs(row["threshold"] - 0.5),
            row["threshold"],
        ),
    )
    return float(best["threshold"]), float(best["score"]), rows


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
