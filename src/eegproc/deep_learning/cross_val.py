from __future__ import annotations

import gc
import inspect
import itertools
import multiprocessing as mp
import os
import queue
import traceback
import warnings
from pprint import pformat
from typing import Callable, Literal, Mapping

import numpy as np
import tensorflow as tf
from joblib.externals import cloudpickle
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit

__all__ = [
    "PredictionDiagnostics",
    "CompactEpochLogger",
    "HeldOutUserOracleMetrics",
    "MetaLearningSubjectSequence",
    "AlternatingSubjectSetSequence",
    "subject_calibration_cv",
]


def __getattr__(name: str):
    """Provide lazy compatibility exports without reintroducing import cycles."""
    if name == "PredictionDiagnostics":
        from .training_outputs import PredictionDiagnostics

        return PredictionDiagnostics
    if name == "CompactEpochLogger":
        from .training_outputs import CompactEpochLogger

        return CompactEpochLogger
    if name == "MetaLearningSubjectSequence":
        from .generalize_optimization_strats.MetaLearning import (
            MetaLearningSubjectSequence,
        )

        return MetaLearningSubjectSequence
    if name == "AlternatingSubjectSetSequence":
        from .generalize_optimization_strats.AlternatingGroupLearning import (
            AlternatingSubjectSetSequence,
        )

        return AlternatingSubjectSetSequence
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_FIT_RESERVED_KEYS = frozenset({"epochs", "batch_size"})
_CLASSIFICATION_METRICS = frozenset(
    {
        "accuracy",
        "f1",
        "precision",
        "recall",
        # Class-balanced alternatives retained for diagnostics.
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "balanced_accuracy",
        "roc_auc",
        "brier_score",
        "ece",
    }
)

_DEFAULT_ECE_BINS = 15

# Sequence-valued encoder settings need architecture-aware nesting rules.
# The integer is the nesting depth of one architecture value:
#   depth 1: [16, 32] or [2, 2]
#   depth 2: [[3, 3], [3, 3]]
# One additional outer level enumerates multiple candidate architectures.
_DEFAULT_SEQUENCE_HYPERPARAMETER_DEPTHS = {
    "conv_filters": 1,
    "kernel_sizes": 1,
    "pool_after_layers": 1,
    "pool_sizes": 1,
    "gcn_units": 1,
    "temporal_pool_sizes": 1,
    # Dense classifier layer widths are one architecture value, e.g.
    # [128, 64] means Dense(128) -> Dense(64), not two grid candidates.
    # Multiple classifier architectures can still be supplied with one
    # additional nesting level, e.g. [[128, 64], [256, 128]].
    "classification_hidden_units": 1,
    "spatial_pool_sizes": 2,
}
_EMPTY_SEQUENCE_ALLOWED_KEYS = frozenset(
    {
        "pool_after_layers",
        "pool_sizes",
        "spatial_pool_sizes",
    }
)


_JOINT_LOSS_WEIGHT_KEYS = frozenset(
    {
        "ae_loss_weight",
        "vc_loss_weight",
        "vae_beta",
        "vc_alpha",
        "vc_beta",
        "vc_gamma",
        "vc_lambda",
        "subject_loss_weight",
        "mldg_meta_test_weight",
        "use_subject_adversarial",
        # "label_smoothing",
    }
)


def _sequence_structure_depth(value) -> int:
    """Return the maximum list/tuple nesting depth of one value."""
    if not isinstance(value, (list, tuple)):
        return 0
    if not value:
        return 1
    return 1 + max(_sequence_structure_depth(item) for item in value)


def _copy_sequence_value(value):
    """Copy nested list/tuple values into JSON-friendly lists."""
    if isinstance(value, (list, tuple)):
        return [_copy_sequence_value(item) for item in value]
    return value


def _hyperparameter_candidates(
    key: str,
    value,
    sequence_hyperparameter_depths: Mapping[str, int] | None = None,
) -> list:
    """Return candidate values while preserving architecture sequences.

    ``sequence_hyperparameter_depths`` resolves otherwise ambiguous nested
    values. For CNN2D, for example, one ``kernel_sizes`` architecture has
    depth two (``[[3, 3], [3, 3]]``), while a depth-three value enumerates
    several kernel schedules. For CNN1D the same key has depth one.
    """
    sequence_depths = dict(_DEFAULT_SEQUENCE_HYPERPARAMETER_DEPTHS)
    if sequence_hyperparameter_depths:
        for sequence_key, expected_depth in sequence_hyperparameter_depths.items():
            expected_depth = int(expected_depth)
            if expected_depth < 1:
                raise ValueError(
                    "Sequence hyperparameter depths must be >= 1; got "
                    f"{sequence_key!r}: {expected_depth}."
                )
            sequence_depths[str(sequence_key)] = expected_depth

    if key not in sequence_depths:
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError(f"Hyperparameter {key!r} has an empty candidate list.")
            return list(value)
        return [value]

    if value is None:
        return [None]

    if not isinstance(value, (list, tuple)):
        raise TypeError(
            f"Sequence hyperparameter {key!r} must be a list or tuple, "
            f"got {type(value).__name__}."
        )
    if not value:
        if key in _EMPTY_SEQUENCE_ALLOWED_KEYS:
            return [[]]
        raise ValueError(f"Sequence hyperparameter {key!r} cannot be empty.")

    expected_depth = sequence_depths[key]
    actual_depth = _sequence_structure_depth(value)

    if actual_depth <= expected_depth:
        return [_copy_sequence_value(value)]

    if actual_depth == expected_depth + 1:
        candidates = [_copy_sequence_value(item) for item in value]
        if key not in _EMPTY_SEQUENCE_ALLOWED_KEYS and any(
            isinstance(candidate, list) and not candidate for candidate in candidates
        ):
            raise ValueError(
                f"Sequence hyperparameter {key!r} contains an empty candidate."
            )
        return candidates

    raise ValueError(
        f"Sequence hyperparameter {key!r} has nesting depth {actual_depth}, "
        f"but one architecture expects depth {expected_depth}. Use depth "
        f"{expected_depth} for one architecture or {expected_depth + 1} "
        "to enumerate candidates."
    )


def _warn_if_joint_loss_weights_vary(
    grid_configs: list[dict],
    selection_metric: str,
) -> None:
    """Warn when joint_loss selection compares models with incompatible weight settings."""
    if selection_metric != "joint_loss":
        return

    weight_keys = sorted(_JOINT_LOSS_WEIGHT_KEYS)
    weight_profiles = [
        tuple(config.get(key) for key in weight_keys)
        for config in grid_configs
    ]
    if len({profile for profile in weight_profiles}) > 1:
        warnings.warn(
            "selection_metric='joint_loss' was requested while joint-loss "
            "weight settings vary across configurations. This may make model "
            "comparison inconsistent.",
            UserWarning,
            stacklevel=3,
        )


def _expand_hyperparameter_grid(
    hp: dict | None,
    sequence_hyperparameter_depths: Mapping[str, int] | None = None,
) -> list[dict]:
    """Expand a hyperparameter dictionary into a Cartesian-product grid."""
    if not hp:
        return [{}]

    keys = list(hp)
    candidate_values = [
        _hyperparameter_candidates(
            key,
            hp[key],
            sequence_hyperparameter_depths=sequence_hyperparameter_depths,
        )
        for key in keys
    ]
    return [
        dict(zip(keys, combination))
        for combination in itertools.product(*candidate_values)
    ]


def _split_config(config: dict) -> tuple[dict, dict]:
    """Split a flat config into model-builder kwargs and model.fit kwargs."""
    model_hp = {k: v for k, v in config.items() if k not in _FIT_RESERVED_KEYS}
    fit_hp = {k: v for k, v in config.items() if k in _FIT_RESERVED_KEYS}
    return model_hp, fit_hp


def _build_model_with_fold_training_context(
    model_builder_function: Callable[..., tf.keras.Model],
    model_hp: dict,
    *,
    training_features: np.ndarray,
    training_labels: np.ndarray,
    training_subject_ids: np.ndarray,
    training_trial_ids: np.ndarray,
) -> tf.keras.Model:
    """Build a model with leakage-safe fold-local training context when supported.

    Some architectures, including SIC, need the actual gradient-training
    partition at construction time. SIC uses it to estimate its fixed
    MTLFuseNet-style mutual-information adjacency and to determine the
    fold-local subject-adversarial class count.

    To preserve compatibility with older EEGProc builders, training-context
    arguments are supplied only when the builder explicitly declares them in
    its signature. Validation and test samples are never included.
    """
    builder_kwargs = dict(model_hp)
    training_context = {
        "training_features": training_features,
        "training_labels": training_labels,
        "training_subject_ids": training_subject_ids,
        "training_trial_ids": training_trial_ids,
    }

    try:
        parameters = inspect.signature(model_builder_function).parameters
    except (TypeError, ValueError):
        # Preserve legacy behavior for unusual callables whose signatures
        # cannot be inspected.
        return model_builder_function(**builder_kwargs)

    accepted_context_keys = [
        key for key in training_context if key in parameters
    ]
    for key in accepted_context_keys:
        if key in builder_kwargs:
            raise ValueError(
                f"{key!r} must be supplied fold-locally by loso_cv; do not "
                "put it in the hyperparameter configuration."
            )
        builder_kwargs[key] = training_context[key]

    if accepted_context_keys:
        print(
            "Model builder fold-training context: "
            + ", ".join(accepted_context_keys),
            flush=True,
        )

    return model_builder_function(**builder_kwargs)


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


def _prepare_fit_inputs_with_subject_ids(
    model: tf.keras.Model,
    X: np.ndarray,
    subject_ids: np.ndarray,
):
    """Attach fold-local subject labels only when the model requests them.

    Subject-adversarial models expose ``prepare_fit_inputs``. The method maps
    the fitting subjects to contiguous fold-local classes and returns a Keras
    input dictionary. Ordinary models continue receiving the original EEG
    tensor unchanged. Validation and test inputs are intentionally left raw so
    held-out identities never contribute to the adversarial loss.
    """
    prepare = getattr(model, "prepare_fit_inputs", None)
    if prepare is None or not getattr(model, "use_subject_adversarial", False):
        return X
    return prepare(X, subject_ids)


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
        # MTLFuseNet convention: binary F1 for class 1.
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
        "MTLFuseNet binary class-1 convention."
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


def _is_trial_tensor(X: np.ndarray) -> bool:
    """Return True for hierarchical inputs shaped ``(N, W, T, F)``."""
    return np.asarray(X).ndim == 4


def _count_windows_for_indices(
    feature_array: np.ndarray,
    indices: np.ndarray,
) -> int:
    """Count underlying windows represented by selected samples.

    Rank-4 hierarchical inputs contain one trial per first-axis sample and one
    window axis at position 1. Rank-3 legacy inputs contain one window per
    first-axis sample.
    """
    features = np.asarray(feature_array)
    selected_count = int(len(indices))
    if features.ndim == 4:
        return selected_count * int(features.shape[1])
    if features.ndim == 3:
        return selected_count
    raise ValueError(
        "feature_array must be rank 3 or 4 when counting windows; "
        f"got {features.shape}."
    )


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
    MTLFuseNet convention: class 1 is the positive class and no macro averaging
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


class TrialValidationMetrics(tf.keras.callbacks.Callback):
    """Compute deterministic trial-level validation metrics each epoch.

    Hierarchical models are scored directly from one output per trial. Legacy
    window models are still aggregated within each (subject_id, trial_id) pair.
    The resulting values are added to the Keras epoch logs as
    ``val_trial_f1``, ``val_trial_balanced_accuracy``, and ``val_trial_loss``
    so callbacks such as EarlyStopping can monitor them.
    """

    def __init__(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        subject_ids_val: np.ndarray,
        trial_ids_val: np.ndarray,
        batch_size: int | None = None,
    ) -> None:
        super().__init__()
        self.X_val = np.asarray(X_val)
        self.y_val = np.asarray(y_val)
        self.subject_ids_val = np.asarray(subject_ids_val)
        self.trial_ids_val = np.asarray(trial_ids_val)
        self.batch_size = batch_size

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if logs is None:
            return

        # Use posterior-mean inference here. It is deterministic, inexpensive,
        # and avoids Monte Carlo sampling noise in the stopping decision.
        probabilities_model = _predict_probabilities(
            model=self.model,
            X=self.X_val,
            batch_size=self.batch_size,
            n_prediction_latent_samples=0,
            latent_sampling_seed=None,
        )

        if _is_trial_tensor(self.X_val):
            trial_aggregation = _direct_trial_aggregation(
                probabilities=probabilities_model,
                y_true=self.y_val,
                subject_ids=self.subject_ids_val,
                trial_ids=self.trial_ids_val,
                n_windows_per_trial=self.X_val.shape[1],
            )
        else:
            trial_aggregation = _aggregate_window_probabilities_by_trial(
                probabilities=probabilities_model,
                y_true=self.y_val,
                subject_ids=self.subject_ids_val,
                trial_ids=self.trial_ids_val,
            )

        probabilities_trial = trial_aggregation["probabilities"]
        y_true_trial = trial_aggregation["y_true"]
        y_pred_trial = trial_aggregation["y_pred"]
        expected_labels = list(range(probabilities_trial.shape[1]))

        if probabilities_trial.shape[1] == 2:
            val_trial_f1 = f1_score(
                y_true_trial,
                y_pred_trial,
                average="binary",
                pos_label=1,
                zero_division=0,
            )
        else:
            val_trial_f1 = f1_score(
                y_true_trial,
                y_pred_trial,
                average="macro",
                labels=expected_labels,
                zero_division=0,
            )
        logs["val_trial_f1"] = float(val_trial_f1)
        logs["val_trial_macro_f1"] = float(
            f1_score(
                y_true_trial,
                y_pred_trial,
                average="macro",
                labels=expected_labels,
                zero_division=0,
            )
        )
        logs["val_trial_balanced_accuracy"] = float(
            recall_score(
                y_true_trial,
                y_pred_trial,
                average="macro",
                labels=expected_labels,
                zero_division=0,
            )
        )
        logs["val_trial_loss"] = _probability_log_loss(
            y_true=y_true_trial,
            probabilities=probabilities_trial,
        )


class EvaluationLevelValidationMetrics(tf.keras.callbacks.Callback):
    """Add complete window- or trial-level validation metrics to epoch logs.

    This is used by nested subject-calibration LOSO folds so source-model epoch
    selection is based only on source-validation subjects. The outer target
    subject and all of its calibration/evaluation trials remain untouched.
    """

    _METRICS = (
        "accuracy",
        "f1",
        "precision",
        "recall",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "balanced_accuracy",
        "roc_auc",
        "brier_score",
        "ece",
    )

    def __init__(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        subject_ids_val: np.ndarray,
        trial_ids_val: np.ndarray,
        evaluation_level: Literal["window", "trial"],
        batch_size: int | None = None,
        ece_bins: int = _DEFAULT_ECE_BINS,
    ) -> None:
        super().__init__()
        self.X_val = np.asarray(X_val)
        self.y_val = np.asarray(y_val)
        self.subject_ids_val = np.asarray(subject_ids_val)
        self.trial_ids_val = np.asarray(trial_ids_val)
        self.evaluation_level = evaluation_level
        self.batch_size = batch_size
        self.ece_bins = int(ece_bins)

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if logs is None:
            return
        probabilities = _predict_probabilities(
            model=self.model,
            X=self.X_val,
            batch_size=self.batch_size,
            n_prediction_latent_samples=0,
            latent_sampling_seed=None,
        )
        if self.evaluation_level == "trial":
            if _is_trial_tensor(self.X_val):
                aggregation = _direct_trial_aggregation(
                    probabilities=probabilities,
                    y_true=self.y_val,
                    subject_ids=self.subject_ids_val,
                    trial_ids=self.trial_ids_val,
                    n_windows_per_trial=self.X_val.shape[1],
                )
            else:
                aggregation = _aggregate_window_probabilities_by_trial(
                    probabilities=probabilities,
                    y_true=self.y_val,
                    subject_ids=self.subject_ids_val,
                    trial_ids=self.trial_ids_val,
                )
            probabilities = aggregation["probabilities"]
            y_true = aggregation["y_true"]
            y_pred = aggregation["y_pred"]
        else:
            y_true = _as_numpy_1d(self.y_val).astype(np.int64)
            y_pred = _predict_labels(probabilities)

        scores = _classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            probabilities=probabilities,
            metrics=self._METRICS,
            n_classes=probabilities.shape[1],
            ece_bins=self.ece_bins,
        )
        prefix = f"val_{self.evaluation_level}_"
        for metric_name, value in scores.items():
            logs[f"{prefix}{metric_name}"] = float(value)
        logs[f"{prefix}loss"] = _probability_log_loss(
            y_true=y_true,
            probabilities=probabilities,
        )


class HeldOutUserOracleMetrics(tf.keras.callbacks.Callback):
    """Print reporting-only metrics on the outer held-out user each epoch.

    This callback deliberately keeps its values out of the Keras ``logs``
    dictionary. Consequently, the oracle metrics cannot be consumed by
    EarlyStopping, checkpoint selection, or any other training callback. The
    target data are used only for inference after an epoch has finished; they
    never enter a gradient calculation.
    """

    _METRICS = (
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "brier_score",
    )

    def __init__(
        self,
        X_target: np.ndarray,
        y_target: np.ndarray,
        subject_ids_target: np.ndarray,
        trial_ids_target: np.ndarray,
        target_subject,
        evaluation_level: Literal["window", "trial"],
        batch_size: int | None = None,
        decision_threshold: float = 0.5,
        diagnostic_thresholds: list[float] | tuple[float, ...] | None = None,
        ece_bins: int = _DEFAULT_ECE_BINS,
    ) -> None:
        super().__init__()
        self.X_target = np.asarray(X_target)
        self.y_target = np.asarray(y_target)
        self.subject_ids_target = np.asarray(subject_ids_target)
        self.trial_ids_target = np.asarray(trial_ids_target)
        self.target_subject = _python_scalar(target_subject)
        self.evaluation_level = evaluation_level
        self.batch_size = batch_size
        self.decision_threshold = float(decision_threshold)
        raw_thresholds = (
            (self.decision_threshold,)
            if diagnostic_thresholds is None
            else (*diagnostic_thresholds, self.decision_threshold)
        )
        normalized_thresholds = _normalize_decision_thresholds(raw_thresholds)
        self.diagnostic_thresholds = (
            self.decision_threshold,
            *(
                threshold
                for threshold in normalized_thresholds
                if threshold != self.decision_threshold
            ),
        )
        self.ece_bins = int(ece_bins)
        self.history: list[dict] = []

    def on_train_begin(self, logs: dict | None = None) -> None:
        print(
            "[ORACLE] Held-out-user metrics are reporting only and are not "
            "used for gradients, early stopping, threshold selection, or "
            "checkpoint selection. Official evaluation threshold="
            f"{self.decision_threshold:.4f}; reporting-only thresholds="
            f"{list(self.diagnostic_thresholds)}.",
            flush=True,
        )

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        probabilities = _predict_probabilities(
            model=self.model,
            X=self.X_target,
            batch_size=self.batch_size,
            n_prediction_latent_samples=0,
            latent_sampling_seed=None,
        )

        if self.evaluation_level == "trial":
            if _is_trial_tensor(self.X_target):
                aggregation = _direct_trial_aggregation(
                    probabilities=probabilities,
                    y_true=self.y_target,
                    subject_ids=self.subject_ids_target,
                    trial_ids=self.trial_ids_target,
                    n_windows_per_trial=self.X_target.shape[1],
                    decision_threshold=self.decision_threshold,
                )
            else:
                aggregation = _aggregate_window_probabilities_by_trial(
                    probabilities=probabilities,
                    y_true=self.y_target,
                    subject_ids=self.subject_ids_target,
                    trial_ids=self.trial_ids_target,
                    decision_threshold=self.decision_threshold,
                )
            probabilities = aggregation["probabilities"]
            y_true = aggregation["y_true"]
        else:
            y_true = _as_numpy_1d(self.y_target).astype(np.int64)

        for threshold in self.diagnostic_thresholds:
            y_pred = _predict_labels(
                probabilities,
                decision_threshold=threshold,
            )
            scores = _classification_metrics(
                y_true=y_true,
                y_pred=y_pred,
                probabilities=probabilities,
                metrics=self._METRICS,
                n_classes=probabilities.shape[1],
                ece_bins=self.ece_bins,
            )
            is_official = bool(threshold == self.decision_threshold)
            row = {
                "target_subject": self.target_subject,
                "epoch": int(epoch) + 1,
                "evaluation_level": self.evaluation_level,
                "decision_threshold": float(threshold),
                "official_decision_threshold": self.decision_threshold,
                "is_official_decision_threshold": is_official,
                **{name: float(value) for name, value in scores.items()},
            }
            if probabilities.shape[1] == 2:
                row["predicted_class_1_fraction"] = float(np.mean(y_pred == 1))
                row["true_class_1_fraction"] = float(np.mean(y_true == 1))
            self.history.append(row)

            parts = [
                f"{self.evaluation_level}_accuracy={row['accuracy']:.4f}",
                (
                    f"{self.evaluation_level}_balanced_accuracy="
                    f"{row['balanced_accuracy']:.4f}"
                ),
                f"{self.evaluation_level}_roc_auc={row['roc_auc']:.4f}",
                f"{self.evaluation_level}_brier_score={row['brier_score']:.4f}",
            ]
            if probabilities.shape[1] == 2:
                parts.extend(
                    [
                        f"pred1={row['predicted_class_1_fraction']:.4f}",
                        f"true1={row['true_class_1_fraction']:.4f}",
                    ]
                )
            print(
                f"[ORACLE held-out user={self.target_subject!r} "
                f"epoch={int(epoch) + 1} threshold={threshold:.4f} "
                f"official={str(is_official).lower()}] "
                + "  ".join(parts),
                flush=True,
            )


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


def _validate_processed_alignment(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    trial_ids: np.ndarray,
    partition_name: str,
) -> None:
    """Ensure fold-local preprocessing preserved sample order and count."""
    lengths = (len(X), len(y), len(subject_ids), len(trial_ids))
    if len(set(lengths)) != 1:
        raise ValueError(
            f"Preprocessing changed the number of {partition_name} samples or "
            "misaligned labels/IDs. Sample creation, removal, reordering, and "
            "resampling must occur before loso_cv. Got lengths "
            f"X/y/subject/trial={lengths}."
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

        numeric_values = np.asarray(values, dtype=np.float64)
        finite_values = numeric_values[np.isfinite(numeric_values)]
        if not len(finite_values):
            mean_scores[metric_name] = float("nan")
            std_scores[metric_name] = float("nan")
            continue

        mean_scores[metric_name] = float(np.mean(finite_values))
        std_scores[metric_name] = float(np.std(finite_values))

    return mean_scores, std_scores


# ---------------------------------------------------------------------
# Fold evaluation
# ---------------------------------------------------------------------


def _evaluate_trial_tensor_fold(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    subject_ids_test: np.ndarray,
    trial_ids_test: np.ndarray,
    fold_index: int,
    metrics: list[str] | tuple[str, ...],
    batch_size: int | None,
    n_prediction_latent_samples: int,
    latent_sampling_seed: int | None,
    log_predictions: bool,
    log_variational_intervals: bool,
    n_uncertainty_samples: int,
    ci_level: float,
    decision_threshold: float = 0.5,
    ece_bins: int = _DEFAULT_ECE_BINS,
    print_results: bool = True,
) -> dict:
    """Evaluate a model that emits one classifier prediction per trial."""
    y_true_trial = _as_numpy_1d(y_test).astype(np.int64)
    probabilities_trial = _predict_probabilities(
        model=model,
        X=X_test,
        batch_size=batch_size,
        n_prediction_latent_samples=n_prediction_latent_samples,
        latent_sampling_seed=latent_sampling_seed,
    )
    y_pred_trial = _predict_labels(
        probabilities_trial,
        decision_threshold=decision_threshold,
    )
    if print_results:
        _print_probability_diagnostics(
            label=f"fold {fold_index} test trial",
            probabilities=probabilities_trial,
            y_true=y_true_trial,
        )
    keras_evaluation = _keras_evaluation_results(
        model=model,
        X=X_test,
        y=y_test,
        batch_size=batch_size,
    )
    keras_model_loss = float(keras_evaluation["loss"])
    decoder_accuracy = keras_evaluation.get("decoder_accuracy")
    decoder_r2 = keras_evaluation.get("decoder_r2")

    trial_scores = _level_scores(
        y_true=y_true_trial,
        y_pred=y_pred_trial,
        probabilities=probabilities_trial,
        metrics=metrics,
        ece_bins=ece_bins,
    )
    trial_scores["joint_loss"] = keras_model_loss
    if decoder_accuracy is not None:
        trial_scores["decoder_accuracy"] = float(decoder_accuracy)
    if decoder_r2 is not None:
        trial_scores["decoder_r2"] = float(decoder_r2)

    n_trials = int(len(y_true_trial))
    n_windows_per_trial = int(X_test.shape[1])
    n_windows = n_trials * n_windows_per_trial
    fold_scores = {
        "fold": int(fold_index),
        "evaluation_level": "trial",
        "classification_level": "trial",
        "n_samples": n_trials,
        "n_windows": n_windows,
        "n_trials": n_trials,
        "windows_per_trial": n_windows_per_trial,
        "keras_model_loss": keras_model_loss,
        "joint_loss": keras_model_loss,
        **(
            {"decoder_accuracy": float(decoder_accuracy)}
            if decoder_accuracy is not None
            else {}
        ),
        "prediction_latent_samples": int(n_prediction_latent_samples),
        "decision_threshold": float(decision_threshold),
        **trial_scores,
        **_prefix_scores(trial_scores, "trial"),
    }

    window_fold_metrics = {
        "fold": int(fold_index),
        "n_windows": n_windows,
        "classification_available": False,
        "joint_loss": keras_model_loss,
        **(
            {"decoder_accuracy": float(decoder_accuracy)}
            if decoder_accuracy is not None
            else {}
        ),
    }
    trial_fold_metrics = {
        "fold": int(fold_index),
        "n_trials": n_trials,
        "windows_per_trial": n_windows_per_trial,
        **trial_scores,
    }

    user_rows: list[dict] = []
    for subject_id in np.unique(subject_ids_test):
        trial_mask = subject_ids_test == subject_id
        user_trial_scores = _level_scores(
            y_true=y_true_trial[trial_mask],
            y_pred=y_pred_trial[trial_mask],
            probabilities=probabilities_trial[trial_mask],
            metrics=metrics,
            ece_bins=ece_bins,
        )
        user_rows.append(
            {
                "fold": int(fold_index),
                "subject_id": _python_scalar(subject_id),
                "evaluation_level": "trial",
                "classification_level": "trial",
                "n_samples": int(trial_mask.sum()),
                "n_windows": int(trial_mask.sum()) * n_windows_per_trial,
                "n_trials": int(trial_mask.sum()),
                **user_trial_scores,
                **_prefix_scores(user_trial_scores, "trial"),
            }
        )

    trial_aggregation = _direct_trial_aggregation(
        probabilities=probabilities_trial,
        y_true=y_true_trial,
        subject_ids=subject_ids_test,
        trial_ids=trial_ids_test,
        n_windows_per_trial=n_windows_per_trial,
        decision_threshold=decision_threshold,
    )
    trial_prediction_rows = (
        _make_trial_prediction_log(fold_index, trial_aggregation)
        if log_predictions
        else []
    )

    window_interval_rows: list[dict] = []
    trial_interval_rows: list[dict] = []
    if log_variational_intervals:
        window_interval_rows, trial_interval_rows = _make_variational_interval_logs(
            model=model,
            X=X_test,
            y_true=y_true_trial,
            subject_ids=subject_ids_test,
            trial_ids=trial_ids_test,
            fold_index=fold_index,
            n_uncertainty_samples=n_uncertainty_samples,
            ci_level=ci_level,
            decision_threshold=decision_threshold,
        )

    if print_results:
        _print_metric_row(
            title=f"Fold {fold_index} metrics (trial primary)",
            row=fold_scores,
        )
        _print_user_metrics(user_rows)

    return {
        "fold_metrics": fold_scores,
        "window_fold_metrics": window_fold_metrics,
        "trial_fold_metrics": trial_fold_metrics,
        "user_metrics": user_rows,
        "window_prediction_log": [],
        "trial_prediction_log": trial_prediction_rows,
        "window_variational_interval_log": window_interval_rows,
        "trial_variational_interval_log": trial_interval_rows,
    }


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
    n_prediction_latent_samples: int = 0,
    latent_sampling_seed: int | None = None,
    log_predictions: bool = True,
    log_variational_intervals: bool = False,
    n_uncertainty_samples: int = 30,
    ci_level: float = 0.95,
    decision_threshold: float = 0.5,
    ece_bins: int = _DEFAULT_ECE_BINS,
    print_results: bool = True,
) -> dict:
    """Evaluate one outer fold at the model's native classification level."""
    _validate_evaluation_level(evaluation_level, "evaluation_level")
    if _is_trial_tensor(X_test):
        if evaluation_level != "trial":
            raise ValueError(
                "Hierarchical rank-4 inputs produce trial-level classifier "
                "outputs; evaluation_level must be 'trial'."
            )
        return _evaluate_trial_tensor_fold(
            model=model,
            X_test=X_test,
            y_test=y_test,
            subject_ids_test=subject_ids_test,
            trial_ids_test=trial_ids_test,
            fold_index=fold_index,
            metrics=metrics,
            batch_size=batch_size,
            n_prediction_latent_samples=n_prediction_latent_samples,
            latent_sampling_seed=latent_sampling_seed,
            log_predictions=log_predictions,
            log_variational_intervals=log_variational_intervals,
            n_uncertainty_samples=n_uncertainty_samples,
            ci_level=ci_level,
            decision_threshold=decision_threshold,
            ece_bins=ece_bins,
            print_results=print_results,
        )
    y_true_window = _as_numpy_1d(y_test).astype(np.int64)

    probabilities_window = _predict_probabilities(
        model=model,
        X=X_test,
        batch_size=batch_size,
        n_prediction_latent_samples=n_prediction_latent_samples,
        latent_sampling_seed=latent_sampling_seed,
    )
    y_pred_window = _predict_labels(
        probabilities_window,
        decision_threshold=decision_threshold,
    )
    if print_results:
        _print_probability_diagnostics(
            label=f"fold {fold_index} test window",
            probabilities=probabilities_window,
            y_true=y_true_window,
        )

    # model.evaluate() is retained as a diagnostic because joint Keras models
    # may include reconstruction/regularization terms beyond classification.
    # It also exposes decoder_accuracy for continuous reconstruction quality.
    keras_evaluation = _keras_evaluation_results(
        model=model,
        X=X_test,
        y=y_test,
        batch_size=batch_size,
    )
    keras_model_loss = keras_evaluation["loss"]
    decoder_accuracy = keras_evaluation.get("decoder_accuracy")
    decoder_r2 = keras_evaluation.get("decoder_r2")

    window_scores = _level_scores(
        y_true=y_true_window,
        y_pred=y_pred_window,
        probabilities=probabilities_window,
        metrics=metrics,
        ece_bins=ece_bins,
    )
    # ``loss`` above is classifier probability log loss. ``joint_loss`` is
    # the model's complete weighted VAE + VC objective returned by Keras.
    window_scores["joint_loss"] = float(keras_model_loss)
    if decoder_accuracy is not None:
        window_scores["decoder_accuracy"] = float(decoder_accuracy)
    if decoder_r2 is not None:
        window_scores["decoder_r2"] = float(decoder_r2)

    trial_aggregation = _aggregate_window_probabilities_by_trial(
        probabilities=probabilities_window,
        y_true=y_true_window,
        subject_ids=subject_ids_test,
        trial_ids=trial_ids_test,
        decision_threshold=decision_threshold,
    )
    if print_results:
        _print_probability_diagnostics(
            label=f"fold {fold_index} test trial-aggregated",
            probabilities=trial_aggregation["probabilities"],
            y_true=trial_aggregation["y_true"],
        )
    trial_scores = _level_scores(
        y_true=trial_aggregation["y_true"],
        y_pred=trial_aggregation["y_pred"],
        probabilities=trial_aggregation["probabilities"],
        metrics=metrics,
        ece_bins=ece_bins,
    )
    if decoder_accuracy is not None:
        trial_scores["decoder_accuracy"] = float(decoder_accuracy)
    if decoder_r2 is not None:
        trial_scores["decoder_r2"] = float(decoder_r2)

    primary_scores = trial_scores if evaluation_level == "trial" else window_scores
    fold_scores = {
        "fold": int(fold_index),
        "n_samples": int(
            len(trial_aggregation["y_true"])
            if evaluation_level == "trial"
            else len(y_true_window)
        ),
        "n_windows": int(len(y_true_window)),
        "n_trials": int(len(trial_aggregation["y_true"])),
        "keras_model_loss": float(keras_model_loss),
        "joint_loss": float(keras_model_loss),
        **(
            {"decoder_accuracy": float(decoder_accuracy)}
            if decoder_accuracy is not None
            else {}
        ),
        "prediction_latent_samples": int(n_prediction_latent_samples),
        "decision_threshold": float(decision_threshold),
        **primary_scores,
        **_prefix_scores(window_scores, "window"),
        **_prefix_scores(trial_scores, "trial"),
    }

    window_fold_metrics = {
        "fold": int(fold_index),
        "n_windows": int(len(y_true_window)),
        "keras_model_loss": float(keras_model_loss),
        "joint_loss": float(keras_model_loss),
        "decision_threshold": float(decision_threshold),
        **window_scores,
    }
    trial_fold_metrics = {
        "fold": int(fold_index),
        "n_trials": int(len(trial_aggregation["y_true"])),
        "decision_threshold": float(decision_threshold),
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
            ece_bins=ece_bins,
        )
        user_trial_scores = _level_scores(
            y_true=trial_aggregation["y_true"][trial_mask],
            y_pred=trial_aggregation["y_pred"][trial_mask],
            probabilities=trial_aggregation["probabilities"][trial_mask],
            metrics=metrics,
            ece_bins=ece_bins,
        )
        user_primary_scores = (
            user_trial_scores if evaluation_level == "trial" else user_window_scores
        )

        user_rows.append(
            {
                "fold": int(fold_index),
                "subject_id": _python_scalar(subject_id),
                "evaluation_level": evaluation_level,
                "n_samples": int(
                    trial_mask.sum()
                    if evaluation_level == "trial"
                    else window_mask.sum()
                ),
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

    if print_results:
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
        "window_prediction_log": window_prediction_rows,
        "trial_prediction_log": trial_prediction_rows,
        "window_variational_interval_log": window_interval_rows,
        "trial_variational_interval_log": trial_interval_rows,
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
            token.strip() for token in visible_devices.split(",") if token.strip()
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
            f"GPU {int(requested_gpu_id)} " f"(CUDA_VISIBLE_DEVICES={cuda_token})"
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
                process for process in processes if process.exitcode not in (None, 0)
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
            raise RuntimeError(f"Received duplicate result for fold {fold_number}.")

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

        payload_size_mb = len(worker_state_payload) / (1024**2)
        if payload_size_mb >= 256.0:
            print(
                f"Warning: serialized worker state is {payload_size_mb:.1f} MiB. "
                "Each spawned worker will hold its own host-memory copy.",
                flush=True,
            )

        for worker_index in range(n_workers):
            requested_gpu_id = gpu_ids[worker_index] if gpu_ids is not None else None
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
    n_prediction_latent_samples: int,
    latent_sampling_seed: int | None,
    n_uncertainty_samples: int,
    ci_level: float,
    validation_subjects_per_fold: int,
    validation_seed: int | None,
    early_stopping_patience: int | None,
    early_stopping_min_delta: float,
    early_stopping_monitor: str,
    early_stopping_mode: Literal["auto", "min", "max"],
    restore_best_weights: bool,
    prediction_diagnostics: bool,
    prediction_diagnostics_every_n_epochs: int,
    prediction_diagnostics_max_samples: int,
    prediction_diagnostics_threshold_tolerance: float,
    prediction_diagnostics_seed: int | None,
    decision_thresholds: tuple[float, ...],
    threshold_selection_metric: str,
    threshold_selection_level: Literal["window", "trial"],
    verbose: int,
    extra_fit_kwargs: dict,
    fold_description: str | None = None,
    alternate_subject_sets: bool = False,
    alternating_subject_seed: int | None = 42,
    use_mldg: bool = False,
    mldg_meta_train_subjects: int = 6,
    mldg_meta_test_subjects: int = 2,
    mldg_samples_per_subject: int = 4,
    mldg_seed: int | None = 42,
) -> dict:
    """Train and evaluate one LOSO fold with optional seeded validation.

    The LOSO test subject is never used by ``model.fit``. When
    ``validation_subjects_per_fold`` is positive, that many subjects are drawn
    deterministically from the outer-training pool and excluded from gradient
    updates. They provide ``validation_data`` for early stopping without adding
    another model fit.
    """
    test_mask = subject_id_array == test_subject
    test_indices = np.where(test_mask)[0]
    outer_train_indices = np.where(~test_mask)[0]

    if len(outer_train_indices) == 0 or len(test_indices) == 0:
        raise ValueError(
            f"Invalid LOSO split: train={len(outer_train_indices)}, "
            f"test={len(test_indices)} samples."
        )

    outer_train_subjects = np.sort(np.unique(subject_id_array[outer_train_indices]))
    validation_candidate_subjects = outer_train_subjects
    if validation_subjects_per_fold < 0:
        raise ValueError("validation_subjects_per_fold must be >= 0.")
    if alternate_subject_sets and use_mldg:
        raise ValueError("alternate_subject_sets and use_mldg are mutually exclusive.")
    if mldg_meta_train_subjects < 1 or mldg_meta_test_subjects < 1:
        raise ValueError("MLDG A/B subject counts must both be at least 1.")
    if mldg_samples_per_subject < 1:
        raise ValueError("mldg_samples_per_subject must be at least 1.")
    if mldg_seed is not None and mldg_seed < 0:
        raise ValueError("mldg_seed must be >= 0 or None.")
    if alternate_subject_sets and validation_subjects_per_fold != 0:
        raise ValueError(
            "alternate_subject_sets uses all non-test subjects and therefore "
            "requires validation_subjects_per_fold=0."
        )
    if validation_subjects_per_fold > 0 and validation_subjects_per_fold >= len(
        validation_candidate_subjects
    ):
        raise ValueError(
            "validation_subjects_per_fold must leave at least one eligible "
            "subject outside validation. Got "
            f"{validation_subjects_per_fold} validation subjects from "
            f"{len(validation_candidate_subjects)} eligible subjects."
        )

    if validation_subjects_per_fold > 0:
        base_seed = 0 if validation_seed is None else int(validation_seed)
        fold_seed = np.random.SeedSequence([base_seed, int(fold_number)])
        rng = np.random.default_rng(fold_seed)
        validation_subjects = np.sort(
            rng.choice(
                validation_candidate_subjects,
                size=validation_subjects_per_fold,
                replace=False,
            )
        )
        validation_mask_relative = np.isin(
            subject_id_array[outer_train_indices],
            validation_subjects,
        )
        validation_indices = outer_train_indices[validation_mask_relative]
        fit_train_indices = outer_train_indices[~validation_mask_relative]
    else:
        validation_subjects = np.asarray([], dtype=outer_train_subjects.dtype)
        validation_indices = np.asarray([], dtype=np.int64)
        fit_train_indices = outer_train_indices

    sample_level = "trials" if feature_array.ndim == 4 else "windows"
    if fold_description is None:
        fold_description = (
            f"LOSO test subject={_python_scalar(test_subject)!r} "
            f"(fit_train={len(fit_train_indices)}, "
            f"validation={len(validation_indices)}, "
            f"test={len(test_indices)} {sample_level})"
        )
    else:
        fold_description = (
            f"{fold_description} (fit_train={len(fit_train_indices)}, "
            f"validation={len(validation_indices)}, "
            f"test={len(test_indices)} {sample_level})"
        )
    _print_fold_header(
        fold_number,
        total_folds,
        fold_description,
    )
    if len(validation_subjects):
        print(
            "Seeded validation subjects: "
            f"{[_python_scalar(value) for value in validation_subjects]}",
            flush=True,
        )

    # The current preprocessing callback API supports only one train/eval pair.
    # Refuse an ambiguous three-way fit rather than leaking validation subjects
    # into a fitted transform or fitting inconsistent transforms for val/test.
    if validation_subjects_per_fold > 0 and preprocessing_strategy is not None:
        raise ValueError(
            "Seeded subject-level validation currently requires "
            "preprocessing_strategy=None. Preprocess before loso_cv or extend "
            "the strategy API to transform train/validation/test from one "
            "fold-local fitted state."
        )

    X_fit_train = feature_array[fit_train_indices]
    y_fit_train = label_array[fit_train_indices]
    X_validation = feature_array[validation_indices]
    y_validation = label_array[validation_indices]
    X_test = feature_array[test_indices]
    y_test = label_array[test_indices]

    subject_ids_fit_train = subject_id_array[fit_train_indices]
    subject_ids_validation = subject_id_array[validation_indices]
    subject_ids_test = subject_id_array[test_indices]
    trial_ids_fit_train = trial_id_array[fit_train_indices]
    trial_ids_validation = trial_id_array[validation_indices]
    trial_ids_test = trial_id_array[test_indices]

    if validation_subjects_per_fold == 0:
        X_fit_train, y_fit_train, X_test, y_test = _apply_preprocessing_strategy(
            preprocessing_strategy=preprocessing_strategy,
            X_train=X_fit_train,
            y_train=y_fit_train,
            X_eval=X_test,
            y_eval=y_test,
            train_indices=fit_train_indices,
            eval_indices=test_indices,
        )

    _validate_processed_alignment(
        X_fit_train,
        y_fit_train,
        subject_ids_fit_train,
        trial_ids_fit_train,
        "LOSO-fit-training",
    )
    if validation_subjects_per_fold > 0:
        _validate_processed_alignment(
            X_validation,
            y_validation,
            subject_ids_validation,
            trial_ids_validation,
            "LOSO-validation",
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

    from .training_outputs import CompactEpochLogger, PredictionDiagnostics

    fit_call_kwargs = dict(extra_fit_kwargs)
    callbacks = list(fit_call_kwargs.pop("callbacks", []))
    prediction_diagnostics_callback: PredictionDiagnostics | None = None

    if prediction_diagnostics:
        prediction_diagnostics_callback = PredictionDiagnostics(
            X_train=X_fit_train,
            y_train=y_fit_train,
            X_val=(X_validation if validation_subjects_per_fold > 0 else None),
            y_val=(y_validation if validation_subjects_per_fold > 0 else None),
            fold_number=fold_number,
            batch_size=current_batch_size,
            every_n_epochs=prediction_diagnostics_every_n_epochs,
            max_samples=prediction_diagnostics_max_samples,
            threshold_tolerance=prediction_diagnostics_threshold_tolerance,
            seed=(
                None
                if prediction_diagnostics_seed is None
                else int(prediction_diagnostics_seed) + int(fold_number)
            ),
        )
        callbacks.append(prediction_diagnostics_callback)

    oracle_reporting_threshold = (
        float(decision_thresholds[0])
        if len(decision_thresholds) == 1
        else 0.5
    )
    oracle_metrics_callback = HeldOutUserOracleMetrics(
        X_target=X_test,
        y_target=y_test,
        subject_ids_target=subject_ids_test,
        trial_ids_target=trial_ids_test,
        target_subject=test_subject,
        evaluation_level=evaluation_level,
        batch_size=current_batch_size,
        decision_threshold=oracle_reporting_threshold,
    )
    callbacks.append(oracle_metrics_callback)

    if validation_subjects_per_fold > 0:
        if early_stopping_monitor in {
            "val_trial_f1",
            "val_trial_balanced_accuracy",
            "val_trial_loss",
        }:
            # This callback must run before CompactEpochLogger and EarlyStopping
            # so the custom metric is available to both callbacks.
            callbacks.append(
                TrialValidationMetrics(
                    X_val=X_validation,
                    y_val=y_validation,
                    subject_ids_val=subject_ids_validation,
                    trial_ids_val=trial_ids_validation,
                    batch_size=current_batch_size,
                )
            )

    if verbose:
        callbacks.append(CompactEpochLogger(fold_number=fold_number))

    if validation_subjects_per_fold > 0 and early_stopping_patience is not None:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=early_stopping_monitor,
                patience=int(early_stopping_patience),
                min_delta=float(early_stopping_min_delta),
                mode=early_stopping_mode,
                restore_best_weights=bool(restore_best_weights),
                verbose=1 if verbose else 0,
            )
        )

    if callbacks:
        fit_call_kwargs["callbacks"] = callbacks

    tf.keras.backend.clear_session()
    model = _build_model_with_fold_training_context(
        model_builder_function,
        model_hp,
        training_features=X_fit_train,
        training_labels=y_fit_train,
        training_subject_ids=subject_ids_fit_train,
        training_trial_ids=trial_ids_fit_train,
    )
    X_fit_train_for_fit = _prepare_fit_inputs_with_subject_ids(
        model,
        X_fit_train,
        subject_ids_fit_train,
    )

    epochs_ran = 0
    best_epoch: int | None = None
    best_monitored_value: float | None = None
    stopped_early = False

    try:
        y_fit_train_ids = _as_numpy_1d(y_fit_train)
        classes, counts = np.unique(y_fit_train_ids, return_counts=True)

        class_weight = {
            int(class_id): len(y_fit_train_ids) / (len(classes) * count)
            for class_id, count in zip(classes, counts)
        }

        validation_data = (
            (X_validation, y_validation) if validation_subjects_per_fold > 0 else None
        )
        if use_mldg:
            from .generalize_optimization_strats.MetaLearning import (
                MetaLearningSubjectSequence,
            )

            fold_mldg_seed = (
                None if mldg_seed is None else int(mldg_seed) + int(fold_number)
            )
            effective_class_weight = (
                class_weight if bool(getattr(model, "use_class_weight", True)) else None
            )
            mldg_sequence = MetaLearningSubjectSequence(
                X=X_fit_train,
                y=y_fit_train,
                subject_ids=subject_ids_fit_train,
                model=model,
                meta_train_subjects=mldg_meta_train_subjects,
                meta_test_subjects=mldg_meta_test_subjects,
                samples_per_subject=mldg_samples_per_subject,
                class_weight=effective_class_weight,
                seed=fold_mldg_seed,
            )
            print(
                "First-order MLDG episodes (natural within-subject labels): "
                f"A_subjects={mldg_meta_train_subjects}, "
                f"B_subjects={mldg_meta_test_subjects}, "
                f"samples_per_subject={mldg_samples_per_subject}, "
                f"steps_per_epoch={len(mldg_sequence)}",
                flush=True,
            )
            subject_set_a = np.asarray([], dtype=subject_ids_fit_train.dtype)
            subject_set_b = np.asarray([], dtype=subject_ids_fit_train.dtype)
            history = model.fit(
                mldg_sequence,
                validation_data=validation_data,
                verbose=0,
                **fit_hp,
                **fit_call_kwargs,
            )
        elif alternate_subject_sets:
            from .generalize_optimization_strats.AlternatingGroupLearning import (
                AlternatingSubjectSetSequence,
            )

            if validation_subjects_per_fold > 0:
                raise ValueError(
                    "alternate_subject_sets requires validation_subjects_per_fold=0."
                )
            fold_alt_seed = (
                None
                if alternating_subject_seed is None
                else int(alternating_subject_seed) + int(fold_number)
            )
            subject_set_a, subject_set_b = _balanced_two_subject_sets(
                subject_ids_fit_train,
                y_fit_train,
                seed=fold_alt_seed,
            )
            print(
                "Alternating subject sets: "
                f"A={[_python_scalar(v) for v in subject_set_a]} | "
                f"B={[_python_scalar(v) for v in subject_set_b]}",
                flush=True,
            )
            alternating_sequence = AlternatingSubjectSetSequence(
                X=X_fit_train,
                y=y_fit_train,
                subject_ids=subject_ids_fit_train,
                subject_set_a=subject_set_a,
                subject_set_b=subject_set_b,
                batch_size=current_batch_size,
                model=model,
                class_weight=class_weight,
                seed=fold_alt_seed,
            )
            history = model.fit(
                alternating_sequence,
                validation_data=None,
                verbose=0,
                **fit_hp,
                **fit_call_kwargs,
            )
        else:
            subject_set_a = np.asarray([], dtype=subject_ids_fit_train.dtype)
            subject_set_b = np.asarray([], dtype=subject_ids_fit_train.dtype)
            history = model.fit(
                X_fit_train_for_fit,
                y_fit_train,
                validation_data=validation_data,
                class_weight=class_weight,
                verbose=0,
                **fit_hp,
                **fit_call_kwargs,
            )

        epochs_ran = int(len(history.history.get("loss", [])))
        requested_epochs = int(fit_hp.get("epochs", epochs_ran))
        stopped_early = bool(epochs_ran < requested_epochs)

        monitored_history = history.history.get(early_stopping_monitor)
        if monitored_history:
            monitored_values = np.asarray(monitored_history, dtype=np.float64)
            finite_mask = np.isfinite(monitored_values)
            if np.any(finite_mask):
                candidate_indices = np.where(finite_mask)[0]
                candidate_values = monitored_values[finite_mask]
                if early_stopping_mode == "max":
                    local_best = int(np.argmax(candidate_values))
                elif early_stopping_mode == "min":
                    local_best = int(np.argmin(candidate_values))
                else:
                    maximize_tokens = ("acc", "auc", "f1", "precision", "recall")
                    maximize = any(
                        token in early_stopping_monitor.lower()
                        for token in maximize_tokens
                    )
                    local_best = int(
                        np.argmax(candidate_values)
                        if maximize
                        else np.argmin(candidate_values)
                    )
                best_index = int(candidate_indices[local_best])
                best_epoch = best_index + 1
                best_monitored_value = float(monitored_values[best_index])
        if best_epoch is None and epochs_ran > 0:
            best_epoch = epochs_ran

        selected_decision_threshold = float(decision_thresholds[0])
        threshold_validation_score: float | None = None
        threshold_search_results: list[dict] = []
        if validation_subjects_per_fold > 0:
            validation_probabilities = _predict_probabilities(
                model=model,
                X=X_validation,
                batch_size=current_batch_size,
                n_prediction_latent_samples=n_prediction_latent_samples,
                latent_sampling_seed=latent_sampling_seed,
            )
            if threshold_selection_level == "trial":
                if _is_trial_tensor(X_validation):
                    threshold_validation = _direct_trial_aggregation(
                        probabilities=validation_probabilities,
                        y_true=y_validation,
                        subject_ids=subject_ids_validation,
                        trial_ids=trial_ids_validation,
                        n_windows_per_trial=X_validation.shape[1],
                    )
                else:
                    threshold_validation = _aggregate_window_probabilities_by_trial(
                        probabilities=validation_probabilities,
                        y_true=y_validation,
                        subject_ids=subject_ids_validation,
                        trial_ids=trial_ids_validation,
                    )
                threshold_probabilities = threshold_validation["probabilities"]
                threshold_y_true = threshold_validation["y_true"]
            else:
                threshold_probabilities = validation_probabilities
                threshold_y_true = _as_numpy_1d(y_validation).astype(np.int64)

            (
                selected_decision_threshold,
                threshold_validation_score,
                threshold_search_results,
            ) = _select_binary_decision_threshold(
                probabilities=threshold_probabilities,
                y_true=threshold_y_true,
                thresholds=decision_thresholds,
                metric=threshold_selection_metric,
            )
            print(
                f"Fold {fold_number} selected decision threshold "
                f"{selected_decision_threshold:.4f} from validation "
                f"{threshold_selection_level}_{threshold_selection_metric}="
                f"{threshold_validation_score:.6f}",
                flush=True,
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
            n_prediction_latent_samples=n_prediction_latent_samples,
            latent_sampling_seed=latent_sampling_seed,
            log_predictions=log_predictions,
            log_variational_intervals=log_variational_intervals,
            n_uncertainty_samples=n_uncertainty_samples,
            ci_level=ci_level,
            decision_threshold=selected_decision_threshold,
        )
    finally:
        del model
        gc.collect()
        tf.keras.backend.clear_session()

    def count_trials(subject_ids: np.ndarray, trial_ids: np.ndarray) -> int:
        return int(len(set(zip(subject_ids.tolist(), trial_ids.tolist()))))

    subject_ids_outer_train = subject_id_array[outer_train_indices]
    trial_ids_outer_train = trial_id_array[outer_train_indices]

    fold_record = {
        "fold_number": int(fold_number),
        "left_out_subjects": [_python_scalar(test_subject)],
        "validation_subjects": [
            _python_scalar(value) for value in validation_subjects.tolist()
        ],
        "n_train_windows": _count_windows_for_indices(
            feature_array, outer_train_indices
        ),
        "n_fit_train_windows": _count_windows_for_indices(
            feature_array, fit_train_indices
        ),
        "n_validation_windows": _count_windows_for_indices(
            feature_array, validation_indices
        ),
        "n_test_windows": _count_windows_for_indices(feature_array, test_indices),
        "n_train_trials": count_trials(subject_ids_outer_train, trial_ids_outer_train),
        "n_fit_train_trials": count_trials(subject_ids_fit_train, trial_ids_fit_train),
        "n_validation_trials": count_trials(
            subject_ids_validation, trial_ids_validation
        ),
        "n_test_trials": count_trials(subject_ids_test, trial_ids_test),
        "epochs_ran": int(epochs_ran),
        "best_epoch": None if best_epoch is None else int(best_epoch),
        "best_monitored_value": best_monitored_value,
        "stopped_early": bool(stopped_early),
        "decision_threshold": float(selected_decision_threshold),
        "alternate_subject_sets": bool(alternate_subject_sets),
        "use_mldg": bool(use_mldg),
        "mldg_meta_train_subjects": int(mldg_meta_train_subjects) if use_mldg else 0,
        "mldg_meta_test_subjects": int(mldg_meta_test_subjects) if use_mldg else 0,
        "mldg_samples_per_subject": int(mldg_samples_per_subject) if use_mldg else 0,
        "subject_set_a": (
            [_python_scalar(value) for value in subject_set_a.tolist()]
            if alternate_subject_sets
            else []
        ),
        "subject_set_b": (
            [_python_scalar(value) for value in subject_set_b.tolist()]
            if alternate_subject_sets
            else []
        ),
    }

    prediction_diagnostics_log = (
        []
        if prediction_diagnostics_callback is None
        else list(prediction_diagnostics_callback.history)
    )

    return {
        "outer_fold_number": int(fold_number),
        "fold_record": fold_record,
        "prediction_diagnostics_log": prediction_diagnostics_log,
        "oracle_epoch_log": list(oracle_metrics_callback.history),
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


def _compact_loso_training_result(fold_output: dict) -> dict:
    """Return the small per-fold training summary for one configuration."""
    fold_record = fold_output["fold_record"]
    return {
        "fold_number": int(fold_record["fold_number"]),
        "epochs_ran": int(fold_record["epochs_ran"]),
        "best_epoch": fold_record["best_epoch"],
        "best_monitored_value": fold_record["best_monitored_value"],
        "stopped_early": bool(fold_record["stopped_early"]),
        "decision_threshold": float(fold_record["decision_threshold"]),
    }


def _aggregate_loso_config_result(
    config_index: int,
    config: dict,
    fold_outputs: list[dict],
    metrics: tuple[str, ...],
    selection_metric: str,
    selection_level: Literal["window", "trial"],
) -> dict:
    """Aggregate a complete LOSO evaluation for one configuration."""
    fold_outputs = sorted(
        fold_outputs,
        key=lambda row: int(row["outer_fold_number"]),
    )

    fold_metrics = [dict(row["fold_metrics"]) for row in fold_outputs]
    window_fold_metrics = [dict(row["window_fold_metrics"]) for row in fold_outputs]
    trial_fold_metrics = [dict(row["trial_fold_metrics"]) for row in fold_outputs]

    mean_scores, std_scores = _mean_std_rows(
        fold_metrics,
        ["loss", "joint_loss", "keras_model_loss", *metrics, "decoder_accuracy"],
    )
    window_mean_scores, window_std_scores = _mean_std_rows(
        window_fold_metrics,
        ["loss", "joint_loss", "keras_model_loss", *metrics, "decoder_accuracy"],
    )
    trial_mean_scores, trial_std_scores = _mean_std_rows(
        trial_fold_metrics,
        ["loss", "joint_loss", "keras_model_loss", *metrics, "decoder_accuracy"],
    )

    selection_means = (
        trial_mean_scores if selection_level == "trial" else window_mean_scores
    )
    selection_stds = (
        trial_std_scores if selection_level == "trial" else window_std_scores
    )

    if selection_metric not in selection_means:
        raise ValueError(
            f"Selection metric {selection_metric!r} was not produced for "
            f"configuration {config_index}. Available metrics: "
            f"{sorted(selection_means)}"
        )

    return {
        "config_index": int(config_index),
        "config": dict(config),
        "selection_score": float(selection_means[selection_metric]),
        "selection_score_std": float(selection_stds[selection_metric]),
        "window_mean_scores": window_mean_scores,
        "window_std_scores": window_std_scores,
        "trial_mean_scores": trial_mean_scores,
        "trial_std_scores": trial_std_scores,
        "fold_metrics": fold_metrics,
        "fold_training": [_compact_loso_training_result(row) for row in fold_outputs],
        "oracle_epoch_log": [
            oracle_row
            for fold_output in fold_outputs
            for oracle_row in fold_output.get("oracle_epoch_log", [])
        ],
    }


def _loso_config_sort_key(
    config_result: dict,
    selection_metric: str,
    selection_level: Literal["window", "trial"],
    maximize_metric: bool,
) -> tuple[float, float, float, int]:
    """Return a deterministic ranking key for flat LOSO grid search.

    The primary criterion is the mean selected metric across held-out subjects.
    Ties are resolved by lower between-subject standard deviation, then lower
    mean log loss, then the earlier configuration index.
    """
    mean_key = f"{selection_level}_mean_scores"
    std_key = f"{selection_level}_std_scores"
    mean_scores = config_result[mean_key]
    std_scores = config_result[std_key]

    primary = float(mean_scores[selection_metric])
    primary_std = float(std_scores[selection_metric])
    mean_loss = float(mean_scores.get("loss", np.inf))

    if not np.isfinite(primary):
        primary_rank = np.inf
    else:
        primary_rank = -primary if maximize_metric else primary

    if not np.isfinite(primary_std):
        primary_std = np.inf
    if not np.isfinite(mean_loss):
        mean_loss = np.inf

    return (
        float(primary_rank),
        float(primary_std),
        float(mean_loss),
        int(config_result["config_index"]),
    )


def _choose_best_loso_config_index(
    config_results: list[dict],
    selection_metric: str,
    selection_level: Literal["window", "trial"],
    maximize_metric: bool,
) -> int:
    """Choose the global configuration after every config completes LOSO."""
    if not config_results:
        raise ValueError("No LOSO configuration results were produced.")

    best_result = min(
        config_results,
        key=lambda row: _loso_config_sort_key(
            config_result=row,
            selection_metric=selection_metric,
            selection_level=selection_level,
            maximize_metric=maximize_metric,
        ),
    )

    best_score = float(best_result["selection_score"])
    if not np.isfinite(best_score):
        raise RuntimeError(
            "All LOSO configurations produced a non-finite selection score."
        )

    return int(best_result["config_index"])


def _validate_loso_cv_arguments(
    subject_id_array: np.ndarray,
    trial_id_array: np.ndarray,
    selection_metric: str,
    selection_level: Literal["window", "trial"],
    maximize_metric: bool | None,
    metrics: tuple[str, ...],
    n_prediction_latent_samples: int,
    ci_level: float,
    n_jobs: int,
    cpus_per_worker: int | None,
    validation_subjects_per_fold: int,
    alternate_subject_sets: bool,
    use_mldg: bool,
    mldg_meta_train_subjects: int,
    mldg_meta_test_subjects: int,
    mldg_samples_per_subject: int,
    mldg_seed: int | None,
    validation_seed: int | None,
    early_stopping_patience: int | None,
    early_stopping_min_delta: float,
    early_stopping_monitor: str,
    early_stopping_mode: Literal["auto", "min", "max"],
    prediction_diagnostics_every_n_epochs: int,
    prediction_diagnostics_max_samples: int,
    prediction_diagnostics_threshold_tolerance: float,
    decision_thresholds: list[float] | tuple[float, ...],
    threshold_selection_level: Literal["window", "trial"],
    max_folds: int | None,
) -> tuple[bool, tuple[float, ...], np.ndarray]:
    """Validate user-facing LOSO CV options and return normalized values."""
    if n_jobs < 1:
        raise ValueError("n_jobs must be >= 1.")
    if cpus_per_worker is not None and cpus_per_worker < 1:
        raise ValueError("cpus_per_worker must be >= 1 when provided.")
    if validation_subjects_per_fold < 0:
        raise ValueError("validation_subjects_per_fold must be >= 0.")
    if alternate_subject_sets and use_mldg:
        raise ValueError("alternate_subject_sets and use_mldg are mutually exclusive.")
    if mldg_meta_train_subjects < 1 or mldg_meta_test_subjects < 1:
        raise ValueError("MLDG A/B subject counts must both be at least 1.")
    if mldg_samples_per_subject < 1:
        raise ValueError("mldg_samples_per_subject must be at least 1.")
    if mldg_seed is not None and mldg_seed < 0:
        raise ValueError("mldg_seed must be >= 0 or None.")
    if validation_seed is not None and validation_seed < 0:
        raise ValueError("validation_seed must be >= 0 or None.")
    if early_stopping_patience is not None and early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be >= 0 or None.")
    if early_stopping_min_delta < 0.0:
        raise ValueError("early_stopping_min_delta must be >= 0.")
    if early_stopping_mode not in {"auto", "min", "max"}:
        raise ValueError("early_stopping_mode must be 'auto', 'min', or 'max'.")
    if not early_stopping_monitor:
        raise ValueError("early_stopping_monitor must be a non-empty string.")
    if prediction_diagnostics_every_n_epochs < 1:
        raise ValueError("prediction_diagnostics_every_n_epochs must be at least 1.")
    if prediction_diagnostics_max_samples < 1:
        raise ValueError("prediction_diagnostics_max_samples must be at least 1.")
    if prediction_diagnostics_threshold_tolerance < 0.0:
        raise ValueError(
            "prediction_diagnostics_threshold_tolerance must be non-negative."
        )
    if n_prediction_latent_samples < 0:
        raise ValueError("n_prediction_latent_samples must be >= 0.")
    if not (0.0 < ci_level < 1.0):
        raise ValueError("ci_level must be between 0 and 1.")

    decision_thresholds = _normalize_decision_thresholds(decision_thresholds)
    _validate_evaluation_level(threshold_selection_level, "threshold_selection_level")

    if len(decision_thresholds) > 1 and validation_subjects_per_fold == 0:
        raise ValueError(
            "Testing multiple decision thresholds requires fold-local validation "
            "subjects. Set validation_subjects_per_fold >= 1."
        )
    if (
        early_stopping_monitor
        in {"val_trial_f1", "val_trial_balanced_accuracy", "val_trial_loss"}
        and validation_subjects_per_fold == 0
        and early_stopping_patience is not None
    ):
        raise ValueError(
            f"{early_stopping_monitor} requires at least one fold-local "
            "validation subject. Set validation_subjects_per_fold >= 1."
        )

    if selection_metric not in {"loss", "joint_loss", *metrics}:
        raise ValueError(
            f"selection_metric={selection_metric!r} is unavailable. "
            f"Use 'loss', 'joint_loss', or one of metrics={list(metrics)}."
        )

    if maximize_metric is None:
        maximize_metric = selection_metric not in {"loss", "joint_loss"}

    unique_subjects = np.sort(np.unique(subject_id_array))
    if len(unique_subjects) < 2:
        raise ValueError(
            "LOSO CV requires at least two unique subjects. "
            f"Got {len(unique_subjects)}."
        )
    if validation_subjects_per_fold >= len(unique_subjects) - 1:
        raise ValueError(
            "validation_subjects_per_fold must leave at least one gradient-"
            "training subject after the LOSO test subject is removed. Got "
            f"{validation_subjects_per_fold} validation subjects for "
            f"{len(unique_subjects)} total subjects."
        )
    if use_mldg:
        available_mldg_subjects = len(unique_subjects) - 1 - validation_subjects_per_fold
        required_mldg_subjects = int(mldg_meta_train_subjects) + int(
            mldg_meta_test_subjects
        )
        if required_mldg_subjects > available_mldg_subjects:
            raise ValueError(
                "MLDG requires "
                f"{required_mldg_subjects} episodic subjects, but each LOSO "
                f"fold leaves only {available_mldg_subjects} gradient-training "
                "subjects after removing test and validation subjects."
            )

    if max_folds is not None and max_folds < 1:
        raise ValueError("max_folds must be >= 1 when provided.")

    test_subjects = (
        unique_subjects[: min(max_folds, len(unique_subjects))]
        if max_folds is not None
        else unique_subjects
    )

    return maximize_metric, decision_thresholds, test_subjects


def _print_loso_cv_summary(
    num_configs: int,
    total_folds: int,
    num_subjects: int,
    max_folds: int | None,
    metrics: tuple[str, ...],
    selection_level: str,
    selection_metric: str,
    maximize_metric: bool,
    evaluation_level: str,
    log_predictions: bool,
    prediction_diagnostics: bool,
    log_variational_intervals: bool,
    decision_thresholds: tuple[float, ...],
    threshold_selection_level: str,
    threshold_selection_metric: str,
    n_prediction_latent_samples: int,
    validation_subjects_per_fold: int,
    validation_seed: int | None,
    early_stopping_monitor: str,
    early_stopping_patience: int | None,
    restore_best_weights: bool,
    use_mldg: bool,
    mldg_meta_train_subjects: int,
    mldg_meta_test_subjects: int,
    mldg_samples_per_subject: int,
    mldg_seed: int | None,
    alternate_subject_sets: bool,
    effective_n_jobs: int,
    normalized_gpu_ids: tuple[int, ...] | None,
) -> None:
    print(
        f"\nFlat LOSO hyperparameter search — {num_configs} "
        f"configuration{'s' if num_configs != 1 else ''}, "
        f"{total_folds} fold{'s' if total_folds != 1 else ''} each"
    )
    print(f"Total available subjects: {num_subjects}")
    print(f"Total LOSO model fits: {num_configs * total_folds}")
    if max_folds is not None:
        print(
            f"Smoke-test fold limit: {total_folds} of "
            f"{num_subjects} subjects per configuration"
        )
    print(f"Requested metrics: {list(metrics)}")
    print(
        f"Configuration selection: {selection_level}-level "
        f"{selection_metric} "
        f"({'maximize' if maximize_metric else 'minimize'})"
    )
    print(f"Primary reported metrics: {evaluation_level}-level")
    print(f"Prediction logging: {log_predictions}")
    print(f"Prediction diagnostics: {prediction_diagnostics}")
    print(f"Variational interval logging: {log_variational_intervals}")
    print(
        "Decision thresholds: "
        f"{list(decision_thresholds)}; selection="
        f"{threshold_selection_level}_{threshold_selection_metric}"
    )
    print(
        "Prediction latent mode: "
        + (
            "posterior mean"
            if n_prediction_latent_samples == 0
            else f"MC average over {n_prediction_latent_samples} latent sample(s)"
        )
    )
    if validation_subjects_per_fold > 0:
        print(
            "Per-fold validation: "
            f"{validation_subjects_per_fold} seeded subject(s), "
            f"seed={validation_seed}, monitor={early_stopping_monitor}, "
            f"patience={early_stopping_patience}, "
            f"restore_best_weights={restore_best_weights}"
        )
    else:
        print("Per-fold validation: disabled")
    if use_mldg:
        print(
            "Optimization: first-order MLDG with natural within-subject labels "
            f"(A={mldg_meta_train_subjects} subjects, "
            f"B={mldg_meta_test_subjects} subjects, "
            f"samples/subject={mldg_samples_per_subject}, seed={mldg_seed})"
        )
    elif alternate_subject_sets:
        print("Optimization: alternating fixed subject sets")
    else:
        print("Optimization: ordinary shuffled minibatches")
    print(f"Fold workers: {effective_n_jobs}")
    if effective_n_jobs > 1 and normalized_gpu_ids is None:
        print("Worker devices: CPU-only")
    elif normalized_gpu_ids is not None:
        print(f"Worker devices: GPUs {list(normalized_gpu_ids)}")
    else:
        print("Worker device: current TensorFlow default")


def _build_loso_common_worker_state(
    model_builder_function: Callable[..., tf.keras.Model],
    feature_array: np.ndarray,
    label_array: np.ndarray,
    subject_id_array: np.ndarray,
    trial_id_array: np.ndarray,
    batch_size: int,
    preprocessing_strategy: Callable | None,
    evaluation_level: Literal["window", "trial"],
    metrics: tuple[str, ...],
    log_predictions: bool,
    log_variational_intervals: bool,
    n_prediction_latent_samples: int,
    latent_sampling_seed: int | None,
    n_uncertainty_samples: int,
    ci_level: float,
    validation_subjects_per_fold: int,
    validation_seed: int | None,
    early_stopping_patience: int | None,
    early_stopping_min_delta: float,
    early_stopping_monitor: str,
    early_stopping_mode: Literal["auto", "min", "max"],
    restore_best_weights: bool,
    prediction_diagnostics: bool,
    prediction_diagnostics_every_n_epochs: int,
    prediction_diagnostics_max_samples: int,
    prediction_diagnostics_threshold_tolerance: float,
    prediction_diagnostics_seed: int | None,
    decision_thresholds: tuple[float, ...],
    threshold_selection_metric: str,
    threshold_selection_level: Literal["window", "trial"],
    verbose: int,
    extra_fit_kwargs: dict,
    alternate_subject_sets: bool,
    alternating_subject_seed: int | None,
    use_mldg: bool,
    mldg_meta_train_subjects: int,
    mldg_meta_test_subjects: int,
    mldg_samples_per_subject: int,
    mldg_seed: int | None,
) -> dict:
    return {
        "total_folds": None,
        "model_builder_function": model_builder_function,
        "feature_array": feature_array,
        "label_array": label_array,
        "subject_id_array": subject_id_array,
        "trial_id_array": trial_id_array,
        "batch_size": batch_size,
        "preprocessing_strategy": preprocessing_strategy,
        "evaluation_level": evaluation_level,
        "metrics": metrics,
        "log_predictions": log_predictions,
        "log_variational_intervals": log_variational_intervals,
        "n_prediction_latent_samples": n_prediction_latent_samples,
        "latent_sampling_seed": latent_sampling_seed,
        "n_uncertainty_samples": n_uncertainty_samples,
        "ci_level": ci_level,
        "validation_subjects_per_fold": validation_subjects_per_fold,
        "validation_seed": validation_seed,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "early_stopping_monitor": early_stopping_monitor,
        "early_stopping_mode": early_stopping_mode,
        "restore_best_weights": restore_best_weights,
        "prediction_diagnostics": bool(prediction_diagnostics),
        "prediction_diagnostics_every_n_epochs": int(prediction_diagnostics_every_n_epochs),
        "prediction_diagnostics_max_samples": int(prediction_diagnostics_max_samples),
        "prediction_diagnostics_threshold_tolerance": float(
            prediction_diagnostics_threshold_tolerance
        ),
        "prediction_diagnostics_seed": prediction_diagnostics_seed,
        "decision_thresholds": decision_thresholds,
        "threshold_selection_metric": threshold_selection_metric,
        "threshold_selection_level": threshold_selection_level,
        "verbose": verbose,
        "extra_fit_kwargs": extra_fit_kwargs,
        "alternate_subject_sets": bool(alternate_subject_sets),
        "alternating_subject_seed": alternating_subject_seed,
        "use_mldg": bool(use_mldg),
        "mldg_meta_train_subjects": int(mldg_meta_train_subjects),
        "mldg_meta_test_subjects": int(mldg_meta_test_subjects),
        "mldg_samples_per_subject": int(mldg_samples_per_subject),
        "mldg_seed": mldg_seed,
    }


def loso_cv(
    model_builder_function: Callable[..., tf.keras.Model],
    feature_array: np.ndarray,
    label_array: np.ndarray,
    subject_id_array: np.ndarray,
    trial_id_array: np.ndarray | None = None,
    n_epochs: int = 50,
    batch_size: int = 2,
    hyperparameters: dict | None = None,
    preprocessing_strategy: Callable | None = None,
    evaluation_level: Literal["window", "trial"] = "trial",
    selection_metric: str = "f1",
    selection_level: Literal["window", "trial"] = "trial",
    maximize_metric: bool | None = None,
    metrics: list[str] | tuple[str, ...] = (
        "accuracy",
        "f1",
        "precision",
        "recall",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "balanced_accuracy",
        "binary_f1",
        "binary_precision",
        "binary_recall",
        "roc_auc",
    ),
    log_predictions: bool = True,
    log_variational_intervals: bool = False,
    n_prediction_latent_samples: int = 0,
    latent_sampling_seed: int | None = None,
    n_uncertainty_samples: int = 30,
    ci_level: float = 0.95,
    validation_subjects_per_fold: int = 0,
    validation_seed: int | None = 42,
    early_stopping_patience: int | None = 5,
    early_stopping_min_delta: float = 0.0,
    early_stopping_monitor: str = "val_loss",
    early_stopping_mode: Literal["auto", "min", "max"] = "min",
    restore_best_weights: bool = True,
    prediction_diagnostics: bool = False,
    prediction_diagnostics_every_n_epochs: int = 1,
    prediction_diagnostics_max_samples: int = 256,
    prediction_diagnostics_threshold_tolerance: float = 0.01,
    prediction_diagnostics_seed: int | None = 42,
    decision_thresholds: list[float] | tuple[float, ...] = (0.5,),
    threshold_selection_metric: Literal[
        "accuracy", "f1", "balanced_accuracy", "binary_f1"
    ] = "f1",
    threshold_selection_level: Literal["window", "trial"] = "trial",
    verbose: int = 0,
    extra_fit_kwargs: dict | None = None,
    n_jobs: int = 1,
    gpu_ids: list[int] | tuple[int, ...] | None = None,
    cpus_per_worker: int | None = None,
    max_folds: int | None = None,
    alternate_subject_sets: bool = False,
    alternating_subject_seed: int | None = 42,
    use_mldg: bool = False,
    mldg_meta_train_subjects: int = 6,
    mldg_meta_test_subjects: int = 2,
    mldg_samples_per_subject: int = 4,
    mldg_seed: int | None = 42,
) -> dict:
    """Run a flat hyperparameter search using complete LOSO evaluations.

    For every Cartesian-product hyperparameter configuration, each unique
    subject is held out exactly once. The configuration is therefore evaluated
    on the same complete set of subject-wise folds. After all configurations
    finish, one global configuration is selected from its mean LOSO metric.

    This is *not* nested cross-validation: the held-out LOSO results are used
    both to compare configurations and to report the selected configuration's
    cross-validation performance. This behavior is intentional for a practical
    flat LOSO hyperparameter search.

    Hyperparameter grid
    -------------------
    Scalar values may be supplied directly or as candidate lists/tuples. The
    Cartesian product is evaluated with a complete LOSO run per configuration.

    Sequence-valued encoder settings preserve one complete architecture before
    the Cartesian product is expanded. ``sequence_hyperparameter_depths``
    specifies the nesting depth of one value, resolving CNN1D/CNN2D ambiguity
    for keys such as ``kernel_sizes``. GCN ``gcn_units`` and temporal/spatial
    pooling schedules are preserved in the same way. One additional outer list
    level enumerates multiple architecture candidates.

    ``n_epochs`` and ``batch_size`` provide defaults and are overridden when
    ``hyperparameters`` contains ``epochs`` or ``batch_size``.

    Seeded validation and early stopping
    ------------------------------------
    When ``validation_subjects_per_fold`` is positive, that many subjects are
    sampled deterministically from each outer-training pool. They are excluded
    from gradient updates and passed to ``model.fit`` as ``validation_data``.
    The same fold-local validation subjects are reused for every hyperparameter
    configuration, while the LOSO test subject remains untouched. This adds no
    extra fits; it only changes each fit from train/test to train/validation/test.

    Selection
    ---------
    ``selection_level`` determines whether configurations are ranked using
    window- or trial-level scores. Hierarchical rank-4 inputs require trial-level
    selection. For binary tasks, ``selection_metric='f1'`` uses the MTLFuseNet
    convention: class 1 is positive. ``precision`` and ``recall`` follow the same
    convention. Explicit ``macro_*`` metrics and ``balanced_accuracy`` remain
    available for class-balanced diagnostics, while ``roc_auc`` uses the class-1
    probability.
    Classification metrics are maximized; probability loss and joint loss are minimized unless
    ``maximize_metric`` is explicitly supplied. Ties use lower between-subject
    standard deviation, lower mean log loss, then the earlier grid index.

    Returned results
    ----------------
    ``config_results`` contains per-fold and aggregate metrics for every
    configuration. Top-level prediction logs, user metrics, and fold metadata
    correspond only to the globally selected configuration. Selected fold
    metrics remain available through ``config_results[best_config_index]``.

    Concurrency
    -----------
    LOSO folds for one configuration run concurrently. The next configuration
    starts after the current configuration's folds complete. With one worker per
    GPU, this prevents multiple models from competing for the same GPU while
    bounding parent-process memory to approximately one configuration's logs.

    Smoke testing
    -------------
    ``max_folds`` deterministically limits every configuration to the first N
    sorted subjects. Leave it as ``None`` for complete LOSO evaluation.
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
            "Pass one trial ID per sample, aligned with feature_array."
        )

    _validate_evaluation_level(evaluation_level, "evaluation_level")
    _validate_evaluation_level(selection_level, "selection_level")

    feature_array = np.asarray(feature_array)
    label_array = np.asarray(label_array)
    subject_id_array = np.asarray(subject_id_array)
    trial_id_array = np.asarray(trial_id_array)

    if feature_array.ndim not in {3, 4}:
        raise ValueError(
            "feature_array must be rank 3 for window samples or rank 4 for "
            f"grouped trial samples; got {feature_array.shape}."
        )
    if feature_array.ndim == 4:
        if selection_level != "trial":
            raise ValueError(
                "Grouped rank-4 trial inputs require selection_level='trial'."
            )
        if evaluation_level != "trial":
            raise ValueError(
                "Grouped rank-4 trial inputs require evaluation_level='trial'."
            )

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

    unique_subjects = np.sort(np.unique(subject_id_array))

    maximize_metric, decision_thresholds, test_subjects = _validate_loso_cv_arguments(
        subject_id_array=subject_id_array,
        trial_id_array=trial_id_array,
        selection_metric=selection_metric,
        selection_level=selection_level,
        maximize_metric=maximize_metric,
        metrics=metrics,
        n_prediction_latent_samples=n_prediction_latent_samples,
        ci_level=ci_level,
        n_jobs=n_jobs,
        cpus_per_worker=cpus_per_worker,
        validation_subjects_per_fold=validation_subjects_per_fold,
        alternate_subject_sets=alternate_subject_sets,
        use_mldg=use_mldg,
        mldg_meta_train_subjects=mldg_meta_train_subjects,
        mldg_meta_test_subjects=mldg_meta_test_subjects,
        mldg_samples_per_subject=mldg_samples_per_subject,
        mldg_seed=mldg_seed,
        validation_seed=validation_seed,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        early_stopping_monitor=early_stopping_monitor,
        early_stopping_mode=early_stopping_mode,
        prediction_diagnostics_every_n_epochs=prediction_diagnostics_every_n_epochs,
        prediction_diagnostics_max_samples=prediction_diagnostics_max_samples,
        prediction_diagnostics_threshold_tolerance=prediction_diagnostics_threshold_tolerance,
        decision_thresholds=decision_thresholds,
        threshold_selection_level=threshold_selection_level,
        max_folds=max_folds,
    )

    effective_hyperparameters = {
        "epochs": n_epochs,
        "batch_size": batch_size,
        **(hyperparameters or {}),
    }
    sequence_hyperparameter_depths = getattr(
        model_builder_function,
        "_sequence_hyperparameter_depths",
        None,
    )
    grid_configs = _expand_hyperparameter_grid(
        effective_hyperparameters,
        sequence_hyperparameter_depths=sequence_hyperparameter_depths,
    )
    _warn_if_joint_loss_weights_vary(grid_configs, selection_metric)
    if not grid_configs:
        raise ValueError("The hyperparameter grid produced no configurations.")

    total_folds = len(test_subjects)
    total_model_fits = len(grid_configs) * total_folds
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

    _print_loso_cv_summary(
        num_configs=len(grid_configs),
        total_folds=total_folds,
        num_subjects=len(unique_subjects),
        max_folds=max_folds,
        metrics=metrics,
        selection_level=selection_level,
        selection_metric=selection_metric,
        maximize_metric=maximize_metric,
        evaluation_level=evaluation_level,
        log_predictions=log_predictions,
        prediction_diagnostics=prediction_diagnostics,
        log_variational_intervals=log_variational_intervals,
        decision_thresholds=decision_thresholds,
        threshold_selection_level=threshold_selection_level,
        threshold_selection_metric=threshold_selection_metric,
        n_prediction_latent_samples=n_prediction_latent_samples,
        validation_subjects_per_fold=validation_subjects_per_fold,
        validation_seed=validation_seed,
        early_stopping_monitor=early_stopping_monitor,
        early_stopping_patience=early_stopping_patience,
        restore_best_weights=restore_best_weights,
        use_mldg=use_mldg,
        mldg_meta_train_subjects=mldg_meta_train_subjects,
        mldg_meta_test_subjects=mldg_meta_test_subjects,
        mldg_samples_per_subject=mldg_samples_per_subject,
        mldg_seed=mldg_seed,
        alternate_subject_sets=alternate_subject_sets,
        effective_n_jobs=effective_n_jobs,
        normalized_gpu_ids=normalized_gpu_ids,
    )

    tasks = [
        (fold_number, _python_scalar(test_subject))
        for fold_number, test_subject in enumerate(test_subjects, start=1)
    ]

    common_worker_state = {
        "total_folds": total_folds,
        "model_builder_function": model_builder_function,
        "feature_array": feature_array,
        "label_array": label_array,
        "subject_id_array": subject_id_array,
        "trial_id_array": trial_id_array,
        "batch_size": batch_size,
        "preprocessing_strategy": preprocessing_strategy,
        "evaluation_level": evaluation_level,
        "metrics": metrics,
        "log_predictions": log_predictions,
        "log_variational_intervals": log_variational_intervals,
        "n_prediction_latent_samples": n_prediction_latent_samples,
        "latent_sampling_seed": latent_sampling_seed,
        "n_uncertainty_samples": n_uncertainty_samples,
        "ci_level": ci_level,
        "validation_subjects_per_fold": validation_subjects_per_fold,
        "validation_seed": validation_seed,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "early_stopping_monitor": early_stopping_monitor,
        "early_stopping_mode": early_stopping_mode,
        "restore_best_weights": restore_best_weights,
        "prediction_diagnostics": bool(prediction_diagnostics),
        "prediction_diagnostics_every_n_epochs": int(
            prediction_diagnostics_every_n_epochs
        ),
        "prediction_diagnostics_max_samples": int(prediction_diagnostics_max_samples),
        "prediction_diagnostics_threshold_tolerance": float(
            prediction_diagnostics_threshold_tolerance
        ),
        "prediction_diagnostics_seed": prediction_diagnostics_seed,
        "decision_thresholds": decision_thresholds,
        "threshold_selection_metric": threshold_selection_metric,
        "threshold_selection_level": threshold_selection_level,
        "verbose": verbose,
        "extra_fit_kwargs": extra_fit_kwargs,
        "alternate_subject_sets": bool(alternate_subject_sets),
        "alternating_subject_seed": alternating_subject_seed,
        "use_mldg": bool(use_mldg),
        "mldg_meta_train_subjects": int(mldg_meta_train_subjects),
        "mldg_meta_test_subjects": int(mldg_meta_test_subjects),
        "mldg_samples_per_subject": int(mldg_samples_per_subject),
        "mldg_seed": mldg_seed,
    }

    config_results: list[dict] = []
    best_so_far_result: dict | None = None
    best_fold_outputs: list[dict] | None = None

    for config_index, config in enumerate(grid_configs):
        print("\n" + "#" * 80)
        print(
            f"Configuration {config_index + 1} / {len(grid_configs)} "
            f"({total_folds} LOSO fits)"
        )
        _print_config("Configuration:", config)

        worker_state = {
            **common_worker_state,
            "fixed_config": config,
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
                worker_name_prefix=f"LOSOConfig{config_index + 1}Worker",
                worker_description=(f"LOSO-fold for configuration {config_index + 1}"),
            )

        fold_outputs.sort(key=lambda row: row["outer_fold_number"])
        config_result = _aggregate_loso_config_result(
            config_index=config_index,
            config=config,
            fold_outputs=fold_outputs,
            metrics=metrics,
            selection_metric=selection_metric,
            selection_level=selection_level,
        )
        config_results.append(config_result)

        if best_so_far_result is None or _loso_config_sort_key(
            config_result=config_result,
            selection_metric=selection_metric,
            selection_level=selection_level,
            maximize_metric=bool(maximize_metric),
        ) < _loso_config_sort_key(
            config_result=best_so_far_result,
            selection_metric=selection_metric,
            selection_level=selection_level,
            maximize_metric=bool(maximize_metric),
        ):
            best_so_far_result = config_result
            best_fold_outputs = fold_outputs

        print(
            f"\nConfiguration {config_index + 1} complete: "
            f"mean {selection_level}_{selection_metric}="
            f"{config_result['selection_score']:.6f} ± "
            f"{config_result['selection_score_std']:.6f}",
            flush=True,
        )

    best_config_index = _choose_best_loso_config_index(
        config_results=config_results,
        selection_metric=selection_metric,
        selection_level=selection_level,
        maximize_metric=bool(maximize_metric),
    )
    best_config_result = config_results[best_config_index]
    best_config = dict(best_config_result["config"])

    if (
        best_so_far_result is None
        or best_fold_outputs is None
        or int(best_so_far_result["config_index"]) != best_config_index
    ):
        raise RuntimeError(
            "Internal LOSO grid-search error: selected configuration logs "
            "were not retained correctly."
        )

    # Surface one canonical copy of each selected-configuration artifact.
    results = {
        "cv_strategy": "flat_loso_hyperparameter_search",
        "hyperparameter_search": True,
        "n_configs": int(len(grid_configs)),
        "n_subjects": int(len(unique_subjects)),
        "n_evaluated_folds_per_config": int(total_folds),
        "n_total_loso_fits": int(total_model_fits),
        "max_folds": max_folds,
        "selection_metric": selection_metric,
        "selection_level": selection_level,
        "evaluation_level": evaluation_level,
        "maximize_metric": bool(maximize_metric),
        "selection_score": float(best_config_result["selection_score"]),
        "selection_score_std": float(best_config_result["selection_score_std"]),
        "n_prediction_latent_samples": int(n_prediction_latent_samples),
        "latent_sampling_seed": latent_sampling_seed,
        "validation_subjects_per_fold": int(validation_subjects_per_fold),
        "validation_seed": validation_seed,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": float(early_stopping_min_delta),
        "early_stopping_monitor": early_stopping_monitor,
        "early_stopping_mode": early_stopping_mode,
        "restore_best_weights": bool(restore_best_weights),
        "use_mldg": bool(use_mldg),
        "mldg_meta_train_subjects": int(mldg_meta_train_subjects) if use_mldg else 0,
        "mldg_meta_test_subjects": int(mldg_meta_test_subjects) if use_mldg else 0,
        "mldg_samples_per_subject": int(mldg_samples_per_subject) if use_mldg else 0,
        "mldg_seed": mldg_seed if use_mldg else None,
        "config_results": config_results,
        "best_config_index": int(best_config_index),
        "best_config": best_config,
        "user_metrics": [],
        "fold_results": [],
    }
    if log_predictions:
        if feature_array.ndim == 3:
            results["window_prediction_log"] = []
        results["trial_prediction_log"] = []
    if log_variational_intervals:
        if feature_array.ndim == 3:
            results["window_variational_interval_log"] = []
        results["trial_variational_interval_log"] = []
    if prediction_diagnostics:
        results["prediction_diagnostics_log"] = []

    for fold_output in best_fold_outputs:
        results["user_metrics"].extend(fold_output["user_metrics"])
        if log_predictions:
            if feature_array.ndim == 3:
                results["window_prediction_log"].extend(
                    fold_output["window_prediction_log"]
                )
            results["trial_prediction_log"].extend(fold_output["trial_prediction_log"])
        if log_variational_intervals:
            if feature_array.ndim == 3:
                results["window_variational_interval_log"].extend(
                    fold_output["window_variational_interval_log"]
                )
            results["trial_variational_interval_log"].extend(
                fold_output["trial_variational_interval_log"]
            )
        if prediction_diagnostics:
            results["prediction_diagnostics_log"].extend(
                fold_output.get("prediction_diagnostics_log", [])
            )
        results["fold_results"].append(dict(fold_output["fold_record"]))

    print("\nFlat LOSO hyperparameter search complete")
    print("=" * 80)
    print(
        f"Selected configuration {best_config_index + 1} / "
        f"{len(grid_configs)} using {selection_level}-level "
        f"{selection_metric}."
    )
    _print_config("Best configuration:", best_config)
    print(
        f"Selection score: {best_config_result['selection_score']:.6f} ± "
        f"{best_config_result['selection_score_std']:.6f}"
    )
    print("Selected configuration primary mean scores:")
    print(
        pformat(
            best_config_result[f"{evaluation_level}_mean_scores"],
            indent=4,
            width=120,
            sort_dicts=False,
        )
    )
    print("Selected configuration primary score standard deviations:")
    print(
        pformat(
            best_config_result[f"{evaluation_level}_std_scores"],
            indent=4,
            width=120,
            sort_dicts=False,
        )
    )
    print("Selected configuration window-level mean scores:")
    print(
        pformat(
            best_config_result["window_mean_scores"],
            indent=4,
            width=120,
            sort_dicts=False,
        )
    )
    print("Selected configuration trial-level mean scores:")
    print(
        pformat(
            best_config_result["trial_mean_scores"],
            indent=4,
            width=120,
            sort_dicts=False,
        )
    )

    return results


# ---------------------------------------------------------------------
# Subject-independent pretraining + few-shot subject calibration
# ---------------------------------------------------------------------


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


def _normalize_calibration_levels(
    calibration_levels,
    *,
    calibration_trials: int,
    calibration_folds: int,
) -> tuple[tuple[int, int], ...]:
    """Return unique ``(shots, folds)`` pairs in requested reporting order."""
    raw_levels = (
        [(calibration_trials, calibration_folds)]
        if calibration_levels is None
        else list(calibration_levels)
    )
    if not raw_levels:
        raise ValueError("calibration_levels must contain at least one pair.")

    normalized: list[tuple[int, int]] = []
    seen_shots: set[int] = set()
    for pair in raw_levels:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(
                "Each calibration level must be a (shots, folds) pair; "
                f"got {pair!r}."
            )
        shots, folds = (int(pair[0]), int(pair[1]))
        if shots < 1:
            raise ValueError("Calibration shots must be >= 1.")
        if folds < 1:
            raise ValueError("Calibration folds must be >= 1.")
        if shots in seen_shots:
            raise ValueError(
                f"Calibration shots must be unique; {shots}-shot was repeated."
            )
        seen_shots.add(shots)
        normalized.append((shots, folds))
    return tuple(normalized)


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


def _tag_subject_calibration_evaluation(
    evaluation: dict,
    *,
    target_subject,
    calibration_fold: int | None,
    stage: str,
    calibration_shots: int | None = None,
) -> dict:
    """Annotate evaluation rows so zero-shot/calibrated logs stay separable."""
    tagged = dict(evaluation)
    for key in ("fold_metrics", "window_fold_metrics", "trial_fold_metrics"):
        if key in tagged:
            tagged[key] = {
                **dict(tagged[key]),
                "target_subject": _python_scalar(target_subject),
                "calibration_fold": calibration_fold,
                "calibration_shots": calibration_shots,
                "stage": stage,
            }

    for key in (
        "user_metrics",
        "window_prediction_log",
        "trial_prediction_log",
        "window_variational_interval_log",
        "trial_variational_interval_log",
    ):
        rows = []
        for row in tagged.get(key, []):
            rows.append(
                {
                    **dict(row),
                    "target_subject": _python_scalar(target_subject),
                    "calibration_fold": calibration_fold,
                    "calibration_shots": calibration_shots,
                    "stage": stage,
                }
            )
        tagged[key] = rows
    return tagged


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


def _save_zero_shot_source_model(
    model: tf.keras.Model,
    *,
    output_dir: str | os.PathLike[str],
    subject_number: int,
    target_subject,
) -> dict:
    """Persist the untouched source-trained model for one LOSO target."""
    resolved_dir = os.path.abspath(os.fspath(output_dir))
    os.makedirs(resolved_dir, exist_ok=True)
    target_token = _safe_model_filename_token(target_subject)
    filename = (
        f"loso_fold_{int(subject_number):04d}_target_{target_token}_zero_shot.keras"
    )
    model_path = os.path.join(resolved_dir, filename)
    model.save(model_path, overwrite=True)
    return {
        "stage": "zero_shot_source_model",
        "format": "keras",
        "path": model_path,
        "filename": filename,
        "target_subject": _python_scalar(target_subject),
        "loso_fold": int(subject_number),
        "load_with_compile": False,
    }


def _run_subject_calibration_subject(
    subject_number: int,
    target_subject,
    total_subjects: int,
    *,
    model_builder_function: Callable[..., tf.keras.Model],
    feature_array: np.ndarray,
    label_array: np.ndarray,
    subject_id_array: np.ndarray,
    trial_id_array: np.ndarray,
    fixed_config: dict,
    source_epochs: int,
    source_batch_size: int,
    validation_subjects_per_fold: int,
    validation_seed: int | None,
    early_stopping_patience: int | None,
    early_stopping_min_delta: float,
    early_stopping_monitor: str,
    early_stopping_mode: Literal["auto", "min", "max"],
    restore_best_weights: bool,
    calibration_epochs: int,
    calibration_batch_size: int,
    calibration_trials: int,
    calibration_folds: int,
    calibration_levels: tuple[tuple[int, int], ...],
    calibration_learning_rate: float,
    calibration_optimizer: str,
    calibration_weight_decay: float,
    calibration_seed: int | None,
    stratify_calibration: bool,
    allow_overlapping_calibration_folds: bool,
    evaluation_level: Literal["window", "trial"],
    metrics: tuple[str, ...],
    ece_bins: int,
    decision_threshold: float,
    prediction_diagnostics_thresholds: tuple[float, ...],
    prediction_diagnostics: bool,
    prediction_diagnostics_metric: str,
    prediction_diagnostics_every_n_epochs: int,
    prediction_diagnostics_max_samples: int,
    prediction_diagnostics_threshold_tolerance: float,
    prediction_diagnostics_seed: int | None,
    n_prediction_latent_samples: int,
    latent_sampling_seed: int | None,
    log_predictions: bool,
    log_variational_intervals: bool,
    n_uncertainty_samples: int,
    ci_level: float,
    source_use_class_weight: bool,
    calibration_use_class_weight: bool,
    source_fit_kwargs: dict,
    calibration_fit_kwargs: dict,
    source_model_output_dir: str | os.PathLike[str] | None,
    verbose: int,
) -> dict:
    """Train one source model and run all target-subject calibration folds."""
    target_mask = subject_id_array == target_subject
    target_indices = np.flatnonzero(target_mask)
    outer_source_indices = np.flatnonzero(~target_mask)
    if not len(target_indices) or not len(outer_source_indices):
        raise ValueError(
            f"Invalid source/target split for subject {target_subject!r}: "
            f"source={len(outer_source_indices)}, target={len(target_indices)}."
        )

    outer_source_subjects = np.sort(
        np.unique(subject_id_array[outer_source_indices])
    )
    if validation_subjects_per_fold < 0:
        raise ValueError("validation_subjects_per_fold must be >= 0.")
    if validation_subjects_per_fold >= len(outer_source_subjects):
        raise ValueError(
            "validation_subjects_per_fold must leave at least one non-target "
            f"subject for source fitting; got {validation_subjects_per_fold} "
            f"from {len(outer_source_subjects)} source subjects."
        )
    if validation_subjects_per_fold:
        base_seed = 0 if validation_seed is None else int(validation_seed)
        validation_rng = np.random.default_rng(
            np.random.SeedSequence([base_seed, int(subject_number)])
        )
        validation_subjects = np.sort(
            validation_rng.choice(
                outer_source_subjects,
                size=int(validation_subjects_per_fold),
                replace=False,
            )
        )
        validation_mask = np.isin(
            subject_id_array[outer_source_indices],
            validation_subjects,
        )
        source_validation_indices = outer_source_indices[validation_mask]
        source_indices = outer_source_indices[~validation_mask]
    else:
        validation_subjects = np.asarray([], dtype=outer_source_subjects.dtype)
        source_validation_indices = np.asarray([], dtype=np.int64)
        source_indices = outer_source_indices

    X_source = feature_array[source_indices]
    y_source = label_array[source_indices]
    source_subject_ids = subject_id_array[source_indices]
    source_trial_ids = trial_id_array[source_indices]
    X_source_validation = feature_array[source_validation_indices]
    y_source_validation = label_array[source_validation_indices]
    source_validation_subject_ids = subject_id_array[source_validation_indices]
    source_validation_trial_ids = trial_id_array[source_validation_indices]

    X_target = feature_array[target_indices]
    y_target = label_array[target_indices]
    target_subject_ids = subject_id_array[target_indices]
    target_trial_ids = trial_id_array[target_indices]

    fold_seed = (
        None
        if calibration_seed is None
        else int(calibration_seed) + int(subject_number) - 1
    )
    calibration_splits_by_level: list[tuple[int, int, list[dict]]] = []
    target_trial_count = int(len(np.unique(target_trial_ids)))
    for level_index, (shots, folds) in enumerate(calibration_levels):
        level_seed = (
            None
            if fold_seed is None
            else int(fold_seed) + level_index * 1_000_003
        )
        calibration_splits_by_level.append(
            (
                int(shots),
                int(folds),
                _make_subject_calibration_splits(
                    labels=y_target,
                    trial_ids=target_trial_ids,
                    calibration_trials=int(shots),
                    calibration_folds=int(folds),
                    seed=level_seed,
                    stratify=stratify_calibration,
                    allow_overlapping_folds=(
                        bool(allow_overlapping_calibration_folds)
                        or int(shots) * int(folds) != target_trial_count
                    ),
                ),
            )
        )

    model_hp, ignored_fit_hp = _split_config(fixed_config)
    if ignored_fit_hp:
        raise ValueError(
            "subject_calibration_cv uses explicit source/calibration epoch and "
            "batch-size arguments. Remove epochs/batch_size from fixed_config."
        )

    tf.keras.backend.clear_session()
    try:
        try:
            model = model_builder_function(
                training_features=X_source,
                training_labels=y_source,
                training_subject_ids=source_subject_ids,
                training_trial_ids=source_trial_ids,
                **model_hp,
            )
        except TypeError as exc:
            raise TypeError(
                "For subject_calibration_cv, model_builder_function must accept "
                "training_features, training_labels, training_subject_ids, and "
                "training_trial_ids (directly or through **kwargs). v6 uses "
                "training_features to construct the MTLFuseNet MI adjacency "
                "without target-subject leakage."
            ) from exc

        X_source_for_fit = _prepare_fit_inputs_with_subject_ids(
            model,
            X_source,
            source_subject_ids,
        )
        source_kwargs = dict(source_fit_kwargs)
        source_callbacks = list(source_kwargs.pop("callbacks", []))
        prediction_diagnostics_callback = None
        if prediction_diagnostics:
            from .training_outputs import PredictionDiagnostics

            prediction_diagnostics_callback = PredictionDiagnostics(
                X_train=X_source,
                y_train=y_source,
                X_val=(X_source_validation if len(source_validation_indices) else None),
                y_val=(y_source_validation if len(source_validation_indices) else None),
                fold_number=int(subject_number),
                batch_size=int(source_batch_size),
                every_n_epochs=int(prediction_diagnostics_every_n_epochs),
                max_samples=int(prediction_diagnostics_max_samples),
                threshold_tolerance=float(
                    prediction_diagnostics_threshold_tolerance
                ),
                reported_metric=str(prediction_diagnostics_metric),
                decision_threshold=float(decision_threshold),
                decision_thresholds=prediction_diagnostics_thresholds,
                ece_bins=int(ece_bins),
                seed=(
                    None
                    if prediction_diagnostics_seed is None
                    else int(prediction_diagnostics_seed) + int(subject_number)
                ),
            )
            source_callbacks.append(prediction_diagnostics_callback)

        oracle_metrics_callback = HeldOutUserOracleMetrics(
            X_target=X_target,
            y_target=y_target,
            subject_ids_target=target_subject_ids,
            trial_ids_target=target_trial_ids,
            target_subject=target_subject,
            evaluation_level=evaluation_level,
            batch_size=int(1),
            decision_threshold=float(decision_threshold),
            diagnostic_thresholds=prediction_diagnostics_thresholds,
            ece_bins=int(ece_bins),
        )
        source_callbacks.append(oracle_metrics_callback)

        source_class_weight = (
            _class_weight_from_labels(y_source) if source_use_class_weight else None
        )
        if source_class_weight is not None:
            source_kwargs["class_weight"] = source_class_weight

        source_validation_data = None
        if len(source_validation_indices):
            source_validation_data = (X_source_validation, y_source_validation)
            source_callbacks.append(
                EvaluationLevelValidationMetrics(
                    X_val=X_source_validation,
                    y_val=y_source_validation,
                    subject_ids_val=source_validation_subject_ids,
                    trial_ids_val=source_validation_trial_ids,
                    evaluation_level=evaluation_level,
                    batch_size=int(source_batch_size),
                    ece_bins=ece_bins,
                )
            )
            if early_stopping_patience is not None:
                source_callbacks.append(
                    tf.keras.callbacks.EarlyStopping(
                        monitor=early_stopping_monitor,
                        patience=int(early_stopping_patience),
                        min_delta=float(early_stopping_min_delta),
                        mode=early_stopping_mode,
                        restore_best_weights=bool(restore_best_weights),
                        verbose=1 if verbose else 0,
                    )
                )
        if source_callbacks:
            source_kwargs["callbacks"] = source_callbacks

        source_history = model.fit(
            X_source_for_fit,
            y_source,
            validation_data=source_validation_data,
            epochs=int(source_epochs),
            batch_size=int(source_batch_size),
            verbose=int(verbose),
            **source_kwargs,
        )

        source_epochs_ran = int(len(source_history.history.get("loss", [])))
        source_best_epoch: int | None = None
        source_best_monitored_value: float | None = None
        monitored_history = source_history.history.get(early_stopping_monitor)
        if monitored_history:
            monitored_values = np.asarray(monitored_history, dtype=np.float64)
            finite_indices = np.flatnonzero(np.isfinite(monitored_values))
            if len(finite_indices):
                finite_values = monitored_values[finite_indices]
                if early_stopping_mode == "max":
                    best_local_index = int(np.argmax(finite_values))
                elif early_stopping_mode == "min":
                    best_local_index = int(np.argmin(finite_values))
                else:
                    maximize_tokens = (
                        "acc",
                        "auc",
                        "f1",
                        "precision",
                        "recall",
                    )
                    maximize = any(
                        token in early_stopping_monitor.lower()
                        for token in maximize_tokens
                    )
                    best_local_index = int(
                        np.argmax(finite_values)
                        if maximize
                        else np.argmin(finite_values)
                    )
                best_history_index = int(finite_indices[best_local_index])
                source_best_epoch = best_history_index + 1
                source_best_monitored_value = float(
                    monitored_values[best_history_index]
                )
        if source_best_epoch is None and source_epochs_ran:
            source_best_epoch = source_epochs_ran

        # Persist the exact population/source model used for zero-shot LOSO
        # evaluation. This happens before calibration changes trainability or
        # compiles a subject-specific optimizer.
        prepare_zero_shot = getattr(
            model, "prepare_for_zero_shot_evaluation", None
        )
        if prepare_zero_shot is not None:
            prepare_zero_shot()
        zero_shot_model_artifact = (
            None
            if source_model_output_dir is None
            else _save_zero_shot_source_model(
                model,
                output_dir=source_model_output_dir,
                subject_number=subject_number,
                target_subject=target_subject,
            )
        )

        # get_weights() captures only model state, not optimizer state. That is
        # intentional: every calibration fold gets a newly compiled calibration
        # optimizer from prepare_for_subject_calibration().
        source_weights = [np.array(value, copy=True) for value in model.get_weights()]

        all_target_evaluation = _evaluate_classification_fold(
            model=model,
            X_test=X_target,
            y_test=y_target,
            subject_ids_test=target_subject_ids,
            trial_ids_test=target_trial_ids,
            fold_index=subject_number,
            metrics=metrics,
            evaluation_level=evaluation_level,
            batch_size=source_batch_size,
            n_prediction_latent_samples=n_prediction_latent_samples,
            latent_sampling_seed=latent_sampling_seed,
            log_predictions=log_predictions,
            log_variational_intervals=log_variational_intervals,
            n_uncertainty_samples=n_uncertainty_samples,
            ci_level=ci_level,
            decision_threshold=decision_threshold,
            ece_bins=ece_bins,
            print_results=False,
        )
        all_target_evaluation = _tag_subject_calibration_evaluation(
            all_target_evaluation,
            target_subject=target_subject,
            calibration_fold=None,
            stage="zero_shot_all_trials",
        )

        prepare_calibration = getattr(
            model,
            "prepare_for_subject_calibration",
            None,
        )
        if prepare_calibration is None:
            raise AttributeError(
                "The model must implement prepare_for_subject_calibration(" 
                "learning_rate=..., optimizer_name=..., weight_decay=...). "
                "That method should freeze the subject-independent/generative "
                "representation, leave only the intended classification head "
                "trainable, and compile a fresh calibration optimizer."
            )

        metric_names = ("loss", *metrics, "joint_loss", "decoder_r2")
        calibration_level_outputs: list[dict] = []
        evaluation_offset = (int(subject_number) - 1) * sum(
            level_folds for _, level_folds in calibration_levels
        )

        for calibration_shots, level_folds, calibration_splits in calibration_splits_by_level:
            calibration_rows: list[dict] = []
            fold_outputs: list[dict] = []

            for split in calibration_splits:
                calibration_fold = int(split["calibration_fold"])
                calibration_mask = np.isin(
                    target_trial_ids,
                    np.asarray(split["calibration_trial_ids"]),
                )
                evaluation_mask = np.isin(
                    target_trial_ids,
                    np.asarray(split["evaluation_trial_ids"]),
                )
                calibration_local_indices = np.flatnonzero(calibration_mask)
                evaluation_local_indices = np.flatnonzero(evaluation_mask)
                if not len(calibration_local_indices) or not len(evaluation_local_indices):
                    raise RuntimeError(
                        f"{calibration_shots}-shot calibration fold "
                        f"{calibration_fold} produced an empty calibration or "
                        "evaluation partition."
                    )

                X_calibration = X_target[calibration_local_indices]
                y_calibration = y_target[calibration_local_indices]
                X_evaluation = X_target[evaluation_local_indices]
                y_evaluation = y_target[evaluation_local_indices]
                evaluation_subject_ids = target_subject_ids[evaluation_local_indices]
                evaluation_trial_ids = target_trial_ids[evaluation_local_indices]

                # Every shot/fold continuation starts from the identical strict
                # LOSO source checkpoint and receives a fresh optimizer.
                model.set_weights(source_weights)
                prepare_zero_shot = getattr(
                    model, "prepare_for_zero_shot_evaluation", None
                )
                if prepare_zero_shot is not None:
                    prepare_zero_shot()
                evaluation_index = evaluation_offset + calibration_fold
                zero_shot = _evaluate_classification_fold(
                    model=model,
                    X_test=X_evaluation,
                    y_test=y_evaluation,
                    subject_ids_test=evaluation_subject_ids,
                    trial_ids_test=evaluation_trial_ids,
                    fold_index=evaluation_index,
                    metrics=metrics,
                    evaluation_level=evaluation_level,
                    batch_size=source_batch_size,
                    n_prediction_latent_samples=n_prediction_latent_samples,
                    latent_sampling_seed=latent_sampling_seed,
                    log_predictions=log_predictions,
                    log_variational_intervals=log_variational_intervals,
                    n_uncertainty_samples=n_uncertainty_samples,
                    ci_level=ci_level,
                    decision_threshold=decision_threshold,
                    ece_bins=ece_bins,
                    print_results=False,
                )
                zero_shot = _tag_subject_calibration_evaluation(
                    zero_shot,
                    target_subject=target_subject,
                    calibration_fold=calibration_fold,
                    calibration_shots=calibration_shots,
                    stage="zero_shot_paired",
                )

                model.set_weights(source_weights)
                prepare_calibration(
                    learning_rate=float(calibration_learning_rate),
                    optimizer_name=str(calibration_optimizer),
                    weight_decay=float(calibration_weight_decay),
                )
                trainable_names = [
                    variable.name for variable in model.trainable_variables
                ]
                if not trainable_names:
                    raise RuntimeError(
                        "prepare_for_subject_calibration() left no trainable variables."
                    )

                prepare_inputs = getattr(model, "prepare_calibration_inputs", None)
                X_calibration_for_fit = (
                    prepare_inputs(X_calibration)
                    if prepare_inputs is not None
                    else X_calibration
                )
                calibration_kwargs = dict(calibration_fit_kwargs)
                calibration_class_weight = (
                    _class_weight_from_labels(y_calibration)
                    if calibration_use_class_weight
                    else None
                )
                if calibration_class_weight is not None:
                    calibration_kwargs["class_weight"] = calibration_class_weight

                calibration_history = model.fit(
                    X_calibration_for_fit,
                    y_calibration,
                    epochs=int(calibration_epochs),
                    batch_size=int(calibration_batch_size),
                    # Calibration continuations are intentionally quiet; the
                    # terminal reports only final mean accuracy per shot level.
                    verbose=0,
                    **calibration_kwargs,
                )

                calibrated = _evaluate_classification_fold(
                    model=model,
                    X_test=X_evaluation,
                    y_test=y_evaluation,
                    subject_ids_test=evaluation_subject_ids,
                    trial_ids_test=evaluation_trial_ids,
                    fold_index=evaluation_index,
                    metrics=metrics,
                    evaluation_level=evaluation_level,
                    batch_size=calibration_batch_size,
                    n_prediction_latent_samples=n_prediction_latent_samples,
                    latent_sampling_seed=latent_sampling_seed,
                    log_predictions=log_predictions,
                    log_variational_intervals=log_variational_intervals,
                    n_uncertainty_samples=n_uncertainty_samples,
                    ci_level=ci_level,
                    decision_threshold=decision_threshold,
                    ece_bins=ece_bins,
                    print_results=False,
                )
                calibrated = _tag_subject_calibration_evaluation(
                    calibrated,
                    target_subject=target_subject,
                    calibration_fold=calibration_fold,
                    calibration_shots=calibration_shots,
                    stage="post_calibration",
                )

                zero_scores = _calibration_score_dict(zero_shot, metric_names)
                calibrated_scores = _calibration_score_dict(
                    calibrated, metric_names
                )
                delta_scores = {
                    metric_name: float(
                        calibrated_scores[metric_name] - zero_scores[metric_name]
                    )
                    for metric_name in zero_scores.keys() & calibrated_scores.keys()
                }
                calibration_row = {
                    "target_subject": _python_scalar(target_subject),
                    "calibration_shots": int(calibration_shots),
                    "calibration_fold": calibration_fold,
                    "partition_mode": split["partition_mode"],
                    "calibration_trial_ids": list(split["calibration_trial_ids"]),
                    "evaluation_trial_ids": list(split["evaluation_trial_ids"]),
                    "calibration_class_counts": dict(split["calibration_class_counts"]),
                    "evaluation_class_counts": dict(split["evaluation_class_counts"]),
                    "n_calibration_samples": int(len(calibration_local_indices)),
                    "n_evaluation_samples": int(len(evaluation_local_indices)),
                    "calibration_epochs_ran": int(
                        len(calibration_history.history.get("loss", []))
                    ),
                    "calibration_final_history": _final_history_values(
                        calibration_history
                    ),
                    "calibration_trainable_variables": trainable_names,
                    "zero_shot_scores": zero_scores,
                    "calibrated_scores": calibrated_scores,
                    "delta_scores": delta_scores,
                }
                calibration_rows.append(calibration_row)
                fold_outputs.append(
                    {
                        "calibration_shots": int(calibration_shots),
                        "split": dict(split),
                        "zero_shot": zero_shot,
                        "calibrated": calibrated,
                    }
                )

            zero_rows = [row["zero_shot_scores"] for row in calibration_rows]
            calibrated_rows = [
                row["calibrated_scores"] for row in calibration_rows
            ]
            delta_rows = [row["delta_scores"] for row in calibration_rows]
            paired_zero_mean, paired_zero_std = _mean_std_rows(
                zero_rows, list(metric_names)
            )
            calibrated_mean, calibrated_std = _mean_std_rows(
                calibrated_rows, list(metric_names)
            )
            delta_mean, delta_std = _mean_std_rows(
                delta_rows, list(metric_names)
            )
            calibration_level_outputs.append(
                {
                    "calibration_shots": int(calibration_shots),
                    "calibration_folds": int(level_folds),
                    "calibration_runs": calibration_rows,
                    "fold_outputs": fold_outputs,
                    "summary": {
                        "paired_zero_shot_mean_scores": paired_zero_mean,
                        "paired_zero_shot_std_scores": paired_zero_std,
                        "calibrated_mean_scores": calibrated_mean,
                        "calibrated_std_scores": calibrated_std,
                        "delta_mean_scores": delta_mean,
                        "delta_std_scores": delta_std,
                    },
                }
            )
            evaluation_offset += int(level_folds)

        zero_all_scores = _calibration_score_dict(
            all_target_evaluation,
            metric_names,
        )
        calibration_summaries = {
            str(level["calibration_shots"]): dict(level["summary"])
            for level in calibration_level_outputs
        }
        subject_summary = {
            "target_subject": _python_scalar(target_subject),
            "zero_shot_all_trials_scores": zero_all_scores,
            "calibration_levels": calibration_summaries,
        }
        return {
            "subject_number": int(subject_number),
            "target_subject": _python_scalar(target_subject),
            "source_subjects": [
                _python_scalar(value)
                for value in np.sort(np.unique(source_subject_ids)).tolist()
            ],
            "source_validation_subjects": [
                _python_scalar(value) for value in validation_subjects.tolist()
            ],
            "n_source_samples": int(len(source_indices)),
            "n_source_validation_samples": int(len(source_validation_indices)),
            "n_outer_source_samples": int(len(outer_source_indices)),
            "n_target_samples": int(len(target_indices)),
            "n_target_trials": int(len(np.unique(target_trial_ids))),
            "source_training": {
                "epochs_ran": source_epochs_ran,
                "best_epoch": source_best_epoch,
                "best_monitored_value": source_best_monitored_value,
                "early_stopping_monitor": early_stopping_monitor,
                "stopped_early": bool(source_epochs_ran < int(source_epochs)),
                "restore_best_weights": bool(restore_best_weights),
                "final_history": _final_history_values(source_history),
                "class_weight": source_class_weight,
                "prediction_diagnostics_metric": (
                    str(prediction_diagnostics_metric)
                    if prediction_diagnostics
                    else None
                ),
            },
            "prediction_diagnostics_log": (
                []
                if prediction_diagnostics_callback is None
                else list(prediction_diagnostics_callback.history)
            ),
            "oracle_epoch_log": list(oracle_metrics_callback.history),
            "zero_shot_model": zero_shot_model_artifact,
            "zero_shot_all_trials": all_target_evaluation,
            "calibration_levels": calibration_level_outputs,
            "subject_summary": subject_summary,
        }
    finally:
        if "model" in locals():
            del model
        gc.collect()
        tf.keras.backend.clear_session()


def _subject_calibration_process_main(
    worker_state_payload: bytes,
    task_queue,
    result_queue,
    gpu_id: int | None,
    cpus_per_worker: int | None,
    assigned_device_label: str | None,
) -> None:
    """Run target-subject calibration evaluations in a persistent worker."""
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
            subject_number, target_subject = task
            try:
                output = _run_subject_calibration_subject(
                    subject_number=subject_number,
                    target_subject=target_subject,
                    **worker_state,
                )
                result_queue.put(("ok", int(subject_number), output))
            except BaseException:
                result_queue.put(
                    ("error", int(subject_number), traceback.format_exc())
                )
                return
    except BaseException:
        result_queue.put(("error", -1, traceback.format_exc()))
    finally:
        tf.keras.backend.clear_session()
        gc.collect()


def subject_calibration_cv(
    model_builder_function: Callable[..., tf.keras.Model],
    feature_array: np.ndarray,
    label_array: np.ndarray,
    subject_id_array: np.ndarray,
    trial_id_array: np.ndarray,
    fixed_config: dict | None = None,
    source_epochs: int = 100,
    source_batch_size: int = 64,
    calibration_epochs: int = 20,
    calibration_batch_size: int = 6,
    *,
    calibration_trials: int = 6,
    calibration_folds: int = 3,
    calibration_levels: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None = None,
    calibration_selection_shots: int | None = None,
    calibration_learning_rate: float = 1e-4,
    calibration_optimizer: str = "adamw",
    calibration_weight_decay: float = 0.0,
    calibration_seed: int | None = 42,
    stratify_calibration: bool = True,
    allow_overlapping_calibration_folds: bool = False,
    validation_subjects_per_fold: int = 0,
    validation_seed: int | None = 42,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    early_stopping_monitor: str = "val_loss",
    early_stopping_mode: Literal["auto", "min", "max"] = "auto",
    restore_best_weights: bool = True,
    evaluation_level: Literal["window", "trial"] = "trial",
    metrics: list[str] | tuple[str, ...] = (
        "accuracy",
        "f1",
        "precision",
        "recall",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "balanced_accuracy",
        "roc_auc",
        "brier_score",
        "ece",
    ),
    ece_bins: int = _DEFAULT_ECE_BINS,
    decision_threshold: float = 0.5,
    prediction_diagnostics_thresholds: (
        list[float] | tuple[float, ...] | None
    ) = None,
    prediction_diagnostics: bool = False,
    prediction_diagnostics_metric: str = "accuracy",
    prediction_diagnostics_every_n_epochs: int = 1,
    prediction_diagnostics_max_samples: int = 256,
    prediction_diagnostics_threshold_tolerance: float = 0.01,
    prediction_diagnostics_seed: int | None = 42,
    n_prediction_latent_samples: int = 0,
    latent_sampling_seed: int | None = None,
    log_predictions: bool = True,
    log_variational_intervals: bool = False,
    n_uncertainty_samples: int = 30,
    ci_level: float = 0.95,
    source_use_class_weight: bool = False,
    calibration_use_class_weight: bool = False,
    source_fit_kwargs: dict | None = None,
    calibration_fit_kwargs: dict | None = None,
    verbose: int = 0,
    n_jobs: int = 1,
    gpu_ids: list[int] | tuple[int, ...] | None = None,
    cpus_per_worker: int | None = None,
    max_subjects: int | None = None,
    target_subjects: list[int] | tuple[int, ...] | None = None,
    source_model_output_dir: str | os.PathLike[str] | None = None,
) -> dict:
    """Strict LOSO pretraining followed by one or more calibration levels.

    For each target subject, a fresh subject-independent source model is first
    trained using the other subjects. When ``validation_subjects_per_fold`` is
    positive, a seeded subset of those source subjects is excluded from gradient
    updates and used only for source checkpoint selection. The target subject is
    never used to construct, optimize, or select that source model. This is
    especially important for v6's MTLFuseNet-style GCN: the four ``training_*``
    arrays contain only gradient-training source subjects, so its fixed
    mutual-information adjacency remains training-only.

    The source model is evaluated zero-shot on all target trials once. Pass
    ``calibration_levels=((3, 6), (6, 3), ...)`` to request ``(shots, folds)``
    pairs. If omitted, the legacy ``calibration_trials``/``calibration_folds``
    pair is used. The same source checkpoint is restored for every level and
    fold; it is never retrained between calibration levels. Exact partitions
    are used when ``shots * folds`` equals the retained trial count, otherwise
    seeded repeated holdouts evaluate on every non-calibration trial. For each
    fold:

      1. restore the exact source-pretrained weights;
      2. evaluate zero-shot on the trials not used for calibration;
      3. call ``model.prepare_for_subject_calibration(...)``;
      4. fine-tune only the model-defined calibration parameters;
      5. evaluate the calibrated model on the same held-out target trials.

    When ``source_model_output_dir`` is provided, the exact source-trained
    0-shot model for every LOSO target is saved there in native ``.keras``
    format before calibration begins. Calibrated fold continuations are not
    saved. Load these source models with ``compile=False`` before calling the
    model's calibration preparation hook, which installs a fresh optimizer.

    Thus the default DREAMER protocol remains 6 calibration trials + 12
    evaluation trials repeated three times, while arbitrary valid shot/fold
    pairs can be evaluated in the same LOSO run.

    ``model_builder_function`` contract
    ------------------------------------
    In addition to ``fixed_config`` model kwargs, the builder must accept:
    ``training_features``, ``training_labels``, ``training_subject_ids``, and
    ``training_trial_ids``. The first of these is where v6 should estimate its
    training-only MI graph.

    ``prepare_for_subject_calibration`` contract
    --------------------------------------------
    The returned model must implement::

        model.prepare_for_subject_calibration(
            learning_rate=...,
            optimizer_name=...,
            weight_decay=...,
        )

    The method owns the architecture-specific freezing policy and must compile
    a *fresh optimizer*. For deterministic SIC it should freeze the MTL GCN,
    spectral GRU, temporal encoder, and subject-invariance machinery, leaving
    only the intended emotion-classification head suffix trainable.

    Aggregation
    -----------
    Metrics are first averaged across folds within each subject and shot level,
    then across subjects. Calibration folds are never treated as independent
    subjects. JSON output retains every requested metric at every shot level;
    terminal output reports final mean accuracy, balanced accuracy, and Brier
    score for 0-shot and each calibrated level.
    """
    feature_array = np.asarray(feature_array)
    label_array = np.asarray(label_array)
    subject_id_array = np.asarray(subject_id_array).reshape(-1)
    trial_id_array = np.asarray(trial_id_array).reshape(-1)
    fixed_config = dict(fixed_config or {})
    source_fit_kwargs = dict(source_fit_kwargs or {})
    calibration_fit_kwargs = dict(calibration_fit_kwargs or {})
    calibration_levels = _normalize_calibration_levels(
        calibration_levels,
        calibration_trials=calibration_trials,
        calibration_folds=calibration_folds,
    )
    calibration_level_shots = tuple(shots for shots, _ in calibration_levels)
    if calibration_selection_shots is None:
        calibration_selection_shots = max(calibration_level_shots)
    calibration_selection_shots = int(calibration_selection_shots)
    if calibration_selection_shots not in calibration_level_shots:
        raise ValueError(
            "calibration_selection_shots must match one configured shot level; "
            f"got {calibration_selection_shots}, available={calibration_level_shots}."
        )

    if feature_array.ndim not in {3, 4}:
        raise ValueError(
            "feature_array must be rank 3 (window samples) or rank 4 "
            f"(grouped trial samples); got {feature_array.shape}."
        )
    lengths = (
        len(feature_array),
        len(label_array),
        len(subject_id_array),
        len(trial_id_array),
    )
    if len(set(lengths)) != 1:
        raise ValueError(
            "feature_array, label_array, subject_id_array, and trial_id_array "
            f"must align; got lengths {lengths}."
        )
    if source_epochs < 1 or calibration_epochs < 1:
        raise ValueError("source_epochs and calibration_epochs must be >= 1.")
    if source_batch_size < 1 or calibration_batch_size < 1:
        raise ValueError(
            "source_batch_size and calibration_batch_size must be >= 1."
        )
    if validation_subjects_per_fold < 0:
        raise ValueError("validation_subjects_per_fold must be >= 0.")
    if early_stopping_patience is not None and early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be >= 1 when provided.")
    if early_stopping_min_delta < 0.0:
        raise ValueError("early_stopping_min_delta must be non-negative.")
    if early_stopping_mode not in {"auto", "min", "max"}:
        raise ValueError("early_stopping_mode must be 'auto', 'min', or 'max'.")
    if not isinstance(early_stopping_monitor, str) or not early_stopping_monitor:
        raise ValueError("early_stopping_monitor must be a non-empty string.")
    if calibration_learning_rate <= 0.0:
        raise ValueError("calibration_learning_rate must be positive.")
    if calibration_weight_decay < 0.0:
        raise ValueError("calibration_weight_decay must be non-negative.")
    if calibration_optimizer not in {"adam", "adamw"}:
        raise ValueError("calibration_optimizer must be 'adam' or 'adamw'.")
    decision_threshold = float(decision_threshold)
    if not 0.0 < decision_threshold < 1.0:
        raise ValueError("decision_threshold must lie strictly between 0 and 1.")
    raw_diagnostic_thresholds = (
        (decision_threshold,)
        if prediction_diagnostics_thresholds is None
        else (*prediction_diagnostics_thresholds, decision_threshold)
    )
    normalized_diagnostic_thresholds = _normalize_decision_thresholds(
        raw_diagnostic_thresholds
    )
    prediction_diagnostics_thresholds = (
        decision_threshold,
        *(
            threshold
            for threshold in normalized_diagnostic_thresholds
            if threshold != decision_threshold
        ),
    )
    prediction_diagnostics_metric = str(
        prediction_diagnostics_metric
    ).strip().lower()
    if prediction_diagnostics_metric not in _CLASSIFICATION_METRICS:
        raise ValueError(
            "prediction_diagnostics_metric must be one of "
            f"{sorted(_CLASSIFICATION_METRICS)}; got "
            f"{prediction_diagnostics_metric!r}."
        )
    if int(prediction_diagnostics_every_n_epochs) < 1:
        raise ValueError("prediction_diagnostics_every_n_epochs must be >= 1.")
    if int(prediction_diagnostics_max_samples) < 1:
        raise ValueError("prediction_diagnostics_max_samples must be >= 1.")
    if float(prediction_diagnostics_threshold_tolerance) < 0.0:
        raise ValueError(
            "prediction_diagnostics_threshold_tolerance must be non-negative."
        )
    if n_prediction_latent_samples < 0:
        raise ValueError("n_prediction_latent_samples must be >= 0.")
    if int(ece_bins) < 2:
        raise ValueError("ece_bins must be >= 2.")
    if not 0.0 < float(ci_level) < 1.0:
        raise ValueError("ci_level must lie between 0 and 1.")
    if n_jobs < 1:
        raise ValueError("n_jobs must be >= 1.")
    if cpus_per_worker is not None and cpus_per_worker < 1:
        raise ValueError("cpus_per_worker must be >= 1 when provided.")
    if max_subjects is not None and int(max_subjects) < 1:
        raise ValueError("max_subjects must be >= 1 when provided.")
    if source_model_output_dir is not None:
        source_model_output_dir = os.path.abspath(
            os.fspath(source_model_output_dir)
        )
        os.makedirs(source_model_output_dir, exist_ok=True)
    if feature_array.ndim == 4 and evaluation_level != "trial":
        raise ValueError(
            "Rank-4 grouped-trial inputs require evaluation_level='trial'."
        )
    _validate_evaluation_level(evaluation_level, "evaluation_level")

    for forbidden_key in ("epochs", "batch_size"):
        if forbidden_key in fixed_config:
            raise ValueError(
                f"Remove {forbidden_key!r} from fixed_config; "
                "subject_calibration_cv has separate source/calibration values."
            )
    for fit_name, fit_kwargs in (
        ("source_fit_kwargs", source_fit_kwargs),
        ("calibration_fit_kwargs", calibration_fit_kwargs),
    ):
        duplicates = {"epochs", "batch_size", "verbose", "class_weight"}.intersection(
            fit_kwargs
        )
        if duplicates:
            raise ValueError(
                f"{fit_name} must not override managed fit arguments: "
                f"{sorted(duplicates)}."
            )
        if "validation_data" in fit_kwargs:
            raise ValueError(
                f"{fit_name} must not provide validation_data. Source-validation "
                "subjects are managed by subject_calibration_cv, the target "
                "subject cannot select the source model, and the six calibration "
                "trials are intentionally used only for fitting."
            )

    metrics = tuple(metrics)
    for metric in metrics:
        if metric not in _CLASSIFICATION_METRICS:
            raise ValueError(
                f"Unsupported metric {metric!r}. Supported metrics: "
                f"{sorted(_CLASSIFICATION_METRICS)}"
            )

    unique_subjects = np.sort(np.unique(subject_id_array))
    if len(unique_subjects) < 2:
        raise ValueError("subject_calibration_cv requires at least two subjects.")
    if validation_subjects_per_fold >= len(unique_subjects) - 1:
        raise ValueError(
            "validation_subjects_per_fold must leave at least one non-target "
            "subject for source fitting. Got "
            f"{validation_subjects_per_fold} validation subjects with "
            f"{len(unique_subjects)} total subjects."
        )
    if target_subjects is None:
        target_subjects = unique_subjects
        if max_subjects is not None:
            target_subjects = target_subjects[: int(max_subjects)]
    else:
        target_subjects = np.asarray(target_subjects)
        missing_subjects = target_subjects[
            ~np.isin(target_subjects, unique_subjects)
        ]
        if len(missing_subjects):
            raise ValueError(
                f"Unknown target subject IDs: {missing_subjects.tolist()}. "
                f"Available subjects: {unique_subjects.tolist()}."
            )
    total_subjects = int(len(target_subjects))

    # Fail early if a usable calibration/evaluation split cannot be formed for
    # any selected subject, rather than discovering it after pretraining.
    for target_subject in target_subjects:
        target_trial_count = len(
            np.unique(trial_id_array[subject_id_array == target_subject])
        )
        invalid_shots = [
            shots for shots, _ in calibration_levels if shots >= target_trial_count
        ]
        if invalid_shots:
            raise ValueError(
                f"Target subject {_python_scalar(target_subject)!r} has "
                f"{target_trial_count} retained trials. Every calibration level "
                "must leave at least one evaluation trial; invalid shot levels="
                f"{invalid_shots}."
            )

    effective_n_jobs = min(int(n_jobs), total_subjects)
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
                f"n_jobs={effective_n_jobs} requires at least that many GPU IDs; "
                f"got gpu_ids={normalized_gpu_ids}."
            )
        normalized_gpu_ids = normalized_gpu_ids[:effective_n_jobs]

    tasks = [
        (subject_number, target_subject)
        for subject_number, target_subject in enumerate(target_subjects, start=1)
    ]
    worker_state = {
        "total_subjects": total_subjects,
        "model_builder_function": model_builder_function,
        "feature_array": feature_array,
        "label_array": label_array,
        "subject_id_array": subject_id_array,
        "trial_id_array": trial_id_array,
        "fixed_config": fixed_config,
        "source_epochs": int(source_epochs),
        "source_batch_size": int(source_batch_size),
        "validation_subjects_per_fold": int(validation_subjects_per_fold),
        "validation_seed": validation_seed,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": float(early_stopping_min_delta),
        "early_stopping_monitor": str(early_stopping_monitor),
        "early_stopping_mode": str(early_stopping_mode),
        "restore_best_weights": bool(restore_best_weights),
        "calibration_epochs": int(calibration_epochs),
        "calibration_batch_size": int(calibration_batch_size),
        "calibration_trials": int(calibration_trials),
        "calibration_folds": int(calibration_folds),
        "calibration_levels": calibration_levels,
        "calibration_learning_rate": float(calibration_learning_rate),
        "calibration_optimizer": str(calibration_optimizer),
        "calibration_weight_decay": float(calibration_weight_decay),
        "calibration_seed": calibration_seed,
        "stratify_calibration": bool(stratify_calibration),
        "allow_overlapping_calibration_folds": bool(
            allow_overlapping_calibration_folds
        ),
        "evaluation_level": evaluation_level,
        "metrics": metrics,
        "ece_bins": int(ece_bins),
        "decision_threshold": decision_threshold,
        "prediction_diagnostics_thresholds": (
            prediction_diagnostics_thresholds
        ),
        "prediction_diagnostics": bool(prediction_diagnostics),
        "prediction_diagnostics_metric": prediction_diagnostics_metric,
        "prediction_diagnostics_every_n_epochs": int(
            prediction_diagnostics_every_n_epochs
        ),
        "prediction_diagnostics_max_samples": int(
            prediction_diagnostics_max_samples
        ),
        "prediction_diagnostics_threshold_tolerance": float(
            prediction_diagnostics_threshold_tolerance
        ),
        "prediction_diagnostics_seed": prediction_diagnostics_seed,
        "n_prediction_latent_samples": int(n_prediction_latent_samples),
        "latent_sampling_seed": latent_sampling_seed,
        "log_predictions": bool(log_predictions),
        "log_variational_intervals": bool(log_variational_intervals),
        "n_uncertainty_samples": int(n_uncertainty_samples),
        "ci_level": float(ci_level),
        "source_use_class_weight": bool(source_use_class_weight),
        "calibration_use_class_weight": bool(calibration_use_class_weight),
        "source_fit_kwargs": source_fit_kwargs,
        "calibration_fit_kwargs": calibration_fit_kwargs,
        "source_model_output_dir": source_model_output_dir,
        "verbose": int(verbose),
    }

    if effective_n_jobs == 1 and normalized_gpu_ids is None:
        subject_outputs = [
            _run_subject_calibration_subject(
                subject_number=subject_number,
                target_subject=target_subject,
                **worker_state,
            )
            for subject_number, target_subject in tasks
        ]
    else:
        subject_outputs = _run_spawned_fold_pool(
            worker_target=_subject_calibration_process_main,
            worker_state=worker_state,
            tasks=tasks,
            n_workers=effective_n_jobs,
            gpu_ids=normalized_gpu_ids,
            cpus_per_worker=cpus_per_worker,
            worker_name_prefix="SubjectCalibrationWorker",
            worker_description="target-subject calibration",
        )
    subject_outputs.sort(key=lambda row: int(row["subject_number"]))

    metric_names = ("loss", *metrics, "joint_loss", "decoder_r2")
    subject_summary_rows: list[dict] = []
    zero_all_subject_rows: list[dict] = []
    level_subject_rows = {
        str(shots): {
            "paired_zero": [],
            "calibrated": [],
            "delta": [],
        }
        for shots, _ in calibration_levels
    }
    total_calibration_folds = sum(folds for _, folds in calibration_levels)

    results = {
        "cv_strategy": "subject_independent_pretraining_with_subject_calibration",
        "protocol_name": "strict LOSO with multi-level calibration continuation",
        "n_subjects": total_subjects,
        "n_source_model_fits": total_subjects,
        "n_calibration_fits": total_subjects * int(total_calibration_folds),
        "calibration_plan": [
            {"shots": int(shots), "folds": int(folds)}
            for shots, folds in calibration_levels
        ],
        "calibration_selection_shots": int(calibration_selection_shots),
        "calibration_seed": calibration_seed,
        "stratify_calibration": bool(stratify_calibration),
        "allow_overlapping_calibration_folds": bool(
            allow_overlapping_calibration_folds
        ),
        "source_epochs": int(source_epochs),
        "source_batch_size": int(source_batch_size),
        "validation_subjects_per_fold": int(validation_subjects_per_fold),
        "validation_seed": validation_seed,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": float(early_stopping_min_delta),
        "early_stopping_monitor": str(early_stopping_monitor),
        "early_stopping_mode": str(early_stopping_mode),
        "restore_best_weights": bool(restore_best_weights),
        "calibration_epochs": int(calibration_epochs),
        "calibration_batch_size": int(calibration_batch_size),
        "calibration_learning_rate": float(calibration_learning_rate),
        "calibration_optimizer": str(calibration_optimizer),
        "calibration_weight_decay": float(calibration_weight_decay),
        "decision_threshold": decision_threshold,
        "prediction_diagnostics": {
            "enabled": bool(prediction_diagnostics),
            "official_decision_threshold": decision_threshold,
            "decision_thresholds": list(prediction_diagnostics_thresholds),
            "reported_metric": prediction_diagnostics_metric,
            "every_n_epochs": int(prediction_diagnostics_every_n_epochs),
            "max_samples": int(prediction_diagnostics_max_samples),
            "threshold_tolerance": float(
                prediction_diagnostics_threshold_tolerance
            ),
            "seed": prediction_diagnostics_seed,
        },
        "evaluation_level": evaluation_level,
        "metrics": list(metrics),
        "ece_bins": int(ece_bins),
        "fixed_config": fixed_config,
        "zero_shot_source_models": {
            "saved": source_model_output_dir is not None,
            "directory": source_model_output_dir,
            "format": "keras",
            "load_with_compile": False,
            "models": [
                subject_output["zero_shot_model"]
                for subject_output in subject_outputs
                if subject_output.get("zero_shot_model") is not None
            ],
        },
        "subject_results": subject_outputs,
        "subject_summary_rows": subject_summary_rows,
        "oracle_epoch_log": [],
    }
    if prediction_diagnostics:
        results["prediction_diagnostics_log"] = []
    if log_predictions:
        if feature_array.ndim == 3:
            results["window_prediction_log"] = []
        results["trial_prediction_log"] = []
    if log_variational_intervals:
        if feature_array.ndim == 3:
            results["window_variational_interval_log"] = []
        results["trial_variational_interval_log"] = []

    for subject_output in subject_outputs:
        results["oracle_epoch_log"].extend(
            subject_output.get("oracle_epoch_log", [])
        )
        if prediction_diagnostics:
            results["prediction_diagnostics_log"].extend(
                subject_output.get("prediction_diagnostics_log", [])
            )
        summary = subject_output["subject_summary"]
        zero_all = dict(summary["zero_shot_all_trials_scores"])
        zero_all_subject_rows.append(zero_all)

        flat_summary = {"target_subject": subject_output["target_subject"]}
        zero_shot_model = subject_output.get("zero_shot_model")
        if zero_shot_model is not None:
            flat_summary["zero_shot_model_path"] = zero_shot_model["path"]
            flat_summary["zero_shot_model_filename"] = zero_shot_model[
                "filename"
            ]
        flat_summary.update({f"zero_shot_all_{k}": v for k, v in zero_all.items()})
        for level in subject_output["calibration_levels"]:
            shots_key = str(level["calibration_shots"])
            level_summary = dict(level["summary"])
            paired_zero = dict(level_summary["paired_zero_shot_mean_scores"])
            calibrated = dict(level_summary["calibrated_mean_scores"])
            delta = dict(level_summary["delta_mean_scores"])
            level_subject_rows[shots_key]["paired_zero"].append(paired_zero)
            level_subject_rows[shots_key]["calibrated"].append(calibrated)
            level_subject_rows[shots_key]["delta"].append(delta)
            flat_summary.update(
                {f"{shots_key}_shot_paired_zero_{k}": v for k, v in paired_zero.items()}
            )
            flat_summary.update(
                {f"{shots_key}_shot_calibrated_{k}": v for k, v in calibrated.items()}
            )
            flat_summary.update(
                {f"{shots_key}_shot_delta_{k}": v for k, v in delta.items()}
            )
        subject_summary_rows.append(flat_summary)

        if log_predictions:
            evaluations = [subject_output["zero_shot_all_trials"]]
            for level in subject_output["calibration_levels"]:
                for fold_output in level["fold_outputs"]:
                    evaluations.extend(
                        [fold_output["zero_shot"], fold_output["calibrated"]]
                    )
            for evaluation in evaluations:
                if feature_array.ndim == 3:
                    results["window_prediction_log"].extend(
                        evaluation.get("window_prediction_log", [])
                    )
                results["trial_prediction_log"].extend(
                    evaluation.get("trial_prediction_log", [])
                )

        if log_variational_intervals:
            evaluations = [subject_output["zero_shot_all_trials"]]
            for level in subject_output["calibration_levels"]:
                for fold_output in level["fold_outputs"]:
                    evaluations.extend(
                        [fold_output["zero_shot"], fold_output["calibrated"]]
                    )
            for evaluation in evaluations:
                if feature_array.ndim == 3:
                    results["window_variational_interval_log"].extend(
                        evaluation.get("window_variational_interval_log", [])
                    )
                results["trial_variational_interval_log"].extend(
                    evaluation.get("trial_variational_interval_log", [])
                )

    zero_all_mean, zero_all_std = _mean_std_rows(
        zero_all_subject_rows, list(metric_names)
    )
    overall_calibration_levels: dict[str, dict] = {}
    for shots, folds in calibration_levels:
        shots_key = str(shots)
        rows = level_subject_rows[shots_key]
        paired_zero_mean, paired_zero_std = _mean_std_rows(
            rows["paired_zero"], list(metric_names)
        )
        calibrated_mean, calibrated_std = _mean_std_rows(
            rows["calibrated"], list(metric_names)
        )
        delta_mean, delta_std = _mean_std_rows(
            rows["delta"], list(metric_names)
        )
        overall_calibration_levels[shots_key] = {
            "calibration_shots": int(shots),
            "calibration_folds": int(folds),
            "paired_zero_shot_mean_scores": paired_zero_mean,
            "paired_zero_shot_std_scores": paired_zero_std,
            "calibrated_mean_scores": calibrated_mean,
            "calibrated_std_scores": calibrated_std,
            "delta_mean_scores": delta_mean,
            "delta_std_scores": delta_std,
            "delta_definition": "post_calibration_minus_paired_zero_shot",
        }
    selected_level = overall_calibration_levels[str(calibration_selection_shots)]
    results["overall"] = {
        "aggregation_unit": "subject",
        "n_subjects": total_subjects,
        "zero_shot_all_trials_mean_scores": zero_all_mean,
        "zero_shot_all_trials_std_scores": zero_all_std,
        "calibration_selection_shots": int(calibration_selection_shots),
        "calibration_levels": overall_calibration_levels,
        # Compatibility aliases point to the explicitly selected shot level.
        "paired_zero_shot_mean_scores": selected_level["paired_zero_shot_mean_scores"],
        "paired_zero_shot_std_scores": selected_level["paired_zero_shot_std_scores"],
        "calibrated_mean_scores": selected_level["calibrated_mean_scores"],
        "calibrated_std_scores": selected_level["calibrated_std_scores"],
        "delta_mean_scores": selected_level["delta_mean_scores"],
        "delta_std_scores": selected_level["delta_std_scores"],
        "delta_definition": "post_calibration_minus_paired_zero_shot",
    }

    final_metric_names = ("accuracy", "balanced_accuracy", "brier_score")

    def _format_final_means(mean_scores: dict) -> str:
        parts = []
        for metric_name in final_metric_names:
            value = mean_scores.get(metric_name)
            formatted_value = "N/A" if value is None else f"{float(value):.6f}"
            parts.append(f"{metric_name}={formatted_value}")
        return ", ".join(parts)

    print("\nFinal mean metrics by calibration level")
    print("=" * 80)
    print(f"0-shot: {_format_final_means(zero_all_mean)}")
    for shots, _ in calibration_levels:
        calibrated_means = overall_calibration_levels[str(shots)][
            "calibrated_mean_scores"
        ]
        print(f"{shots}-shot: {_format_final_means(calibrated_means)}")
    return results

def fixed_loso_cv(
    model_builder_function: Callable[..., tf.keras.Model],
    feature_array: np.ndarray,
    label_array: np.ndarray,
    subject_id_array: np.ndarray,
    trial_id_array: np.ndarray,
    fixed_config: dict,
    n_epochs: int,
    batch_size: int,
    *,
    preprocessing_strategy: Callable | None = None,
    evaluation_level: Literal["window", "trial"] = "trial",
    selection_metric: str = "balanced_accuracy",
    selection_level: Literal["window", "trial"] = "trial",
    maximize_metric: bool | None = None,
    metrics: list[str] | tuple[str, ...] = (
        "accuracy",
        "f1",
        "precision",
        "recall",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "balanced_accuracy",
    ),
    log_predictions: bool = True,
    log_variational_intervals: bool = False,
    n_prediction_latent_samples: int = 0,
    latent_sampling_seed: int | None = None,
    n_uncertainty_samples: int = 30,
    ci_level: float = 0.95,
    decision_threshold: float = 0.5,
    prediction_diagnostics: bool = False,
    prediction_diagnostics_every_n_epochs: int = 1,
    prediction_diagnostics_max_samples: int = 256,
    prediction_diagnostics_threshold_tolerance: float = 0.01,
    prediction_diagnostics_seed: int | None = 42,
    verbose: int = 0,
    extra_fit_kwargs: dict | None = None,
    n_jobs: int = 1,
    gpu_ids: list[int] | tuple[int, ...] | None = None,
    cpus_per_worker: int | None = None,
    max_folds: int | None = None,
    alternate_subject_sets: bool = False,
    alternating_subject_seed: int | None = 42,
    use_mldg: bool = False,
    mldg_meta_train_subjects: int = 6,
    mldg_meta_test_subjects: int = 2,
    mldg_samples_per_subject: int = 4,
    mldg_seed: int | None = 42,
) -> dict:
    """Evaluate one fixed configuration with strict LOSOCV and no validation.

    Every fold trains for exactly ``n_epochs`` on all non-test subjects. No
    validation subjects are removed, no validation data are passed to Keras,
    no early-stopping callback is installed, and the supplied decision
    threshold is applied unchanged to every held-out subject.

    This is intended as a post-selection diagnostic after another CV run has
    already chosen the hyperparameters, epoch count, and threshold. It does not
    perform another hyperparameter or threshold search.
    """
    if n_epochs < 1:
        raise ValueError("n_epochs must be at least 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    decision_threshold = float(decision_threshold)
    if not 0.0 < decision_threshold < 1.0:
        raise ValueError("decision_threshold must be strictly between 0 and 1.")

    model_config = dict(fixed_config)
    # The explicit post-selection values must override anything retained from
    # the original search result.
    model_config.pop("epochs", None)
    model_config.pop("batch_size", None)

    print(
        "\nFixed-config no-validation LOSOCV — "
        f"epochs={int(n_epochs)}, batch_size={int(batch_size)}, "
        f"decision_threshold={decision_threshold:.4f}",
        flush=True,
    )

    results = loso_cv(
        model_builder_function=model_builder_function,
        feature_array=feature_array,
        label_array=label_array,
        subject_id_array=subject_id_array,
        trial_id_array=trial_id_array,
        n_epochs=int(n_epochs),
        batch_size=int(batch_size),
        hyperparameters=model_config,
        preprocessing_strategy=preprocessing_strategy,
        evaluation_level=evaluation_level,
        selection_metric=selection_metric,
        selection_level=selection_level,
        maximize_metric=maximize_metric,
        metrics=metrics,
        log_predictions=log_predictions,
        log_variational_intervals=log_variational_intervals,
        n_prediction_latent_samples=n_prediction_latent_samples,
        latent_sampling_seed=latent_sampling_seed,
        n_uncertainty_samples=n_uncertainty_samples,
        ci_level=ci_level,
        validation_subjects_per_fold=0,
        validation_seed=None,
        early_stopping_patience=None,
        early_stopping_min_delta=0.0,
        early_stopping_monitor="loss",
        early_stopping_mode="min",
        restore_best_weights=False,
        prediction_diagnostics=prediction_diagnostics,
        prediction_diagnostics_every_n_epochs=(prediction_diagnostics_every_n_epochs),
        prediction_diagnostics_max_samples=prediction_diagnostics_max_samples,
        prediction_diagnostics_threshold_tolerance=(
            prediction_diagnostics_threshold_tolerance
        ),
        prediction_diagnostics_seed=prediction_diagnostics_seed,
        decision_thresholds=(decision_threshold,),
        threshold_selection_metric="balanced_accuracy",
        threshold_selection_level=selection_level,
        verbose=verbose,
        extra_fit_kwargs=extra_fit_kwargs,
        n_jobs=n_jobs,
        gpu_ids=gpu_ids,
        cpus_per_worker=cpus_per_worker,
        max_folds=max_folds,
        alternate_subject_sets=alternate_subject_sets,
        alternating_subject_seed=alternating_subject_seed,
        use_mldg=use_mldg,
        mldg_meta_train_subjects=mldg_meta_train_subjects,
        mldg_meta_test_subjects=mldg_meta_test_subjects,
        mldg_samples_per_subject=mldg_samples_per_subject,
        mldg_seed=mldg_seed,
    )

    if int(results.get("n_configs", 0)) != 1:
        raise RuntimeError(
            "fixed_loso_cv expected exactly one configuration, but loso_cv "
            f"reported {results.get('n_configs')}."
        )

    results.update(
        {
            "cv_strategy": "fixed_loso_no_validation",
            "hyperparameter_search": False,
            "post_selection_diagnostic": True,
            "fixed_epochs": int(n_epochs),
            "fixed_batch_size": int(batch_size),
            "fixed_decision_threshold": decision_threshold,
            "validation_subjects_per_fold": 0,
            "early_stopping_patience": None,
            "restore_best_weights": False,
        }
    )
    return results
