"""Array coercion helpers. NumPy only - deliberately free of TensorFlow so the
lowest layer stays cheap to import and trivial to unit-test."""

from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotations only
    import tensorflow as tf

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


def _python_scalar(value):
    """Convert numpy scalars to plain Python scalars for logs/JSON."""
    if isinstance(value, np.generic):
        return value.item()
    return value
