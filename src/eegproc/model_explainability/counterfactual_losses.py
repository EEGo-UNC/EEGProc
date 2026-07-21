"""Losses for reconstructed-space EEG counterfactual optimization.

The deterministic objective is

    L_cf =
        lambda_valid * L_valid
        + lambda_signal * d(D(z_cf), x)

where:

- ``z_cf`` is the optimized latent sequence;
- ``D`` is the frozen decoder;
- ``x`` is the original preprocessed EEG feature window.

There is deliberately no latent-space proximity term. The latent variable is
only the optimization parameter; counterfactual size is defined and evaluated
in the decoder's reconstructed signal space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import tensorflow as tf


DistanceMetric = Literal["mse", "mae", "rmse"]


@dataclass(frozen=True, slots=True)
class CounterfactualLossWeights:
    """Weights for validity and reconstructed-signal fidelity."""

    validity: float = 1.0
    signal_proximity: float = 0.10

    def __post_init__(self) -> None:
        if self.validity < 0.0:
            raise ValueError("validity weight must be non-negative.")
        if self.signal_proximity < 0.0:
            raise ValueError("signal_proximity weight must be non-negative.")
        if self.validity == 0.0 and self.signal_proximity == 0.0:
            raise ValueError("At least one loss weight must be positive.")


def normalize_target_classes(
    target_class: int | tf.Tensor,
    batch_size: int | tf.Tensor,
) -> tf.Tensor:
    """Return one integer target class for each batch element."""
    targets = tf.cast(
        tf.reshape(tf.convert_to_tensor(target_class), [-1]),
        tf.int32,
    )
    batch_size = tf.cast(batch_size, tf.int32)

    targets = tf.cond(
        tf.equal(tf.size(targets), 1),
        lambda: tf.repeat(targets, repeats=batch_size),
        lambda: targets,
    )
    tf.debugging.assert_equal(
        tf.shape(targets)[0],
        batch_size,
        message=(
            "target_class must be a scalar or contain one class ID per "
            "batch element."
        ),
    )
    return targets


def target_probabilities(
    logits: tf.Tensor,
    target_class: int | tf.Tensor,
) -> tf.Tensor:
    """Extract the target-class softmax probability for each example."""
    logits = tf.convert_to_tensor(logits)
    if logits.shape.rank != 2:
        raise ValueError(
            "logits must have shape (batch, n_classes); "
            f"received {logits.shape}."
        )

    targets = normalize_target_classes(target_class, tf.shape(logits)[0])
    probabilities = tf.nn.softmax(logits, axis=-1)
    indices = tf.stack(
        [tf.range(tf.shape(logits)[0], dtype=tf.int32), targets],
        axis=1,
    )
    return tf.gather_nd(probabilities, indices)


def targeted_cross_entropy_loss(
    logits: tf.Tensor,
    target_class: int | tf.Tensor,
) -> tf.Tensor:
    """Targeted sparse cross-entropy averaged over the batch."""
    logits = tf.convert_to_tensor(logits)
    targets = normalize_target_classes(target_class, tf.shape(logits)[0])
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=targets,
        logits=logits,
    )
    return tf.reduce_mean(losses)


def target_probability_hinge_loss(
    logits: tf.Tensor,
    target_class: int | tf.Tensor,
    minimum_probability: float = 0.80,
    epsilon: float = 1e-7,
) -> tf.Tensor:
    """Penalize the target class only while its probability is below a threshold.

    For each example:

        max(0, log(p_min) - log p(y_target | z_cf))

    The validity loss is zero after the requested probability is reached.
    """
    if not 0.0 < minimum_probability < 1.0:
        raise ValueError("minimum_probability must be strictly between 0 and 1.")

    logits = tf.convert_to_tensor(logits)
    targets = normalize_target_classes(target_class, tf.shape(logits)[0])
    log_probabilities = tf.nn.log_softmax(logits, axis=-1)

    indices = tf.stack(
        [tf.range(tf.shape(logits)[0], dtype=tf.int32), targets],
        axis=1,
    )
    target_log_probability = tf.gather_nd(log_probabilities, indices)

    dtype = logits.dtype
    threshold = tf.clip_by_value(
        tf.cast(minimum_probability, dtype),
        tf.cast(epsilon, dtype),
        tf.cast(1.0 - epsilon, dtype),
    )
    losses = tf.nn.relu(tf.math.log(threshold) - target_log_probability)
    return tf.reduce_mean(losses)


def tensor_distance(
    candidate: tf.Tensor,
    reference: tf.Tensor,
    metric: DistanceMetric = "mse",
    epsilon: float = 1e-8,
) -> tf.Tensor:
    """Compute a batch-mean distance over all non-batch dimensions."""
    candidate = tf.convert_to_tensor(candidate)
    reference = tf.cast(tf.convert_to_tensor(reference), candidate.dtype)

    tf.debugging.assert_equal(
        tf.shape(candidate),
        tf.shape(reference),
        message="Candidate and reference tensors must have identical shapes.",
    )

    difference = tf.reshape(
        candidate - reference,
        [tf.shape(candidate)[0], -1],
    )

    if metric == "mse":
        per_example = tf.reduce_mean(tf.square(difference), axis=1)
    elif metric == "mae":
        per_example = tf.reduce_mean(tf.abs(difference), axis=1)
    elif metric == "rmse":
        per_example = tf.sqrt(
            tf.reduce_mean(tf.square(difference), axis=1)
            + tf.cast(epsilon, candidate.dtype)
        )
    else:
        raise ValueError(
            f"Unknown metric {metric!r}; choose 'mse', 'mae', or 'rmse'."
        )

    return tf.reduce_mean(per_example)


def reconstructed_input_proximity_loss(
    counterfactual_reconstruction: tf.Tensor,
    original_features: tf.Tensor,
    metric: DistanceMetric = "mse",
) -> tf.Tensor:
    """Distance from decoded counterfactual EEG to the original feature window."""
    return tensor_distance(
        candidate=counterfactual_reconstruction,
        reference=original_features,
        metric=metric,
    )


def decoded_change_distance(
    counterfactual_reconstruction: tf.Tensor,
    original_reconstruction: tf.Tensor,
    metric: DistanceMetric = "mse",
) -> tf.Tensor:
    """Diagnostic distance between counterfactual and original reconstructions."""
    return tensor_distance(
        candidate=counterfactual_reconstruction,
        reference=original_reconstruction,
        metric=metric,
    )


def counterfactual_objective(
    *,
    logits: tf.Tensor,
    target_class: int | tf.Tensor,
    counterfactual_reconstruction: tf.Tensor,
    original_features: tf.Tensor,
    weights: CounterfactualLossWeights,
    target_probability: float = 0.80,
    signal_metric: DistanceMetric = "mse",
) -> dict[str, tf.Tensor]:
    """Compute the reconstructed-space counterfactual objective."""
    validity = target_probability_hinge_loss(
        logits=logits,
        target_class=target_class,
        minimum_probability=target_probability,
    )
    reconstructed_input = reconstructed_input_proximity_loss(
        counterfactual_reconstruction=counterfactual_reconstruction,
        original_features=original_features,
        metric=signal_metric,
    )

    dtype = validity.dtype
    weighted_validity = tf.cast(weights.validity, dtype) * validity
    weighted_signal = (
        tf.cast(weights.signal_proximity, dtype) * reconstructed_input
    )
    total = weighted_validity + weighted_signal

    return {
        "total": total,
        "validity": validity,
        "reconstructed_input_proximity": reconstructed_input,
        "weighted_validity": weighted_validity,
        "weighted_reconstructed_input_proximity": weighted_signal,
    }


def diagonal_gaussian_kl(
    mean_q: tf.Tensor,
    log_var_q: tf.Tensor,
    mean_p: tf.Tensor,
    log_var_p: tf.Tensor,
) -> tf.Tensor:
    """Batch-mean KL(q || p) for diagonal Gaussian distributions.

    This is retained for a later distributional counterfactual objective. It is
    not part of the deterministic reconstructed-space objective.
    """
    mean_q = tf.convert_to_tensor(mean_q)
    dtype = mean_q.dtype
    log_var_q = tf.cast(log_var_q, dtype)
    mean_p = tf.cast(mean_p, dtype)
    log_var_p = tf.cast(log_var_p, dtype)

    tf.debugging.assert_equal(tf.shape(mean_q), tf.shape(log_var_q))
    tf.debugging.assert_equal(tf.shape(mean_q), tf.shape(mean_p))
    tf.debugging.assert_equal(tf.shape(mean_q), tf.shape(log_var_p))

    variance_ratio = tf.exp(log_var_q - log_var_p)
    mean_term = tf.square(mean_p - mean_q) * tf.exp(-log_var_p)
    elementwise = (
        log_var_p
        - log_var_q
        + variance_ratio
        + mean_term
        - 1.0
    )

    per_example = 0.5 * tf.reduce_sum(
        tf.reshape(elementwise, [tf.shape(elementwise)[0], -1]),
        axis=1,
    )
    return tf.reduce_mean(per_example)


def symmetric_diagonal_gaussian_kl(
    mean_a: tf.Tensor,
    log_var_a: tf.Tensor,
    mean_b: tf.Tensor,
    log_var_b: tf.Tensor,
) -> tf.Tensor:
    """Symmetric KL distance for future distributional counterfactuals."""
    return 0.5 * (
        diagonal_gaussian_kl(mean_a, log_var_a, mean_b, log_var_b)
        + diagonal_gaussian_kl(mean_b, log_var_b, mean_a, log_var_a)
    )
