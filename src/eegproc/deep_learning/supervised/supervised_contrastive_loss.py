"""Supervised contrastive regularization for EEG representations.

The loss operates on a rank-2 embedding produced by an encoder or by the
pre-logit portion of a classifier. It is intentionally independent of any
particular prediction head so it can be used with dense, hybrid, or
variational classifiers.
"""

from __future__ import annotations

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="EEGProc")
class SupervisedContrastiveLoss(tf.keras.layers.Layer):
    """Compute supervised contrastive loss and batch diagnostics.

    By default, a positive pair must share the same class label and come from
    different subjects. Same-label samples from the same subject are then
    excluded from both the positive set and the denominator. This focuses the
    objective on subject-invariant class structure instead of letting repeated
    windows from one subject dominate the contrastive signal.

    The returned ``positive_pairs`` value counts directed anchor-positive
    pairs. Anchors without a valid positive are excluded from the loss mean and
    reflected in ``valid_anchor_fraction``.
    """

    def __init__(
        self,
        temperature: float = 0.10,
        cross_subject_only: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        self.temperature = float(temperature)
        self.cross_subject_only = bool(cross_subject_only)

    def call(
        self,
        embeddings: tf.Tensor,
        labels: tf.Tensor,
        subject_ids: tf.Tensor | None = None,
        sample_weight: tf.Tensor | None = None,
    ) -> dict[str, tf.Tensor]:
        """Return the scalar loss and positive-pair diagnostics."""
        embeddings = tf.convert_to_tensor(embeddings)
        if embeddings.shape.rank != 2:
            raise ValueError("SupCon embeddings must be rank 2.")

        # Accumulate similarities in float32 for mixed-precision stability.
        # Gradients still propagate to the original embedding tensor.
        embeddings_float = tf.cast(embeddings, tf.float32)
        labels = tf.cast(tf.reshape(labels, [-1]), tf.int32)
        batch_size = tf.shape(embeddings_float)[0]
        tf.debugging.assert_equal(
            batch_size,
            tf.shape(labels)[0],
            message="labels must align with the SupCon embedding batch.",
        )

        normalized = tf.math.l2_normalize(
            embeddings_float,
            axis=-1,
            epsilon=1e-12,
        )
        similarity = tf.matmul(normalized, normalized, transpose_b=True)
        similarity /= tf.cast(self.temperature, similarity.dtype)

        non_self = tf.logical_not(tf.eye(batch_size, dtype=tf.bool))
        same_label = tf.equal(labels[:, tf.newaxis], labels[tf.newaxis, :])

        if self.cross_subject_only:
            if subject_ids is None:
                zero = tf.zeros((), dtype=similarity.dtype)
                return {
                    "loss": zero,
                    "valid_anchor_fraction": zero,
                    "positive_pairs": zero,
                }
            subject_ids = tf.cast(tf.reshape(subject_ids, [-1]), tf.int32)
            tf.debugging.assert_equal(
                tf.shape(subject_ids)[0],
                batch_size,
                message="subject_ids must align with the SupCon embedding batch.",
            )
            same_subject = tf.equal(
                subject_ids[:, tf.newaxis],
                subject_ids[tf.newaxis, :],
            )
            positive_mask = tf.logical_and(
                non_self,
                tf.logical_and(same_label, tf.logical_not(same_subject)),
            )
            ignored_same_subject_positive = tf.logical_and(
                non_self,
                tf.logical_and(same_label, same_subject),
            )
            denominator_mask = tf.logical_and(
                non_self,
                tf.logical_not(ignored_same_subject_positive),
            )
        else:
            positive_mask = tf.logical_and(non_self, same_label)
            denominator_mask = non_self

        # A finite mask value keeps homogeneous or single-sample batches from
        # producing NaNs when an anchor has no eligible denominator entries.
        mask_value = tf.cast(-1e9, similarity.dtype)
        masked_similarity = tf.where(
            denominator_mask,
            similarity,
            mask_value,
        )
        row_max = tf.stop_gradient(
            tf.reduce_max(masked_similarity, axis=1, keepdims=True)
        )
        stabilized = similarity - row_max
        masked_stabilized = tf.where(
            denominator_mask,
            stabilized,
            mask_value,
        )
        log_denominator = tf.reduce_logsumexp(
            masked_stabilized,
            axis=1,
            keepdims=True,
        )
        log_probability = stabilized - log_denominator

        positive_count = tf.reduce_sum(
            tf.cast(positive_mask, similarity.dtype),
            axis=1,
        )
        positive_log_probability = tf.where(
            positive_mask,
            log_probability,
            tf.zeros_like(log_probability),
        )
        mean_positive_log_probability = tf.math.divide_no_nan(
            tf.reduce_sum(positive_log_probability, axis=1),
            positive_count,
        )
        valid_anchor = positive_count > 0.0
        per_anchor_loss = tf.where(
            valid_anchor,
            -mean_positive_log_probability,
            tf.zeros_like(mean_positive_log_probability),
        )

        anchor_weights = tf.cast(valid_anchor, per_anchor_loss.dtype)
        if sample_weight is not None:
            weights = tf.cast(
                tf.reshape(sample_weight, [-1]),
                per_anchor_loss.dtype,
            )
            tf.debugging.assert_equal(
                tf.shape(weights)[0],
                batch_size,
                message="sample_weight must align with the SupCon embedding batch.",
            )
            anchor_weights *= weights

        loss = tf.math.divide_no_nan(
            tf.reduce_sum(per_anchor_loss * anchor_weights),
            tf.reduce_sum(anchor_weights),
        )
        return {
            "loss": loss,
            "valid_anchor_fraction": tf.reduce_mean(
                tf.cast(valid_anchor, similarity.dtype)
            ),
            "positive_pairs": tf.reduce_sum(
                tf.cast(positive_mask, similarity.dtype)
            ),
        }

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "temperature": self.temperature,
                "cross_subject_only": self.cross_subject_only,
            }
        )
        return config
