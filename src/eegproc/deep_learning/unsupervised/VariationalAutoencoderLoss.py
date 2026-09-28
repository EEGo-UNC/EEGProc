from __future__ import annotations

from typing import Literal

import tensorflow as tf

ReconstructionLoss = Literal["mse", "mae", "huber"]
FeatureReduction = Literal["sum", "mean"]


@tf.keras.utils.register_keras_serializable(package="eegproc")
class GradientReversal(tf.keras.layers.Layer):
    """Identity layer in the forward pass that reverses encoder gradients.

    The layer returns its input unchanged during the forward pass. During
    backpropagation, the incoming gradient is multiplied by
    ``-adversarial_weight``. Place it between the encoder representation and
    the subject-identification head:

    ``subject_logits = subject_head(GradientReversal(weight)(z_mean))``

    The subject head can then minimize ordinary subject cross-entropy, while
    the encoder receives the opposite gradient and learns to make subject
    identity harder to recover.

    Parameters
    ----------
    adversarial_weight:
        Non-negative strength of the reversed gradient reaching the encoder.
        This is the main hyperparameter controlling subject invariance. A
        value of 0.0 disables the adversarial encoder gradient; 1.0 reverses
        it at full strength.
    """

    def __init__(
        self,
        adversarial_weight: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if adversarial_weight < 0.0:
            raise ValueError(
                "adversarial_weight must be non-negative, "
                f"got {adversarial_weight}."
            )
        self.adversarial_weight = float(adversarial_weight)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        inputs = tf.convert_to_tensor(inputs)
        adversarial_weight = tf.cast(
            self.adversarial_weight,
            inputs.dtype,
        )

        @tf.custom_gradient
        def _reverse_gradient(x: tf.Tensor):
            def grad(upstream_gradient: tf.Tensor) -> tf.Tensor:
                return -adversarial_weight * upstream_gradient

            return tf.identity(x), grad

        return _reverse_gradient(inputs)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "adversarial_weight": self.adversarial_weight,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="eegproc")
class VariationalAutoencoderLoss:
    """Compute a configurable VAE objective with optional subject adversity.

    For each sample, the implemented scalar objective is

    ``reconstruction + beta * latent_kl + subject_loss_weight * subject_ce``.

    The subject term is deliberately *positive*. To make it adversarial for
    the encoder, ``subject_pred`` must be produced by a subject head whose
    input passes through :class:`GradientReversal`. This gives the desired
    parameter-specific optimization:

    - the subject head minimizes subject cross-entropy;
    - the encoder maximizes subject cross-entropy, scaled by the gradient-
      reversal ``adversarial_weight``;
    - the decoder is unaffected by the subject term.

    Both VAE terms can be reduced with either ``"sum"`` or ``"mean"`` across
    their non-batch coordinates:

    - ``sum/sum`` is the usual summed ELBO convention;
    - ``mean/mean`` is a dimension-normalized objective that is often easier
      to combine with O(1) classifier losses.

    Parameters
    ----------
    reconstruction:
        Element-wise reconstruction penalty: ``"mse"``, ``"mae"``, or
        ``"huber"``.
    beta:
        Non-negative multiplier on the reduced KL term.
    feature_reduction:
        Reduction across every non-batch reconstruction dimension.
    kl_reduction:
        Reduction across latent coordinates. When omitted, defaults to
        ``feature_reduction``.
    huber_delta:
        Positive transition point for Huber loss.
    log_var_clip_min, log_var_clip_max:
        Bounds applied to log variance before exponentiation.
    subject_loss_weight:
        Non-negative multiplier on subject cross-entropy in the scalar loss.
        Keep this at 1.0 in the usual gradient-reversal setup so the subject
        head remains strong. Control the encoder-side adversarial pressure
        primarily through ``GradientReversal(adversarial_weight=...)``.
        Set this to 0.0 to exclude the subject term from the objective.
    subject_from_logits:
        Whether ``subject_pred`` contains unnormalized logits. Using logits is
        recommended for numerical stability.
    """

    def __init__(
        self,
        reconstruction: ReconstructionLoss = "mse",
        beta: float = 1.0,
        feature_reduction: FeatureReduction = "mean",
        kl_reduction: FeatureReduction | None = None,
        huber_delta: float = 1.0,
        log_var_clip_min: float = -20.0,
        log_var_clip_max: float = 20.0,
        subject_loss_weight: float = 0.0,
        subject_from_logits: bool = True,
    ) -> None:
        if reconstruction not in {"mse", "mae", "huber"}:
            raise ValueError(
                "reconstruction must be one of {'mse', 'mae', 'huber'}, "
                f"got {reconstruction!r}."
            )
        if feature_reduction not in {"sum", "mean"}:
            raise ValueError(
                "feature_reduction must be 'sum' or 'mean', "
                f"got {feature_reduction!r}."
            )

        resolved_kl_reduction = kl_reduction or feature_reduction
        if resolved_kl_reduction not in {"sum", "mean"}:
            raise ValueError(
                "kl_reduction must be 'sum', 'mean', or None, "
                f"got {kl_reduction!r}."
            )
        if beta < 0.0:
            raise ValueError(f"beta must be non-negative, got {beta}.")
        if huber_delta <= 0.0:
            raise ValueError(
                f"huber_delta must be positive, got {huber_delta}."
            )
        if log_var_clip_min >= log_var_clip_max:
            raise ValueError(
                "log_var_clip_min must be smaller than log_var_clip_max; "
                f"got {log_var_clip_min} and {log_var_clip_max}."
            )
        if subject_loss_weight < 0.0:
            raise ValueError(
                "subject_loss_weight must be non-negative, "
                f"got {subject_loss_weight}."
            )

        self.reconstruction = reconstruction
        self.beta = float(beta)
        self.feature_reduction = feature_reduction
        self.kl_reduction = resolved_kl_reduction
        self.huber_delta = float(huber_delta)
        self.log_var_clip_min = float(log_var_clip_min)
        self.log_var_clip_max = float(log_var_clip_max)
        self.subject_loss_weight = float(subject_loss_weight)
        self.subject_from_logits = bool(subject_from_logits)

    def __call__(
        self,
        x_true: tf.Tensor,
        x_pred: tf.Tensor,
        z_mean: tf.Tensor,
        z_log_var: tf.Tensor,
        subject_true: tf.Tensor | None = None,
        subject_pred: tf.Tensor | None = None,
        include_subject_loss: bool = True,
    ) -> dict[str, tf.Tensor]:
        """Return scalar batch losses and reduced per-sample components.

        ``subject_true`` must contain integer, fold-local subject IDs and
        ``subject_pred`` must contain one logit/probability vector per sample.
        Set ``include_subject_loss=False`` for validation/test batches whose
        subjects were not part of the subject head's training classes.
        """
        reconstruction_loss_per_sample = self.compute_reconstruction_loss(
            x_true=x_true,
            x_pred=x_pred,
        )
        kl_loss_per_sample = self.compute_kl_loss(
            z_mean=z_mean,
            z_log_var=z_log_var,
        )

        weighted_kl_loss_per_sample = (
            tf.cast(self.beta, kl_loss_per_sample.dtype) * kl_loss_per_sample
        )

        subject_loss_per_sample = self._resolve_subject_loss(
            subject_true=subject_true,
            subject_pred=subject_pred,
            reference_loss=reconstruction_loss_per_sample,
            include_subject_loss=include_subject_loss,
        )
        weighted_subject_loss_per_sample = (
            tf.cast(
                self.subject_loss_weight,
                subject_loss_per_sample.dtype,
            )
            * subject_loss_per_sample
        )

        total_loss_per_sample = (
            reconstruction_loss_per_sample
            + weighted_kl_loss_per_sample
            + weighted_subject_loss_per_sample
        )

        return {
            "total_loss": tf.reduce_mean(total_loss_per_sample),
            "reconstruction_loss": tf.reduce_mean(
                reconstruction_loss_per_sample
            ),
            # Kept as ``kl_loss`` for compatibility. Its exact interpretation
            # is controlled by ``kl_reduction``.
            "kl_loss": tf.reduce_mean(kl_loss_per_sample),
            "weighted_kl_loss": tf.reduce_mean(
                weighted_kl_loss_per_sample
            ),
            "subject_loss": tf.reduce_mean(subject_loss_per_sample),
            "weighted_subject_loss": tf.reduce_mean(
                weighted_subject_loss_per_sample
            ),
            "reconstruction_loss_per_sample": reconstruction_loss_per_sample,
            "kl_loss_per_sample": kl_loss_per_sample,
            "subject_loss_per_sample": subject_loss_per_sample,
            "weighted_subject_loss_per_sample": (
                weighted_subject_loss_per_sample
            ),
        }

    def _resolve_subject_loss(
        self,
        subject_true: tf.Tensor | None,
        subject_pred: tf.Tensor | None,
        reference_loss: tf.Tensor,
        include_subject_loss: bool,
    ) -> tf.Tensor:
        if not include_subject_loss:
            return tf.zeros_like(reference_loss)

        if subject_true is None and subject_pred is None:
            if self.subject_loss_weight > 0.0:
                raise ValueError(
                    "subject_true and subject_pred are required when "
                    "subject_loss_weight is greater than 0."
                )
            return tf.zeros_like(reference_loss)

        if subject_true is None or subject_pred is None:
            raise ValueError(
                "subject_true and subject_pred must either both be provided "
                "or both be omitted."
            )

        subject_loss_per_sample = self.compute_subject_loss(
            subject_true=subject_true,
            subject_pred=subject_pred,
        )
        tf.debugging.assert_equal(
            tf.shape(subject_loss_per_sample)[0],
            tf.shape(reference_loss)[0],
            message=(
                "Subject loss and reconstruction loss must have the same "
                "batch dimension."
            ),
        )
        return tf.cast(subject_loss_per_sample, reference_loss.dtype)

    def compute_subject_loss(
        self,
        subject_true: tf.Tensor,
        subject_pred: tf.Tensor,
    ) -> tf.Tensor:
        """Compute sparse subject cross-entropy for every batch sample.

        Subject targets are ordinary integer subject IDs. They are not
        inverted, randomized, or replaced with uniform targets. The gradient
        reversal applied before the subject head creates the adversarial
        encoder update.
        """
        subject_pred = tf.convert_to_tensor(subject_pred)
        subject_true = tf.convert_to_tensor(subject_true)

        tf.debugging.assert_rank_at_least(
            subject_pred,
            2,
            message=(
                "subject_pred must include batch and subject-class "
                "dimensions."
            ),
        )
        tf.debugging.assert_all_finite(
            subject_pred,
            "subject_pred contains NaN or Inf values.",
        )

        # Permit labels shaped [batch, 1] in addition to the preferred [batch].
        if subject_true.shape.rank == 2 and subject_true.shape[-1] == 1:
            subject_true = tf.squeeze(subject_true, axis=-1)

        subject_true = tf.cast(subject_true, tf.int32)
        per_position_loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true=subject_true,
            y_pred=subject_pred,
            from_logits=self.subject_from_logits,
        )

        # If a temporal subject head emits [batch, time, classes], average its
        # per-time losses into one value per sample. Standard [batch, classes]
        # logits already produce a rank-1 [batch] loss and pass through.
        reduce_axes = tf.range(1, tf.rank(per_position_loss))
        return tf.cond(
            tf.greater(tf.rank(per_position_loss), 1),
            lambda: tf.reduce_mean(per_position_loss, axis=reduce_axes),
            lambda: per_position_loss,
        )

    def compute_reconstruction_loss(
        self,
        x_true: tf.Tensor,
        x_pred: tf.Tensor,
    ) -> tf.Tensor:
        """Compute one reduced reconstruction loss for every sample."""
        x_pred = tf.convert_to_tensor(x_pred)
        x_true = tf.cast(tf.convert_to_tensor(x_true), x_pred.dtype)

        tf.debugging.assert_rank_at_least(
            x_true,
            2,
            message="x_true must include batch and feature dimensions.",
        )
        tf.debugging.assert_equal(
            tf.shape(x_true),
            tf.shape(x_pred),
            message="x_true and x_pred must have identical shapes.",
        )
        tf.debugging.assert_all_finite(
            x_true,
            "x_true contains NaN or Inf values.",
        )
        tf.debugging.assert_all_finite(
            x_pred,
            "x_pred contains NaN or Inf values.",
        )

        error = x_true - x_pred
        if self.reconstruction == "mse":
            elementwise_loss = tf.square(error)
        elif self.reconstruction == "mae":
            elementwise_loss = tf.abs(error)
        else:
            elementwise_loss = self.compute_huber_loss(
                error=error,
                delta=self.huber_delta,
            )

        reduce_axes = tf.range(1, tf.rank(elementwise_loss))
        return self._reduce_non_batch_coordinates(
            elementwise_loss,
            reduce_axes,
            self.feature_reduction,
        )

    @staticmethod
    def _reduce_non_batch_coordinates(
        values: tf.Tensor,
        axes: tf.Tensor,
        reduction: FeatureReduction,
    ) -> tf.Tensor:
        if reduction == "sum":
            return tf.reduce_sum(values, axis=axes)
        return tf.reduce_mean(values, axis=axes)

    @staticmethod
    def compute_huber_loss(
        error: tf.Tensor,
        delta: float = 1.0,
    ) -> tf.Tensor:
        """Return element-wise Huber loss."""
        delta_tensor = tf.cast(delta, error.dtype)
        abs_error = tf.abs(error)
        quadratic = tf.minimum(abs_error, delta_tensor)
        linear = abs_error - quadratic
        return 0.5 * tf.square(quadratic) + delta_tensor * linear

    def compute_kl_loss(
        self,
        z_mean: tf.Tensor,
        z_log_var: tf.Tensor,
    ) -> tf.Tensor:
        """Compute one reduced KL value for every sample.

        ``z_log_var`` must contain ``log(sigma**2)``, not ``log(sigma)``.
        """
        z_mean = tf.convert_to_tensor(z_mean)
        z_log_var = tf.cast(tf.convert_to_tensor(z_log_var), z_mean.dtype)

        tf.debugging.assert_rank_at_least(
            z_mean,
            2,
            message="z_mean must include batch and latent dimensions.",
        )
        tf.debugging.assert_equal(
            tf.shape(z_mean),
            tf.shape(z_log_var),
            message="z_mean and z_log_var must have identical shapes.",
        )
        tf.debugging.assert_all_finite(
            z_mean,
            "z_mean contains NaN or Inf values.",
        )
        tf.debugging.assert_all_finite(
            z_log_var,
            "z_log_var contains NaN or Inf values.",
        )

        clipped_log_var = tf.clip_by_value(
            z_log_var,
            tf.cast(self.log_var_clip_min, z_log_var.dtype),
            tf.cast(self.log_var_clip_max, z_log_var.dtype),
        )
        kl_per_coordinate = 0.5 * (
            tf.square(z_mean)
            + tf.exp(clipped_log_var)
            - 1.0
            - clipped_log_var
        )

        if self.kl_reduction == "sum":
            return tf.reduce_sum(kl_per_coordinate, axis=-1)
        return tf.reduce_mean(kl_per_coordinate, axis=-1)

    def get_config(self) -> dict:
        """Return a Keras-serializable configuration."""
        return {
            "reconstruction": self.reconstruction,
            "beta": self.beta,
            "feature_reduction": self.feature_reduction,
            "kl_reduction": self.kl_reduction,
            "huber_delta": self.huber_delta,
            "log_var_clip_min": self.log_var_clip_min,
            "log_var_clip_max": self.log_var_clip_max,
            "subject_loss_weight": self.subject_loss_weight,
            "subject_from_logits": self.subject_from_logits,
        }
