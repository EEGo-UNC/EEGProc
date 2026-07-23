from __future__ import annotations

from typing import Literal

import tensorflow as tf

ReconstructionLoss = Literal["mse", "mae", "huber"]
FeatureReduction = Literal["sum", "mean"]


@tf.keras.utils.register_keras_serializable(package="eegproc")
class VariationalAutoencoderLoss:
    """Compute a configurable diagonal-Gaussian VAE objective.

    For each sample, the loss is

    ``reconstruction_loss + beta * latent_kl_loss``.

    Both terms can be reduced with either ``"sum"`` or ``"mean"`` across
    their non-batch coordinates:

    - ``sum/sum`` is the usual summed ELBO convention.
    - ``mean/mean`` is a dimension-normalized objective that is often easier
      to combine with an O(1) classifier loss, but ``beta=1`` under this
      convention is not numerically identical to the standard summed ELBO.

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
        Reduction across latent coordinates. When omitted, it defaults to
        ``feature_reduction`` so the reconstruction and KL terms use
        consistent reduction conventions.
    huber_delta:
        Positive transition point for Huber loss.
    log_var_clip_min, log_var_clip_max:
        Bounds applied to log variance before exponentiation. These prevent
        ``exp(z_log_var)`` from overflowing while preserving a broad range of
        posterior variances.
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

        self.reconstruction = reconstruction
        self.beta = float(beta)
        self.feature_reduction = feature_reduction
        self.kl_reduction = resolved_kl_reduction
        self.huber_delta = float(huber_delta)
        self.log_var_clip_min = float(log_var_clip_min)
        self.log_var_clip_max = float(log_var_clip_max)

    def __call__(
        self,
        x_true: tf.Tensor,
        x_pred: tf.Tensor,
        z_mean: tf.Tensor,
        z_log_var: tf.Tensor,
    ) -> dict[str, tf.Tensor]:
        """Return scalar batch losses and the reduced per-sample components."""
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
        total_loss_per_sample = (
            reconstruction_loss_per_sample + weighted_kl_loss_per_sample
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
            "reconstruction_loss_per_sample": reconstruction_loss_per_sample,
            "kl_loss_per_sample": kl_loss_per_sample,
        }

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
        }
