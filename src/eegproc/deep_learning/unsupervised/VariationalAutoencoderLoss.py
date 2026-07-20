from __future__ import annotations

from typing import Literal, Dict

import tensorflow as tf

ReconstructionLoss = Literal["mse", "mae", "huber"]
FeatureReduction = Literal["sum", "mean"]


class VariationalAutoencoderLoss:
    """
    Generic variational autoencoder loss that learns an approximate posterior q(z|x)
    over a latent variable z given an input x.

    L = reconstruction_loss(x, x_hat) + beta * KL(q(z|x) || p(z))

    where p(z) is assumed to be N(0, I).

    Parameters
    ----------
    reconstruction:
        Reconstruction loss type. Usually "mse" or "huber" for EEG/time-series data.
    beta:
        Weight on the KL divergence term. beta=1 gives the standard VAE loss.
    feature_reduction:
        Whether to sum or average reconstruction loss across non-batch dimensions.
        "sum" is closer to the standard ELBO formulation.
        "mean" is often more stable across different input sizes.
    huber_delta:
        Threshold where Huber loss changes from quadratic to linear.
        Only used when reconstruction="huber".
    """

    def __init__(
        self,
        reconstruction: ReconstructionLoss = "mse",
        beta: float = 1.0,
        feature_reduction: FeatureReduction = "mean",
        huber_delta: float = 1.0,
    ) -> None:
        self.reconstruction = reconstruction
        self.beta = beta
        self.feature_reduction = feature_reduction
        self.huber_delta = huber_delta

    def __call__(
        self,
        x_true: tf.Tensor,
        x_pred: tf.Tensor,
        z_mean: tf.Tensor,
        z_log_var: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        """
        Compute total VAE loss and individual loss components.

        Parameters
        ----------
        x_true:
            Original input.
            Shape: (batch, ...)
        x_pred:
            Reconstructed input.
            Shape: (batch, ...)
        z_mean:
            Latent Gaussian mean.
            Shape: (batch, latent_dim)
        z_log_var:
            Latent Gaussian log variance.
            Shape: (batch, latent_dim)

        Returns
        -------
        dict with:
            total_loss
            reconstruction_loss
            kl_loss
        """

        reconstruction_loss = self.compute_reconstruction_loss(
            x_true=x_true,
            x_pred=x_pred,
        )

        kl_loss = self.compute_kl_loss(
            z_mean=z_mean,
            z_log_var=z_log_var,
        )

        weighted_kl_loss = tf.cast(self.beta, kl_loss.dtype) * kl_loss
        total_loss_per_sample = reconstruction_loss + weighted_kl_loss

        return {
            "total_loss": tf.reduce_mean(total_loss_per_sample),
            "reconstruction_loss": tf.reduce_mean(reconstruction_loss),
            "kl_loss": tf.reduce_mean(kl_loss),
            "weighted_kl_loss": tf.reduce_mean(weighted_kl_loss),
        }

    def compute_reconstruction_loss(
        self,
        x_true: tf.Tensor,
        x_pred: tf.Tensor,
    ) -> tf.Tensor:
        """
        Computes per-sample reconstruction loss.
        """

        x_true = tf.cast(x_true, x_pred.dtype)
        error = x_true - x_pred

        if self.reconstruction == "mse":
            loss = tf.square(error)
        elif self.reconstruction == "mae":
            loss = tf.abs(error)
        elif self.reconstruction == "huber":
            loss = self.compute_huber_loss(
                error=error,
                delta=self.huber_delta,
            )
        else:
            raise ValueError(f"Unknown reconstruction loss: {self.reconstruction}")

        reduce_axes = tf.range(1, tf.rank(loss))

        if self.feature_reduction == "sum":
            return tf.reduce_sum(loss, axis=reduce_axes)

        if self.feature_reduction == "mean":
            return tf.reduce_mean(loss, axis=reduce_axes)

        raise ValueError(f"Unknown feature reduction: {self.feature_reduction}")

    @staticmethod
    def compute_huber_loss(
        error: tf.Tensor,
        delta: float = 1.0,
    ) -> tf.Tensor:
        """
        Element-wise Huber loss. MSE for small errors, MAE for large errors.
        """

        abs_error = tf.abs(error)
        quadratic = tf.minimum(abs_error, delta)
        linear = abs_error - quadratic
        return 0.5 * tf.square(quadratic) + delta * linear

    @staticmethod
    def compute_kl_loss(
        z_mean: tf.Tensor,
        z_log_var: tf.Tensor,
    ) -> tf.Tensor:
        """
        KL divergence between q(z|x) = N(z_mean, exp(z_log_var))
        and p(z) = N(0, I).
        """

        return -0.5 * tf.reduce_mean(
            1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
            axis=-1,
        )
