"""Joint autoencoder + variational-classifier architecture for EEG.

This module defines a first-pass semi-supervised model that shares a single
encoder pathway and optimizes reconstruction and classification objectives in
one gradient step.

Pipeline
--------
1. Raw EEG input -> ``encoder`` (e.g., ``CNN1DEncoder``) -> latent sequence.
2. Latent sequence -> temporal pooling -> ``variational_classifier`` logits.
3. Latent sequence -> ``decoder`` (e.g., ``CNN1DDecoder``) -> reconstruction.

Training objective
------------------
The model minimizes a weighted sum in a single backward pass:

    total_loss = ae_loss_weight * reconstruction_loss
               + vc_loss_weight * variational_classifier_loss

Both weights are tunable and default to ``0.5``.
"""

from __future__ import annotations

from collections.abc import Callable

import tensorflow as tf
from tensorflow.keras import layers


class JointAutoencoderVariationalClassifierV1(tf.keras.Model):
    """Combined autoencoder and variational-classification model.

    Parameters
    ----------
    encoder : tf.keras.Model
        Sequence encoder that maps ``(batch, timesteps, n_features)`` to
        ``(batch, t_latent, emb_dim)``.
    decoder : tf.keras.Model
        Sequence decoder that reconstructs inputs from the encoder output.
    variational_classifier : tf.keras.layers.Layer
        Classification head that accepts pooled latent vectors of shape
        ``(batch, emb_dim)`` and returns class logits.
    ae_loss_weight : float, default=0.5
        Weight for reconstruction loss.
    vc_loss_weight : float, default=0.5
        Weight for variational-classifier loss.
    reconstruction_loss_fn : Callable | None, default=None
        Loss used for the autoencoder branch. Defaults to mean squared error.
    vc_alpha : float, default=1.0
        Cross-entropy coefficient for ``variational_classifier.vc_loss``.
    vc_beta : float, default=0.0
        Encoder/prior KL coefficient for ``variational_classifier.vc_loss``.
    vc_gamma : float, default=1e-4
        Gaussian analytic term coefficient for ``variational_classifier.vc_loss``.
    vc_lambda : float, default=0.0
        Class-prior KL coefficient for ``variational_classifier.vc_loss``.
    name : str, default="joint_autoencoder_variational_classifier"
        Keras model name.
    """

    def __init__(
        self,
        encoder: tf.keras.Model,
        decoder: tf.keras.Model,
        variational_classifier: tf.keras.layers.Layer,
        ae_loss_weight: float = 0.5,
        vc_loss_weight: float = 0.5,
        reconstruction_loss_fn: Callable | None = None,
        vc_alpha: float = 1.0,
        vc_beta: float = 0.0,
        vc_gamma: float = 1e-4,
        vc_lambda: float = 0.0,
        name: str = "joint_autoencoder_variational_classifier",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        if ae_loss_weight < 0.0 or vc_loss_weight < 0.0:
            raise ValueError("Loss weights must be non-negative.")
        if ae_loss_weight == 0.0 and vc_loss_weight == 0.0:
            raise ValueError("At least one loss weight must be > 0.")

        self.encoder = encoder
        self.decoder = decoder
        self.variational_classifier = variational_classifier

        self.ae_loss_weight = float(ae_loss_weight)
        self.vc_loss_weight = float(vc_loss_weight)

        self.reconstruction_loss_fn = (
            reconstruction_loss_fn
            if reconstruction_loss_fn is not None
            else tf.keras.losses.MeanSquaredError(
                reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
            )
        )

        self.vc_alpha = float(vc_alpha)
        self.vc_beta = float(vc_beta)
        self.vc_gamma = float(vc_gamma)
        self.vc_lambda = float(vc_lambda)

        self.temporal_pool = layers.GlobalAveragePooling1D(name="latent_temporal_pool")

        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vc_loss_tracker = tf.keras.metrics.Mean(name="vc_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vc_loss_tracker,
        ]

    def call(self, inputs, training: bool = False):
        """Run the joint forward pass.

        Parameters
        ----------
        inputs : tf.Tensor
            Raw EEG tensor of shape ``(batch, timesteps, n_features)``.

        Returns
        -------
        dict[str, tf.Tensor]
            ``latent_sequence``: encoder output.
            ``pooled_latent``: pooled latent vectors for classification.
            ``logits``: classifier logits.
            ``reconstruction``: decoder output.
        """
        latent_sequence = self.encoder(inputs, training=training)
        pooled_latent = self.temporal_pool(latent_sequence)
        logits = self.variational_classifier(pooled_latent, training=training)
        reconstruction = self.decoder(latent_sequence, training=training)

        return {
            "latent_sequence": latent_sequence,
            "pooled_latent": pooled_latent,
            "logits": logits,
            "reconstruction": reconstruction,
        }

    def _compute_weighted_losses(self, x, y, training: bool):
        outputs = self(x, training=training)

        reconstruction_loss = self.reconstruction_loss_fn(x, outputs["reconstruction"])

        y = tf.cast(tf.reshape(y, [-1]), tf.int32)
        vc_loss = self.variational_classifier.vc_loss(
            mh=outputs["pooled_latent"],
            y=y,
            alpha=self.vc_alpha,
            beta=self.vc_beta,
            gamma=self.vc_gamma,
            lambda_=self.vc_lambda,
        )

        total_loss = (
            self.ae_loss_weight * reconstruction_loss
            + self.vc_loss_weight * vc_loss
        )

        return total_loss, reconstruction_loss, vc_loss

    def train_step(self, data):
        """Train one step with a single gradient update on weighted loss."""
        if isinstance(data, tuple):
            if len(data) == 2:
                x, y = data
            elif len(data) == 3:
                x, y, _sample_weight = data
            else:
                raise ValueError("Expected (x, y) or (x, y, sample_weight).")
        else:
            raise ValueError("Expected data as (x, y) tuple.")

        with tf.GradientTape() as tape:
            total_loss, reconstruction_loss, vc_loss = self._compute_weighted_losses(
                x=x,
                y=y,
                training=True,
            )

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vc_loss_tracker.update_state(vc_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vc_loss": self.vc_loss_tracker.result(),
        }

    def test_step(self, data):
        """Evaluate one step using the same weighted objective."""
        if isinstance(data, tuple):
            if len(data) == 2:
                x, y = data
            elif len(data) == 3:
                x, y, _sample_weight = data
            else:
                raise ValueError("Expected (x, y) or (x, y, sample_weight).")
        else:
            raise ValueError("Expected data as (x, y) tuple.")

        total_loss, reconstruction_loss, vc_loss = self._compute_weighted_losses(
            x=x,
            y=y,
            training=False,
        )

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vc_loss_tracker.update_state(vc_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vc_loss": self.vc_loss_tracker.result(),
        }

    def get_config(self) -> dict:
        """Return lightweight serializable configuration for loss settings."""
        config = super().get_config()
        config.update(
            {
                "ae_loss_weight": self.ae_loss_weight,
                "vc_loss_weight": self.vc_loss_weight,
                "vc_alpha": self.vc_alpha,
                "vc_beta": self.vc_beta,
                "vc_gamma": self.vc_gamma,
                "vc_lambda": self.vc_lambda,
            }
        )
        return config
