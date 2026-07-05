"""Joint autoencoder + variational-classifier architecture for EEG.

This module defines a second-pass semi-supervised model that shares a single
encoder pathway and optimizes reconstruction and classification objectives in
one gradient step, while also allowing the variational-classifier head's
internal losses to be trained explicitly.

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

In addition, the variational-classifier discriminator can optionally receive
its own gradient step by enabling ``update_discriminator``.
"""

from __future__ import annotations

from collections.abc import Callable

import tensorflow as tf
from tensorflow.keras import layers

from ..unsupervised.VariationalAutoencoderLoss import VariationalAutoencoderLoss


class JointAutoencoderVariationalClassifierV2(tf.keras.Model):
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
        Loss used for the autoencoder branch. Defaults to a VAE loss with
        mean-squared reconstruction and KL regularization.
    vc_alpha : float, default=1.0
        Cross-entropy coefficient for ``variational_classifier.vc_loss``.
    vc_beta : float, default=0.0
        Encoder/prior KL coefficient for ``variational_classifier.vc_loss``.
    vc_gamma : float, default=1e-4
        Gaussian analytic term coefficient for ``variational_classifier.vc_loss``.
    vc_lambda : float, default=0.0
        Class-prior KL coefficient for ``variational_classifier.vc_loss``.
    update_discriminator : bool, default=False
        Whether to run a separate discriminator gradient step after the main
        joint update.
    name : str, default="joint_autoencoder_variational_classifier_v2"
        Keras model name.
    """

    def __init__(
        self,
        encoder: tf.keras.Model,
        decoder: tf.keras.Model,
        variational_classifier: tf.keras.layers.Layer,
        ae_loss_weight: float = 0.5,
        vc_loss_weight: float = 0.5,
        reconstruction_loss_fn: VariationalAutoencoderLoss | None = None,
        vc_alpha: float = 1.0,
        vc_beta: float = 0.0,
        vc_gamma: float = 1e-4,
        vc_lambda: float = 0.0,
        update_discriminator: bool = False,
        name: str = "joint_autoencoder_variational_classifier_v2",
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
            else VariationalAutoencoderLoss(
                reconstruction="mse",
                beta=1.0,
                feature_reduction="mean",
            )
        )

        self.vc_alpha = float(vc_alpha)
        self.vc_beta = float(vc_beta)
        self.vc_gamma = float(vc_gamma)
        self.vc_lambda = float(vc_lambda)
        self.update_discriminator = bool(update_discriminator)

        self.temporal_pool = layers.GlobalAveragePooling1D(name="latent_temporal_pool")

        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vc_loss_tracker = tf.keras.metrics.Mean(name="vc_loss")
        self.accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(
            name="accuracy"
        )

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vc_loss_tracker,
            self.accuracy_tracker,
        ]

    def compile(self, optimizer=None, discriminator_optimizer=None, **kwargs):
        """Compile the model, eagerly preparing the discriminator optimizer.

        The discriminator optimizer must be created here (or lazily, but only
        from a guaranteed-eager context) rather than inside ``train_step``.
        ``train_step`` is traced into a graph the first time ``model.fit()``
        runs it, and cloning ``self.optimizer`` via
        ``self.optimizer.__class__.from_config(self.optimizer.get_config())``
        requires reading the optimizer's current learning rate with
        ``.numpy()`` -- which raises ``NotImplementedError`` during tracing.
        Building the clone here, at ``compile()`` time, guarantees it happens
        eagerly, before any graph tracing can occur.

        Parameters
        ----------
        discriminator_optimizer : tf.keras.optimizers.Optimizer | None
            Optimizer to use for the discriminator's own gradient step. If
            omitted and ``update_discriminator`` is True, a fresh optimizer
            is cloned from ``optimizer``'s config.
        """
        super().compile(optimizer=optimizer, **kwargs)

        if self.update_discriminator:
            if discriminator_optimizer is not None:
                self._discriminator_optimizer = discriminator_optimizer
            elif not hasattr(self, "_discriminator_optimizer"):
                self._discriminator_optimizer = self.optimizer.__class__.from_config(
                    self.optimizer.get_config()
                )

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

        latent_sequence = outputs["latent_sequence"]
        z_mean = tf.reshape(latent_sequence, [tf.shape(latent_sequence)[0], -1])
        z_log_var = tf.zeros_like(z_mean)

        vae_losses = self.reconstruction_loss_fn(
            x_true=x,
            x_pred=outputs["reconstruction"],
            z_mean=z_mean,
            z_log_var=z_log_var,
        )
        reconstruction_loss = vae_losses["reconstruction_loss"]
        ae_loss = vae_losses["total_loss"]

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
            self.ae_loss_weight * ae_loss
            + self.vc_loss_weight * vc_loss
        )

        return total_loss, reconstruction_loss, vc_loss, outputs

    def _discriminator_variables(self):
        if not hasattr(self.variational_classifier, "disc_w"):
            return []
        return [self.variational_classifier.disc_w, self.variational_classifier.disc_b]

    def _apply_gradients(self, optimizer, gradients, variables):
        filtered_pairs = [
            (gradient, variable)
            for gradient, variable in zip(gradients, variables)
            if gradient is not None
        ]
        if filtered_pairs:
            optimizer.apply_gradients(filtered_pairs)

    def train_step(self, data):
        """Train one step with a weighted joint loss and optional discriminator update."""
        if isinstance(data, tuple):
            if len(data) == 2:
                x, y = data
            elif len(data) == 3:
                x, y, _sample_weight = data
            else:
                raise ValueError("Expected (x, y) or (x, y, sample_weight).")
        else:
            raise ValueError("Expected data as (x, y) tuple.")

        disc_vars = self._discriminator_variables()
        disc_var_ids = {id(variable) for variable in disc_vars}
        main_variables = [
            variable for variable in self.trainable_variables if id(variable) not in disc_var_ids
        ]

        discriminator_optimizer = None
        if self.update_discriminator and disc_vars:
            # Built eagerly in compile() -- never construct an optimizer here.
            # train_step gets traced into a tf.function graph the first time
            # model.fit() runs it, and cloning an optimizer via
            # self.optimizer.__class__.from_config(self.optimizer.get_config())
            # reads the learning rate with .numpy(), which raises
            # NotImplementedError during tracing.
            discriminator_optimizer = getattr(self, "_discriminator_optimizer", None)
            if discriminator_optimizer is None:
                raise RuntimeError(
                    "update_discriminator=True but no discriminator optimizer "
                    "was built. This should have happened in compile(); did "
                    "you forget to call model.compile(...)?"
                )

        with tf.GradientTape() as tape:
            total_loss, reconstruction_loss, vc_loss, outputs = self._compute_weighted_losses(
                x=x,
                y=y,
                training=True,
            )

        main_gradients = tape.gradient(total_loss, main_variables)
        self._apply_gradients(self.optimizer, main_gradients, main_variables)

        if discriminator_optimizer is not None:
            with tf.GradientTape() as disc_tape:
                disc_loss = self.variational_classifier.discriminator_loss(
                    tf.stop_gradient(outputs["pooled_latent"]),
                    tf.cast(tf.reshape(y, [-1]), tf.int32),
                )
            disc_gradients = disc_tape.gradient(disc_loss, disc_vars)
            self._apply_gradients(discriminator_optimizer, disc_gradients, disc_vars)

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vc_loss_tracker.update_state(vc_loss)
        self.accuracy_tracker.update_state(y, outputs["logits"])

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vc_loss": self.vc_loss_tracker.result(),
            "accuracy": self.accuracy_tracker.result(),
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

        total_loss, reconstruction_loss, vc_loss, outputs = self._compute_weighted_losses(
            x=x,
            y=y,
            training=False,
        )

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vc_loss_tracker.update_state(vc_loss)
        self.accuracy_tracker.update_state(y, outputs["logits"])

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vc_loss": self.vc_loss_tracker.result(),
            "accuracy": self.accuracy_tracker.result(),
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
                "update_discriminator": self.update_discriminator,
            }
        )
        return config