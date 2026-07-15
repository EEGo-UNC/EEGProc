"""Joint autoencoder, recurrent encoder, and variational classifier.

This model shares a CNN encoder between two branches:

1. ``latent_sequence -> decoder -> reconstruction``
2. ``latent_sequence -> classification_model -> variational_classifier``

The classification model is expected to be a recurrent feature extractor,
such as ``BiLSTMClassifier.build_feature_extractor()``, that returns one 2D
sequence-level embedding per sample. The joint model owns the final
``VariationalClassifier`` and all of its losses.

Training objective
------------------
The main gradient step minimizes::

    total_loss = ae_loss_weight * autoencoder_loss
               + vc_loss_weight * variational_classifier_loss

When ``update_discriminator=True``, the variational classifier's optional
discriminator receives a separate gradient step using the same recurrent
classification embedding.
"""

from __future__ import annotations

import tensorflow as tf

from ..unsupervised.VariationalAutoencoderLoss import VariationalAutoencoderLoss


class JointAutoencoderVariationalClassifierV2(tf.keras.Model):
    """Combine a sequence autoencoder with recurrent variational classification.

    Parameters
    ----------
    encoder : tf.keras.Model
        Sequence encoder mapping raw EEG shaped
        ``(batch, timesteps, n_features)`` to a latent sequence shaped
        ``(batch, latent_timesteps, latent_features)``.
    decoder : tf.keras.Model
        Decoder that reconstructs raw EEG from ``latent_sequence``.
    classification_model : tf.keras.Model
        Recurrent feature extractor mapping ``latent_sequence`` to one
        classification embedding per sample, shaped
        ``(batch, classification_features)``. This should not contain its own
        dense/softmax/variational classification head.
    variational_classifier : tf.keras.layers.Layer
        Final variational classification head accepting the recurrent
        classification embedding and returning class logits.
    ae_loss_weight : float, default=0.5
        Weight assigned to the autoencoder loss.
    vc_loss_weight : float, default=0.5
        Weight assigned to the variational-classifier loss.
    reconstruction_loss_fn : VariationalAutoencoderLoss | None
        Autoencoder loss object. Defaults to mean-squared reconstruction with
        the existing VAE-loss interface.
    vc_alpha, vc_beta, vc_gamma, vc_lambda : float
        Coefficients forwarded to ``variational_classifier.vc_loss``.
    update_discriminator : bool, default=False
        Whether to train the variational classifier's discriminator in a
        separate gradient step.
    """

    def __init__(
        self,
        encoder: tf.keras.Model,
        decoder: tf.keras.Model,
        classification_model: tf.keras.Model,
        variational_classifier: tf.keras.layers.Layer,
        ae_loss_weight: float = 0.5,
        vc_loss_weight: float = 0.5,
        reconstruction_loss_fn: VariationalAutoencoderLoss | None = None,
        vc_alpha: float = 1.0,
        vc_beta: float = 1.0,
        vc_gamma: float = 0.0,
        vc_lambda: float = 1.0,
        update_discriminator: bool = False,
        name: str = "joint_autoencoder_variational_classifier_v2",
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)

        if ae_loss_weight < 0.0 or vc_loss_weight < 0.0:
            raise ValueError("Loss weights must be non-negative.")
        if ae_loss_weight == 0.0 and vc_loss_weight == 0.0:
            raise ValueError("At least one loss weight must be greater than 0.")
        if classification_model is None:
            raise ValueError("classification_model must be provided.")

        self.encoder = encoder
        self.decoder = decoder
        self.classification_model = classification_model
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

        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vc_loss_tracker = tf.keras.metrics.Mean(name="vc_loss")
        self.accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(
            name="accuracy"
        )

    @property
    def metrics(self) -> list[tf.keras.metrics.Metric]:
        """Metrics reset automatically by Keras at each epoch/evaluation."""
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vc_loss_tracker,
            self.accuracy_tracker,
        ]

    def compile(
        self,
        optimizer=None,
        discriminator_optimizer=None,
        **kwargs,
    ) -> None:
        """Compile and eagerly prepare the optional discriminator optimizer."""
        kwargs.setdefault("jit_compile", False)
        super().compile(optimizer=optimizer, **kwargs)

        if not self.update_discriminator:
            return

        if discriminator_optimizer is not None:
            self._discriminator_optimizer = discriminator_optimizer
        elif not hasattr(self, "_discriminator_optimizer"):
            if self.optimizer is None:
                raise ValueError(
                    "An optimizer is required when update_discriminator=True."
                )
            self._discriminator_optimizer = (
                self.optimizer.__class__.from_config(
                    self.optimizer.get_config()
                )
            )

    def call(self, inputs, training: bool = False) -> dict[str, tf.Tensor]:
        """Run the shared encoder and both downstream branches."""
        latent_sequence = self.encoder(inputs, training=training)

        classification_latent = self.classification_model(
            latent_sequence,
            training=training,
        )
        logits = self.variational_classifier(
            classification_latent,
            training=training,
        )

        reconstruction = self.decoder(
            latent_sequence,
            training=training,
        )

        return {
            "latent_sequence": latent_sequence,
            "classification_latent": classification_latent,
            "logits": logits,
            "reconstruction": reconstruction,
        }

    @staticmethod
    def _unpack_data(data):
        """Return ``x`` and ``y`` from Keras two- or three-item batches."""
        if not isinstance(data, tuple):
            raise ValueError("Expected data as an (x, y) tuple.")
        if len(data) == 2:
            return data[0], data[1]
        if len(data) == 3:
            x, y, _sample_weight = data
            return x, y
        raise ValueError("Expected (x, y) or (x, y, sample_weight).")

    @staticmethod
    def _flatten_labels(y) -> tf.Tensor:
        """Convert integer labels shaped ``(batch,)`` or ``(batch, 1)`` to 1D."""
        return tf.cast(tf.reshape(y, [-1]), tf.int32)

    def _compute_weighted_losses(
        self,
        x,
        y,
        training: bool,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, dict[str, tf.Tensor]]:
        outputs = self(x, training=training)

        latent_sequence = outputs["latent_sequence"]
        z_mean = tf.reshape(
            latent_sequence,
            [tf.shape(latent_sequence)[0], -1],
        )
        z_log_var = tf.zeros_like(z_mean)

        vae_losses = self.reconstruction_loss_fn(
            x_true=x,
            x_pred=outputs["reconstruction"],
            z_mean=z_mean,
            z_log_var=z_log_var,
        )
        reconstruction_loss = vae_losses["reconstruction_loss"]
        ae_loss = vae_losses["total_loss"]

        y_flat = self._flatten_labels(y)
        vc_loss = self.variational_classifier.vc_loss(
            mh=outputs["classification_latent"],
            y=y_flat,
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

    def _discriminator_variables(self) -> list[tf.Variable]:
        """Return discriminator variables when the VC head defines them."""
        if not hasattr(self.variational_classifier, "disc_w"):
            return []
        return [
            self.variational_classifier.disc_w,
            self.variational_classifier.disc_b,
        ]

    @staticmethod
    def _apply_gradients(optimizer, gradients, variables) -> None:
        """Apply only gradients that are connected to the current loss."""
        gradient_variable_pairs = [
            (gradient, variable)
            for gradient, variable in zip(gradients, variables)
            if gradient is not None
        ]
        if gradient_variable_pairs:
            optimizer.apply_gradients(gradient_variable_pairs)

    def train_step(self, data) -> dict[str, tf.Tensor]:
        """Run one joint update and an optional discriminator-only update."""
        x, y = self._unpack_data(data)
        y_flat = self._flatten_labels(y)

        with tf.GradientTape() as tape:
            (
                total_loss,
                reconstruction_loss,
                vc_loss,
                outputs,
            ) = self._compute_weighted_losses(
                x=x,
                y=y_flat,
                training=True,
            )

        # The first forward call creates all lazily built variables before the
        # trainable-variable list is inspected.
        discriminator_variables = self._discriminator_variables()
        discriminator_variable_ids = {
            id(variable) for variable in discriminator_variables
        }
        main_variables = [
            variable
            for variable in self.trainable_variables
            if id(variable) not in discriminator_variable_ids
        ]

        main_gradients = tape.gradient(total_loss, main_variables)
        self._apply_gradients(
            self.optimizer,
            main_gradients,
            main_variables,
        )

        if self.update_discriminator and discriminator_variables:
            discriminator_optimizer = getattr(
                self,
                "_discriminator_optimizer",
                None,
            )
            if discriminator_optimizer is None:
                raise RuntimeError(
                    "update_discriminator=True but no discriminator optimizer "
                    "was created. Call model.compile(...) first."
                )

            with tf.GradientTape() as discriminator_tape:
                discriminator_loss = (
                    self.variational_classifier.discriminator_loss(
                        tf.stop_gradient(
                            outputs["classification_latent"]
                        ),
                        y_flat,
                    )
                )

            discriminator_gradients = discriminator_tape.gradient(
                discriminator_loss,
                discriminator_variables,
            )
            self._apply_gradients(
                discriminator_optimizer,
                discriminator_gradients,
                discriminator_variables,
            )

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vc_loss_tracker.update_state(vc_loss)
        self.accuracy_tracker.update_state(y_flat, outputs["logits"])

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vc_loss": self.vc_loss_tracker.result(),
            "accuracy": self.accuracy_tracker.result(),
        }

    def test_step(self, data) -> dict[str, tf.Tensor]:
        """Evaluate with the same joint objective, without updating weights."""
        x, y = self._unpack_data(data)
        y_flat = self._flatten_labels(y)

        total_loss, reconstruction_loss, vc_loss, outputs = (
            self._compute_weighted_losses(
                x=x,
                y=y_flat,
                training=False,
            )
        )

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vc_loss_tracker.update_state(vc_loss)
        self.accuracy_tracker.update_state(y_flat, outputs["logits"])

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vc_loss": self.vc_loss_tracker.result(),
            "accuracy": self.accuracy_tracker.result(),
        }

    def get_config(self) -> dict:
        """Return scalar loss settings used by this subclassed model."""
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
