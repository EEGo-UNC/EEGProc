"""Joint variational autoencoder, recurrent encoder, and classifier.

The shared CNN encoder first produces a deterministic feature sequence. Two
learned projections parameterize a diagonal Gaussian posterior at every latent
timestep::

    encoder_output -> z_mean, z_log_var
    z = z_mean + exp(0.5 * z_log_var) * epsilon

The sampled latent sequence is used by both downstream branches during
training:

1. ``z -> decoder -> reconstruction``
2. ``z -> classification_model -> variational_classifier``

At evaluation/inference time, callers can independently choose whether to use
``z_mean`` or sample from ``q(z|x)``. Latent sampling is deliberately separated
from Keras' ``training`` flag so Monte Carlo prediction can keep dropout disabled
and BatchNorm in inference mode.

Training objective
------------------
The main gradient step minimizes::

    autoencoder_loss = reconstruction_loss
                     + vae_beta * KL(q(z|x) || N(0, I))

    total_loss = ae_loss_weight * autoencoder_loss
               + vc_loss_weight * variational_classifier_loss

When ``update_discriminator=True``, the variational classifier's optional
discriminator receives a separate gradient step using the recurrent
classification embedding.
"""

from __future__ import annotations

import tensorflow as tf

from ..unsupervised.VariationalAutoencoderLoss import VariationalAutoencoderLoss


class JointAutoencoderVariationalClassifierV2(tf.keras.Model):
    """Combine a sequence VAE with recurrent variational classification.

    Parameters
    ----------
    encoder : tf.keras.Model
        Sequence encoder mapping raw EEG shaped
        ``(batch, timesteps, n_features)`` to a deterministic feature sequence
        shaped ``(batch, latent_timesteps, encoder_features)``.
    decoder : tf.keras.Model
        Decoder mapping the sampled latent sequence back to raw EEG.
    classification_model : tf.keras.Model
        Recurrent feature extractor mapping the sampled latent sequence to one
        classification embedding per sample.
    variational_classifier : tf.keras.layers.Layer
        Final variational classification head accepting the recurrent
        classification embedding and returning class logits.
    latent_features : int
        Width of the learned Gaussian latent sequence. In the current builder,
        this matches the CNN encoder output width so the existing decoder can
        consume the sampled latent sequence without a shape adapter.
    ae_loss_weight : float, default=0.5
        Weight assigned to the complete VAE loss.
    vc_loss_weight : float, default=0.5
        Weight assigned to the variational-classifier loss.
    vae_beta : float, default=1.0
        Non-negative multiplier on ``KL(q(z|x) || N(0, I))``.
    reconstruction_loss_fn : VariationalAutoencoderLoss | None
        Optional preconfigured VAE loss object. When omitted, one is created
        with mean-squared reconstruction and ``beta=vae_beta``.
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
        latent_features: int,
        ae_loss_weight: float = 0.5,
        vc_loss_weight: float = 0.5,
        vae_beta: float = 1.0,
        reconstruction_loss_fn: VariationalAutoencoderLoss | None = None,
        vc_alpha: float = 1.0,
        vc_beta: float = 1.0,
        vc_gamma: float = 0.0,
        vc_lambda: float = 1.0,
        update_discriminator: bool = False,
        name: str = "joint_autoencoder_variational_classifier_v2",
        **kwargs,
    ) -> None:
        # ``latent_features`` and ``vae_beta`` are consumed explicitly here.
        # They must not leak through **kwargs into tf.keras.Model.__init__.
        super().__init__(name=name, **kwargs)

        if encoder is None:
            raise ValueError("encoder must be provided.")
        if decoder is None:
            raise ValueError("decoder must be provided.")
        if classification_model is None:
            raise ValueError("classification_model must be provided.")
        if variational_classifier is None:
            raise ValueError("variational_classifier must be provided.")
        if latent_features < 1:
            raise ValueError("latent_features must be at least 1.")
        if ae_loss_weight < 0.0 or vc_loss_weight < 0.0:
            raise ValueError("Loss weights must be non-negative.")
        if ae_loss_weight == 0.0 and vc_loss_weight == 0.0:
            raise ValueError("At least one loss weight must be greater than 0.")
        if vae_beta < 0.0:
            raise ValueError("vae_beta must be non-negative.")

        self.encoder = encoder
        self.decoder = decoder
        self.classification_model = classification_model
        self.variational_classifier = variational_classifier

        self.latent_features = int(latent_features)
        self.ae_loss_weight = float(ae_loss_weight)
        self.vc_loss_weight = float(vc_loss_weight)
        self.vae_beta = float(vae_beta)

        # These two trainable projections turn the deterministic CNN output
        # into a true diagonal-Gaussian posterior q(z|x).
        self.z_mean_projection = tf.keras.layers.Dense(
            self.latent_features,
            name="z_mean",
        )
        self.z_log_var_projection = tf.keras.layers.Dense(
            self.latent_features,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="z_log_var",
        )

        self.reconstruction_loss_fn = (
            reconstruction_loss_fn
            if reconstruction_loss_fn is not None
            else VariationalAutoencoderLoss(
                reconstruction="mse",
                beta=self.vae_beta,
                feature_reduction="mean",
            )
        )

        self.vc_alpha = float(vc_alpha)
        self.vc_beta = float(vc_beta)
        self.vc_gamma = float(vc_gamma)
        self.vc_lambda = float(vc_lambda)
        self.update_discriminator = bool(update_discriminator)

        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.autoencoder_loss_tracker = tf.keras.metrics.Mean(
            name="autoencoder_loss"
        )
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.weighted_kl_loss_tracker = tf.keras.metrics.Mean(
            name="weighted_kl_loss"
        )
        self.vc_loss_tracker = tf.keras.metrics.Mean(name="vc_loss")
        self.accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(
            name="accuracy"
        )

    @property
    def metrics(self) -> list[tf.keras.metrics.Metric]:
        """Metrics reset automatically by Keras each epoch/evaluation."""
        return [
            self.total_loss_tracker,
            self.autoencoder_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.weighted_kl_loss_tracker,
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
            self._discriminator_optimizer = self.optimizer.__class__.from_config(
                self.optimizer.get_config()
            )

    @staticmethod
    def _reparameterize(
        z_mean: tf.Tensor,
        z_log_var: tf.Tensor,
    ) -> tf.Tensor:
        """Draw a differentiable sample from a diagonal Gaussian posterior."""
        epsilon = tf.random.normal(
            shape=tf.shape(z_mean),
            mean=0.0,
            stddev=1.0,
            dtype=z_mean.dtype,
        )
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def _latent_for_mode(
        self,
        z_mean: tf.Tensor,
        z_log_var: tf.Tensor,
        sample_latent: bool | tf.Tensor,
    ) -> tf.Tensor:
        """Sample from ``q(z|x)`` when requested, otherwise return ``z_mean``."""
        if tf.is_tensor(sample_latent):
            return tf.cond(
                tf.cast(sample_latent, tf.bool),
                lambda: self._reparameterize(z_mean, z_log_var),
                lambda: z_mean,
            )

        return (
            self._reparameterize(z_mean, z_log_var)
            if bool(sample_latent)
            else z_mean
        )

    def _posterior_parameters(
        self,
        inputs: tf.Tensor,
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Return encoder output and diagonal-Gaussian posterior parameters."""
        encoder_output = self.encoder(inputs, training=training)

        if encoder_output.shape.rank != 3:
            raise ValueError(
                "encoder must return a rank-3 sequence shaped "
                "(batch, latent_timesteps, encoder_features); got "
                f"{encoder_output.shape}."
            )

        z_mean = self.z_mean_projection(encoder_output)
        z_log_var = self.z_log_var_projection(encoder_output)
        return encoder_output, z_mean, z_log_var

    def _classify_latent(
        self,
        latent_sequence: tf.Tensor,
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Classify one batch of latent sequences."""
        classification_latent = self.classification_model(
            latent_sequence,
            training=training,
        )
        logits = self.variational_classifier(
            classification_latent,
            training=training,
        )
        return classification_latent, logits

    def call(
        self,
        inputs,
        training: bool = False,
        sample_latent: bool | None = None,
    ) -> dict[str, tf.Tensor]:
        """Run the complete model with independent latent-sampling control.

        ``sample_latent=None`` preserves the original behavior: training samples
        from the posterior and inference uses the posterior mean. Passing
        ``sample_latent=True`` with ``training=False`` samples the VAE posterior
        while keeping dropout disabled and BatchNorm in inference mode.
        """
        if sample_latent is None:
            sample_latent = training

        encoder_output, z_mean, z_log_var = self._posterior_parameters(
            inputs,
            training=training,
        )
        latent_sequence = self._latent_for_mode(
            z_mean=z_mean,
            z_log_var=z_log_var,
            sample_latent=sample_latent,
        )

        classification_latent, logits = self._classify_latent(
            latent_sequence,
            training=training,
        )
        reconstruction = self.decoder(
            latent_sequence,
            training=training,
        )

        return {
            "encoder_output": encoder_output,
            "z_mean": z_mean,
            "z_log_var": z_log_var,
            "latent_sequence": latent_sequence,
            "classification_latent": classification_latent,
            "logits": logits,
            "reconstruction": reconstruction,
        }

    def predict_mc_probabilities(
        self,
        inputs,
        n_samples: int = 30,
        seed: int | tuple[int, int] | None = None,
    ) -> dict[str, tf.Tensor]:
        """Average classifier probabilities across posterior latent samples.

        The CNN encoder is evaluated once. ``n_samples`` latent sequences are
        then drawn from ``q(z|x)`` and classified in one vectorized recurrent
        pass. Dropout remains disabled and BatchNorm uses moving statistics.

        ``n_samples=1`` performs one random posterior draw. Use ``seed`` for a
        reproducible draw or Monte Carlo estimate.
        """
        if n_samples < 1:
            raise ValueError("n_samples must be at least 1.")

        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        _encoder_output, z_mean, z_log_var = self._posterior_parameters(
            inputs,
            training=False,
        )

        epsilon_shape = tf.concat(
            [tf.constant([int(n_samples)], dtype=tf.int32), tf.shape(z_mean)],
            axis=0,
        )
        if seed is None:
            epsilon = tf.random.normal(
                shape=epsilon_shape,
                dtype=z_mean.dtype,
            )
        else:
            if isinstance(seed, int):
                stateless_seed = tf.constant([seed, 0], dtype=tf.int32)
            else:
                if len(seed) != 2:
                    raise ValueError("seed tuple must contain exactly two integers.")
                stateless_seed = tf.constant(seed, dtype=tf.int32)
            epsilon = tf.random.stateless_normal(
                shape=epsilon_shape,
                seed=stateless_seed,
                dtype=z_mean.dtype,
            )

        z_std = tf.exp(0.5 * z_log_var)
        z_samples = z_mean[tf.newaxis, ...] + z_std[tf.newaxis, ...] * epsilon

        sample_shape = tf.shape(z_samples)
        z_flat = tf.reshape(
            z_samples,
            [
                sample_shape[0] * sample_shape[1],
                sample_shape[2],
                sample_shape[3],
            ],
        )
        _classification_latent, logits_flat = self._classify_latent(
            z_flat,
            training=False,
        )
        probabilities_flat = tf.nn.softmax(logits_flat, axis=-1)
        n_classes = tf.shape(probabilities_flat)[-1]
        probability_samples = tf.reshape(
            probabilities_flat,
            [sample_shape[0], sample_shape[1], n_classes],
        )

        return {
            "mean_probabilities": tf.reduce_mean(probability_samples, axis=0),
            "probability_samples": probability_samples,
            "z_mean": z_mean,
            "z_log_var": z_log_var,
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
    ) -> tuple[
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        dict[str, tf.Tensor],
    ]:
        outputs = self(x, training=training)

        # Flatten the latent timestep and feature axes only for the analytic KL
        # calculation. The decoder and BiLSTM still consume the 3D sequence.
        z_mean_flat = tf.reshape(
            outputs["z_mean"],
            [tf.shape(outputs["z_mean"])[0], -1],
        )
        z_log_var_flat = tf.reshape(
            outputs["z_log_var"],
            [tf.shape(outputs["z_log_var"])[0], -1],
        )

        vae_losses = self.reconstruction_loss_fn(
            x_true=x,
            x_pred=outputs["reconstruction"],
            z_mean=z_mean_flat,
            z_log_var=z_log_var_flat,
        )
        reconstruction_loss = vae_losses["reconstruction_loss"]
        autoencoder_loss = vae_losses["total_loss"]

        # Track the unweighted Gaussian KL explicitly so VAE-beta and encoder
        # architecture searches can be diagnosed independently of the weighted
        # total autoencoder loss. This does not alter the optimized objective.
        kl_per_sample = -0.5 * tf.reduce_sum(
            1.0
            + z_log_var_flat
            - tf.square(z_mean_flat)
            - tf.exp(z_log_var_flat),
            axis=1,
        )
        kl_loss = tf.reduce_mean(kl_per_sample)
        weighted_kl_loss = tf.cast(self.vae_beta, kl_loss.dtype) * kl_loss

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
            self.ae_loss_weight * autoencoder_loss
            + self.vc_loss_weight * vc_loss
        )
        return (
            total_loss,
            autoencoder_loss,
            reconstruction_loss,
            kl_loss,
            weighted_kl_loss,
            vc_loss,
            outputs,
        )

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
        """Apply only gradients connected to the current loss."""
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
                autoencoder_loss,
                reconstruction_loss,
                kl_loss,
                weighted_kl_loss,
                vc_loss,
                outputs,
            ) = self._compute_weighted_losses(
                x=x,
                y=y_flat,
                training=True,
            )

        # The first forward call creates all lazily built variables, including
        # the z_mean and z_log_var projections, before variables are collected.
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
                        tf.stop_gradient(outputs["classification_latent"]),
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
        self.autoencoder_loss_tracker.update_state(autoencoder_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.weighted_kl_loss_tracker.update_state(weighted_kl_loss)
        self.vc_loss_tracker.update_state(vc_loss)
        self.accuracy_tracker.update_state(y_flat, outputs["logits"])

        return {
            "loss": self.total_loss_tracker.result(),
            "autoencoder_loss": self.autoencoder_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "weighted_kl_loss": self.weighted_kl_loss_tracker.result(),
            "vc_loss": self.vc_loss_tracker.result(),
            "accuracy": self.accuracy_tracker.result(),
        }

    def test_step(self, data) -> dict[str, tf.Tensor]:
        """Evaluate with the posterior mean and without updating weights."""
        x, y = self._unpack_data(data)
        y_flat = self._flatten_labels(y)

        (
            total_loss,
            autoencoder_loss,
            reconstruction_loss,
            kl_loss,
            weighted_kl_loss,
            vc_loss,
            outputs,
        ) = self._compute_weighted_losses(
            x=x,
            y=y_flat,
            training=False,
        )

        self.total_loss_tracker.update_state(total_loss)
        self.autoencoder_loss_tracker.update_state(autoencoder_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.weighted_kl_loss_tracker.update_state(weighted_kl_loss)
        self.vc_loss_tracker.update_state(vc_loss)
        self.accuracy_tracker.update_state(y_flat, outputs["logits"])

        return {
            "loss": self.total_loss_tracker.result(),
            "autoencoder_loss": self.autoencoder_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "weighted_kl_loss": self.weighted_kl_loss_tracker.result(),
            "vc_loss": self.vc_loss_tracker.result(),
            "accuracy": self.accuracy_tracker.result(),
        }

    def get_config(self) -> dict:
        """Return scalar settings used by this subclassed model."""
        config = super().get_config()
        config.update(
            {
                "latent_features": self.latent_features,
                "ae_loss_weight": self.ae_loss_weight,
                "vc_loss_weight": self.vc_loss_weight,
                "vae_beta": self.vae_beta,
                "vc_alpha": self.vc_alpha,
                "vc_beta": self.vc_beta,
                "vc_gamma": self.vc_gamma,
                "vc_lambda": self.vc_lambda,
                "update_discriminator": self.update_discriminator,
            }
        )
        return config
