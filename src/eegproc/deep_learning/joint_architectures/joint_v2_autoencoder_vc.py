"""Joint variational autoencoder, recurrent encoder, and classifier.

The shared CNN encoder first produces a deterministic feature sequence. Two
learned projections parameterize a diagonal Gaussian posterior at every latent
timestep::

    encoder_output -> z_mean, z_log_var
    z = z_mean + exp(0.5 * z_log_var) * epsilon

Each training sample is a complete trial shaped
``(n_windows, window_timesteps, n_features)``. The trial and window dimensions
are flattened only while the shared encoder and decoder process individual
windows. The two downstream branches are:

1. ``window z -> decoder -> one reconstruction per window``
2. ``window z -> window classification_model -> trial_classification_model
   -> one variational-classifier prediction per trial``

At evaluation/inference time, callers can independently choose whether to use
``z_mean`` or sample from ``q(z|x)``. Latent sampling is deliberately separated
from Keras' ``training`` flag so Monte Carlo prediction can keep dropout disabled
and BatchNorm in inference mode.

Training objective
------------------
The main gradient step minimizes::

    autoencoder_loss = mean_reconstruction_loss
                     + vae_beta * mean_latent_coordinate_kl

    base_total_loss = ae_loss_weight * autoencoder_loss
                    + vc_loss_weight * variational_classifier_loss

    total_loss = base_total_loss + keras_layer_regularization_losses

The default autoencoder objective is dimension-normalized (mean/mean), not the
standard summed ELBO. This keeps it on a practical scale when combined with the
classifier objective. Graph adjacency and other Keras ``add_loss`` penalties
are explicitly included in the custom training objective.

When ``update_discriminator=True``, the variational classifier's optional
discriminator receives a separate gradient step using the recurrent
classification embedding.
"""

from __future__ import annotations

import importlib

import tensorflow as tf

from ..unsupervised.VariationalAutoencoderLoss import VariationalAutoencoderLoss


def _serialize_keras_component(component):
    """Serialize a nested Keras object for full-model saving."""
    return tf.keras.utils.serialize_keras_object(component)


def _deserialize_keras_component(config):
    """Deserialize a nested Keras object, with an import-based fallback.

    The fallback supports EEGProc custom classes that expose ``get_config`` /
    ``from_config`` but have not yet been added to Keras' global registry.
    """
    if config is None or isinstance(config, (tf.keras.Model, tf.keras.layers.Layer)):
        return config

    try:
        return tf.keras.utils.deserialize_keras_object(config)
    except (TypeError, ValueError, ImportError) as original_error:
        if not isinstance(config, dict):
            raise
        module_name = config.get("module")
        class_name = config.get("class_name")
        object_config = config.get("config")
        if not module_name or not class_name or not isinstance(object_config, dict):
            raise original_error

        module = importlib.import_module(module_name)
        object_class = getattr(module, class_name)
        if hasattr(object_class, "from_config"):
            return object_class.from_config(object_config)
        return object_class(**object_config)

@tf.keras.utils.register_keras_serializable(package="EEGProc")
class DecoderReconstructionAccuracy(tf.keras.metrics.Metric):
    """Global reconstruction R² reported as ``decoder_accuracy``.

    The decoder predicts continuous EEG values, so categorical accuracy is not
    defined. This metric accumulates sufficient statistics across all batches
    and reports the coefficient of determination::

        1 - sum((x - x_hat)^2) / sum((x - mean(x))^2)

    A value of 1.0 is a perfect reconstruction, 0.0 matches predicting the
    global target mean, and negative values indicate a worse reconstruction.
    """

    def __init__(
        self,
        name: str = "decoder_accuracy",
        dtype=None,
        **kwargs,
    ) -> None:
        super().__init__(name=name, dtype=dtype, **kwargs)
        metric_dtype = self.dtype or tf.keras.backend.floatx()
        self.squared_error_sum = self.add_weight(
            name="squared_error_sum",
            initializer="zeros",
            dtype=metric_dtype,
        )
        self.target_sum = self.add_weight(
            name="target_sum",
            initializer="zeros",
            dtype=metric_dtype,
        )
        self.target_squared_sum = self.add_weight(
            name="target_squared_sum",
            initializer="zeros",
            dtype=metric_dtype,
        )
        self.target_count = self.add_weight(
            name="target_count",
            initializer="zeros",
            dtype=metric_dtype,
        )

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        del sample_weight  # Reconstruction loss is currently unweighted too.
        y_true = tf.cast(y_true, self.dtype)
        y_pred = tf.cast(y_pred, self.dtype)
        error = y_true - y_pred

        self.squared_error_sum.assign_add(tf.reduce_sum(tf.square(error)))
        self.target_sum.assign_add(tf.reduce_sum(y_true))
        self.target_squared_sum.assign_add(tf.reduce_sum(tf.square(y_true)))
        self.target_count.assign_add(tf.cast(tf.size(y_true), self.dtype))

    def result(self):
        target_total_sum_of_squares = (
            self.target_squared_sum
            - tf.math.divide_no_nan(
                tf.square(self.target_sum),
                self.target_count,
            )
        )
        epsilon = tf.cast(tf.keras.backend.epsilon(), self.dtype)
        ordinary_r2 = 1.0 - tf.math.divide_no_nan(
            self.squared_error_sum,
            target_total_sum_of_squares,
        )
        perfect_constant_reconstruction = tf.cast(
            self.squared_error_sum <= epsilon,
            self.dtype,
        )
        return tf.where(
            target_total_sum_of_squares > epsilon,
            ordinary_r2,
            perfect_constant_reconstruction,
        )

    def reset_state(self) -> None:
        for variable in self.variables:
            variable.assign(tf.zeros_like(variable))


@tf.keras.utils.register_keras_serializable(package="EEGProc")
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
        Window-level recurrent feature extractor mapping each sampled latent
        window sequence to one window embedding.
    trial_classification_model : tf.keras.Model
        Trial-level recurrent feature extractor mapping the ordered sequence of
        window embeddings to one embedding per trial.
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
        with mean-squared reconstruction, mean reconstruction reduction, mean
        KL-coordinate reduction, and ``beta=vae_beta``.
    z_log_var_clip_min, z_log_var_clip_max : float
        Bounds applied to the posterior log variance before sampling and loss
        computation to prevent exponential overflow.
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
        trial_classification_model: tf.keras.Model,
        variational_classifier: tf.keras.layers.Layer,
        latent_features: int,
        ae_loss_weight: float = 0.5,
        vc_loss_weight: float = 0.5,
        vae_beta: float = 1.0,
        reconstruction_loss_fn: VariationalAutoencoderLoss | None = None,
        z_log_var_clip_min: float = -20.0,
        z_log_var_clip_max: float = 20.0,
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
        if trial_classification_model is None:
            raise ValueError("trial_classification_model must be provided.")
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
        if z_log_var_clip_min >= z_log_var_clip_max:
            raise ValueError(
                "z_log_var_clip_min must be smaller than "
                "z_log_var_clip_max."
            )

        self.encoder = encoder
        self.decoder = decoder
        self.classification_model = classification_model
        self.trial_classification_model = trial_classification_model
        self.variational_classifier = variational_classifier

        self.latent_features = int(latent_features)
        self.ae_loss_weight = float(ae_loss_weight)
        self.vc_loss_weight = float(vc_loss_weight)
        self.vae_beta = float(vae_beta)
        self.z_log_var_clip_min = float(z_log_var_clip_min)
        self.z_log_var_clip_max = float(z_log_var_clip_max)

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
                kl_reduction="mean",
                log_var_clip_min=self.z_log_var_clip_min,
                log_var_clip_max=self.z_log_var_clip_max,
            )
        )

        self.vc_alpha = float(vc_alpha)
        self.vc_beta = float(vc_beta)
        self.vc_gamma = float(vc_gamma)
        self.vc_lambda = float(vc_lambda)
        self.update_discriminator = bool(update_discriminator)

        if self.vc_gamma > 0.0 and not self.update_discriminator:
            raise ValueError(
                "vc_gamma is positive, but update_discriminator=False. "
                "A discriminator KL term requires a trained discriminator."
            )

        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.base_total_loss_tracker = tf.keras.metrics.Mean(
            name="base_total_loss"
        )
        self.regularization_loss_tracker = tf.keras.metrics.Mean(
            name="regularization_loss"
        )
        self.autoencoder_loss_tracker = tf.keras.metrics.Mean(name="autoencoder_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.weighted_kl_loss_tracker = tf.keras.metrics.Mean(name="weighted_kl_loss")
        self.decoder_accuracy_tracker = DecoderReconstructionAccuracy(
            name="decoder_accuracy"
        )

        self.vc_loss_tracker = tf.keras.metrics.Mean(name="vc_loss")
        self.vc_cross_entropy_tracker = tf.keras.metrics.Mean(
            name="vc_cross_entropy"
        )
        self.weighted_vc_cross_entropy_tracker = tf.keras.metrics.Mean(
            name="weighted_vc_cross_entropy"
        )
        self.vc_latent_kl_tracker = tf.keras.metrics.Mean(name="vc_latent_kl")
        self.weighted_vc_latent_kl_tracker = tf.keras.metrics.Mean(
            name="weighted_vc_latent_kl"
        )
        self.vc_discriminator_kl_tracker = tf.keras.metrics.Mean(
            name="vc_discriminator_kl"
        )
        self.weighted_vc_discriminator_kl_tracker = tf.keras.metrics.Mean(
            name="weighted_vc_discriminator_kl"
        )
        self.vc_class_prior_kl_tracker = tf.keras.metrics.Mean(
            name="vc_class_prior_kl"
        )
        self.weighted_vc_class_prior_kl_tracker = tf.keras.metrics.Mean(
            name="weighted_vc_class_prior_kl"
        )
        self.vc_discriminator_loss_tracker = tf.keras.metrics.Mean(
            name="vc_discriminator_loss"
        )

        self.accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(
            name="accuracy"
        )
        self.true_class_fraction_trackers = [
            tf.keras.metrics.Mean(name=f"true_class_{class_index}_fraction")
            for class_index in range(self.variational_classifier.n_classes)
        ]
        self.predicted_class_fraction_trackers = [
            tf.keras.metrics.Mean(name=f"predicted_class_{class_index}_fraction")
            for class_index in range(self.variational_classifier.n_classes)
        ]

    @property
    def metrics(self) -> list[tf.keras.metrics.Metric]:
        """Metrics reset automatically by Keras each epoch/evaluation."""
        metrics: list[tf.keras.metrics.Metric] = [
            self.total_loss_tracker,
            self.base_total_loss_tracker,
            self.regularization_loss_tracker,
        ]
        if self.ae_loss_weight > 0.0:
            metrics.extend(
                [
                    self.autoencoder_loss_tracker,
                    self.reconstruction_loss_tracker,
                    self.kl_loss_tracker,
                    self.weighted_kl_loss_tracker,
                    self.decoder_accuracy_tracker,
                ]
            )
        metrics.extend(
            [
                self.vc_loss_tracker,
                self.vc_cross_entropy_tracker,
                self.weighted_vc_cross_entropy_tracker,
                self.vc_latent_kl_tracker,
                self.weighted_vc_latent_kl_tracker,
                self.vc_discriminator_kl_tracker,
                self.weighted_vc_discriminator_kl_tracker,
                self.vc_class_prior_kl_tracker,
                self.weighted_vc_class_prior_kl_tracker,
                self.vc_discriminator_loss_tracker,
                self.accuracy_tracker,
                *self.true_class_fraction_trackers,
                *self.predicted_class_fraction_trackers,
            ]
        )
        return metrics

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
            self._reparameterize(z_mean, z_log_var) if bool(sample_latent) else z_mean
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
        raw_z_log_var = self.z_log_var_projection(encoder_output)

        tf.debugging.assert_equal(
            tf.shape(z_mean),
            tf.shape(raw_z_log_var),
            message="z_mean and z_log_var projections must have identical shapes.",
        )
        tf.debugging.assert_all_finite(
            z_mean,
            "z_mean contains NaN or Inf values.",
        )
        tf.debugging.assert_all_finite(
            raw_z_log_var,
            "Unclipped z_log_var contains NaN or Inf values.",
        )

        z_log_var = tf.clip_by_value(
            raw_z_log_var,
            tf.cast(self.z_log_var_clip_min, raw_z_log_var.dtype),
            tf.cast(self.z_log_var_clip_max, raw_z_log_var.dtype),
        )
        return encoder_output, z_mean, z_log_var

    @staticmethod
    def _flatten_trial_windows(
        inputs: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Flatten ``(trial, window)`` while preserving both dimensions."""
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        if inputs.shape.rank != 4:
            raise ValueError(
                "Trial-level input must have rank 4 and shape "
                "(batch_trials, n_windows, window_timesteps, n_features); "
                f"got {inputs.shape}."
            )

        input_shape = tf.shape(inputs)
        batch_trials = input_shape[0]
        n_windows = input_shape[1]
        flat_windows = tf.reshape(
            inputs,
            [
                batch_trials * n_windows,
                input_shape[2],
                input_shape[3],
            ],
        )
        return flat_windows, batch_trials, n_windows

    @staticmethod
    def _restore_trial_window_axes(
        flat_tensor: tf.Tensor,
        batch_trials: tf.Tensor,
        n_windows: tf.Tensor,
    ) -> tf.Tensor:
        """Restore leading ``(batch_trials, n_windows)`` dimensions."""
        flat_shape = tf.shape(flat_tensor)
        target_shape = tf.concat(
            [
                tf.reshape(batch_trials, [1]),
                tf.reshape(n_windows, [1]),
                flat_shape[1:],
            ],
            axis=0,
        )
        return tf.reshape(flat_tensor, target_shape)

    def _classify_flat_latents(
        self,
        latent_windows_flat: tf.Tensor,
        batch_trials: tf.Tensor,
        n_windows: tf.Tensor,
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Return window embeddings, one trial embedding, and trial logits."""
        window_embeddings_flat = self.classification_model(
            latent_windows_flat,
            training=training,
        )
        if window_embeddings_flat.shape.rank != 2:
            raise ValueError(
                "classification_model must return one rank-2 embedding per "
                f"window; got {window_embeddings_flat.shape}."
            )

        window_embeddings = tf.reshape(
            window_embeddings_flat,
            [batch_trials, n_windows, tf.shape(window_embeddings_flat)[-1]],
        )
        trial_classification_latent = self.trial_classification_model(
            window_embeddings,
            training=training,
        )
        if trial_classification_latent.shape.rank != 2:
            raise ValueError(
                "trial_classification_model must return one rank-2 embedding "
                f"per trial; got {trial_classification_latent.shape}."
            )

        logits = self.variational_classifier(
            trial_classification_latent,
            training=training,
        )
        return window_embeddings, trial_classification_latent, logits

    def call(
        self,
        inputs,
        training: bool = False,
        sample_latent: bool | None = None,
        include_reconstruction: bool | None = None,
    ) -> dict[str, tf.Tensor]:
        """Run window reconstruction and trial classification together.

        Inputs are complete trials shaped ``(B, W, T, F)``. The encoder and
        decoder operate on ``B * W`` independent windows, while the classifier
        receives the ordered ``W`` window embeddings and emits one prediction
        per trial.
        """
        if sample_latent is None:
            sample_latent = training
        if include_reconstruction is None:
            include_reconstruction = self.ae_loss_weight > 0.0

        flat_inputs, batch_trials, n_windows = self._flatten_trial_windows(inputs)
        encoder_output_flat, z_mean_flat, z_log_var_flat = (
            self._posterior_parameters(flat_inputs, training=training)
        )
        latent_windows_flat = self._latent_for_mode(
            z_mean=z_mean_flat,
            z_log_var=z_log_var_flat,
            sample_latent=sample_latent,
        )

        (
            window_classification_latent,
            trial_classification_latent,
            logits,
        ) = self._classify_flat_latents(
            latent_windows_flat,
            batch_trials=batch_trials,
            n_windows=n_windows,
            training=training,
        )

        outputs = {
            "encoder_output": self._restore_trial_window_axes(
                encoder_output_flat, batch_trials, n_windows
            ),
            "z_mean": self._restore_trial_window_axes(
                z_mean_flat, batch_trials, n_windows
            ),
            "z_log_var": self._restore_trial_window_axes(
                z_log_var_flat, batch_trials, n_windows
            ),
            "latent_sequence": self._restore_trial_window_axes(
                latent_windows_flat, batch_trials, n_windows
            ),
            "window_classification_latent": window_classification_latent,
            # Backwards-compatible key now intentionally denotes the trial
            # embedding consumed by the variational classifier.
            "classification_latent": trial_classification_latent,
            "trial_classification_latent": trial_classification_latent,
            "logits": logits,
        }

        if include_reconstruction:
            reconstruction_flat = self.decoder(
                latent_windows_flat,
                training=training,
            )
            outputs["reconstruction"] = self._restore_trial_window_axes(
                reconstruction_flat,
                batch_trials,
                n_windows,
            )

        return outputs

    def predict_mc_probabilities(
        self,
        inputs,
        n_samples: int = 30,
        seed: int | tuple[int, int] | None = None,
    ) -> dict[str, tf.Tensor]:
        """Average trial probabilities across posterior window-latent draws."""
        if n_samples < 1:
            raise ValueError("n_samples must be at least 1.")

        flat_inputs, batch_trials, n_windows = self._flatten_trial_windows(inputs)
        _encoder_output, z_mean_flat, z_log_var_flat = self._posterior_parameters(
            flat_inputs,
            training=False,
        )

        epsilon_shape = tf.concat(
            [tf.constant([int(n_samples)], dtype=tf.int32), tf.shape(z_mean_flat)],
            axis=0,
        )
        if seed is None:
            epsilon = tf.random.normal(shape=epsilon_shape, dtype=z_mean_flat.dtype)
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
                dtype=z_mean_flat.dtype,
            )

        z_std = tf.exp(0.5 * z_log_var_flat)
        z_samples = (
            z_mean_flat[tf.newaxis, ...]
            + z_std[tf.newaxis, ...] * epsilon
        )
        sample_shape = tf.shape(z_samples)
        z_samples_flat = tf.reshape(
            z_samples,
            [
                sample_shape[0] * sample_shape[1],
                sample_shape[2],
                sample_shape[3],
            ],
        )

        _window_embeddings, _trial_latent, logits_flat = (
            self._classify_flat_latents(
                z_samples_flat,
                batch_trials=sample_shape[0] * batch_trials,
                n_windows=n_windows,
                training=False,
            )
        )
        probabilities_flat = tf.nn.softmax(logits_flat, axis=-1)
        n_classes = tf.shape(probabilities_flat)[-1]
        probability_samples = tf.reshape(
            probabilities_flat,
            [sample_shape[0], batch_trials, n_classes],
        )

        return {
            "mean_probabilities": tf.reduce_mean(probability_samples, axis=0),
            "probability_samples": probability_samples,
            "z_mean": self._restore_trial_window_axes(
                z_mean_flat, batch_trials, n_windows
            ),
            "z_log_var": self._restore_trial_window_axes(
                z_log_var_flat, batch_trials, n_windows
            ),
        }

    @staticmethod
    def _unpack_data(data):
        """Return x, y, and optional sample weights from a Keras batch."""
        if not isinstance(data, tuple):
            raise ValueError("Expected data as an (x, y) tuple.")
        if len(data) == 2:
            return data[0], data[1], None
        if len(data) == 3:
            return data[0], data[1], data[2]
        raise ValueError("Expected (x, y) or (x, y, sample_weight).")

    @staticmethod
    def _flatten_labels(y) -> tf.Tensor:
        """Convert sparse labels or one-hot labels into class-id vectors."""
        y_tensor = tf.convert_to_tensor(y)
        if (
            y_tensor.shape.rank == 2
            and y_tensor.shape[-1] is not None
            and y_tensor.shape[-1] > 1
        ):
            return tf.argmax(y_tensor, axis=-1, output_type=tf.int32)
        return tf.cast(tf.reshape(y_tensor, [-1]), tf.int32)

    def _compute_weighted_losses(
        self,
        x,
        y,
        training: bool,
        sample_weight=None,
    ) -> tuple[dict[str, tf.Tensor], dict[str, tf.Tensor]]:
        autoencoder_enabled = self.ae_loss_weight > 0.0
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        outputs = self(
            x,
            training=training,
            include_reconstruction=autoencoder_enabled,
        )

        if autoencoder_enabled:
            if x.shape.rank != 4:
                raise ValueError(
                    "Window-level reconstruction requires trial input shaped "
                    f"(B, W, T, F); got {x.shape}."
                )
            x_shape = tf.shape(x)
            n_window_samples = x_shape[0] * x_shape[1]
            x_windows_flat = tf.reshape(
                x,
                [n_window_samples, x_shape[2], x_shape[3]],
            )
            reconstruction_flat = tf.reshape(
                outputs["reconstruction"],
                [n_window_samples, x_shape[2], x_shape[3]],
            )
            z_mean_flat = tf.reshape(
                outputs["z_mean"],
                [n_window_samples, -1],
            )
            z_log_var_flat = tf.reshape(
                outputs["z_log_var"],
                [n_window_samples, -1],
            )
            vae_losses = self.reconstruction_loss_fn(
                x_true=x_windows_flat,
                x_pred=reconstruction_flat,
                z_mean=z_mean_flat,
                z_log_var=z_log_var_flat,
            )
        else:
            zero = tf.zeros((), dtype=outputs["logits"].dtype)
            vae_losses = {
                "total_loss": zero,
                "reconstruction_loss": zero,
                "kl_loss": zero,
                "weighted_kl_loss": zero,
            }

        y_flat = self._flatten_labels(y)
        tf.debugging.assert_equal(
            tf.shape(y_flat)[0],
            tf.shape(outputs["logits"])[0],
            message=(
                "The classifier must emit one prediction per trial and labels "
                "must contain exactly one class ID per trial."
            ),
        )
        vc_losses = self.variational_classifier.vc_loss_components(
            mh=outputs["trial_classification_latent"],
            y=y_flat,
            alpha=self.vc_alpha,
            beta=self.vc_beta,
            gamma=self.vc_gamma,
            lambda_=self.vc_lambda,
            logits=outputs["logits"],
            sample_weight=sample_weight,
        )

        base_total_loss = (
            self.ae_loss_weight * vae_losses["total_loss"]
            + self.vc_loss_weight * vc_losses["total_loss"]
        )

        if self.losses:
            regularization_loss = tf.add_n(
                [
                    tf.cast(layer_loss, base_total_loss.dtype)
                    for layer_loss in self.losses
                ]
            )
        else:
            regularization_loss = tf.zeros_like(base_total_loss)

        total_loss = base_total_loss + regularization_loss

        losses = {
            "total_loss": total_loss,
            "base_total_loss": base_total_loss,
            "regularization_loss": regularization_loss,
            "autoencoder_loss": vae_losses["total_loss"],
            "reconstruction_loss": vae_losses["reconstruction_loss"],
            "kl_loss": vae_losses["kl_loss"],
            "weighted_kl_loss": vae_losses["weighted_kl_loss"],
            "vc_loss": vc_losses["total_loss"],
            "vc_cross_entropy": vc_losses["cross_entropy"],
            "weighted_vc_cross_entropy": vc_losses[
                "weighted_cross_entropy"
            ],
            "vc_latent_kl": vc_losses["latent_posterior_kl"],
            "weighted_vc_latent_kl": vc_losses[
                "weighted_latent_posterior_kl"
            ],
            "vc_discriminator_kl": vc_losses["discriminator_kl"],
            "weighted_vc_discriminator_kl": vc_losses[
                "weighted_discriminator_kl"
            ],
            "vc_class_prior_kl": vc_losses["class_prior_kl"],
            "weighted_vc_class_prior_kl": vc_losses[
                "weighted_class_prior_kl"
            ],
        }
        return losses, outputs

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

    def _update_trackers(
        self,
        losses: dict[str, tf.Tensor],
        outputs: dict[str, tf.Tensor],
        x_true: tf.Tensor,
        y_flat: tf.Tensor,
        sample_weight,
        discriminator_loss: tf.Tensor,
    ) -> None:
        self.total_loss_tracker.update_state(losses["total_loss"])
        self.base_total_loss_tracker.update_state(losses["base_total_loss"])
        self.regularization_loss_tracker.update_state(
            losses["regularization_loss"]
        )
        if self.ae_loss_weight > 0.0:
            self.autoencoder_loss_tracker.update_state(losses["autoencoder_loss"])
            self.reconstruction_loss_tracker.update_state(
                losses["reconstruction_loss"]
            )
            self.kl_loss_tracker.update_state(losses["kl_loss"])
            self.weighted_kl_loss_tracker.update_state(losses["weighted_kl_loss"])
            self.decoder_accuracy_tracker.update_state(
                x_true,
                outputs["reconstruction"],
            )
        self.vc_loss_tracker.update_state(losses["vc_loss"])
        self.vc_cross_entropy_tracker.update_state(losses["vc_cross_entropy"])
        self.weighted_vc_cross_entropy_tracker.update_state(
            losses["weighted_vc_cross_entropy"]
        )
        self.vc_latent_kl_tracker.update_state(losses["vc_latent_kl"])
        self.weighted_vc_latent_kl_tracker.update_state(
            losses["weighted_vc_latent_kl"]
        )
        self.vc_discriminator_kl_tracker.update_state(
            losses["vc_discriminator_kl"]
        )
        self.weighted_vc_discriminator_kl_tracker.update_state(
            losses["weighted_vc_discriminator_kl"]
        )
        self.vc_class_prior_kl_tracker.update_state(
            losses["vc_class_prior_kl"]
        )
        self.weighted_vc_class_prior_kl_tracker.update_state(
            losses["weighted_vc_class_prior_kl"]
        )
        self.vc_discriminator_loss_tracker.update_state(discriminator_loss)

        self.accuracy_tracker.update_state(
            y_flat,
            outputs["logits"],
            sample_weight=sample_weight,
        )
        predicted_classes = tf.argmax(
            outputs["logits"],
            axis=-1,
            output_type=tf.int32,
        )
        for class_index, tracker in enumerate(
            self.true_class_fraction_trackers
        ):
            tracker.update_state(
                tf.cast(tf.equal(y_flat, class_index), tf.float32)
            )
        for class_index, tracker in enumerate(
            self.predicted_class_fraction_trackers
        ):
            tracker.update_state(
                tf.cast(tf.equal(predicted_classes, class_index), tf.float32)
            )

    def _metric_results(self) -> dict[str, tf.Tensor]:
        return {metric.name: metric.result() for metric in self.metrics}

    def train_step(self, data) -> dict[str, tf.Tensor]:
        """Run one joint update and an optional discriminator-only update."""
        x, y, sample_weight = self._unpack_data(data)
        y_flat = self._flatten_labels(y)

        with tf.GradientTape() as tape:
            losses, outputs = self._compute_weighted_losses(
                x=x,
                y=y_flat,
                training=True,
                sample_weight=sample_weight,
            )

        discriminator_variables = self._discriminator_variables()
        discriminator_variable_ids = {
            id(variable) for variable in discriminator_variables
        }
        main_variables = [
            variable
            for variable in self.trainable_variables
            if id(variable) not in discriminator_variable_ids
        ]
        main_gradients = tape.gradient(losses["total_loss"], main_variables)
        self._apply_gradients(self.optimizer, main_gradients, main_variables)

        discriminator_loss = tf.zeros((), dtype=losses["total_loss"].dtype)
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

        self._update_trackers(
            losses=losses,
            outputs=outputs,
            x_true=x,
            y_flat=y_flat,
            sample_weight=sample_weight,
            discriminator_loss=discriminator_loss,
        )
        return self._metric_results()

    def test_step(self, data) -> dict[str, tf.Tensor]:
        """Evaluate with the posterior mean and without updating weights."""
        x, y, sample_weight = self._unpack_data(data)
        y_flat = self._flatten_labels(y)
        losses, outputs = self._compute_weighted_losses(
            x=x,
            y=y_flat,
            training=False,
            sample_weight=sample_weight,
        )
        discriminator_loss = tf.zeros((), dtype=losses["total_loss"].dtype)
        self._update_trackers(
            losses=losses,
            outputs=outputs,
            x_true=x,
            y_flat=y_flat,
            sample_weight=sample_weight,
            discriminator_loss=discriminator_loss,
        )
        return self._metric_results()

    def predict_step(self, data):
        """Return trial logits without running the reconstruction decoder."""
        if isinstance(data, tuple):
            inputs = data[0]
        else:
            inputs = data
        outputs = self(
            inputs,
            training=False,
            sample_latent=False,
            include_reconstruction=False,
        )
        return outputs["logits"]

    def get_config(self) -> dict:
        """Return a complete serializable configuration for this model."""
        config = super().get_config()
        config.update(
            {
                "encoder": _serialize_keras_component(self.encoder),
                "decoder": _serialize_keras_component(self.decoder),
                "classification_model": _serialize_keras_component(
                    self.classification_model
                ),
                "trial_classification_model": _serialize_keras_component(
                    self.trial_classification_model
                ),
                "variational_classifier": _serialize_keras_component(
                    self.variational_classifier
                ),
                "latent_features": self.latent_features,
                "ae_loss_weight": self.ae_loss_weight,
                "vc_loss_weight": self.vc_loss_weight,
                "vae_beta": self.vae_beta,
                "reconstruction_loss_fn": _serialize_keras_component(
                    self.reconstruction_loss_fn
                ),
                "z_log_var_clip_min": self.z_log_var_clip_min,
                "z_log_var_clip_max": self.z_log_var_clip_max,
                "vc_alpha": self.vc_alpha,
                "vc_beta": self.vc_beta,
                "vc_gamma": self.vc_gamma,
                "vc_lambda": self.vc_lambda,
                "update_discriminator": self.update_discriminator,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict):
        """Reconstruct the joint model and all nested components."""
        config = dict(config)
        for key in (
            "encoder",
            "decoder",
            "classification_model",
            "trial_classification_model",
            "variational_classifier",
            "reconstruction_loss_fn",
        ):
            if key in config:
                config[key] = _deserialize_keras_component(config[key])
        return cls(**config)

