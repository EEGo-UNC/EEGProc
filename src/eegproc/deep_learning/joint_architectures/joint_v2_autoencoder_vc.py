"""Window-level joint variational autoencoder and selectable classifier.

Each sample is one EEG window. The VAE reconstructs that window, a recurrent
feature extractor summarizes its latent temporal sequence, and either a dense
or variational classification head emits one prediction for the same window.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping

import numpy as np
import tensorflow as tf

from ..unsupervised.VariationalAutoencoderLoss import (
    GradientReversal,
    VariationalAutoencoderLoss,
)


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
    """Joint window-level VAE with a dense or variational classifier.

    Every sample is one EEG window shaped ``(timesteps, n_features)``. The
    encoder and decoder operate on that window, while ``classification_model``
    summarizes the sampled latent sequence into one embedding and the
    selected classification head emits one prediction for the same window.

    ``use_class_weight`` controls whether a ``class_weight`` dictionary passed
    by external training utilities is honored. This lets the existing LOSO
    implementation keep calculating fold-local weights while the model can
    explicitly disable them without modifying cross_val.py.
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
        z_log_var_clip_min: float = -20.0,
        z_log_var_clip_max: float = 20.0,
        vc_alpha: float = 1.0,
        vc_beta: float = 1.0,
        vc_gamma: float = 0.0,
        vc_lambda: float = 1.0,
        update_discriminator: bool = False,
        use_class_weight: bool = True,
        use_subject_adversarial: bool = False,
        n_subject_classes: int | None = None,
        subject_adversarial_weight: float = 0.05,
        subject_loss_weight: float = 1.0,
        subject_hidden_units: int = 64,
        subject_dropout: float = 0.0,
        subject_latent_mode: str = "mean",
        subject_mc_samples: int = 5,
        name: str = "joint_autoencoder_variational_classifier_v2",
        **kwargs,
    ) -> None:
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
        if z_log_var_clip_min >= z_log_var_clip_max:
            raise ValueError(
                "z_log_var_clip_min must be smaller than z_log_var_clip_max."
            )
        if subject_adversarial_weight < 0.0:
            raise ValueError("subject_adversarial_weight must be non-negative.")
        if subject_loss_weight < 0.0:
            raise ValueError("subject_loss_weight must be non-negative.")
        if subject_hidden_units < 1:
            raise ValueError("subject_hidden_units must be at least 1.")
        if not 0.0 <= subject_dropout < 1.0:
            raise ValueError("subject_dropout must be in [0, 1).")
        subject_latent_mode = str(subject_latent_mode).lower()
        if subject_latent_mode not in {"mean", "mc"}:
            raise ValueError(
                "subject_latent_mode must be 'mean' or 'mc', "
                f"got {subject_latent_mode!r}."
            )
        if subject_mc_samples < 1:
            raise ValueError("subject_mc_samples must be at least 1.")
        if n_subject_classes is not None and int(n_subject_classes) < 2:
            raise ValueError("n_subject_classes must be at least 2 when supplied.")

        self.encoder = encoder
        self.decoder = decoder
        self.classification_model = classification_model
        self.variational_classifier = variational_classifier

        self.latent_features = int(latent_features)
        self.ae_loss_weight = float(ae_loss_weight)
        self.vc_loss_weight = float(vc_loss_weight)
        self.vae_beta = float(vae_beta)
        self.z_log_var_clip_min = float(z_log_var_clip_min)
        self.z_log_var_clip_max = float(z_log_var_clip_max)
        self.use_class_weight = bool(use_class_weight)
        self.use_subject_adversarial = bool(use_subject_adversarial)
        self.n_subject_classes = (
            None if n_subject_classes is None else int(n_subject_classes)
        )
        self.subject_adversarial_weight = float(subject_adversarial_weight)
        self.subject_loss_weight = float(subject_loss_weight)
        self.subject_hidden_units = int(subject_hidden_units)
        self.subject_dropout = float(subject_dropout)
        self.subject_latent_mode = subject_latent_mode
        self.subject_mc_samples = int(subject_mc_samples)

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

        # The subject head is configured lazily from each fold's fitting
        # subjects. This keeps its output dimension fold-local and prevents
        # validation/test identities from becoming subject classes.
        self.subject_pooling = None
        self.subject_gradient_reversal = None
        self.subject_hidden = None
        self.subject_dropout_layer = None
        self.subject_classifier = None
        if self.use_subject_adversarial and self.n_subject_classes is not None:
            self._configure_subject_head(self.n_subject_classes)

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

        classifier_supports_discriminator = bool(
            getattr(
                self.variational_classifier,
                "supports_discriminator",
                hasattr(self.variational_classifier, "discriminator"),
            )
        )
        if (
            classifier_supports_discriminator
            and self.vc_gamma > 0.0
            and not self.update_discriminator
        ):
            raise ValueError(
                "vc_gamma is positive, but update_discriminator=False. "
                "A discriminator KL term requires a trained discriminator."
            )

        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.base_total_loss_tracker = tf.keras.metrics.Mean(name="base_total_loss")
        self.regularization_loss_tracker = tf.keras.metrics.Mean(
            name="regularization_loss"
        )
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
        self.decoder_accuracy_tracker = DecoderReconstructionAccuracy(
            name="decoder_accuracy"
        )
        self.subject_loss_tracker = tf.keras.metrics.Mean(name="subject_loss")
        self.weighted_subject_loss_tracker = tf.keras.metrics.Mean(
            name="weighted_subject_loss"
        )
        self.subject_accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(
            name="subject_accuracy"
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

    def _configure_subject_head(self, n_subject_classes: int) -> None:
        """Create the fold-local subject discriminator exactly once."""
        n_subject_classes = int(n_subject_classes)
        if n_subject_classes < 2:
            raise ValueError("Subject-adversarial training requires at least 2 subjects.")
        if self.subject_classifier is not None:
            if self.n_subject_classes != n_subject_classes:
                raise ValueError(
                    "The subject head is already configured for "
                    f"{self.n_subject_classes} classes, not {n_subject_classes}."
                )
            return

        self.n_subject_classes = n_subject_classes
        self.subject_pooling = tf.keras.layers.GlobalAveragePooling1D(
            name="subject_latent_pool"
        )
        self.subject_gradient_reversal = GradientReversal(
            adversarial_weight=self.subject_adversarial_weight,
            name="subject_gradient_reversal",
        )
        self.subject_hidden = tf.keras.layers.Dense(
            self.subject_hidden_units,
            activation="relu",
            name="subject_hidden",
        )
        self.subject_dropout_layer = tf.keras.layers.Dropout(
            self.subject_dropout,
            name="subject_dropout",
        )
        self.subject_classifier = tf.keras.layers.Dense(
            n_subject_classes,
            activation=None,
            name="subject_logits",
        )

    def prepare_fit_inputs(self, eeg_inputs, subject_ids):
        """Attach deterministic fold-local subject targets to training inputs."""
        if not self.use_subject_adversarial:
            return eeg_inputs

        eeg_array = np.asarray(eeg_inputs)
        subjects = np.asarray(subject_ids).reshape(-1)
        if len(eeg_array) != len(subjects):
            raise ValueError(
                "EEG samples and subject IDs must align for adversarial training; "
                f"got {len(eeg_array)} and {len(subjects)}."
            )
        unique_subjects = np.sort(np.unique(subjects))
        self._configure_subject_head(len(unique_subjects))
        subject_to_class = {
            value.item() if isinstance(value, np.generic) else value: index
            for index, value in enumerate(unique_subjects)
        }
        fold_local_ids = np.asarray(
            [
                subject_to_class[
                    value.item() if isinstance(value, np.generic) else value
                ]
                for value in subjects
            ],
            dtype=np.int32,
        )
        return {"eeg": eeg_array, "subject_id": fold_local_ids}

    @staticmethod
    def _split_eeg_and_subject_inputs(inputs):
        if isinstance(inputs, Mapping):
            if "eeg" not in inputs:
                raise ValueError("Training input dictionaries must contain an 'eeg' key.")
            return inputs["eeg"], inputs.get("subject_id")
        return inputs, None

    def _subject_head_forward(
        self,
        latent_sequence: tf.Tensor,
        training: bool,
    ) -> tf.Tensor:
        if self.subject_classifier is None:
            raise RuntimeError(
                "Subject head is not configured. Call prepare_fit_inputs(...) "
                "before fitting this fold."
            )
        pooled = self.subject_pooling(latent_sequence)
        reversed_features = self.subject_gradient_reversal(pooled)
        hidden = self.subject_hidden(reversed_features)
        hidden = self.subject_dropout_layer(hidden, training=training)
        return self.subject_classifier(hidden)

    def _subject_logits_from_posterior(
        self,
        z_mean: tf.Tensor,
        z_log_var: tf.Tensor,
        training: bool,
    ) -> tf.Tensor:
        if self.subject_latent_mode == "mean":
            return self._subject_head_forward(z_mean, training=training)

        sample_count = self.subject_mc_samples
        epsilon_shape = tf.concat(
            [tf.constant([sample_count], dtype=tf.int32), tf.shape(z_mean)],
            axis=0,
        )
        epsilon = tf.random.normal(epsilon_shape, dtype=z_mean.dtype)
        latent_samples = (
            z_mean[tf.newaxis, ...]
            + tf.exp(0.5 * z_log_var)[tf.newaxis, ...] * epsilon
        )
        shape = tf.shape(latent_samples)
        flat_samples = tf.reshape(
            latent_samples,
            [shape[0] * shape[1], shape[2], shape[3]],
        )
        flat_logits = self._subject_head_forward(flat_samples, training=training)
        logits_by_sample = tf.reshape(
            flat_logits,
            [shape[0], shape[1], self.n_subject_classes],
        )
        # Average the predictive distribution across Monte Carlo draws. The
        # returned normalized log-probabilities are valid logits for sparse CE.
        log_probabilities = tf.nn.log_softmax(logits_by_sample, axis=-1)
        return (
            tf.reduce_logsumexp(log_probabilities, axis=0)
            - tf.math.log(tf.cast(sample_count, log_probabilities.dtype))
        )

    @property
    def metrics(self) -> list[tf.keras.metrics.Metric]:
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
        if self.use_subject_adversarial:
            metrics.extend(
                [
                    self.subject_loss_tracker,
                    self.weighted_subject_loss_tracker,
                    self.subject_accuracy_tracker,
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

    def fit(self, *args, **kwargs):
        """Optionally discard externally supplied class weights.

        ``cross_val.loso_cv`` currently always supplies fold-local class
        weights. Removing them here keeps the toggle local to this model and
        applies consistently to inner fits, outer fits, and the final fit.
        """
        if not self.use_class_weight:
            kwargs.pop("class_weight", None)
        return super().fit(*args, **kwargs)

    def compile(self, optimizer=None, discriminator_optimizer=None, **kwargs) -> None:
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
    def _reparameterize(z_mean: tf.Tensor, z_log_var: tf.Tensor) -> tf.Tensor:
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
        encoder_output = self.encoder(inputs, training=training)
        if encoder_output.shape.rank != 3:
            raise ValueError(
                "encoder must return (batch, latent_timesteps, features); got "
                f"{encoder_output.shape}."
            )

        z_mean = self.z_mean_projection(encoder_output)
        raw_z_log_var = self.z_log_var_projection(encoder_output)
        tf.debugging.assert_equal(tf.shape(z_mean), tf.shape(raw_z_log_var))
        tf.debugging.assert_all_finite(z_mean, "z_mean contains NaN or Inf values.")
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

    def _classify_latents(
        self,
        latent_sequence: tf.Tensor,
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        classification_latent = self.classification_model(
            latent_sequence,
            training=training,
        )
        if classification_latent.shape.rank != 2:
            raise ValueError(
                "classification_model must return one rank-2 embedding per "
                f"window; got {classification_latent.shape}."
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
        include_reconstruction: bool | None = None,
        include_subject_adversarial: bool | None = None,
    ) -> dict[str, tf.Tensor]:
        """Run reconstruction and classification for independent windows."""
        inputs, _subject_ids = self._split_eeg_and_subject_inputs(inputs)
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        if inputs.shape.rank != 3:
            raise ValueError(
                "Window-level input must have shape (batch, timesteps, "
                f"features); got {inputs.shape}."
            )
        if sample_latent is None:
            sample_latent = training
        if include_reconstruction is None:
            include_reconstruction = self.ae_loss_weight > 0.0
        if include_subject_adversarial is None:
            include_subject_adversarial = (
                self.use_subject_adversarial and training is True
            )

        encoder_output, z_mean, z_log_var = self._posterior_parameters(
            inputs,
            training=training,
        )
        # The classifier uses the deterministic posterior mean during training.
        # Feeding a fresh unit-variance posterior sample to every four-second
        # window can overwhelm its affective signal and produce a constant-class
        # solution. Sampling remains available for the decoder and explicit MC
        # prediction.
        classification_sequence = z_mean
        if include_reconstruction:
            decoder_latent_sequence = self._latent_for_mode(
                z_mean=z_mean,
                z_log_var=z_log_var,
                sample_latent=sample_latent,
            )
        else:
            decoder_latent_sequence = z_mean

        classification_latent, logits = self._classify_latents(
            classification_sequence,
            training=training,
        )

        probabilities = tf.nn.softmax(logits, axis=-1)
        if self.variational_classifier.n_classes == 2:
            logit_margin = logits[:, 1] - logits[:, 0]
        else:
            top_logits = tf.math.top_k(logits, k=2, sorted=True).values
            logit_margin = top_logits[:, 0] - top_logits[:, 1]

        outputs = {
            "encoder_output": encoder_output,
            "z_mean": z_mean,
            "z_log_var": z_log_var,
            "latent_sequence": decoder_latent_sequence,
            "classification_latent_sequence": classification_sequence,
            "window_classification_latent": classification_latent,
            "classification_latent": classification_latent,
            "logits": logits,
            "probabilities": probabilities,
            "logit_margin": logit_margin,
        }
        if include_reconstruction:
            outputs["reconstruction"] = self.decoder(
                decoder_latent_sequence,
                training=training,
            )
        if include_subject_adversarial:
            if not self.use_subject_adversarial:
                raise ValueError(
                    "include_subject_adversarial=True, but the subject branch "
                    "is disabled."
                )
            outputs["subject_logits"] = self._subject_logits_from_posterior(
                z_mean=z_mean,
                z_log_var=z_log_var,
                training=training,
            )
        return outputs

    def predict_diagnostics(
        self,
        inputs,
        batch_size: int | None = None,
    ) -> dict[str, tf.Tensor]:
        """Return deterministic internal tensors for prediction diagnostics.

        This method never samples the VAE posterior and never reconstructs the
        input. It therefore inspects the exact posterior-mean path used by
        ``predict_step`` and ordinary deterministic evaluation.
        """
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        if inputs.shape.rank != 3:
            raise ValueError(
                "Prediction diagnostics expect rank-3 window inputs; got "
                f"{inputs.shape}."
            )

        n_samples = int(tf.shape(inputs)[0].numpy())
        effective_batch_size = n_samples if batch_size is None else int(batch_size)
        if effective_batch_size < 1:
            raise ValueError("batch_size must be at least 1 when provided.")

        keys = (
            "encoder_output",
            "z_mean",
            "z_log_var",
            "classification_latent_sequence",
            "classification_latent",
            "logits",
            "probabilities",
            "logit_margin",
        )
        collected: dict[str, list[tf.Tensor]] = {key: [] for key in keys}

        for start in range(0, n_samples, effective_batch_size):
            batch_outputs = self(
                inputs[start : start + effective_batch_size],
                training=False,
                sample_latent=False,
                include_reconstruction=False,
            )
            for key in keys:
                collected[key].append(batch_outputs[key])

        return {
            key: tf.concat(values, axis=0)
            for key, values in collected.items()
        }

    def predict_mc_probabilities(
        self,
        inputs,
        n_samples: int = 30,
        seed: int | tuple[int, int] | None = None,
    ) -> dict[str, tf.Tensor]:
        """Average window probabilities across posterior latent draws."""
        if n_samples < 1:
            raise ValueError("n_samples must be at least 1.")
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        if inputs.shape.rank != 3:
            raise ValueError(
                "Monte Carlo prediction expects rank-3 window inputs; got "
                f"{inputs.shape}."
            )

        _encoder_output, z_mean, z_log_var = self._posterior_parameters(
            inputs,
            training=False,
        )
        epsilon_shape = tf.concat(
            [tf.constant([int(n_samples)], dtype=tf.int32), tf.shape(z_mean)],
            axis=0,
        )
        if seed is None:
            epsilon = tf.random.normal(epsilon_shape, dtype=z_mean.dtype)
        else:
            if isinstance(seed, int):
                stateless_seed = tf.constant([seed, 0], dtype=tf.int32)
            else:
                if len(seed) != 2:
                    raise ValueError("seed tuple must contain exactly two integers.")
                stateless_seed = tf.constant(seed, dtype=tf.int32)
            epsilon = tf.random.stateless_normal(
                epsilon_shape,
                seed=stateless_seed,
                dtype=z_mean.dtype,
            )

        z_samples = (
            z_mean[tf.newaxis, ...]
            + tf.exp(0.5 * z_log_var)[tf.newaxis, ...] * epsilon
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
        _classification_latent, logits_flat = self._classify_latents(
            z_samples_flat,
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
        if not isinstance(data, tuple):
            raise ValueError("Expected data as an (x, y) tuple.")
        if len(data) == 2:
            return data[0], data[1], None
        if len(data) == 3:
            return data[0], data[1], data[2]
        raise ValueError("Expected (x, y) or (x, y, sample_weight).")

    @staticmethod
    def _flatten_labels(y) -> tf.Tensor:
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
        eeg_inputs, subject_ids = self._split_eeg_and_subject_inputs(x)
        eeg_inputs = tf.convert_to_tensor(eeg_inputs, dtype=tf.float32)
        if eeg_inputs.shape.rank != 3:
            raise ValueError(
                "Window-level training expects (B, T, F); got "
                f"{eeg_inputs.shape}."
            )

        subject_enabled_for_batch = (
            self.use_subject_adversarial and subject_ids is not None
        )
        outputs = self(
            eeg_inputs,
            training=training,
            include_reconstruction=autoencoder_enabled,
            include_subject_adversarial=subject_enabled_for_batch,
        )

        zero = tf.zeros((), dtype=outputs["logits"].dtype)
        if autoencoder_enabled:
            batch_size = tf.shape(eeg_inputs)[0]
            z_mean_flat = tf.reshape(outputs["z_mean"], [batch_size, -1])
            z_log_var_flat = tf.reshape(outputs["z_log_var"], [batch_size, -1])
            vae_losses = self.reconstruction_loss_fn(
                x_true=eeg_inputs,
                x_pred=outputs["reconstruction"],
                z_mean=z_mean_flat,
                z_log_var=z_log_var_flat,
                include_subject_loss=False,
            )
            autoencoder_loss = (
                vae_losses["reconstruction_loss"]
                + vae_losses["weighted_kl_loss"]
            )
        else:
            vae_losses = {
                "reconstruction_loss": zero,
                "kl_loss": zero,
                "weighted_kl_loss": zero,
            }
            autoencoder_loss = zero

        if subject_enabled_for_batch:
            subject_ids = tf.cast(tf.reshape(subject_ids, [-1]), tf.int32)
            tf.debugging.assert_equal(
                tf.shape(subject_ids)[0],
                tf.shape(outputs["subject_logits"])[0],
                message="Subject labels must align with EEG windows.",
            )
            subject_loss_per_sample = self.reconstruction_loss_fn.compute_subject_loss(
                subject_true=subject_ids,
                subject_pred=outputs["subject_logits"],
            )
            subject_loss = tf.reduce_mean(subject_loss_per_sample)
            weighted_subject_loss = (
                tf.cast(self.subject_loss_weight, subject_loss.dtype)
                * subject_loss
            )
            outputs["subject_targets"] = subject_ids
        else:
            subject_loss = zero
            weighted_subject_loss = zero

        y_flat = self._flatten_labels(y)
        tf.debugging.assert_equal(
            tf.shape(y_flat)[0],
            tf.shape(outputs["logits"])[0],
            message=(
                "The classifier must emit one prediction per window and labels "
                "must contain one class ID per window."
            ),
        )
        vc_losses = self.variational_classifier.vc_loss_components(
            mh=outputs["classification_latent"],
            y=y_flat,
            alpha=self.vc_alpha,
            beta=self.vc_beta,
            gamma=self.vc_gamma,
            lambda_=self.vc_lambda,
            logits=outputs["logits"],
            sample_weight=sample_weight,
        )

        base_total_loss = (
            self.ae_loss_weight * autoencoder_loss
            + self.vc_loss_weight * vc_losses["total_loss"]
            + weighted_subject_loss
        )
        if self.losses:
            regularization_loss = tf.add_n(
                [tf.cast(layer_loss, base_total_loss.dtype) for layer_loss in self.losses]
            )
        else:
            regularization_loss = tf.zeros_like(base_total_loss)
        total_loss = base_total_loss + regularization_loss

        losses = {
            "total_loss": total_loss,
            "base_total_loss": base_total_loss,
            "regularization_loss": regularization_loss,
            "autoencoder_loss": autoencoder_loss,
            "reconstruction_loss": vae_losses["reconstruction_loss"],
            "kl_loss": vae_losses["kl_loss"],
            "weighted_kl_loss": vae_losses["weighted_kl_loss"],
            "subject_loss": subject_loss,
            "weighted_subject_loss": weighted_subject_loss,
            "vc_loss": vc_losses["total_loss"],
            "vc_cross_entropy": vc_losses["cross_entropy"],
            "weighted_vc_cross_entropy": vc_losses["weighted_cross_entropy"],
            "vc_latent_kl": vc_losses["latent_posterior_kl"],
            "weighted_vc_latent_kl": vc_losses["weighted_latent_posterior_kl"],
            "vc_discriminator_kl": vc_losses["discriminator_kl"],
            "weighted_vc_discriminator_kl": vc_losses["weighted_discriminator_kl"],
            "vc_class_prior_kl": vc_losses["class_prior_kl"],
            "weighted_vc_class_prior_kl": vc_losses["weighted_class_prior_kl"],
        }
        return losses, outputs

    def _discriminator_variables(self) -> list[tf.Variable]:
        if not hasattr(self.variational_classifier, "disc_w"):
            return []
        return [
            self.variational_classifier.disc_w,
            self.variational_classifier.disc_b,
        ]

    @staticmethod
    def _apply_gradients(optimizer, gradients, variables) -> None:
        pairs = [
            (gradient, variable)
            for gradient, variable in zip(gradients, variables)
            if gradient is not None
        ]
        if pairs:
            optimizer.apply_gradients(pairs)

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
        self.regularization_loss_tracker.update_state(losses["regularization_loss"])
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
        if self.use_subject_adversarial:
            self.subject_loss_tracker.update_state(losses["subject_loss"])
            self.weighted_subject_loss_tracker.update_state(
                losses["weighted_subject_loss"]
            )
            if "subject_targets" in outputs and "subject_logits" in outputs:
                self.subject_accuracy_tracker.update_state(
                    outputs["subject_targets"],
                    outputs["subject_logits"],
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
        self.vc_class_prior_kl_tracker.update_state(losses["vc_class_prior_kl"])
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
        for class_index, tracker in enumerate(self.true_class_fraction_trackers):
            tracker.update_state(tf.cast(tf.equal(y_flat, class_index), tf.float32))
        for class_index, tracker in enumerate(
            self.predicted_class_fraction_trackers
        ):
            tracker.update_state(
                tf.cast(tf.equal(predicted_classes, class_index), tf.float32)
            )

    def _metric_results(self) -> dict[str, tf.Tensor]:
        return {metric.name: metric.result() for metric in self.metrics}

    def train_step(self, data) -> dict[str, tf.Tensor]:
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
        discriminator_variable_ids = {id(v) for v in discriminator_variables}
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
                discriminator_loss = self.variational_classifier.discriminator_loss(
                    tf.stop_gradient(outputs["classification_latent"]),
                    y_flat,
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
            x_true=self._split_eeg_and_subject_inputs(x)[0],
            y_flat=y_flat,
            sample_weight=sample_weight,
            discriminator_loss=discriminator_loss,
        )
        return self._metric_results()

    def test_step(self, data) -> dict[str, tf.Tensor]:
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
            x_true=self._split_eeg_and_subject_inputs(x)[0],
            y_flat=y_flat,
            sample_weight=sample_weight,
            discriminator_loss=discriminator_loss,
        )
        return self._metric_results()

    def predict_step(self, data):
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
        config = super().get_config()
        config.update(
            {
                "encoder": _serialize_keras_component(self.encoder),
                "decoder": _serialize_keras_component(self.decoder),
                "classification_model": _serialize_keras_component(
                    self.classification_model
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
                "use_class_weight": self.use_class_weight,
                "use_subject_adversarial": self.use_subject_adversarial,
                "n_subject_classes": self.n_subject_classes,
                "subject_adversarial_weight": self.subject_adversarial_weight,
                "subject_loss_weight": self.subject_loss_weight,
                "subject_hidden_units": self.subject_hidden_units,
                "subject_dropout": self.subject_dropout,
                "subject_latent_mode": self.subject_latent_mode,
                "subject_mc_samples": self.subject_mc_samples,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict):
        config = dict(config)
        # Old hierarchical checkpoints are not shape-compatible, but dropping
        # this key produces a clearer migration path than passing an unknown
        # argument into the window-level constructor.
        config.pop("trial_classification_model", None)
        for key in (
            "encoder",
            "decoder",
            "classification_model",
            "variational_classifier",
            "reconstruction_loss_fn",
        ):
            if key in config:
                config[key] = _deserialize_keras_component(config[key])
        return cls(**config)
