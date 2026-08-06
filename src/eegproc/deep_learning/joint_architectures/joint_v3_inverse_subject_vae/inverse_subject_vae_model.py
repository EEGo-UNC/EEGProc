"""Subject-adversarial STS variational autoencoder for EEG reconstruction.

This module reuses the spatiotemporal BiLSTM encoder, spatiospectral GCN
encoder, and cooperative dual-path decoder from ``joint_sts_model`` while
removing the emotion-classification pathway entirely.

Training alternates two updates per batch:

1. Subject-classifier update
   The encoder posterior mean is stop-gradient detached and the subject head
   learns to identify the fold-local training subjects.
2. Adversarial VAE update
   The encoder and decoder minimize reconstruction + beta-weighted KL. The
   subject cross-entropy is routed through gradient reversal, so the encoder is
   simultaneously trained to make subject identification difficult. Subject
   head variables are not updated in this phase.

Validation and held-out-subject testing are reconstruction-only. They use the
posterior mean, making ``decoder_accuracy`` a deterministic dataset-level R^2.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import tensorflow as tf

try:
    from ..joint_v3_sts.joint_sts_model import (
        DecoderReconstructionR2,
        DualPathSTSDecoder,
        build_spatiotemporal_bilstm_encoder,
    )
    from ...unsupervised.Convolutions.GCN import GCNEncoder
    from ...unsupervised.VariationalAutoencoderLoss import (
        GradientReversal,
        VariationalAutoencoderLoss,
    )
except ImportError:
    from eegproc.deep_learning.joint_architectures.joint_v3_sts.joint_sts_model import (
        DecoderReconstructionR2,
        DualPathSTSDecoder,
        build_spatiotemporal_bilstm_encoder,
    )
    from eegproc.deep_learning.unsupervised.Convolutions.GCN import GCNEncoder
    from eegproc.deep_learning.unsupervised.VariationalAutoencoderLoss import (
        GradientReversal,
        VariationalAutoencoderLoss,
    )


def _serialize_keras_component(component):
    if component is None:
        return None
    return tf.keras.utils.serialize_keras_object(component)


def _deserialize_keras_component(config):
    if config is None or isinstance(
        config,
        (tf.keras.Model, tf.keras.layers.Layer, tf.keras.optimizers.Optimizer),
    ):
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


def _deduplicate_variables(variables: Sequence[tf.Variable]) -> list[tf.Variable]:
    seen: set[int] = set()
    output: list[tf.Variable] = []
    for variable in variables:
        variable_id = id(variable)
        if variable_id not in seen:
            output.append(variable)
            seen.add(variable_id)
    return output


def _resolve_temporal_pool_sizes(
    temporal_pool_sizes: Sequence[int] | None,
    t_down: int,
) -> tuple[int, ...]:
    t_down = int(t_down)
    if t_down < 1:
        raise ValueError("t_down must be at least 1.")
    if temporal_pool_sizes is None:
        pools = () if t_down == 1 else (t_down,)
    else:
        pools = tuple(int(value) for value in temporal_pool_sizes)
    if any(value < 1 for value in pools):
        raise ValueError("Every temporal pool size must be at least 1.")
    effective = int(np.prod(pools, dtype=np.int64)) if pools else 1
    if effective != t_down:
        raise ValueError(
            f"t_down={t_down}, but temporal_pool_sizes={pools} produces {effective}."
        )
    return pools


def _build_optimizer(
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
) -> tf.keras.optimizers.Optimizer:
    optimizer_name = str(optimizer_name).lower()
    learning_rate = float(learning_rate)
    weight_decay = float(weight_decay)
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative.")
    if optimizer_name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if optimizer_name == "adamw":
        return tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
    raise ValueError("optimizer_name must be 'adam' or 'adamw'.")


@tf.keras.utils.register_keras_serializable(package="EEGProc")
class InverseSubjectSTSVAE(tf.keras.Model):
    """Fused STS VAE trained to suppress fold-local subject identity."""

    def __init__(
        self,
        *,
        temporal_encoder: tf.keras.Model,
        spectral_encoder: tf.keras.Model,
        decoder: tf.keras.Model,
        fusion_dim: int = 64,
        latent_features: int = 32,
        fusion_dropout: float = 0.20,
        activation: str = "relu",
        vae_loss_weight: float = 1.0,
        vae_beta: float = 0.30,
        reconstruction_loss_fn: VariationalAutoencoderLoss | None = None,
        z_log_var_clip_min: float = -20.0,
        z_log_var_clip_max: float = 20.0,
        n_subject_classes: int | None = None,
        subject_adversarial_weight: float = 1.0,
        subject_loss_weight: float = 1.0,
        subject_hidden_units: int = 64,
        subject_dropout: float = 0.0,
        subject_steps_per_batch: int = 1,
        vae_steps_per_batch: int = 1,
        name: str = "inverse_subject_sts_vae",
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        if temporal_encoder is None or spectral_encoder is None or decoder is None:
            raise ValueError("temporal_encoder, spectral_encoder, and decoder are required.")
        if int(fusion_dim) < 1 or int(latent_features) < 1:
            raise ValueError("fusion_dim and latent_features must be positive.")
        if not 0.0 <= float(fusion_dropout) < 1.0:
            raise ValueError("fusion_dropout must be in [0, 1).")
        if float(vae_loss_weight) <= 0.0 or float(vae_beta) < 0.0:
            raise ValueError("vae_loss_weight must be positive and vae_beta non-negative.")
        if float(z_log_var_clip_min) >= float(z_log_var_clip_max):
            raise ValueError("z_log_var_clip_min must be below z_log_var_clip_max.")
        if float(subject_adversarial_weight) < 0.0:
            raise ValueError("subject_adversarial_weight must be non-negative.")
        if float(subject_loss_weight) < 0.0:
            raise ValueError("subject_loss_weight must be non-negative.")
        if int(subject_hidden_units) < 1:
            raise ValueError("subject_hidden_units must be positive.")
        if not 0.0 <= float(subject_dropout) < 1.0:
            raise ValueError("subject_dropout must be in [0, 1).")
        if int(subject_steps_per_batch) < 1 or int(vae_steps_per_batch) < 1:
            raise ValueError("Both alternating step counts must be at least 1.")
        if n_subject_classes is not None and int(n_subject_classes) < 2:
            raise ValueError("n_subject_classes must be at least 2.")

        self.temporal_encoder = temporal_encoder
        self.spectral_encoder = spectral_encoder
        self.decoder = decoder

        self.fusion_dim = int(fusion_dim)
        self.latent_features = int(latent_features)
        self.fusion_dropout_rate = float(fusion_dropout)
        self.activation_name = str(activation)
        self.vae_loss_weight = float(vae_loss_weight)
        self.vae_beta = float(vae_beta)
        self.z_log_var_clip_min = float(z_log_var_clip_min)
        self.z_log_var_clip_max = float(z_log_var_clip_max)

        self.fusion_projection = tf.keras.layers.Conv1D(
            self.fusion_dim,
            kernel_size=1,
            padding="same",
            activation=None,
            name="inverse_subject_fusion_projection",
        )
        self.fusion_normalization = tf.keras.layers.LayerNormalization(
            axis=-1,
            name="inverse_subject_fusion_normalization",
        )
        self.fusion_activation = tf.keras.layers.Activation(
            self.activation_name,
            name="inverse_subject_fusion_activation",
        )
        self.fusion_dropout = tf.keras.layers.Dropout(
            self.fusion_dropout_rate,
            name="inverse_subject_fusion_dropout",
        )
        self.z_mean_projection = tf.keras.layers.Conv1D(
            self.latent_features,
            kernel_size=1,
            padding="same",
            activation=None,
            name="inverse_subject_z_mean",
        )
        self.z_log_var_projection = tf.keras.layers.Conv1D(
            self.latent_features,
            kernel_size=1,
            padding="same",
            activation=None,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="inverse_subject_z_log_var",
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

        self.n_subject_classes = (
            None if n_subject_classes is None else int(n_subject_classes)
        )
        self.subject_adversarial_weight = float(subject_adversarial_weight)
        self.subject_loss_weight = float(subject_loss_weight)
        self.subject_hidden_units = int(subject_hidden_units)
        self.subject_dropout_rate = float(subject_dropout)
        self.subject_steps_per_batch = int(subject_steps_per_batch)
        self.vae_steps_per_batch = int(vae_steps_per_batch)

        self.subject_pooling = tf.keras.layers.GlobalAveragePooling1D(
            name="inverse_subject_pool"
        )
        self.subject_gradient_reversal = GradientReversal(
            adversarial_weight=self.subject_adversarial_weight,
            name="inverse_subject_gradient_reversal",
        )
        self.subject_hidden = tf.keras.layers.Dense(
            self.subject_hidden_units,
            activation=self.activation_name,
            name="inverse_subject_hidden",
        )
        self.subject_dropout_layer = tf.keras.layers.Dropout(
            self.subject_dropout_rate,
            name="inverse_subject_dropout",
        )
        self.subject_classifier: tf.keras.layers.Dense | None = None
        if self.n_subject_classes is not None:
            self._configure_subject_head(self.n_subject_classes)

        # Compatibility flags used by EEGProc-style input preparation.
        self.requires_subject_ids = True
        self.use_subject_adversarial = True

        self.vae_optimizer: tf.keras.optimizers.Optimizer | None = None
        self.subject_optimizer: tf.keras.optimizers.Optimizer | None = None

        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.vae_objective_tracker = tf.keras.metrics.Mean(name="vae_objective")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.weighted_kl_loss_tracker = tf.keras.metrics.Mean(
            name="weighted_kl_loss"
        )
        self.regularization_loss_tracker = tf.keras.metrics.Mean(
            name="regularization_loss"
        )
        self.decoder_accuracy_tracker = DecoderReconstructionR2(
            name="decoder_accuracy"
        )
        self.subject_classifier_loss_tracker = tf.keras.metrics.Mean(
            name="subject_classifier_loss"
        )
        self.subject_adversarial_loss_tracker = tf.keras.metrics.Mean(
            name="subject_adversarial_loss"
        )
        self.weighted_subject_adversarial_loss_tracker = tf.keras.metrics.Mean(
            name="weighted_subject_adversarial_loss"
        )
        self.subject_accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(
            name="subject_accuracy"
        )

    @property
    def metrics(self) -> list[tf.keras.metrics.Metric]:
        return [
            self.total_loss_tracker,
            self.vae_objective_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.weighted_kl_loss_tracker,
            self.regularization_loss_tracker,
            self.decoder_accuracy_tracker,
            self.subject_classifier_loss_tracker,
            self.subject_adversarial_loss_tracker,
            self.weighted_subject_adversarial_loss_tracker,
            self.subject_accuracy_tracker,
        ]

    def compile(
        self,
        *,
        vae_optimizer: tf.keras.optimizers.Optimizer,
        subject_optimizer: tf.keras.optimizers.Optimizer,
        **kwargs,
    ) -> None:
        if vae_optimizer is None or subject_optimizer is None:
            raise ValueError("vae_optimizer and subject_optimizer are required.")
        kwargs.setdefault("jit_compile", False)
        super().compile(optimizer=vae_optimizer, **kwargs)
        self.vae_optimizer = vae_optimizer
        self.subject_optimizer = subject_optimizer

    def get_compile_config(self) -> dict:
        config = super().get_compile_config()
        config.pop("optimizer", None)
        config.update(
            {
                "vae_optimizer": _serialize_keras_component(self.vae_optimizer),
                "subject_optimizer": _serialize_keras_component(
                    self.subject_optimizer
                ),
            }
        )
        return config

    def compile_from_config(self, config: dict) -> None:
        config = dict(config)
        config.pop("optimizer", None)
        self.compile(
            vae_optimizer=_deserialize_keras_component(
                config.pop("vae_optimizer")
            ),
            subject_optimizer=_deserialize_keras_component(
                config.pop("subject_optimizer")
            ),
            **config,
        )

    def _configure_subject_head(self, n_subject_classes: int) -> None:
        n_subject_classes = int(n_subject_classes)
        if n_subject_classes < 2:
            raise ValueError("Subject adversity requires at least two subjects.")
        if self.subject_classifier is not None:
            if self.n_subject_classes != n_subject_classes:
                raise ValueError(
                    "The subject head is already configured for "
                    f"{self.n_subject_classes} classes, not {n_subject_classes}."
                )
            return
        self.n_subject_classes = n_subject_classes
        self.subject_classifier = tf.keras.layers.Dense(
            self.n_subject_classes,
            activation=None,
            name="inverse_subject_logits",
        )

    def prepare_fit_inputs(self, eeg_inputs, subject_ids):
        """Attach contiguous fold-local subject labels to training EEG."""
        eeg_array = np.asarray(eeg_inputs, dtype=np.float32)
        subjects = np.asarray(subject_ids).reshape(-1)
        if len(eeg_array) != len(subjects):
            raise ValueError(
                "EEG samples and subject IDs must align; got "
                f"{len(eeg_array)} and {len(subjects)}."
            )
        unique_subjects = np.sort(np.unique(subjects))
        self._configure_subject_head(len(unique_subjects))
        subject_to_class = {
            value.item() if isinstance(value, np.generic) else value: index
            for index, value in enumerate(unique_subjects)
        }
        remapped = np.asarray(
            [
                subject_to_class[
                    value.item() if isinstance(value, np.generic) else value
                ]
                for value in subjects
            ],
            dtype=np.int32,
        )
        return {"eeg": eeg_array, "subject_id": remapped}

    @staticmethod
    def _split_inputs(inputs):
        if isinstance(inputs, Mapping):
            if "eeg" not in inputs:
                raise ValueError("Input mappings must contain an 'eeg' key.")
            return inputs["eeg"], inputs.get("subject_id")
        return inputs, None

    @staticmethod
    def _reparameterize(z_mean: tf.Tensor, z_log_var: tf.Tensor) -> tf.Tensor:
        epsilon = tf.random.normal(tf.shape(z_mean), dtype=z_mean.dtype)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def _encode_fused(self, eeg_inputs: tf.Tensor, training: bool) -> dict[str, tf.Tensor]:
        temporal_sequence = self.temporal_encoder(eeg_inputs, training=training)
        spectral_sequence = self.spectral_encoder(eeg_inputs, training=training)
        if temporal_sequence.shape.rank != 3 or spectral_sequence.shape.rank != 3:
            raise ValueError("Both STS encoders must return rank-3 sequences.")
        tf.debugging.assert_equal(
            tf.shape(temporal_sequence)[:2],
            tf.shape(spectral_sequence)[:2],
            message="Temporal and spectral latent sequences must align.",
        )
        fused = tf.concat([temporal_sequence, spectral_sequence], axis=-1)
        fused = self.fusion_projection(fused)
        fused = self.fusion_normalization(fused)
        fused = self.fusion_activation(fused)
        fused = self.fusion_dropout(fused, training=training)
        z_mean = self.z_mean_projection(fused)
        raw_z_log_var = self.z_log_var_projection(fused)
        tf.debugging.assert_all_finite(z_mean, "z_mean contains NaN or Inf.")
        tf.debugging.assert_all_finite(
            raw_z_log_var,
            "Unclipped z_log_var contains NaN or Inf.",
        )
        z_log_var = tf.clip_by_value(
            raw_z_log_var,
            tf.cast(self.z_log_var_clip_min, raw_z_log_var.dtype),
            tf.cast(self.z_log_var_clip_max, raw_z_log_var.dtype),
        )
        return {
            "temporal_sequence": temporal_sequence,
            "spectral_sequence": spectral_sequence,
            "fused_sequence": fused,
            "z_mean": z_mean,
            "z_log_var": z_log_var,
        }

    def _subject_logits(
        self,
        latent_sequence: tf.Tensor,
        *,
        training: bool,
        reverse_gradient: bool,
    ) -> tf.Tensor:
        if self.subject_classifier is None:
            raise RuntimeError(
                "Subject head is not configured. Call prepare_fit_inputs first."
            )
        pooled = self.subject_pooling(latent_sequence)
        if reverse_gradient:
            pooled = self.subject_gradient_reversal(pooled)
        hidden = self.subject_hidden(pooled)
        hidden = self.subject_dropout_layer(hidden, training=training)
        return self.subject_classifier(hidden)

    def call(
        self,
        inputs,
        training: bool = False,
        sample_latent: bool | None = None,
        include_subject_logits: bool = False,
        reverse_subject_gradient: bool = True,
    ) -> dict[str, tf.Tensor]:
        eeg_inputs, _subject_ids = self._split_inputs(inputs)
        eeg_inputs = tf.convert_to_tensor(eeg_inputs, dtype=tf.float32)
        if eeg_inputs.shape.rank != 3:
            raise ValueError(
                "InverseSubjectSTSVAE expects (batch, timesteps, features); "
                f"got {eeg_inputs.shape}."
            )
        if sample_latent is None:
            sample_latent = bool(training)
        encoded = self._encode_fused(eeg_inputs, training=training)
        latent_sequence = (
            self._reparameterize(encoded["z_mean"], encoded["z_log_var"])
            if sample_latent
            else encoded["z_mean"]
        )
        reconstruction = self.decoder(latent_sequence, training=training)
        outputs = {
            **encoded,
            "encoder_output": encoded["fused_sequence"],
            "latent_sequence": latent_sequence,
            "reconstruction": reconstruction,
        }
        if include_subject_logits:
            outputs["subject_logits"] = self._subject_logits(
                encoded["z_mean"],
                training=training,
                reverse_gradient=reverse_subject_gradient,
            )
        return outputs

    def _regularization_loss(self, dtype: tf.dtypes.DType) -> tf.Tensor:
        if not self.losses:
            return tf.zeros((), dtype=dtype)
        return tf.add_n([tf.cast(value, dtype) for value in self.losses])

    def _vae_components(
        self,
        eeg_inputs: tf.Tensor,
        *,
        training: bool,
        include_subject_adversarial: bool,
        subject_ids: tf.Tensor | None,
    ) -> tuple[dict[str, tf.Tensor], dict[str, tf.Tensor]]:
        outputs = self(
            eeg_inputs,
            training=training,
            sample_latent=training,
            include_subject_logits=include_subject_adversarial,
            reverse_subject_gradient=True,
        )
        batch_size = tf.shape(eeg_inputs)[0]
        z_mean_flat = tf.reshape(outputs["z_mean"], [batch_size, -1])
        z_log_var_flat = tf.reshape(outputs["z_log_var"], [batch_size, -1])
        components = self.reconstruction_loss_fn(
            x_true=eeg_inputs,
            x_pred=outputs["reconstruction"],
            z_mean=z_mean_flat,
            z_log_var=z_log_var_flat,
            include_subject_loss=False,
        )
        vae_loss = components["reconstruction_loss"] + components["weighted_kl_loss"]
        zero = tf.zeros((), dtype=vae_loss.dtype)
        if include_subject_adversarial:
            if subject_ids is None:
                raise ValueError("Subject IDs are required for adversarial training.")
            subject_ids = tf.cast(tf.reshape(subject_ids, [-1]), tf.int32)
            per_sample = tf.keras.losses.sparse_categorical_crossentropy(
                subject_ids,
                outputs["subject_logits"],
                from_logits=True,
            )
            subject_loss = tf.reduce_mean(per_sample)
            weighted_subject_loss = (
                tf.cast(self.subject_loss_weight, subject_loss.dtype) * subject_loss
            )
        else:
            subject_loss = zero
            weighted_subject_loss = zero
        regularization_loss = self._regularization_loss(vae_loss.dtype)
        objective = (
            tf.cast(self.vae_loss_weight, vae_loss.dtype) * vae_loss
            + weighted_subject_loss
            + regularization_loss
        )
        return (
            {
                "objective": objective,
                "vae_loss": vae_loss,
                "reconstruction_loss": components["reconstruction_loss"],
                "kl_loss": components["kl_loss"],
                "weighted_kl_loss": components["weighted_kl_loss"],
                "subject_adversarial_loss": subject_loss,
                "weighted_subject_adversarial_loss": weighted_subject_loss,
                "regularization_loss": regularization_loss,
            },
            outputs,
        )

    def _vae_variables(self) -> list[tf.Variable]:
        components: list[Any] = [
            self.temporal_encoder,
            self.spectral_encoder,
            self.fusion_projection,
            self.fusion_normalization,
            self.fusion_activation,
            self.fusion_dropout,
            self.z_mean_projection,
            self.z_log_var_projection,
            self.decoder,
        ]
        variables: list[tf.Variable] = []
        for component in components:
            variables.extend(component.trainable_variables)
        return _deduplicate_variables(variables)

    def _subject_variables(self) -> list[tf.Variable]:
        if self.subject_classifier is None:
            return []
        variables: list[tf.Variable] = []
        for component in (
            self.subject_hidden,
            self.subject_dropout_layer,
            self.subject_classifier,
        ):
            variables.extend(component.trainable_variables)
        return _deduplicate_variables(variables)

    @staticmethod
    def _apply_gradients(optimizer, gradients, variables) -> None:
        pairs = [
            (gradient, variable)
            for gradient, variable in zip(gradients, variables)
            if gradient is not None
        ]
        if pairs:
            optimizer.apply_gradients(pairs)

    def _update_reconstruction_metrics(
        self,
        losses: dict[str, tf.Tensor],
        outputs: dict[str, tf.Tensor],
        eeg_inputs: tf.Tensor,
    ) -> None:
        self.total_loss_tracker.update_state(losses["objective"])
        self.vae_objective_tracker.update_state(losses["vae_loss"])
        self.reconstruction_loss_tracker.update_state(losses["reconstruction_loss"])
        self.kl_loss_tracker.update_state(losses["kl_loss"])
        self.weighted_kl_loss_tracker.update_state(losses["weighted_kl_loss"])
        self.regularization_loss_tracker.update_state(losses["regularization_loss"])
        self.decoder_accuracy_tracker.update_state(
            eeg_inputs,
            outputs["reconstruction"],
        )
        self.subject_adversarial_loss_tracker.update_state(
            losses["subject_adversarial_loss"]
        )
        self.weighted_subject_adversarial_loss_tracker.update_state(
            losses["weighted_subject_adversarial_loss"]
        )

    def train_step(self, data) -> dict[str, tf.Tensor]:
        if self.vae_optimizer is None or self.subject_optimizer is None:
            raise RuntimeError("Call model.compile(...) before model.fit(...).")
        x, _y, _sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        eeg_inputs, subject_ids = self._split_inputs(x)
        if subject_ids is None:
            raise ValueError(
                "Training requires fold-local subject IDs. Use prepare_fit_inputs."
            )
        eeg_inputs = tf.convert_to_tensor(eeg_inputs, dtype=tf.float32)
        subject_ids = tf.cast(tf.reshape(subject_ids, [-1]), tf.int32)

        subject_loss = tf.zeros((), dtype=tf.float32)
        subject_logits = None
        for _ in range(self.subject_steps_per_batch):
            with tf.GradientTape() as subject_tape:
                encoded = self._encode_fused(eeg_inputs, training=False)
                detached_latent = tf.stop_gradient(encoded["z_mean"])
                subject_logits = self._subject_logits(
                    detached_latent,
                    training=True,
                    reverse_gradient=False,
                )
                subject_loss = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(
                        subject_ids,
                        subject_logits,
                        from_logits=True,
                    )
                )
            subject_variables = self._subject_variables()
            subject_gradients = subject_tape.gradient(
                subject_loss,
                subject_variables,
            )
            self._apply_gradients(
                self.subject_optimizer,
                subject_gradients,
                subject_variables,
            )

        vae_losses = vae_outputs = None
        for _ in range(self.vae_steps_per_batch):
            with tf.GradientTape() as vae_tape:
                vae_losses, vae_outputs = self._vae_components(
                    eeg_inputs,
                    training=True,
                    include_subject_adversarial=True,
                    subject_ids=subject_ids,
                )
            vae_variables = self._vae_variables()
            vae_gradients = vae_tape.gradient(
                vae_losses["objective"],
                vae_variables,
            )
            self._apply_gradients(
                self.vae_optimizer,
                vae_gradients,
                vae_variables,
            )

        self._update_reconstruction_metrics(vae_losses, vae_outputs, eeg_inputs)
        self.subject_classifier_loss_tracker.update_state(subject_loss)
        if subject_logits is not None:
            self.subject_accuracy_tracker.update_state(subject_ids, subject_logits)
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data) -> dict[str, tf.Tensor]:
        x, _y, _sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        eeg_inputs, _subject_ids = self._split_inputs(x)
        eeg_inputs = tf.convert_to_tensor(eeg_inputs, dtype=tf.float32)
        losses, outputs = self._vae_components(
            eeg_inputs,
            training=False,
            include_subject_adversarial=False,
            subject_ids=None,
        )
        self._update_reconstruction_metrics(losses, outputs, eeg_inputs)
        return {metric.name: metric.result() for metric in self.metrics}

    def predict_step(self, data):
        inputs = data[0] if isinstance(data, tuple) else data
        return self(
            inputs,
            training=False,
            sample_latent=False,
            include_subject_logits=False,
        )["reconstruction"]

    def reconstruct(self, inputs, batch_size: int | None = None) -> np.ndarray:
        """Return deterministic posterior-mean reconstructions."""
        return np.asarray(self.predict(inputs, batch_size=batch_size, verbose=0))

    def encode(self, inputs, batch_size: int | None = None) -> dict[str, np.ndarray]:
        """Return deterministic fused posterior tensors in batches."""
        eeg_inputs, _subject_ids = self._split_inputs(inputs)
        eeg_array = np.asarray(eeg_inputs, dtype=np.float32)
        effective_batch_size = len(eeg_array) if batch_size is None else int(batch_size)
        if effective_batch_size < 1:
            raise ValueError("batch_size must be positive.")
        collected = {"z_mean": [], "z_log_var": [], "fused_sequence": []}
        for start in range(0, len(eeg_array), effective_batch_size):
            encoded = self._encode_fused(
                tf.convert_to_tensor(
                    eeg_array[start : start + effective_batch_size],
                    dtype=tf.float32,
                ),
                training=False,
            )
            for key in collected:
                collected[key].append(np.asarray(encoded[key].numpy()))
        return {key: np.concatenate(value, axis=0) for key, value in collected.items()}

    def get_adjacency_matrices(self) -> dict[str, dict[str, tf.Tensor]]:
        output: dict[str, dict[str, tf.Tensor]] = {}
        if hasattr(self.spectral_encoder, "get_adjacency_matrices"):
            output["encoder"] = self.spectral_encoder.get_adjacency_matrices()
        if hasattr(self.decoder, "get_adjacency_matrices"):
            output["decoder"] = self.decoder.get_adjacency_matrices()
        return output

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "temporal_encoder": _serialize_keras_component(
                    self.temporal_encoder
                ),
                "spectral_encoder": _serialize_keras_component(
                    self.spectral_encoder
                ),
                "decoder": _serialize_keras_component(self.decoder),
                "fusion_dim": self.fusion_dim,
                "latent_features": self.latent_features,
                "fusion_dropout": self.fusion_dropout_rate,
                "activation": self.activation_name,
                "vae_loss_weight": self.vae_loss_weight,
                "vae_beta": self.vae_beta,
                "reconstruction_loss_fn": _serialize_keras_component(
                    self.reconstruction_loss_fn
                ),
                "z_log_var_clip_min": self.z_log_var_clip_min,
                "z_log_var_clip_max": self.z_log_var_clip_max,
                "n_subject_classes": self.n_subject_classes,
                "subject_adversarial_weight": self.subject_adversarial_weight,
                "subject_loss_weight": self.subject_loss_weight,
                "subject_hidden_units": self.subject_hidden_units,
                "subject_dropout": self.subject_dropout_rate,
                "subject_steps_per_batch": self.subject_steps_per_batch,
                "vae_steps_per_batch": self.vae_steps_per_batch,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict):
        config = dict(config)
        for key in (
            "temporal_encoder",
            "spectral_encoder",
            "decoder",
            "reconstruction_loss_fn",
        ):
            config[key] = _deserialize_keras_component(config[key])
        return cls(**config)


def build_inverse_subject_sts_vae(
    input_shape: tuple[int, int],
    *,
    n_channels: int = 14,
    n_bands: int = 3,
    t_down: int = 2,
    temporal_pool_sizes: Sequence[int] | None = (2,),
    bilstm_units: int = 64,
    n_bilstm_layers: int = 1,
    bilstm_dropout: float = 0.30,
    temporal_emb_dim: int = 32,
    gcn_units: Sequence[int] = (64, 32),
    spectral_emb_dim: int = 32,
    gcn_dropout: float = 0.20,
    gcn_activation: str = "relu",
    gcn_use_batch_norm: bool = False,
    graph_self_loop_bias: float = 2.0,
    graph_identity_mix: float = 0.0,
    graph_adjacency_reg_weight: float = 1e-4,
    fusion_dim: int = 64,
    latent_features: int = 32,
    fusion_dropout: float = 0.20,
    decoder_temporal_units: int = 64,
    decoder_bilstm_layers: int = 1,
    decoder_graph_output_units: int = 16,
    decoder_branch_feature_dim: int = 64,
    decoder_fusion_units: int = 64,
    decoder_dropout: float = 0.20,
    activation: str = "relu",
    reconstruction_loss: str = "mse",
    vae_loss_weight: float = 1.0,
    vae_beta: float = 0.30,
    n_subject_classes: int | None = None,
    subject_adversarial_weight: float = 1.0,
    subject_loss_weight: float = 1.0,
    subject_hidden_units: int = 64,
    subject_dropout: float = 0.0,
    subject_steps_per_batch: int = 1,
    vae_steps_per_batch: int = 1,
    optimizer_name: str = "adamw",
    vae_learning_rate: float = 5e-5,
    subject_learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    model_name: str = "inverse_subject_sts_vae",
) -> InverseSubjectSTSVAE:
    """Build and compile the reconstruction-only subject-adversarial STS VAE."""
    timesteps, n_features = map(int, input_shape)
    n_channels = int(n_channels)
    n_bands = int(n_bands)
    if n_features != n_channels * n_bands:
        raise ValueError(
            "STS input must satisfy n_features = n_channels * n_bands; got "
            f"{n_features} != {n_channels} * {n_bands}."
        )
    pools = _resolve_temporal_pool_sizes(temporal_pool_sizes, t_down)

    temporal_encoder = build_spatiotemporal_bilstm_encoder(
        timesteps=timesteps,
        n_features=n_features,
        lstm_units=int(bilstm_units),
        n_bilstm_layers=int(n_bilstm_layers),
        dropout=float(bilstm_dropout),
        temporal_pool_sizes=pools,
        t_down=int(t_down),
        emb_dim=int(temporal_emb_dim),
        name="inverse_subject_spatiotemporal_bilstm",
    )
    spectral_encoder = GCNEncoder(
        timesteps=timesteps,
        t_down=int(t_down),
        n_channels=n_channels,
        n_bands=n_bands,
        gcn_units=tuple(int(value) for value in gcn_units),
        temporal_pool_sizes=pools,
        emb_dim=int(spectral_emb_dim),
        dropout=float(gcn_dropout),
        activation=str(gcn_activation),
        use_batch_norm=bool(gcn_use_batch_norm),
        graph_self_loop_bias=float(graph_self_loop_bias),
        graph_identity_mix=float(graph_identity_mix),
        graph_adjacency_reg_weight=float(graph_adjacency_reg_weight),
        name="inverse_subject_spatiospectral_gcn",
    )
    decoder = DualPathSTSDecoder(
        timesteps=timesteps,
        n_channels=n_channels,
        n_bands=n_bands,
        t_down=int(t_down),
        temporal_pool_sizes=pools,
        gcn_units=tuple(int(value) for value in gcn_units),
        temporal_decoder_units=int(decoder_temporal_units),
        n_temporal_decoder_bilstm_layers=int(decoder_bilstm_layers),
        graph_output_units=int(decoder_graph_output_units),
        branch_feature_dim=int(decoder_branch_feature_dim),
        fusion_units=int(decoder_fusion_units),
        dropout=float(decoder_dropout),
        activation=str(activation),
        use_batch_norm=bool(gcn_use_batch_norm),
        graph_self_loop_bias=float(graph_self_loop_bias),
        graph_identity_mix=float(graph_identity_mix),
        graph_adjacency_reg_weight=float(graph_adjacency_reg_weight),
        name="inverse_subject_dual_path_decoder",
    )
    reconstruction_loss_fn = VariationalAutoencoderLoss(
        reconstruction=str(reconstruction_loss),
        beta=float(vae_beta),
        feature_reduction="mean",
        kl_reduction="mean",
        log_var_clip_min=-20.0,
        log_var_clip_max=20.0,
    )
    model = InverseSubjectSTSVAE(
        temporal_encoder=temporal_encoder,
        spectral_encoder=spectral_encoder,
        decoder=decoder,
        fusion_dim=int(fusion_dim),
        latent_features=int(latent_features),
        fusion_dropout=float(fusion_dropout),
        activation=str(activation),
        vae_loss_weight=float(vae_loss_weight),
        vae_beta=float(vae_beta),
        reconstruction_loss_fn=reconstruction_loss_fn,
        n_subject_classes=n_subject_classes,
        subject_adversarial_weight=float(subject_adversarial_weight),
        subject_loss_weight=float(subject_loss_weight),
        subject_hidden_units=int(subject_hidden_units),
        subject_dropout=float(subject_dropout),
        subject_steps_per_batch=int(subject_steps_per_batch),
        vae_steps_per_batch=int(vae_steps_per_batch),
        name=model_name,
    )
    model.compile(
        vae_optimizer=_build_optimizer(
            optimizer_name,
            vae_learning_rate,
            weight_decay,
        ),
        subject_optimizer=_build_optimizer(
            optimizer_name,
            subject_learning_rate,
            weight_decay,
        ),
    )
    return model


__all__ = [
    "InverseSubjectSTSVAE",
    "build_inverse_subject_sts_vae",
]
