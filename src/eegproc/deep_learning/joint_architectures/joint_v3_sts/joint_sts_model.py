"""Spatiotemporal-spatiospectral fused VAE classifier for EEG.

This module implements the STS architecture proposed for EEGProc:

    EEG window
      |-- spatiotemporal BiLSTM encoder --|
      |-- spatiospectral GCN encoder -----|--> sequence fusion --> q(z | x)
                                                         |-- classifier
                                                         |-- fused decoder
                                                         |-- subject adversary

The two principal objectives are optimized *alternately* inside every Keras
``train_step``:

1. Classification phase
   Updates both encoders, fusion/posterior layers, the classification pathway,
   the optional subject-adversarial head, and optional SupCon regularization.
2. VAE phase
   Recomputes the fused posterior after the classification update and updates
   both encoders, fusion/posterior layers, and the single graph-aware decoder
   with reconstruction + beta-weighted KL loss.

The decoder therefore reconstructs from the *fused* latent sequence rather
than from either branch independently. Classification defaults to a standard
DenseClassifier (dense logits + softmax probabilities), while the existing
HybridClassifier and VariationalClassifier remain selectable ablations.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import tensorflow as tf

try:
    from ..supervised.rnn_architectures import BiLSTMClassifier
    from ..supervised.variational_classifier import (
        DenseClassifier,
        HybridClassifier,
        VariationalClassifier,
    )
    from ..unsupervised.Convolutions.GCN import GCNDecoder, GCNEncoder
    from ..unsupervised.VariationalAutoencoderLoss import (
        GradientReversal,
        VariationalAutoencoderLoss,
    )
except ImportError:
    # Supports direct execution/import in the same style as the existing
    # joint_v2_autoencoder_vc module.
    from eegproc.deep_learning.supervised.rnn_architectures import (
        BiLSTMClassifier,
    )
    from eegproc.deep_learning.supervised.variational_classifier import (
        DenseClassifier,
        HybridClassifier,
        VariationalClassifier,
    )
    from eegproc.deep_learning.unsupervised.Convolutions.GCN import (
        GCNDecoder,
        GCNEncoder,
    )
    from eegproc.deep_learning.unsupervised.VariationalAutoencoderLoss import (
        GradientReversal,
        VariationalAutoencoderLoss,
    )


def _serialize_keras_component(component):
    """Serialize a nested Keras object."""
    if component is None:
        return None
    return tf.keras.utils.serialize_keras_object(component)


def _deserialize_keras_component(config):
    """Deserialize an EEGProc/Keras object with an import fallback."""
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


def _as_positive_int_tuple(
    name: str,
    values: Sequence[int] | None,
    *,
    allow_empty: bool = False,
) -> tuple[int, ...]:
    if values is None:
        normalized: tuple[int, ...] = ()
    else:
        normalized = tuple(int(value) for value in values)
    if not normalized and not allow_empty:
        raise ValueError(f"{name} must contain at least one value.")
    if any(value < 1 for value in normalized):
        raise ValueError(f"Every {name} value must be >= 1; got {normalized}.")
    return normalized


def _resolve_temporal_pool_sizes(
    temporal_pool_sizes: Sequence[int] | None,
    t_down: int,
) -> tuple[int, ...]:
    t_down = int(t_down)
    if t_down < 1:
        raise ValueError(f"t_down must be >= 1, got {t_down}.")
    pools = (
        (() if t_down == 1 else (t_down,))
        if temporal_pool_sizes is None
        else _as_positive_int_tuple(
            "temporal_pool_sizes",
            temporal_pool_sizes,
            allow_empty=True,
        )
    )
    effective_downsampling = int(np.prod(pools, dtype=np.int64)) if pools else 1
    if effective_downsampling != t_down:
        raise ValueError(
            f"t_down={t_down}, but temporal_pool_sizes={pools} produces "
            f"{effective_downsampling}."
        )
    return pools


def _deduplicate_variables(variables: Sequence[tf.Variable]) -> list[tf.Variable]:
    """Preserve variable order while removing shared-variable duplicates."""
    seen: set[int] = set()
    output: list[tf.Variable] = []
    for variable in variables:
        variable_id = id(variable)
        if variable_id not in seen:
            output.append(variable)
            seen.add(variable_id)
    return output


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


def build_spatiotemporal_bilstm_encoder(
    *,
    timesteps: int,
    n_features: int,
    lstm_units: int = 64,
    n_bilstm_layers: int = 1,
    dropout: float = 0.30,
    temporal_pool_sizes: Sequence[int] | None = (2,),
    t_down: int = 2,
    emb_dim: int = 32,
    name: str = "sts_temporal_encoder",
) -> tf.keras.Model:
    """Build a sequence-preserving BiLSTM encoder for fusion.

    ``BiLSTMClassifier.build_feature_extractor()`` intentionally collapses the
    temporal axis. Fused decoding needs a latent sequence, so this builder
    reuses the existing BiLSTM recurrent stack and replaces only its final
    global pooling with the same temporal downsampling used by the GCN branch.
    """
    timesteps = int(timesteps)
    n_features = int(n_features)
    emb_dim = int(emb_dim)
    if timesteps < 1 or n_features < 1 or emb_dim < 1:
        raise ValueError("timesteps, n_features, and emb_dim must be positive.")
    pools = _resolve_temporal_pool_sizes(temporal_pool_sizes, t_down)

    builder = BiLSTMClassifier(
        timesteps=timesteps,
        n_features=n_features,
        n_classes=2,  # The standalone head is not built or used here.
        lstm_units=int(lstm_units),
        n_bilstm_layers=int(n_bilstm_layers),
        dropout=float(dropout),
        name=name,
    )
    inputs = tf.keras.layers.Input(
        shape=(timesteps, n_features),
        name=f"{name}_input",
    )
    sequence = builder.recurrent_stack(inputs)
    for index, pool_size in enumerate(pools):
        sequence = tf.keras.layers.MaxPool1D(
            pool_size=pool_size,
            padding="same",
            name=f"{name}_pool_{index}",
        )(sequence)
    sequence = tf.keras.layers.Conv1D(
        emb_dim,
        kernel_size=1,
        padding="same",
        activation=None,
        name=f"{name}_sequence_projection",
    )(sequence)
    return tf.keras.Model(inputs, sequence, name=name)


@tf.keras.utils.register_keras_serializable(package="EEGProc")
class DecoderReconstructionR2(tf.keras.metrics.Metric):
    """Dataset-level reconstruction coefficient of determination."""

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
        del sample_weight
        y_true = tf.cast(y_true, self.dtype)
        y_pred = tf.cast(y_pred, self.dtype)
        error = y_true - y_pred
        self.squared_error_sum.assign_add(tf.reduce_sum(tf.square(error)))
        self.target_sum.assign_add(tf.reduce_sum(y_true))
        self.target_squared_sum.assign_add(tf.reduce_sum(tf.square(y_true)))
        self.target_count.assign_add(tf.cast(tf.size(y_true), self.dtype))

    def result(self):
        total_sum_of_squares = self.target_squared_sum - tf.math.divide_no_nan(
            tf.square(self.target_sum),
            self.target_count,
        )
        epsilon = tf.cast(tf.keras.backend.epsilon(), self.dtype)
        ordinary_r2 = 1.0 - tf.math.divide_no_nan(
            self.squared_error_sum,
            total_sum_of_squares,
        )
        perfect_constant = tf.cast(self.squared_error_sum <= epsilon, self.dtype)
        return tf.where(
            total_sum_of_squares > epsilon,
            ordinary_r2,
            perfect_constant,
        )

    def reset_state(self) -> None:
        for variable in self.variables:
            variable.assign(tf.zeros_like(variable))


@tf.keras.utils.register_keras_serializable(package="EEGProc")
class JointSTSModel(tf.keras.Model):
    """Parallel BiLSTM-GCN fused VAE with alternating optimization.

    Classification phase objective::

        L_cls = classification_loss_weight * L_classifier
                + subject_loss_weight * CE_subject(GRL(z_mean))
                + supcon_weight * L_supcon
                + regularization

    VAE phase objective::

        L_vae = vae_loss_weight * (L_reconstruction + beta * KL)
                + regularization

    The phases use separate gradient tapes and optimizer states. Shared encoder,
    fusion, and posterior variables are therefore updated once by each objective
    per batch. Subject adversity belongs to the classification phase by design,
    preventing it from being counted twice during a two-update batch.
    """

    def __init__(
        self,
        temporal_encoder: tf.keras.Model,
        spectral_encoder: tf.keras.Model,
        decoder: tf.keras.Model,
        classifier: tf.keras.layers.Layer,
        fusion_dim: int = 64,
        latent_features: int = 32,
        classification_hidden_units: int = 64,
        fusion_dropout: float = 0.20,
        classification_dropout: float = 0.30,
        activation: str = "relu",
        classification_loss_weight: float = 1.0,
        vae_loss_weight: float = 1.0,
        vae_beta: float = 0.30,
        reconstruction_loss_fn: VariationalAutoencoderLoss | None = None,
        z_log_var_clip_min: float = -20.0,
        z_log_var_clip_max: float = 20.0,
        vc_alpha: float = 1.0,
        vc_beta: float = 0.0,
        vc_gamma: float = 0.0,
        vc_lambda: float = 0.0,
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
        use_supcon: bool = False,
        supcon_weight: float = 0.03,
        supcon_temperature: float = 0.10,
        supcon_cross_subject_only: bool = True,
        classification_steps_per_batch: int = 1,
        vae_steps_per_batch: int = 1,
        name: str = "joint_sts_model",
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)

        for component_name, component in (
            ("temporal_encoder", temporal_encoder),
            ("spectral_encoder", spectral_encoder),
            ("decoder", decoder),
            ("classifier", classifier),
        ):
            if component is None:
                raise ValueError(f"{component_name} must be provided.")
        if fusion_dim < 1 or latent_features < 1:
            raise ValueError("fusion_dim and latent_features must be positive.")
        if classification_hidden_units < 1:
            raise ValueError("classification_hidden_units must be positive.")
        if not 0.0 <= fusion_dropout < 1.0:
            raise ValueError("fusion_dropout must be in [0, 1).")
        if not 0.0 <= classification_dropout < 1.0:
            raise ValueError("classification_dropout must be in [0, 1).")
        if classification_loss_weight <= 0.0:
            raise ValueError("classification_loss_weight must be positive.")
        if vae_loss_weight <= 0.0:
            raise ValueError("vae_loss_weight must be positive.")
        if vae_beta < 0.0:
            raise ValueError("vae_beta must be non-negative.")
        if z_log_var_clip_min >= z_log_var_clip_max:
            raise ValueError("z_log_var_clip_min must be below z_log_var_clip_max.")
        if subject_adversarial_weight < 0.0 or subject_loss_weight < 0.0:
            raise ValueError("Subject loss weights must be non-negative.")
        if subject_hidden_units < 1:
            raise ValueError("subject_hidden_units must be positive.")
        if not 0.0 <= subject_dropout < 1.0:
            raise ValueError("subject_dropout must be in [0, 1).")
        subject_latent_mode = str(subject_latent_mode).lower()
        if subject_latent_mode not in {"mean", "mc"}:
            raise ValueError("subject_latent_mode must be 'mean' or 'mc'.")
        if subject_mc_samples < 1:
            raise ValueError("subject_mc_samples must be positive.")
        if supcon_weight < 0.0:
            raise ValueError("supcon_weight must be non-negative.")
        if supcon_temperature <= 0.0:
            raise ValueError("supcon_temperature must be positive.")
        if classification_steps_per_batch < 1 or vae_steps_per_batch < 1:
            raise ValueError("Both alternating step counts must be at least 1.")
        if n_subject_classes is not None and int(n_subject_classes) < 2:
            raise ValueError("n_subject_classes must be at least 2.")

        self.temporal_encoder = temporal_encoder
        self.spectral_encoder = spectral_encoder
        self.decoder = decoder
        self.classifier = classifier

        self.fusion_dim = int(fusion_dim)
        self.latent_features = int(latent_features)
        self.classification_hidden_units = int(classification_hidden_units)
        self.fusion_dropout_rate = float(fusion_dropout)
        self.classification_dropout_rate = float(classification_dropout)
        self.activation_name = str(activation)

        self.fusion_projection = tf.keras.layers.Conv1D(
            self.fusion_dim,
            kernel_size=1,
            padding="same",
            activation=None,
            name="sts_fusion_projection",
        )
        self.fusion_normalization = tf.keras.layers.LayerNormalization(
            axis=-1,
            name="sts_fusion_normalization",
        )
        self.fusion_activation = tf.keras.layers.Activation(
            self.activation_name,
            name="sts_fusion_activation",
        )
        self.fusion_dropout = tf.keras.layers.Dropout(
            self.fusion_dropout_rate,
            name="sts_fusion_dropout",
        )
        self.z_mean_projection = tf.keras.layers.Conv1D(
            self.latent_features,
            kernel_size=1,
            padding="same",
            activation=None,
            name="sts_z_mean",
        )
        self.z_log_var_projection = tf.keras.layers.Conv1D(
            self.latent_features,
            kernel_size=1,
            padding="same",
            activation=None,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="sts_z_log_var",
        )

        self.classification_pool = tf.keras.layers.GlobalAveragePooling1D(
            name="sts_classification_pool",
        )
        self.classification_projection = tf.keras.layers.Dense(
            self.classification_hidden_units,
            activation=self.activation_name,
            name="sts_classification_projection",
        )
        self.classification_normalization = tf.keras.layers.LayerNormalization(
            axis=-1,
            name="sts_classification_normalization",
        )
        self.classification_dropout = tf.keras.layers.Dropout(
            self.classification_dropout_rate,
            name="sts_classification_dropout",
        )

        self.classification_loss_weight = float(classification_loss_weight)
        self.vae_loss_weight = float(vae_loss_weight)
        self.vae_beta = float(vae_beta)
        self.z_log_var_clip_min = float(z_log_var_clip_min)
        self.z_log_var_clip_max = float(z_log_var_clip_max)
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
        self.use_class_weight = bool(use_class_weight)

        classifier_supports_discriminator = bool(
            getattr(
                self.classifier,
                "supports_discriminator",
                hasattr(self.classifier, "discriminator_loss"),
            )
        )
        if (
            classifier_supports_discriminator
            and self.vc_gamma > 0.0
            and not self.update_discriminator
        ):
            raise ValueError(
                "vc_gamma is positive, but update_discriminator=False. "
                "Enable discriminator updates or set vc_gamma=0."
            )

        self.subject_adversarial_enabled = bool(use_subject_adversarial)
        self.use_supcon = bool(use_supcon)
        self.supcon_weight = float(supcon_weight)
        self.supcon_temperature = float(supcon_temperature)
        self.supcon_cross_subject_only = bool(supcon_cross_subject_only)
        self.requires_subject_ids = bool(
            self.subject_adversarial_enabled
            or (self.use_supcon and self.supcon_cross_subject_only)
        )
        # Compatibility with the current cross-validation metadata helper.
        self.use_subject_adversarial = self.requires_subject_ids

        self.n_subject_classes = (
            None if n_subject_classes is None else int(n_subject_classes)
        )
        self.subject_adversarial_weight = float(subject_adversarial_weight)
        self.subject_loss_weight = float(subject_loss_weight)
        self.subject_hidden_units = int(subject_hidden_units)
        self.subject_dropout_rate = float(subject_dropout)
        self.subject_latent_mode = subject_latent_mode
        self.subject_mc_samples = int(subject_mc_samples)

        self.subject_pooling = None
        self.subject_gradient_reversal = None
        self.subject_hidden = None
        self.subject_dropout_layer = None
        self.subject_classifier = None
        if self.subject_adversarial_enabled and self.n_subject_classes is not None:
            self._configure_subject_head(self.n_subject_classes)

        self.classification_steps_per_batch = int(classification_steps_per_batch)
        self.vae_steps_per_batch = int(vae_steps_per_batch)
        self.classification_optimizer = None
        self.vae_optimizer = None
        self.discriminator_optimizer = None

        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.classification_objective_tracker = tf.keras.metrics.Mean(
            name="classification_objective"
        )
        self.vae_objective_tracker = tf.keras.metrics.Mean(name="vae_objective")
        self.classification_regularization_tracker = tf.keras.metrics.Mean(
            name="classification_regularization_loss"
        )
        self.vae_regularization_tracker = tf.keras.metrics.Mean(
            name="vae_regularization_loss"
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

        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.weighted_kl_loss_tracker = tf.keras.metrics.Mean(
            name="weighted_kl_loss"
        )
        self.decoder_accuracy_tracker = DecoderReconstructionR2(
            name="decoder_accuracy"
        )

        self.subject_loss_tracker = tf.keras.metrics.Mean(name="subject_loss")
        self.weighted_subject_loss_tracker = tf.keras.metrics.Mean(
            name="weighted_subject_loss"
        )
        self.subject_accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(
            name="subject_accuracy"
        )

        self.supcon_loss_tracker = tf.keras.metrics.Mean(name="supcon_loss")
        self.weighted_supcon_loss_tracker = tf.keras.metrics.Mean(
            name="weighted_supcon_loss"
        )
        self.supcon_valid_anchor_fraction_tracker = tf.keras.metrics.Mean(
            name="supcon_valid_anchor_fraction"
        )
        self.supcon_positive_pairs_tracker = tf.keras.metrics.Mean(
            name="supcon_positive_pairs"
        )

        self.accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(
            name="accuracy"
        )
        n_classes = int(getattr(self.classifier, "n_classes", 2))
        self.true_class_fraction_trackers = [
            tf.keras.metrics.Mean(name=f"true_class_{index}_fraction")
            for index in range(n_classes)
        ]
        self.predicted_class_fraction_trackers = [
            tf.keras.metrics.Mean(name=f"predicted_class_{index}_fraction")
            for index in range(n_classes)
        ]

    @property
    def metrics(self) -> list[tf.keras.metrics.Metric]:
        metrics: list[tf.keras.metrics.Metric] = [
            self.total_loss_tracker,
            self.classification_objective_tracker,
            self.vae_objective_tracker,
            self.classification_regularization_tracker,
            self.vae_regularization_tracker,
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
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.weighted_kl_loss_tracker,
            self.decoder_accuracy_tracker,
        ]
        if self.subject_adversarial_enabled:
            metrics.extend(
                [
                    self.subject_loss_tracker,
                    self.weighted_subject_loss_tracker,
                    self.subject_accuracy_tracker,
                ]
            )
        if self.use_supcon:
            metrics.extend(
                [
                    self.supcon_loss_tracker,
                    self.weighted_supcon_loss_tracker,
                    self.supcon_valid_anchor_fraction_tracker,
                    self.supcon_positive_pairs_tracker,
                ]
            )
        metrics.extend(
            [
                self.accuracy_tracker,
                *self.true_class_fraction_trackers,
                *self.predicted_class_fraction_trackers,
            ]
        )
        return metrics

    def compile(
        self,
        classification_optimizer: tf.keras.optimizers.Optimizer,
        vae_optimizer: tf.keras.optimizers.Optimizer,
        discriminator_optimizer: tf.keras.optimizers.Optimizer | None = None,
        **kwargs,
    ) -> None:
        """Compile with independent optimizers for the alternating phases."""
        if classification_optimizer is None or vae_optimizer is None:
            raise ValueError(
                "classification_optimizer and vae_optimizer are required."
            )
        kwargs.setdefault("jit_compile", False)
        super().compile(optimizer=classification_optimizer, **kwargs)
        self.classification_optimizer = classification_optimizer
        self.vae_optimizer = vae_optimizer
        if self.update_discriminator:
            if discriminator_optimizer is None:
                discriminator_optimizer = (
                    classification_optimizer.__class__.from_config(
                        classification_optimizer.get_config()
                    )
                )
            self.discriminator_optimizer = discriminator_optimizer
        else:
            self.discriminator_optimizer = None

    def get_compile_config(self) -> dict:
        """Serialize all optimizers used by the alternating train step."""
        config = super().get_compile_config()
        config.pop("optimizer", None)
        config.update(
            {
                "classification_optimizer": _serialize_keras_component(
                    self.classification_optimizer
                ),
                "vae_optimizer": _serialize_keras_component(self.vae_optimizer),
                "discriminator_optimizer": _serialize_keras_component(
                    self.discriminator_optimizer
                ),
            }
        )
        return config

    def compile_from_config(self, config: dict) -> None:
        """Restore the alternating optimizer configuration after loading."""
        config = dict(config)
        config.pop("optimizer", None)
        classification_optimizer = _deserialize_keras_component(
            config.pop("classification_optimizer")
        )
        vae_optimizer = _deserialize_keras_component(
            config.pop("vae_optimizer")
        )
        discriminator_optimizer = _deserialize_keras_component(
            config.pop("discriminator_optimizer", None)
        )
        self.compile(
            classification_optimizer=classification_optimizer,
            vae_optimizer=vae_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            **config,
        )

    def fit(self, *args, **kwargs):
        if not self.use_class_weight:
            kwargs.pop("class_weight", None)
        return super().fit(*args, **kwargs)

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
        self.subject_pooling = tf.keras.layers.GlobalAveragePooling1D(
            name="sts_subject_pool"
        )
        self.subject_gradient_reversal = GradientReversal(
            adversarial_weight=self.subject_adversarial_weight,
            name="sts_subject_gradient_reversal",
        )
        self.subject_hidden = tf.keras.layers.Dense(
            self.subject_hidden_units,
            activation=self.activation_name,
            name="sts_subject_hidden",
        )
        self.subject_dropout_layer = tf.keras.layers.Dropout(
            self.subject_dropout_rate,
            name="sts_subject_dropout",
        )
        self.subject_classifier = tf.keras.layers.Dense(
            self.n_subject_classes,
            activation=None,
            name="sts_subject_logits",
        )

    def prepare_fit_inputs(self, eeg_inputs, subject_ids):
        """Attach contiguous fold-local subject IDs to EEG inputs."""
        if not self.requires_subject_ids:
            return eeg_inputs

        eeg_array = np.asarray(eeg_inputs)
        subjects = np.asarray(subject_ids).reshape(-1)
        if len(eeg_array) != len(subjects):
            raise ValueError(
                "EEG samples and subject IDs must align; got "
                f"{len(eeg_array)} and {len(subjects)}."
            )
        unique_subjects = np.sort(np.unique(subjects))
        if self.subject_adversarial_enabled:
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
    def _split_eeg_and_subject_inputs(inputs):
        if isinstance(inputs, Mapping):
            if "eeg" not in inputs:
                raise ValueError("Input mappings must contain an 'eeg' key.")
            return inputs["eeg"], inputs.get("subject_id")
        return inputs, None

    @staticmethod
    def _flatten_labels(y) -> tf.Tensor:
        labels = tf.convert_to_tensor(y)
        if (
            labels.shape.rank == 2
            and labels.shape[-1] is not None
            and labels.shape[-1] > 1
        ):
            return tf.argmax(labels, axis=-1, output_type=tf.int32)
        return tf.cast(tf.reshape(labels, [-1]), tf.int32)

    @staticmethod
    def _reparameterize(z_mean: tf.Tensor, z_log_var: tf.Tensor) -> tf.Tensor:
        epsilon = tf.random.normal(tf.shape(z_mean), dtype=z_mean.dtype)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def _encode_fused(
        self,
        eeg_inputs: tf.Tensor,
        training: bool,
    ) -> dict[str, tf.Tensor]:
        temporal_sequence = self.temporal_encoder(eeg_inputs, training=training)
        spectral_sequence = self.spectral_encoder(eeg_inputs, training=training)
        if temporal_sequence.shape.rank != 3 or spectral_sequence.shape.rank != 3:
            raise ValueError(
                "Both STS encoders must return rank-3 latent sequences; got "
                f"{temporal_sequence.shape} and {spectral_sequence.shape}."
            )
        tf.debugging.assert_equal(
            tf.shape(temporal_sequence)[:2],
            tf.shape(spectral_sequence)[:2],
            message=(
                "BiLSTM and GCN sequences must have matching batch/time axes. "
                "Use the same t_down and temporal_pool_sizes."
            ),
        )
        fused = tf.concat(
            [temporal_sequence, spectral_sequence],
            axis=-1,
            name="sts_feature_concatenation",
        )
        fused = self.fusion_projection(fused)
        fused = self.fusion_normalization(fused)
        fused = self.fusion_activation(fused)
        fused = self.fusion_dropout(fused, training=training)

        z_mean = self.z_mean_projection(fused)
        raw_z_log_var = self.z_log_var_projection(fused)
        tf.debugging.assert_all_finite(z_mean, "STS z_mean contains NaN or Inf.")
        tf.debugging.assert_all_finite(
            raw_z_log_var,
            "STS unclipped z_log_var contains NaN or Inf.",
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

    def _classification_embedding(
        self,
        latent_sequence: tf.Tensor,
        training: bool,
    ) -> tf.Tensor:
        embedding = self.classification_pool(latent_sequence)
        embedding = self.classification_projection(embedding)
        embedding = self.classification_normalization(embedding)
        return self.classification_dropout(embedding, training=training)

    def _subject_head_forward(
        self,
        latent_sequence: tf.Tensor,
        training: bool,
    ) -> tf.Tensor:
        if self.subject_classifier is None:
            raise RuntimeError(
                "Subject head is not configured. Call prepare_fit_inputs(...) "
                "before fitting the fold."
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
        samples = (
            z_mean[tf.newaxis, ...]
            + tf.exp(0.5 * z_log_var)[tf.newaxis, ...] * epsilon
        )
        sample_shape = tf.shape(samples)
        flat_samples = tf.reshape(
            samples,
            [sample_shape[0] * sample_shape[1], sample_shape[2], sample_shape[3]],
        )
        flat_logits = self._subject_head_forward(flat_samples, training=training)
        logits_by_sample = tf.reshape(
            flat_logits,
            [sample_shape[0], sample_shape[1], self.n_subject_classes],
        )
        log_probabilities = tf.nn.log_softmax(logits_by_sample, axis=-1)
        return tf.reduce_logsumexp(log_probabilities, axis=0) - tf.math.log(
            tf.cast(sample_count, log_probabilities.dtype)
        )

    def call(
        self,
        inputs,
        training: bool = False,
        sample_latent: bool | None = None,
        include_reconstruction: bool = True,
        include_subject_adversarial: bool | None = None,
    ) -> dict[str, tf.Tensor]:
        eeg_inputs, _subject_ids = self._split_eeg_and_subject_inputs(inputs)
        eeg_inputs = tf.convert_to_tensor(eeg_inputs, dtype=tf.float32)
        if eeg_inputs.shape.rank != 3:
            raise ValueError(
                "JointSTSModel expects EEG windows shaped "
                f"(batch, timesteps, features); got {eeg_inputs.shape}."
            )
        if sample_latent is None:
            sample_latent = bool(training)
        if include_subject_adversarial is None:
            include_subject_adversarial = (
                self.subject_adversarial_enabled and training is True
            )

        encoded = self._encode_fused(eeg_inputs, training=training)
        classification_latent = self._classification_embedding(
            encoded["z_mean"],
            training=training,
        )
        logits = self.classifier(classification_latent, training=training)
        probabilities = tf.nn.softmax(logits, axis=-1)

        outputs = {
            **encoded,
            # Compatibility alias used by the existing prediction diagnostics.
            "encoder_output": encoded["fused_sequence"],
            "classification_latent_sequence": encoded["z_mean"],
            "classification_latent": classification_latent,
            "logits": logits,
            "probabilities": probabilities,
        }
        n_classes = int(getattr(self.classifier, "n_classes", 2))
        if n_classes == 2:
            outputs["logit_margin"] = logits[:, 1] - logits[:, 0]
        else:
            top_logits = tf.math.top_k(logits, k=2, sorted=True).values
            outputs["logit_margin"] = top_logits[:, 0] - top_logits[:, 1]

        if include_reconstruction:
            latent_for_decoder = (
                self._reparameterize(encoded["z_mean"], encoded["z_log_var"])
                if sample_latent
                else encoded["z_mean"]
            )
            outputs["latent_sequence"] = latent_for_decoder
            outputs["reconstruction"] = self.decoder(
                latent_for_decoder,
                training=training,
            )

        if include_subject_adversarial:
            if not self.subject_adversarial_enabled:
                raise ValueError(
                    "include_subject_adversarial=True, but the branch is disabled."
                )
            outputs["subject_logits"] = self._subject_logits_from_posterior(
                z_mean=encoded["z_mean"],
                z_log_var=encoded["z_log_var"],
                training=training,
            )
        return outputs

    @staticmethod
    def _supervised_contrastive_loss(
        embeddings: tf.Tensor,
        labels: tf.Tensor,
        temperature: float,
        subject_ids: tf.Tensor | None = None,
        cross_subject_only: bool = True,
        sample_weight: tf.Tensor | None = None,
    ) -> dict[str, tf.Tensor]:
        """Compute SupCon, optionally using only cross-subject positives."""
        embeddings = tf.convert_to_tensor(embeddings)
        if embeddings.shape.rank != 2:
            raise ValueError("SupCon embeddings must be rank 2.")
        labels = tf.cast(tf.reshape(labels, [-1]), tf.int32)
        tf.debugging.assert_equal(tf.shape(embeddings)[0], tf.shape(labels)[0])

        normalized = tf.math.l2_normalize(embeddings, axis=-1, epsilon=1e-12)
        similarity = tf.matmul(normalized, normalized, transpose_b=True)
        similarity /= tf.cast(temperature, similarity.dtype)

        batch_size = tf.shape(similarity)[0]
        non_self = tf.logical_not(tf.eye(batch_size, dtype=tf.bool))
        same_label = tf.equal(labels[:, tf.newaxis], labels[tf.newaxis, :])

        if cross_subject_only:
            if subject_ids is None:
                zero = tf.zeros((), dtype=embeddings.dtype)
                return {
                    "loss": zero,
                    "valid_anchor_fraction": zero,
                    "positive_pairs": zero,
                }
            subject_ids = tf.cast(tf.reshape(subject_ids, [-1]), tf.int32)
            tf.debugging.assert_equal(tf.shape(subject_ids)[0], batch_size)
            same_subject = tf.equal(
                subject_ids[:, tf.newaxis],
                subject_ids[tf.newaxis, :],
            )
            positive_mask = tf.logical_and(
                non_self,
                tf.logical_and(same_label, tf.logical_not(same_subject)),
            )
            ignored_same_subject_positive = tf.logical_and(
                non_self,
                tf.logical_and(same_label, same_subject),
            )
            denominator_mask = tf.logical_and(
                non_self,
                tf.logical_not(ignored_same_subject_positive),
            )
        else:
            positive_mask = tf.logical_and(non_self, same_label)
            denominator_mask = non_self

        large_negative = tf.cast(-1e9, similarity.dtype)
        masked_similarity = tf.where(
            denominator_mask,
            similarity,
            large_negative,
        )
        row_max = tf.stop_gradient(
            tf.reduce_max(masked_similarity, axis=1, keepdims=True)
        )
        stabilized = similarity - row_max
        exp_similarity = tf.exp(stabilized) * tf.cast(
            denominator_mask,
            stabilized.dtype,
        )
        log_denominator = tf.math.log(
            tf.reduce_sum(exp_similarity, axis=1, keepdims=True)
            + tf.cast(tf.keras.backend.epsilon(), stabilized.dtype)
        )
        log_probability = stabilized - log_denominator

        positive_count = tf.reduce_sum(
            tf.cast(positive_mask, stabilized.dtype),
            axis=1,
        )
        mean_positive_log_probability = tf.math.divide_no_nan(
            tf.reduce_sum(
                log_probability * tf.cast(positive_mask, stabilized.dtype),
                axis=1,
            ),
            positive_count,
        )
        valid_anchor = positive_count > 0.0
        per_anchor_loss = tf.where(
            valid_anchor,
            -mean_positive_log_probability,
            tf.zeros_like(mean_positive_log_probability),
        )
        anchor_weights = tf.cast(valid_anchor, per_anchor_loss.dtype)
        if sample_weight is not None:
            weights = tf.cast(tf.reshape(sample_weight, [-1]), per_anchor_loss.dtype)
            tf.debugging.assert_equal(tf.shape(weights)[0], batch_size)
            anchor_weights *= weights
        loss = tf.math.divide_no_nan(
            tf.reduce_sum(per_anchor_loss * anchor_weights),
            tf.reduce_sum(anchor_weights),
        )
        return {
            "loss": loss,
            "valid_anchor_fraction": tf.reduce_mean(
                tf.cast(valid_anchor, embeddings.dtype)
            ),
            "positive_pairs": tf.reduce_sum(
                tf.cast(positive_mask, embeddings.dtype)
            ),
        }

    def _regularization_loss(self, dtype: tf.dtypes.DType) -> tf.Tensor:
        if not self.losses:
            return tf.zeros((), dtype=dtype)
        return tf.add_n([tf.cast(loss, dtype) for loss in self.losses])

    def _classification_losses(
        self,
        x,
        y_flat: tf.Tensor,
        training: bool,
        sample_weight=None,
    ) -> tuple[dict[str, tf.Tensor], dict[str, tf.Tensor]]:
        eeg_inputs, subject_ids = self._split_eeg_and_subject_inputs(x)
        subject_enabled = self.subject_adversarial_enabled and subject_ids is not None
        outputs = self(
            eeg_inputs,
            training=training,
            sample_latent=False,
            include_reconstruction=False,
            include_subject_adversarial=subject_enabled,
        )
        tf.debugging.assert_equal(
            tf.shape(y_flat)[0],
            tf.shape(outputs["logits"])[0],
        )

        vc_losses = self.classifier.vc_loss_components(
            mh=outputs["classification_latent"],
            y=y_flat,
            alpha=self.vc_alpha,
            beta=self.vc_beta,
            gamma=self.vc_gamma,
            lambda_=self.vc_lambda,
            logits=outputs["logits"],
            sample_weight=sample_weight,
        )

        zero = tf.zeros((), dtype=outputs["logits"].dtype)
        if subject_enabled:
            subject_ids = tf.cast(tf.reshape(subject_ids, [-1]), tf.int32)
            subject_loss_per_sample = (
                self.reconstruction_loss_fn.compute_subject_loss(
                    subject_true=subject_ids,
                    subject_pred=outputs["subject_logits"],
                )
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

        if (
            self.use_supcon
            and self.supcon_cross_subject_only
            and training
            and subject_ids is None
        ):
            raise ValueError(
                "Cross-subject SupCon requires subject IDs during fitting. "
                "Use prepare_fit_inputs(...)."
            )
        if self.use_supcon:
            supcon = self._supervised_contrastive_loss(
                embeddings=outputs["classification_latent"],
                labels=y_flat,
                temperature=self.supcon_temperature,
                subject_ids=subject_ids,
                cross_subject_only=self.supcon_cross_subject_only,
                sample_weight=sample_weight,
            )
            weighted_supcon_loss = (
                tf.cast(self.supcon_weight, supcon["loss"].dtype)
                * supcon["loss"]
            )
        else:
            supcon = {
                "loss": zero,
                "valid_anchor_fraction": zero,
                "positive_pairs": zero,
            }
            weighted_supcon_loss = zero

        regularization_loss = self._regularization_loss(outputs["logits"].dtype)
        objective = (
            tf.cast(self.classification_loss_weight, vc_losses["total_loss"].dtype)
            * vc_losses["total_loss"]
            + weighted_subject_loss
            + weighted_supcon_loss
            + regularization_loss
        )
        losses = {
            "objective": objective,
            "regularization_loss": regularization_loss,
            "vc_loss": vc_losses["total_loss"],
            "vc_cross_entropy": vc_losses["cross_entropy"],
            "weighted_vc_cross_entropy": vc_losses["weighted_cross_entropy"],
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
            "subject_loss": subject_loss,
            "weighted_subject_loss": weighted_subject_loss,
            "supcon_loss": supcon["loss"],
            "weighted_supcon_loss": weighted_supcon_loss,
            "supcon_valid_anchor_fraction": supcon["valid_anchor_fraction"],
            "supcon_positive_pairs": supcon["positive_pairs"],
        }
        return losses, outputs

    def _vae_losses(
        self,
        x,
        training: bool,
    ) -> tuple[dict[str, tf.Tensor], dict[str, tf.Tensor]]:
        eeg_inputs, _subject_ids = self._split_eeg_and_subject_inputs(x)
        eeg_inputs = tf.convert_to_tensor(eeg_inputs, dtype=tf.float32)
        outputs = self(
            eeg_inputs,
            training=training,
            sample_latent=training,
            include_reconstruction=True,
            include_subject_adversarial=False,
        )
        batch_size = tf.shape(eeg_inputs)[0]
        z_mean_flat = tf.reshape(outputs["z_mean"], [batch_size, -1])
        z_log_var_flat = tf.reshape(outputs["z_log_var"], [batch_size, -1])
        vae_components = self.reconstruction_loss_fn(
            x_true=eeg_inputs,
            x_pred=outputs["reconstruction"],
            z_mean=z_mean_flat,
            z_log_var=z_log_var_flat,
            include_subject_loss=False,
        )
        vae_loss = (
            vae_components["reconstruction_loss"]
            + vae_components["weighted_kl_loss"]
        )
        regularization_loss = self._regularization_loss(vae_loss.dtype)
        objective = (
            tf.cast(self.vae_loss_weight, vae_loss.dtype) * vae_loss
            + regularization_loss
        )
        return (
            {
                "objective": objective,
                "regularization_loss": regularization_loss,
                "vae_loss": vae_loss,
                "reconstruction_loss": vae_components["reconstruction_loss"],
                "kl_loss": vae_components["kl_loss"],
                "weighted_kl_loss": vae_components["weighted_kl_loss"],
            },
            outputs,
        )

    def _discriminator_variables(self) -> list[tf.Variable]:
        if not hasattr(self.classifier, "disc_w"):
            return []
        return [self.classifier.disc_w, self.classifier.disc_b]

    def _classification_variables(self) -> list[tf.Variable]:
        components: list[Any] = [
            self.temporal_encoder,
            self.spectral_encoder,
            self.fusion_projection,
            self.fusion_normalization,
            self.fusion_activation,
            self.fusion_dropout,
            self.z_mean_projection,
            self.z_log_var_projection,
            self.classification_pool,
            self.classification_projection,
            self.classification_normalization,
            self.classification_dropout,
            self.classifier,
        ]
        if self.subject_classifier is not None:
            components.extend(
                [
                    self.subject_pooling,
                    self.subject_gradient_reversal,
                    self.subject_hidden,
                    self.subject_dropout_layer,
                    self.subject_classifier,
                ]
            )
        variables: list[tf.Variable] = []
        for component in components:
            variables.extend(component.trainable_variables)
        discriminator_ids = {id(variable) for variable in self._discriminator_variables()}
        return [
            variable
            for variable in _deduplicate_variables(variables)
            if id(variable) not in discriminator_ids
        ]

    def _vae_variables(self) -> list[tf.Variable]:
        components = [
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

    @staticmethod
    def _apply_gradients(
        optimizer: tf.keras.optimizers.Optimizer,
        gradients,
        variables,
    ) -> None:
        pairs = [
            (gradient, variable)
            for gradient, variable in zip(gradients, variables)
            if gradient is not None
        ]
        if pairs:
            optimizer.apply_gradients(pairs)

    @staticmethod
    def _unpack_data(data):
        if not isinstance(data, tuple):
            raise ValueError("Expected data as (x, y) or (x, y, sample_weight).")
        if len(data) == 2:
            return data[0], data[1], None
        if len(data) == 3:
            return data[0], data[1], data[2]
        raise ValueError("Expected data as (x, y) or (x, y, sample_weight).")

    def _update_trackers(
        self,
        classification_losses: dict[str, tf.Tensor],
        classification_outputs: dict[str, tf.Tensor],
        vae_losses: dict[str, tf.Tensor],
        vae_outputs: dict[str, tf.Tensor],
        x,
        y_flat: tf.Tensor,
        sample_weight,
        discriminator_loss: tf.Tensor,
    ) -> None:
        total_loss = classification_losses["objective"] + vae_losses["objective"]
        self.total_loss_tracker.update_state(total_loss)
        self.classification_objective_tracker.update_state(
            classification_losses["objective"]
        )
        self.vae_objective_tracker.update_state(vae_losses["objective"])
        self.classification_regularization_tracker.update_state(
            classification_losses["regularization_loss"]
        )
        self.vae_regularization_tracker.update_state(
            vae_losses["regularization_loss"]
        )

        self.vc_loss_tracker.update_state(classification_losses["vc_loss"])
        self.vc_cross_entropy_tracker.update_state(
            classification_losses["vc_cross_entropy"]
        )
        self.weighted_vc_cross_entropy_tracker.update_state(
            classification_losses["weighted_vc_cross_entropy"]
        )
        self.vc_latent_kl_tracker.update_state(
            classification_losses["vc_latent_kl"]
        )
        self.weighted_vc_latent_kl_tracker.update_state(
            classification_losses["weighted_vc_latent_kl"]
        )
        self.vc_discriminator_kl_tracker.update_state(
            classification_losses["vc_discriminator_kl"]
        )
        self.weighted_vc_discriminator_kl_tracker.update_state(
            classification_losses["weighted_vc_discriminator_kl"]
        )
        self.vc_class_prior_kl_tracker.update_state(
            classification_losses["vc_class_prior_kl"]
        )
        self.weighted_vc_class_prior_kl_tracker.update_state(
            classification_losses["weighted_vc_class_prior_kl"]
        )
        self.vc_discriminator_loss_tracker.update_state(discriminator_loss)

        self.reconstruction_loss_tracker.update_state(
            vae_losses["reconstruction_loss"]
        )
        self.kl_loss_tracker.update_state(vae_losses["kl_loss"])
        self.weighted_kl_loss_tracker.update_state(
            vae_losses["weighted_kl_loss"]
        )
        eeg_inputs, _subject_ids = self._split_eeg_and_subject_inputs(x)
        self.decoder_accuracy_tracker.update_state(
            eeg_inputs,
            vae_outputs["reconstruction"],
        )

        if self.subject_adversarial_enabled:
            self.subject_loss_tracker.update_state(
                classification_losses["subject_loss"]
            )
            self.weighted_subject_loss_tracker.update_state(
                classification_losses["weighted_subject_loss"]
            )
            if (
                "subject_targets" in classification_outputs
                and "subject_logits" in classification_outputs
            ):
                self.subject_accuracy_tracker.update_state(
                    classification_outputs["subject_targets"],
                    classification_outputs["subject_logits"],
                )

        if self.use_supcon:
            self.supcon_loss_tracker.update_state(
                classification_losses["supcon_loss"]
            )
            self.weighted_supcon_loss_tracker.update_state(
                classification_losses["weighted_supcon_loss"]
            )
            self.supcon_valid_anchor_fraction_tracker.update_state(
                classification_losses["supcon_valid_anchor_fraction"]
            )
            self.supcon_positive_pairs_tracker.update_state(
                classification_losses["supcon_positive_pairs"]
            )

        self.accuracy_tracker.update_state(
            y_flat,
            classification_outputs["logits"],
            sample_weight=sample_weight,
        )
        predicted = tf.argmax(
            classification_outputs["logits"],
            axis=-1,
            output_type=tf.int32,
        )
        for class_index, tracker in enumerate(self.true_class_fraction_trackers):
            tracker.update_state(tf.cast(tf.equal(y_flat, class_index), tf.float32))
        for class_index, tracker in enumerate(
            self.predicted_class_fraction_trackers
        ):
            tracker.update_state(
                tf.cast(tf.equal(predicted, class_index), tf.float32)
            )

    def _metric_results(self) -> dict[str, tf.Tensor]:
        return {metric.name: metric.result() for metric in self.metrics}

    def train_step(self, data) -> dict[str, tf.Tensor]:
        if self.classification_optimizer is None or self.vae_optimizer is None:
            raise RuntimeError("Call model.compile(...) before model.fit(...).")
        x, y, sample_weight = self._unpack_data(data)
        y_flat = self._flatten_labels(y)

        classification_losses = classification_outputs = None
        discriminator_loss = tf.zeros((), dtype=tf.float32)
        for _ in range(self.classification_steps_per_batch):
            with tf.GradientTape() as classification_tape:
                classification_losses, classification_outputs = (
                    self._classification_losses(
                        x=x,
                        y_flat=y_flat,
                        training=True,
                        sample_weight=sample_weight,
                    )
                )
            classification_variables = self._classification_variables()
            classification_gradients = classification_tape.gradient(
                classification_losses["objective"],
                classification_variables,
            )
            self._apply_gradients(
                self.classification_optimizer,
                classification_gradients,
                classification_variables,
            )

            discriminator_variables = self._discriminator_variables()
            if self.update_discriminator and discriminator_variables:
                if self.discriminator_optimizer is None:
                    raise RuntimeError(
                        "Discriminator updates are enabled without an optimizer."
                    )
                with tf.GradientTape() as discriminator_tape:
                    discriminator_outputs = self(
                        self._split_eeg_and_subject_inputs(x)[0],
                        training=True,
                        sample_latent=False,
                        include_reconstruction=False,
                        include_subject_adversarial=False,
                    )
                    discriminator_loss = self.classifier.discriminator_loss(
                        tf.stop_gradient(
                            discriminator_outputs["classification_latent"]
                        ),
                        y_flat,
                    )
                discriminator_gradients = discriminator_tape.gradient(
                    discriminator_loss,
                    discriminator_variables,
                )
                self._apply_gradients(
                    self.discriminator_optimizer,
                    discriminator_gradients,
                    discriminator_variables,
                )

        vae_losses = vae_outputs = None
        for _ in range(self.vae_steps_per_batch):
            with tf.GradientTape() as vae_tape:
                vae_losses, vae_outputs = self._vae_losses(x=x, training=True)
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

        self._update_trackers(
            classification_losses=classification_losses,
            classification_outputs=classification_outputs,
            vae_losses=vae_losses,
            vae_outputs=vae_outputs,
            x=x,
            y_flat=y_flat,
            sample_weight=sample_weight,
            discriminator_loss=discriminator_loss,
        )
        return self._metric_results()

    def test_step(self, data) -> dict[str, tf.Tensor]:
        x, y, sample_weight = self._unpack_data(data)
        y_flat = self._flatten_labels(y)
        classification_losses, classification_outputs = (
            self._classification_losses(
                x=x,
                y_flat=y_flat,
                training=False,
                sample_weight=sample_weight,
            )
        )
        vae_losses, vae_outputs = self._vae_losses(x=x, training=False)
        discriminator_loss = tf.zeros((), dtype=classification_losses["objective"].dtype)
        self._update_trackers(
            classification_losses=classification_losses,
            classification_outputs=classification_outputs,
            vae_losses=vae_losses,
            vae_outputs=vae_outputs,
            x=x,
            y_flat=y_flat,
            sample_weight=sample_weight,
            discriminator_loss=discriminator_loss,
        )
        return self._metric_results()

    def predict_step(self, data):
        inputs = data[0] if isinstance(data, tuple) else data
        outputs = self(
            inputs,
            training=False,
            sample_latent=False,
            include_reconstruction=False,
            include_subject_adversarial=False,
        )
        return outputs["logits"]

    def predict_diagnostics(
        self,
        inputs,
        batch_size: int | None = None,
    ) -> dict[str, tf.Tensor]:
        """Return deterministic tensors expected by EEGProc diagnostics."""
        eeg_inputs, _subject_ids = self._split_eeg_and_subject_inputs(inputs)
        eeg_inputs = tf.convert_to_tensor(eeg_inputs, dtype=tf.float32)
        n_samples = int(tf.shape(eeg_inputs)[0].numpy())
        effective_batch_size = n_samples if batch_size is None else int(batch_size)
        if effective_batch_size < 1:
            raise ValueError("batch_size must be positive.")

        keys = (
            "encoder_output",
            "temporal_sequence",
            "spectral_sequence",
            "fused_sequence",
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
            outputs = self(
                eeg_inputs[start : start + effective_batch_size],
                training=False,
                sample_latent=False,
                include_reconstruction=False,
                include_subject_adversarial=False,
            )
            for key in keys:
                collected[key].append(outputs[key])
        return {key: tf.concat(values, axis=0) for key, values in collected.items()}

    def predict_mc_probabilities(
        self,
        inputs,
        n_samples: int = 30,
        seed: int | tuple[int, int] | None = None,
    ) -> dict[str, tf.Tensor]:
        """Average predictions over samples from the fused posterior."""
        if n_samples < 1:
            raise ValueError("n_samples must be at least 1.")
        eeg_inputs, _subject_ids = self._split_eeg_and_subject_inputs(inputs)
        eeg_inputs = tf.convert_to_tensor(eeg_inputs, dtype=tf.float32)
        encoded = self._encode_fused(eeg_inputs, training=False)
        z_mean = encoded["z_mean"]
        z_log_var = encoded["z_log_var"]
        epsilon_shape = tf.concat(
            [tf.constant([int(n_samples)], dtype=tf.int32), tf.shape(z_mean)],
            axis=0,
        )
        if seed is None:
            epsilon = tf.random.normal(epsilon_shape, dtype=z_mean.dtype)
        else:
            stateless_seed = (
                tf.constant([seed, 0], dtype=tf.int32)
                if isinstance(seed, int)
                else tf.constant(seed, dtype=tf.int32)
            )
            epsilon = tf.random.stateless_normal(
                epsilon_shape,
                seed=stateless_seed,
                dtype=z_mean.dtype,
            )
        samples = (
            z_mean[tf.newaxis, ...]
            + tf.exp(0.5 * z_log_var)[tf.newaxis, ...] * epsilon
        )
        shape = tf.shape(samples)
        flat_samples = tf.reshape(
            samples,
            [shape[0] * shape[1], shape[2], shape[3]],
        )
        flat_embeddings = self._classification_embedding(
            flat_samples,
            training=False,
        )
        flat_logits = self.classifier(flat_embeddings, training=False)
        flat_probabilities = tf.nn.softmax(flat_logits, axis=-1)
        n_classes = tf.shape(flat_probabilities)[-1]
        probability_samples = tf.reshape(
            flat_probabilities,
            [shape[0], shape[1], n_classes],
        )
        return {
            "mean_probabilities": tf.reduce_mean(probability_samples, axis=0),
            "probability_samples": probability_samples,
            "z_mean": z_mean,
            "z_log_var": z_log_var,
        }

    def get_adjacency_matrices(self) -> dict[str, dict[str, tf.Tensor]]:
        """Expose learned encoder and decoder electrode graphs."""
        output: dict[str, dict[str, tf.Tensor]] = {}
        if hasattr(self.spectral_encoder, "get_adjacency_matrices"):
            output["spectral_encoder"] = (
                self.spectral_encoder.get_adjacency_matrices()
            )
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
                "classifier": _serialize_keras_component(self.classifier),
                "fusion_dim": self.fusion_dim,
                "latent_features": self.latent_features,
                "classification_hidden_units": self.classification_hidden_units,
                "fusion_dropout": self.fusion_dropout_rate,
                "classification_dropout": self.classification_dropout_rate,
                "activation": self.activation_name,
                "classification_loss_weight": self.classification_loss_weight,
                "vae_loss_weight": self.vae_loss_weight,
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
                "use_subject_adversarial": self.subject_adversarial_enabled,
                "n_subject_classes": self.n_subject_classes,
                "subject_adversarial_weight": self.subject_adversarial_weight,
                "subject_loss_weight": self.subject_loss_weight,
                "subject_hidden_units": self.subject_hidden_units,
                "subject_dropout": self.subject_dropout_rate,
                "subject_latent_mode": self.subject_latent_mode,
                "subject_mc_samples": self.subject_mc_samples,
                "use_supcon": self.use_supcon,
                "supcon_weight": self.supcon_weight,
                "supcon_temperature": self.supcon_temperature,
                "supcon_cross_subject_only": self.supcon_cross_subject_only,
                "classification_steps_per_batch": (
                    self.classification_steps_per_batch
                ),
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
            "classifier",
            "reconstruction_loss_fn",
        ):
            if key in config:
                config[key] = _deserialize_keras_component(config[key])
        return cls(**config)


def build_joint_sts_model(
    input_shape: tuple[int, int],
    *,
    n_classes: int = 2,
    n_channels: int = 14,
    n_bands: int = 4,
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
    classification_hidden_units: int = 64,
    classification_dropout: float = 0.30,
    activation: str = "relu",
    classifier_head: str = "dense",
    classifier_kwargs: dict | None = None,
    label_smoothing: float = 0.0,
    classification_loss_weight: float = 1.0,
    vae_loss_weight: float = 1.0,
    vae_beta: float = 0.30,
    vc_alpha: float = 1.0,
    vc_beta: float = 0.0,
    vc_gamma: float = 0.0,
    vc_lambda: float = 0.0,
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
    use_supcon: bool = False,
    supcon_weight: float = 0.03,
    supcon_temperature: float = 0.10,
    supcon_cross_subject_only: bool = True,
    classification_steps_per_batch: int = 1,
    vae_steps_per_batch: int = 1,
    optimizer_name: str = "adamw",
    classification_learning_rate: float = 1e-4,
    vae_learning_rate: float = 5e-5,
    discriminator_learning_rate: float | None = None,
    weight_decay: float = 1e-4,
    reconstruction_loss: str = "mse",
    model_name: str = "joint_sts_model",
) -> JointSTSModel:
    """Build and compile the fused STS model.

    The default classifier is ``DenseClassifier`` to match the requested dense
    softmax design. ``classifier_head='hybrid'`` or ``'variational'`` retains
    the existing VC regularizers without changing the alternating optimizer.
    """
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
        lstm_units=bilstm_units,
        n_bilstm_layers=n_bilstm_layers,
        dropout=bilstm_dropout,
        temporal_pool_sizes=pools,
        t_down=t_down,
        emb_dim=temporal_emb_dim,
        name="sts_spatiotemporal_bilstm",
    )
    spectral_encoder = GCNEncoder(
        timesteps=timesteps,
        t_down=t_down,
        n_channels=n_channels,
        n_bands=n_bands,
        gcn_units=tuple(int(value) for value in gcn_units),
        temporal_pool_sizes=pools,
        emb_dim=int(spectral_emb_dim),
        dropout=float(gcn_dropout),
        activation=gcn_activation,
        use_batch_norm=bool(gcn_use_batch_norm),
        graph_self_loop_bias=float(graph_self_loop_bias),
        graph_identity_mix=float(graph_identity_mix),
        graph_adjacency_reg_weight=float(graph_adjacency_reg_weight),
        name="sts_spatiospectral_gcn",
    )
    decoder = GCNDecoder.from_encoder(
        spectral_encoder,
        name="sts_fused_graph_decoder",
    )

    classifier_config = {
        "n_classes": int(n_classes),
        "label_smoothing": float(label_smoothing),
        "name": f"sts_{classifier_head}_classifier",
    }
    classifier_config.update(classifier_kwargs or {})
    classifier_head = str(classifier_head).lower()
    if classifier_head == "dense":
        classifier = DenseClassifier(**classifier_config)
    elif classifier_head == "hybrid":
        classifier = HybridClassifier(**classifier_config)
    elif classifier_head == "variational":
        classifier = VariationalClassifier(**classifier_config)
    else:
        raise ValueError(
            "classifier_head must be 'dense', 'hybrid', or 'variational'."
        )

    reconstruction_loss_fn = VariationalAutoencoderLoss(
        reconstruction=reconstruction_loss,
        beta=float(vae_beta),
        feature_reduction="mean",
        kl_reduction="mean",
        log_var_clip_min=-20.0,
        log_var_clip_max=20.0,
    )
    model = JointSTSModel(
        temporal_encoder=temporal_encoder,
        spectral_encoder=spectral_encoder,
        decoder=decoder,
        classifier=classifier,
        fusion_dim=fusion_dim,
        latent_features=latent_features,
        classification_hidden_units=classification_hidden_units,
        fusion_dropout=fusion_dropout,
        classification_dropout=classification_dropout,
        activation=activation,
        classification_loss_weight=classification_loss_weight,
        vae_loss_weight=vae_loss_weight,
        vae_beta=vae_beta,
        reconstruction_loss_fn=reconstruction_loss_fn,
        vc_alpha=vc_alpha,
        vc_beta=vc_beta,
        vc_gamma=vc_gamma,
        vc_lambda=vc_lambda,
        update_discriminator=update_discriminator,
        use_class_weight=use_class_weight,
        use_subject_adversarial=use_subject_adversarial,
        n_subject_classes=n_subject_classes,
        subject_adversarial_weight=subject_adversarial_weight,
        subject_loss_weight=subject_loss_weight,
        subject_hidden_units=subject_hidden_units,
        subject_dropout=subject_dropout,
        subject_latent_mode=subject_latent_mode,
        subject_mc_samples=subject_mc_samples,
        use_supcon=use_supcon,
        supcon_weight=supcon_weight,
        supcon_temperature=supcon_temperature,
        supcon_cross_subject_only=supcon_cross_subject_only,
        classification_steps_per_batch=classification_steps_per_batch,
        vae_steps_per_batch=vae_steps_per_batch,
        name=model_name,
    )

    classification_optimizer = _build_optimizer(
        optimizer_name=optimizer_name,
        learning_rate=classification_learning_rate,
        weight_decay=weight_decay,
    )
    vae_optimizer = _build_optimizer(
        optimizer_name=optimizer_name,
        learning_rate=vae_learning_rate,
        weight_decay=weight_decay,
    )
    discriminator_optimizer = None
    if update_discriminator:
        discriminator_optimizer = _build_optimizer(
            optimizer_name=optimizer_name,
            learning_rate=(
                classification_learning_rate
                if discriminator_learning_rate is None
                else discriminator_learning_rate
            ),
            weight_decay=weight_decay,
        )
    model.compile(
        classification_optimizer=classification_optimizer,
        vae_optimizer=vae_optimizer,
        discriminator_optimizer=discriminator_optimizer,
    )
    return model


__all__ = [
    "DecoderReconstructionR2",
    "JointSTSModel",
    "build_joint_sts_model",
    "build_spatiotemporal_bilstm_encoder",
]
