"""SIC: Subject Invariant Calibrator.

MTLFuseNet GCN/GRU -> temporal BiLSTM -> variational z -> dense-softmax.

Architecture
------------
For each EEG window::

    channel-band EEG
        -> MTLFuseNet-style fixed-MI shared GCN
        -> spectral GRU across frequency bands
        -> temporal BiLSTM
        -> q(z | x) = N(z_mean, diag(exp(z_log_var)))

The pooled z representation feeds a conventional dense -> softmax emotion
classifier.  EEGProc's existing VariationalClassifier is used as an auxiliary
VC target/regularizer on the dense classifier embedding; the dense logits are
still the actual prediction logits.

The generative path uses EEGProc's existing graph-aware MTL decoder::

    z sequence
        -> temporal projection / upsampling
        -> fixed-MI MTL-style graph decoding
        -> reconstructed channel-band EEG

The decoder is intentionally simpler than the encoder: it is a reconstruction
module, not a claimed mathematical inverse of the GCN/GRU/BiLSTM encoder.

Subject invariance can be imposed directly on pooled z with ordinary gradient
reversal.  There is no adversarial takeover/recovery controller: emotion/VC,
VAE, and (when enabled) subject-adversarial objectives are optimized together
in the same source update.

V-REx can additionally regularize source training.  Each source subject present
in a minibatch is treated as an environment.  SIC computes the classification
cross-entropy risk separately for each subject and adds the variance of those
risks to the ordinary source objective.  This uses the same batched forward
pass and a single backward/optimizer step.

Subject calibration
-------------------
``prepare_for_subject_calibration`` freezes the complete representation,
posterior, decoder, subject adversary, and VC target parameters.  It then
unfreezes only the last ``calibration_unfreeze_layers`` prediction layers:

    1 -> softmax/logits only
    2 -> last dense hidden block + softmax
    3 -> last two dense hidden blocks + softmax
    ...

The calibration train step uses the dense-softmax classification objective and,
when enabled, the same frozen VC target.  Therefore VC regularization can shape
any unfrozen dense hidden representation while a softmax-only calibration
reduces naturally to fitting the output decision boundary.

This file is designed for ``cross_val.subject_calibration_cv``.  Its builder
accepts ``training_features`` and computes the fixed MI adjacency from source
subjects only, avoiding target-subject leakage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import tensorflow as tf

try:
    from ...supervised.variational_classifier import VariationalClassifier
    from ...unsupervised.GNN.GCNMTL import (
        GCNMTLEncoder,
        GCNMTLDecoder,
        compute_mtl_shared_mi_adjacency,
    )
except ImportError:
    from eegproc.deep_learning.supervised.variational_classifier import (
        VariationalClassifier,
    )
    from eegproc.deep_learning.unsupervised.GNN.GCNMTL import (
        GCNMTLEncoder,
        GCNMTLDecoder,
        compute_mtl_shared_mi_adjacency,
    )


SIC_BUILDER_API_VERSION = 4
JOINT_V6_BUILDER_API_VERSION = SIC_BUILDER_API_VERSION


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


def _as_positive_tuple(name: str, values: Sequence[int]) -> tuple[int, ...]:
    output = tuple(int(value) for value in values)
    if not output or any(value < 1 for value in output):
        raise ValueError(f"{name} must contain positive integers; got {values}.")
    return output


def _resolve_temporal_pool_sizes(
    temporal_pool_sizes: Sequence[int] | None,
    t_down: int,
) -> tuple[int, ...]:
    t_down = int(t_down)
    if t_down < 1:
        raise ValueError("t_down must be >= 1.")
    if temporal_pool_sizes is None:
        pools = () if t_down == 1 else (t_down,)
    else:
        pools = tuple(int(value) for value in temporal_pool_sizes)
    if any(value < 1 for value in pools):
        raise ValueError("temporal_pool_sizes values must be >= 1.")
    effective = int(np.prod(pools, dtype=np.int64)) if pools else 1
    if effective != t_down:
        raise ValueError(
            f"t_down={t_down}, but temporal_pool_sizes={pools} gives {effective}."
        )
    return pools


def _deduplicate_variables(variables):
    seen: set[int] = set()
    output = []
    for variable in variables:
        identifier = id(variable)
        if identifier not in seen:
            seen.add(identifier)
            output.append(variable)
    return output


def _serialize_keras_object(value):
    if value is None:
        return None
    return tf.keras.utils.serialize_keras_object(value)


def _deserialize_keras_object(value):
    if value is None:
        return None
    return tf.keras.utils.deserialize_keras_object(value)


def _source_only_mi_adjacency(
    training_features,
    *,
    n_channels: int,
    n_bands: int,
    n_neighbors: int,
    random_state: int,
    zero_diagonal: bool,
    band_reduction: str,
    max_observations: int | None,
) -> np.ndarray:
    """Compute one fixed MI graph from source-training features only.

    ``training_features`` may be rank 3 (windows, time, features) or rank 4
    (trials, windows, time, features).  All leading axes are observations for
    MI estimation.  Optional row subsampling keeps this step practical.
    """
    x = np.asarray(training_features, dtype=np.float32)
    expected_features = int(n_channels) * int(n_bands)
    if x.ndim < 2 or x.shape[-1] != expected_features:
        raise ValueError(
            "training_features must end in n_channels*n_bands features; "
            f"got {x.shape}, expected last dimension {expected_features}."
        )
    observations = x.reshape(-1, expected_features)
    if max_observations is not None:
        max_observations = int(max_observations)
        if max_observations < 4:
            raise ValueError("mi_max_observations must be >= 4 or None.")
        if len(observations) > max_observations:
            rng = np.random.default_rng(int(random_state))
            indices = rng.choice(
                len(observations),
                size=max_observations,
                replace=False,
            )
            observations = observations[indices]

    return compute_mtl_shared_mi_adjacency(
        observations,
        n_channels=int(n_channels),
        n_bands=int(n_bands),
        n_neighbors=int(n_neighbors),
        random_state=int(random_state),
        zero_diagonal=bool(zero_diagonal),
        band_reduction=str(band_reduction),
    )


@tf.keras.utils.register_keras_serializable(package="EEGProc")
class GradientReversal(tf.keras.layers.Layer):
    """Identity forward pass with a negative scaled encoder gradient."""

    def __init__(self, adversarial_weight: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        if float(adversarial_weight) < 0.0:
            raise ValueError("adversarial_weight must be non-negative.")
        self.adversarial_weight = float(adversarial_weight)

    def call(self, inputs):
        scale = self.adversarial_weight

        @tf.custom_gradient
        def reverse_gradient(x):
            def gradient(dy):
                return -tf.cast(scale, dy.dtype) * dy

            return x, gradient

        return reverse_gradient(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({"adversarial_weight": self.adversarial_weight})
        return config


@tf.keras.utils.register_keras_serializable(package="EEGProc")
class SICModel(tf.keras.Model):
    """SIC with serial or parallel feature-fusion encoder and focal classifier."""

    def __init__(
        self,
        *,
        graph_encoder: GCNMTLEncoder,
        decoder: GCNMTLDecoder,
        classification_level: str = "trial",
        n_classes: int = 2,
        bilstm_units: int = 128,
        n_bilstm_layers: int = 1,
        bilstm_dropout: float = 0.30,
        architecture_mode: str = "serial",
        fusion_units: int = 128,
        fusion_dropout: float = 0.20,
        temporal_downsample_factor: int = 1,
        z_dim: int = 64,
        z_log_var_clip_min: float = -20.0,
        z_log_var_clip_max: float = 20.0,
        classification_hidden_units: Sequence[int] = (128,),
        classification_dropout: float = 0.20,
        activation: str = "relu",
        label_smoothing: float = 0.0,
        focal_gamma: float = 1.0,
        focal_alpha: float | None = None,
        vc_loss_weight: float = 1.0,
        vc_alpha: float = 1.0,
        vc_beta: float = 0.5,
        vc_gamma: float = 0.0,
        vc_lambda: float = 0.0,
        update_vc_discriminator: bool = False,
        vae_loss_weight: float = 0.10,
        vae_beta: float = 0.05,
        use_vrex: bool = False,
        vrex_penalty_weight: float = 1.0,
        use_subject_adversarial: bool = True,
        n_subject_classes: int | None = None,
        subject_adversarial_weight: float = 0.8,
        subject_loss_weight: float = 1.0,
        subject_hidden_units: int = 64,
        subject_dropout: float = 0.0,
        calibration_unfreeze_layers: int = 1,
        calibration_use_vc_target: bool = True,
        calibration_vc_alpha: float | None = None,
        calibration_vc_beta: float | None = None,
        calibration_vc_gamma: float | None = None,
        calibration_vc_lambda: float | None = None,
        use_class_weight: bool = False,
        name: str = "sic_subject_invariant_calibrator",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        classification_level = str(classification_level).lower()
        if classification_level not in {"window", "trial"}:
            raise ValueError("classification_level must be 'window' or 'trial'.")
        if graph_encoder is None or decoder is None:
            raise ValueError("graph_encoder and decoder are required.")
        if int(n_classes) < 2:
            raise ValueError("n_classes must be >= 2.")
        if int(bilstm_units) < 1 or int(n_bilstm_layers) < 1 or int(z_dim) < 1:
            raise ValueError("BiLSTM and z dimensions must be positive.")
        architecture_mode = str(architecture_mode).lower()
        if architecture_mode not in {"serial", "feature_fusion"}:
            raise ValueError(
                "architecture_mode must be 'serial' or 'feature_fusion'."
            )
        if int(fusion_units) < 1:
            raise ValueError("fusion_units must be positive.")
        if not 0.0 <= float(fusion_dropout) < 1.0:
            raise ValueError("fusion_dropout must be in [0, 1).")
        if int(temporal_downsample_factor) < 1:
            raise ValueError("temporal_downsample_factor must be >= 1.")
        if float(focal_gamma) < 0.0:
            raise ValueError("focal_gamma must be non-negative.")
        if focal_alpha is not None and not 0.0 <= float(focal_alpha) <= 1.0:
            raise ValueError("focal_alpha must be in [0, 1] or None.")
        if float(z_log_var_clip_min) >= float(z_log_var_clip_max):
            raise ValueError("z_log_var_clip_min must be less than max.")
        hidden_units = tuple(int(value) for value in classification_hidden_units)
        if any(value < 1 for value in hidden_units):
            raise ValueError("classification_hidden_units must be positive.")
        if not 0.0 <= float(classification_dropout) < 1.0:
            raise ValueError("classification_dropout must be in [0, 1).")
        if not 0.0 <= float(bilstm_dropout) < 1.0:
            raise ValueError("bilstm_dropout must be in [0, 1).")
        if not 0.0 <= float(label_smoothing) < 1.0:
            raise ValueError("label_smoothing must be in [0, 1).")
        for loss_name, value in (
            ("vc_loss_weight", vc_loss_weight),
            ("vae_loss_weight", vae_loss_weight),
            ("vae_beta", vae_beta),
            ("subject_adversarial_weight", subject_adversarial_weight),
            ("subject_loss_weight", subject_loss_weight),
        ):
            if float(value) < 0.0:
                raise ValueError(f"{loss_name} must be non-negative.")
        if int(subject_hidden_units) < 1:
            raise ValueError("subject_hidden_units must be positive.")
        if not 0.0 <= float(subject_dropout) < 1.0:
            raise ValueError("subject_dropout must be in [0, 1).")
        if float(vrex_penalty_weight) < 0.0:
            raise ValueError("vrex_penalty_weight must be non-negative.")
        calibration_unfreeze_layers = int(calibration_unfreeze_layers)
        max_calibration_layers = len(hidden_units) + 1
        if not 1 <= calibration_unfreeze_layers <= max_calibration_layers:
            raise ValueError(
                "calibration_unfreeze_layers must be between 1 and "
                f"{max_calibration_layers}; got {calibration_unfreeze_layers}."
            )

        self.graph_encoder = graph_encoder
        self.decoder = decoder
        self.classification_level = classification_level
        self.n_classes = int(n_classes)
        self.bilstm_units = int(bilstm_units)
        self.n_bilstm_layers = int(n_bilstm_layers)
        self.bilstm_dropout_rate = float(bilstm_dropout)
        self.architecture_mode = architecture_mode
        self.fusion_units = int(fusion_units)
        self.fusion_dropout_rate = float(fusion_dropout)
        self.temporal_downsample_factor = int(temporal_downsample_factor)
        self.z_dim = int(z_dim)
        self.z_log_var_clip_min = float(z_log_var_clip_min)
        self.z_log_var_clip_max = float(z_log_var_clip_max)
        self.classification_hidden_units = hidden_units
        self.classification_dropout_rate = float(classification_dropout)
        self.activation_name = str(activation)
        self.label_smoothing = float(label_smoothing)
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = None if focal_alpha is None else float(focal_alpha)

        self.vc_loss_weight = float(vc_loss_weight)
        self.vc_alpha = float(vc_alpha)
        self.vc_beta = float(vc_beta)
        self.vc_gamma = float(vc_gamma)
        self.vc_lambda = float(vc_lambda)
        self.update_vc_discriminator = bool(update_vc_discriminator)
        self.vae_loss_weight = float(vae_loss_weight)
        self.vae_beta = float(vae_beta)
        self.use_class_weight = bool(use_class_weight)

        self.use_vrex = bool(use_vrex)
        self.vrex_penalty_weight = float(vrex_penalty_weight)

        self.subject_adversarial_enabled = bool(use_subject_adversarial)
        self.n_subject_classes = (
            None if n_subject_classes is None else int(n_subject_classes)
        )
        self.subject_adversarial_weight = float(subject_adversarial_weight)
        self.subject_loss_weight = float(subject_loss_weight)
        self.subject_hidden_units = int(subject_hidden_units)
        self.subject_dropout_rate = float(subject_dropout)

        self.calibration_unfreeze_layers = calibration_unfreeze_layers
        self.calibration_use_vc_target = bool(calibration_use_vc_target)
        self.calibration_vc_alpha = (
            self.vc_alpha if calibration_vc_alpha is None else float(calibration_vc_alpha)
        )
        self.calibration_vc_beta = (
            self.vc_beta if calibration_vc_beta is None else float(calibration_vc_beta)
        )
        self.calibration_vc_gamma = (
            self.vc_gamma if calibration_vc_gamma is None else float(calibration_vc_gamma)
        )
        self.calibration_vc_lambda = (
            self.vc_lambda if calibration_vc_lambda is None else float(calibration_vc_lambda)
        )
        self.calibration_mode = False

        self.requires_subject_ids = self.subject_adversarial_enabled or self.use_vrex
        # Current EEGProc cross_val uses ``use_subject_adversarial`` as the
        # compatibility gate for attaching source subject IDs.  Keep that gate
        # true whenever V-REx needs subject metadata; actual GRL computation is
        # still controlled exclusively by ``subject_adversarial_enabled``.
        self.use_subject_adversarial = self.requires_subject_ids

        # MTL encoder already performs GCN + spectral GRU.  This recurrent
        # stack is exclusively temporal, preserving the reduced time axis.
        self.temporal_bilstms: list[tf.keras.layers.Layer] = []
        self.temporal_norms: list[tf.keras.layers.Layer] = []
        self.temporal_dropouts: list[tf.keras.layers.Layer] = []
        for index in range(self.n_bilstm_layers):
            self.temporal_bilstms.append(
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(
                        self.bilstm_units,
                        return_sequences=True,
                        name=f"v6_temporal_lstm_{index}",
                    ),
                    merge_mode="concat",
                    name=f"v6_temporal_bilstm_{index}",
                )
            )
            self.temporal_norms.append(
                tf.keras.layers.LayerNormalization(
                    axis=-1,
                    name=f"v6_temporal_bilstm_ln_{index}",
                )
            )
            self.temporal_dropouts.append(
                tf.keras.layers.Dropout(
                    self.bilstm_dropout_rate,
                    name=f"v6_temporal_bilstm_dropout_{index}",
                )
            )

        # In serial mode the BiLSTM consumes the GCN-GRU sequence.
        # In feature_fusion mode the BiLSTM is an independent raw-EEG branch;
        # it is downsampled to the GCN-GRU temporal resolution and fused before z.
        self.parallel_temporal_pool = tf.keras.layers.AveragePooling1D(
            pool_size=self.temporal_downsample_factor,
            strides=self.temporal_downsample_factor,
            padding="same",
            name="v6_parallel_bilstm_pool",
        )
        self.fusion_projection = tf.keras.layers.Dense(
            self.fusion_units,
            activation=self.activation_name,
            name="v6_feature_fusion_dense",
        )
        self.fusion_norm = tf.keras.layers.LayerNormalization(
            axis=-1,
            name="v6_feature_fusion_ln",
        )
        self.fusion_dropout_layer = tf.keras.layers.Dropout(
            self.fusion_dropout_rate,
            name="v6_feature_fusion_dropout",
        )

        self.z_mean_projection = tf.keras.layers.Dense(
            self.z_dim,
            activation=None,
            name="v6_z_mean",
        )
        self.z_log_var_projection = tf.keras.layers.Dense(
            self.z_dim,
            activation=None,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="v6_z_log_var",
        )
        self.z_pool = tf.keras.layers.GlobalAveragePooling1D(name="v6_z_pool")

        # Dense-softmax prediction head. Hidden blocks are explicit so the
        # calibration policy can unfreeze exactly the last k prediction layers.
        self.classification_dense_layers: list[tf.keras.layers.Layer] = []
        self.classification_norm_layers: list[tf.keras.layers.Layer] = []
        self.classification_dropout_layers: list[tf.keras.layers.Layer] = []
        for index, units in enumerate(self.classification_hidden_units):
            self.classification_dense_layers.append(
                tf.keras.layers.Dense(
                    units,
                    activation=self.activation_name,
                    name=f"v6_classifier_dense_{index}",
                )
            )
            self.classification_norm_layers.append(
                tf.keras.layers.LayerNormalization(
                    axis=-1,
                    name=f"v6_classifier_ln_{index}",
                )
            )
            self.classification_dropout_layers.append(
                tf.keras.layers.Dropout(
                    self.classification_dropout_rate,
                    name=f"v6_classifier_dropout_{index}",
                )
            )
        self.logits_layer = tf.keras.layers.Dense(
            self.n_classes,
            activation=None,
            name="v6_classifier_logits",
        )

        vc_dim = (
            self.classification_hidden_units[-1]
            if self.classification_hidden_units
            else self.z_dim
        )
        self.vc_target = VariationalClassifier(
            n_classes=self.n_classes,
            latent_dim=vc_dim,
            label_smoothing=self.label_smoothing,
            name="v6_vc_target",
        )
        # We pass externally produced dense logits into vc_loss_components, so
        # explicitly build the target's Gaussian/discriminator parameters.
        self.vc_target.build(tf.TensorShape([None, vc_dim]))

        self.subject_gradient_reversal = None
        self.subject_hidden = None
        self.subject_dropout_layer = None
        self.subject_logits_layer = None
        if self.subject_adversarial_enabled and self.n_subject_classes is not None:
            self._configure_subject_head(self.n_subject_classes)

        self.main_optimizer = None
        self.vc_discriminator_optimizer = None

        # Keras metrics.  The same classification metrics are emitted both
        # during source training and calibration; cross_val additionally logs
        # paired zero-shot and post-calibration trial metrics.
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.vc_loss_tracker = tf.keras.metrics.Mean(name="vc_loss")
        self.cross_entropy_tracker = tf.keras.metrics.Mean(name="cross_entropy")
        self.focal_loss_tracker = tf.keras.metrics.Mean(name="focal_loss")
        self.accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(
            name="accuracy"
        )
        self.vae_loss_tracker = tf.keras.metrics.Mean(name="vae_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.vrex_penalty_tracker = tf.keras.metrics.Mean(name="vrex_penalty")
        self.vrex_subject_risk_mean_tracker = tf.keras.metrics.Mean(
            name="vrex_subject_risk_mean"
        )
        self.vrex_subjects_per_batch_tracker = tf.keras.metrics.Mean(
            name="vrex_subjects_per_batch"
        )
        self.subject_loss_tracker = tf.keras.metrics.Mean(name="subject_loss")
        self.subject_accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(
            name="subject_accuracy"
        )

    @property
    def metrics(self):
        output = [
            self.loss_tracker,
            self.vc_loss_tracker,
            self.cross_entropy_tracker,
            self.focal_loss_tracker,
            self.accuracy_tracker,
            self.vae_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
        if self.use_vrex:
            output.extend(
                [
                    self.vrex_penalty_tracker,
                    self.vrex_subject_risk_mean_tracker,
                    self.vrex_subjects_per_batch_tracker,
                ]
            )
        if self.subject_adversarial_enabled:
            output.extend(
                [
                    self.subject_loss_tracker,
                    self.subject_accuracy_tracker,
                ]
            )
        return output

    def compile(
        self,
        main_optimizer,
        vc_discriminator_optimizer=None,
        **kwargs,
    ):
        if main_optimizer is None:
            raise ValueError("main_optimizer is required.")
        kwargs.setdefault("jit_compile", False)
        super().compile(optimizer=main_optimizer, **kwargs)
        self.main_optimizer = main_optimizer
        self.vc_discriminator_optimizer = vc_discriminator_optimizer

    def fit(self, *args, **kwargs):
        if not self.use_class_weight:
            kwargs.pop("class_weight", None)
        return super().fit(*args, **kwargs)

    @staticmethod
    def _flatten_labels(labels):
        labels = tf.convert_to_tensor(labels)
        if (
            labels.shape.rank == 2
            and labels.shape[-1] is not None
            and labels.shape[-1] > 1
        ):
            return tf.argmax(labels, axis=-1, output_type=tf.int32)
        return tf.cast(tf.reshape(labels, [-1]), tf.int32)

    @staticmethod
    def _split_eeg_and_subject_inputs(inputs):
        if isinstance(inputs, Mapping):
            if "eeg" not in inputs:
                raise ValueError("Input mappings must contain an 'eeg' key.")
            return inputs["eeg"], inputs.get("subject_id")
        return inputs, None

    def _configure_subject_head(self, n_subject_classes: int):
        n_subject_classes = int(n_subject_classes)
        if n_subject_classes < 2:
            raise ValueError("Subject adversity requires at least two subjects.")
        if self.subject_logits_layer is not None:
            if self.n_subject_classes != n_subject_classes:
                raise ValueError(
                    "Subject head already configured for "
                    f"{self.n_subject_classes}, not {n_subject_classes}."
                )
            return
        self.n_subject_classes = n_subject_classes
        self.subject_gradient_reversal = GradientReversal(
            adversarial_weight=self.subject_adversarial_weight,
            name="v6_subject_gradient_reversal",
        )
        self.subject_hidden = tf.keras.layers.Dense(
            self.subject_hidden_units,
            activation=self.activation_name,
            name="v6_subject_hidden",
        )
        self.subject_dropout_layer = tf.keras.layers.Dropout(
            self.subject_dropout_rate,
            name="v6_subject_dropout",
        )
        self.subject_logits_layer = tf.keras.layers.Dense(
            self.n_subject_classes,
            activation=None,
            name="v6_subject_logits",
        )

    def prepare_fit_inputs(self, eeg_inputs, subject_ids):
        """Attach contiguous source-fold subject labels for adversarial training."""
        if not self.requires_subject_ids:
            return eeg_inputs
        eeg_array = np.asarray(eeg_inputs)
        subjects = np.asarray(subject_ids).reshape(-1)
        if len(eeg_array) != len(subjects):
            raise ValueError("EEG inputs and subject IDs must align.")
        unique_subjects = np.sort(np.unique(subjects))
        if self.subject_adversarial_enabled:
            self._configure_subject_head(len(unique_subjects))
        mapping = {
            value.item() if isinstance(value, np.generic) else value: index
            for index, value in enumerate(unique_subjects)
        }
        remapped = np.asarray(
            [
                mapping[value.item() if isinstance(value, np.generic) else value]
                for value in subjects
            ],
            dtype=np.int32,
        )
        return {"eeg": eeg_array, "subject_id": remapped}

    def prepare_calibration_inputs(self, eeg_inputs):
        """Calibration intentionally has no subject-adversarial input."""
        return np.asarray(eeg_inputs)

    def _flatten_trial_windows(self, eeg_inputs):
        eeg_inputs = tf.convert_to_tensor(eeg_inputs, dtype=tf.float32)
        if eeg_inputs.shape.rank != 4:
            raise ValueError(
                "Trial mode expects (batch, windows, timesteps, features); "
                f"got {eeg_inputs.shape}."
            )
        shape = tf.shape(eeg_inputs)
        return (
            tf.reshape(eeg_inputs, [shape[0] * shape[1], shape[2], shape[3]]),
            shape[0],
            shape[1],
        )

    def _temporal_encode(self, graph_sequence, training: bool):
        x = graph_sequence
        for bilstm, norm, dropout in zip(
            self.temporal_bilstms,
            self.temporal_norms,
            self.temporal_dropouts,
        ):
            x = bilstm(x, training=training)
            x = norm(x)
            x = dropout(x, training=training)
        return x

    def _posterior_from_flat_windows(self, flat_windows, training: bool):
        graph_sequence = self.graph_encoder(flat_windows, training=training)

        if self.architecture_mode == "serial":
            temporal_sequence = self._temporal_encode(
                graph_sequence,
                training=training,
            )
            fused_sequence = temporal_sequence
            bilstm_sequence = temporal_sequence
        else:
            # Independent temporal branch directly from the raw EEG feature
            # sequence. No GCN/GRU features enter this BiLSTM branch.
            bilstm_sequence = self._temporal_encode(
                flat_windows,
                training=training,
            )
            pooled_bilstm = self.parallel_temporal_pool(bilstm_sequence)

            # The builder sets temporal_downsample_factor=t_down, so the two
            # branches normally align exactly. The assertion catches any future
            # encoder pooling change rather than silently mis-fusing tensors.
            tf.debugging.assert_equal(
                tf.shape(graph_sequence)[1],
                tf.shape(pooled_bilstm)[1],
                message=(
                    "GCN-GRU and BiLSTM branch sequence lengths do not match "
                    "for feature fusion."
                ),
            )
            fused_sequence = tf.concat(
                [graph_sequence, pooled_bilstm],
                axis=-1,
            )
            fused_sequence = self.fusion_projection(fused_sequence)
            fused_sequence = self.fusion_norm(fused_sequence)
            fused_sequence = self.fusion_dropout_layer(
                fused_sequence,
                training=training,
            )
            temporal_sequence = fused_sequence

        z_mean = self.z_mean_projection(fused_sequence)
        raw_log_var = self.z_log_var_projection(fused_sequence)
        z_log_var = tf.clip_by_value(
            raw_log_var,
            self.z_log_var_clip_min,
            self.z_log_var_clip_max,
        )
        return {
            "graph_sequence": graph_sequence,
            "temporal_sequence": temporal_sequence,
            "bilstm_sequence": bilstm_sequence,
            "fused_sequence": fused_sequence,
            "z_mean": z_mean,
            "z_log_var": z_log_var,
        }

    @staticmethod
    def _reparameterize(z_mean, z_log_var, seed=None):
        if seed is None:
            epsilon = tf.random.normal(tf.shape(z_mean), dtype=z_mean.dtype)
        else:
            if isinstance(seed, tuple):
                stateless_seed = tf.constant(seed, dtype=tf.int32)
            else:
                stateless_seed = tf.constant([int(seed), 0], dtype=tf.int32)
            epsilon = tf.random.stateless_normal(
                tf.shape(z_mean),
                seed=stateless_seed,
                dtype=z_mean.dtype,
            )
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def _pool_z_for_prediction(self, flat_z, batch_size=None, n_windows=None):
        window_z = self.z_pool(flat_z)
        if self.classification_level == "window":
            return window_z, window_z
        window_z = tf.reshape(window_z, [batch_size, n_windows, self.z_dim])
        trial_z = tf.reduce_mean(window_z, axis=1)
        return trial_z, window_z

    def _classifier_forward(self, pooled_z, training: bool):
        x = pooled_z
        for dense, norm, dropout in zip(
            self.classification_dense_layers,
            self.classification_norm_layers,
            self.classification_dropout_layers,
        ):
            x = dense(x)
            x = norm(x)
            # During calibration, frozen earlier classifier blocks are also
            # deterministic. Dropout remains active only in blocks that were
            # explicitly selected for fine-tuning.
            block_training = bool(training) and bool(dense.trainable)
            x = dropout(x, training=block_training)
        classification_embedding = x
        logits = self.logits_layer(classification_embedding)
        return classification_embedding, logits

    def _encode(
        self,
        eeg_inputs,
        *,
        training: bool,
        sample_latent: bool,
        seed=None,
        classifier_training: bool | None = None,
    ):
        eeg_inputs = tf.convert_to_tensor(eeg_inputs, dtype=tf.float32)
        if classifier_training is None:
            classifier_training = bool(training)
        if self.classification_level == "window":
            if eeg_inputs.shape.rank != 3:
                raise ValueError(
                    "Window mode expects (batch, timesteps, features); "
                    f"got {eeg_inputs.shape}."
                )
            flat_windows = eeg_inputs
            batch_size = n_windows = None
        else:
            flat_windows, batch_size, n_windows = self._flatten_trial_windows(
                eeg_inputs
            )

        posterior = self._posterior_from_flat_windows(
            flat_windows,
            training=training,
        )
        z = (
            self._reparameterize(
                posterior["z_mean"],
                posterior["z_log_var"],
                seed=seed,
            )
            if sample_latent
            else posterior["z_mean"]
        )
        pooled_z, window_z = self._pool_z_for_prediction(
            z,
            batch_size=batch_size,
            n_windows=n_windows,
        )
        pooled_z_mean, window_z_mean = self._pool_z_for_prediction(
            posterior["z_mean"],
            batch_size=batch_size,
            n_windows=n_windows,
        )
        classification_embedding, logits = self._classifier_forward(
            pooled_z,
            training=bool(classifier_training),
        )
        posterior.update(
            {
                "flat_windows": flat_windows,
                "z": z,
                "pooled_z": pooled_z,
                "window_z": window_z,
                "pooled_z_mean": pooled_z_mean,
                "window_z_mean": window_z_mean,
                "classification_embedding": classification_embedding,
                "logits": logits,
                "probabilities": tf.nn.softmax(logits, axis=-1),
                "batch_size": batch_size,
                "n_windows": n_windows,
            }
        )
        return posterior

    def call(self, inputs, training=False, sample_latent: bool | None = None):
        eeg_inputs, _ = self._split_eeg_and_subject_inputs(inputs)
        if sample_latent is None:
            sample_latent = bool(training) and not self.calibration_mode
        outputs = self._encode(
            eeg_inputs,
            training=bool(training),
            sample_latent=bool(sample_latent),
        )
        return outputs["logits"]

    def _per_sample_focal_loss(self, logits, y_flat):
        """Sparse binary/multiclass focal loss from deterministic logits."""
        y_flat = tf.cast(tf.reshape(y_flat, [-1]), tf.int32)
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_flat,
            logits=logits,
        )
        probs = tf.nn.softmax(logits, axis=-1)
        row_ids = tf.range(tf.shape(y_flat)[0], dtype=tf.int32)
        gather_ids = tf.stack([row_ids, y_flat], axis=1)
        p_t = tf.gather_nd(probs, gather_ids)
        modulating = tf.pow(
            tf.maximum(1.0 - p_t, tf.keras.backend.epsilon()),
            tf.cast(self.focal_gamma, p_t.dtype),
        )
        loss = modulating * ce

        if self.focal_alpha is not None:
            if self.n_classes != 2:
                raise ValueError(
                    "Scalar focal_alpha is currently defined only for binary SIC."
                )
            alpha = tf.cast(self.focal_alpha, loss.dtype)
            alpha_t = tf.where(
                tf.equal(y_flat, 1),
                alpha,
                1.0 - alpha,
            )
            loss = alpha_t * loss
        return loss

    @staticmethod
    def _weighted_mean(values, sample_weight):
        if sample_weight is None:
            return tf.reduce_mean(values)
        weights = tf.cast(tf.reshape(sample_weight, [-1]), values.dtype)
        return tf.math.divide_no_nan(
            tf.reduce_sum(values * weights),
            tf.reduce_sum(weights),
        )

    def _vc_components(
        self,
        classification_embedding,
        logits,
        y_flat,
        sample_weight,
        *,
        calibration: bool,
    ):
        # The deterministic classifier term is focal loss. The VC target still
        # contributes its latent/distribution regularizers, but its CE term is
        # replaced so we do not optimize both CE and focal simultaneously.
        focal_per_sample = self._per_sample_focal_loss(logits, y_flat)
        focal_loss = self._weighted_mean(focal_per_sample, sample_weight)

        if calibration and not self.calibration_use_vc_target:
            zero = tf.zeros((), dtype=focal_loss.dtype)
            return {
                "total_loss": focal_loss,
                "cross_entropy": focal_loss,
                "weighted_cross_entropy": focal_loss,
                "focal_loss": focal_loss,
                "latent_posterior_kl": zero,
                "weighted_latent_posterior_kl": zero,
                "discriminator_kl": zero,
                "weighted_discriminator_kl": zero,
                "class_prior_kl": zero,
                "weighted_class_prior_kl": zero,
            }

        alpha = self.calibration_vc_alpha if calibration else self.vc_alpha
        components = self.vc_target.vc_loss_components(
            mh=classification_embedding,
            y=y_flat,
            alpha=alpha,
            beta=(self.calibration_vc_beta if calibration else self.vc_beta),
            gamma=(self.calibration_vc_gamma if calibration else self.vc_gamma),
            lambda_=(self.calibration_vc_lambda if calibration else self.vc_lambda),
            logits=logits,
            sample_weight=sample_weight,
        )

        # VariationalClassifier's alpha-weighted deterministic CE is replaced
        # by alpha-weighted focal loss, preserving all non-deterministic VC
        # terms exactly as configured.
        dtype = components["total_loss"].dtype
        ce_term = tf.cast(components["weighted_cross_entropy"], dtype)
        focal_term = tf.cast(alpha, dtype) * tf.cast(focal_loss, dtype)
        components = dict(components)
        components["total_loss"] = (
            tf.cast(components["total_loss"], dtype) - ce_term + focal_term
        )
        components["focal_loss"] = tf.cast(focal_loss, dtype)
        return components

    def _vae_components(self, outputs, training: bool):
        reconstruction = self.decoder(outputs["z"], training=training)
        reconstruction_loss = tf.reduce_mean(
            tf.square(outputs["flat_windows"] - reconstruction)
        )
        z_mean = outputs["z_mean"]
        z_log_var = outputs["z_log_var"]
        kl_values = -0.5 * (
            1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        kl_loss = tf.reduce_mean(kl_values)
        vae_loss = reconstruction_loss + tf.cast(
            self.vae_beta,
            kl_loss.dtype,
        ) * kl_loss
        return {
            "vae_loss": vae_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "reconstruction": reconstruction,
        }

    def _subject_logits(self, pooled_z, training: bool, use_grl: bool):
        if self.subject_logits_layer is None:
            raise RuntimeError("Subject head has not been configured.")
        x = self.subject_gradient_reversal(pooled_z) if use_grl else pooled_z
        x = self.subject_hidden(x)
        x = self.subject_dropout_layer(x, training=training)
        return self.subject_logits_layer(x)

    def _subject_components(self, pooled_z, subject_ids, training: bool, use_grl: bool):
        if not self.subject_adversarial_enabled or subject_ids is None:
            zero = tf.zeros((), dtype=pooled_z.dtype)
            return {
                "subject_loss": zero,
                "subject_logits": None,
                "subject_targets": None,
            }
        targets = tf.cast(tf.reshape(subject_ids, [-1]), tf.int32)
        logits = self._subject_logits(
            pooled_z,
            training=training,
            use_grl=use_grl,
        )
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=targets,
                logits=logits,
            )
        )
        return {
            "subject_loss": loss,
            "subject_logits": logits,
            "subject_targets": targets,
        }

    def _regularization_loss(self, dtype):
        if not self.losses:
            return tf.zeros((), dtype=dtype)
        return tf.add_n([tf.cast(value, dtype) for value in self.losses])

    @staticmethod
    def _apply_gradients(optimizer, gradients, variables):
        pairs = [
            (gradient, variable)
            for gradient, variable in zip(gradients, variables)
            if gradient is not None
        ]
        if pairs:
            optimizer.apply_gradients(pairs)

    def _subject_head_variables(self):
        if self.subject_logits_layer is None:
            return []
        variables = []
        for component in (self.subject_hidden, self.subject_logits_layer):
            variables.extend(component.trainable_variables)
        return _deduplicate_variables(variables)

    def _vc_discriminator_variables(self):
        if not self.update_vc_discriminator:
            return []
        variables = []
        for attribute in ("disc_w", "disc_b"):
            variable = getattr(self.vc_target, attribute, None)
            if variable is not None:
                variables.append(variable)
        return variables

    def _vrex_components(self, logits, y_flat, subject_ids, sample_weight):
        """Return subject-wise classification risks and the V-REx penalty.

        The deterministic SIC classifier uses focal loss. V-REx therefore
        adds ``lambda * Var(R_s)`` where ``R_s`` is the mean focal risk for one
        source subject represented in the current minibatch.
        """
        dtype = logits.dtype
        zero = tf.zeros((), dtype=dtype)
        if not self.use_vrex or subject_ids is None:
            return {
                "penalty": zero,
                "mean_subject_risk": zero,
                "n_subjects": zero,
                "subject_risks": tf.zeros((0,), dtype=dtype),
            }

        targets = tf.cast(tf.reshape(subject_ids, [-1]), tf.int32)
        per_sample = self._per_sample_focal_loss(logits, y_flat)
        weights = None
        if sample_weight is not None:
            weights = tf.cast(tf.reshape(sample_weight, [-1]), per_sample.dtype)

        unique_subjects = tf.unique(targets).y

        def risk_for_subject(subject_id):
            mask = tf.cast(tf.equal(targets, subject_id), per_sample.dtype)
            if weights is None:
                return tf.math.divide_no_nan(
                    tf.reduce_sum(per_sample * mask),
                    tf.reduce_sum(mask),
                )
            subject_weights = weights * mask
            return tf.math.divide_no_nan(
                tf.reduce_sum(per_sample * subject_weights),
                tf.reduce_sum(subject_weights),
            )

        subject_risks = tf.map_fn(
            risk_for_subject,
            unique_subjects,
            fn_output_signature=per_sample.dtype,
        )
        mean_subject_risk = tf.reduce_mean(subject_risks)
        penalty = tf.math.reduce_variance(subject_risks)
        return {
            "penalty": penalty,
            "mean_subject_risk": mean_subject_risk,
            "n_subjects": tf.cast(tf.size(unique_subjects), dtype),
            "subject_risks": subject_risks,
        }

    def _update_metrics(
        self,
        *,
        total_loss,
        vc_components,
        outputs,
        y_flat,
        sample_weight,
        vae_components=None,
        subject_components=None,
        vrex_components=None,
    ):
        zero = tf.zeros((), dtype=total_loss.dtype)
        self.loss_tracker.update_state(total_loss)
        self.vc_loss_tracker.update_state(vc_components["total_loss"])
        self.cross_entropy_tracker.update_state(vc_components["cross_entropy"])
        self.focal_loss_tracker.update_state(
            vc_components.get("focal_loss", vc_components["cross_entropy"])
        )
        self.accuracy_tracker.update_state(
            y_flat,
            outputs["logits"],
            sample_weight=sample_weight,
        )
        self.vae_loss_tracker.update_state(
            zero if vae_components is None else vae_components["vae_loss"]
        )
        self.reconstruction_loss_tracker.update_state(
            zero
            if vae_components is None
            else vae_components["reconstruction_loss"]
        )
        self.kl_loss_tracker.update_state(
            zero if vae_components is None else vae_components["kl_loss"]
        )
        if self.use_vrex:
            self.vrex_penalty_tracker.update_state(
                zero if vrex_components is None else vrex_components["penalty"]
            )
            self.vrex_subject_risk_mean_tracker.update_state(
                zero
                if vrex_components is None
                else vrex_components["mean_subject_risk"]
            )
            self.vrex_subjects_per_batch_tracker.update_state(
                zero if vrex_components is None else vrex_components["n_subjects"]
            )
        if self.subject_adversarial_enabled:
            subject_loss = (
                zero
                if subject_components is None
                else subject_components["subject_loss"]
            )
            self.subject_loss_tracker.update_state(subject_loss)
            if (
                subject_components is not None
                and subject_components["subject_logits"] is not None
            ):
                self.subject_accuracy_tracker.update_state(
                    subject_components["subject_targets"],
                    subject_components["subject_logits"],
                )

    def _source_train_step(self, x, y_flat, sample_weight):
        eeg_inputs, subject_ids = self._split_eeg_and_subject_inputs(x)
        if self.use_vrex and subject_ids is None:
            raise ValueError(
                "V-REx source training requires subject IDs in each minibatch."
            )

        with tf.GradientTape() as tape:
            outputs = self._encode(
                eeg_inputs,
                training=True,
                sample_latent=True,
            )
            vc_components = self._vc_components(
                outputs["classification_embedding"],
                outputs["logits"],
                y_flat,
                sample_weight,
                calibration=False,
            )
            vae_components = self._vae_components(outputs, training=True)
            subject_components = self._subject_components(
                outputs["pooled_z_mean"],
                subject_ids,
                training=True,
                use_grl=True,
            )
            vrex_components = self._vrex_components(
                outputs["logits"],
                y_flat,
                subject_ids,
                sample_weight,
            )
            dtype = vc_components["total_loss"].dtype
            total = (
                tf.cast(self.vc_loss_weight, dtype)
                * vc_components["total_loss"]
                + tf.cast(self.vae_loss_weight, dtype)
                * tf.cast(vae_components["vae_loss"], dtype)
                + tf.cast(self.subject_loss_weight, dtype)
                * tf.cast(subject_components["subject_loss"], dtype)
                + tf.cast(self.vrex_penalty_weight, dtype)
                * tf.cast(vrex_components["penalty"], dtype)
                + self._regularization_loss(dtype)
            )
        variables = self.trainable_variables
        gradients = tape.gradient(total, variables)
        self._apply_gradients(self.main_optimizer, gradients, variables)

        if self.update_vc_discriminator:
            if self.vc_discriminator_optimizer is None:
                raise RuntimeError(
                    "update_vc_discriminator=True requires a discriminator optimizer."
                )
            embedding_frozen = tf.stop_gradient(outputs["classification_embedding"])
            with tf.GradientTape() as disc_tape:
                disc_loss = self.vc_target.discriminator_loss(
                    embedding_frozen,
                    y_flat,
                )
            disc_variables = self._vc_discriminator_variables()
            disc_gradients = disc_tape.gradient(disc_loss, disc_variables)
            self._apply_gradients(
                self.vc_discriminator_optimizer,
                disc_gradients,
                disc_variables,
            )

        self._update_metrics(
            total_loss=total,
            vc_components=vc_components,
            outputs=outputs,
            y_flat=y_flat,
            sample_weight=sample_weight,
            vae_components=vae_components,
            subject_components=subject_components,
            vrex_components=vrex_components,
        )

    def _calibration_train_step(self, x, y_flat, sample_weight):
        eeg_inputs, _ = self._split_eeg_and_subject_inputs(x)
        # Backbone layers are frozen and posterior mean is used so six-trial
        # calibration fits a stable subject-specific decision boundary.
        with tf.GradientTape() as tape:
            outputs = self._encode(
                eeg_inputs,
                training=False,
                sample_latent=False,
                classifier_training=True,
            )
            vc_components = self._vc_components(
                outputs["classification_embedding"],
                outputs["logits"],
                y_flat,
                sample_weight,
                calibration=True,
            )
            dtype = vc_components["total_loss"].dtype
            total = (
                tf.cast(self.vc_loss_weight, dtype)
                * vc_components["total_loss"]
                + self._regularization_loss(dtype)
            )
        variables = self.trainable_variables
        gradients = tape.gradient(total, variables)
        self._apply_gradients(self.main_optimizer, gradients, variables)
        self._update_metrics(
            total_loss=total,
            vc_components=vc_components,
            outputs=outputs,
            y_flat=y_flat,
            sample_weight=sample_weight,
            vae_components=None,
            subject_components=None,
            vrex_components=None,
        )

    def train_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        y_flat = self._flatten_labels(y)
        if self.main_optimizer is None:
            raise RuntimeError("Call model.compile(...) before model.fit(...).")
        if self.calibration_mode:
            self._calibration_train_step(x, y_flat, sample_weight)
        else:
            self._source_train_step(x, y_flat, sample_weight)
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        y_flat = self._flatten_labels(y)
        eeg_inputs, subject_ids = self._split_eeg_and_subject_inputs(x)
        outputs = self._encode(
            eeg_inputs,
            training=False,
            sample_latent=False,
        )
        vc_components = self._vc_components(
            outputs["classification_embedding"],
            outputs["logits"],
            y_flat,
            sample_weight,
            calibration=self.calibration_mode,
        )
        if self.calibration_mode:
            vae_components = None
            subject_components = None
            total = (
                tf.cast(self.vc_loss_weight, vc_components["total_loss"].dtype)
                * vc_components["total_loss"]
            )
        else:
            vae_components = self._vae_components(outputs, training=False)
            subject_components = self._subject_components(
                outputs["pooled_z_mean"],
                subject_ids,
                training=False,
                use_grl=False,
            )
            dtype = vc_components["total_loss"].dtype
            total = (
                tf.cast(self.vc_loss_weight, dtype)
                * vc_components["total_loss"]
                + tf.cast(self.vae_loss_weight, dtype)
                * tf.cast(vae_components["vae_loss"], dtype)
                + tf.cast(self.subject_loss_weight, dtype)
                * tf.cast(subject_components["subject_loss"], dtype)
            )
        self._update_metrics(
            total_loss=total,
            vc_components=vc_components,
            outputs=outputs,
            y_flat=y_flat,
            sample_weight=sample_weight,
            vae_components=vae_components,
            subject_components=subject_components,
            vrex_components=None,
        )
        return {metric.name: metric.result() for metric in self.metrics}

    def predict_step(self, data):
        x = data[0] if isinstance(data, tuple) else data
        return self(x, training=False, sample_latent=False)

    def prepare_for_zero_shot_evaluation(self):
        """Restore source-evaluation semantics after a calibration fold.

        ``subject_calibration_cv`` restores source weights between folds.  This
        hook resets the loss/inference mode as well, so paired zero-shot metrics
        are evaluated as the original population model rather than in the
        calibration-only loss mode.
        """
        self.calibration_mode = False
        return self

    def prepare_for_subject_calibration(
        self,
        *,
        learning_rate: float,
        optimizer_name: str = "adamw",
        weight_decay: float = 0.0,
        unfreeze_layers: int | None = None,
    ):
        """Freeze the population model and unfreeze only the final k head layers.

        ``unfreeze_layers`` counts prediction layers backward from the output:
        1 means logits/softmax only, 2 means the final hidden dense block plus
        logits, and so on.  The VC target is frozen and acts as a source-trained
        target for any unfrozen hidden classification representation.
        """
        if unfreeze_layers is None:
            unfreeze_layers = self.calibration_unfreeze_layers
        unfreeze_layers = int(unfreeze_layers)
        max_layers = len(self.classification_dense_layers) + 1
        if not 1 <= unfreeze_layers <= max_layers:
            raise ValueError(
                f"unfreeze_layers must be in [1, {max_layers}], got "
                f"{unfreeze_layers}."
            )

        self.calibration_mode = True

        # Freeze every major source-trained subsystem first.
        self.graph_encoder.trainable = False
        for layer in self.temporal_bilstms:
            layer.trainable = False
        for layer in self.temporal_norms:
            layer.trainable = False
        for layer in self.temporal_dropouts:
            layer.trainable = False
        self.parallel_temporal_pool.trainable = False
        self.fusion_projection.trainable = False
        self.fusion_norm.trainable = False
        self.fusion_dropout_layer.trainable = False
        self.z_mean_projection.trainable = False
        self.z_log_var_projection.trainable = False
        self.decoder.trainable = False
        self.vc_target.trainable = False
        if self.subject_gradient_reversal is not None:
            self.subject_gradient_reversal.trainable = False
        if self.subject_hidden is not None:
            self.subject_hidden.trainable = False
        if self.subject_dropout_layer is not None:
            self.subject_dropout_layer.trainable = False
        if self.subject_logits_layer is not None:
            self.subject_logits_layer.trainable = False

        # Freeze the whole dense prediction head, then unfreeze exactly the
        # requested suffix.  A hidden block consists of Dense + LN + Dropout.
        for dense, norm, dropout in zip(
            self.classification_dense_layers,
            self.classification_norm_layers,
            self.classification_dropout_layers,
        ):
            dense.trainable = False
            norm.trainable = False
            dropout.trainable = False
        self.logits_layer.trainable = True

        hidden_to_unfreeze = max(0, unfreeze_layers - 1)
        if hidden_to_unfreeze:
            start = len(self.classification_dense_layers) - hidden_to_unfreeze
            for index in range(start, len(self.classification_dense_layers)):
                self.classification_dense_layers[index].trainable = True
                self.classification_norm_layers[index].trainable = True
                self.classification_dropout_layers[index].trainable = True

        optimizer = _build_optimizer(
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        # A fresh optimizer is required because each calibration fold restores
        # source weights and must not inherit optimizer moments from pretraining
        # or another target calibration fold.
        self.compile(
            main_optimizer=optimizer,
            vc_discriminator_optimizer=None,
            run_eagerly=False,
            jit_compile=False,
        )
        return {
            "calibration_unfreeze_layers": unfreeze_layers,
            "trainable_variables": [variable.name for variable in self.trainable_variables],
            "calibration_use_vc_target": self.calibration_use_vc_target,
        }

    def predict_mc_probabilities(self, inputs, n_samples: int = 30, seed=None):
        """Posterior predictive probabilities from VAE latent sampling."""
        if int(n_samples) < 1:
            raise ValueError("n_samples must be >= 1.")
        eeg_inputs, _ = self._split_eeg_and_subject_inputs(inputs)
        eeg_inputs = tf.convert_to_tensor(eeg_inputs, dtype=tf.float32)

        if self.classification_level == "window":
            flat_windows = eeg_inputs
            batch_size = tf.shape(eeg_inputs)[0]
            n_windows = None
        else:
            flat_windows, batch_size, n_windows = self._flatten_trial_windows(
                eeg_inputs
            )
        posterior = self._posterior_from_flat_windows(flat_windows, training=False)
        z_mean = posterior["z_mean"]
        z_log_var = posterior["z_log_var"]

        if seed is None:
            epsilon = tf.random.normal(
                tf.concat(
                    [tf.constant([int(n_samples)], dtype=tf.int32), tf.shape(z_mean)],
                    axis=0,
                ),
                dtype=z_mean.dtype,
            )
        else:
            if isinstance(seed, tuple):
                stateless_seed = tf.constant(seed, dtype=tf.int32)
            else:
                stateless_seed = tf.constant([int(seed), 0], dtype=tf.int32)
            epsilon = tf.random.stateless_normal(
                tf.concat(
                    [tf.constant([int(n_samples)], dtype=tf.int32), tf.shape(z_mean)],
                    axis=0,
                ),
                seed=stateless_seed,
                dtype=z_mean.dtype,
            )
        z_samples = z_mean[tf.newaxis, ...] + tf.exp(
            0.5 * z_log_var[tf.newaxis, ...]
        ) * epsilon

        # Pool each sampled latent sequence to one vector per window.
        pooled_windows = tf.reduce_mean(z_samples, axis=2)
        if self.classification_level == "trial":
            pooled_windows = tf.reshape(
                pooled_windows,
                [int(n_samples), batch_size, n_windows, self.z_dim],
            )
            pooled = tf.reduce_mean(pooled_windows, axis=2)
        else:
            pooled = pooled_windows

        sample_count = tf.shape(pooled)[0]
        sample_batch = tf.shape(pooled)[1]
        flat_pooled = tf.reshape(pooled, [sample_count * sample_batch, self.z_dim])
        embedding, logits = self._classifier_forward(flat_pooled, training=False)
        del embedding
        probabilities = tf.nn.softmax(logits, axis=-1)
        probabilities = tf.reshape(
            probabilities,
            [sample_count, sample_batch, self.n_classes],
        )
        return {
            "probability_samples": probabilities,
            "mean_probabilities": tf.reduce_mean(probabilities, axis=0),
        }

    def get_latent_distribution(self, inputs):
        """Return the counterfactual VAE posterior and pooled representation."""
        eeg_inputs, _ = self._split_eeg_and_subject_inputs(inputs)
        outputs = self._encode(
            eeg_inputs,
            training=False,
            sample_latent=False,
        )
        return {
            "z_mean": outputs["z_mean"],
            "z_log_var": outputs["z_log_var"],
            "pooled_z": outputs["pooled_z_mean"],
            "classification_embedding": outputs["classification_embedding"],
            "probabilities": outputs["probabilities"],
        }

    def decode_latent(self, latent_sequence):
        return self.decoder(latent_sequence, training=False)

    def get_adjacency_matrices(self):
        return {
            "mtl_raw_adjacency": self.graph_encoder.get_raw_adjacency_matrix(),
            "mtl_normalized_adjacency": self.graph_encoder.get_adjacency_matrix(),
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "graph_encoder": _serialize_keras_object(self.graph_encoder),
                "decoder": _serialize_keras_object(self.decoder),
                "classification_level": self.classification_level,
                "n_classes": self.n_classes,
                "bilstm_units": self.bilstm_units,
                "n_bilstm_layers": self.n_bilstm_layers,
                "bilstm_dropout": self.bilstm_dropout_rate,
                "architecture_mode": self.architecture_mode,
                "fusion_units": self.fusion_units,
                "fusion_dropout": self.fusion_dropout_rate,
                "temporal_downsample_factor": self.temporal_downsample_factor,
                "z_dim": self.z_dim,
                "z_log_var_clip_min": self.z_log_var_clip_min,
                "z_log_var_clip_max": self.z_log_var_clip_max,
                "classification_hidden_units": self.classification_hidden_units,
                "classification_dropout": self.classification_dropout_rate,
                "activation": self.activation_name,
                "label_smoothing": self.label_smoothing,
                "focal_gamma": self.focal_gamma,
                "focal_alpha": self.focal_alpha,
                "vc_loss_weight": self.vc_loss_weight,
                "vc_alpha": self.vc_alpha,
                "vc_beta": self.vc_beta,
                "vc_gamma": self.vc_gamma,
                "vc_lambda": self.vc_lambda,
                "update_vc_discriminator": self.update_vc_discriminator,
                "vae_loss_weight": self.vae_loss_weight,
                "vae_beta": self.vae_beta,
                "use_vrex": self.use_vrex,
                "vrex_penalty_weight": self.vrex_penalty_weight,
                "use_subject_adversarial": self.subject_adversarial_enabled,
                "n_subject_classes": self.n_subject_classes,
                "subject_adversarial_weight": self.subject_adversarial_weight,
                "subject_loss_weight": self.subject_loss_weight,
                "subject_hidden_units": self.subject_hidden_units,
                "subject_dropout": self.subject_dropout_rate,
                "calibration_unfreeze_layers": self.calibration_unfreeze_layers,
                "calibration_use_vc_target": self.calibration_use_vc_target,
                "calibration_vc_alpha": self.calibration_vc_alpha,
                "calibration_vc_beta": self.calibration_vc_beta,
                "calibration_vc_gamma": self.calibration_vc_gamma,
                "calibration_vc_lambda": self.calibration_vc_lambda,
                "use_class_weight": self.use_class_weight,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config["graph_encoder"] = _deserialize_keras_object(config["graph_encoder"])
        config["decoder"] = _deserialize_keras_object(config["decoder"])
        return cls(**config)


def build_sic_model(
    input_shape,
    *,
    training_features=None,
    training_labels=None,
    training_subject_ids=None,
    training_trial_ids=None,
    adjacency=None,
    classification_level: str = "trial",
    n_classes: int = 2,
    n_channels: int = 14,
    n_bands: int = 3,
    t_down: int = 2,
    temporal_pool_sizes: Sequence[int] | None = (2,),
    gcn_units: Sequence[int] = (64, 32),
    gcn_dropout: float = 0.10,
    gcn_activation: str = "relu",
    gcn_use_batch_norm: bool = False,
    spectral_gru_units: int = 384,
    spectral_gru_dropout: float = 0.20,
    graph_add_self_loops: bool = True,
    graph_symmetrize: bool = True,
    graph_epsilon: float = 1e-8,
    mi_n_neighbors: int = 3,
    mi_random_state: int = 42,
    mi_zero_diagonal: bool = False,
    mi_band_reduction: str = "mean",
    mi_max_observations: int | None = 15000,
    bilstm_units: int = 128,
    n_bilstm_layers: int = 1,
    bilstm_dropout: float = 0.30,
    architecture_mode: str = "serial",
    fusion_units: int = 128,
    fusion_dropout: float = 0.20,
    z_dim: int = 64,
    z_log_var_clip_min: float = -20.0,
    z_log_var_clip_max: float = 20.0,
    classification_hidden_units: Sequence[int] = (128,),
    classification_dropout: float = 0.20,
    activation: str = "relu",
    label_smoothing: float = 0.0,
    focal_gamma: float = 1.0,
    focal_alpha: float | None = None,
    vc_loss_weight: float = 1.0,
    vc_alpha: float = 1.0,
    vc_beta: float = 0.5,
    vc_gamma: float = 0.0,
    vc_lambda: float = 0.0,
    update_vc_discriminator: bool = False,
    vae_loss_weight: float = 0.10,
    vae_beta: float = 0.05,
    use_vrex: bool = False,
    vrex_penalty_weight: float = 1.0,
    use_subject_adversarial: bool = True,
    n_subject_classes: int | None = None,
    subject_adversarial_weight: float = 0.8,
    subject_loss_weight: float = 1.0,
    subject_hidden_units: int = 64,
    subject_dropout: float = 0.0,
    calibration_unfreeze_layers: int = 1,
    calibration_use_vc_target: bool = True,
    calibration_vc_alpha: float | None = None,
    calibration_vc_beta: float | None = None,
    calibration_vc_gamma: float | None = None,
    calibration_vc_lambda: float | None = None,
    decoder_dropout: float = 0.10,
    optimizer_name: str = "adamw",
    learning_rate: float = 1e-4,
    vc_discriminator_learning_rate: float | None = None,
    weight_decay: float = 5e-5,
    use_class_weight: bool = False,
    model_name: str = "sic_subject_invariant_calibrator",
    **unused_kwargs,
) -> SICModel:
    """Build SIC, computing its fixed MI graph from source data when needed.

    The ``training_*`` arguments are accepted explicitly for
    ``subject_calibration_cv``. Only ``training_features`` and
    ``training_subject_ids`` are needed by this builder; labels/trial IDs are
    accepted to keep the builder contract uniform and to make leakage auditing
    straightforward.
    """
    del training_labels, training_trial_ids, unused_kwargs

    classification_level = str(classification_level).lower()
    input_shape = tuple(int(value) for value in input_shape)
    if classification_level == "window":
        if len(input_shape) != 2:
            raise ValueError(
                "Window-level v6 expects input_shape=(timesteps, features); "
                f"got {input_shape}."
            )
        timesteps, n_features = input_shape
        dummy_shape = (1, timesteps, n_features)
    elif classification_level == "trial":
        if len(input_shape) != 3:
            raise ValueError(
                "Trial-level v6 expects input_shape=(windows, timesteps, features); "
                f"got {input_shape}."
            )
        n_trial_windows, timesteps, n_features = input_shape
        dummy_shape = (1, n_trial_windows, timesteps, n_features)
    else:
        raise ValueError("classification_level must be 'window' or 'trial'.")

    expected_features = int(n_channels) * int(n_bands)
    if n_features != expected_features:
        raise ValueError(
            f"Input features={n_features}, expected {n_channels}*{n_bands}="
            f"{expected_features}."
        )
    pools = _resolve_temporal_pool_sizes(temporal_pool_sizes, t_down)
    gcn_units = _as_positive_tuple("gcn_units", gcn_units)

    if adjacency is None:
        if training_features is None:
            raise ValueError(
                "SIC requires either adjacency=... or training_features=... so "
                "the MTLFuseNet MI graph can be estimated from source data only."
            )
        adjacency = _source_only_mi_adjacency(
            training_features,
            n_channels=int(n_channels),
            n_bands=int(n_bands),
            n_neighbors=int(mi_n_neighbors),
            random_state=int(mi_random_state),
            zero_diagonal=bool(mi_zero_diagonal),
            band_reduction=str(mi_band_reduction),
            max_observations=mi_max_observations,
        )
    else:
        adjacency = np.asarray(adjacency, dtype=np.float32)

    if n_subject_classes is None and training_subject_ids is not None:
        n_subject_classes = int(len(np.unique(np.asarray(training_subject_ids))))

    graph_encoder = GCNMTLEncoder(
        timesteps=int(timesteps),
        t_down=int(t_down),
        adjacency=adjacency,
        n_channels=int(n_channels),
        n_bands=int(n_bands),
        gcn_units=gcn_units,
        temporal_pool_sizes=pools,
        emb_dim=None,
        dropout=float(gcn_dropout),
        activation=str(gcn_activation),
        use_batch_norm=bool(gcn_use_batch_norm),
        use_spectral_gru=True,
        spectral_gru_units=int(spectral_gru_units),
        spectral_gru_dropout=float(spectral_gru_dropout),
        graph_add_self_loops=bool(graph_add_self_loops),
        graph_symmetrize=bool(graph_symmetrize),
        graph_epsilon=float(graph_epsilon),
        name="v6_mtl_gcn_gru_encoder",
    )

    decoder = GCNMTLDecoder(
        timesteps=int(timesteps),
        n_channels=int(n_channels),
        n_bands=int(n_bands),
        t_down=int(t_down),
        gcn_units=gcn_units,
        temporal_pool_sizes=pools,
        adjacency=adjacency,
        emb_dim=int(z_dim),
        dropout=float(decoder_dropout),
        activation=str(activation),
        use_batch_norm=bool(gcn_use_batch_norm),
        graph_add_self_loops=bool(graph_add_self_loops),
        graph_symmetrize=bool(graph_symmetrize),
        graph_epsilon=float(graph_epsilon),
        name="sic_mtl_graph_decoder",
    )

    model = SICModel(
        graph_encoder=graph_encoder,
        decoder=decoder,
        classification_level=classification_level,
        n_classes=int(n_classes),
        bilstm_units=int(bilstm_units),
        n_bilstm_layers=int(n_bilstm_layers),
        bilstm_dropout=float(bilstm_dropout),
        architecture_mode=str(architecture_mode),
        fusion_units=int(fusion_units),
        fusion_dropout=float(fusion_dropout),
        temporal_downsample_factor=int(t_down),
        z_dim=int(z_dim),
        z_log_var_clip_min=float(z_log_var_clip_min),
        z_log_var_clip_max=float(z_log_var_clip_max),
        classification_hidden_units=tuple(
            int(value) for value in classification_hidden_units
        ),
        classification_dropout=float(classification_dropout),
        activation=str(activation),
        label_smoothing=float(label_smoothing),
        focal_gamma=float(focal_gamma),
        focal_alpha=focal_alpha,
        vc_loss_weight=float(vc_loss_weight),
        vc_alpha=float(vc_alpha),
        vc_beta=float(vc_beta),
        vc_gamma=float(vc_gamma),
        vc_lambda=float(vc_lambda),
        update_vc_discriminator=bool(update_vc_discriminator),
        vae_loss_weight=float(vae_loss_weight),
        vae_beta=float(vae_beta),
        use_vrex=bool(use_vrex),
        vrex_penalty_weight=float(vrex_penalty_weight),
        use_subject_adversarial=bool(use_subject_adversarial),
        n_subject_classes=n_subject_classes,
        subject_adversarial_weight=float(subject_adversarial_weight),
        subject_loss_weight=float(subject_loss_weight),
        subject_hidden_units=int(subject_hidden_units),
        subject_dropout=float(subject_dropout),
        calibration_unfreeze_layers=int(calibration_unfreeze_layers),
        calibration_use_vc_target=bool(calibration_use_vc_target),
        calibration_vc_alpha=calibration_vc_alpha,
        calibration_vc_beta=calibration_vc_beta,
        calibration_vc_gamma=calibration_vc_gamma,
        calibration_vc_lambda=calibration_vc_lambda,
        use_class_weight=bool(use_class_weight),
        name=model_name,
    )

    main_optimizer = _build_optimizer(
        optimizer_name=optimizer_name,
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    resolved_disc_lr = (
        float(learning_rate)
        if vc_discriminator_learning_rate is None
        else float(vc_discriminator_learning_rate)
    )
    vc_discriminator_optimizer = (
        _build_optimizer(
            optimizer_name=optimizer_name,
            learning_rate=resolved_disc_lr,
            weight_decay=float(weight_decay),
        )
        if bool(update_vc_discriminator)
        else None
    )

    model.compile(
        main_optimizer=main_optimizer,
        vc_discriminator_optimizer=vc_discriminator_optimizer,
        run_eagerly=False,
        jit_compile=False,
    )

    # Build every stateful branch before the first fit. This avoids Keras 3
    # creating new nested-layer variables after the outer model has been marked
    # built (the failure mode that affected earlier subject-adversarial models).
    if bool(use_subject_adversarial) and n_subject_classes is not None:
        _ = model._subject_logits(
            tf.zeros((1, int(z_dim)), dtype=tf.float32),
            training=False,
            use_grl=False,
        )
    latent_timesteps = int(np.ceil(float(timesteps) / float(t_down)))
    _ = decoder(
        tf.zeros((1, latent_timesteps, int(z_dim)), dtype=tf.float32),
        training=False,
    )
    if not bool(use_subject_adversarial) or n_subject_classes is not None:
        _ = model(tf.zeros(dummy_shape, dtype=tf.float32), training=False)
    return model


# Compatibility aliases for earlier v6 naming.
JointV6Model = SICModel
JointSTSModelV6 = SICModel
build_joint_v6_model = build_sic_model
build_joint_sts_model_v6 = build_sic_model
