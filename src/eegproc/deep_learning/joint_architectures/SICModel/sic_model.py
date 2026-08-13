"""SIC: Subject Invariant Calibrator model.

Parallel MTLFuseNet GCN-GRU and temporal BiLSTM encoders with head fusion.

Architecture
------------
For each EEG window, two independent deterministic encoders produce feature
sequences::

    channel-band EEG -> fixed-MI GCN -> spectral GRU -> h_g
    channel-band EEG -> temporal BiLSTM              -> h_b

After matching their temporal lengths, the complete encoder feature vectors
are concatenated directly. There is no learned projection, bottleneck, or
fusion network between the encoders and the prediction heads::

    h_joint = concat([h_g, h_b]) -> pooling -> dense feature blocks -> VC logits

With the default widths, h_g has 384 features and h_b has 2 * 128 = 256
features, so both active branches give the final heads all 640 features. Fusion
occurs in the classifier and subject-adversarial heads. EEGProc's existing
VariationalClassifier is the sole logits-producing classification head; there
is no parallel dense-logit head. Focal classification and the VC regularizers
are optimized jointly from that one head and its one logits tensor. The VC is
not an encoder VAE.

There is no encoder posterior, reparameterization, KL loss, reconstruction
loss, or decoder. Subject invariance can be imposed directly on the pooled
concatenated deterministic representation with ordinary gradient reversal.

V-REx can additionally regularize source training.  Each source subject present
in a minibatch is treated as an environment.  SIC computes the focal risk
separately for each subject and adds the variance of those risks to the ordinary
source objective.  This uses the same batched forward pass and a single
backward/optimizer step.

First-order MLDG is available as a mutually exclusive source-training method.
Every source epoch is composed of subject-disjoint meta-train/meta-test
episodes.  A temporary plain-SGD step is taken on the meta-train objective,
the meta-test focal-classification gradient is evaluated at those adapted
parameters, and the original parameters receive one combined outer-optimizer
update.  The temporary assignment is detached, so no Hessian or
gradient-through-gradient path is constructed.

Subject calibration
-------------------
``prepare_for_subject_calibration`` freezes the complete representation and
subject adversary. It keeps the VC classification head trainable and unfreezes
only the requested dense suffix before that head:

    1 -> VC logits head only
    2 -> last dense hidden block + VC logits head
    3 -> last two dense hidden blocks + VC logits head
    ...

The calibration train step uses focal loss and, when enabled, the VC
regularizers computed jointly with the same VC logits. The VC head therefore
adapts in every calibration fold instead of acting as a frozen source target.
For backward CLI compatibility, ``calibration_use_vc_target=false`` disables
only the auxiliary VC regularizers; it does not replace or freeze the VC
classification head that produces the logits.

This file is designed for ``cross_val.subject_calibration_cv``.  Its builder
accepts ``training_features`` and computes the fixed MI adjacency from source
subjects only, avoiding target-subject leakage.

Ablations are controlled by ``use_gcn_gru_branch`` and
``use_bilstm_branch``. At least one encoder branch must remain enabled.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import tensorflow as tf

try:
    from ...generalize_optimization_strats.MetaLearning import (
        fit_sic_mldg,
        run_sic_mldg_train_step,
    )
except ImportError:
    from eegproc.deep_learning.generalize_optimization_strats.MetaLearning import (
        fit_sic_mldg,
        run_sic_mldg_train_step,
    )

try:
    from ...supervised.variational_classifier import VariationalClassifier
    from ...unsupervised.GNN.GCNMTL import (
        GCNMTLEncoder,
        compute_mtl_shared_mi_adjacency,
    )
except ImportError:
    from eegproc.deep_learning.supervised.variational_classifier import (
        VariationalClassifier,
    )
    from eegproc.deep_learning.unsupervised.GNN.GCNMTL import (
        GCNMTLEncoder,
        compute_mtl_shared_mi_adjacency,
    )


SIC_BUILDER_API_VERSION = 10
JOINT_V6_BUILDER_API_VERSION = SIC_BUILDER_API_VERSION


_TRAINING_METHOD_ALIASES = {
    "normal": "erm",
    "standard": "erm",
    "joint": "erm",
    "erm": "erm",
    "vrex": "vrex",
    "v-rex": "vrex",
    "v_rex": "vrex",
    "mldg": "mldg",
    "fo_mldg": "mldg",
    "first_order_mldg": "mldg",
}


def _resolve_training_method(
    training_method: str | None,
    use_vrex: bool,
) -> str:
    """Resolve the new method selector while preserving old V-REx configs."""
    if training_method is None:
        return "vrex" if bool(use_vrex) else "erm"
    normalized = str(training_method).strip().lower().replace("-", "_")
    normalized = _TRAINING_METHOD_ALIASES.get(normalized, normalized)
    if normalized not in {"erm", "vrex", "mldg"}:
        raise ValueError(
            "training_method must be one of 'erm', 'vrex', or 'mldg'; "
            f"got {training_method!r}."
        )
    if bool(use_vrex) and normalized != "vrex":
        raise ValueError(
            "use_vrex=true conflicts with training_method="
            f"{training_method!r}. Use training_method='vrex' or remove the "
            "legacy use_vrex setting."
        )
    return normalized


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
class StreamingBalancedAccuracy(tf.keras.metrics.Metric):
    """Balanced accuracy accumulated from one epoch-wide confusion matrix."""

    def __init__(self, n_classes: int = 2, name="balanced_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_classes = int(n_classes)
        if self.n_classes < 2:
            raise ValueError("n_classes must be >= 2.")
        self.confusion_matrix = self.add_weight(
            name="confusion_matrix",
            shape=(self.n_classes, self.n_classes),
            initializer="zeros",
            dtype=tf.float32,
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.convert_to_tensor(y_pred)
        if y_pred.shape.rank is not None and y_pred.shape.rank > 1:
            y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        else:
            y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.int32)
        y_pred = tf.reshape(y_pred, [-1])

        weights = None
        if sample_weight is not None:
            weights = tf.cast(tf.reshape(sample_weight, [-1]), tf.float32)

        matrix = tf.math.confusion_matrix(
            y_true,
            y_pred,
            num_classes=self.n_classes,
            weights=weights,
            dtype=tf.float32,
        )
        self.confusion_matrix.assign_add(matrix)

    def result(self):
        true_counts = tf.reduce_sum(self.confusion_matrix, axis=1)
        recalls = tf.math.divide_no_nan(
            tf.linalg.diag_part(self.confusion_matrix),
            true_counts,
        )
        return tf.reduce_mean(recalls)

    def reset_state(self):
        self.confusion_matrix.assign(tf.zeros_like(self.confusion_matrix))

    def get_config(self):
        config = super().get_config()
        config.update({"n_classes": self.n_classes})
        return config


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
    """SIC with direct deterministic GCN-GRU/BiLSTM feature concatenation."""

    def __init__(
        self,
        *,
        graph_encoder: GCNMTLEncoder | None,
        classification_level: str = "trial",
        n_classes: int = 2,
        gcn_gru_feature_dim: int = 384,
        bilstm_units: int = 128,
        n_bilstm_layers: int = 1,
        bilstm_dropout: float = 0.30,
        use_gcn_gru_branch: bool = True,
        use_bilstm_branch: bool = True,
        temporal_downsample_factor: int = 1,
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
        training_method: str | None = None,
        use_vrex: bool = False,
        vrex_penalty_weight: float = 1.0,
        mldg_meta_train_subjects: int | None = None,
        mldg_meta_test_subjects: int = 4,
        mldg_trials_per_subject: int = 1,
        mldg_steps_per_epoch: int | None = None,
        mldg_inner_learning_rate: float = 1e-4,
        mldg_meta_test_weight: float = 1.0,
        mldg_seed: int | None = 42,
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
        use_gcn_gru_branch = bool(use_gcn_gru_branch)
        use_bilstm_branch = bool(use_bilstm_branch)
        if not use_gcn_gru_branch and not use_bilstm_branch:
            raise ValueError(
                "At least one of use_gcn_gru_branch/use_bilstm_branch must be true."
            )
        if use_gcn_gru_branch and graph_encoder is None:
            raise ValueError("graph_encoder is required when use_gcn_gru_branch=true.")
        if int(n_classes) < 2:
            raise ValueError("n_classes must be >= 2.")
        if int(gcn_gru_feature_dim) < 1:
            raise ValueError("gcn_gru_feature_dim must be positive.")
        if int(bilstm_units) < 1 or int(n_bilstm_layers) < 1:
            raise ValueError("BiLSTM dimensions must be positive.")
        if int(temporal_downsample_factor) < 1:
            raise ValueError("temporal_downsample_factor must be >= 1.")
        if float(focal_gamma) < 0.0:
            raise ValueError("focal_gamma must be non-negative.")
        if focal_alpha is not None and not 0.0 <= float(focal_alpha) <= 1.0:
            raise ValueError("focal_alpha must be in [0, 1] or None.")
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
        resolved_training_method = _resolve_training_method(
            training_method,
            use_vrex,
        )
        if mldg_meta_train_subjects is not None and int(mldg_meta_train_subjects) < 1:
            raise ValueError("mldg_meta_train_subjects must be >= 1 or null.")
        if int(mldg_meta_test_subjects) < 1:
            raise ValueError("mldg_meta_test_subjects must be >= 1.")
        if int(mldg_trials_per_subject) < 1:
            raise ValueError("mldg_trials_per_subject must be >= 1.")
        if mldg_steps_per_epoch is not None and int(mldg_steps_per_epoch) < 1:
            raise ValueError("mldg_steps_per_epoch must be >= 1 or null.")
        if float(mldg_inner_learning_rate) <= 0.0:
            raise ValueError("mldg_inner_learning_rate must be positive.")
        if float(mldg_meta_test_weight) < 0.0:
            raise ValueError("mldg_meta_test_weight must be non-negative.")
        calibration_unfreeze_layers = int(calibration_unfreeze_layers)
        max_calibration_layers = len(hidden_units) + 1
        if not 1 <= calibration_unfreeze_layers <= max_calibration_layers:
            raise ValueError(
                "calibration_unfreeze_layers must be between 1 and "
                f"{max_calibration_layers}; got {calibration_unfreeze_layers}."
            )

        self.graph_encoder = graph_encoder
        self.classification_level = classification_level
        self.n_classes = int(n_classes)
        self.gcn_gru_feature_dim = int(gcn_gru_feature_dim)
        self.bilstm_units = int(bilstm_units)
        self.bilstm_feature_dim = 2 * self.bilstm_units
        self.n_bilstm_layers = int(n_bilstm_layers)
        self.bilstm_dropout_rate = float(bilstm_dropout)
        self.use_gcn_gru_branch = use_gcn_gru_branch
        self.use_bilstm_branch = use_bilstm_branch
        self.temporal_downsample_factor = int(temporal_downsample_factor)
        self.combined_feature_dim = (
            self.gcn_gru_feature_dim * int(self.use_gcn_gru_branch)
            + self.bilstm_feature_dim * int(self.use_bilstm_branch)
        )
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
        self.use_class_weight = bool(use_class_weight)
        self.supports_latent_sampling = False

        self.training_method = resolved_training_method
        self.use_vrex = self.training_method == "vrex"
        self.use_mldg = self.training_method == "mldg"
        self.vrex_penalty_weight = float(vrex_penalty_weight)
        self.mldg_meta_train_subjects = (
            None if mldg_meta_train_subjects is None else int(mldg_meta_train_subjects)
        )
        self.mldg_meta_test_subjects = int(mldg_meta_test_subjects)
        self.mldg_trials_per_subject = int(mldg_trials_per_subject)
        self.mldg_steps_per_epoch = (
            None if mldg_steps_per_epoch is None else int(mldg_steps_per_epoch)
        )
        self.mldg_inner_learning_rate = float(mldg_inner_learning_rate)
        self.mldg_meta_test_weight = float(mldg_meta_test_weight)
        self.mldg_seed = None if mldg_seed is None else int(mldg_seed)

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
            self.vc_alpha
            if calibration_vc_alpha is None
            else float(calibration_vc_alpha)
        )
        self.calibration_vc_beta = (
            self.vc_beta if calibration_vc_beta is None else float(calibration_vc_beta)
        )
        self.calibration_vc_gamma = (
            self.vc_gamma
            if calibration_vc_gamma is None
            else float(calibration_vc_gamma)
        )
        self.calibration_vc_lambda = (
            self.vc_lambda
            if calibration_vc_lambda is None
            else float(calibration_vc_lambda)
        )
        self.calibration_mode = False

        self.requires_subject_ids = (
            self.subject_adversarial_enabled or self.use_vrex or self.use_mldg
        )
        # Current EEGProc cross_val uses ``use_subject_adversarial`` as the
        # compatibility gate for attaching source subject IDs.  Keep that gate
        # true whenever V-REx or MLDG needs subject metadata; actual GRL
        # computation is still controlled exclusively by
        # ``subject_adversarial_enabled``.
        self.use_subject_adversarial = self.requires_subject_ids
        self._source_subject_ids = None
        self._source_trial_ids = None

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

        # The BiLSTM is an independent raw-EEG branch. Pool only its temporal
        # axis so it can be concatenated position-for-position with the
        # reduced-time GCN-GRU output. Feature widths are left untouched.
        self.parallel_temporal_pool = tf.keras.layers.AveragePooling1D(
            pool_size=self.temporal_downsample_factor,
            strides=self.temporal_downsample_factor,
            padding="same",
            name="v6_parallel_bilstm_pool",
        )
        self.feature_pool = tf.keras.layers.GlobalAveragePooling1D(
            name="v6_feature_pool"
        )

        # Dense feature blocks followed by one VC logits head. The blocks are
        # explicit so calibration can unfreeze exactly the last k layers.
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
        vc_dim = (
            self.classification_hidden_units[-1]
            if self.classification_hidden_units
            else self.combined_feature_dim
        )
        vc_focal_alpha = (
            None
            if self.focal_alpha is None
            else (1.0 - self.focal_alpha, self.focal_alpha)
        )
        self.vc_target = VariationalClassifier(
            n_classes=self.n_classes,
            latent_dim=vc_dim,
            label_smoothing=self.label_smoothing,
            focal_gamma=self.focal_gamma,
            focal_alpha=vc_focal_alpha,
            name="v6_vc_target",
        )
        # This is the sole classification head. Explicitly build its learned
        # Gaussian class parameters and optional discriminator parameters.
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
        self.focal_loss_tracker = tf.keras.metrics.Mean(name="focal_loss")
        metric_prefix = self.classification_level
        self.accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(
            name=f"{metric_prefix}_accuracy"
        )
        self.balanced_accuracy_tracker = StreamingBalancedAccuracy(
            n_classes=self.n_classes,
            name=f"{metric_prefix}_balanced_accuracy",
        )
        # Exact epoch-wide class fractions. During validation Keras exposes
        # these as val_predicted_class_1_fraction / val_true_class_1_fraction.
        self.predicted_class_1_fraction_tracker = tf.keras.metrics.Mean(
            name="predicted_class_1_fraction"
        )
        self.true_class_1_fraction_tracker = tf.keras.metrics.Mean(
            name="true_class_1_fraction"
        )
        self.vrex_penalty_tracker = tf.keras.metrics.Mean(name="vrex_penalty")
        self.vrex_subject_risk_mean_tracker = tf.keras.metrics.Mean(
            name="vrex_subject_risk_mean"
        )
        self.vrex_subjects_per_batch_tracker = tf.keras.metrics.Mean(
            name="vrex_subjects_per_batch"
        )
        self.mldg_meta_train_loss_tracker = tf.keras.metrics.Mean(
            name="mldg_meta_train_loss"
        )
        self.mldg_meta_test_loss_tracker = tf.keras.metrics.Mean(
            name="mldg_meta_test_loss"
        )
        self.mldg_meta_train_subjects_tracker = tf.keras.metrics.Mean(
            name="mldg_meta_train_subjects"
        )
        self.mldg_meta_test_subjects_tracker = tf.keras.metrics.Mean(
            name="mldg_meta_test_subjects"
        )
        self.mldg_gradient_cosine_tracker = tf.keras.metrics.Mean(
            name="mldg_gradient_cosine"
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
            self.focal_loss_tracker,
            self.accuracy_tracker,
            self.balanced_accuracy_tracker,
            self.predicted_class_1_fraction_tracker,
            self.true_class_1_fraction_tracker,
        ]
        if self.use_vrex:
            output.extend(
                [
                    self.vrex_penalty_tracker,
                    self.vrex_subject_risk_mean_tracker,
                    self.vrex_subjects_per_batch_tracker,
                ]
            )
        if self.use_mldg:
            output.extend(
                [
                    self.mldg_meta_train_loss_tracker,
                    self.mldg_meta_test_loss_tracker,
                    self.mldg_meta_train_subjects_tracker,
                    self.mldg_meta_test_subjects_tracker,
                    self.mldg_gradient_cosine_tracker,
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

    def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        epochs=1,
        verbose="auto",
        callbacks=None,
        validation_split=0.0,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_batch_size=None,
        validation_freq=1,
        **kwargs,
    ):
        if not self.use_class_weight:
            class_weight = None
        if not self.use_mldg or self.calibration_mode:
            return super().fit(
                x=x,
                y=y,
                batch_size=batch_size,
                epochs=epochs,
                verbose=verbose,
                callbacks=callbacks,
                validation_split=validation_split,
                validation_data=validation_data,
                shuffle=shuffle,
                class_weight=class_weight,
                sample_weight=sample_weight,
                initial_epoch=initial_epoch,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                validation_batch_size=validation_batch_size,
                validation_freq=validation_freq,
                **kwargs,
            )

        return fit_sic_mldg(
            self,
            x=x,
            y=y,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            validation_split=validation_split,
            validation_data=validation_data,
            class_weight=class_weight,
            sample_weight=sample_weight,
            initial_epoch=initial_epoch,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            validation_batch_size=validation_batch_size,
            validation_freq=validation_freq,
            **kwargs,
        )

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

    def set_source_training_metadata(self, subject_ids, trial_ids):
        """Retain source-only IDs so MLDG can keep complete trials together."""
        if subject_ids is None or trial_ids is None:
            self._source_subject_ids = None
            self._source_trial_ids = None
            return
        subjects = np.asarray(subject_ids).reshape(-1)
        trials = np.asarray(trial_ids).reshape(-1)
        if len(subjects) != len(trials):
            raise ValueError("Source subject IDs and trial IDs must align.")
        self._source_subject_ids = subjects.copy()
        self._source_trial_ids = trials.copy()

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
        output = {"eeg": eeg_array, "subject_id": remapped}
        matching_trial_ids = None
        if self._source_subject_ids is not None and self._source_trial_ids is not None:
            if len(subjects) == len(self._source_subject_ids) and np.array_equal(
                subjects,
                self._source_subject_ids,
            ):
                matching_trial_ids = self._source_trial_ids
            else:
                # cross_val may build from all source subjects and subsequently
                # reserve whole validation subjects. Boolean subject filtering
                # preserves row order, so recover the aligned trial IDs without
                # exposing target-subject data or guessing individual windows.
                source_mask = np.isin(
                    self._source_subject_ids,
                    np.unique(subjects),
                )
                candidate_subjects = self._source_subject_ids[source_mask]
                if len(candidate_subjects) == len(subjects) and np.array_equal(
                    candidate_subjects,
                    subjects,
                ):
                    matching_trial_ids = self._source_trial_ids[source_mask]
        if self.use_mldg and matching_trial_ids is not None:
            output["trial_id"] = np.asarray(matching_trial_ids).copy()
        return output

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

    def _temporal_encode(self, eeg_sequence, training: bool):
        x = eeg_sequence
        for bilstm, norm, dropout in zip(
            self.temporal_bilstms,
            self.temporal_norms,
            self.temporal_dropouts,
        ):
            x = bilstm(x, training=training)
            x = norm(x)
            x = dropout(x, training=training)
        return x

    def _features_from_flat_windows(self, flat_windows, training: bool):
        graph_sequence = (
            self.graph_encoder(flat_windows, training=training)
            if self.use_gcn_gru_branch
            else None
        )
        bilstm_sequence = (
            self._temporal_encode(flat_windows, training=training)
            if self.use_bilstm_branch
            else None
        )
        pooled_bilstm = (
            self.parallel_temporal_pool(bilstm_sequence)
            if bilstm_sequence is not None
            else None
        )

        if graph_sequence is not None and pooled_bilstm is not None:
            tf.debugging.assert_equal(
                tf.shape(graph_sequence)[1],
                tf.shape(pooled_bilstm)[1],
                message=(
                    "GCN-GRU and BiLSTM feature sequence lengths do not match."
                ),
            )
        feature_parts = [
            value
            for value in (graph_sequence, pooled_bilstm)
            if value is not None
        ]
        combined_sequence = (
            feature_parts[0]
            if len(feature_parts) == 1
            else tf.concat(feature_parts, axis=-1)
        )
        tf.debugging.assert_equal(
            tf.shape(combined_sequence)[-1],
            self.combined_feature_dim,
            message="Concatenated encoder feature width is inconsistent.",
        )
        return {
            "graph_sequence": graph_sequence,
            "bilstm_sequence": bilstm_sequence,
            "pooled_bilstm_sequence": pooled_bilstm,
            "combined_feature_sequence": combined_sequence,
        }

    def _pool_features_for_prediction(
        self,
        flat_features,
        batch_size=None,
        n_windows=None,
    ):
        window_features = self.feature_pool(flat_features)
        if self.classification_level == "window":
            return window_features, window_features
        window_features = tf.reshape(
            window_features,
            [batch_size, n_windows, self.combined_feature_dim],
        )
        trial_features = tf.reduce_mean(window_features, axis=1)
        return trial_features, window_features

    def _classifier_forward(self, pooled_features, training: bool):
        x = pooled_features
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
        logits = self.vc_target(
            classification_embedding,
            training=bool(training),
        )
        return classification_embedding, logits

    def _encode(
        self,
        eeg_inputs,
        *,
        training: bool,
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

        encoder_outputs = self._features_from_flat_windows(
            flat_windows,
            training=training,
        )
        pooled_features, window_features = self._pool_features_for_prediction(
            encoder_outputs["combined_feature_sequence"],
            batch_size=batch_size,
            n_windows=n_windows,
        )
        classification_embedding, logits = self._classifier_forward(
            pooled_features,
            training=bool(classifier_training),
        )
        encoder_outputs.update(
            {
                "flat_windows": flat_windows,
                "pooled_features": pooled_features,
                "window_features": window_features,
                "classification_embedding": classification_embedding,
                "logits": logits,
                "probabilities": tf.nn.softmax(logits, axis=-1),
                "batch_size": batch_size,
                "n_windows": n_windows,
            }
        )
        return encoder_outputs

    def call(self, inputs, training=False):
        eeg_inputs, _ = self._split_eeg_and_subject_inputs(inputs)
        outputs = self._encode(
            eeg_inputs,
            training=bool(training),
        )
        return outputs["logits"]

    def _per_sample_focal_loss(self, logits, y_flat):
        """Focal loss from the same sole VC logits used for prediction."""
        y_flat = tf.cast(tf.reshape(y_flat, [-1]), tf.int32)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        probs = tf.exp(log_probs)
        row_ids = tf.range(tf.shape(y_flat)[0], dtype=tf.int32)
        gather_ids = tf.stack([row_ids, y_flat], axis=1)
        p_t = tf.gather_nd(probs, gather_ids)
        log_p_t = tf.gather_nd(log_probs, gather_ids)
        modulating = tf.pow(
            tf.maximum(1.0 - p_t, tf.keras.backend.epsilon()),
            tf.cast(self.focal_gamma, p_t.dtype),
        )
        if self.label_smoothing:
            hard_targets = tf.one_hot(
                y_flat,
                depth=self.n_classes,
                dtype=logits.dtype,
            )
            smoothing = tf.cast(self.label_smoothing, logits.dtype)
            smoothed_targets = (
                (1.0 - smoothing) * hard_targets
                + smoothing / tf.cast(self.n_classes, logits.dtype)
            )
            base_loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=smoothed_targets,
                logits=logits,
            )
        else:
            base_loss = -log_p_t
        loss = modulating * base_loss

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
        if calibration and not self.calibration_use_vc_target:
            # The VC head still produces and learns the logits. This switch
            # disables only its auxiliary variational regularizers.
            focal_per_sample = self._per_sample_focal_loss(logits, y_flat)
            focal_loss = self._weighted_mean(focal_per_sample, sample_weight)
            zero = tf.zeros((), dtype=focal_loss.dtype)
            return {
                "total_loss": focal_loss,
                "classification_loss": focal_loss,
                "weighted_classification_loss": focal_loss,
                "focal_loss": focal_loss,
                "weighted_focal_loss": focal_loss,
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
            include_classification=True,
        )
        # Focal classification and every enabled VC term were calculated in
        # one joint objective from the exact logits returned by vc_target.
        components = dict(components)
        # Do not expose the VariationalClassifier's legacy cross-entropy
        # aliases: older code used those names for the focal value, which made
        # logs incorrectly show identical cross_entropy and focal_loss fields.
        components.pop("cross_entropy", None)
        components.pop("weighted_cross_entropy", None)
        return components

    def _subject_logits(self, pooled_features, training: bool, use_grl: bool):
        if self.subject_logits_layer is None:
            raise RuntimeError("Subject head has not been configured.")
        x = (
            self.subject_gradient_reversal(pooled_features)
            if use_grl
            else pooled_features
        )
        x = self.subject_hidden(x)
        x = self.subject_dropout_layer(x, training=training)
        return self.subject_logits_layer(x)

    def _subject_components(
        self,
        pooled_features,
        subject_ids,
        training: bool,
        use_grl: bool,
    ):
        if not self.subject_adversarial_enabled or subject_ids is None:
            zero = tf.zeros((), dtype=pooled_features.dtype)
            return {
                "subject_loss": zero,
                "subject_logits": None,
                "subject_targets": None,
            }
        targets = tf.cast(tf.reshape(subject_ids, [-1]), tf.int32)
        logits = self._subject_logits(
            pooled_features,
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
        subject_components=None,
        vrex_components=None,
    ):
        zero = tf.zeros((), dtype=total_loss.dtype)
        self.loss_tracker.update_state(total_loss)
        self.vc_loss_tracker.update_state(vc_components["total_loss"])
        self.focal_loss_tracker.update_state(vc_components["focal_loss"])
        # Reporting metrics are deliberately unweighted. Class/sample weights
        # may shape the optimization loss, but "window_accuracy" and
        # "window_balanced_accuracy" should describe the model's natural
        # predictions on the observed windows.
        self.accuracy_tracker.update_state(
            y_flat,
            outputs["logits"],
        )
        self.balanced_accuracy_tracker.update_state(
            y_flat,
            outputs["logits"],
        )
        predicted_ids = tf.argmax(
            outputs["logits"],
            axis=-1,
            output_type=tf.int32,
        )
        self.predicted_class_1_fraction_tracker.update_state(
            tf.cast(tf.equal(predicted_ids, 1), tf.float32)
        )
        self.true_class_1_fraction_tracker.update_state(
            tf.cast(tf.equal(tf.cast(y_flat, tf.int32), 1), tf.float32)
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
            )
            vc_components = self._vc_components(
                outputs["classification_embedding"],
                outputs["logits"],
                y_flat,
                sample_weight,
                calibration=False,
            )
            subject_components = self._subject_components(
                outputs["pooled_features"],
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
                tf.cast(self.vc_loss_weight, dtype) * vc_components["total_loss"]
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
            subject_components=subject_components,
            vrex_components=vrex_components,
        )

    def _mldg_train_step(self, x, y_flat, sample_weight):
        run_sic_mldg_train_step(self, x, y_flat, sample_weight)

    def _calibration_train_step(self, x, y_flat, sample_weight):
        eeg_inputs, _ = self._split_eeg_and_subject_inputs(x)
        # Backbone layers are frozen so calibration fits the selected dense
        # suffix and the sole VC logits-producing classification head.
        with tf.GradientTape() as tape:
            outputs = self._encode(
                eeg_inputs,
                training=False,
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
            total = tf.cast(self.vc_loss_weight, dtype) * vc_components[
                "total_loss"
            ] + self._regularization_loss(dtype)
        variables = self.trainable_variables
        gradients = tape.gradient(total, variables)
        self._apply_gradients(self.main_optimizer, gradients, variables)
        self._update_metrics(
            total_loss=total,
            vc_components=vc_components,
            outputs=outputs,
            y_flat=y_flat,
            sample_weight=sample_weight,
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
        elif self.use_mldg:
            self._mldg_train_step(x, y_flat, sample_weight)
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
        )
        vc_components = self._vc_components(
            outputs["classification_embedding"],
            outputs["logits"],
            y_flat,
            sample_weight,
            calibration=self.calibration_mode,
        )
        if self.calibration_mode:
            subject_components = None
            total = (
                tf.cast(self.vc_loss_weight, vc_components["total_loss"].dtype)
                * vc_components["total_loss"]
            )
        else:
            subject_components = self._subject_components(
                outputs["pooled_features"],
                subject_ids,
                training=False,
                use_grl=False,
            )
            dtype = vc_components["total_loss"].dtype
            total = (
                tf.cast(self.vc_loss_weight, dtype) * vc_components["total_loss"]
                + tf.cast(self.subject_loss_weight, dtype)
                * tf.cast(subject_components["subject_loss"], dtype)
            )
        self._update_metrics(
            total_loss=total,
            vc_components=vc_components,
            outputs=outputs,
            y_flat=y_flat,
            sample_weight=sample_weight,
            subject_components=subject_components,
            vrex_components=None,
        )
        return {metric.name: metric.result() for metric in self.metrics}

    def predict_step(self, data):
        x = data[0] if isinstance(data, tuple) else data
        return self(x, training=False)

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
        """Freeze the backbone and adapt the final k classification layers.

        ``unfreeze_layers`` counts prediction layers backward from the output:
        1 means the VC logits head only, 2 means the final hidden dense block
        plus the VC head, and so on. The VC parameters always remain trainable
        because this module is the classifier rather than a frozen target.
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
        if self.graph_encoder is not None:
            self.graph_encoder.trainable = False
        for layer in self.temporal_bilstms:
            layer.trainable = False
        for layer in self.temporal_norms:
            layer.trainable = False
        for layer in self.temporal_dropouts:
            layer.trainable = False
        self.parallel_temporal_pool.trainable = False
        self.feature_pool.trainable = False
        if self.subject_gradient_reversal is not None:
            self.subject_gradient_reversal.trainable = False
        if self.subject_hidden is not None:
            self.subject_hidden.trainable = False
        if self.subject_dropout_layer is not None:
            self.subject_dropout_layer.trainable = False
        if self.subject_logits_layer is not None:
            self.subject_logits_layer.trainable = False

        # Freeze every dense feature block, then unfreeze exactly the requested
        # suffix. A hidden block consists of Dense + LN + Dropout; the final
        # prediction layer is the VC classification head.
        for dense, norm, dropout in zip(
            self.classification_dense_layers,
            self.classification_norm_layers,
            self.classification_dropout_layers,
        ):
            dense.trainable = False
            norm.trainable = False
            dropout.trainable = False
        self.vc_target.trainable = True

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
            "trainable_variables": [
                variable.name for variable in self.trainable_variables
            ],
            "calibration_use_vc_target": self.calibration_use_vc_target,
        }

    def get_encoder_features(self, inputs):
        """Return both deterministic branches and their direct concatenation."""
        eeg_inputs, _ = self._split_eeg_and_subject_inputs(inputs)
        outputs = self._encode(
            eeg_inputs,
            training=False,
        )
        return {
            "gcn_gru_features": outputs["graph_sequence"],
            "bilstm_features": outputs["pooled_bilstm_sequence"],
            "combined_feature_sequence": outputs[
                "combined_feature_sequence"
            ],
            "pooled_features": outputs["pooled_features"],
            "window_features": outputs["window_features"],
            "classification_embedding": outputs["classification_embedding"],
            "probabilities": outputs["probabilities"],
        }

    def get_adjacency_matrices(self):
        if self.graph_encoder is None:
            return {}
        return {
            "mtl_raw_adjacency": self.graph_encoder.get_raw_adjacency_matrix(),
            "mtl_normalized_adjacency": self.graph_encoder.get_adjacency_matrix(),
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "graph_encoder": (
                    None
                    if self.graph_encoder is None
                    else _serialize_keras_object(self.graph_encoder)
                ),
                "classification_level": self.classification_level,
                "n_classes": self.n_classes,
                "gcn_gru_feature_dim": self.gcn_gru_feature_dim,
                "bilstm_units": self.bilstm_units,
                "n_bilstm_layers": self.n_bilstm_layers,
                "bilstm_dropout": self.bilstm_dropout_rate,
                "use_gcn_gru_branch": self.use_gcn_gru_branch,
                "use_bilstm_branch": self.use_bilstm_branch,
                "temporal_downsample_factor": self.temporal_downsample_factor,
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
                "training_method": self.training_method,
                "use_vrex": self.use_vrex,
                "vrex_penalty_weight": self.vrex_penalty_weight,
                "mldg_meta_train_subjects": self.mldg_meta_train_subjects,
                "mldg_meta_test_subjects": self.mldg_meta_test_subjects,
                "mldg_trials_per_subject": self.mldg_trials_per_subject,
                "mldg_steps_per_epoch": self.mldg_steps_per_epoch,
                "mldg_inner_learning_rate": self.mldg_inner_learning_rate,
                "mldg_meta_test_weight": self.mldg_meta_test_weight,
                "mldg_seed": self.mldg_seed,
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
        config["graph_encoder"] = (
            None
            if config["graph_encoder"] is None
            else _deserialize_keras_object(config["graph_encoder"])
        )
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
    use_gcn_gru_branch: bool = True,
    use_bilstm_branch: bool = True,
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
    training_method: str | None = None,
    use_vrex: bool = False,
    vrex_penalty_weight: float = 1.0,
    mldg_meta_train_subjects: int | None = None,
    mldg_meta_test_subjects: int = 4,
    mldg_trials_per_subject: int = 1,
    mldg_steps_per_epoch: int | None = None,
    mldg_inner_learning_rate: float = 1e-4,
    mldg_meta_test_weight: float = 1.0,
    mldg_seed: int | None = 42,
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
    optimizer_name: str = "adamw",
    learning_rate: float = 1e-4,
    vc_discriminator_learning_rate: float | None = None,
    weight_decay: float = 5e-5,
    use_class_weight: bool = False,
    model_name: str = "sic_subject_invariant_calibrator",
) -> SICModel:
    """Build SIC, computing its fixed MI graph from source data when needed.

    The ``training_*`` arguments are accepted explicitly for
    ``subject_calibration_cv``. ``training_features`` estimates the source-only
    graph, ``training_subject_ids`` configures source metadata/adversity, and
    ``training_trial_ids`` keeps complete MLDG trials together. Labels are
    accepted to keep the builder contract uniform and leakage-auditable.
    """
    del training_labels

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

    use_gcn_gru_branch = bool(use_gcn_gru_branch)
    use_bilstm_branch = bool(use_bilstm_branch)
    if not use_gcn_gru_branch and not use_bilstm_branch:
        raise ValueError(
            "At least one of use_gcn_gru_branch/use_bilstm_branch must be true."
        )
    combined_feature_dim = (
        int(spectral_gru_units) * int(use_gcn_gru_branch)
        + 2 * int(bilstm_units) * int(use_bilstm_branch)
    )

    needs_graph = use_gcn_gru_branch
    if adjacency is None and needs_graph:
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
    elif adjacency is not None:
        adjacency = np.asarray(adjacency, dtype=np.float32)

    if n_subject_classes is None and training_subject_ids is not None:
        n_subject_classes = int(len(np.unique(np.asarray(training_subject_ids))))

    graph_encoder = None
    if use_gcn_gru_branch:
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

    model = SICModel(
        graph_encoder=graph_encoder,
        classification_level=classification_level,
        n_classes=int(n_classes),
        gcn_gru_feature_dim=int(spectral_gru_units),
        bilstm_units=int(bilstm_units),
        n_bilstm_layers=int(n_bilstm_layers),
        bilstm_dropout=float(bilstm_dropout),
        use_gcn_gru_branch=use_gcn_gru_branch,
        use_bilstm_branch=use_bilstm_branch,
        temporal_downsample_factor=int(t_down),
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
        training_method=training_method,
        use_vrex=bool(use_vrex),
        vrex_penalty_weight=float(vrex_penalty_weight),
        mldg_meta_train_subjects=mldg_meta_train_subjects,
        mldg_meta_test_subjects=int(mldg_meta_test_subjects),
        mldg_trials_per_subject=int(mldg_trials_per_subject),
        mldg_steps_per_epoch=mldg_steps_per_epoch,
        mldg_inner_learning_rate=float(mldg_inner_learning_rate),
        mldg_meta_test_weight=float(mldg_meta_test_weight),
        mldg_seed=mldg_seed,
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
    model.set_source_training_metadata(
        training_subject_ids,
        training_trial_ids,
    )

    # Build every stateful branch before the first fit. This avoids Keras 3
    # creating new nested-layer variables after the outer model has been marked
    # built (the failure mode that affected earlier subject-adversarial models).
    if bool(use_subject_adversarial) and n_subject_classes is not None:
        _ = model._subject_logits(
            tf.zeros((1, combined_feature_dim), dtype=tf.float32),
            training=False,
            use_grl=False,
        )
    if not bool(use_subject_adversarial) or n_subject_classes is not None:
        _ = model(tf.zeros(dummy_shape, dtype=tf.float32), training=False)
    return model


# Compatibility aliases for earlier v6 naming.
JointV6Model = SICModel
JointSTSModelV6 = SICModel
build_joint_v6_model = build_sic_model
build_joint_sts_model_v6 = build_sic_model
