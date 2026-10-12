"""SIC: Subject Invariant Calibrator model.

Parallel MTLFuseNet GCN-GRU and temporal BiLSTM encoders with head fusion.

Architecture
------------
For each EEG window, two independent deterministic encoders produce feature
sequences::

    channel-band EEG -> fixed-MI GCN -> spectral GRU -> h_g
    channel-band EEG -> temporal BiLSTM              -> h_b

The complete encoder feature vectors are concatenated directly at every
original EEG timestep. There is no learned projection, bottleneck, temporal
downsampling, or fusion network between the encoders and the recurrent
classifier::

    h_joint = concat([h_g, h_b])       # (window, 128, 640 by default)
    ordered windows -> reshape         # (trial, windows * 128, 640)
    full-trial sequence -> BiGRU/GRU final state -> VC logits

With the default widths, h_g has 384 features and h_b has 2 * 128 = 256
features, so both active branches give the final heads all 640 features. Fusion
occurs in the classifier and subject-adversarial heads. EEGProc's existing
VariationalClassifier is the sole logits-producing classification head; there
is no parallel dense-logit head. Focal classification and the VC regularizers
are optimized jointly from that one head and its one logits tensor. The VC is
not an encoder VAE.

There is no encoder posterior, reparameterization, or encoder KL loss. An
optional deterministic reconstruction objective decodes each active encoder
branch independently back to the original EEG window::

    h_g -> GCNMTLDecoder -> x_hat_g
    h_b -> GCNMTLDecoder -> x_hat_b

The two decoders never consume the concatenated feature sequence. When both
branches are active, SIC v15 additionally learns one bounded scalar that fuses
their same-shaped outputs in the original EEG space::

    alpha = sigmoid(joint_reconstruction_logit)
    x_hat_joint = alpha * x_hat_g + (1 - alpha) * x_hat_b

The optimized reconstruction objective is the joint reconstruction MSE plus a
configurable auxiliary weight times the mean independent-branch MSE. The
auxiliary losses keep both representations reconstructive instead of allowing
the joint output to hide a collapsed branch. Single-branch ablations retain the
v11 reconstruction objective exactly. Subject invariance can be imposed
directly on the recurrent trial representation (or per-window representation
in window mode) with ordinary gradient reversal.

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

For trial-level classification, the recurrent classifier consumes every
encoded timestep from every window in its original chronological order. There
is no mean, max, attention, or downsampling operation on either the within-
window or cross-window temporal axes. Window-level classification uses the
same recurrent classifier on the complete encoded window sequence.

Subject calibration
-------------------
``prepare_for_subject_calibration`` freezes the complete representation and
subject adversary. By default it also freezes both reconstruction decoders,
preserving the historical classifier-only calibration behavior. Set
``calibration_freeze_decoder=false`` to continue training every active decoder
with a weighted reconstruction term. The VC classification head remains
trainable and calibration can also unfreeze the full-sequence recurrent
classifier:

    1 -> VC logits head only
    2 -> full-sequence GRU/BiGRU + VC logits head

The calibration train step supports focal or cross-entropy classification loss
and, when enabled, VC regularizers computed from the same VC logits. When the
decoder is unfrozen, its objective is ``calibration_decoder_loss_weight`` times
the normalized joint-plus-auxiliary reconstruction objective. The VC head
therefore adapts in every calibration fold instead of acting as a frozen source
target.
For backward CLI compatibility, ``calibration_use_vc_target=false`` disables
only the auxiliary VC regularizers; it does not replace or freeze the VC
classification head that produces the logits.

This file is designed for ``cross_val.subject_calibration_cv``.  Its builder
accepts ``training_features`` and computes the fixed MI adjacency from source
subjects only, avoiding target-subject leakage.

Ablations are controlled by ``use_gcn_gru_branch``, ``use_bilstm_branch``, and
``use_decoder``. At least one encoder branch must remain enabled.
"""

from __future__ import annotations

from collections.abc import Sequence
import numpy as np
import tensorflow as tf

from eegproc.deep_learning.domain_generalization.meta_learning import run_mldg_train_step as run_sic_mldg_train_step
from eegproc.deep_learning.unsupervised.GNN.GCNMTL import GCNMTLDecoder, GCNMTLEncoder
from eegproc.deep_learning.joint_architectures.sic_common.model import (
    SICModelCore,
    _resolve_training_method,
    _normalize_calibration_loss,
    _build_optimizer,
    _as_positive_tuple,
    _resolve_recurrent_widths,
    _deduplicate_variables,
    _serialize_keras_object,
    _deserialize_keras_object,
    _source_only_mi_adjacency,
    StreamingBalancedAccuracy,
    StreamingR2,
    GradientReversal,
)


SIC_BUILDER_API_VERSION = 15
JOINT_V6_BUILDER_API_VERSION = SIC_BUILDER_API_VERSION


@tf.keras.utils.register_keras_serializable(package="EEGProc")
class ConvexReconstructionFusion(tf.keras.layers.Layer):
    """Learn one convex mixing weight for two same-shaped reconstructions."""

    def __init__(self, initial_alpha: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        initial_alpha = float(initial_alpha)
        if not np.isfinite(initial_alpha) or not 0.0 < initial_alpha < 1.0:
            raise ValueError("initial_alpha must be finite and strictly in (0, 1).")
        self.initial_alpha = initial_alpha
        self.mix_logit = None

    def build(self, input_shape):
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
            raise ValueError(
                "ConvexReconstructionFusion expects exactly two input shapes."
            )
        if not tf.TensorShape(input_shape[0]).is_compatible_with(
            tf.TensorShape(input_shape[1])
        ):
            raise ValueError(
                "GCN-GRU and BiLSTM reconstructions must have compatible shapes."
            )
        initial_logit = np.log(self.initial_alpha) - np.log1p(
            -self.initial_alpha
        )
        self.mix_logit = self.add_weight(
            name="mix_logit",
            shape=(),
            initializer=tf.keras.initializers.Constant(initial_logit),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError(
                "ConvexReconstructionFusion expects [gcn_gru, bilstm]."
            )
        gcn_gru_reconstruction, bilstm_reconstruction = inputs
        tf.debugging.assert_equal(
            tf.shape(gcn_gru_reconstruction),
            tf.shape(bilstm_reconstruction),
            message=(
                "GCN-GRU and BiLSTM reconstruction shapes must match before "
                "joint fusion."
            ),
        )
        alpha = tf.cast(self.alpha, gcn_gru_reconstruction.dtype)
        return (
            alpha * gcn_gru_reconstruction
            + (1.0 - alpha) * bilstm_reconstruction
        )

    @property
    def alpha(self):
        if self.mix_logit is None:
            return tf.convert_to_tensor(self.initial_alpha, dtype=self.compute_dtype)
        return tf.sigmoid(self.mix_logit)

    def get_config(self):
        config = super().get_config()
        config.update({"initial_alpha": self.initial_alpha})
        return config


@tf.keras.utils.register_keras_serializable(package="EEGProc")
class SICModel(SICModelCore):
    """SIC with direct deterministic GCN-GRU/BiLSTM feature concatenation."""

    def __init__(
        self,
        *,
        graph_encoder: GCNMTLEncoder | None,
        gcn_gru_decoder: GCNMTLDecoder | None = None,
        bilstm_decoder: GCNMTLDecoder | None = None,
        classification_level: str = "trial",
        n_classes: int = 2,
        gcn_gru_feature_dim: int = 384,
        bilstm_units: int = 128,
        n_bilstm_layers: int = 1,
        bilstm_dropout: float = 0.30,
        use_gcn_gru_branch: bool = True,
        use_bilstm_branch: bool = True,
        use_decoder: bool = False,
        classifier_rnn_type: str = "bigru",
        classifier_rnn_units: int | Sequence[int] = 128,
        n_classifier_rnn_layers: int | None = None,
        classifier_rnn_dropout: float = 0.20,
        classification_hidden_units: Sequence[int] | None = None,
        classification_dropout: float | None = None,
        activation: str = "relu",
        label_smoothing: float = 0.0,
        focal_gamma: float = 1.0,
        focal_alpha: float | None = None,
        vc_loss_weight: float = 1.0,
        vc_alpha: float = 1.0,
        vc_beta: float = 0.5,
        vc_gamma: float = 0.0,
        vc_lambda: float = 0.0,
        vc_logit_scale: float = 1.0,
        update_vc_discriminator: bool = False,
        reconstruction_loss_weight: float = 0.10,
        joint_reconstruction_auxiliary_weight: float = 0.25,
        joint_reconstruction_initial_alpha: float = 0.5,
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
        calibration_vc_loss_weight: float | None = None,
        calibration_freeze_decoder: bool = True,
        calibration_decoder_loss_weight: float = 1.0,
        calibration_loss: str = "focal",
        calibration_label_smoothing: float | None = None,
        calibration_focal_gamma: float | None = None,
        calibration_focal_alpha: float | None = None,
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
        use_decoder = bool(use_decoder)
        if not use_gcn_gru_branch and not use_bilstm_branch:
            raise ValueError(
                "At least one of use_gcn_gru_branch/use_bilstm_branch must be true."
            )
        if use_gcn_gru_branch and graph_encoder is None:
            raise ValueError("graph_encoder is required when use_gcn_gru_branch=true.")
        if use_decoder and use_gcn_gru_branch and gcn_gru_decoder is None:
            raise ValueError(
                "gcn_gru_decoder is required when the GCN-GRU branch decoder is enabled."
            )
        if use_decoder and use_bilstm_branch and bilstm_decoder is None:
            raise ValueError(
                "bilstm_decoder is required when the BiLSTM branch decoder is enabled."
            )
        if int(n_classes) < 2:
            raise ValueError("n_classes must be >= 2.")
        if int(gcn_gru_feature_dim) < 1:
            raise ValueError("gcn_gru_feature_dim must be positive.")
        if int(bilstm_units) < 1 or int(n_bilstm_layers) < 1:
            raise ValueError("BiLSTM dimensions must be positive.")
        if float(focal_gamma) < 0.0:
            raise ValueError("focal_gamma must be non-negative.")
        if focal_alpha is not None and not 0.0 <= float(focal_alpha) <= 1.0:
            raise ValueError("focal_alpha must be in [0, 1] or None.")
        calibration_loss = _normalize_calibration_loss(calibration_loss)
        resolved_calibration_label_smoothing = (
            float(label_smoothing)
            if calibration_label_smoothing is None
            else float(calibration_label_smoothing)
        )
        resolved_calibration_focal_gamma = (
            float(focal_gamma)
            if calibration_focal_gamma is None
            else float(calibration_focal_gamma)
        )
        resolved_calibration_focal_alpha = (
            focal_alpha
            if calibration_focal_alpha is None
            else float(calibration_focal_alpha)
        )
        resolved_calibration_vc_loss_weight = (
            float(vc_loss_weight)
            if calibration_vc_loss_weight is None
            else float(calibration_vc_loss_weight)
        )
        if not 0.0 <= resolved_calibration_label_smoothing < 1.0:
            raise ValueError("calibration_label_smoothing must be in [0, 1).")
        if resolved_calibration_focal_gamma < 0.0:
            raise ValueError("calibration_focal_gamma must be non-negative.")
        if (
            resolved_calibration_focal_alpha is not None
            and not 0.0 <= float(resolved_calibration_focal_alpha) <= 1.0
        ):
            raise ValueError("calibration_focal_alpha must be in [0, 1] or None.")
        if resolved_calibration_vc_loss_weight < 0.0:
            raise ValueError("calibration_vc_loss_weight must be non-negative.")
        if (
            not np.isfinite(float(calibration_decoder_loss_weight))
            or float(calibration_decoder_loss_weight) < 0.0
        ):
            raise ValueError(
                "calibration_decoder_loss_weight must be finite and non-negative."
            )
        requested_classifier_widths = _resolve_recurrent_widths(
            "classifier_rnn_units",
            classifier_rnn_units,
            n_classifier_rnn_layers,
        )

        # Map configurations serialized before API v11 onto the recurrent
        # classifier. Unequal legacy widths now map to tapered RNN layers.
        if classification_hidden_units is not None:
            legacy_units = tuple(
                int(value) for value in classification_hidden_units
            )
            if not legacy_units or any(value < 1 for value in legacy_units):
                raise ValueError(
                    "classification_hidden_units must contain positive values."
                )
            if requested_classifier_widths not in {
                (128,),
                legacy_units,
            }:
                raise ValueError(
                    "classifier_rnn_units conflicts with the legacy "
                    "classification_hidden_units setting."
                )
            requested_classifier_widths = legacy_units
        if classification_dropout is not None:
            if float(classifier_rnn_dropout) != 0.20 and (
                float(classifier_rnn_dropout) != float(classification_dropout)
            ):
                raise ValueError(
                    "classifier_rnn_dropout conflicts with the legacy "
                    "classification_dropout setting."
                )
            classifier_rnn_dropout = float(classification_dropout)
        # The old classifier Dense stack also used this activation. After its
        # removal, the setting remains relevant only to the subject-adversary
        # hidden layer and to loading older serialized configurations.

        classifier_rnn_type = (
            str(classifier_rnn_type).strip().lower().replace("-", "_")
        )
        classifier_rnn_type = {
            "bi_gru": "bigru",
            "bidirectional_gru": "bigru",
        }.get(classifier_rnn_type, classifier_rnn_type)
        if classifier_rnn_type not in {"gru", "bigru"}:
            raise ValueError("classifier_rnn_type must be 'gru' or 'bigru'.")
        if not 0.0 <= float(classifier_rnn_dropout) < 1.0:
            raise ValueError("classifier_rnn_dropout must be in [0, 1).")
        if not 0.0 <= float(bilstm_dropout) < 1.0:
            raise ValueError("bilstm_dropout must be in [0, 1).")
        if not 0.0 <= float(label_smoothing) < 1.0:
            raise ValueError("label_smoothing must be in [0, 1).")
        if not np.isfinite(float(vc_logit_scale)) or float(vc_logit_scale) <= 0.0:
            raise ValueError("vc_logit_scale must be finite and positive.")
        for loss_name, value in (
            ("vc_loss_weight", vc_loss_weight),
            ("reconstruction_loss_weight", reconstruction_loss_weight),
            (
                "joint_reconstruction_auxiliary_weight",
                joint_reconstruction_auxiliary_weight,
            ),
            ("subject_adversarial_weight", subject_adversarial_weight),
            ("subject_loss_weight", subject_loss_weight),
        ):
            if not np.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{loss_name} must be finite and non-negative.")
        if (
            not np.isfinite(float(joint_reconstruction_initial_alpha))
            or not 0.0 < float(joint_reconstruction_initial_alpha) < 1.0
        ):
            raise ValueError(
                "joint_reconstruction_initial_alpha must be finite and in (0, 1)."
            )
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
        max_calibration_layers = 2
        if not 1 <= calibration_unfreeze_layers <= max_calibration_layers:
            raise ValueError(
                "calibration_unfreeze_layers must be between 1 and "
                f"{max_calibration_layers}; got {calibration_unfreeze_layers}."
            )

        self.graph_encoder = graph_encoder
        self.gcn_gru_decoder = gcn_gru_decoder
        self.bilstm_decoder = bilstm_decoder
        self.classification_level = classification_level
        self.n_classes = int(n_classes)
        self.gcn_gru_feature_dim = int(gcn_gru_feature_dim)
        self.bilstm_units = int(bilstm_units)
        self.bilstm_feature_dim = 2 * self.bilstm_units
        self.n_bilstm_layers = int(n_bilstm_layers)
        self.bilstm_dropout_rate = float(bilstm_dropout)
        self.use_gcn_gru_branch = use_gcn_gru_branch
        self.use_bilstm_branch = use_bilstm_branch
        self.use_decoder = use_decoder
        self.combined_feature_dim = (
            self.gcn_gru_feature_dim * int(self.use_gcn_gru_branch)
            + self.bilstm_feature_dim * int(self.use_bilstm_branch)
        )
        self.classifier_rnn_type = classifier_rnn_type
        self.classifier_rnn_units = requested_classifier_widths
        self.n_classifier_rnn_layers = len(self.classifier_rnn_units)
        self.classifier_rnn_dropout_rate = float(classifier_rnn_dropout)
        self.activation_name = str(activation)
        self.label_smoothing = float(label_smoothing)
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = None if focal_alpha is None else float(focal_alpha)

        self.vc_loss_weight = float(vc_loss_weight)
        self.vc_alpha = float(vc_alpha)
        self.vc_beta = float(vc_beta)
        self.vc_gamma = float(vc_gamma)
        self.vc_lambda = float(vc_lambda)
        self.vc_logit_scale = float(vc_logit_scale)
        self.update_vc_discriminator = bool(update_vc_discriminator)
        self.reconstruction_loss_weight = float(reconstruction_loss_weight)
        self.joint_reconstruction_auxiliary_weight = float(
            joint_reconstruction_auxiliary_weight
        )
        self.joint_reconstruction_initial_alpha = float(
            joint_reconstruction_initial_alpha
        )
        self.use_joint_reconstruction = bool(
            self.use_decoder
            and self.use_gcn_gru_branch
            and self.use_bilstm_branch
        )
        self.joint_reconstruction_fusion = (
            ConvexReconstructionFusion(
                initial_alpha=self.joint_reconstruction_initial_alpha,
                name="sic_joint_reconstruction_fusion",
            )
            if self.use_joint_reconstruction
            else None
        )
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
        self.calibration_vc_loss_weight = resolved_calibration_vc_loss_weight
        self.calibration_freeze_decoder = bool(calibration_freeze_decoder)
        self.calibration_decoder_loss_weight = float(
            calibration_decoder_loss_weight
        )
        self.calibration_loss = calibration_loss
        self.calibration_label_smoothing = resolved_calibration_label_smoothing
        self.calibration_focal_gamma = resolved_calibration_focal_gamma
        self.calibration_focal_alpha = resolved_calibration_focal_alpha
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

        self._build_sequence_heads(vc_logit_scale=self.vc_logit_scale)

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.vc_loss_tracker = tf.keras.metrics.Mean(name="vc_loss")
        self.classification_loss_tracker = tf.keras.metrics.Mean(
            name="classification_loss"
        )
        self.focal_loss_tracker = tf.keras.metrics.Mean(name="focal_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.decoder_r2_tracker = StreamingR2(name="decoder_r2")
        self.gcn_gru_reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="gcn_gru_reconstruction_loss"
        )
        self.gcn_gru_decoder_r2_tracker = StreamingR2(
            name="gcn_gru_decoder_r2"
        )
        self.bilstm_reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="bilstm_reconstruction_loss"
        )
        self.bilstm_decoder_r2_tracker = StreamingR2(
            name="bilstm_decoder_r2"
        )
        self.joint_reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="joint_reconstruction_loss"
        )
        self.joint_decoder_r2_tracker = StreamingR2(
            name="joint_decoder_r2"
        )
        self.joint_reconstruction_alpha_tracker = tf.keras.metrics.Mean(
            name="joint_reconstruction_alpha"
        )
        self.joint_reconstruction_gain_tracker = tf.keras.metrics.Mean(
            name="joint_reconstruction_gain_vs_best_branch"
        )
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

    def _extra_decoder_metrics(self):
        if not self.use_joint_reconstruction:
            return []
        return [
            self.joint_reconstruction_loss_tracker,
            self.joint_decoder_r2_tracker,
            self.joint_reconstruction_alpha_tracker,
            self.joint_reconstruction_gain_tracker,
        ]

    def _source_callbacks(self, callbacks):
        if len(getattr(self, "_fold_devices", ())) > 1:
            from .multi_gpu import FoldGPUMemory

            return list(callbacks or ()) + [FoldGPUMemory(self._fold_devices)]
        return callbacks

    def _reconstruction_components(self, outputs, training: bool):
        """Blend independent reconstructions and their auxiliary branch loss."""
        if "device_shards" in outputs:
            from .multi_gpu import reconstruction_on_devices

            return reconstruction_on_devices(self, outputs, training=training)
        components = super()._reconstruction_components(outputs, training)
        dtype = outputs["combined_feature_sequence"].dtype
        zero = tf.zeros((), dtype=dtype)
        branch_reconstruction_loss = components["reconstruction_loss"]
        components.update(
            branch_reconstruction_loss=branch_reconstruction_loss,
            joint_reconstruction_loss=zero,
            joint_reconstruction_alpha=zero,
            joint_reconstruction_gain_vs_best_branch=zero,
            joint_reconstruction=None,
        )
        if not self.use_decoder:
            return components
        target = tf.cast(outputs["flat_windows"], dtype)
        if self.use_joint_reconstruction:
            gcn_gru_reconstruction = components["gcn_gru_reconstruction"]
            bilstm_reconstruction = components["bilstm_reconstruction"]
            if gcn_gru_reconstruction is None or bilstm_reconstruction is None:
                raise RuntimeError(
                    "Joint reconstruction requires both active decoder outputs."
                )
            joint_reconstruction = self.joint_reconstruction_fusion(
                [gcn_gru_reconstruction, bilstm_reconstruction]
            )
            tf.debugging.assert_equal(
                tf.shape(joint_reconstruction),
                tf.shape(target),
                message=(
                    "Joint decoder output must match the original flattened "
                    "EEG windows."
                ),
            )
            joint_reconstruction_loss = tf.reduce_mean(
                tf.square(target - joint_reconstruction)
            )
            best_branch_loss = tf.minimum(
                components["gcn_gru_reconstruction_loss"],
                components["bilstm_reconstruction_loss"],
            )
            auxiliary_weight = tf.cast(
                self.joint_reconstruction_auxiliary_weight,
                dtype,
            )
            components["joint_reconstruction"] = joint_reconstruction
            components["joint_reconstruction_loss"] = joint_reconstruction_loss
            components["joint_reconstruction_alpha"] = tf.cast(
                self.joint_reconstruction_fusion.alpha,
                dtype,
            )
            components["joint_reconstruction_gain_vs_best_branch"] = (
                best_branch_loss - joint_reconstruction_loss
            )
            components["reconstruction_loss"] = tf.math.divide_no_nan(
                joint_reconstruction_loss
                + auxiliary_weight * branch_reconstruction_loss,
                1.0 + auxiliary_weight,
            )
        else:
            components["reconstruction_loss"] = branch_reconstruction_loss
        return components

    def _update_extra_decoder_metrics(self, outputs, reconstruction_components, zero):
        if self.use_joint_reconstruction:
            joint_loss = (
                zero
                if reconstruction_components is None
                else reconstruction_components["joint_reconstruction_loss"]
            )
            joint_alpha = (
                zero
                if reconstruction_components is None
                else reconstruction_components["joint_reconstruction_alpha"]
            )
            joint_gain = (
                zero
                if reconstruction_components is None
                else reconstruction_components[
                    "joint_reconstruction_gain_vs_best_branch"
                ]
            )
            self.joint_reconstruction_loss_tracker.update_state(joint_loss)
            self.joint_reconstruction_alpha_tracker.update_state(joint_alpha)
            self.joint_reconstruction_gain_tracker.update_state(joint_gain)
            joint_reconstruction = (
                None
                if reconstruction_components is None
                else reconstruction_components["joint_reconstruction"]
            )
            if joint_reconstruction is not None:
                self.joint_decoder_r2_tracker.update_state(
                    outputs["flat_windows"],
                    joint_reconstruction,
                )

    def configure_fold_devices(
        self,
        devices,
        *,
        deterministic_execution: bool = False,
    ):
        """Place MLDG trial forwards on a fold's isolated GPU group.

        Device placement is runtime-only: checkpoints remain ordinary v15
        models, and calibration/inference retain their existing execution.

        TensorFlow's graph executor may run independent device shards in a
        different order from one process to the next.  That is desirable for
        throughput, but even tiny floating-point differences are amplified by
        recurrent MLDG training.  Deterministic execution therefore runs the
        same multi-device code eagerly: the Python device loop becomes the
        fixed execution order while activations remain split across GPUs.
        """
        from .multi_gpu import validate_fold_devices

        self._fold_devices = validate_fold_devices(self, devices)
        self._deterministic_fold_execution = bool(deterministic_execution)
        if self._deterministic_fold_execution and len(self._fold_devices) > 1:
            # Recompile before the first fit so Keras does not wrap train_step
            # in a tf.function that can schedule independent GPU branches in
            # completion order.  Reuse the freshly-created optimizers; no
            # optimizer state exists at this point.
            self.compile(
                main_optimizer=self.main_optimizer,
                vc_discriminator_optimizer=self.vc_discriminator_optimizer,
                run_eagerly=True,
                jit_compile=False,
            )
        self.train_function = None

    def _encode_mldg(self, eeg_inputs, *, training):
        devices = getattr(self, "_fold_devices", ())
        if len(devices) < 2:
            return self._encode(eeg_inputs, training=training)
        from .multi_gpu import encode_on_devices

        return encode_on_devices(self, eeg_inputs, devices, training=training)

    def _mldg_train_step(self, x, y_flat, sample_weight):
        run_sic_mldg_train_step(
            self, x, y_flat, sample_weight, encode=self._encode_mldg
        )

    def _set_extra_decoder_trainable(self, trainable):
        if self.joint_reconstruction_fusion is not None:
            self.joint_reconstruction_fusion.trainable = trainable

    def reconstruct_branches(self, inputs):
        """Return independent and, when available, joint EEG reconstructions.

        Values preserve the input rank: window models return ``(N, T, F)`` and
        trial models return ``(N, W, T, F)``. The ``joint`` entry is present
        only when both decoder branches are active.
        """
        if not self.use_decoder:
            raise RuntimeError("Decoder is disabled for this SIC configuration.")
        eeg_inputs, _ = self._split_eeg_and_subject_inputs(inputs)
        eeg_inputs = tf.convert_to_tensor(eeg_inputs, dtype=tf.float32)
        if self.classification_level == "window":
            flat_windows = eeg_inputs
        else:
            flat_windows, _, _ = self._flatten_trial_windows(eeg_inputs)
        outputs = self._features_from_flat_windows(flat_windows, training=False)
        outputs["flat_windows"] = flat_windows
        components = self._reconstruction_components(outputs, training=False)

        reconstructions = {}
        for branch, key in (
            ("gcn_gru", "gcn_gru_reconstruction"),
            ("bilstm", "bilstm_reconstruction"),
            ("joint", "joint_reconstruction"),
        ):
            reconstruction = components[key]
            if reconstruction is None:
                continue
            if self.classification_level == "trial":
                reconstruction = tf.reshape(reconstruction, tf.shape(eeg_inputs))
            reconstructions[branch] = reconstruction
        return reconstructions

    def reconstruct(self, inputs, branch: str | None = None):
        """Reconstruct EEG from one path or return every reconstruction."""
        reconstructions = self.reconstruct_branches(inputs)
        if branch is None:
            return reconstructions
        normalized = str(branch).strip().lower().replace("-", "_")
        normalized = {
            "gcn": "gcn_gru",
            "graph": "gcn_gru",
            "temporal": "bilstm",
            "fused": "joint",
            "fusion": "joint",
        }.get(normalized, normalized)
        if normalized not in reconstructions:
            raise ValueError(
                f"Reconstruction branch {branch!r} is unavailable; active branches "
                f"are {sorted(reconstructions)}."
            )
        return reconstructions[normalized]

    def reconstruct_joint(self, inputs):
        """Return the learned convex joint reconstruction."""
        if not self.use_joint_reconstruction:
            raise RuntimeError(
                "Joint reconstruction requires the decoder and both encoder branches."
            )
        return self.reconstruct(inputs, branch="joint")

    @classmethod
    def from_config(cls, config):
        model = super().from_config(config)
        if model.joint_reconstruction_fusion is not None:
            # Deserialization calls the classifier, which never invokes the
            # reconstruction path. Materialize the scalar before Keras restores
            # weights or calibration snapshots them, including learned alpha.
            decoder = model.gcn_gru_decoder
            shape = (None, decoder.timesteps, decoder.n_channels * decoder.n_bands)
            model.joint_reconstruction_fusion.build([shape, shape])
        return model

    def get_config(self):
        config = super().get_config()
        config.update(
            vc_logit_scale=self.vc_logit_scale,
            joint_reconstruction_auxiliary_weight=self.joint_reconstruction_auxiliary_weight,
            joint_reconstruction_initial_alpha=self.joint_reconstruction_initial_alpha,
        )
        return config


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
    use_decoder: bool = True,
    classifier_rnn_type: str = "bigru",
    classifier_rnn_units: int | Sequence[int] = 128,
    n_classifier_rnn_layers: int | None = None,
    classifier_rnn_dropout: float = 0.20,
    classification_hidden_units: Sequence[int] | None = None,
    classification_dropout: float | None = None,
    activation: str = "relu",
    label_smoothing: float = 0.0,
    focal_gamma: float = 1.0,
    focal_alpha: float | None = None,
    vc_loss_weight: float = 1.0,
    vc_alpha: float = 1.0,
    vc_beta: float = 0.5,
    vc_gamma: float = 0.0,
    vc_lambda: float = 0.0,
    vc_logit_scale: float = 1.0,
    update_vc_discriminator: bool = False,
    reconstruction_loss_weight: float = 0.10,
    joint_reconstruction_auxiliary_weight: float = 0.25,
    joint_reconstruction_initial_alpha: float = 0.5,
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
    calibration_vc_loss_weight: float | None = None,
    calibration_freeze_decoder: bool = True,
    calibration_decoder_loss_weight: float = 1.0,
    calibration_loss: str = "focal",
    calibration_label_smoothing: float | None = None,
    calibration_focal_gamma: float | None = None,
    calibration_focal_alpha: float | None = None,
    decoder_dropout: float = 0.10,
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
    gcn_units = _as_positive_tuple("gcn_units", gcn_units)

    use_gcn_gru_branch = bool(use_gcn_gru_branch)
    use_bilstm_branch = bool(use_bilstm_branch)
    use_decoder = bool(use_decoder)
    if not use_gcn_gru_branch and not use_bilstm_branch:
        raise ValueError(
            "At least one of use_gcn_gru_branch/use_bilstm_branch must be true."
        )
    if not 0.0 <= float(decoder_dropout) < 1.0:
        raise ValueError("decoder_dropout must be in [0, 1).")
    needs_graph = use_gcn_gru_branch or use_decoder
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
            # GCNMTLEncoder retains these historical constructor arguments.
            # A fixed factor of one and no pools disable all downsampling;
            # neither value is exposed as a SIC hyperparameter.
            t_down=1,
            adjacency=adjacency,
            n_channels=int(n_channels),
            n_bands=int(n_bands),
            gcn_units=gcn_units,
            temporal_pool_sizes=(),
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

    def build_branch_decoder(*, emb_dim: int, name: str) -> GCNMTLDecoder:
        # Each call creates a fully independent decoder and therefore an
        # independent reconstruction gradient for one encoder branch.
        return GCNMTLDecoder(
            timesteps=int(timesteps),
            n_channels=int(n_channels),
            n_bands=int(n_bands),
            t_down=1,
            gcn_units=gcn_units,
            temporal_pool_sizes=(),
            adjacency=adjacency,
            emb_dim=int(emb_dim),
            dropout=float(decoder_dropout),
            activation=str(gcn_activation),
            use_batch_norm=bool(gcn_use_batch_norm),
            graph_add_self_loops=bool(graph_add_self_loops),
            graph_symmetrize=bool(graph_symmetrize),
            graph_epsilon=float(graph_epsilon),
            name=name,
        )

    gcn_gru_decoder = (
        build_branch_decoder(
            emb_dim=int(spectral_gru_units),
            name="sic_gcn_gru_decoder",
        )
        if use_decoder and use_gcn_gru_branch
        else None
    )
    bilstm_decoder = (
        build_branch_decoder(
            emb_dim=2 * int(bilstm_units),
            name="sic_bilstm_decoder",
        )
        if use_decoder and use_bilstm_branch
        else None
    )

    model = SICModel(
        graph_encoder=graph_encoder,
        gcn_gru_decoder=gcn_gru_decoder,
        bilstm_decoder=bilstm_decoder,
        classification_level=classification_level,
        n_classes=int(n_classes),
        gcn_gru_feature_dim=int(spectral_gru_units),
        bilstm_units=int(bilstm_units),
        n_bilstm_layers=int(n_bilstm_layers),
        bilstm_dropout=float(bilstm_dropout),
        use_gcn_gru_branch=use_gcn_gru_branch,
        use_bilstm_branch=use_bilstm_branch,
        use_decoder=use_decoder,
        classifier_rnn_type=str(classifier_rnn_type),
        classifier_rnn_units=(
            tuple(int(value) for value in classifier_rnn_units)
            if isinstance(classifier_rnn_units, Sequence)
            and not isinstance(classifier_rnn_units, (str, bytes))
            else int(classifier_rnn_units)
        ),
        n_classifier_rnn_layers=(
            None
            if n_classifier_rnn_layers is None
            else int(n_classifier_rnn_layers)
        ),
        classifier_rnn_dropout=float(classifier_rnn_dropout),
        classification_hidden_units=(
            None
            if classification_hidden_units is None
            else tuple(int(value) for value in classification_hidden_units)
        ),
        classification_dropout=(
            None
            if classification_dropout is None
            else float(classification_dropout)
        ),
        activation=activation,
        label_smoothing=float(label_smoothing),
        focal_gamma=float(focal_gamma),
        focal_alpha=focal_alpha,
        vc_loss_weight=float(vc_loss_weight),
        vc_alpha=float(vc_alpha),
        vc_beta=float(vc_beta),
        vc_gamma=float(vc_gamma),
        vc_lambda=float(vc_lambda),
        vc_logit_scale=float(vc_logit_scale),
        update_vc_discriminator=bool(update_vc_discriminator),
        reconstruction_loss_weight=float(reconstruction_loss_weight),
        joint_reconstruction_auxiliary_weight=float(
            joint_reconstruction_auxiliary_weight
        ),
        joint_reconstruction_initial_alpha=float(
            joint_reconstruction_initial_alpha
        ),
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
        calibration_vc_loss_weight=calibration_vc_loss_weight,
        calibration_freeze_decoder=bool(calibration_freeze_decoder),
        calibration_decoder_loss_weight=float(
            calibration_decoder_loss_weight
        ),
        calibration_loss=calibration_loss,
        calibration_label_smoothing=calibration_label_smoothing,
        calibration_focal_gamma=calibration_focal_gamma,
        calibration_focal_alpha=calibration_focal_alpha,
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
            tf.zeros(
                (1, model.classification_embedding_dim),
                dtype=tf.float32,
            ),
            training=False,
            use_grl=False,
        )
    gcn_gru_dummy_reconstruction = None
    bilstm_dummy_reconstruction = None
    if gcn_gru_decoder is not None:
        gcn_gru_dummy_reconstruction = gcn_gru_decoder(
            tf.zeros(
                (1, int(timesteps), int(spectral_gru_units)),
                dtype=tf.float32,
            ),
            training=False,
        )
    if bilstm_decoder is not None:
        bilstm_dummy_reconstruction = bilstm_decoder(
            tf.zeros(
                (1, int(timesteps), 2 * int(bilstm_units)),
                dtype=tf.float32,
            ),
            training=False,
        )
    if model.joint_reconstruction_fusion is not None:
        _ = model.joint_reconstruction_fusion(
            [gcn_gru_dummy_reconstruction, bilstm_dummy_reconstruction]
        )
    if not bool(use_subject_adversarial) or n_subject_classes is not None:
        _ = model(tf.zeros(dummy_shape, dtype=tf.float32), training=False)
    return model


# Compatibility aliases for earlier v6 naming.
JointV6Model = SICModel
JointSTSModelV6 = SICModel
build_joint_v6_model = build_sic_model
build_joint_sts_model_v6 = build_sic_model
