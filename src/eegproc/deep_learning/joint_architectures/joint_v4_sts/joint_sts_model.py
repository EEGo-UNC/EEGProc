"""Joint v4 STS: band-separated GCN -> BiLSTM -> classifier.

The main emotion path is always sequential:

    EEG window
      -> independent per-band GCN stacks
      -> band-fused graph sequence
      -> BiLSTM
      -> bilstm_emb_dim projection
      -> pooled shared representation
      -> emotion classifier

Two optional auxiliary objectives can be enabled independently:
  * Variational autoencoder reconstruction from the graph sequence.
  * Subject-adversarial classification from the shared BiLSTM representation.

The model supports two input levels:
  * ``window``: one EEG window is one supervised sample.
  * ``trial``: each supervised sample contains an ordered sequence of windows.
    Every window is graph-encoded independently, then the BiLSTM runs across
    the sequence of window embeddings and produces one trial prediction.

MLDG is a run-level training mode. It applies first-order MLDG to the emotion
classification pathway. Optional VAE reconstruction remains a separate
auxiliary update and is not included in the meta-test emotion objective.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import tensorflow as tf

try:
    from ...unsupervised.GNN.GCN_band_separated import (
        BandSeparatedGCNEncoder,
        GCNDecoder,
    )
except ImportError:
    from eegproc.deep_learning.unsupervised.GNN.GCN_band_separated import (
        BandSeparatedGCNEncoder,
        GCNDecoder,
    )


# The SLURM launchers intentionally verify this value before beginning a run.
JOINT_STS_BUILDER_API_VERSION = 6


@tf.keras.utils.register_keras_serializable(package="EEGProc")
class SparseCategoricalFocalLoss(tf.keras.losses.Loss):
    """Sparse categorical focal loss with logits support."""

    def __init__(
        self,
        gamma: float = 0.0,
        alpha: float | Sequence[float] | None = None,
        from_logits: bool = True,
        name: str = "sparse_categorical_focal_loss",
        **kwargs,
    ):
        super().__init__(
            name=name,
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
            **kwargs,
        )
        if float(gamma) < 0.0:
            raise ValueError("gamma must be non-negative.")
        self.gamma = float(gamma)
        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (list, tuple)):
            self.alpha = tuple(float(value) for value in alpha)
        else:
            self.alpha = (float(alpha),)
        if self.alpha is not None:
            if any(value < 0.0 for value in self.alpha):
                raise ValueError("focal alpha weights must be non-negative.")
            if not any(value > 0.0 for value in self.alpha):
                raise ValueError("At least one focal alpha weight must be positive.")
        self.from_logits = bool(from_logits)

    def call(self, y_true, y_pred):
        labels = tf.cast(tf.reshape(y_true, [-1]), tf.int32)

        if self.from_logits:
            probabilities = tf.nn.softmax(y_pred, axis=-1)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels,
                logits=y_pred,
            )
        else:
            probabilities = tf.convert_to_tensor(y_pred)
            probabilities = tf.math.divide_no_nan(
                probabilities,
                tf.reduce_sum(probabilities, axis=-1, keepdims=True),
            )
            cross_entropy = tf.keras.losses.sparse_categorical_crossentropy(
                labels,
                probabilities,
                from_logits=False,
            )

        row_ids = tf.range(tf.shape(labels)[0], dtype=tf.int32)
        true_probability = tf.gather_nd(
            probabilities,
            tf.stack([row_ids, labels], axis=1),
        )
        focal_factor = tf.pow(
            tf.maximum(
                1.0 - true_probability,
                tf.cast(tf.keras.backend.epsilon(), true_probability.dtype),
            ),
            tf.cast(self.gamma, true_probability.dtype),
        )
        loss = tf.cast(cross_entropy, true_probability.dtype) * focal_factor

        if self.alpha is not None:
            alpha = tf.constant(self.alpha, dtype=loss.dtype)
            if len(self.alpha) == 1:
                loss *= alpha[0]
            else:
                tf.debugging.assert_less(
                    labels,
                    tf.shape(alpha)[0],
                    message="focal_alpha must contain one weight per class.",
                )
                loss *= tf.gather(alpha, labels)

        return loss

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "gamma": self.gamma,
                "alpha": self.alpha,
                "from_logits": self.from_logits,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="EEGProc")
class GradientReversal(tf.keras.layers.Layer):
    """Identity forward pass with a negative scaled backward gradient."""

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


def _serialize_keras_object(value):
    if value is None:
        return None
    return tf.keras.utils.serialize_keras_object(value)


def _deserialize_keras_object(value):
    if value is None:
        return None
    return tf.keras.utils.deserialize_keras_object(value)


def _deduplicate_variables(variables):
    seen = set()
    output = []
    for variable in variables:
        identifier = id(variable)
        if identifier not in seen:
            seen.add(identifier)
            output.append(variable)
    return output


def _build_optimizer(
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
):
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
class JointSTSModel(tf.keras.Model):
    """Band-separated GCN -> BiLSTM classifier with optional auxiliary heads."""

    def __init__(
        self,
        graph_encoder: tf.keras.Model,
        decoder: tf.keras.Model | None = None,
        classification_level: str = "trial",
        n_classes: int = 2,
        bilstm_units: int = 128,
        n_bilstm_layers: int = 1,
        bilstm_dropout: float = 0.30,
        bilstm_emb_dim: int = 64,
        classification_hidden_units: int = 64,
        classification_dropout: float = 0.30,
        activation: str = "relu",
        focal_gamma: float = 0.0,
        focal_alpha: float | Sequence[float] | None = None,
        use_vae: bool = False,
        vae_loss_weight: float = 0.10,
        vae_beta: float = 0.05,
        use_class_weight: bool = False,
        use_subject_adversarial: bool = False,
        n_subject_classes: int | None = None,
        subject_adversarial_weight: float = 0.30,
        subject_loss_weight: float = 0.30,
        subject_hidden_units: int = 64,
        subject_dropout: float = 0.0,
        use_mldg: bool = False,
        mldg_inner_learning_rate: float = 1e-4,
        mldg_meta_test_weight: float = 1.0,
        name: str = "joint_v4_sts_model",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        classification_level = str(classification_level).lower()
        if classification_level not in {"window", "trial"}:
            raise ValueError("classification_level must be 'window' or 'trial'.")
        if graph_encoder is None:
            raise ValueError("graph_encoder must be provided.")
        if int(n_classes) < 2:
            raise ValueError("n_classes must be at least 2.")
        if int(bilstm_units) < 1 or int(n_bilstm_layers) < 1:
            raise ValueError("BiLSTM dimensions must be positive.")
        if int(bilstm_emb_dim) < 1:
            raise ValueError("bilstm_emb_dim must be positive.")
        if int(classification_hidden_units) < 1:
            raise ValueError("classification_hidden_units must be positive.")
        if not 0.0 <= float(bilstm_dropout) < 1.0:
            raise ValueError("bilstm_dropout must be in [0, 1).")
        if not 0.0 <= float(classification_dropout) < 1.0:
            raise ValueError("classification_dropout must be in [0, 1).")
        if float(vae_loss_weight) < 0.0 or float(vae_beta) < 0.0:
            raise ValueError("VAE weights must be non-negative.")
        if use_vae and decoder is None:
            raise ValueError("decoder is required when use_vae=True.")
        if float(subject_adversarial_weight) < 0.0:
            raise ValueError("subject_adversarial_weight must be non-negative.")
        if float(subject_loss_weight) < 0.0:
            raise ValueError("subject_loss_weight must be non-negative.")
        if int(subject_hidden_units) < 1:
            raise ValueError("subject_hidden_units must be positive.")
        if not 0.0 <= float(subject_dropout) < 1.0:
            raise ValueError("subject_dropout must be in [0, 1).")
        if float(mldg_inner_learning_rate) <= 0.0:
            raise ValueError("mldg_inner_learning_rate must be positive.")
        if float(mldg_meta_test_weight) < 0.0:
            raise ValueError("mldg_meta_test_weight must be non-negative.")

        self.graph_encoder = graph_encoder
        self.decoder = decoder
        self.classification_level = classification_level
        self.n_classes = int(n_classes)
        self.bilstm_units = int(bilstm_units)
        self.n_bilstm_layers = int(n_bilstm_layers)
        self.bilstm_dropout_rate = float(bilstm_dropout)
        self.bilstm_emb_dim = int(bilstm_emb_dim)
        self.classification_hidden_units = int(classification_hidden_units)
        self.classification_dropout_rate = float(classification_dropout)
        self.activation_name = str(activation)

        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = (
            None
            if focal_alpha is None
            else tuple(
                float(value)
                for value in (
                    focal_alpha
                    if isinstance(focal_alpha, (list, tuple))
                    else (focal_alpha,)
                )
            )
        )
        self.emotion_loss_fn = SparseCategoricalFocalLoss(
            gamma=self.focal_gamma,
            alpha=self.focal_alpha,
            from_logits=True,
        )

        self.use_vae = bool(use_vae)
        self.vae_loss_weight = float(vae_loss_weight)
        self.vae_beta = float(vae_beta)
        self.use_class_weight = bool(use_class_weight)

        self.subject_adversarial_enabled = bool(use_subject_adversarial)
        self.use_subject_adversarial = self.subject_adversarial_enabled
        self.n_subject_classes = (
            None if n_subject_classes is None else int(n_subject_classes)
        )
        self.subject_adversarial_weight = float(subject_adversarial_weight)
        self.subject_loss_weight = float(subject_loss_weight)
        self.subject_hidden_units = int(subject_hidden_units)
        self.subject_dropout_rate = float(subject_dropout)

        self.use_mldg = bool(use_mldg)
        self.mldg_inner_learning_rate = float(mldg_inner_learning_rate)
        self.mldg_meta_test_weight = float(mldg_meta_test_weight)

        # The subject IDs are only required as network inputs when the
        # subject-adversarial branch is enabled. MLDG receives subject IDs
        # separately through MetaLearningSubjectSequence.
        self.requires_subject_ids = self.subject_adversarial_enabled

        # In trial mode the graph encoder produces one sequence per window.
        # This pooling converts each graph sequence into one ordered window
        # embedding before the BiLSTM runs across the full trial.
        self.window_graph_pool = tf.keras.layers.GlobalAveragePooling1D(
            name="v4_window_graph_pool"
        )

        self.bilstm_layers = []
        self.bilstm_norms = []
        self.bilstm_dropouts = []
        for index in range(self.n_bilstm_layers):
            self.bilstm_layers.append(
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(
                        self.bilstm_units,
                        return_sequences=True,
                        name=f"v4_lstm_{index}",
                    ),
                    merge_mode="concat",
                    name=f"v4_bilstm_{index}",
                )
            )
            self.bilstm_norms.append(
                tf.keras.layers.LayerNormalization(
                    axis=-1,
                    name=f"v4_bilstm_ln_{index}",
                )
            )
            self.bilstm_dropouts.append(
                tf.keras.layers.Dropout(
                    self.bilstm_dropout_rate,
                    name=f"v4_bilstm_dropout_{index}",
                )
            )

        # Explicit BiLSTM embedding size requested by the v4 API.
        self.bilstm_embedding_projection = tf.keras.layers.Dense(
            self.bilstm_emb_dim,
            activation=None,
            name="v4_bilstm_embedding_projection",
        )
        self.bilstm_embedding_norm = tf.keras.layers.LayerNormalization(
            axis=-1,
            name="v4_bilstm_embedding_ln",
        )
        self.bilstm_embedding_activation = tf.keras.layers.Activation(
            self.activation_name,
            name="v4_bilstm_embedding_activation",
        )

        self.sequence_pool = tf.keras.layers.GlobalAveragePooling1D(
            name="v4_sequence_pool"
        )
        self.classification_hidden = tf.keras.layers.Dense(
            self.classification_hidden_units,
            activation=self.activation_name,
            name="v4_classification_hidden",
        )
        self.classification_norm = tf.keras.layers.LayerNormalization(
            axis=-1,
            name="v4_classification_ln",
        )
        self.classification_dropout = tf.keras.layers.Dropout(
            self.classification_dropout_rate,
            name="v4_classification_dropout",
        )
        self.logits_layer = tf.keras.layers.Dense(
            self.n_classes,
            activation=None,
            name="v4_logits",
        )

        # VAE posterior is applied to every graph-encoded EEG window.
        latent_features = int(getattr(self.graph_encoder, "emb_dim", self.bilstm_emb_dim))
        self.vae_latent_features = latent_features
        self.z_mean_projection = tf.keras.layers.Conv1D(
            latent_features,
            kernel_size=1,
            padding="same",
            activation=None,
            name="v4_vae_z_mean",
        )
        self.z_log_var_projection = tf.keras.layers.Conv1D(
            latent_features,
            kernel_size=1,
            padding="same",
            activation=None,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="v4_vae_z_log_var",
        )

        # Subject-adversarial head is configured fold-locally because LOSO
        # training pools can contain different subject ID sets.
        self.subject_gradient_reversal = None
        self.subject_hidden = None
        self.subject_dropout_layer = None
        self.subject_logits_layer = None
        if self.subject_adversarial_enabled and self.n_subject_classes is not None:
            self._configure_subject_head(self.n_subject_classes)

        self.classification_optimizer = None
        self.vae_optimizer = None

        # Trackers exposed to Keras fit/evaluate.
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.emotion_loss_tracker = tf.keras.metrics.Mean(name="emotion_loss")
        self.accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(
            name="accuracy"
        )
        self.subject_loss_tracker = tf.keras.metrics.Mean(name="subject_loss")
        self.subject_accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(
            name="subject_accuracy"
        )
        self.vae_loss_tracker = tf.keras.metrics.Mean(name="vae_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.mldg_meta_train_loss_tracker = tf.keras.metrics.Mean(
            name="mldg_meta_train_loss"
        )
        self.mldg_meta_test_loss_tracker = tf.keras.metrics.Mean(
            name="mldg_meta_test_loss"
        )
        self.mldg_meta_test_accuracy_tracker = (
            tf.keras.metrics.SparseCategoricalAccuracy(
                name="mldg_meta_test_accuracy"
            )
        )

    @property
    def metrics(self):
        output = [
            self.loss_tracker,
            self.emotion_loss_tracker,
            self.accuracy_tracker,
        ]
        if self.subject_adversarial_enabled:
            output.extend(
                [
                    self.subject_loss_tracker,
                    self.subject_accuracy_tracker,
                ]
            )
        if self.use_vae:
            output.extend(
                [
                    self.vae_loss_tracker,
                    self.reconstruction_loss_tracker,
                    self.kl_loss_tracker,
                ]
            )
        if self.use_mldg:
            output.extend(
                [
                    self.mldg_meta_train_loss_tracker,
                    self.mldg_meta_test_loss_tracker,
                    self.mldg_meta_test_accuracy_tracker,
                ]
            )
        return output

    def compile(
        self,
        classification_optimizer,
        vae_optimizer=None,
        **kwargs,
    ):
        if classification_optimizer is None:
            raise ValueError("classification_optimizer is required.")
        if self.use_vae and vae_optimizer is None:
            raise ValueError("vae_optimizer is required when use_vae=True.")
        kwargs.setdefault("jit_compile", False)
        super().compile(optimizer=classification_optimizer, **kwargs)
        self.classification_optimizer = classification_optimizer
        self.vae_optimizer = vae_optimizer

    def get_compile_config(self):
        config = super().get_compile_config()
        config.pop("optimizer", None)
        config.update(
            {
                "classification_optimizer": _serialize_keras_object(
                    self.classification_optimizer
                ),
                "vae_optimizer": _serialize_keras_object(self.vae_optimizer),
            }
        )
        return config

    def compile_from_config(self, config):
        config = dict(config)
        config.pop("optimizer", None)
        classification_optimizer = _deserialize_keras_object(
            config.pop("classification_optimizer")
        )
        vae_optimizer = _deserialize_keras_object(
            config.pop("vae_optimizer", None)
        )
        self.compile(
            classification_optimizer=classification_optimizer,
            vae_optimizer=vae_optimizer,
            **config,
        )

    def fit(self, *args, **kwargs):
        # --no-class-weight must remain authoritative even if an external CV
        # helper tries to supply class_weight.
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

    def _configure_subject_head(self, n_subject_classes):
        n_subject_classes = int(n_subject_classes)
        if n_subject_classes < 2:
            raise ValueError("Subject adversity requires at least two subjects.")
        if self.subject_logits_layer is not None:
            if self.n_subject_classes != n_subject_classes:
                raise ValueError(
                    "Subject head already configured for "
                    f"{self.n_subject_classes} classes, not {n_subject_classes}."
                )
            return

        self.n_subject_classes = n_subject_classes
        self.subject_gradient_reversal = GradientReversal(
            adversarial_weight=self.subject_adversarial_weight,
            name="v4_subject_gradient_reversal",
        )
        self.subject_hidden = tf.keras.layers.Dense(
            self.subject_hidden_units,
            activation=self.activation_name,
            name="v4_subject_hidden",
        )
        self.subject_dropout_layer = tf.keras.layers.Dropout(
            self.subject_dropout_rate,
            name="v4_subject_dropout",
        )
        self.subject_logits_layer = tf.keras.layers.Dense(
            self.n_subject_classes,
            activation=None,
            name="v4_subject_logits",
        )

    def prepare_fit_inputs(self, eeg_inputs, subject_ids):
        """Attach contiguous fold-local subject labels when required."""
        if not self.requires_subject_ids:
            return eeg_inputs

        import numpy as np

        eeg_array = np.asarray(eeg_inputs)
        subjects = np.asarray(subject_ids).reshape(-1)
        if len(eeg_array) != len(subjects):
            raise ValueError(
                "EEG samples and subject IDs must align; "
                f"got {len(eeg_array)} and {len(subjects)}."
            )

        unique_subjects = np.sort(np.unique(subjects))
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

    def _flatten_trial_windows(self, eeg_inputs):
        eeg_inputs = tf.convert_to_tensor(eeg_inputs, dtype=tf.float32)
        if eeg_inputs.shape.rank != 4:
            raise ValueError(
                "Trial mode expects (batch, windows, timesteps, features); "
                f"got {eeg_inputs.shape}."
            )
        shape = tf.shape(eeg_inputs)
        flat = tf.reshape(
            eeg_inputs,
            [shape[0] * shape[1], shape[2], shape[3]],
        )
        return flat, shape[0], shape[1]

    def _encode_for_classification(self, eeg_inputs, training):
        eeg_inputs = tf.convert_to_tensor(eeg_inputs, dtype=tf.float32)

        if self.classification_level == "window":
            if eeg_inputs.shape.rank != 3:
                raise ValueError(
                    "Window mode expects (batch, timesteps, features); "
                    f"got {eeg_inputs.shape}."
                )
            graph_sequence = self.graph_encoder(
                eeg_inputs,
                training=training,
            )
            sequence_for_bilstm = graph_sequence
            graph_output_for_diagnostics = graph_sequence
            window_embeddings = None

        else:
            flat_windows, batch_size, n_windows = self._flatten_trial_windows(
                eeg_inputs
            )
            flat_graph_sequence = self.graph_encoder(
                flat_windows,
                training=training,
            )
            flat_window_embeddings = self.window_graph_pool(flat_graph_sequence)
            embedding_dim = tf.shape(flat_window_embeddings)[-1]
            window_embeddings = tf.reshape(
                flat_window_embeddings,
                [batch_size, n_windows, embedding_dim],
            )
            sequence_for_bilstm = window_embeddings

            graph_shape = tf.shape(flat_graph_sequence)
            graph_output_for_diagnostics = tf.reshape(
                flat_graph_sequence,
                [
                    batch_size,
                    n_windows,
                    graph_shape[1],
                    graph_shape[2],
                ],
            )

        sequence = sequence_for_bilstm
        for bilstm, norm, dropout in zip(
            self.bilstm_layers,
            self.bilstm_norms,
            self.bilstm_dropouts,
        ):
            sequence = bilstm(sequence, training=training)
            sequence = norm(sequence)
            sequence = dropout(sequence, training=training)

        bilstm_embedding_sequence = self.bilstm_embedding_projection(sequence)
        bilstm_embedding_sequence = self.bilstm_embedding_norm(
            bilstm_embedding_sequence
        )
        bilstm_embedding_sequence = self.bilstm_embedding_activation(
            bilstm_embedding_sequence
        )

        shared_embedding = self.sequence_pool(bilstm_embedding_sequence)
        hidden = self.classification_hidden(shared_embedding)
        hidden = self.classification_norm(hidden)
        hidden = self.classification_dropout(hidden, training=training)
        logits = self.logits_layer(hidden)

        return {
            "graph_sequence": graph_output_for_diagnostics,
            "window_embeddings": window_embeddings,
            "bilstm_sequence": sequence,
            "bilstm_embedding_sequence": bilstm_embedding_sequence,
            "shared_embedding": shared_embedding,
            "classification_latent": hidden,
            "logits": logits,
            "probabilities": tf.nn.softmax(logits, axis=-1),
        }

    def call(self, inputs, training=False):
        eeg_inputs, _ = self._split_eeg_and_subject_inputs(inputs)
        return self._encode_for_classification(
            eeg_inputs,
            training=training,
        )["logits"]

    def _subject_logits(self, shared_embedding, training):
        if self.subject_logits_layer is None:
            raise RuntimeError(
                "Subject head is not configured. The fold must call "
                "prepare_fit_inputs(...) before fitting."
            )
        x = self.subject_gradient_reversal(shared_embedding)
        x = self.subject_hidden(x)
        x = self.subject_dropout_layer(x, training=training)
        return self.subject_logits_layer(x)

    def _regularization_loss(self, dtype):
        if not self.losses:
            return tf.zeros((), dtype=dtype)
        return tf.add_n([tf.cast(loss, dtype) for loss in self.losses])

    def _classification_losses(
        self,
        x,
        y_flat,
        training,
        sample_weight=None,
        include_subject_adversarial=True,
    ):
        eeg_inputs, subject_ids = self._split_eeg_and_subject_inputs(x)
        outputs = self._encode_for_classification(
            eeg_inputs,
            training=training,
        )

        emotion_loss = self.emotion_loss_fn(
            y_flat,
            outputs["logits"],
            sample_weight=sample_weight,
        )

        zero = tf.zeros((), dtype=emotion_loss.dtype)
        subject_loss = zero
        subject_logits = None
        subject_targets = None

        subject_enabled = (
            include_subject_adversarial
            and self.subject_adversarial_enabled
            and subject_ids is not None
        )
        if subject_enabled:
            subject_targets = tf.cast(
                tf.reshape(subject_ids, [-1]),
                tf.int32,
            )
            subject_logits = self._subject_logits(
                outputs["shared_embedding"],
                training=training,
            )
            per_subject_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=subject_targets,
                logits=subject_logits,
            )
            subject_loss = tf.reduce_mean(per_subject_loss)

        regularization_loss = self._regularization_loss(emotion_loss.dtype)
        objective = (
            emotion_loss
            + tf.cast(self.subject_loss_weight, emotion_loss.dtype) * subject_loss
            + regularization_loss
        )

        outputs.update(
            {
                "subject_logits": subject_logits,
                "subject_targets": subject_targets,
            }
        )
        return {
            "objective": objective,
            "emotion_loss": emotion_loss,
            "subject_loss": subject_loss,
            "regularization_loss": regularization_loss,
        }, outputs

    @staticmethod
    def _reparameterize(z_mean, z_log_var):
        epsilon = tf.random.normal(tf.shape(z_mean), dtype=z_mean.dtype)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def _vae_losses(self, x, training):
        if not self.use_vae:
            zero = tf.zeros((), dtype=tf.float32)
            return {
                "objective": zero,
                "vae_loss": zero,
                "reconstruction_loss": zero,
                "kl_loss": zero,
            }, None

        eeg_inputs, _ = self._split_eeg_and_subject_inputs(x)
        eeg_inputs = tf.convert_to_tensor(eeg_inputs, dtype=tf.float32)

        if self.classification_level == "trial":
            flat_windows, _, _ = self._flatten_trial_windows(eeg_inputs)
        else:
            flat_windows = eeg_inputs

        graph_sequence = self.graph_encoder(
            flat_windows,
            training=training,
        )
        z_mean = self.z_mean_projection(graph_sequence)
        raw_z_log_var = self.z_log_var_projection(graph_sequence)
        z_log_var = tf.clip_by_value(raw_z_log_var, -20.0, 20.0)
        z = (
            self._reparameterize(z_mean, z_log_var)
            if training
            else z_mean
        )
        reconstruction = self.decoder(z, training=training)

        reconstruction_loss = tf.reduce_mean(
            tf.square(flat_windows - reconstruction)
        )
        kl_per_value = -0.5 * (
            1.0
            + z_log_var
            - tf.square(z_mean)
            - tf.exp(z_log_var)
        )
        kl_loss = tf.reduce_mean(kl_per_value)
        vae_loss = (
            reconstruction_loss
            + tf.cast(self.vae_beta, kl_loss.dtype) * kl_loss
        )
        objective = (
            tf.cast(self.vae_loss_weight, vae_loss.dtype)
            * vae_loss
        )

        return {
            "objective": objective,
            "vae_loss": vae_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }, {
            "z_mean": z_mean,
            "z_log_var": z_log_var,
            "latent_sequence": z,
            "reconstruction": reconstruction,
        }

    def _classification_variables(self):
        components = [
            self.graph_encoder,
            *self.bilstm_layers,
            *self.bilstm_norms,
            self.bilstm_embedding_projection,
            self.bilstm_embedding_norm,
            self.classification_hidden,
            self.classification_norm,
            self.logits_layer,
        ]
        if self.subject_logits_layer is not None:
            components.extend(
                [
                    self.subject_gradient_reversal,
                    self.subject_hidden,
                    self.subject_logits_layer,
                ]
            )

        variables = []
        for component in components:
            variables.extend(component.trainable_variables)
        return _deduplicate_variables(variables)

    def _vae_variables(self):
        if not self.use_vae:
            return []
        components = [
            self.graph_encoder,
            self.z_mean_projection,
            self.z_log_var_projection,
            self.decoder,
        ]
        variables = []
        for component in components:
            variables.extend(component.trainable_variables)
        return _deduplicate_variables(variables)

    @staticmethod
    def _apply_gradients(optimizer, gradients, variables):
        pairs = [
            (gradient, variable)
            for gradient, variable in zip(gradients, variables)
            if gradient is not None
        ]
        if pairs:
            optimizer.apply_gradients(pairs)

    @staticmethod
    def _dense_gradient(gradient, variable):
        if gradient is None:
            return None
        if isinstance(gradient, tf.IndexedSlices):
            gradient = tf.convert_to_tensor(gradient)
        return tf.cast(gradient, variable.dtype)

    def _combine_first_order_gradients(
        self,
        gradients_a,
        gradients_b,
        variables,
    ):
        output = []
        for gradient_a, gradient_b, variable in zip(
            gradients_a,
            gradients_b,
            variables,
        ):
            gradient_a = self._dense_gradient(gradient_a, variable)
            gradient_b = self._dense_gradient(gradient_b, variable)
            if gradient_a is None and gradient_b is None:
                output.append(None)
            elif gradient_a is None:
                output.append(
                    tf.cast(self.mldg_meta_test_weight, variable.dtype)
                    * gradient_b
                )
            elif gradient_b is None:
                output.append(gradient_a)
            else:
                output.append(
                    gradient_a
                    + tf.cast(self.mldg_meta_test_weight, variable.dtype)
                    * gradient_b
                )
        return output

    @staticmethod
    def _unpack_data(data):
        return tf.keras.utils.unpack_x_y_sample_weight(data)

    @classmethod
    def _unpack_mldg_episode(cls, data):
        x, y, sample_weight = cls._unpack_data(data)

        if not isinstance(x, Mapping) or not isinstance(y, Mapping):
            raise ValueError(
                "MLDG expects x/y mappings containing meta_train/meta_test."
            )
        if "meta_train" not in x or "meta_test" not in x:
            raise ValueError("MLDG x is missing meta_train or meta_test.")
        if "meta_train" not in y or "meta_test" not in y:
            raise ValueError("MLDG y is missing meta_train or meta_test.")

        if sample_weight is None:
            sample_weight_a = sample_weight_b = None
        elif isinstance(sample_weight, Mapping):
            sample_weight_a = sample_weight.get("meta_train")
            sample_weight_b = sample_weight.get("meta_test")
        else:
            raise ValueError("MLDG sample_weight must be a mapping or None.")

        return (
            x["meta_train"],
            y["meta_train"],
            sample_weight_a,
            x["meta_test"],
            y["meta_test"],
            sample_weight_b,
        )

    @classmethod
    def _merge_episode_inputs(cls, x_a, x_b):
        eeg_a, subject_a = cls._split_eeg_and_subject_inputs(x_a)
        eeg_b, subject_b = cls._split_eeg_and_subject_inputs(x_b)
        merged_eeg = tf.concat([eeg_a, eeg_b], axis=0)

        if subject_a is None or subject_b is None:
            return merged_eeg
        return {
            "eeg": merged_eeg,
            "subject_id": tf.concat([subject_a, subject_b], axis=0),
        }

    def _update_metrics(
        self,
        classification_losses,
        classification_outputs,
        y_flat,
        sample_weight,
        vae_losses=None,
        meta_test_loss=None,
        meta_test_outputs=None,
        meta_test_y=None,
        meta_test_sample_weight=None,
    ):
        total = classification_losses["objective"]
        if vae_losses is not None:
            total += tf.cast(vae_losses["objective"], total.dtype)
        if meta_test_loss is not None:
            total += (
                tf.cast(self.mldg_meta_test_weight, total.dtype)
                * tf.cast(meta_test_loss, total.dtype)
            )

        self.loss_tracker.update_state(total)
        self.emotion_loss_tracker.update_state(
            classification_losses["emotion_loss"]
        )
        self.accuracy_tracker.update_state(
            y_flat,
            classification_outputs["logits"],
            sample_weight=sample_weight,
        )

        if self.subject_adversarial_enabled:
            self.subject_loss_tracker.update_state(
                classification_losses["subject_loss"]
            )
            if (
                classification_outputs.get("subject_logits") is not None
                and classification_outputs.get("subject_targets") is not None
            ):
                self.subject_accuracy_tracker.update_state(
                    classification_outputs["subject_targets"],
                    classification_outputs["subject_logits"],
                )

        if self.use_vae and vae_losses is not None:
            self.vae_loss_tracker.update_state(vae_losses["vae_loss"])
            self.reconstruction_loss_tracker.update_state(
                vae_losses["reconstruction_loss"]
            )
            self.kl_loss_tracker.update_state(vae_losses["kl_loss"])

        if self.use_mldg and meta_test_loss is not None:
            self.mldg_meta_train_loss_tracker.update_state(
                classification_losses["emotion_loss"]
            )
            self.mldg_meta_test_loss_tracker.update_state(meta_test_loss)
            if meta_test_outputs is not None and meta_test_y is not None:
                self.mldg_meta_test_accuracy_tracker.update_state(
                    meta_test_y,
                    meta_test_outputs["logits"],
                    sample_weight=meta_test_sample_weight,
                )

    def _standard_train_step(self, data):
        x, y, sample_weight = self._unpack_data(data)
        y_flat = self._flatten_labels(y)

        with tf.GradientTape() as tape:
            classification_losses, classification_outputs = (
                self._classification_losses(
                    x,
                    y_flat,
                    training=True,
                    sample_weight=sample_weight,
                    include_subject_adversarial=True,
                )
            )
        classification_variables = self._classification_variables()
        gradients = tape.gradient(
            classification_losses["objective"],
            classification_variables,
        )
        self._apply_gradients(
            self.classification_optimizer,
            gradients,
            classification_variables,
        )

        vae_losses = None
        if self.use_vae:
            with tf.GradientTape() as vae_tape:
                vae_losses, _ = self._vae_losses(x, training=True)
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

        self._update_metrics(
            classification_losses,
            classification_outputs,
            y_flat,
            sample_weight,
            vae_losses=vae_losses,
        )
        return {metric.name: metric.result() for metric in self.metrics}

    def _mldg_train_step(self, data):
        (
            x_a,
            y_a,
            sample_weight_a,
            x_b,
            y_b,
            sample_weight_b,
        ) = self._unpack_mldg_episode(data)

        y_a_flat = self._flatten_labels(y_a)
        y_b_flat = self._flatten_labels(y_b)

        with tf.GradientTape() as meta_train_tape:
            losses_a, outputs_a = self._classification_losses(
                x_a,
                y_a_flat,
                training=True,
                sample_weight=sample_weight_a,
                include_subject_adversarial=True,
            )

        variables = self._classification_variables()
        gradients_a = meta_train_tape.gradient(
            losses_a["objective"],
            variables,
        )

        original_values = [tf.identity(variable) for variable in variables]
        for variable, gradient in zip(variables, gradients_a):
            gradient = self._dense_gradient(gradient, variable)
            if gradient is not None:
                variable.assign_sub(
                    tf.cast(self.mldg_inner_learning_rate, variable.dtype)
                    * gradient
                )

        # Meta-test B deliberately evaluates emotion only. Subject adversity
        # and VAE reconstruction are not part of the pseudo-unseen objective.
        with tf.GradientTape() as meta_test_tape:
            losses_b, outputs_b = self._classification_losses(
                x_b,
                y_b_flat,
                training=False,
                sample_weight=sample_weight_b,
                include_subject_adversarial=False,
            )
            meta_test_emotion_loss = losses_b["emotion_loss"]

        gradients_b = meta_test_tape.gradient(
            meta_test_emotion_loss,
            variables,
        )

        for variable, original_value in zip(variables, original_values):
            variable.assign(original_value)

        combined_gradients = self._combine_first_order_gradients(
            gradients_a,
            gradients_b,
            variables,
        )
        self._apply_gradients(
            self.classification_optimizer,
            combined_gradients,
            variables,
        )

        vae_losses = None
        if self.use_vae:
            x_vae = self._merge_episode_inputs(x_a, x_b)
            with tf.GradientTape() as vae_tape:
                vae_losses, _ = self._vae_losses(
                    x_vae,
                    training=True,
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

        self._update_metrics(
            losses_a,
            outputs_a,
            y_a_flat,
            sample_weight_a,
            vae_losses=vae_losses,
            meta_test_loss=meta_test_emotion_loss,
            meta_test_outputs=outputs_b,
            meta_test_y=y_b_flat,
            meta_test_sample_weight=sample_weight_b,
        )
        return {metric.name: metric.result() for metric in self.metrics}

    def train_step(self, data):
        if self.classification_optimizer is None:
            raise RuntimeError("Call model.compile(...) before model.fit(...).")
        if self.use_vae and self.vae_optimizer is None:
            raise RuntimeError(
                "VAE is enabled but no vae_optimizer is configured."
            )
        if self.use_mldg:
            return self._mldg_train_step(data)
        return self._standard_train_step(data)

    def test_step(self, data):
        x, y, sample_weight = self._unpack_data(data)
        y_flat = self._flatten_labels(y)

        classification_losses, classification_outputs = (
            self._classification_losses(
                x,
                y_flat,
                training=False,
                sample_weight=sample_weight,
                include_subject_adversarial=True,
            )
        )
        vae_losses = None
        if self.use_vae:
            vae_losses, _ = self._vae_losses(x, training=False)

        self._update_metrics(
            classification_losses,
            classification_outputs,
            y_flat,
            sample_weight,
            vae_losses=vae_losses,
        )
        return {metric.name: metric.result() for metric in self.metrics}

    def predict_step(self, data):
        x = data[0] if isinstance(data, tuple) else data
        return self(x, training=False)

    def predict_diagnostics(self, inputs, batch_size=None):
        eeg_inputs, _ = self._split_eeg_and_subject_inputs(inputs)
        eeg_inputs = tf.convert_to_tensor(eeg_inputs, dtype=tf.float32)

        n_samples = int(tf.shape(eeg_inputs)[0].numpy())
        effective_batch_size = (
            n_samples if batch_size is None else int(batch_size)
        )
        if effective_batch_size < 1:
            raise ValueError("batch_size must be positive.")

        collected = {
            "encoder_output": [],
            "graph_sequence": [],
            "bilstm_sequence": [],
            "bilstm_embedding_sequence": [],
            "classification_latent": [],
            "logits": [],
            "probabilities": [],
            "logit_margin": [],
        }

        for start in range(0, n_samples, effective_batch_size):
            batch = eeg_inputs[start : start + effective_batch_size]
            outputs = self._encode_for_classification(
                batch,
                training=False,
            )
            logits = outputs["logits"]
            if self.n_classes == 2:
                margin = logits[:, 1] - logits[:, 0]
            else:
                top_logits = tf.math.top_k(
                    logits,
                    k=2,
                    sorted=True,
                ).values
                margin = top_logits[:, 0] - top_logits[:, 1]

            # encoder_output remains batch-aligned in both modes.
            encoder_output = (
                outputs["graph_sequence"]
                if self.classification_level == "window"
                else outputs["window_embeddings"]
            )
            values = {
                "encoder_output": encoder_output,
                "graph_sequence": outputs["graph_sequence"],
                "bilstm_sequence": outputs["bilstm_sequence"],
                "bilstm_embedding_sequence": outputs[
                    "bilstm_embedding_sequence"
                ],
                "classification_latent": outputs["classification_latent"],
                "logits": logits,
                "probabilities": outputs["probabilities"],
                "logit_margin": margin,
            }
            for key, value in values.items():
                collected[key].append(value)

        return {
            key: tf.concat(values, axis=0)
            for key, values in collected.items()
        }

    def predict_mc_probabilities(
        self,
        inputs,
        n_samples=1,
        seed=None,
    ):
        # The VAE is auxiliary; the emotion classifier itself is deterministic.
        del seed
        if int(n_samples) < 1:
            raise ValueError("n_samples must be at least 1.")
        logits = self(inputs, training=False)
        probabilities = tf.nn.softmax(logits, axis=-1)
        return {
            "mean_probabilities": probabilities,
            "probability_samples": tf.repeat(
                probabilities[tf.newaxis, ...],
                repeats=int(n_samples),
                axis=0,
            ),
        }

    def get_adjacency_matrices(self):
        if hasattr(self.graph_encoder, "get_adjacency_matrices"):
            return {
                "band_separated_gcn": (
                    self.graph_encoder.get_adjacency_matrices()
                )
            }
        return {}

    def get_band_features(self, inputs, training=False):
        eeg_inputs, _ = self._split_eeg_and_subject_inputs(inputs)
        eeg_inputs = tf.convert_to_tensor(eeg_inputs, dtype=tf.float32)

        if self.classification_level == "window":
            return self.graph_encoder.get_band_features(
                eeg_inputs,
                training=training,
            )

        flat_windows, batch_size, n_windows = self._flatten_trial_windows(
            eeg_inputs
        )
        flat_features = self.graph_encoder.get_band_features(
            flat_windows,
            training=training,
        )
        output = {}
        for key, value in flat_features.items():
            shape = tf.shape(value)
            output[key] = tf.reshape(
                value,
                [
                    batch_size,
                    n_windows,
                    shape[1],
                    shape[2],
                ],
            )
        return output

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
                "bilstm_emb_dim": self.bilstm_emb_dim,
                "classification_hidden_units": self.classification_hidden_units,
                "classification_dropout": self.classification_dropout_rate,
                "activation": self.activation_name,
                "focal_gamma": self.focal_gamma,
                "focal_alpha": self.focal_alpha,
                "use_vae": self.use_vae,
                "vae_loss_weight": self.vae_loss_weight,
                "vae_beta": self.vae_beta,
                "use_class_weight": self.use_class_weight,
                "use_subject_adversarial": (
                    self.subject_adversarial_enabled
                ),
                "n_subject_classes": self.n_subject_classes,
                "subject_adversarial_weight": (
                    self.subject_adversarial_weight
                ),
                "subject_loss_weight": self.subject_loss_weight,
                "subject_hidden_units": self.subject_hidden_units,
                "subject_dropout": self.subject_dropout_rate,
                "use_mldg": self.use_mldg,
                "mldg_inner_learning_rate": self.mldg_inner_learning_rate,
                "mldg_meta_test_weight": self.mldg_meta_test_weight,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config["graph_encoder"] = _deserialize_keras_object(
            config["graph_encoder"]
        )
        config["decoder"] = _deserialize_keras_object(
            config.get("decoder")
        )
        return cls(**config)


def build_joint_sts_model(
    input_shape,
    *,
    classification_level="trial",
    n_classes=2,
    n_channels=14,
    n_bands=3,
    t_down=2,
    temporal_pool_sizes=(2,),
    gcn_units=(128, 64),
    spectral_emb_dim=128,
    gcn_dropout=0.20,
    gcn_activation="relu",
    gcn_use_batch_norm=False,
    graph_self_loop_bias=2.0,
    graph_identity_mix=0.0,
    graph_adjacency_reg_weight=1e-4,
    bilstm_units=128,
    n_bilstm_layers=1,
    bilstm_dropout=0.30,
    bilstm_emb_dim=64,
    classification_hidden_units=64,
    classification_dropout=0.30,
    activation="relu",
    focal_gamma=0.0,
    focal_alpha=None,
    use_vae=False,
    vae_loss_weight=0.10,
    vae_beta=0.05,
    vae_learning_rate=5e-5,
    use_class_weight=False,
    use_subject_adversarial=False,
    n_subject_classes=None,
    subject_adversarial_weight=0.30,
    subject_loss_weight=0.30,
    subject_hidden_units=64,
    subject_dropout=0.0,
    use_mldg=False,
    mldg_inner_learning_rate=1e-4,
    mldg_meta_test_weight=1.0,
    optimizer_name="adamw",
    classification_learning_rate=1e-4,
    weight_decay=1e-4,
    model_name="joint_v4_sts_model",
    **unused_kwargs,
):
    """Build and compile the complete v4 model."""

    classification_level = str(classification_level).lower()
    input_shape = tuple(int(value) for value in input_shape)

    if classification_level == "window":
        if len(input_shape) != 2:
            raise ValueError(
                "Window-level v4 expects input_shape=(timesteps, features); "
                f"got {input_shape}."
            )
        window_timesteps, n_features = input_shape
        dummy_shape = (1, window_timesteps, n_features)

    elif classification_level == "trial":
        if len(input_shape) != 3:
            raise ValueError(
                "Trial-level v4 expects "
                "input_shape=(windows, timesteps, features); "
                f"got {input_shape}."
            )
        n_trial_windows, window_timesteps, n_features = input_shape
        dummy_shape = (
            1,
            n_trial_windows,
            window_timesteps,
            n_features,
        )

    else:
        raise ValueError("classification_level must be 'window' or 'trial'.")

    expected_features = int(n_channels) * int(n_bands)
    if n_features != expected_features:
        raise ValueError(
            "Input feature count must equal n_channels * n_bands; "
            f"got {n_features} != {n_channels} * {n_bands}."
        )

    graph_encoder = BandSeparatedGCNEncoder(
        timesteps=window_timesteps,
        t_down=int(t_down),
        n_channels=int(n_channels),
        n_bands=int(n_bands),
        gcn_units=tuple(int(value) for value in gcn_units),
        temporal_pool_sizes=(
            None
            if temporal_pool_sizes is None
            else tuple(int(value) for value in temporal_pool_sizes)
        ),
        emb_dim=int(spectral_emb_dim),
        dropout=float(gcn_dropout),
        activation=str(gcn_activation),
        use_batch_norm=bool(gcn_use_batch_norm),
        graph_self_loop_bias=float(graph_self_loop_bias),
        graph_identity_mix=float(graph_identity_mix),
        graph_adjacency_reg_weight=float(
            graph_adjacency_reg_weight
        ),
        name="v4_band_separated_gcn",
    )

    decoder = (
        GCNDecoder.from_encoder(
            graph_encoder,
            name="v4_vae_decoder",
        )
        if bool(use_vae)
        else None
    )

    model = JointSTSModel(
        graph_encoder=graph_encoder,
        decoder=decoder,
        classification_level=classification_level,
        n_classes=int(n_classes),
        bilstm_units=int(bilstm_units),
        n_bilstm_layers=int(n_bilstm_layers),
        bilstm_dropout=float(bilstm_dropout),
        bilstm_emb_dim=int(bilstm_emb_dim),
        classification_hidden_units=int(classification_hidden_units),
        classification_dropout=float(classification_dropout),
        activation=str(activation),
        focal_gamma=float(focal_gamma),
        focal_alpha=focal_alpha,
        use_vae=bool(use_vae),
        vae_loss_weight=float(vae_loss_weight),
        vae_beta=float(vae_beta),
        use_class_weight=bool(use_class_weight),
        use_subject_adversarial=bool(use_subject_adversarial),
        n_subject_classes=n_subject_classes,
        subject_adversarial_weight=float(subject_adversarial_weight),
        subject_loss_weight=float(subject_loss_weight),
        subject_hidden_units=int(subject_hidden_units),
        subject_dropout=float(subject_dropout),
        use_mldg=bool(use_mldg),
        mldg_inner_learning_rate=float(mldg_inner_learning_rate),
        mldg_meta_test_weight=float(mldg_meta_test_weight),
        name=model_name,
    )

    classification_optimizer = _build_optimizer(
        optimizer_name=optimizer_name,
        learning_rate=float(classification_learning_rate),
        weight_decay=float(weight_decay),
    )
    vae_optimizer = (
        _build_optimizer(
            optimizer_name=optimizer_name,
            learning_rate=float(vae_learning_rate),
            weight_decay=float(weight_decay),
        )
        if bool(use_vae)
        else None
    )

    model.compile(
        classification_optimizer=classification_optimizer,
        vae_optimizer=vae_optimizer,
        jit_compile=False,
    )

    # Build immediately only when the subject-adversarial head is disabled
    # (or already has a known class count). When subject classes are fold-local,
    # prepare_fit_inputs(...) must create the subject head before Keras marks
    # the model as built; Keras 3 disallows adding new state afterward.
    if not bool(use_subject_adversarial) or n_subject_classes is not None:
        _ = model(
            tf.zeros(dummy_shape, dtype=tf.float32),
            training=False,
        )
    return model
