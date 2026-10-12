"""SIC sequence, loss and training behavior shared by v11 and v15.

Version modules own the architecture/configuration and reconstruction objective.
This base keeps the identical source/calibration steps in one place and delegates
first-order meta-learning to EEGProc's domain_generalization package.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import numpy as np
import tensorflow as tf

from ...domain_generalization.meta_learning import fit_mldg as fit_sic_mldg
from ...supervised.rnn_architectures import BiGRUClassifier, GRUClassifier
from ...supervised.variational_classifier import VariationalClassifier
from ...unsupervised.GNN.GCNMTL import compute_mtl_shared_mi_adjacency


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


_CALIBRATION_LOSS_ALIASES = {
    "ce": "cross_entropy",
    "crossentropy": "cross_entropy",
    "cross-entropy": "cross_entropy",
    "categorical_crossentropy": "cross_entropy",
    "sparse_categorical_crossentropy": "cross_entropy",
    "focal": "focal",
    "focal_loss": "focal",
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


def _normalize_calibration_loss(value: str) -> str:
    normalized = str(value).strip().lower().replace(" ", "_")
    normalized = _CALIBRATION_LOSS_ALIASES.get(normalized, normalized)
    if normalized not in {"focal", "cross_entropy"}:
        raise ValueError(
            "calibration_loss must be 'focal' or 'cross_entropy'; "
            f"got {value!r}."
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


def _resolve_recurrent_widths(
    name: str,
    values: int | Sequence[int],
    n_layers: int | None,
) -> tuple[int, ...]:
    """Resolve a repeated scalar width or one explicit width per RNN layer."""
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        widths = tuple(int(value) for value in values)
        if not widths or any(width < 1 for width in widths):
            raise ValueError(f"{name} must contain positive integers.")
        if n_layers is not None and int(n_layers) != len(widths):
            raise ValueError(
                f"{name} has {len(widths)} entries but n_classifier_rnn_layers="
                f"{n_layers}."
            )
        return widths

    width = int(values)
    if width < 1:
        raise ValueError(f"{name} must be positive.")
    resolved_layers = 1 if n_layers is None else int(n_layers)
    if resolved_layers < 1:
        raise ValueError("n_classifier_rnn_layers must be positive.")
    return (width,) * resolved_layers


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
class StreamingR2(tf.keras.metrics.Metric):
    """Epoch-wide coefficient of determination for EEG reconstruction."""

    def __init__(self, name="decoder_r2", **kwargs):
        super().__init__(name=name, **kwargs)
        self.squared_error_sum = self.add_weight(
            name="squared_error_sum",
            initializer="zeros",
        )
        self.target_sum = self.add_weight(name="target_sum", initializer="zeros")
        self.target_squared_sum = self.add_weight(
            name="target_squared_sum",
            initializer="zeros",
        )
        self.target_count = self.add_weight(
            name="target_count",
            initializer="zeros",
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        del sample_weight
        y_true = tf.cast(tf.reshape(y_true, [-1]), self.dtype)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), self.dtype)
        self.squared_error_sum.assign_add(
            tf.reduce_sum(tf.square(y_true - y_pred))
        )
        self.target_sum.assign_add(tf.reduce_sum(y_true))
        self.target_squared_sum.assign_add(tf.reduce_sum(tf.square(y_true)))
        self.target_count.assign_add(tf.cast(tf.size(y_true), self.dtype))

    def result(self):
        target_total_sum_of_squares = self.target_squared_sum - tf.math.divide_no_nan(
            tf.square(self.target_sum),
            self.target_count,
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

    def reset_state(self):
        for variable in self.variables:
            variable.assign(tf.zeros_like(variable))


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


class SICModelCore(tf.keras.Model):
    """Common sequence processing and optimizer steps; no version-specific layers."""

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

        if graph_sequence is not None and bilstm_sequence is not None:
            tf.debugging.assert_equal(
                tf.shape(graph_sequence)[1],
                tf.shape(bilstm_sequence)[1],
                message=(
                    "GCN-GRU and BiLSTM feature sequence lengths do not match."
                ),
            )
        feature_parts = [
            value
            for value in (graph_sequence, bilstm_sequence)
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
            "combined_feature_sequence": combined_sequence,
        }

    def _sequence_features_for_prediction(
        self,
        flat_features,
        batch_size=None,
        n_windows=None,
        training: bool = False,
    ):
        # Keep every encoded timestep. In trial mode, restore the window axis
        # and then merge only the adjacent window/time axes, which preserves
        # chronological order exactly:
        #   (batch * windows, time, features)
        #       -> (batch, windows, time, features)
        #       -> (batch, windows * time, features)
        if self.classification_level == "window":
            window_feature_sequences = flat_features
            classifier_sequence = flat_features
        else:
            steps_per_window = tf.shape(flat_features)[1]
            window_feature_sequences = tf.reshape(
                flat_features,
                [
                    batch_size,
                    n_windows,
                    steps_per_window,
                    self.combined_feature_dim,
                ],
            )
            classifier_sequence = tf.reshape(
                window_feature_sequences,
                [batch_size, -1, self.combined_feature_dim],
            )

        recurrent_training = bool(training) and bool(
            self.trial_recurrent_classifier.trainable
        )
        sequence_features = self.trial_recurrent_classifier(
            classifier_sequence,
            training=recurrent_training,
        )
        return (
            sequence_features,
            window_feature_sequences,
            classifier_sequence,
        )

    def _classifier_forward(self, pooled_features, training: bool):
        # The recurrent full-sequence state goes directly to the sole
        # logits-producing VC head.
        classification_embedding = pooled_features
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
        (
            pooled_features,
            window_features,
            classifier_sequence,
        ) = self._sequence_features_for_prediction(
            encoder_outputs["combined_feature_sequence"],
            batch_size=batch_size,
            n_windows=n_windows,
            training=bool(classifier_training),
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
                "classifier_sequence": classifier_sequence,
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

    def _per_sample_cross_entropy(
        self,
        logits,
        y_flat,
        *,
        label_smoothing: float,
    ):
        y_flat = tf.cast(tf.reshape(y_flat, [-1]), tf.int32)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        row_ids = tf.range(tf.shape(y_flat)[0], dtype=tf.int32)
        gather_ids = tf.stack([row_ids, y_flat], axis=1)
        log_p_t = tf.gather_nd(log_probs, gather_ids)
        if label_smoothing:
            hard_targets = tf.one_hot(
                y_flat,
                depth=self.n_classes,
                dtype=logits.dtype,
            )
            smoothing = tf.cast(label_smoothing, logits.dtype)
            smoothed_targets = (
                (1.0 - smoothing) * hard_targets
                + smoothing / tf.cast(self.n_classes, logits.dtype)
            )
            return tf.nn.softmax_cross_entropy_with_logits(
                labels=smoothed_targets,
                logits=logits,
            )
        return -log_p_t

    def _per_sample_focal_loss(
        self,
        logits,
        y_flat,
        *,
        gamma: float,
        alpha: float | None,
        label_smoothing: float,
    ):
        """Focal loss from the same sole VC logits used for prediction."""
        y_flat = tf.cast(tf.reshape(y_flat, [-1]), tf.int32)
        probs = tf.nn.softmax(logits, axis=-1)
        row_ids = tf.range(tf.shape(y_flat)[0], dtype=tf.int32)
        gather_ids = tf.stack([row_ids, y_flat], axis=1)
        p_t = tf.gather_nd(probs, gather_ids)
        base_loss = self._per_sample_cross_entropy(
            logits,
            y_flat,
            label_smoothing=float(label_smoothing),
        )
        modulating = tf.pow(
            tf.maximum(1.0 - p_t, tf.keras.backend.epsilon()),
            tf.cast(gamma, p_t.dtype),
        )
        loss = modulating * base_loss

        if alpha is not None:
            if self.n_classes != 2:
                raise ValueError(
                    "Scalar focal_alpha is currently defined only for binary SIC."
                )
            alpha = tf.cast(alpha, loss.dtype)
            alpha_t = tf.where(
                tf.equal(y_flat, 1),
                alpha,
                1.0 - alpha,
            )
            loss = alpha_t * loss
        return loss

    def _per_sample_calibration_loss(self, logits, y_flat):
        if self.calibration_loss == "cross_entropy":
            return self._per_sample_cross_entropy(
                logits,
                y_flat,
                label_smoothing=self.calibration_label_smoothing,
            )
        return self._per_sample_focal_loss(
            logits,
            y_flat,
            gamma=self.calibration_focal_gamma,
            alpha=self.calibration_focal_alpha,
            label_smoothing=self.calibration_label_smoothing,
        )

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
        calibration_classification_loss = None
        calibration_weighted_classification_loss = None
        if calibration:
            calibration_per_sample = self._per_sample_calibration_loss(
                logits,
                y_flat,
            )
            calibration_classification_loss = tf.reduce_mean(
                calibration_per_sample
            )
            calibration_weighted_classification_loss = self._weighted_mean(
                calibration_per_sample,
                sample_weight,
            )

        if calibration and not self.calibration_use_vc_target:
            # The VC head still produces and learns the logits. This switch
            # disables only its auxiliary variational regularizers.
            selected_loss = calibration_weighted_classification_loss
            zero = tf.zeros((), dtype=selected_loss.dtype)
            focal_loss = (
                selected_loss if self.calibration_loss == "focal" else zero
            )
            return {
                "total_loss": selected_loss,
                "classification_loss": calibration_classification_loss,
                "weighted_classification_loss": selected_loss,
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
            # Calibration owns a selectable classification loss below while
            # retaining the VC auxiliary terms from the same logits/embedding.
            include_classification=not calibration,
        )
        # The selected classification loss and every enabled VC term are
        # combined below from this exact logits/embedding pair.
        components = dict(components)
        # Do not expose the VariationalClassifier's legacy cross-entropy
        # aliases: older code used those names for the focal value, which made
        # logs incorrectly show identical cross_entropy and focal_loss fields.
        components.pop("cross_entropy", None)
        components.pop("weighted_cross_entropy", None)
        if calibration:
            classification_weight = tf.cast(
                self.calibration_vc_alpha,
                calibration_weighted_classification_loss.dtype,
            )
            replacement_weighted_classification = (
                classification_weight
                * calibration_weighted_classification_loss
            )
            components["total_loss"] = (
                components["total_loss"]
                + replacement_weighted_classification
            )
            components["classification_loss"] = (
                calibration_classification_loss
            )
            components["weighted_classification_loss"] = (
                replacement_weighted_classification
            )
            zero = tf.zeros_like(calibration_classification_loss)
            components["focal_loss"] = (
                calibration_classification_loss
                if self.calibration_loss == "focal"
                else zero
            )
            components["weighted_focal_loss"] = (
                replacement_weighted_classification
                if self.calibration_loss == "focal"
                else zero
            )
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
        per_sample = self._per_sample_focal_loss(
            logits,
            y_flat,
            gamma=self.focal_gamma,
            alpha=self.focal_alpha,
            label_smoothing=self.label_smoothing,
        )
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
            reconstruction_components = self._reconstruction_components(
                outputs,
                training=True,
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
                + tf.cast(self.reconstruction_loss_weight, dtype)
                * tf.cast(
                    reconstruction_components["reconstruction_loss"],
                    dtype,
                )
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
            reconstruction_components=reconstruction_components,
            subject_components=subject_components,
            vrex_components=vrex_components,
        )

    def _calibration_train_step(self, x, y_flat, sample_weight):
        eeg_inputs, _ = self._split_eeg_and_subject_inputs(x)
        # Backbone layers are frozen so calibration fits the selected
        # recurrent classifier (when requested), the sole VC logits head, and
        # optionally the active reconstruction decoders.
        decoder_training = not self.calibration_freeze_decoder
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
            reconstruction_components = self._reconstruction_components(
                outputs,
                training=decoder_training,
            )
            dtype = vc_components["total_loss"].dtype
            total = (
                tf.cast(self.calibration_vc_loss_weight, dtype)
                * vc_components["total_loss"]
            )
            if decoder_training:
                total = (
                    total
                    + tf.cast(self.calibration_decoder_loss_weight, dtype)
                    * tf.cast(
                        reconstruction_components["reconstruction_loss"],
                        dtype,
                    )
                )
            total = total + self._regularization_loss(dtype)
        variables = self.trainable_variables
        gradients = tape.gradient(total, variables)
        self._apply_gradients(self.main_optimizer, gradients, variables)
        self._update_metrics(
            total_loss=total,
            vc_components=vc_components,
            outputs=outputs,
            y_flat=y_flat,
            sample_weight=sample_weight,
            reconstruction_components=reconstruction_components,
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
        reconstruction_components = self._reconstruction_components(
            outputs,
            training=False,
        )
        if self.calibration_mode:
            subject_components = None
            dtype = vc_components["total_loss"].dtype
            total = (
                tf.cast(
                    self.calibration_vc_loss_weight,
                    dtype,
                )
                * vc_components["total_loss"]
            )
            if not self.calibration_freeze_decoder:
                total = (
                    total
                    + tf.cast(self.calibration_decoder_loss_weight, dtype)
                    * tf.cast(
                        reconstruction_components["reconstruction_loss"],
                        dtype,
                    )
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
                + tf.cast(self.reconstruction_loss_weight, dtype)
                * tf.cast(
                    reconstruction_components["reconstruction_loss"],
                    dtype,
                )
                + tf.cast(self.subject_loss_weight, dtype)
                * tf.cast(subject_components["subject_loss"], dtype)
            )
        self._update_metrics(
            total_loss=total,
            vc_components=vc_components,
            outputs=outputs,
            y_flat=y_flat,
            sample_weight=sample_weight,
            reconstruction_components=reconstruction_components,
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
        calibration-only loss mode. A model loaded with ``compile=False`` also
        receives an evaluation-only optimizer so Keras ``evaluate`` can execute;
        calibration replaces it with a fresh optimizer before any fitting.
        """
        self.calibration_mode = False
        if self.main_optimizer is None:
            self.compile(
                main_optimizer=_build_optimizer(
                    optimizer_name="adam",
                    learning_rate=1e-4,
                    weight_decay=0.0,
                ),
                vc_discriminator_optimizer=None,
                run_eagerly=False,
                jit_compile=False,
            )
        return self

    def get_encoder_features(self, inputs):
        """Return both deterministic branches and their direct concatenation."""
        eeg_inputs, _ = self._split_eeg_and_subject_inputs(inputs)
        outputs = self._encode(
            eeg_inputs,
            training=False,
        )
        return {
            "gcn_gru_features": outputs["graph_sequence"],
            "bilstm_features": outputs["bilstm_sequence"],
            "combined_feature_sequence": outputs[
                "combined_feature_sequence"
            ],
            "classifier_sequence": outputs["classifier_sequence"],
            "pooled_features": outputs["pooled_features"],
            "window_features": outputs["window_features"],
            "classification_embedding": outputs["classification_embedding"],
            "probabilities": outputs["probabilities"],
        }

    def decode_branch_feature_sequence(self, branch: str, feature_sequence):
        """Decode one branch feature sequence without crossing branch inputs."""
        if not self.use_decoder:
            raise RuntimeError("Decoder is disabled for this SIC configuration.")
        branch = str(branch).strip().lower().replace("-", "_")
        if branch in {"gcn", "gcn_gru", "graph"}:
            if self.gcn_gru_decoder is None:
                raise RuntimeError("The GCN-GRU branch decoder is unavailable.")
            return self.gcn_gru_decoder(feature_sequence, training=False)
        if branch in {"bilstm", "temporal"}:
            if self.bilstm_decoder is None:
                raise RuntimeError("The BiLSTM branch decoder is unavailable.")
            return self.bilstm_decoder(feature_sequence, training=False)
        raise ValueError("branch must be 'gcn_gru' or 'bilstm'.")

    def get_adjacency_matrices(self):
        if self.graph_encoder is None:
            return {}
        return {
            "mtl_raw_adjacency": self.graph_encoder.get_raw_adjacency_matrix(),
            "mtl_normalized_adjacency": self.graph_encoder.get_adjacency_matrix(),
        }

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config["graph_encoder"] = (
            None
            if config["graph_encoder"] is None
            else _deserialize_keras_object(config["graph_encoder"])
        )
        for key in ("gcn_gru_decoder", "bilstm_decoder"):
            config[key] = (
                None
                if config.get(key) is None
                else _deserialize_keras_object(config[key])
            )
        model = cls(**config)
        # Prediction does not call the adversary. Build it before Keras restores
        # its saved weights, just as the public model builder does for training.
        if model.subject_logits_layer is not None:
            model._subject_logits(
                tf.zeros((1, model.classification_embedding_dim), dtype=tf.float32),
                training=False,
                use_grl=False,
            )
        return model

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
        # Source training follows the serialized use_class_weight setting.
        # Calibration class weighting is controlled independently by
        # cross_val and is therefore accepted whenever calibration_mode is on.
        if not self.use_class_weight and not self.calibration_mode:
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

        callbacks = self._source_callbacks(callbacks)
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

    def _source_callbacks(self, callbacks):
        return callbacks

    @property
    def metrics(self):
        output = [
            self.loss_tracker,
            self.vc_loss_tracker,
            self.classification_loss_tracker,
            self.focal_loss_tracker,
            self.accuracy_tracker,
            self.balanced_accuracy_tracker,
            self.predicted_class_1_fraction_tracker,
            self.true_class_1_fraction_tracker,
        ]
        if self.use_decoder:
            output.extend(
                [
                    self.reconstruction_loss_tracker,
                    self.decoder_r2_tracker,
                ]
            )
            if self.use_gcn_gru_branch:
                output.extend(
                    [
                        self.gcn_gru_reconstruction_loss_tracker,
                        self.gcn_gru_decoder_r2_tracker,
                    ]
                )
            if self.use_bilstm_branch:
                output.extend(
                    [
                        self.bilstm_reconstruction_loss_tracker,
                        self.bilstm_decoder_r2_tracker,
                    ]
                )
            output.extend(self._extra_decoder_metrics())
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

    def _extra_decoder_metrics(self):
        return []

    def _update_metrics(
        self,
        *,
        total_loss,
        vc_components,
        outputs,
        y_flat,
        sample_weight,
        reconstruction_components=None,
        subject_components=None,
        vrex_components=None,
    ):
        zero = tf.zeros((), dtype=total_loss.dtype)
        self.loss_tracker.update_state(total_loss)
        self.vc_loss_tracker.update_state(vc_components["total_loss"])
        self.classification_loss_tracker.update_state(
            vc_components["classification_loss"]
        )
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
        if self.use_decoder:
            reconstruction_loss = (
                zero
                if reconstruction_components is None
                else reconstruction_components["reconstruction_loss"]
            )
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)

            if self.use_gcn_gru_branch:
                graph_loss = (
                    zero
                    if reconstruction_components is None
                    else reconstruction_components["gcn_gru_reconstruction_loss"]
                )
                self.gcn_gru_reconstruction_loss_tracker.update_state(graph_loss)
                graph_reconstruction = (
                    None
                    if reconstruction_components is None
                    else reconstruction_components["gcn_gru_reconstruction"]
                )
                if graph_reconstruction is not None:
                    self.gcn_gru_decoder_r2_tracker.update_state(
                        outputs["flat_windows"],
                        graph_reconstruction,
                    )
                    self.decoder_r2_tracker.update_state(
                        outputs["flat_windows"],
                        graph_reconstruction,
                    )

            if self.use_bilstm_branch:
                bilstm_loss = (
                    zero
                    if reconstruction_components is None
                    else reconstruction_components["bilstm_reconstruction_loss"]
                )
                self.bilstm_reconstruction_loss_tracker.update_state(bilstm_loss)
                bilstm_reconstruction = (
                    None
                    if reconstruction_components is None
                    else reconstruction_components["bilstm_reconstruction"]
                )
                if bilstm_reconstruction is not None:
                    self.bilstm_decoder_r2_tracker.update_state(
                        outputs["flat_windows"],
                        bilstm_reconstruction,
                    )
                    self.decoder_r2_tracker.update_state(
                        outputs["flat_windows"],
                        bilstm_reconstruction,
                    )
            self._update_extra_decoder_metrics(outputs, reconstruction_components, zero)
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

    def _update_extra_decoder_metrics(self, outputs, reconstruction_components, zero):
        pass

    def prepare_for_subject_calibration(
        self,
        *,
        learning_rate: float,
        optimizer_name: str = "adamw",
        weight_decay: float = 0.0,
        unfreeze_layers: int | None = None,
        freeze_decoder: bool | None = None,
        decoder_loss_weight: float | None = None,
    ):
        """Freeze the backbone and adapt classification plus optional decoders.

        ``unfreeze_layers`` counts prediction layers backward from the output:
        1 means the VC logits head only; 2 means the full-sequence recurrent
        classifier plus the VC head. The VC parameters always remain
        trainable because this module is the classifier rather than a frozen
        target. ``freeze_decoder`` defaults to the serialized calibration
        setting. When false, every active decoder is trainable and its mean
        reconstruction loss is added with ``decoder_loss_weight``.
        """
        if unfreeze_layers is None:
            unfreeze_layers = self.calibration_unfreeze_layers
        unfreeze_layers = int(unfreeze_layers)
        max_layers = 2 if self.trial_recurrent_classifier is not None else 1
        if not 1 <= unfreeze_layers <= max_layers:
            raise ValueError(
                f"unfreeze_layers must be in [1, {max_layers}], got "
                f"{unfreeze_layers}."
            )
        if freeze_decoder is None:
            freeze_decoder = self.calibration_freeze_decoder
        freeze_decoder = bool(freeze_decoder)
        if decoder_loss_weight is None:
            decoder_loss_weight = self.calibration_decoder_loss_weight
        decoder_loss_weight = float(decoder_loss_weight)
        if not np.isfinite(decoder_loss_weight) or decoder_loss_weight < 0.0:
            raise ValueError(
                "decoder_loss_weight must be finite and non-negative."
            )
        if not freeze_decoder and not self.use_decoder:
            raise ValueError(
                "calibration_freeze_decoder=false requires a checkpoint with "
                "use_decoder=true."
            )
        if not freeze_decoder and not any(
            decoder is not None
            for decoder in (self.gcn_gru_decoder, self.bilstm_decoder)
        ):
            raise ValueError(
                "calibration_freeze_decoder=false requires at least one "
                "active decoder."
            )

        self.calibration_mode = True
        self.calibration_freeze_decoder = freeze_decoder
        self.calibration_decoder_loss_weight = decoder_loss_weight

        # Freeze every major source-trained subsystem first.
        if self.graph_encoder is not None:
            self.graph_encoder.trainable = False
        if self.gcn_gru_decoder is not None:
            self.gcn_gru_decoder.trainable = not freeze_decoder
        if self.bilstm_decoder is not None:
            self.bilstm_decoder.trainable = not freeze_decoder
        self._set_extra_decoder_trainable(not freeze_decoder)
        for layer in self.temporal_bilstms:
            layer.trainable = False
        for layer in self.temporal_norms:
            layer.trainable = False
        for layer in self.temporal_dropouts:
            layer.trainable = False
        if self.subject_gradient_reversal is not None:
            self.subject_gradient_reversal.trainable = False
        if self.subject_hidden is not None:
            self.subject_hidden.trainable = False
        if self.subject_dropout_layer is not None:
            self.subject_dropout_layer.trainable = False
        if self.subject_logits_layer is not None:
            self.subject_logits_layer.trainable = False

        if self.trial_recurrent_classifier is not None:
            self.trial_recurrent_classifier.trainable = False
        self.vc_target.trainable = True

        if unfreeze_layers == 2:
            self.trial_recurrent_classifier.trainable = True

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
            "calibration_loss": self.calibration_loss,
            "calibration_label_smoothing": self.calibration_label_smoothing,
            "calibration_focal_gamma": self.calibration_focal_gamma,
            "calibration_focal_alpha": self.calibration_focal_alpha,
            "calibration_vc_loss_weight": self.calibration_vc_loss_weight,
            "calibration_freeze_decoder": self.calibration_freeze_decoder,
            "calibration_decoder_loss_weight": (
                self.calibration_decoder_loss_weight
            ),
        }

    def _set_extra_decoder_trainable(self, trainable):
        pass

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "graph_encoder": (
                    None
                    if self.graph_encoder is None
                    else _serialize_keras_object(self.graph_encoder)
                ),
                "gcn_gru_decoder": (
                    None
                    if self.gcn_gru_decoder is None
                    else _serialize_keras_object(self.gcn_gru_decoder)
                ),
                "bilstm_decoder": (
                    None
                    if self.bilstm_decoder is None
                    else _serialize_keras_object(self.bilstm_decoder)
                ),
                "classification_level": self.classification_level,
                "n_classes": self.n_classes,
                "gcn_gru_feature_dim": self.gcn_gru_feature_dim,
                "bilstm_units": self.bilstm_units,
                "n_bilstm_layers": self.n_bilstm_layers,
                "bilstm_dropout": self.bilstm_dropout_rate,
                "use_gcn_gru_branch": self.use_gcn_gru_branch,
                "use_bilstm_branch": self.use_bilstm_branch,
                "use_decoder": self.use_decoder,
                "classifier_rnn_type": self.classifier_rnn_type,
                "classifier_rnn_units": self.classifier_rnn_units,
                "n_classifier_rnn_layers": self.n_classifier_rnn_layers,
                "classifier_rnn_dropout": self.classifier_rnn_dropout_rate,
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
                "reconstruction_loss_weight": self.reconstruction_loss_weight,
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
                "calibration_vc_loss_weight": self.calibration_vc_loss_weight,
                "calibration_freeze_decoder": self.calibration_freeze_decoder,
                "calibration_decoder_loss_weight": (
                    self.calibration_decoder_loss_weight
                ),
                "calibration_loss": self.calibration_loss,
                "calibration_label_smoothing": (
                    self.calibration_label_smoothing
                ),
                "calibration_focal_gamma": self.calibration_focal_gamma,
                "calibration_focal_alpha": self.calibration_focal_alpha,
                "use_class_weight": self.use_class_weight,
            }
        )
        return config

    def _reconstruction_components(self, outputs, training: bool):
        """Reconstruct EEG independently from each active encoder branch.

        The GCN-GRU decoder consumes only ``graph_sequence`` and the BiLSTM
        decoder consumes only ``bilstm_sequence``. Their losses are averaged,
        rather than summed, so toggling a branch does not silently change the
        scale controlled by ``reconstruction_loss_weight``.
        """
        dtype = outputs["combined_feature_sequence"].dtype
        zero = tf.zeros((), dtype=dtype)
        components = {
            "reconstruction_loss": zero,
            "gcn_gru_reconstruction_loss": zero,
            "bilstm_reconstruction_loss": zero,
            "gcn_gru_reconstruction": None,
            "bilstm_reconstruction": None,
        }
        if not self.use_decoder:
            return components

        target = tf.cast(outputs["flat_windows"], dtype)
        branch_losses = []

        if self.use_gcn_gru_branch:
            reconstruction = self.gcn_gru_decoder(
                outputs["graph_sequence"],
                training=bool(training),
            )
            tf.debugging.assert_equal(
                tf.shape(reconstruction),
                tf.shape(target),
                message=(
                    "GCN-GRU decoder output must match the original flattened "
                    "EEG windows."
                ),
            )
            branch_loss = tf.reduce_mean(tf.square(target - reconstruction))
            components["gcn_gru_reconstruction"] = reconstruction
            components["gcn_gru_reconstruction_loss"] = branch_loss
            branch_losses.append(branch_loss)

        if self.use_bilstm_branch:
            reconstruction = self.bilstm_decoder(
                outputs["bilstm_sequence"],
                training=bool(training),
            )
            tf.debugging.assert_equal(
                tf.shape(reconstruction),
                tf.shape(target),
                message=(
                    "BiLSTM decoder output must match the original flattened "
                    "EEG windows."
                ),
            )
            branch_loss = tf.reduce_mean(tf.square(target - reconstruction))
            components["bilstm_reconstruction"] = reconstruction
            components["bilstm_reconstruction_loss"] = branch_loss
            branch_losses.append(branch_loss)

        if not branch_losses:
            raise RuntimeError("Decoder is enabled but no active branch can be decoded.")
        components["reconstruction_loss"] = tf.add_n(branch_losses) / tf.cast(
            len(branch_losses),
            dtype,
        )
        return components

    def _build_sequence_heads(self, *, vc_logit_scale=1.0):
        # MTL encoder already performs GCN + spectral GRU. This recurrent
        # stack is exclusively temporal and preserves all original timesteps.
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

        # The reusable GRU/BiGRU consumes the entire encoded temporal sequence.
        # Trial inputs are reshaped from (windows, timesteps, features) to
        # (windows * timesteps, features) before reaching this summarizer.
        if self.classifier_rnn_type == "bigru":
            rnn_builder = BiGRUClassifier(
                timesteps=None,
                n_features=self.combined_feature_dim,
                n_classes=self.n_classes,
                gru_units=self.classifier_rnn_units,
                n_bigru_layers=self.n_classifier_rnn_layers,
                dropout=self.classifier_rnn_dropout_rate,
                name="v6_full_sequence_bigru",
            )
            self.classification_embedding_dim = (
                2 * self.classifier_rnn_units[-1]
            )
        else:
            rnn_builder = GRUClassifier(
                timesteps=None,
                n_features=self.combined_feature_dim,
                n_classes=self.n_classes,
                gru_units=self.classifier_rnn_units,
                n_gru_layers=self.n_classifier_rnn_layers,
                dropout=self.classifier_rnn_dropout_rate,
                name="v6_full_sequence_gru",
            )
            self.classification_embedding_dim = self.classifier_rnn_units[-1]
        self.trial_recurrent_classifier = rnn_builder.build_sequence_summarizer()

        vc_dim = self.classification_embedding_dim
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
            logit_scale=vc_logit_scale,
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
