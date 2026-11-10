"""Idea 1: a frozen real-EEG identifier guides decoder-only optimization."""
import numpy as np
import tensorflow as tf

from .data import encode_subjects


def build_identifier(training_eeg, n_subjects, *, filters=32):
    """Fit normalization on source training EEG only; return subject logits."""
    if n_subjects < 2:
        raise ValueError("Subject identification requires at least two source subjects.")
    shape = tuple(training_eeg.shape[1:])
    normalizer = tf.keras.layers.Normalization(axis=-1)
    # Flatten trial/window/time axes consistently for channelwise statistics.
    normalizer.adapt(np.reshape(training_eeg, [-1, shape[-1]]))
    inputs = tf.keras.Input(shape=shape)
    x = normalizer(inputs)
    x = tf.keras.layers.Reshape((-1, shape[-1]))(x)
    x = tf.keras.layers.Conv1D(filters, 9, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv1D(filters, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    logits = tf.keras.layers.Dense(n_subjects)(x)
    return tf.keras.Model(inputs, logits, name="real_eeg_subject_identifier")


def decoder_layers(model):
    layers = [getattr(model, name, None) for name in
              ("gcn_gru_decoder", "bilstm_decoder", "joint_reconstruction_fusion")]
    layers = [layer for layer in layers if layer is not None]
    if not getattr(model, "use_decoder", False) or not layers:
        raise ValueError("A SIC model with trained decoders is required.")
    return layers


def decoder_variables(model):
    result, seen = [], set()
    for layer in decoder_layers(model):
        for variable in layer.trainable_variables:
            if id(variable) not in seen:
                result.append(variable)
                seen.add(id(variable))
    return result


def selected_reconstruction(components, input_shape):
    for key in ("joint_reconstruction", "gcn_gru_reconstruction", "bilstm_reconstruction"):
        if components.get(key) is not None:
            return tf.reshape(components[key], input_shape)
    raise ValueError("No active reconstruction path.")


def subject_loss(identifier, real, reconstructed, targets, *, mode):
    logits = identifier(reconstructed, training=False)
    if mode == "cross_entropy":
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=targets, logits=logits))
    if mode == "kl":
        real_logp = tf.stop_gradient(tf.nn.log_softmax(identifier(real, training=False)))
        decoded_logp = tf.nn.log_softmax(logits)
        return tf.reduce_mean(tf.reduce_sum(
            tf.exp(real_logp) * (real_logp - decoded_logp), axis=-1))
    raise ValueError("identity loss must be 'cross_entropy' or 'kl'.")


class DecoderIdentityTrainer:
    """Update only branch decoders/fusion; encoder, VC and identifier stay fixed.

    Call on an experiment-specific model instance: its trainability flags are
    changed deliberately. A frozen identifier remains differentiable in EEG.
    """
    def __init__(self, model, identifier, *, identity_weight=0.1,
                 reconstruction_weight=1.0, loss_mode="cross_entropy",
                 learning_rate=1e-4):
        for name, value in (("identity_weight", identity_weight),
                            ("reconstruction_weight", reconstruction_weight)):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if identity_weight == reconstruction_weight == 0:
            raise ValueError("At least one decoder objective must be active.")
        if loss_mode not in {"cross_entropy", "kl"}:
            raise ValueError("Unknown identity loss mode.")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        self.model, self.identifier = model, identifier
        self.identity_weight = float(identity_weight)
        self.reconstruction_weight = float(reconstruction_weight)
        self.loss_mode = loss_mode
        decoder_layers(model)
        model.trainable = False
        identifier.trainable = False
        for layer in decoder_layers(model):
            layer.trainable = True
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def objective(self, eeg, targets, *, training):
        eeg = tf.convert_to_tensor(eeg, dtype=tf.float32)
        # Inference-mode encoders plus stop_gradient also protect moving state
        # and avoid an identity-preserving gradient reaching the ISA network.
        outputs = self.model._encode(eeg, training=False)
        outputs = {key: tf.stop_gradient(value) if tf.is_tensor(value) else value
                   for key, value in outputs.items()}
        components = self.model._reconstruction_components(outputs, training=training)
        reconstructed = selected_reconstruction(components, tf.shape(eeg))
        reconstruction = components["reconstruction_loss"]
        identity = subject_loss(self.identifier, eeg, reconstructed,
                                targets, mode=self.loss_mode)
        total = self.reconstruction_weight * reconstruction + self.identity_weight * identity
        tf.debugging.assert_all_finite(total, "Nonfinite decoder objective.")
        return total, {"loss": total, "reconstruction_loss": reconstruction,
                       "identity_loss": identity}

    def step(self, eeg, targets):
        with tf.GradientTape() as tape:
            total, metrics = self.objective(eeg, targets, training=True)
        variables = decoder_variables(self.model)
        gradients = tape.gradient(total, variables)
        pairs = [(gradient, variable) for gradient, variable in zip(gradients, variables)
                 if gradient is not None]
        if not pairs:
            raise RuntimeError("No decoder gradient reached the active objective.")
        for gradient, _ in pairs:
            tf.debugging.assert_all_finite(gradient, "Nonfinite decoder gradient.")
        self.optimizer.apply_gradients(pairs)
        return {key: float(value.numpy()) for key, value in metrics.items()}


def reconstruction_metrics(model, data, *, identifier=None, classes=None,
                           reconstruction_fn=None, batch_size=8):
    """Round-trip task agreement/BA and optional *known-source* ID accuracy."""
    probabilities, decoded_probabilities, mse = [], [], []
    id_real, id_decoded, id_targets = [], [], []
    for start in range(0, len(data.eeg), batch_size):
        eeg = tf.constant(data.eeg[start:start + batch_size])
        if reconstruction_fn is None:
            outputs = model._encode(eeg, training=False)
            components = model._reconstruction_components(outputs, training=False)
            decoded = selected_reconstruction(components, tf.shape(eeg))
        else:
            decoded = reconstruction_fn(eeg)
        probabilities.extend(tf.nn.softmax(model(eeg, training=False)).numpy())
        decoded_probabilities.extend(tf.nn.softmax(model(decoded, training=False)).numpy())
        mse.extend(tf.reduce_mean(tf.square(decoded - eeg), axis=tuple(range(1, eeg.shape.rank))).numpy())
        if identifier is not None:
            id_real.extend(tf.argmax(identifier(eeg, training=False), -1).numpy())
            id_decoded.extend(tf.argmax(identifier(decoded, training=False), -1).numpy())
            id_targets.extend(encode_subjects(data.subject_ids[start:start + batch_size],
                                              classes, allow_unknown=True))
    original, reconstructed = np.argmax(probabilities, -1), np.argmax(decoded_probabilities, -1)
    n_classes = len(probabilities[0])
    support = {str(c): int(np.sum(data.labels == c)) for c in range(n_classes)}
    def balanced_accuracy(predicted):
        if any(count == 0 for count in support.values()):
            return None  # A single observed class does not establish binary BA.
        recalls = [np.mean(predicted[data.labels == c] == c) for c in range(n_classes)]
        return float(np.mean(recalls))
    report = {"n_examples": len(original), "reconstruction_mse": float(np.mean(mse)),
              "round_trip_agreement": float(np.mean(original == reconstructed)),
              "round_trip_flip_rate": float(np.mean(original != reconstructed)),
              "task_class_support": support,
              "original_balanced_accuracy": balanced_accuracy(original),
              "reencoded_balanced_accuracy": balanced_accuracy(reconstructed)}
    if identifier is not None:
        targets = np.asarray(id_targets)
        known = targets >= 0
        report.update(known_subject_examples=int(known.sum()),
                      unknown_subject_examples=int((~known).sum()),
                      real_subject_accuracy=(float(np.mean(np.asarray(id_real)[known] == targets[known]))
                                             if known.any() else None),
                      decoded_subject_accuracy=(float(np.mean(np.asarray(id_decoded)[known] == targets[known]))
                                                if known.any() else None))
    return report
