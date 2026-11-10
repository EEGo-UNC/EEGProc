"""Training orchestration; opt-in experiments leave production runners alone."""
import json
import zipfile

import numpy as np
import tensorflow as tf

from eegproc.deep_learning.joint_architectures.SICModelv15.sic_model import build_sic_model
from .data import (encode_subjects, new_output, validate_splits,
                   vocabulary, write_json)
from .identity import (DecoderIdentityTrainer, build_identifier,
                       reconstruction_metrics)
from .latent_coords import LatentCoordinatesDecoder


def load_sic(path):
    # Build via public config/weight APIs so paired SIC constructors (which
    # create Bidirectional layers) do not share the archive loader's temporary
    # object scope. Restore neither an optimizer nor its training state.
    with zipfile.ZipFile(path) as archive:
        configuration = json.loads(archive.read("config.json"))
    configuration.pop("compile_config", None)
    model = tf.keras.models.model_from_json(json.dumps(configuration))
    model.load_weights(path)
    return model


def _weights(model):
    return [np.array(v, copy=True) for v in model.get_weights()]


def _assert_unchanged(before, model, label):
    after = model.get_weights()
    if len(before) != len(after) or any(not np.array_equal(a, b)
                                      for a, b in zip(before, after)):
        raise RuntimeError(f"Frozen {label} changed during decoder-only training.")


def _average(rows):
    n = sum(count for count, _ in rows)
    return {key: sum(count * values[key] for count, values in rows) / n
            for key in rows[0][1]}


def train_identifier(train, validation, output, *, epochs=30, batch_size=8,
                     learning_rate=1e-3, seed=42):
    validate_splits(train, validation)
    classes = vocabulary(train.subject_ids)
    targets = encode_subjects(train.subject_ids, classes)
    validation_targets = encode_subjects(validation.subject_ids, classes)
    output = new_output(output)
    tf.keras.utils.set_random_seed(seed)
    model = build_identifier(train.eeg, len(classes))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    # train_on_batch avoids a new data-loader threadpool per short EEG fold.
    best, best_weights, history = float("inf"), None, []
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        model.reset_metrics()
        order = rng.permutation(len(train.eeg))
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            model.train_on_batch(train.eeg[index], targets[index])
        losses, correct = [], []
        for start in range(0, len(validation.eeg), batch_size):
            logits = model(validation.eeg[start:start + batch_size], training=False)
            y = validation_targets[start:start + batch_size]
            losses.extend(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits).numpy())
            correct.extend(tf.argmax(logits, -1).numpy() == y)
        row = {"epoch": epoch + 1, "validation_loss": float(np.mean(losses)),
               "validation_accuracy": float(np.mean(correct))}
        history.append(row)
        if not np.isfinite(row["validation_loss"]):
            raise FloatingPointError("Nonfinite subject-identifier validation loss.")
        if row["validation_loss"] < best:
            best, best_weights = row["validation_loss"], _weights(model)
        print(json.dumps(row), flush=True)
    model.set_weights(best_weights)
    model.save(output / "identifier.keras")
    write_json(output / "identifier.json", {
        "subject_ids": classes, "input_shape": list(train.eeg.shape[1:]),
        "outputs": "logits", "normalization": "source_training_only",
        "train_fingerprint": train.fingerprint(),
        "validation_fingerprint": validation.fingerprint(), "seed": seed})
    write_json(output / "history.json", history)
    return model


def train_idea1(model, identifier, classes, train, validation, output, *,
                epochs=30, batch_size=8, learning_rate=1e-4,
                identity_weight=0.1, reconstruction_weight=1.0,
                loss_mode="cross_entropy", seed=42):
    validate_splits(train, validation)
    tf.keras.utils.set_random_seed(seed)
    targets = encode_subjects(train.subject_ids, classes)
    val_targets = encode_subjects(validation.subject_ids, classes,
                                  allow_unknown=(loss_mode == "kl"))
    output = new_output(output)
    before = reconstruction_metrics(model, validation, identifier=identifier,
                                    classes=classes, batch_size=batch_size)
    # Only decoder/fusion weights may differ, including nontrainable state.
    decoder_ids = {id(v) for name in ("gcn_gru_decoder", "bilstm_decoder", "joint_reconstruction_fusion")
                   for v in getattr(getattr(model, name, None), "weights", ())}
    protected = [(v, np.array(v.numpy(), copy=True)) for v in model.weights if id(v) not in decoder_ids]
    identifier_before = _weights(identifier)
    trainer = DecoderIdentityTrainer(model, identifier,
        identity_weight=identity_weight, reconstruction_weight=reconstruction_weight,
        loss_mode=loss_mode, learning_rate=learning_rate)
    rng = np.random.default_rng(seed)
    best, best_weights, history = float("inf"), None, []
    for epoch in range(epochs):
        order = rng.permutation(len(train.eeg))
        rows = []
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            rows.append((len(index), trainer.step(train.eeg[index], targets[index])))
        val_rows = []
        for start in range(0, len(validation.eeg), batch_size):
            eeg = validation.eeg[start:start + batch_size]
            _, metrics = trainer.objective(eeg, val_targets[start:start + batch_size], training=False)
            val_rows.append((len(eeg), {k: float(v.numpy()) for k, v in metrics.items()}))
        row = {"epoch": epoch + 1, **_average(rows),
               **{f"validation_{k}": v for k, v in _average(val_rows).items()}}
        history.append(row)
        if row["validation_loss"] < best:
            best, best_weights = row["validation_loss"], _weights(model)
        print(json.dumps(row), flush=True)
    model.set_weights(best_weights)
    for variable, original in protected:
        if not np.array_equal(variable.numpy(), original):
            raise RuntimeError(f"Non-decoder parameter changed: {variable.name}")
    _assert_unchanged(identifier_before, identifier, "identifier")
    after = reconstruction_metrics(model, validation, identifier=identifier,
                                   classes=classes, batch_size=batch_size)
    model.save(output / "decoder_identity.keras")
    write_json(output / "history.json", history)
    write_json(output / "report.json", {"method": "idea1", "identity_loss": loss_mode,
        "identity_weight": identity_weight, "reconstruction_weight": reconstruction_weight,
        "encoder_and_classifier_unchanged": True, "identifier_unchanged": True,
        "before": before, "after": after, "seed": seed,
        "train_fingerprint": train.fingerprint(), "validation_fingerprint": validation.fingerprint()})
    return model


def _fit_source(model, train, validation, *, epochs, batch_size):
    model.set_source_training_metadata(train.subject_ids, train.trial_ids)
    x = model.prepare_fit_inputs(train.eeg, train.subject_ids)
    # Validation does not train an adversary on validation identities.
    monitor = f"val_{model.balanced_accuracy_tracker.name}"
    history = model.fit(x, train.labels, validation_data=(validation.eeg, validation.labels),
        epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.EarlyStopping(monitor=monitor, mode="max",
                patience=epochs, restore_best_weights=True)])
    values = {key: [float(v) for v in series] for key, series in history.history.items()}
    if monitor not in values:
        raise RuntimeError(f"Missing source checkpoint-selection metric: {monitor}")
    if any(not np.isfinite(v) for series in values.values() for v in series):
        raise FloatingPointError("Nonfinite source-training history.")
    return values


def train_latent_coords(train, validation, config, output, *, phase1_epochs=30,
                        phase2_epochs=30, batch_size=8, isa_weight=0.8, seed=42):
    validate_splits(train, validation)
    if set(map(str, train.subject_ids)) & set(map(str, validation.subject_ids)):
        raise ValueError("Use held-out validation SUBJECTS for the two-phase source model.")
    config = dict(config)
    if not config.get("use_decoder", True):
        raise ValueError("Latent coords requires reconstruction training in phase 1.")
    if config.get("reconstruction_loss_weight", 0.1) <= 0:
        raise ValueError("Phase 1 must train its decoder with positive reconstruction weight.")
    classes = vocabulary(train.subject_ids)
    if len(classes) < 2:
        raise ValueError("ISA requires at least two source training subjects.")
    reserved = {"input_shape", "training_features", "training_labels",
                "training_subject_ids", "training_trial_ids"}
    if reserved & config.keys():
        raise ValueError(f"Model config must not override data-derived keys: {reserved & config.keys()}")
    config.update(use_decoder=True, use_subject_adversarial=True,
                  n_subject_classes=len(classes), subject_adversarial_weight=isa_weight)
    original_subject_loss_weight = float(config.get("subject_loss_weight", 1.0))
    if original_subject_loss_weight <= 0:
        raise ValueError("Phase 2 requires positive subject_loss_weight.")
    config["subject_loss_weight"] = 0.0
    output = new_output(output)
    tf.keras.utils.set_random_seed(seed)
    builder_args = dict(training_features=train.eeg, training_labels=train.labels,
                        training_subject_ids=train.subject_ids, training_trial_ids=train.trial_ids)
    reference = build_sic_model(train.eeg.shape[1:], **builder_args, **config)
    history0 = _fit_source(reference, train, validation, epochs=phase1_epochs,
                           batch_size=batch_size)
    reference.save(output / "pre_isa.keras")
    reference_weights = _weights(reference)
    config["subject_loss_weight"] = original_subject_loss_weight
    invariant = build_sic_model(train.eeg.shape[1:], **builder_args, **config)
    invariant.set_weights(reference_weights)
    history1 = _fit_source(invariant, train, validation, epochs=phase2_epochs,
                           batch_size=batch_size)
    invariant.save(output / "post_isa.keras")
    _assert_unchanged(reference_weights, reference, "pre-ISA reference")
    coordinates = LatentCoordinatesDecoder(reference, invariant)
    report = {
        "method": "latent_coords", "residual": "E0(X)-E1(X)",
        "residual_is_subject_only": False,
        "coordinate_alignment": "raw_paired_initialization",
        "decoder_reference": "pre_isa.keras", "classifier_reference": "post_isa.keras",
        "phase1_subject_loss_weight": 0.0,
        "phase2_subject_loss_weight": original_subject_loss_weight,
        "phase2_grl_weight": isa_weight, "model_config": config,
        "train_fingerprint": train.fingerprint(), "validation_fingerprint": validation.fingerprint(),
        "source_subject_ids": classes, "seed": seed,
        "post_isa_decoder": reconstruction_metrics(invariant, validation, batch_size=batch_size),
        "latent_coords_decoder": reconstruction_metrics(invariant, validation,
            reconstruction_fn=coordinates.reconstruct, batch_size=batch_size),
    }
    write_json(output / "phase1_history.json", history0)
    write_json(output / "phase2_history.json", history1)
    write_json(output / "report.json", report)
    return reference, invariant
