"""Gradient isolation, paired-coordinate decoding and runnable experiment checks."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
# Experiments deliberately live outside the installable library package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.decoder_recovery.data import TrialData, prepare_fold, validate_splits
from experiments.decoder_recovery.identity import (DecoderIdentityTrainer, build_identifier,
                                                   decoder_variables, reconstruction_metrics)
from experiments.decoder_recovery.latent_coords import LatentCoordinatesDecoder, LatentCoordsOptimizer
from experiments.decoder_recovery.training import (load_sic, train_identifier, train_idea1,
                                                   train_latent_coords)
from eegproc.deep_learning.joint_architectures.SICModelv15.sic_model import build_sic_model
from eegproc.model_explainability.counterfactuals.loss import CounterfactualLoss


def config(both=False):
    return dict(n_channels=14, n_bands=3, adjacency=np.eye(14).tolist(),
        use_gcn_gru_branch=both, use_bilstm_branch=True,
        gcn_units=(3, 2), spectral_gru_units=3, bilstm_units=2,
        classifier_rnn_units=3, n_subject_classes=2,
        gcn_dropout=0.0, spectral_gru_dropout=0.0, bilstm_dropout=0.0,
        classifier_rnn_dropout=0.0, decoder_dropout=0.0,
        vc_beta=0.0, vc_gamma=0.0, vc_lambda=0.0, weight_decay=0.0)


def model(both=False):
    return build_sic_model((1, 4, 42), **config(both))


def data(subjects=(0, 0, 1, 1), trial_offset=0):
    eeg = np.random.default_rng(23 + trial_offset).normal(size=(len(subjects), 1, 4, 42)).astype("float32")
    return TrialData(eeg, np.arange(len(subjects)) % 2, np.asarray(subjects),
                     np.arange(len(subjects)) + trial_offset)


@pytest.mark.parametrize("mode", ["cross_entropy", "kl"])
def test_subject_loss_updates_decoder_only_and_checkpoint_round_trip(tmp_path, mode):
    tf.keras.utils.set_random_seed(72)
    sample, source = data(), model()
    identifier = build_identifier(sample.eeg, 2, filters=3)
    original_logits = source(sample.eeg, training=False).numpy()
    all_before = {id(v): v.numpy().copy() for v in source.weights}
    id_before = [v.numpy().copy() for v in identifier.weights]
    trainer = DecoderIdentityTrainer(source, identifier, reconstruction_weight=0.0,
                                     identity_weight=1.0, loss_mode=mode)
    with tf.GradientTape(persistent=True) as tape:
        loss, _ = trainer.objective(sample.eeg, sample.subject_ids.astype("int32"), training=False)
    decoder = decoder_variables(source)
    gradients = tape.gradient(loss, decoder)
    assert any(g is not None and np.any(g.numpy() != 0) for g in gradients)
    # Explicitly watching frozen encoder variables confirms the stop_gradient,
    # independently of the optimizer's restricted parameter list.
    encoder_var = source.temporal_bilstms[0].weights[0]
    with tf.GradientTape() as encoder_tape:
        encoder_tape.watch(encoder_var.value)
        encoder_loss, _ = trainer.objective(sample.eeg, sample.subject_ids.astype("int32"), training=False)
    assert encoder_tape.gradient(encoder_loss, encoder_var.value) is None
    trainer.step(sample.eeg, sample.subject_ids.astype("int32"))
    decoder_ids = {id(v) for v in decoder}
    assert any(not np.array_equal(all_before[id(v)], v.numpy()) for v in decoder)
    for variable in source.weights:
        if id(variable) not in decoder_ids:
            np.testing.assert_array_equal(variable.numpy(), all_before[id(variable)])
    for old, variable in zip(id_before, identifier.weights):
        np.testing.assert_array_equal(old, variable.numpy())
    np.testing.assert_array_equal(original_logits, source(sample.eeg, training=False).numpy())
    path = tmp_path / "decoder.keras"
    source.save(path)
    restored = load_sic(path)
    np.testing.assert_allclose(restored.reconstruct(sample.eeg, "bilstm"),
                               source.reconstruct(sample.eeg, "bilstm"), atol=1e-6)


@pytest.mark.parametrize("both", [False, True])
def test_latent_coords_restore_pre_isa_and_keep_candidate_gradient(both):
    tf.keras.utils.set_random_seed(43)
    original, invariant = model(both), model(both)
    invariant.set_weights(original.get_weights())
    # Simulate a genuine encoder-coordinate change after the shared initialization.
    invariant.temporal_bilstms[0].weights[0].assign_add(
        tf.ones_like(invariant.temporal_bilstms[0].weights[0]) * 0.1)
    eeg = tf.constant(data().eeg[:1])
    coordinates = LatentCoordinatesDecoder(original, invariant)
    residual = coordinates.capture(eeg)
    assert np.any(residual["bilstm"].numpy() != 0)
    expected = original.reconstruct(eeg, "joint" if both else "bilstm")
    np.testing.assert_allclose(coordinates.reconstruct(eeg), expected, atol=2e-6)
    after = invariant.get_encoder_features(eeg)["bilstm_features"]
    candidate = tf.Variable(after)
    weights = [v.numpy().copy() for v in original.weights + invariant.weights]
    with tf.GradientTape() as tape:
        decoded = coordinates.decode_branch("bilstm", candidate, residual)
        loss = tf.reduce_sum(tf.square(decoded))
    gradient = tape.gradient(loss, candidate)
    assert gradient is not None and np.any(gradient.numpy() != 0)
    for old, current in zip(weights, original.weights + invariant.weights):
        np.testing.assert_array_equal(old, current.numpy())
    # A new trial requires a new residual, not an averaged/global difference.
    second = coordinates.capture(tf.constant(data().eeg[1:2]))
    assert not np.array_equal(second["bilstm"], residual["bilstm"])


def test_latent_coords_counterfactual_uses_post_isa_classifier_and_clears_state():
    original, invariant = model(), model()
    invariant.set_weights(original.get_weights())
    invariant.temporal_bilstms[0].weights[0].assign_add(
        tf.ones_like(invariant.temporal_bilstms[0].weights[0]) * 0.1)
    optimizer = LatentCoordsOptimizer(invariant, reference_model=original,
        loss=CounterfactualLoss(physiological_weight=0), max_steps=0, decoder_mode="branches")
    eeg = data().eeg[:1]
    before = [v.numpy().copy() for v in original.weights + invariant.weights]
    result = optimizer.optimize(eeg)
    assert optimizer._residuals is None
    arrays = result["arrays"]
    assert "coordinate_residual_bilstm" in arrays
    np.testing.assert_allclose(arrays["x_reconstructed_bilstm"],
                               original.reconstruct(eeg, "bilstm"), atol=2e-6)
    prediction = result["summary"]["original"]["probabilities"]
    np.testing.assert_allclose(prediction, tf.nn.softmax(invariant(eeg, training=False))[0], atol=1e-6)
    for old, current in zip(before, original.weights + invariant.weights):
        np.testing.assert_array_equal(old, current.numpy())


def test_data_split_protects_complete_trials_and_unknown_subject_metrics():
    sample = data()
    with pytest.raises(ValueError, match="subject/trial"):
        validate_splits(sample, sample)
    identifier = build_identifier(sample.eeg, 2, filters=3)
    unknown = data((99, 99))
    report = reconstruction_metrics(model(), unknown, identifier=identifier, classes=[0, 1])
    assert report["known_subject_examples"] == 0
    assert report["decoded_subject_accuracy"] is None
    assert report["unknown_subject_examples"] == 2


def test_identifier_and_idea1_training_end_to_end(tmp_path):
    train, validation = data(), data(trial_offset=10)
    identifier = train_identifier(train, validation, tmp_path / "identifier", epochs=1, batch_size=2)
    restored_identifier = tf.keras.models.load_model(tmp_path / "identifier" / "identifier.keras", compile=False)
    np.testing.assert_allclose(identifier(train.eeg), restored_identifier(train.eeg), atol=1e-6)
    train_idea1(model(), restored_identifier, [0, 1], train, validation,
                 tmp_path / "idea1", epochs=1, batch_size=2)
    report = json.loads((tmp_path / "idea1" / "report.json").read_text())
    assert report["encoder_and_classifier_unchanged"]
    assert report["identifier_unchanged"]
    assert "round_trip_agreement" in report["after"]


def test_two_phase_pipeline_saves_paired_models_and_recovery_report(tmp_path):
    train, validation = data(), data((2, 2), trial_offset=20)
    reference, invariant = train_latent_coords(train, validation, config(), tmp_path / "coords",
        phase1_epochs=1, phase2_epochs=1, batch_size=2)
    assert reference.subject_loss_weight == 0
    assert invariant.subject_loss_weight > 0
    assert (tmp_path / "coords" / "pre_isa.keras").exists()
    restored = load_sic(tmp_path / "coords" / "post_isa.keras")
    restored_reference = load_sic(tmp_path / "coords" / "pre_isa.keras")
    np.testing.assert_allclose(restored(validation.eeg), invariant(validation.eeg), atol=1e-6)
    np.testing.assert_allclose(LatentCoordinatesDecoder(restored_reference, restored).reconstruct(validation.eeg),
                               reference.reconstruct(validation.eeg, "bilstm"), atol=2e-6)
    report = json.loads((tmp_path / "coords" / "report.json").read_text())
    assert not report["residual_is_subject_only"]
    assert "round_trip_flip_rate" in report["latent_coords_decoder"]


def test_temporal_losses_survive_both_recovery_methods_and_checkpoint_reload(tmp_path):
    train, validation = data(), data((2, 2), trial_offset=20)
    for split in (train, validation):
        # Two chronological windows exercise actual seams, not the one-window
        # zero-seam fallback used by the existing recovery smoke tests.
        split.eeg = np.concatenate((split.eeg, -split.eeg + 0.15), axis=1)
    temporal_config = {**config(both=True),
                       "reconstruction_seam_weight": 0.7,
                       "reconstruction_derivative_weight": 0.3}
    reference, invariant = train_latent_coords(
        train, validation, temporal_config, tmp_path / "coords",
        phase1_epochs=1, phase2_epochs=1, batch_size=2)
    logits = invariant(validation.eeg, training=False).numpy()
    identifier = build_identifier(train.eeg, 2, filters=3)
    train_idea1(invariant, identifier, [0, 1], train, validation,
                tmp_path / "identity", epochs=1, batch_size=2, loss_mode="kl")
    restored = load_sic(tmp_path / "identity" / "decoder_identity.keras")
    assert restored.reconstruction_seam_weight == 0.7
    assert restored.reconstruction_derivative_weight == 0.3
    np.testing.assert_array_equal(restored(validation.eeg, training=False), logits)
    components = restored._reconstruction_components(
        restored._encode(validation.eeg, training=False), training=False)
    assert float(components["reconstruction_seam_loss"]) > 0.0
    assert float(components["reconstruction_derivative_loss"]) > 0.0
    paired_reference = load_sic(tmp_path / "coords" / "pre_isa.keras")
    np.testing.assert_allclose(
        LatentCoordinatesDecoder(paired_reference, restored).reconstruct(validation.eeg),
        reference.reconstruct(validation.eeg, "joint"), atol=2e-6)
    report = json.loads((tmp_path / "identity" / "report.json").read_text())
    assert report["encoder_and_classifier_unchanged"]
    assert report["identifier_unchanged"]


def test_fold_export_reserves_subjects_and_complete_trials(tmp_path):
    sample = data(tuple(np.repeat(np.arange(4), 3)))
    prepare_fold(sample, tmp_path / "fold", test_subjects=[3], validation_subjects=[2])
    splits = {name: TrialData.load(tmp_path / "fold" / f"{name}.npz")
              for name in ("train", "identity_validation", "validation", "test")}
    assert set(splits["train"].subject_ids) == {0, 1}
    assert set(splits["validation"].subject_ids) == {2}
    assert set(splits["test"].subject_ids) == {3}
    assert len(splits["identity_validation"].eeg) == 2
    for name, split in splits.items():
        for other, other_split in splits.items():
            if name != other:
                assert not split.keys() & other_split.keys()


def test_cli_evaluate_and_counterfactual_record_actual_reencoded_prediction(tmp_path):
    from experiments.decoder_recovery.cli import main
    source, sample = model(), data()
    checkpoint, archive = tmp_path / "model.keras", tmp_path / "data.npz"
    source.save(checkpoint)
    np.savez_compressed(archive, eeg=sample.eeg, labels=sample.labels,
                        subject_ids=sample.subject_ids, trial_ids=sample.trial_ids)
    main(["evaluate", "--model", str(checkpoint), "--reference-model", str(checkpoint),
          "--data", str(archive), "--output", str(tmp_path / "evaluation")])
    assert (tmp_path / "evaluation" / "command.json").exists()
    main(["counterfactual", "--model", str(checkpoint), "--reference-model", str(checkpoint),
          "--data", str(archive), "--output", str(tmp_path / "cf"),
          "--decoder-mode", "branches", "--max-steps", "1", "--limit", "1"])
    report = json.loads((tmp_path / "cf" / "trial_00000.json").read_text())
    assert report["counterfactual_validity_prediction_space"] == "decoded_reencoded"
    assert "probabilities" in report["decoded_reencoded"]["bilstm"]["x_prime"]
    with np.load(tmp_path / "cf" / "trial_00000.npz") as arrays:
        actual = tf.nn.softmax(source(arrays["x_prime_bilstm"], training=False))[0]
    np.testing.assert_allclose(report["decoded_reencoded"]["bilstm"]["x_prime"]["probabilities"],
                               actual, atol=1e-6)


def test_slurm_preview_has_both_methods_and_matched_control_without_writes(tmp_path):
    root = Path(__file__).resolve().parents[2]
    script = root / "experiments" / "decoder_recovery" / "run_decoder_recovery.slurm"
    subprocess.run(["bash", "-n", str(script)], check=True)
    env = {**os.environ, "DRY_RUN": "1", "PROJECT_DIR": str(root),
           "VENV_DIR": str(tmp_path / "missing_venv"), "OUT_ROOT": str(tmp_path / "results"),
           "SLURM_ARRAY_TASK_ID": "0"}
    completed = subprocess.run(["bash", str(script)], env=env, text=True,
                               capture_output=True, check=True)
    assert " latent-coords " in completed.stdout
    assert "--identity-weight 0 " in completed.stdout
    assert "--identity-weight 0.1 " in completed.stdout
    assert "independent_probe" in completed.stdout
    assert "--reference-model" in completed.stdout
    assert not (tmp_path / "results").exists()
