"""Seam and derivative reconstruction terms used to train the SIC decoder."""
import numpy as np
import pytest
import tensorflow as tf

from eegproc.deep_learning.joint_architectures.sic_common.model import (
    temporal_reconstruction_losses,
)

TRIALS, WINDOWS, STEPS, FEATURES = 3, 4, 5, 2


def _flat(array):
    return tf.constant(array.reshape(-1, STEPS, FEATURES), tf.float32)


def _reference(target, reconstruction):
    """NumPy version of both terms on (trials, windows, steps, features)."""
    derivative = np.mean(
        (np.diff(reconstruction, axis=2) - np.diff(target, axis=2)) ** 2
    )
    seam_r = reconstruction[:, 1:, 0, :] - reconstruction[:, :-1, -1, :]
    seam_x = target[:, 1:, 0, :] - target[:, :-1, -1, :]
    return np.mean((seam_r - seam_x) ** 2), derivative


@pytest.fixture
def signals():
    rng = np.random.default_rng(7)
    target = rng.normal(size=(TRIALS, WINDOWS, STEPS, FEATURES))
    reconstruction = target + rng.normal(scale=0.3, size=target.shape)
    return target, reconstruction


def test_matches_numpy_reference(signals):
    target, reconstruction = signals
    seam, derivative = temporal_reconstruction_losses(
        _flat(target), _flat(reconstruction), WINDOWS
    )
    expected_seam, expected_derivative = _reference(target, reconstruction)
    np.testing.assert_allclose(seam, expected_seam, rtol=1e-5)
    np.testing.assert_allclose(derivative, expected_derivative, rtol=1e-5)


def test_perfect_reconstruction_is_zero(signals):
    target, _ = signals
    seam, derivative = temporal_reconstruction_losses(_flat(target), _flat(target), WINDOWS)
    assert float(seam) == pytest.approx(0.0, abs=1e-10)
    assert float(derivative) == pytest.approx(0.0, abs=1e-10)


def test_global_offset_is_not_penalized(signals):
    # Plain MSE already handles level; these terms only see changes.
    target, _ = signals
    seam, derivative = temporal_reconstruction_losses(
        _flat(target), _flat(target + 2.5), WINDOWS
    )
    assert float(seam) == pytest.approx(0.0, abs=1e-6)
    assert float(derivative) == pytest.approx(0.0, abs=1e-6)


def test_per_window_offsets_show_up_only_at_seams(signals):
    # A decoder that gets each window's level slightly wrong produces exactly
    # the boundary jumps measured in R(Z); only the seam term should see them.
    target, _ = signals
    offsets = np.arange(WINDOWS, dtype=float)[None, :, None, None] * 0.5
    seam, derivative = temporal_reconstruction_losses(
        _flat(target), _flat(target + offsets), WINDOWS
    )
    assert float(derivative) == pytest.approx(0.0, abs=1e-6)
    assert float(seam) == pytest.approx(0.25, rel=1e-5)


def test_seam_is_zero_without_window_order(signals):
    target, reconstruction = signals
    seam, derivative = temporal_reconstruction_losses(_flat(target), _flat(reconstruction))
    assert float(seam) == 0.0
    assert float(derivative) > 0.0
    single = temporal_reconstruction_losses(
        _flat(target), _flat(reconstruction), TRIALS * WINDOWS
    )
    np.testing.assert_allclose(single[1], derivative)
    one_window = temporal_reconstruction_losses(
        _flat(target)[:1], _flat(reconstruction)[:1], 1
    )
    assert float(one_window[0]) == 0.0


def test_graph_mode_with_dynamic_shapes(signals):
    target, reconstruction = signals
    spec = tf.TensorSpec((None, None, FEATURES), tf.float32)
    graph = tf.function(
        temporal_reconstruction_losses,
        input_signature=(spec, spec, tf.TensorSpec((), tf.int32)),
    )
    eager = temporal_reconstruction_losses(_flat(target), _flat(reconstruction), WINDOWS)
    traced = graph(_flat(target), _flat(reconstruction), tf.constant(WINDOWS))
    np.testing.assert_allclose(traced, eager, rtol=1e-6)


def _build(**extra):
    from eegproc.deep_learning.joint_architectures.SICModelv15.sic_model import (
        build_sic_model,
    )

    tf.keras.utils.set_random_seed(11)
    level = extra.pop("classification_level", "trial")
    shape = (2, 4, 6) if level == "trial" else (4, 6)
    return build_sic_model(
        shape, adjacency=np.eye(2), n_channels=2, n_bands=3,
        gcn_units=(4, 3), spectral_gru_units=4, bilstm_units=3,
        classifier_rnn_units=(4, 3), n_classifier_rnn_layers=2,
        n_subject_classes=4, classification_level=level,
        reconstruction_loss_weight=0.4, decoder_dropout=0.0,
        gcn_dropout=0.0, spectral_gru_dropout=0.0, bilstm_dropout=0.0,
        classifier_rnn_dropout=0.0, **extra,
    )


def test_model_adds_weighted_terms_to_reconstruction_loss():
    eeg = tf.constant(np.random.default_rng(3).normal(size=(4, 2, 4, 6)), tf.float32)
    plain = _build()
    temporal = _build(reconstruction_seam_weight=0.7, reconstruction_derivative_weight=0.3)
    plain(eeg)
    temporal(eeg)
    temporal.set_weights(plain.get_weights())
    a = plain._reconstruction_components(plain._encode(eeg, training=False), False)
    b = temporal._reconstruction_components(temporal._encode(eeg, training=False), False)

    assert float(a["reconstruction_seam_loss"]) == 0.0
    seam, derivative = b["reconstruction_seam_loss"], b["reconstruction_derivative_loss"]
    assert float(seam) > 0.0 and float(derivative) > 0.0
    np.testing.assert_allclose(
        b["reconstruction_loss"],
        a["reconstruction_loss"] + 0.7 * seam + 0.3 * derivative,
        rtol=1e-5,
    )

    # Blended across paths like the MSE: joint plus auxiliary-weighted branch mean.
    target = tf.reshape(eeg, (-1, 4, 6))
    terms = {
        name: temporal_reconstruction_losses(target, b[name], 2)
        for name in ("joint_reconstruction", "gcn_gru_reconstruction", "bilstm_reconstruction")
    }
    branch = [(terms["gcn_gru_reconstruction"][i] + terms["bilstm_reconstruction"][i]) / 2
              for i in (0, 1)]
    for index, value in enumerate((seam, derivative)):
        expected = (terms["joint_reconstruction"][index] + 0.25 * branch[index]) / 1.25
        np.testing.assert_allclose(value, expected, rtol=1e-5)

    config = temporal.get_config()
    assert config["reconstruction_seam_weight"] == 0.7
    assert config["reconstruction_derivative_weight"] == 0.3


def test_seam_weight_requires_trial_level():
    with pytest.raises(ValueError, match="classification_level='trial'"):
        _build(classification_level="window", reconstruction_seam_weight=0.5)
    _build(classification_level="window", reconstruction_derivative_weight=0.5)


def test_cli_can_ablate_configured_seam_and_set_derivative_independently():
    from eegproc.deep_learning.joint_architectures.SICModelv15.sic_model_args import (
        parse_cli,
    )

    args = [
        "--window-overlap", "0.5",
        "--hyperparameters-json",
        '{"reconstruction_seam_weight": 0.7, "reconstruction_derivative_weight": 0.1}',
    ]
    with pytest.raises(ValueError, match="requires --window-overlap 0"):
        parse_cli(args)

    _, config = parse_cli(args + [
        "--reconstruction-seam-weight", "0",
        "--reconstruction-derivative-weight", "0.3",
    ])
    assert config["reconstruction_seam_weight"] == 0.0
    assert config["reconstruction_derivative_weight"] == 0.3


def test_gradient_flows_only_through_window_edges(signals):
    target, reconstruction = signals
    r = tf.Variable(_flat(reconstruction))
    with tf.GradientTape() as tape:
        seam, _ = temporal_reconstruction_losses(_flat(target), r, WINDOWS)
    grad = tape.gradient(seam, r).numpy().reshape(TRIALS, WINDOWS, STEPS, FEATURES)
    assert np.all(grad[:, :, 1:-1, :] == 0.0)
    assert np.any(grad[:, 1:, 0, :] != 0.0) and np.any(grad[:, :-1, -1, :] != 0.0)
