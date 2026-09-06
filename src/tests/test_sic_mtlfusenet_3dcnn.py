"""Shape, geometry, gradient, and serialization tests for SIC's 3D-CNN."""

import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf

from eegproc.deep_learning.joint_architectures.SICModelv11.mtlfusenet_3dcnn import (
    DREAMER_ELECTRODE_GRID,
    MTLFuseNet3DCNNEncoder,
)
from eegproc.deep_learning.joint_architectures.SICModelv11.sic_model import (
    build_sic_model,
)


def _small_encoder(**kwargs):
    return MTLFuseNet3DCNNEncoder(
        filters=(4, 8),
        temporal_kernel_size=3,
        spatial_pool_sizes=(3, 1),
        dropout=0.0,
        **kwargs,
    )


def test_channel_major_features_are_scattered_to_mtl_grid():
    encoder = _small_encoder()
    inputs = np.arange(42, dtype=np.float32).reshape(1, 1, 42)

    grid = encoder._to_spatial_grid(inputs).numpy()

    assert grid.shape == (1, 1, 9, 9, 3)
    for channel, (row, col) in enumerate(DREAMER_ELECTRODE_GRID):
        np.testing.assert_array_equal(grid[0, 0, row, col], inputs[0, 0, 3 * channel:3 * channel + 3])
    assert np.count_nonzero(grid) == np.count_nonzero(inputs)


def test_encoder_preserves_time_and_backpropagates():
    encoder = _small_encoder()
    inputs = tf.random.normal((2, 8, 42), seed=7)

    with tf.GradientTape() as tape:
        outputs = encoder(inputs, training=True)
        loss = tf.reduce_sum(outputs)
    gradients = tape.gradient(loss, encoder.trainable_variables)

    assert outputs.shape == (2, 8, 8)
    assert gradients
    assert all(gradient is not None for gradient in gradients)


def test_sic_fuses_gcn_gru_and_cnn3d_sequences():
    model = build_sic_model(
        input_shape=(2, 8, 42),
        adjacency=np.eye(14, dtype=np.float32),
        n_classes=2,
        gcn_units=(4,),
        spectral_gru_units=6,
        cnn3d_filters=(4, 8),
        cnn3d_temporal_kernel_size=3,
        cnn3d_spatial_pool_sizes=(3, 1),
        cnn3d_dropout=0.0,
        classifier_rnn_units=(5,),
        classifier_rnn_dropout=0.0,
        use_decoder=False,
        use_subject_adversarial=False,
    )
    inputs = tf.random.normal((2, 2, 8, 42), seed=11)

    probabilities = model(inputs, training=False)
    features = model.get_encoder_features(inputs)

    assert probabilities.shape == (2, 2)
    assert features["gcn_gru_features"].shape == (4, 8, 6)
    assert features["cnn3d_features"].shape == (4, 8, 8)
    assert features["combined_feature_sequence"].shape == (4, 8, 14)
    assert features["classifier_sequence"].shape == (2, 16, 14)


def test_each_active_branch_reconstructs_independently():
    model = build_sic_model(
        input_shape=(2, 8, 42),
        adjacency=np.eye(14, dtype=np.float32),
        n_classes=2,
        gcn_units=(4,),
        spectral_gru_units=6,
        cnn3d_filters=(4, 8),
        cnn3d_temporal_kernel_size=3,
        cnn3d_spatial_pool_sizes=(3, 1),
        cnn3d_dropout=0.0,
        classifier_rnn_units=(5,),
        classifier_rnn_dropout=0.0,
        use_decoder=True,
        use_subject_adversarial=False,
    )
    inputs = tf.random.normal((2, 2, 8, 42), seed=12)

    reconstructions = model.reconstruct_branches(inputs)

    assert set(reconstructions) == {"gcn_gru", "cnn3d"}
    assert reconstructions["gcn_gru"].shape == inputs.shape
    assert reconstructions["cnn3d"].shape == inputs.shape
    assert model.decode_branch_feature_sequence(
        "spatiotemporal", model.get_encoder_features(inputs)["cnn3d_features"]
    ).shape == (4, 8, 42)

    logs = model.train_on_batch(
        inputs, np.asarray([0, 1], dtype=np.int32), return_dict=True
    )
    assert "cnn3d_reconstruction_loss" in logs
    assert all(np.isfinite(value) for value in logs.values())


def test_cnn3d_only_model_round_trips_through_keras_config():
    model = build_sic_model(
        input_shape=(2, 8, 42),
        adjacency=np.eye(14, dtype=np.float32),
        n_classes=2,
        cnn3d_filters=(4, 8),
        cnn3d_temporal_kernel_size=3,
        cnn3d_spatial_pool_sizes=(3, 1),
        cnn3d_dropout=0.0,
        classifier_rnn_units=(5,),
        classifier_rnn_dropout=0.0,
        use_gcn_gru_branch=False,
        use_cnn3d_branch=True,
        use_decoder=False,
        use_subject_adversarial=False,
    )
    inputs = tf.random.normal((2, 2, 8, 42), seed=13)
    expected = model(inputs, training=False)

    restored = model.__class__.from_config(model.get_config())
    _ = restored(inputs, training=False)
    restored.set_weights(model.get_weights())

    np.testing.assert_allclose(restored(inputs, training=False), expected, atol=1e-6)

    with tempfile.TemporaryDirectory() as directory:
        model_path = Path(directory) / "sic_cnn3d.keras"
        model.save(model_path)
        loaded = tf.keras.models.load_model(model_path, compile=False)
        np.testing.assert_allclose(
            loaded(inputs, training=False), expected, atol=1e-6
        )
