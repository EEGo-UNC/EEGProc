"""Tests for the standalone DREAMER 3D-CNN + Transformer classifier."""

import numpy as np
import pytest


tf = pytest.importorskip("tensorflow", reason="models require eegproc[deep-learning]")

from eegproc.deep_learning.supervised.cnn_transformer import (  # noqa: E402
    TemporalTransformerBlock,
    build_3dcnn_transformer_classifier,
)


def _small_model():
    return build_3dcnn_transformer_classifier(
        timesteps=8,
        conv_filters=(4, 8),
        temporal_kernel_size=3,
        spatial_pool_sizes=(2, 2),
        embedding_dim=8,
        attention_heads=2,
        feed_forward_dim=16,
        classifier_units=8,
        dropout=0.0,
    )


def test_model_has_two_convolutions_then_temporal_attention():
    model = _small_model()
    encoder = model.get_layer("local_spatiotemporal_encoder")

    assert len(encoder.convolutions) == 2
    assert isinstance(model.get_layer("temporal_transformer_0"), TemporalTransformerBlock)
    assert model.output_shape == (None, 2)


def test_model_backpropagates_and_serializes(tmp_path):
    tf.keras.utils.set_random_seed(7)
    model = _small_model()
    inputs = tf.random.normal((2, 8, 42), seed=7)
    with tf.GradientTape() as tape:
        outputs = model(inputs, training=True)
        loss = tf.reduce_sum(outputs[:, 0])
    gradients = tape.gradient(loss, model.trainable_variables)

    assert outputs.shape == (2, 2)
    np.testing.assert_allclose(outputs.numpy().sum(axis=1), 1.0, rtol=1e-5)
    assert gradients and all(gradient is not None for gradient in gradients)

    path = tmp_path / "cnn_transformer.keras"
    model.save(path)
    restored = tf.keras.models.load_model(path)
    np.testing.assert_allclose(restored(inputs), model(inputs), atol=1e-6)


def test_model_rejects_a_third_convolutional_block():
    with pytest.raises(ValueError, match="two Conv3D blocks"):
        build_3dcnn_transformer_classifier(
            timesteps=8,
            conv_filters=(4, 8, 16),
            spatial_pool_sizes=(2, 2, 1),
        )
