import numpy as np
import tensorflow as tf

from eegproc.deep_learning.supervised import (
    CMHFEConfig,
    CMHFEFeatureExtractor,
    CMHFEModel,
    CMHFEDANNModel,
    GradientReversalLayer,
    binary_labels_to_one_hot,
    binarize_ratings,
)


def test_gradient_reversal_layer_negates_gradients():
    layer = GradientReversalLayer(lambda_=2.0)
    inputs = tf.constant([[1.0, -2.0]], dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(inputs)
        outputs = layer(inputs)
        loss = tf.reduce_sum(outputs)

    gradients = tape.gradient(loss, inputs)
    np.testing.assert_allclose(gradients.numpy(), np.array([[-2.0, -2.0]]))


def test_feature_extractor_preserves_batch_and_embeds_sequences():
    config = CMHFEConfig(
        n_channels=32,
        cnn_filters=(16, 32, 64, 128),
        transformer_embedding_dim=128,
        transformer_heads=4,
        transformer_ffn_dim=256,
        dropout_rate=0.0,
    )
    extractor = CMHFEFeatureExtractor(config)

    x = np.random.randn(2, 32, 64).astype("float32")
    features = extractor(x, training=False)

    assert features.shape == (2, 128)


def test_cmhfe_model_builds_with_two_emotion_heads():
    config = CMHFEConfig(
        n_channels=32,
        cnn_filters=(16, 32, 64, 128),
        transformer_embedding_dim=128,
        transformer_heads=4,
        transformer_ffn_dim=256,
        dropout_rate=0.0,
        learning_rate=1e-3,
    )
    model = CMHFEModel(config)

    x = np.random.randn(3, 32, 64).astype("float32")
    outputs = model.predict(x, verbose=0)

    assert set(outputs) == {"valence", "arousal"}
    assert outputs["valence"].shape == (3, 2)
    assert outputs["arousal"].shape == (3, 2)


def test_cmhfe_dann_model_adds_domain_branch():
    config = CMHFEConfig(
        n_channels=14,
        cnn_filters=(16, 32, 64, 128),
        transformer_embedding_dim=128,
        transformer_heads=4,
        transformer_ffn_dim=256,
        enable_dann=True,
        enable_maxpool=True,
        dropout_rate=0.0,
        learning_rate=1e-3,
        domain_loss_weight=0.7,
        grl_lambda=0.5,
    )
    model = CMHFEDANNModel(config)

    x = np.random.randn(2, 14, 48).astype("float32")
    outputs = model.predict(x, verbose=0)

    assert set(outputs) == {"valence", "arousal", "domain"}
    assert outputs["domain"].shape == (2, 1)


def test_binary_label_helpers_match_thresholding_and_one_hot_encoding():
    ratings = tf.constant([4.5, 5.0, 5.5], dtype=tf.float32)
    labels = binarize_ratings(ratings, threshold=5.0)
    one_hot = binary_labels_to_one_hot(labels, num_classes=2)

    np.testing.assert_array_equal(labels.numpy(), np.array([0, 0, 1]))
    np.testing.assert_array_equal(
        one_hot.numpy(),
        np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
    )