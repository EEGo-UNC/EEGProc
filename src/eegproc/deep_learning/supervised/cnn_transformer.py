"""Standalone 3D-CNN + temporal Transformer classifiers for EEG.

The convolutional encoder supplies local spatiotemporal features as a sequence
of time tokens.  A small pre-normalized Transformer then models relationships
between any pair of tokens before temporal pooling and classification.
"""

from __future__ import annotations

from typing import Any

import tensorflow as tf
from tensorflow.keras import layers

from ..unsupervised.Convolutions.CNN3D import MTLFuseNet3DCNNEncoder


@tf.keras.utils.register_keras_serializable(package="EEGProc")
class LearnedTemporalPositionEmbedding(layers.Layer):
    """Add a learned position vector to every temporal token."""

    def __init__(self, max_length: int, embedding_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.max_length = int(max_length)
        self.embedding_dim = int(embedding_dim)
        if self.max_length < 1 or self.embedding_dim < 1:
            raise ValueError("max_length and embedding_dim must be positive.")
        self.positions = layers.Embedding(
            input_dim=self.max_length,
            output_dim=self.embedding_dim,
            name="position_vectors",
        )

    def build(self, input_shape):
        self.positions.build((None,))
        super().build(input_shape)

    def call(self, inputs):
        length = tf.shape(inputs)[1]
        tf.debugging.assert_less_equal(
            length,
            self.max_length,
            message="Token sequence exceeds the configured positional embedding.",
        )
        return inputs + self.positions(tf.range(length))[tf.newaxis, :, :]

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {"max_length": self.max_length, "embedding_dim": self.embedding_dim}
        )
        return config


@tf.keras.utils.register_keras_serializable(package="EEGProc")
class TemporalTransformerBlock(layers.Layer):
    """Pre-normalized self-attention and feed-forward residual block."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 4,
        feed_forward_dim: int = 256,
        dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embedding_dim = int(embedding_dim)
        self.num_heads = int(num_heads)
        self.feed_forward_dim = int(feed_forward_dim)
        self.dropout_rate = float(dropout)
        if self.embedding_dim < 1 or self.num_heads < 1 or self.feed_forward_dim < 1:
            raise ValueError(
                "embedding_dim, num_heads, and feed_forward_dim must be positive."
            )
        if self.embedding_dim % self.num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads.")
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.attention_norm = layers.LayerNormalization(
            epsilon=1e-6, name="attention_norm"
        )
        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embedding_dim // self.num_heads,
            dropout=self.dropout_rate,
            name="temporal_self_attention",
        )
        self.attention_dropout = layers.Dropout(
            self.dropout_rate, name="attention_dropout"
        )
        self.feed_forward_norm = layers.LayerNormalization(
            epsilon=1e-6, name="feed_forward_norm"
        )
        self.feed_forward = tf.keras.Sequential(
            [
                layers.Dense(
                    self.feed_forward_dim, activation="gelu", name="expand"
                ),
                layers.Dropout(self.dropout_rate, name="inner_dropout"),
                layers.Dense(self.embedding_dim, name="project"),
            ],
            name="feed_forward",
        )
        self.output_dropout = layers.Dropout(self.dropout_rate, name="output_dropout")

    def build(self, input_shape):
        self.attention_norm.build(input_shape)
        self.attention.build(input_shape, input_shape, input_shape)
        self.feed_forward_norm.build(input_shape)
        self.feed_forward.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, training: bool = False):
        normalized = self.attention_norm(inputs)
        attended = self.attention(
            normalized, normalized, normalized, training=training
        )
        x = inputs + self.attention_dropout(attended, training=training)
        transformed = self.feed_forward(
            self.feed_forward_norm(x), training=training
        )
        return x + self.output_dropout(transformed, training=training)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "num_heads": self.num_heads,
                "feed_forward_dim": self.feed_forward_dim,
                "dropout": self.dropout_rate,
            }
        )
        return config


def build_3dcnn_transformer_classifier(
    *,
    timesteps: int,
    n_features: int = 42,
    conv_filters: tuple[int, ...] = (32, 64),
    temporal_kernel_size: int = 7,
    spatial_kernel_size: int = 3,
    spatial_pool_sizes: tuple[int, ...] = (2, 2),
    temporal_pool_sizes: tuple[int, ...] = (),
    embedding_dim: int = 128,
    transformer_layers: int = 1,
    attention_heads: int = 4,
    feed_forward_dim: int = 256,
    classifier_units: int = 128,
    dropout: float = 0.2,
    n_classes: int = 2,
    name: str = "eeg_3dcnn_transformer",
) -> tf.keras.Model:
    """Build a standalone EEG classifier with exactly two Conv3D blocks.

    ``n_features`` is explicit to fail early when the model is connected to a
    representation other than DREAMER's 14 channels x 3 bands.
    """
    conv_filters = tuple(int(value) for value in conv_filters)
    spatial_pool_sizes = tuple(int(value) for value in spatial_pool_sizes)
    temporal_pool_sizes = tuple(int(value) for value in temporal_pool_sizes)
    if len(conv_filters) != 2:
        raise ValueError("The 3D-CNN + Transformer architecture requires two Conv3D blocks.")
    if len(spatial_pool_sizes) != 2:
        raise ValueError("spatial_pool_sizes must contain one value per Conv3D block.")
    if int(n_features) != 42:
        raise ValueError("This DREAMER model expects 42 channel-band features.")
    if int(transformer_layers) < 1:
        raise ValueError("transformer_layers must be at least 1.")

    temporal_downsampling = 1
    for pool_size in temporal_pool_sizes:
        temporal_downsampling *= pool_size

    inputs = layers.Input((int(timesteps), int(n_features)), name="eeg_window")
    encoder = MTLFuseNet3DCNNEncoder(
        timesteps=int(timesteps),
        t_down=temporal_downsampling,
        conv_filters=conv_filters,
        temporal_kernel_size=int(temporal_kernel_size),
        spatial_kernel_size=int(spatial_kernel_size),
        spatial_pool_sizes=spatial_pool_sizes,
        temporal_pool_sizes=temporal_pool_sizes,
        emb_dim=int(embedding_dim),
        dropout=float(dropout),
        name="local_spatiotemporal_encoder",
    )
    x = encoder(inputs)
    max_tokens = (int(timesteps) + temporal_downsampling - 1) // temporal_downsampling
    x = LearnedTemporalPositionEmbedding(
        max_tokens, int(embedding_dim), name="temporal_positions"
    )(x)
    for index in range(int(transformer_layers)):
        x = TemporalTransformerBlock(
            embedding_dim=int(embedding_dim),
            num_heads=int(attention_heads),
            feed_forward_dim=int(feed_forward_dim),
            dropout=float(dropout),
            name=f"temporal_transformer_{index}",
        )(x)
    x = layers.LayerNormalization(epsilon=1e-6, name="token_norm")(x)
    x = layers.GlobalAveragePooling1D(name="temporal_average")(x)
    x = layers.Dense(
        int(classifier_units), activation="gelu", name="classifier_features"
    )(x)
    x = layers.Dropout(float(dropout), name="classifier_dropout")(x)
    outputs = layers.Dense(
        int(n_classes), activation="softmax", name="class_probabilities"
    )(x)
    return tf.keras.Model(inputs, outputs, name=name)
