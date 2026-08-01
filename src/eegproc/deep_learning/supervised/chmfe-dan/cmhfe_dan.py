"""CMHFE and CMHFE-DAN models for raw EEG emotion recognition.

This module keeps the shared feature extractor reusable on its own while also
providing builders for the plain emotion-recognition model and the optional
domain-adversarial variant.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Sequence

import tensorflow as tf
from tensorflow.keras import Model, layers

from ...unsupervised.utils import _ensure_tuple


def binarize_ratings(ratings, threshold: float):
    """Convert continuous ratings to binary labels using a threshold."""
    ratings = tf.convert_to_tensor(ratings)
    return tf.cast(ratings > threshold, tf.int32)


def binary_labels_to_one_hot(labels, num_classes: int = 2):
    """Convert integer class labels into one-hot targets for BCE training."""
    labels = tf.cast(tf.reshape(labels, [-1]), tf.int32)
    return tf.one_hot(labels, depth=num_classes)


@dataclass(slots=True)
class CMHFEConfig:
    """Configuration for CMHFE / CMHFE-DAN models."""

    n_channels: int
    window_length: float = 4.0
    sampling_frequency: float = 128.0
    cnn_filters: tuple[int, ...] = (64, 128, 256, 128)
    conv_kernel_size: int | Sequence[int] = 3
    conv_strides: int | Sequence[int] = 1
    conv_padding: str | Sequence[str] = "same"
    dropout_rate: float = 0.5
    l2_regularization: float = 0.001
    transformer_heads: int = 4
    transformer_embedding_dim: int = 128
    transformer_ffn_dim: int = 512
    enable_dann: bool = False
    enable_maxpool: bool = False
    maxpool_size: int = 2
    maxpool_padding: str = "same"
    domain_loss_weight: float = 1.0
    grl_lambda: float = 0.5
    valence_threshold: float = 5.0
    arousal_threshold: float = 5.0
    num_emotion_classes: int = 2
    learning_rate: float = 1e-3
    batch_size: int = 32

    def __post_init__(self) -> None:
        if self.n_channels < 1:
            raise ValueError("n_channels must be at least 1.")
        if self.window_length <= 0:
            raise ValueError("window_length must be positive.")
        if self.sampling_frequency <= 0:
            raise ValueError("sampling_frequency must be positive.")
        if len(self.cnn_filters) < 1:
            raise ValueError("cnn_filters must contain at least one layer.")
        if self.transformer_heads < 1:
            raise ValueError("transformer_heads must be at least 1.")
        if self.transformer_embedding_dim < 1:
            raise ValueError("transformer_embedding_dim must be at least 1.")
        if self.transformer_ffn_dim < 1:
            raise ValueError("transformer_ffn_dim must be at least 1.")
        if self.dropout_rate < 0.0 or self.dropout_rate >= 1.0:
            raise ValueError("dropout_rate must be in the interval [0, 1).")
        if self.l2_regularization < 0.0:
            raise ValueError("l2_regularization must be non-negative.")
        if self.domain_loss_weight < 0.0:
            raise ValueError("domain_loss_weight must be non-negative.")
        if self.grl_lambda < 0.0:
            raise ValueError("grl_lambda must be non-negative.")
        if self.num_emotion_classes < 2:
            raise ValueError("num_emotion_classes must be at least 2.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        self.conv_kernel_size = _ensure_tuple(
            self.conv_kernel_size,
            len(self.cnn_filters),
            "conv_kernel_size",
        )
        self.conv_strides = _ensure_tuple(
            self.conv_strides,
            len(self.cnn_filters),
            "conv_strides",
        )
        self.conv_padding = _ensure_tuple(
            self.conv_padding,
            len(self.cnn_filters),
            "conv_padding",
        )

        if any(p not in ("same", "valid") for p in self.conv_padding):
            raise ValueError("conv_padding must contain only 'same' or 'valid'.")
        if self.maxpool_padding not in ("same", "valid"):
            raise ValueError("maxpool_padding must be 'same' or 'valid'.")
        if self.transformer_embedding_dim % self.transformer_heads != 0:
            raise ValueError(
                "transformer_embedding_dim must be divisible by transformer_heads."
            )

    @property
    def conv_dropout(self) -> float:
        return self.dropout_rate

    @property
    def transformer_key_dim(self) -> int:
        return self.transformer_embedding_dim // self.transformer_heads

    @property
    def window_samples(self) -> int:
        return int(round(self.window_length * self.sampling_frequency))


def _l2_regularizer(config: CMHFEConfig):
    return tf.keras.regularizers.L2(config.l2_regularization)


def _coerce_config(config: CMHFEConfig | dict) -> CMHFEConfig:
    if isinstance(config, CMHFEConfig):
        return config
    if isinstance(config, dict):
        return CMHFEConfig(**config)
    raise TypeError(f"config must be a CMHFEConfig or dict, got {type(config)!r}.")


@tf.keras.utils.register_keras_serializable(package="eegproc")
class GradientReversalLayer(layers.Layer):
    """Identity in the forward pass and gradient negation in the backward pass."""

    def __init__(self, lambda_: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.lambda_ = float(lambda_)

    def call(self, inputs):
        lambda_ = tf.cast(self.lambda_, inputs.dtype)

        @tf.custom_gradient
        def _reverse(x):
            def grad(dy):
                return -lambda_ * dy

            return x, grad

        return _reverse(inputs)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"lambda_": self.lambda_})
        return config


@tf.keras.utils.register_keras_serializable(package="eegproc")
class TransformerEncoder(layers.Layer):
    """Standard Transformer encoder block with MHSA, residuals, and FFN."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout_rate: float = 0.0,
        l2_regularization: float = 0.0,
        name: str = "transformer_encoder",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be at least 1.")
        if num_heads < 1:
            raise ValueError("num_heads must be at least 1.")
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")
        if ffn_dim < 1:
            raise ValueError("ffn_dim must be at least 1.")

        self.embedding_dim = int(embedding_dim)
        self.num_heads = int(num_heads)
        self.ffn_dim = int(ffn_dim)
        self.dropout_rate = float(dropout_rate)
        self.l2_regularization = float(l2_regularization)

        regularizer = (
            tf.keras.regularizers.L2(self.l2_regularization)
            if self.l2_regularization > 0.0
            else None
        )

        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embedding_dim // self.num_heads,
            dropout=self.dropout_rate,
            name="mhsa",
        )
        self.attention_dropout = layers.Dropout(self.dropout_rate, name="mhsa_dropout")
        self.attention_norm = layers.LayerNormalization(epsilon=1e-6, name="mhsa_norm")
        self.ffn_dense_1 = layers.Dense(
            self.ffn_dim,
            activation=None,
            kernel_regularizer=regularizer,
            name="ffn_dense_1",
        )
        self.ffn_activation = layers.ReLU(name="ffn_relu")
        self.ffn_dropout = layers.Dropout(self.dropout_rate, name="ffn_dropout")
        self.ffn_dense_2 = layers.Dense(
            self.embedding_dim,
            activation=None,
            kernel_regularizer=regularizer,
            name="ffn_dense_2",
        )
        self.ffn_norm = layers.LayerNormalization(epsilon=1e-6, name="ffn_norm")

    def call(self, inputs, training: bool = False):
        attention_output = self.attention(inputs, inputs, training=training)
        attention_output = self.attention_dropout(attention_output, training=training)
        x = self.attention_norm(inputs + attention_output)

        ffn_output = self.ffn_dense_1(x)
        ffn_output = self.ffn_activation(ffn_output)
        ffn_output = self.ffn_dropout(ffn_output, training=training)
        ffn_output = self.ffn_dense_2(ffn_output)
        return self.ffn_norm(x + ffn_output)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "num_heads": self.num_heads,
                "ffn_dim": self.ffn_dim,
                "dropout_rate": self.dropout_rate,
                "l2_regularization": self.l2_regularization,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="eegproc")
class CNN1DFeatureExtractor(Model):
    """Four-block 1D CNN feature extractor for raw EEG windows."""

    def __init__(self, config: CMHFEConfig | dict, name: str = "cnn1d_feature_extractor", **kwargs):
        super().__init__(name=name, **kwargs)
        self.config = _coerce_config(config)
        regularizer = _l2_regularizer(config) if config.l2_regularization > 0.0 else None

        self.permute = layers.Permute((2, 1), name="permute_channels_samples")
        self.conv_blocks: list[tuple[layers.Layer, layers.Layer, layers.Layer, layers.Layer]] = []

        for index, (filters, kernel_size, stride, padding) in enumerate(
            zip(
                self.config.cnn_filters,
                self.config.conv_kernel_size,
                self.config.conv_strides,
                self.config.conv_padding,
            )
        ):
            self.conv_blocks.append(
                (
                    layers.Conv1D(
                        filters=filters,
                        kernel_size=kernel_size,
                        strides=stride,
                        padding=padding,
                        use_bias=False,
                        kernel_regularizer=regularizer,
                        name=f"conv_{index}",
                    ),
                    layers.BatchNormalization(name=f"bn_{index}"),
                    layers.ReLU(name=f"relu_{index}"),
                    layers.Dropout(self.config.conv_dropout, name=f"dropout_{index}"),
                )
            )

        self.sequence_projection = None
        if self.config.cnn_filters[-1] != self.config.transformer_embedding_dim:
            self.sequence_projection = layers.Dense(
                self.config.transformer_embedding_dim,
                activation=None,
                kernel_regularizer=regularizer,
                name="sequence_projection",
            )

    def call(self, inputs, training: bool = False):
        x = self.permute(inputs)

        for conv, batch_norm, activation, dropout in self.conv_blocks:
            x = conv(x)
            x = batch_norm(x, training=training)
            x = activation(x)
            x = dropout(x, training=training)

        if self.sequence_projection is not None:
            x = self.sequence_projection(x)

        return x

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"config": asdict(self.config)})
        return config

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)


@tf.keras.utils.register_keras_serializable(package="eegproc")
class EmotionHead(layers.Layer):
    """Independent emotion head for a single emotion dimension."""

    def __init__(
        self,
        num_classes: int = 2,
        hidden_units: tuple[int, int] = (128, 64),
        l2_regularization: float = 0.0,
        name: str = "emotion_head",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        self.num_classes = int(num_classes)
        self.hidden_units = tuple(hidden_units)
        self.l2_regularization = float(l2_regularization)

        regularizer = (
            tf.keras.regularizers.L2(self.l2_regularization)
            if self.l2_regularization > 0.0
            else None
        )

        self.dense_1 = layers.Dense(
            self.hidden_units[0],
            activation=None,
            kernel_regularizer=regularizer,
            name="dense_1",
        )
        self.relu_1 = layers.ReLU(name="relu_1")
        self.dense_2 = layers.Dense(
            self.hidden_units[1],
            activation=None,
            kernel_regularizer=regularizer,
            name="dense_2",
        )
        self.relu_2 = layers.ReLU(name="relu_2")
        self.output_layer = layers.Dense(
            self.num_classes,
            activation="sigmoid",
            kernel_regularizer=regularizer,
            name="emotion_logits",
        )

    def call(self, inputs, training: bool = False):
        x = self.dense_1(inputs)
        x = self.relu_1(x)
        x = self.dense_2(x)
        x = self.relu_2(x)
        return self.output_layer(x)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "hidden_units": self.hidden_units,
                "l2_regularization": self.l2_regularization,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="eegproc")
class DomainClassifier(layers.Layer):
    """Optional domain classifier for CMHFE-DAN."""

    def __init__(
        self,
        lambda_: float = 1.0,
        hidden_units: tuple[int, int] = (128, 64),
        l2_regularization: float = 0.0,
        name: str = "domain_classifier",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.lambda_ = float(lambda_)
        self.hidden_units = tuple(hidden_units)
        self.l2_regularization = float(l2_regularization)

        regularizer = (
            tf.keras.regularizers.L2(self.l2_regularization)
            if self.l2_regularization > 0.0
            else None
        )

        self.grl = GradientReversalLayer(lambda_=self.lambda_, name="gradient_reversal")
        self.dense_1 = layers.Dense(
            self.hidden_units[0],
            activation=None,
            kernel_regularizer=regularizer,
            name="dense_1",
        )
        self.relu_1 = layers.ReLU(name="relu_1")
        self.dense_2 = layers.Dense(
            self.hidden_units[1],
            activation=None,
            kernel_regularizer=regularizer,
            name="dense_2",
        )
        self.relu_2 = layers.ReLU(name="relu_2")
        self.output_layer = layers.Dense(
            1,
            activation="sigmoid",
            kernel_regularizer=regularizer,
            name="domain_probability",
        )

    def call(self, inputs, training: bool = False):
        x = self.grl(inputs)
        x = self.dense_1(x)
        x = self.relu_1(x)
        x = self.dense_2(x)
        x = self.relu_2(x)
        return self.output_layer(x)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "lambda_": self.lambda_,
                "hidden_units": self.hidden_units,
                "l2_regularization": self.l2_regularization,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="eegproc")
class CMHFEFeatureExtractor(Model):
    """CNN + Transformer shared feature extractor for CMHFE."""

    def __init__(
        self,
        config: CMHFEConfig | dict,
        cnn_encoder: CNN1DFeatureExtractor | None = None,
        transformer_encoder: TransformerEncoder | None = None,
        name: str = "cmhfe_feature_extractor",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.config = _coerce_config(config)
        self.cnn_encoder = cnn_encoder or CNN1DFeatureExtractor(
            config,
            name="cmhfe_cnn_encoder",
        )
        self.max_pool = (
            layers.MaxPool1D(
                pool_size=self.config.maxpool_size,
                padding=self.config.maxpool_padding,
                name="dann_maxpool",
            )
            if self.config.enable_maxpool
            else None
        )
        self.transformer_encoder = transformer_encoder or TransformerEncoder(
            embedding_dim=self.config.transformer_embedding_dim,
            num_heads=self.config.transformer_heads,
            ffn_dim=self.config.transformer_ffn_dim,
            dropout_rate=self.config.dropout_rate,
            l2_regularization=self.config.l2_regularization,
            name="cmhfe_transformer",
        )
        self.global_pool = layers.GlobalAveragePooling1D(name="global_average_pool")

    def call(self, inputs, training: bool = False):
        x = self.cnn_encoder(inputs, training=training)
        if self.max_pool is not None:
            x = self.max_pool(x)
        x = self.transformer_encoder(x, training=training)
        return self.global_pool(x)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"config": asdict(self.config)})
        return config

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)


def _compile_losses(config: CMHFEConfig, enable_dann: bool):
    def emotion_bce():
        bce = tf.keras.losses.BinaryCrossentropy()

        def loss(y_true, y_pred):
            y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
            target = tf.one_hot(y_true, depth=config.num_emotion_classes)
            return bce(target, y_pred)

        return loss

    def domain_bce():
        bce = tf.keras.losses.BinaryCrossentropy()

        def loss(y_true, y_pred):
            y_true = tf.cast(tf.reshape(y_true, tf.shape(y_pred)), y_pred.dtype)
            return bce(y_true, y_pred)

        return loss

    losses = {
        "valence": emotion_bce(),
        "arousal": emotion_bce(),
    }
    loss_weights = {
        "valence": 1.0,
        "arousal": 1.0,
    }
    if enable_dann:
        losses["domain"] = domain_bce()
        loss_weights["domain"] = config.domain_loss_weight
    return losses, loss_weights


def build_cmhfe_model(
    config: CMHFEConfig,
    feature_extractor: CMHFEFeatureExtractor | None = None,
    valence_head: EmotionHead | None = None,
    arousal_head: EmotionHead | None = None,
    name: str = "cmhfe_model",
) -> Model:
    """Build the CMHFE emotion-recognition model without domain adaptation."""
    feature_extractor = feature_extractor or CMHFEFeatureExtractor(config)
    valence_head = valence_head or EmotionHead(
        num_classes=config.num_emotion_classes,
        l2_regularization=config.l2_regularization,
        name="valence_head",
    )
    arousal_head = arousal_head or EmotionHead(
        num_classes=config.num_emotion_classes,
        l2_regularization=config.l2_regularization,
        name="arousal_head",
    )

    inputs = layers.Input(shape=(config.n_channels, None), name="eeg_window")
    shared_features = feature_extractor(inputs)
    outputs = {
        "valence": valence_head(shared_features),
        "arousal": arousal_head(shared_features),
    }

    model = Model(inputs=inputs, outputs=outputs, name=name)
    losses, loss_weights = _compile_losses(config, enable_dann=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=losses,
        loss_weights=loss_weights,
    )
    return model


def build_cmhfe_dann_model(
    config: CMHFEConfig,
    feature_extractor: CMHFEFeatureExtractor | None = None,
    valence_head: EmotionHead | None = None,
    arousal_head: EmotionHead | None = None,
    domain_head: DomainClassifier | None = None,
    name: str = "cmhfe_dann_model",
) -> Model:
    """Build the CMHFE-DAN model with the optional domain classifier branch."""
    feature_extractor = feature_extractor or CMHFEFeatureExtractor(config)
    valence_head = valence_head or EmotionHead(
        num_classes=config.num_emotion_classes,
        l2_regularization=config.l2_regularization,
        name="valence_head",
    )
    arousal_head = arousal_head or EmotionHead(
        num_classes=config.num_emotion_classes,
        l2_regularization=config.l2_regularization,
        name="arousal_head",
    )
    domain_head = domain_head or DomainClassifier(
        lambda_=config.grl_lambda,
        l2_regularization=config.l2_regularization,
        name="domain_head",
    )

    inputs = layers.Input(shape=(config.n_channels, None), name="eeg_window")
    shared_features = feature_extractor(inputs)
    outputs = {
        "valence": valence_head(shared_features),
        "arousal": arousal_head(shared_features),
        "domain": domain_head(shared_features),
    }

    model = Model(inputs=inputs, outputs=outputs, name=name)
    losses, loss_weights = _compile_losses(config, enable_dann=True)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=losses,
        loss_weights=loss_weights,
    )
    return model


def CMHFEModel(
    config: CMHFEConfig,
    feature_extractor: CMHFEFeatureExtractor | None = None,
    valence_head: EmotionHead | None = None,
    arousal_head: EmotionHead | None = None,
    name: str = "cmhfe_model",
) -> Model:
    """Convenience alias matching the architecture name in the specification."""
    return build_cmhfe_model(
        config=config,
        feature_extractor=feature_extractor,
        valence_head=valence_head,
        arousal_head=arousal_head,
        name=name,
    )


def CMHFEDANNModel(
    config: CMHFEConfig,
    feature_extractor: CMHFEFeatureExtractor | None = None,
    valence_head: EmotionHead | None = None,
    arousal_head: EmotionHead | None = None,
    domain_head: DomainClassifier | None = None,
    name: str = "cmhfe_dann_model",
) -> Model:
    """Convenience alias matching the architecture name in the specification."""
    return build_cmhfe_dann_model(
        config=config,
        feature_extractor=feature_extractor,
        valence_head=valence_head,
        arousal_head=arousal_head,
        domain_head=domain_head,
        name=name,
    )


__all__ = [
    "CMHFEConfig",
    "CNN1DFeatureExtractor",
    "CMHFEFeatureExtractor",
    "CMHFEModel",
    "CMHFEDANNModel",
    "DomainClassifier",
    "EmotionHead",
    "GradientReversalLayer",
    "TransformerEncoder",
    "binary_labels_to_one_hot",
    "binarize_ratings",
    "build_cmhfe_dann_model",
    "build_cmhfe_model",
]