"""Supervised RNN classifier architectures for EEG sequence data.

These classes build compiled Keras models ready for training and
cross-validation. The loss function is a parameter so callers can swap
losses without rewriting the model — e.g. the default softmax
cross-entropy today, a variational loss once it is implemented.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Union

import tensorflow as tf
from tensorflow.keras import layers, Model


LossLike = Union[str, Callable, tf.keras.losses.Loss]


class RNNClassifier(ABC):
    """Base class for RNN-based EEG sequence classifiers.

    Subclasses define the recurrent layer type. The shared classifier body is:

        Input
        -> recurrent stack
        -> temporal pooling
        -> global average pooling
        -> Dense logits
        -> Softmax probabilities
    """

    def __init__(
        self,
        timesteps: int,
        n_features: int,
        n_classes: int,
        rnn_units: int = 128,
        n_rnn_layers: int = 2,
        dropout: float = 0.10,
        loss: LossLike = "softmax_crossentropy",
        optimizer: Union[str, tf.keras.optimizers.Optimizer] = "adam",
        metrics: list[str] | None = None,
        name: str = "rnn_classifier",
    ) -> None:
        self.timesteps = timesteps
        self.n_features = n_features
        self.n_classes = n_classes
        self.rnn_units = rnn_units
        self.n_rnn_layers = n_rnn_layers
        self.dropout = dropout
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics if metrics is not None else ["accuracy"]
        self.name = name

    @staticmethod
    def resolve_loss(loss: LossLike) -> Union[str, Callable, tf.keras.losses.Loss]:
        """Map friendly loss aliases to Keras-compatible losses."""
        if loss == "softmax_crossentropy":
            return tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=False
            )

        if loss == "variational":
            raise NotImplementedError(
                "Variational loss is not yet implemented; "
                "use 'softmax_crossentropy' until it is."
            )

        return loss

    @abstractmethod
    def recurrent_layer(self, layer_index: int) -> tf.keras.layers.Layer:
        """Return one recurrent layer/block for this classifier."""
        raise NotImplementedError

    def recurrent_stack(self, x: tf.Tensor) -> tf.Tensor:
        """Apply the recurrent stack."""
        for layer_index in range(self.n_rnn_layers):
            x = self.recurrent_layer(layer_index)(x)
            x = layers.BatchNormalization(
                name=f"{self.name}_bn_{layer_index}"
            )(x)
            x = layers.Dropout(
                self.dropout,
                name=f"{self.name}_do_{layer_index}",
            )(x)

        return x

    def classifier_head(self, x: tf.Tensor) -> tf.Tensor:
        """Apply temporal pooling and classification head."""
        x = layers.MaxPool1D(
            pool_size=2,
            padding="same",
            name="enc_tpool1",
        )(x)

        x = layers.MaxPool1D(
            pool_size=2,
            padding="same",
            name="enc_tpool2",
        )(x)

        x = layers.GlobalAveragePooling1D(name="gap")(x)
        x = layers.Dense(self.n_classes, name="class_logits")(x)
        x = layers.Softmax(name="class_probabilities")(x)

        return x

    def build(self) -> tf.keras.Model:
        """Build and compile the Keras model."""
        resolved_loss = self.resolve_loss(self.loss)

        x_in = layers.Input(
            shape=(self.timesteps, self.n_features),
            name="x",
        )

        x = self.recurrent_stack(x_in)
        output = self.classifier_head(x)

        model = Model(
            inputs=x_in,
            outputs=output,
            name=self.name,
        )

        model.compile(
            optimizer=self.optimizer,
            loss=resolved_loss,
            metrics=self.metrics,
        )

        return model

    def __call__(self) -> tf.keras.Model:
        """Allow classifier instances to be called like builders."""
        return self.build()


class LSTMClassifier(RNNClassifier):
    """Unidirectional LSTM classifier for EEG sequence data."""

    def __init__(
        self,
        timesteps: int,
        n_features: int,
        n_classes: int,
        lstm_units: int = 128,
        n_lstm_layers: int = 2,
        dropout: float = 0.10,
        loss: LossLike = "softmax_crossentropy",
        optimizer: Union[str, tf.keras.optimizers.Optimizer] = "adam",
        metrics: list[str] | None = None,
        name: str = "lstm_classifier",
    ) -> None:
        super().__init__(
            timesteps=timesteps,
            n_features=n_features,
            n_classes=n_classes,
            rnn_units=lstm_units,
            n_rnn_layers=n_lstm_layers,
            dropout=dropout,
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
            name=name,
        )

        self.lstm_units = lstm_units
        self.n_lstm_layers = n_lstm_layers

    def recurrent_layer(self, layer_index: int) -> tf.keras.layers.Layer:
        return layers.LSTM(
            self.rnn_units,
            return_sequences=True,
            name=f"lstm_{layer_index}",
        )


class BiLSTMClassifier(RNNClassifier):
    """Bidirectional LSTM classifier for EEG sequence data."""

    def __init__(
        self,
        timesteps: int,
        n_features: int,
        n_classes: int,
        lstm_units: int = 128,
        n_bilstm_layers: int = 2,
        dropout: float = 0.10,
        loss: LossLike = "softmax_crossentropy",
        optimizer: Union[str, tf.keras.optimizers.Optimizer] = "adam",
        metrics: list[str] | None = None,
        name: str = "bilstm_classifier",
    ) -> None:
        super().__init__(
            timesteps=timesteps,
            n_features=n_features,
            n_classes=n_classes,
            rnn_units=lstm_units,
            n_rnn_layers=n_bilstm_layers,
            dropout=dropout,
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
            name=name,
        )

        self.lstm_units = lstm_units
        self.n_bilstm_layers = n_bilstm_layers

    def recurrent_layer(self, layer_index: int) -> tf.keras.layers.Layer:
        return layers.Bidirectional(
            layers.LSTM(
                self.rnn_units,
                return_sequences=True,
            ),
            merge_mode="concat",
            name=f"bilstm_{layer_index}",
        )