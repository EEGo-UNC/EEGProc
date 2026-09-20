"""Supervised RNN classifier architectures for EEG sequence data.

The builders in this module can be used in two ways:

1. ``build()`` creates a complete, compiled standalone classifier.
2. ``build_feature_extractor()`` creates only the recurrent temporal encoder
   and pooling pathway, returning one sequence-level embedding per sample.
3. ``build_sequence_summarizer()`` returns the final recurrent state without
   temporal pooling. This is useful when every ordered element in a sequence
   must contribute to a trial-level representation.

The feature-extractor form is intended for joint architectures where another
model, such as a variational classifier, owns the final classification head
and loss.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import tensorflow as tf
from tensorflow.keras import Model, layers

from .variational_classifier import VariationalClassifier


LossLike = str | Callable | tf.keras.losses.Loss
OptimizerLike = str | tf.keras.optimizers.Optimizer


class RNNClassifier(ABC):
    """Base builder for RNN-based EEG sequence classifiers.

    Parameters
    ----------
    timesteps : int | None
        Number of temporal samples in each input sequence. ``None`` allows a
        variable-length sequence.
    n_features : int
        Number of features at each timestep.
    n_classes : int
        Number of target classes used by the standalone classifier.
    rnn_units : int or sequence[int], default=128
        A scalar repeats one hidden width across ``n_rnn_layers``. A sequence
        gives the width of each recurrent layer, for example ``(128, 64)``.
    n_rnn_layers : int, default=2
        Number of recurrent layers in the stack.
    dropout : float, default=0.10
        Dropout applied after each recurrent layer.
    loss : str, callable, or tf.keras.losses.Loss
        Standalone-model loss. Use ``"variational"`` to attach a
        ``VariationalClassifier`` head.
    optimizer : str or tf.keras.optimizers.Optimizer, default="adam"
        Optimizer for the standalone compiled model.
    metrics : list[str] | None
        Metrics for the standalone compiled model.
    name : str, default="rnn_classifier"
        Base name used for the model and its layers.
    alpha, beta, gamma, lambda_ : float
        Coefficients forwarded to ``VariationalClassifier.keras_loss`` when
        ``loss="variational"``.
    """

    def __init__(
        self,
        timesteps: int | None,
        n_features: int,
        n_classes: int,
        rnn_units: int | Sequence[int] = 128,
        n_rnn_layers: int = 2,
        dropout: float = 0.10,
        loss: LossLike = "softmax_crossentropy",
        optimizer: OptimizerLike = "adam",
        metrics: list[str] | None = None,
        name: str = "rnn_classifier",
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.0,
        lambda_: float = 0.0,
    ) -> None:
        if timesteps is not None and timesteps < 1:
            raise ValueError("timesteps must be at least 1 or None.")
        if n_features < 1:
            raise ValueError("n_features must be at least 1.")
        if n_classes < 2:
            raise ValueError("n_classes must be at least 2.")
        if n_rnn_layers < 1:
            raise ValueError("n_rnn_layers must be at least 1.")
        if isinstance(rnn_units, Sequence) and not isinstance(
            rnn_units,
            (str, bytes),
        ):
            units_by_layer = tuple(int(units) for units in rnn_units)
            if not units_by_layer or any(units < 1 for units in units_by_layer):
                raise ValueError(
                    "rnn_units sequences must contain positive integers."
                )
            if len(units_by_layer) != int(n_rnn_layers):
                raise ValueError(
                    "The rnn_units sequence length must match n_rnn_layers; "
                    f"got {units_by_layer} and {n_rnn_layers}."
                )
        else:
            scalar_units = int(rnn_units)
            if scalar_units < 1:
                raise ValueError("rnn_units must be at least 1.")
            units_by_layer = (scalar_units,) * int(n_rnn_layers)
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.timesteps = None if timesteps is None else int(timesteps)
        self.n_features = int(n_features)
        self.n_classes = int(n_classes)
        self.rnn_units_by_layer = units_by_layer
        # Preserve a scalar final-width attribute for compatibility with code
        # that inspected this builder before per-layer widths were supported.
        self.rnn_units = self.rnn_units_by_layer[-1]
        self.n_rnn_layers = int(n_rnn_layers)
        self.dropout = float(dropout)
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics if metrics is not None else ["accuracy"]
        self.name = name
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.lambda_ = float(lambda_)

    @staticmethod
    def resolve_loss(loss: LossLike) -> LossLike:
        """Map friendly loss aliases to Keras-compatible losses."""
        if loss == "softmax_crossentropy":
            return tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=False
            )
        return loss

    def uses_variational_head(self) -> bool:
        """Return whether the standalone model should use the VC head."""
        return isinstance(self.loss, str) and self.loss == "variational"

    @abstractmethod
    def recurrent_layer(
        self,
        layer_index: int,
        *,
        return_sequences: bool,
    ) -> tf.keras.layers.Layer:
        """Create one recurrent layer/block for this architecture."""
        raise NotImplementedError

    def recurrent_stack(
        self,
        x: tf.Tensor,
        *,
        return_sequences: bool = True,
    ) -> tf.Tensor:
        """Apply the recurrent stack with an optional final temporal axis."""
        for layer_index in range(self.n_rnn_layers):
            is_final_layer = layer_index == self.n_rnn_layers - 1
            layer_returns_sequence = not is_final_layer or return_sequences
            x = self.recurrent_layer(
                layer_index,
                return_sequences=layer_returns_sequence,
            )(x)
            # Layer normalization is sample-local and does not carry running
            # statistics learned from the training subjects into a held-out
            # subject, making it safer for LOSO EEG generalization.
            x = layers.LayerNormalization(
                axis=-1,
                name=f"{self.name}_ln_{layer_index}",
            )(x)
            x = layers.Dropout(
                self.dropout,
                name=f"{self.name}_dropout_{layer_index}",
            )(x)
        return x

    def build_sequence_summarizer(self) -> tf.keras.Model:
        """Build a recurrent sequence-to-vector model without temporal pooling.

        Every element is processed in order by the recurrent stack. All
        intermediate recurrent layers retain the sequence; only the final
        layer returns its state. For bidirectional architectures, Keras
        concatenates the forward and backward final states.
        """
        x_in = layers.Input(
            shape=(self.timesteps, self.n_features),
            name=f"{self.name}_sequence_input",
        )
        embedding = self.recurrent_stack(x_in, return_sequences=False)

        return Model(
            inputs=x_in,
            outputs=embedding,
            name=f"{self.name}_sequence_summarizer",
        )

    def temporal_embedding(self, x: tf.Tensor) -> tf.Tensor:
        """Collapse an RNN output sequence into one embedding per sample.

        This retains the temporal pooling structure used by the original
        standalone RNN classifiers: two max-pooling stages followed by global
        average pooling.
        """
        x = layers.MaxPool1D(
            pool_size=2,
            padding="same",
            name=f"{self.name}_tpool_0",
        )(x)
        x = layers.MaxPool1D(
            pool_size=2,
            padding="same",
            name=f"{self.name}_tpool_1",
        )(x)
        return layers.GlobalAveragePooling1D(
            name=f"{self.name}_embedding"
        )(x)

    def build_feature_extractor(self) -> tf.keras.Model:
        """Build the RNN stack without a final classification head.

        Returns
        -------
        tf.keras.Model
            Model mapping ``(batch, timesteps, n_features)`` to a 2D
            sequence-level embedding ``(batch, embedding_features)``.

        Notes
        -----
        This is the form that should be passed to the joint autoencoder +
        variational-classifier architecture. It is intentionally not compiled
        and does not create a dense, softmax, or variational classification
        layer.
        """
        x_in = layers.Input(
            shape=(self.timesteps, self.n_features),
            name=f"{self.name}_feature_input",
        )
        x = self.recurrent_stack(x_in, return_sequences=True)
        embedding = self.temporal_embedding(x)

        return Model(
            inputs=x_in,
            outputs=embedding,
            name=f"{self.name}_feature_extractor",
        )

    def classifier_head(
        self,
        embedding: tf.Tensor,
    ) -> tuple[tf.Tensor, VariationalClassifier | None]:
        """Attach the standalone standard or variational classifier head."""
        if self.uses_variational_head():
            vc_head = VariationalClassifier(
                n_classes=self.n_classes,
                name=f"{self.name}_variational_classifier",
            )
            logits = vc_head(embedding)
            probabilities = layers.Softmax(
                name=f"{self.name}_class_probabilities"
            )(logits)
            return probabilities, vc_head

        logits = layers.Dense(
            self.n_classes,
            name=f"{self.name}_class_logits",
        )(embedding)
        probabilities = layers.Softmax(
            name=f"{self.name}_class_probabilities"
        )(logits)
        return probabilities, None

    def build(self) -> tf.keras.Model:
        """Build and compile a complete standalone RNN classifier."""
        x_in = layers.Input(
            shape=(self.timesteps, self.n_features),
            name=f"{self.name}_input",
        )
        x = self.recurrent_stack(x_in, return_sequences=True)
        embedding = self.temporal_embedding(x)
        output, vc_head = self.classifier_head(embedding)

        model = Model(
            inputs=x_in,
            outputs=output,
            name=self.name,
        )

        if vc_head is not None:
            model.compile(
                optimizer=self.optimizer,
                loss=vc_head.keras_loss(
                    alpha=self.alpha,
                    beta=self.beta,
                    gamma=self.gamma,
                    lambda_=self.lambda_,
                ),
                metrics=self.metrics,
            )
            return model

        model.compile(
            optimizer=self.optimizer,
            loss=self.resolve_loss(self.loss),
            metrics=self.metrics,
        )
        return model

    def __call__(self) -> tf.keras.Model:
        """Allow builder instances to create standalone models when called."""
        return self.build()


class LSTMClassifier(RNNClassifier):
    """Unidirectional LSTM classifier for EEG sequence data."""

    def __init__(
        self,
        timesteps: int | None,
        n_features: int,
        n_classes: int,
        lstm_units: int | Sequence[int] = 128,
        n_lstm_layers: int = 2,
        dropout: float = 0.10,
        loss: LossLike = "softmax_crossentropy",
        optimizer: OptimizerLike = "adam",
        metrics: list[str] | None = None,
        name: str = "lstm_classifier",
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.0,
        lambda_: float = 0.0,
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
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            lambda_=lambda_,
        )
        self.lstm_units = self.rnn_units_by_layer
        self.n_lstm_layers = int(n_lstm_layers)

    def recurrent_layer(
        self,
        layer_index: int,
        *,
        return_sequences: bool,
    ) -> tf.keras.layers.Layer:
        return layers.LSTM(
            self.rnn_units_by_layer[layer_index],
            return_sequences=return_sequences,
            name=f"{self.name}_lstm_{layer_index}",
        )


class BiLSTMClassifier(RNNClassifier):
    """Bidirectional LSTM classifier for EEG sequence data."""

    def __init__(
        self,
        timesteps: int | None,
        n_features: int,
        n_classes: int,
        lstm_units: int | Sequence[int] = 128,
        n_bilstm_layers: int = 2,
        dropout: float = 0.10,
        loss: LossLike = "softmax_crossentropy",
        optimizer: OptimizerLike = "adam",
        metrics: list[str] | None = None,
        name: str = "bilstm_classifier",
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.0,
        lambda_: float = 0.0,
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
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            lambda_=lambda_,
        )
        self.lstm_units = self.rnn_units_by_layer
        self.n_bilstm_layers = int(n_bilstm_layers)

    def recurrent_layer(
        self,
        layer_index: int,
        *,
        return_sequences: bool,
    ) -> tf.keras.layers.Layer:
        return layers.Bidirectional(
            layers.LSTM(
                self.rnn_units_by_layer[layer_index],
                return_sequences=return_sequences,
                name=f"{self.name}_lstm_{layer_index}",
            ),
            merge_mode="concat",
            name=f"{self.name}_bilstm_{layer_index}",
        )


class GRUClassifier(RNNClassifier):
    """Unidirectional GRU classifier for EEG sequence data."""

    def __init__(
        self,
        timesteps: int | None,
        n_features: int,
        n_classes: int,
        gru_units: int | Sequence[int] = 128,
        n_gru_layers: int = 2,
        dropout: float = 0.10,
        loss: LossLike = "softmax_crossentropy",
        optimizer: OptimizerLike = "adam",
        metrics: list[str] | None = None,
        name: str = "gru_classifier",
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.0,
        lambda_: float = 0.0,
    ) -> None:
        super().__init__(
            timesteps=timesteps,
            n_features=n_features,
            n_classes=n_classes,
            rnn_units=gru_units,
            n_rnn_layers=n_gru_layers,
            dropout=dropout,
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
            name=name,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            lambda_=lambda_,
        )
        self.gru_units = self.rnn_units_by_layer
        self.n_gru_layers = int(n_gru_layers)

    def recurrent_layer(
        self,
        layer_index: int,
        *,
        return_sequences: bool,
    ) -> tf.keras.layers.Layer:
        return layers.GRU(
            self.rnn_units_by_layer[layer_index],
            return_sequences=return_sequences,
            name=f"{self.name}_gru_{layer_index}",
        )


class BiGRUClassifier(RNNClassifier):
    """Bidirectional GRU classifier for EEG sequence data."""

    def __init__(
        self,
        timesteps: int | None,
        n_features: int,
        n_classes: int,
        gru_units: int | Sequence[int] = 128,
        n_bigru_layers: int = 2,
        dropout: float = 0.10,
        loss: LossLike = "softmax_crossentropy",
        optimizer: OptimizerLike = "adam",
        metrics: list[str] | None = None,
        name: str = "bigru_classifier",
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.0,
        lambda_: float = 0.0,
    ) -> None:
        super().__init__(
            timesteps=timesteps,
            n_features=n_features,
            n_classes=n_classes,
            rnn_units=gru_units,
            n_rnn_layers=n_bigru_layers,
            dropout=dropout,
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
            name=name,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            lambda_=lambda_,
        )
        self.gru_units = self.rnn_units_by_layer
        self.n_bigru_layers = int(n_bigru_layers)

    def recurrent_layer(
        self,
        layer_index: int,
        *,
        return_sequences: bool,
    ) -> tf.keras.layers.Layer:
        return layers.Bidirectional(
            layers.GRU(
                self.rnn_units_by_layer[layer_index],
                return_sequences=return_sequences,
                name=f"{self.name}_gru_{layer_index}",
            ),
            merge_mode="concat",
            name=f"{self.name}_bigru_{layer_index}",
        )
