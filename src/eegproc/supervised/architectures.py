"""Supervised classifier architectures for EEG sequence data.

These factories build *compiled* Keras models ready for training and
cross-validation. The loss function is a parameter so callers can swap
losses without rewriting the model — e.g. the default softmax
cross-entropy today, a variational loss once it is implemented.
"""

from __future__ import annotations

from typing import Callable, Union

import tensorflow as tf
from tensorflow.keras import layers, Model


LossLike = Union[str, Callable, tf.keras.losses.Loss]


def _resolve_loss(loss: LossLike) -> Union[str, Callable, tf.keras.losses.Loss]:
    """Map friendly loss aliases to Keras-compatible losses.

    ``"softmax_crossentropy"`` resolves to sparse categorical crossentropy
    with ``from_logits=False`` (the model output is already softmaxed).
    ``"variational"`` raises ``NotImplementedError`` to lock in the API
    contract until the variational loss work lands. Any other value is
    passed through to ``model.compile`` unchanged.
    """
    if loss == "softmax_crossentropy":
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    if loss == "variational":
        raise NotImplementedError(
            "Variational loss is not yet implemented; "
            "use 'softmax_crossentropy' until it is."
        )
    return loss


def lstm_classifier(
    timesteps: int,
    n_features: int,
    n_classes: int,
    lstm_units: int = 128,
    n_lstm_layers: int = 2,
    dropout: float = 0.10,
    loss: LossLike = "softmax_crossentropy",
    optimizer: Union[str, tf.keras.optimizers.Optimizer] = "adam",
    metrics: list[str] | None = None,
) -> tf.keras.Model:
    """Build a compiled unidirectional LSTM classifier.

    Stacks ``n_lstm_layers`` LSTM layers that pass the full sequence
    forward through time, compresses the time axis by 4 with two
    ``MaxPool1D`` layers, then ``GlobalAveragePooling1D`` collapses to
    one vector and a ``Dense(n_classes)`` + ``Softmax`` produces class
    probabilities.

    Parameters
    ----------
    timesteps : int
        Number of timesteps T in the input sequence.
    n_features : int
        Number of features per timestep (e.g. 84 for 14 electrodes x 6 bands).
    n_classes : int
        Number of output classes.
    lstm_units : int, optional
        Hidden units in each LSTM layer. Default 128.
    n_lstm_layers : int, optional
        How many LSTM layers to stack. Default 2.
    dropout : float, optional
        Dropout rate after each LSTM layer. Default 0.10.
    loss : str or callable, optional
        ``"softmax_crossentropy"`` (default) maps to
        ``SparseCategoricalCrossentropy(from_logits=False)``.
        ``"variational"`` is reserved for future use and raises
        ``NotImplementedError``. Any other Keras-compatible loss string
        or callable is passed through.
    optimizer : str or tf.keras.optimizers.Optimizer, optional
        Default ``"adam"``.
    metrics : list[str] or None, optional
        Defaults to ``["accuracy"]``.

    Returns
    -------
    tf.keras.Model
        Compiled model with input ``(batch, timesteps, n_features)`` and
        output ``(batch, n_classes)`` of class probabilities.

    Examples
    --------
    >>> from eegproc.supervised import lstm_classifier
    >>> model = lstm_classifier(timesteps=128, n_features=84, n_classes=3)
    >>> # Already compiled — fit directly:
    >>> # model.fit(X_train, y_train, epochs=10)
    """
    resolved_loss = _resolve_loss(loss)

    x_in = layers.Input(shape=(timesteps, n_features), name="x")
    x = x_in
    for layer_index in range(n_lstm_layers):
        x = layers.LSTM(
            lstm_units,
            return_sequences=True,
            name=f"lstm_{layer_index}",
        )(x)
        x = layers.BatchNormalization(name=f"lstm_bn_{layer_index}")(x)
        x = layers.Dropout(dropout, name=f"lstm_do_{layer_index}")(x)

    x = layers.MaxPool1D(2, padding="same", name="enc_tpool1")(x)
    x = layers.MaxPool1D(2, padding="same", name="enc_tpool2")(x)

    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(n_classes, name="class_logits")(x)
    output = layers.Softmax(name="class_probabilities")(x)

    model = Model(inputs=x_in, outputs=output, name="lstm_classifier")
    model.compile(
        optimizer=optimizer,
        loss=resolved_loss,
        metrics=metrics if metrics is not None else ["accuracy"],
    )
    return model


def bilstm_classifier(
    timesteps: int,
    n_features: int,
    n_classes: int,
    lstm_units: int = 128,
    n_bilstm_layers: int = 2,
    dropout: float = 0.10,
    loss: LossLike = "softmax_crossentropy",
    optimizer: Union[str, tf.keras.optimizers.Optimizer] = "adam",
    metrics: list[str] | None = None,
) -> tf.keras.Model:
    """Build a compiled bidirectional LSTM classifier.

    Same structure as :func:`lstm_classifier` but each LSTM is wrapped
    in ``Bidirectional`` so the sequence is processed left-to-right and
    right-to-left simultaneously and the two outputs are concatenated.
    Each layer's feature width is therefore ``lstm_units * 2``.

    Parameters
    ----------
    timesteps, n_features, n_classes : int
        Same as :func:`lstm_classifier`.
    lstm_units : int, optional
        Hidden units in *each direction*. Default 128.
    n_bilstm_layers : int, optional
        How many bidirectional layers to stack. Default 2.
    dropout : float, optional
        Dropout rate after each BiLSTM layer. Default 0.10.
    loss, optimizer, metrics
        Same as :func:`lstm_classifier`.

    Returns
    -------
    tf.keras.Model
        Compiled model with input ``(batch, timesteps, n_features)`` and
        output ``(batch, n_classes)`` of class probabilities.
    """
    resolved_loss = _resolve_loss(loss)

    x_in = layers.Input(shape=(timesteps, n_features), name="x")
    x = x_in
    for layer_index in range(n_bilstm_layers):
        x = layers.Bidirectional(
            layers.LSTM(lstm_units, return_sequences=True),
            merge_mode="concat",
            name=f"bilstm_{layer_index}",
        )(x)
        x = layers.BatchNormalization(name=f"bilstm_bn_{layer_index}")(x)
        x = layers.Dropout(dropout, name=f"bilstm_do_{layer_index}")(x)

    x = layers.MaxPool1D(2, padding="same", name="enc_tpool1")(x)
    x = layers.MaxPool1D(2, padding="same", name="enc_tpool2")(x)

    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(n_classes, name="class_logits")(x)
    output = layers.Softmax(name="class_probabilities")(x)

    model = Model(inputs=x_in, outputs=output, name="bilstm_classifier")
    model.compile(
        optimizer=optimizer,
        loss=resolved_loss,
        metrics=metrics if metrics is not None else ["accuracy"],
    )
    return model
