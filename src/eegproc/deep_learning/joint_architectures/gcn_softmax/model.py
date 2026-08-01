"""Deterministic GCN-only classifier for EEG emotion recognition.

The network reuses EEGProc's existing :class:`GCNEncoder`, pools its temporal
sequence, and produces two or more class logits. Cross-entropy is computed from
logits for numerical stability; softmax is applied only when probabilities are
requested by EEGProc's cross-validation utilities.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import prod

import tensorflow as tf
from tensorflow.keras import layers

from eegproc.deep_learning.unsupervised.Convolutions.GCN import GCNEncoder


def _as_positive_int_tuple(values: Sequence[int], name: str) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result or any(value <= 0 for value in result):
        raise ValueError(f"{name} must contain positive integers; got {values!r}.")
    return result


def build_gcn_softmax_classifier(
    input_shape: tuple[int, int],
    *,
    n_classes: int = 2,
    n_channels: int = 14,
    n_bands: int = 4,
    t_down: int = 2,
    gcn_units: Sequence[int] = (32, 16),
    temporal_pool_sizes: Sequence[int] = (2,),
    emb_dim: int = 32,
    dropout: float = 0.30,
    activation: str = "relu",
    use_batch_norm: bool = False,
    temporal_readout: str = "mean_max",
    classifier_units: int = 64,
    classifier_dropout: float = 0.50,
    l2_weight: float = 1e-4,
    learning_rate: float = 1e-4,
    clipnorm: float | None = 1.0,
    name: str = "gcn_softmax_classifier",
) -> tf.keras.Model:
    """Build and compile a GCN-only EEG classifier.

    Parameters
    ----------
    input_shape:
        ``(timesteps, n_channels * n_bands)``.
    temporal_readout:
        ``"mean"``, ``"max"``, or ``"mean_max"``. The default concatenates
        global mean and max temporal summaries.
    classifier_units:
        Width of the optional dense classification layer. Set to ``0`` for a
        linear GCN-to-logits baseline.

    Returns
    -------
    tf.keras.Model
        A compiled model returning unnormalized class logits.
    """
    if len(input_shape) != 2:
        raise ValueError(
            "GCN-only classification expects input_shape=(timesteps, features); "
            f"got {input_shape!r}."
        )

    timesteps, n_features = (int(input_shape[0]), int(input_shape[1]))
    if timesteps <= 0 or n_features <= 0:
        raise ValueError(f"input_shape must be positive; got {input_shape!r}.")
    if n_classes < 2:
        raise ValueError(f"n_classes must be at least 2; got {n_classes}.")
    if n_channels <= 0 or n_bands <= 0:
        raise ValueError("n_channels and n_bands must be positive.")
    if n_features != n_channels * n_bands:
        raise ValueError(
            f"Input has {n_features} features, but n_channels*n_bands is "
            f"{n_channels}*{n_bands}={n_channels * n_bands}."
        )
    if not 0.0 <= float(dropout) < 1.0:
        raise ValueError("dropout must be in [0, 1).")
    if not 0.0 <= float(classifier_dropout) < 1.0:
        raise ValueError("classifier_dropout must be in [0, 1).")
    if classifier_units < 0:
        raise ValueError("classifier_units must be >= 0.")
    if l2_weight < 0.0:
        raise ValueError("l2_weight must be >= 0.")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")

    gcn_units = _as_positive_int_tuple(gcn_units, "gcn_units")
    temporal_pool_sizes = _as_positive_int_tuple(
        temporal_pool_sizes, "temporal_pool_sizes"
    )
    if int(t_down) != prod(temporal_pool_sizes):
        raise ValueError(
            "t_down must equal the product of temporal_pool_sizes; got "
            f"t_down={t_down}, temporal_pool_sizes={temporal_pool_sizes}."
        )

    normalized_readout = temporal_readout.lower().replace("+", "_")
    if normalized_readout not in {"mean", "max", "mean_max"}:
        raise ValueError(
            "temporal_readout must be 'mean', 'max', or 'mean_max'; got "
            f"{temporal_readout!r}."
        )

    inputs = tf.keras.Input(shape=input_shape, name="eeg_window")
    encoder = GCNEncoder(
        timesteps=timesteps,
        t_down=int(t_down),
        n_channels=int(n_channels),
        n_bands=int(n_bands),
        gcn_units=gcn_units,
        temporal_pool_sizes=temporal_pool_sizes,
        emb_dim=int(emb_dim),
        dropout=float(dropout),
        activation=activation,
        use_batch_norm=bool(use_batch_norm),
        name="gcn_encoder",
    )
    sequence_features = encoder(inputs)

    if normalized_readout == "mean":
        features = layers.GlobalAveragePooling1D(name="temporal_mean")(sequence_features)
    elif normalized_readout == "max":
        features = layers.GlobalMaxPooling1D(name="temporal_max")(sequence_features)
    else:
        mean_features = layers.GlobalAveragePooling1D(name="temporal_mean")(
            sequence_features
        )
        max_features = layers.GlobalMaxPooling1D(name="temporal_max")(
            sequence_features
        )
        features = layers.Concatenate(name="temporal_mean_max")(
            [mean_features, max_features]
        )

    regularizer = (
        tf.keras.regularizers.L2(float(l2_weight)) if l2_weight > 0.0 else None
    )

    if classifier_units > 0:
        features = layers.Dense(
            int(classifier_units),
            activation=activation,
            kernel_regularizer=regularizer,
            name="classifier_dense",
        )(features)
        features = layers.Dropout(
            float(classifier_dropout), name="classifier_dropout"
        )(features)
    elif classifier_dropout > 0.0:
        features = layers.Dropout(
            float(classifier_dropout), name="linear_classifier_dropout"
        )(features)

    logits = layers.Dense(
        int(n_classes),
        activation=None,
        kernel_regularizer=regularizer,
        name="class_logits",
    )(features)

    model = tf.keras.Model(inputs=inputs, outputs=logits, name=name)
    optimizer_kwargs: dict[str, float] = {"learning_rate": float(learning_rate)}
    if clipnorm is not None:
        if float(clipnorm) <= 0.0:
            raise ValueError("clipnorm must be positive or None.")
        optimizer_kwargs["clipnorm"] = float(clipnorm)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(**optimizer_kwargs),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ],
    )
    return model
