import math
from collections.abc import Sequence

import tensorflow as tf
from tensorflow.keras import layers

from .BaseEncoder import BaseEncoder


def _ensure_tuple(value, n_layers: int, name: str):
    """Normalize scalar or sequence configuration to a tuple of length n_layers."""
    if isinstance(value, Sequence) and not isinstance(value, str):
        value = tuple(value)
        if len(value) != n_layers:
            raise ValueError(
                f"{name} must have length {n_layers}, got length {len(value)}."
            )
        return value

    return tuple(value for _ in range(n_layers))


def _product(values: Sequence[int]) -> int:
    """Return the product of a sequence of integers."""
    result = 1
    for value in values:
        result *= value
    return result


class CNN1DEncoder(BaseEncoder):
    """Configurable-depth 1D convolutional encoder for EEG sequence data.

    This encoder treats each timestep's feature vector as a flat representation
    and learns temporal patterns using a designer-defined stack of ``Conv1D``
    layers.

    The number of convolutional layers is determined by ``len(conv_filters)``.
    Temporal downsampling is controlled by ``pool_after_layers`` and
    ``pool_sizes``. The product of the pooling sizes must equal ``t_down``.

    Parameters
    ----------
    timesteps : int
        Number of timesteps in each input sequence.
    n_features : int
        Number of features per timestep.
    t_down : int
        Temporal downsampling factor.
    conv_filters : tuple[int, ...], default=(64, 128, 256)
        Number of filters for each Conv1D layer. The length of this tuple
        defines the number of convolutional layers.
    kernel_sizes : int or tuple[int, ...], default=(7, 5, 3)
        Kernel size for each Conv1D layer. If an int is given, the same kernel
        size is used for every layer.
    pool_after_layers : tuple[int, ...], default=(0, 1)
        Zero-indexed convolutional layer indices after which ``MaxPool1D`` is
        applied.
    pool_sizes : int or tuple[int, ...], default=2
        Pool size for each pooling operation. If an int is given, the same pool
        size is used after every layer listed in ``pool_after_layers``.
    emb_dim : int, default=128
        Dimensionality of the latent embedding at each output timestep.
    dropout : float, default=0.10
        Dropout rate applied after each convolutional block.
    activation : str, default="relu"
        Activation function used by convolutional layers.
    use_batch_norm : bool, default=True
        Whether to apply batch normalization after each convolution.
    name : str, default="encoder_1dcnn"
        Name of the Keras model.
    **kwargs
        Additional keyword arguments passed to ``tf.keras.Model``.

    Input shape
    -----------
    ``(batch, timesteps, n_features)``

    Output shape
    ------------
    ``(batch, ceil(timesteps / t_down), emb_dim)``
    """

    def __init__(
        self,
        timesteps: int,
        n_features: int,
        t_down: int,
        conv_filters: tuple[int, ...] = (64, 128, 256),
        kernel_sizes: int | tuple[int, ...] = (7, 5, 3),
        pool_after_layers: tuple[int, ...] = (0, 1),
        pool_sizes: int | tuple[int, ...] = 3,
        emb_dim: int = 128,
        dropout: float = 0.10,
        activation: str = "relu",
        use_batch_norm: bool = True,
        name: str = "encoder_1dcnn",
        **kwargs,
    ):
        super().__init__(
            timesteps=timesteps,
            emb_dim=emb_dim,
            t_down=t_down,
            name=name,
            **kwargs,
        )

        if len(conv_filters) == 0:
            raise ValueError("conv_filters must contain at least one layer.")

        self._n_features = n_features
        self.conv_filters = tuple(conv_filters)
        self.kernel_sizes = _ensure_tuple(
            kernel_sizes,
            len(self.conv_filters),
            "kernel_sizes",
        )
        self.pool_after_layers = tuple(pool_after_layers)
        self.pool_sizes = _ensure_tuple(
            pool_sizes,
            len(self.pool_after_layers),
            "pool_sizes",
        )
        self.dropout_rate = dropout
        self.activation = activation
        self.use_batch_norm = use_batch_norm

        if any(i < 0 or i >= len(self.conv_filters) for i in self.pool_after_layers):
            raise ValueError(
                "pool_after_layers contains an invalid layer index. "
                f"Valid indices are 0 to {len(self.conv_filters) - 1}."
            )

        effective_t_down = _product(self.pool_sizes)
        if effective_t_down != self.t_down:
            raise ValueError(
                f"t_down={self.t_down}, but the configured pooling produces "
                f"a downsampling factor of {effective_t_down}. "
                "Set t_down equal to product(pool_sizes)."
            )

        pool_size_by_layer = dict(zip(self.pool_after_layers, self.pool_sizes))

        self.conv_layers = []
        self.bn_layers = []
        self.pool_layers = []
        self.dropout_layers = []

        for i, (filters, kernel_size) in enumerate(
            zip(self.conv_filters, self.kernel_sizes)
        ):
            self.conv_layers.append(
                layers.Conv1D(
                    filters,
                    kernel_size,
                    padding="same",
                    activation=activation,
                    name=f"enc_conv1d_{i}",
                )
            )

            self.bn_layers.append(
                layers.BatchNormalization(name=f"enc_bn1d_{i}")
                if use_batch_norm
                else None
            )

            self.pool_layers.append(
                layers.MaxPool1D(
                    pool_size_by_layer[i],
                    padding="same",
                    name=f"enc_pool1d_{i}",
                )
                if i in pool_size_by_layer
                else None
            )

            self.dropout_layers.append(
                layers.Dropout(dropout, name=f"enc_do1d_{i}")
            )

        self.seq_emb = layers.Conv1D(
            emb_dim,
            1,
            padding="same",
            activation=None,
            name="seq_emb",
        )

    @property
    def n_features(self) -> int:
        """Number of input features per timestep."""
        return self._n_features

    def call(self, inputs, training: bool = False):
        """Run the configurable-depth 1D CNN encoder forward pass."""
        x = inputs

        for conv, bn, pool, dropout in zip(
            self.conv_layers,
            self.bn_layers,
            self.pool_layers,
            self.dropout_layers,
        ):
            x = conv(x)

            if bn is not None:
                x = bn(x, training=training)

            if pool is not None:
                x = pool(x)

            x = dropout(x, training=training)

        return self.seq_emb(x)

    def get_config(self) -> dict:
        """Return serializable configuration for the encoder."""
        config = super().get_config()
        config.update(
            {
                "n_features": self.n_features,
                "conv_filters": self.conv_filters,
                "kernel_sizes": self.kernel_sizes,
                "pool_after_layers": self.pool_after_layers,
                "pool_sizes": self.pool_sizes,
                "dropout": self.dropout_rate,
                "activation": self.activation,
                "use_batch_norm": self.use_batch_norm,
            }
        )
        return config


class CNN2DEncoder(BaseEncoder):
    """Configurable-depth 2D convolutional encoder over EEG channel-band grids.

    This encoder accepts flattened EEG features at each timestep and reshapes
    them into a 2D ``channels x bands`` grid. Each timestep is processed by a
    designer-defined stack of ``TimeDistributed(Conv2D)`` layers.

    The number of spatial convolutional layers is determined by
    ``len(conv_filters)``. Temporal downsampling is controlled by
    ``temporal_pool_sizes``. The product of ``temporal_pool_sizes`` must equal
    ``t_down``.

    Parameters
    ----------
    timesteps : int
        Number of timesteps in each input sequence.
    t_down : int
        Temporal downsampling factor.
    n_channels : int, default=14
        Number of EEG electrode channels.
    n_bands : int, default=6
        Number of frequency-band features per EEG channel.
    conv_filters : tuple[int, ...], default=(32, 64, 128)
        Number of filters for each Conv2D layer. The length of this tuple
        defines the number of spatial convolutional layers.
    kernel_sizes : tuple[int, int] or tuple[tuple[int, int], ...],
        default=((3, 3), (3, 3), (2, 2))
        Kernel size for each Conv2D layer. If one tuple like ``(3, 3)`` is
        given, it is reused for every layer.
    temporal_pool_sizes : tuple[int, ...], default=(2, 2)
        Temporal pooling operations applied after spatial pooling. For example,
        ``(2, 2)`` gives ``t_down=4``.
    emb_dim : int, default=128
        Dimensionality of the latent embedding at each output timestep.
    dropout : float, default=0.10
        Dropout rate applied after each spatial convolutional block and after
        each temporal pooling layer.
    activation : str, default="relu"
        Activation function used by convolutional layers.
    use_batch_norm : bool, default=True
        Whether to apply batch normalization after each convolution.
    name : str, default="encoder_2dcnn"
        Name of the Keras model.
    **kwargs
        Additional keyword arguments passed to ``tf.keras.Model``.

    Input shape
    -----------
    ``(batch, timesteps, n_channels * n_bands)``

    Output shape
    ------------
    ``(batch, ceil(timesteps / t_down), emb_dim)``
    """

    def __init__(
        self,
        timesteps: int,
        t_down: int,
        n_channels: int = 14,
        n_bands: int = 6,
        conv_filters: tuple[int, ...] = (32, 64, 128),
        kernel_sizes: tuple[int, int] | tuple[tuple[int, int], ...] = (
            (3, 3),
            (3, 3),
            (2, 2),
        ),
        temporal_pool_sizes: tuple[int, ...] = (2, 2),
        emb_dim: int = 128,
        dropout: float = 0.10,
        activation: str = "relu",
        use_batch_norm: bool = True,
        name: str = "encoder_2dcnn",
        **kwargs,
    ):
        super().__init__(
            timesteps=timesteps,
            emb_dim=emb_dim,
            t_down=t_down,
            name=name,
            **kwargs,
        )

        if len(conv_filters) == 0:
            raise ValueError("conv_filters must contain at least one layer.")

        self.n_channels = n_channels
        self.n_bands = n_bands
        self.conv_filters = tuple(conv_filters)
        self.kernel_sizes = self._normalize_2d_kernel_sizes(
            kernel_sizes,
            len(self.conv_filters),
        )
        self.temporal_pool_sizes = tuple(temporal_pool_sizes)
        self.dropout_rate = dropout
        self.activation = activation
        self.use_batch_norm = use_batch_norm

        effective_t_down = _product(self.temporal_pool_sizes)
        if effective_t_down != self.t_down:
            raise ValueError(
                f"t_down={self.t_down}, but temporal_pool_sizes produces "
                f"a downsampling factor of {effective_t_down}. "
                "Set t_down equal to product(temporal_pool_sizes)."
            )

        self.to_grid = layers.Reshape(
            (timesteps, n_channels, n_bands, 1),
            name="to_grid",
        )

        self.conv_layers = []
        self.bn_layers = []
        self.dropout_layers = []

        for i, (filters, kernel_size) in enumerate(
            zip(self.conv_filters, self.kernel_sizes)
        ):
            self.conv_layers.append(
                layers.TimeDistributed(
                    layers.Conv2D(
                        filters,
                        kernel_size,
                        padding="same",
                        activation=activation,
                    ),
                    name=f"enc_conv2d_{i}",
                )
            )

            self.bn_layers.append(
                layers.TimeDistributed(
                    layers.BatchNormalization(),
                    name=f"enc_bn2d_{i}",
                )
                if use_batch_norm
                else None
            )

            self.dropout_layers.append(
                layers.TimeDistributed(
                    layers.Dropout(dropout),
                    name=f"enc_do2d_{i}",
                )
            )

        self.gap2d = layers.TimeDistributed(
            layers.GlobalAveragePooling2D(),
            name="enc_gap2d",
        )

        self.temporal_pool_layers = [
            layers.MaxPool1D(
                pool_size,
                padding="same",
                name=f"enc_tpool_{i}",
            )
            for i, pool_size in enumerate(self.temporal_pool_sizes)
        ]

        self.temporal_dropout_layers = [
            layers.Dropout(
                dropout,
                name=f"enc_tdo_{i}",
            )
            for i, _ in enumerate(self.temporal_pool_sizes)
        ]

        self.seq_emb = layers.Conv1D(
            emb_dim,
            1,
            padding="same",
            activation=None,
            name="seq_emb",
        )

    @staticmethod
    def _normalize_2d_kernel_sizes(kernel_sizes, n_layers: int):
        """Normalize 2D kernel-size configuration to one tuple per layer."""
        if (
            isinstance(kernel_sizes, tuple)
            and len(kernel_sizes) == 2
            and all(isinstance(v, int) for v in kernel_sizes)
        ):
            return tuple(kernel_sizes for _ in range(n_layers))

        kernel_sizes = tuple(kernel_sizes)

        if len(kernel_sizes) != n_layers:
            raise ValueError(
                f"kernel_sizes must have length {n_layers}, "
                f"got length {len(kernel_sizes)}."
            )

        return kernel_sizes

    @property
    def n_features(self) -> int:
        """Flattened number of channel-band features per timestep."""
        return self.n_channels * self.n_bands

    def call(self, inputs, training: bool = False):
        """Run the configurable-depth 2D CNN encoder forward pass."""
        x = self.to_grid(inputs)

        for conv, bn, dropout in zip(
            self.conv_layers,
            self.bn_layers,
            self.dropout_layers,
        ):
            x = conv(x)

            if bn is not None:
                x = bn(x, training=training)

            x = dropout(x, training=training)

        x = self.gap2d(x)

        for pool, dropout in zip(
            self.temporal_pool_layers,
            self.temporal_dropout_layers,
        ):
            x = pool(x)
            x = dropout(x, training=training)

        return self.seq_emb(x)

    def get_config(self) -> dict:
        """Return serializable configuration for the encoder."""
        config = super().get_config()
        config.update(
            {
                "n_channels": self.n_channels,
                "n_bands": self.n_bands,
                "conv_filters": self.conv_filters,
                "kernel_sizes": self.kernel_sizes,
                "temporal_pool_sizes": self.temporal_pool_sizes,
                "dropout": self.dropout_rate,
                "activation": self.activation,
                "use_batch_norm": self.use_batch_norm,
            }
        )
        return config