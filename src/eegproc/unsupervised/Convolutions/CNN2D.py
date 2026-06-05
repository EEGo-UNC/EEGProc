from collections.abc import Sequence

import tensorflow as tf
from tensorflow.keras import layers

from ..BaseEncoder import BaseEncoder
from utils import _ensure_tuple, _product


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



class CNN2DDecoder(tf.keras.Model):
    """Temporal decoder for ``CNN2DEncoder`` latent sequences.

    This decoder reconstructs flattened EEG feature sequences from the latent
    output of ``CNN2DEncoder``. It expects latent inputs of shape
    ``(batch, ceil(timesteps / t_down), emb_dim)`` and reconstructs outputs of
    shape ``(batch, timesteps, n_channels * n_bands)``.

    This is not a true spatial 2D mirror decoder because ``CNN2DEncoder``
    collapses the channel-band grid using ``GlobalAveragePooling2D`` before
    producing the final latent sequence. Therefore, the decoder mirrors the
    temporal compression but reconstructs the final EEG feature vector as a
    flat channel-band representation.

    Parameters
    ----------
    timesteps : int
        Number of timesteps in the original input sequence.
    n_channels : int
        Number of EEG electrode channels.
    n_bands : int
        Number of frequency-band features per EEG channel.
    t_down : int
        Temporal downsampling factor used by the encoder.
    conv_filters : tuple[int, ...]
        Filter schedule from the corresponding ``CNN2DEncoder``. The decoder
        uses the reversed filter schedule for temporal reconstruction.
    temporal_pool_sizes : tuple[int, ...]
        Temporal pooling sizes used by the encoder. The decoder mirrors these
        with ``UpSampling1D`` in reverse order.
    emb_dim : int, default=128
        Dimensionality of the input latent embedding.
    dropout : float, default=0.10
        Dropout rate used after temporal decoder blocks.
    activation : str, default="relu"
        Activation function used in temporal convolutional blocks.
    use_batch_norm : bool, default=True
        Whether to apply batch normalization after decoder convolutions.
    name : str, default="decoder_2dcnn"
        Name of the Keras model.
    **kwargs
        Additional keyword arguments passed to ``tf.keras.Model``.

    Input shape
    -----------
    ``(batch, ceil(timesteps / t_down), emb_dim)``

    Output shape
    ------------
    ``(batch, timesteps, n_channels * n_bands)``
    """

    def __init__(
        self,
        timesteps: int,
        n_channels: int,
        n_bands: int,
        t_down: int,
        conv_filters: tuple[int, ...],
        temporal_pool_sizes: tuple[int, ...],
        emb_dim: int = 128,
        dropout: float = 0.10,
        activation: str = "relu",
        use_batch_norm: bool = True,
        name: str = "decoder_2dcnn",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        if timesteps <= 0:
            raise ValueError(f"timesteps must be positive, got {timesteps}.")
        if n_channels <= 0:
            raise ValueError(f"n_channels must be positive, got {n_channels}.")
        if n_bands <= 0:
            raise ValueError(f"n_bands must be positive, got {n_bands}.")
        if t_down <= 0:
            raise ValueError(f"t_down must be positive, got {t_down}.")
        if len(conv_filters) == 0:
            raise ValueError("conv_filters must contain at least one layer.")

        self.timesteps = timesteps
        self.n_channels = n_channels
        self.n_bands = n_bands
        self.t_down = t_down
        self.conv_filters = tuple(conv_filters)
        self.temporal_pool_sizes = tuple(temporal_pool_sizes)
        self.emb_dim = emb_dim
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

        self.input_projection = layers.Conv1D(
            self.conv_filters[-1],
            1,
            padding="same",
            activation=activation,
            name="dec_input_projection",
        )

        self.upsample_layers = [
            layers.UpSampling1D(
                size=pool_size,
                name=f"dec_upsample_{i}",
            )
            for i, pool_size in enumerate(reversed(self.temporal_pool_sizes))
        ]

        reversed_filters = tuple(reversed(self.conv_filters))

        self.conv_layers = [
            layers.Conv1D(
                filters,
                3,
                padding="same",
                activation=activation,
                name=f"dec_conv1d_{i}",
            )
            for i, filters in enumerate(reversed_filters)
        ]

        self.bn_layers = [
            layers.BatchNormalization(name=f"dec_bn1d_{i}") if use_batch_norm else None
            for i, _ in enumerate(reversed_filters)
        ]

        self.dropout_layers = [
            layers.Dropout(
                dropout,
                name=f"dec_do1d_{i}",
            )
            for i, _ in enumerate(reversed_filters)
        ]

        self.x_hat = layers.Conv1D(
            self.n_features,
            1,
            padding="same",
            activation=None,
            name="x_hat",
        )

    @property
    def n_features(self) -> int:
        """Flattened number of reconstructed channel-band features."""
        return self.n_channels * self.n_bands

    @classmethod
    def from_encoder(
        cls,
        encoder: CNN2DEncoder,
        name: str = "decoder_2dcnn",
    ) -> "CNN2DDecoder":
        """Create a decoder from a configured ``CNN2DEncoder``.

        Parameters
        ----------
        encoder : CNN2DEncoder
            Configured 2D CNN encoder.
        name : str, default="decoder_2dcnn"
            Name of the decoder model.

        Returns
        -------
        CNN2DDecoder
            Decoder whose temporal reconstruction schedule mirrors the encoder.
        """
        if not isinstance(encoder, CNN2DEncoder):
            raise TypeError(
                "CNN2DDecoder.from_encoder only supports CNN2DEncoder. "
                f"Got {type(encoder).__name__}."
            )

        return cls(
            timesteps=encoder.timesteps,
            n_channels=encoder.n_channels,
            n_bands=encoder.n_bands,
            t_down=encoder.t_down,
            conv_filters=encoder.conv_filters,
            temporal_pool_sizes=encoder.temporal_pool_sizes,
            emb_dim=encoder.emb_dim,
            dropout=encoder.dropout_rate,
            activation=encoder.activation,
            use_batch_norm=encoder.use_batch_norm,
            name=name,
        )

    def fix_length(self, x: tf.Tensor) -> tf.Tensor:
        """Trim or pad the reconstructed sequence to exactly ``timesteps``."""
        x = x[:, : self.timesteps, :]

        current_timesteps = tf.shape(x)[1]
        pad_amount = tf.maximum(0, self.timesteps - current_timesteps)

        return tf.pad(
            x,
            paddings=[[0, 0], [0, pad_amount], [0, 0]],
        )

    def call(self, inputs, training: bool = False):
        """Run the decoder forward pass."""
        x = self.input_projection(inputs)

        for upsample in self.upsample_layers:
            x = upsample(x)

        for conv, bn, dropout in zip(
            self.conv_layers,
            self.bn_layers,
            self.dropout_layers,
        ):
            x = conv(x)

            if bn is not None:
                x = bn(x, training=training)

            x = dropout(x, training=training)

        x = self.x_hat(x)

        return self.fix_length(x)

    def compute_output_shape(self, input_shape):
        """Return the decoder output shape."""
        return (input_shape[0], self.timesteps, self.n_features)

    def get_config(self) -> dict:
        """Return serializable configuration for the decoder."""
        config = super().get_config()
        config.update(
            {
                "timesteps": self.timesteps,
                "n_channels": self.n_channels,
                "n_bands": self.n_bands,
                "t_down": self.t_down,
                "conv_filters": self.conv_filters,
                "temporal_pool_sizes": self.temporal_pool_sizes,
                "emb_dim": self.emb_dim,
                "dropout": self.dropout_rate,
                "activation": self.activation,
                "use_batch_norm": self.use_batch_norm,
                "name": self.name,
            }
        )
        return config
