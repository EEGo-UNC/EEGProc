from collections.abc import Sequence

import tensorflow as tf
from tensorflow.keras import layers

from ..BaseEncoder import BaseEncoder
from ..GraphConv import GraphConv
from ..utils import _product


class GCNEncoder(BaseEncoder):
    """Configurable-depth graph convolutional encoder for EEG sequence data.

    This encoder treats EEG electrodes as graph nodes and frequency-band values
    as node features. The expected input shape is
    ``(batch, timesteps, n_channels * n_bands)``. Each timestep is reshaped
    into a node-feature matrix of shape ``(n_channels, n_bands)``.

    The number of graph-convolution layers is determined by ``len(gcn_units)``.
    For example, ``gcn_units=(32, 64)`` creates two graph-convolution layers,
    while ``gcn_units=(32, 64, 128)`` creates three.

    Each ``GCN`` layer learns its own adjacency matrix, allowing the
    encoder to discover functional relationships among EEG channels. After
    graph feature extraction, the node dimension is pooled by averaging across
    channels. Temporal downsampling is then controlled by
    ``temporal_pool_sizes``. The product of ``temporal_pool_sizes`` must equal
    ``t_down``.

    Parameters
    ----------
    timesteps : int
        Number of timesteps in each input sequence.
    t_down : int
        Temporal downsampling factor.
    n_channels : int, default=14
        Number of EEG electrode channels. This is also the number of graph
        nodes.
    n_bands : int, default=6
        Number of frequency-band features per EEG channel.
    gcn_units : tuple[int, ...], default=(32, 64)
        Output dimensionality of each successive ``GCN`` layer. The
        length of this tuple determines the number of graph-convolution layers.
    temporal_pool_sizes : tuple[int, ...], default=(2, 2)
        Temporal pooling operations applied after graph node pooling. For
        example, ``(2, 2)`` gives ``t_down=4``.
    emb_dim : int, default=128
        Dimensionality of the latent embedding at each output timestep.
    dropout : float, default=0.10
        Dropout rate applied after each graph-convolution block and after each
        temporal pooling layer.
    activation : str, default="relu"
        Activation function used inside each ``GCN`` layer.
    use_batch_norm : bool, default=True
        Whether to apply batch normalization after each graph-convolution
        layer.
    name : str, default="encoder_gcn"
        Name of the Keras model.
    **kwargs
        Additional keyword arguments passed to ``tf.keras.Model``.

    Input shape
    -----------
    ``(batch, timesteps, n_channels * n_bands)``

    Output shape
    ------------
    ``(batch, ceil(timesteps / t_down), emb_dim)``

    Notes
    -----
    The learned adjacency matrix of the i-th graph layer can be accessed with:

    ``encoder.gcn_layers[i].layer.A_theta``

    because each ``GCN`` layer is wrapped inside ``TimeDistributed``.
    """

    def __init__(
        self,
        timesteps: int,
        t_down: int,
        n_channels: int = 14,
        n_bands: int = 6,
        gcn_units: tuple[int, ...] = (32, 64),
        temporal_pool_sizes: tuple[int, ...] = (2, 2),
        emb_dim: int = 128,
        dropout: float = 0.10,
        activation: str = "relu",
        use_batch_norm: bool = True,
        name: str = "encoder_gcn",
        **kwargs,
    ):
        super().__init__(
            timesteps=timesteps,
            emb_dim=emb_dim,
            t_down=t_down,
            name=name,
            **kwargs,
        )

        if len(gcn_units) == 0:
            raise ValueError("gcn_units must contain at least one layer.")

        self.n_channels = n_channels
        self.n_bands = n_bands
        self.gcn_units = tuple(gcn_units)
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

        self.to_nodes = layers.Reshape(
            (timesteps, n_channels, n_bands),
            name="to_nodes",
        )

        self.gcn_layers = []
        self.bn_layers = []
        self.dropout_layers = []

        for i, units in enumerate(self.gcn_units):
            self.gcn_layers.append(
                layers.TimeDistributed(
                    GraphConv(
                        units=units,
                        n_nodes=n_channels,
                        activation=activation,
                        name=f"graph_conv_{i}",
                    ),
                    name=f"gcn_{i}",
                )
            )

            self.bn_layers.append(
                layers.TimeDistributed(
                    layers.BatchNormalization(),
                    name=f"gcn_bn_{i}",
                )
                if use_batch_norm
                else None
            )

            self.dropout_layers.append(
                layers.TimeDistributed(
                    layers.Dropout(dropout),
                    name=f"gcn_do_{i}",
                )
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

        self.seq_emb = layers.Dense(
            emb_dim,
            activation=None,
            name="seq_emb",
        )

    @property
    def n_features(self) -> int:
        """Flattened number of channel-band features per timestep."""
        return self.n_channels * self.n_bands

    def call(self, inputs, training: bool = False):
        """Run the configurable-depth graph convolutional encoder forward pass."""
        x = self.to_nodes(inputs)

        for gcn, bn, dropout in zip(
            self.gcn_layers,
            self.bn_layers,
            self.dropout_layers,
        ):
            x = gcn(x, training=training)

            if bn is not None:
                x = bn(x, training=training)

            x = dropout(x, training=training)

        x = tf.reduce_mean(x, axis=2)

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
                "gcn_units": self.gcn_units,
                "temporal_pool_sizes": self.temporal_pool_sizes,
                "dropout": self.dropout_rate,
                "activation": self.activation,
                "use_batch_norm": self.use_batch_norm,
            }
        )
        return config


class GCNDecoder(tf.keras.Model):
    """Temporal decoder for ``GCNEncoder`` latent sequences.

    This decoder reconstructs flattened EEG feature sequences from the latent
    output of ``GCNEncoder``. It expects latent inputs of shape
    ``(batch, ceil(timesteps / t_down), emb_dim)`` and reconstructs outputs of
    shape ``(batch, timesteps, n_channels * n_bands)``.

    This is not a true graph mirror decoder because ``GCNEncoder`` pools away
    the node dimension using ``tf.reduce_mean(x, axis=2)`` before producing the
    final latent sequence. Therefore, the decoder mirrors the temporal
    compression but reconstructs the original EEG feature vector as a flattened
    channel-band representation.
    """

    def __init__(
        self,
        timesteps: int,
        n_channels: int,
        n_bands: int,
        t_down: int,
        gcn_units: tuple[int, ...],
        temporal_pool_sizes: tuple[int, ...],
        emb_dim: int = 128,
        dropout: float = 0.10,
        activation: str = "relu",
        use_batch_norm: bool = True,
        name: str = "decoder_gcn",
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
        if len(gcn_units) == 0:
            raise ValueError("gcn_units must contain at least one layer.")

        self.timesteps = timesteps
        self.n_channels = n_channels
        self.n_bands = n_bands
        self.t_down = t_down
        self.gcn_units = tuple(gcn_units)
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

        reversed_units = tuple(reversed(self.gcn_units))

        self.input_projection = layers.Conv1D(
            reversed_units[0],
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

        self.conv_layers = [
            layers.Conv1D(
                filters,
                3,
                padding="same",
                activation=activation,
                name=f"dec_conv1d_{i}",
            )
            for i, filters in enumerate(reversed_units)
        ]

        self.bn_layers = [
            layers.BatchNormalization(name=f"dec_bn1d_{i}")
            if use_batch_norm
            else None
            for i, _ in enumerate(reversed_units)
        ]

        self.dropout_layers = [
            layers.Dropout(
                dropout,
                name=f"dec_do1d_{i}",
            )
            for i, _ in enumerate(reversed_units)
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
        encoder: GCNEncoder,
        name: str = "decoder_gcn",
    ) -> "GCNDecoder":
        """Create a temporal decoder from a configured ``GCNEncoder``."""
        if not isinstance(encoder, GCNEncoder):
            raise TypeError(
                "GCNDecoder.from_encoder only supports GCNEncoder. "
                f"Got {type(encoder).__name__}."
            )

        return cls(
            timesteps=encoder.timesteps,
            n_channels=encoder.n_channels,
            n_bands=encoder.n_bands,
            t_down=encoder.t_down,
            gcn_units=encoder.gcn_units,
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
        """Run the graph-encoder-compatible temporal decoder forward pass."""
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
                "gcn_units": self.gcn_units,
                "temporal_pool_sizes": self.temporal_pool_sizes,
                "emb_dim": self.emb_dim,
                "dropout": self.dropout_rate,
                "activation": self.activation,
                "use_batch_norm": self.use_batch_norm,
                "name": self.name,
            }
        )
        return config