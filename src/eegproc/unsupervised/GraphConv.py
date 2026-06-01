import tensorflow as tf
from tensorflow.keras import layers

from .BaseEncoder import BaseEncoder
from .GraphConv import GraphConv


class GCNEncoder(BaseEncoder):
    """Graph convolutional encoder for EEG sequence data.

    This encoder treats EEG electrodes as graph nodes and frequency-band
    values as node features. The expected input shape is
    ``(batch, timesteps, n_channels * n_bands)``. Each timestep is reshaped
    into a node-feature matrix of shape ``(n_channels, n_bands)``.

    A stack of ``GraphConv`` layers is applied independently at each timestep
    using ``TimeDistributed``. Each ``GraphConv`` layer learns its own
    adjacency matrix, allowing the model to discover functional relationships
    among EEG channels rather than assuming a fixed electrode topology.

    After graph feature extraction, the node dimension is pooled by averaging
    across channels. The resulting temporal sequence is downsampled using two
    ``MaxPool1D`` layers and projected to ``emb_dim`` latent features per
    downsampled timestep.

    Parameters
    ----------
    timesteps : int
        Number of timesteps in each input sequence.
    n_channels : int, default=14
        Number of EEG electrode channels. This is also the number of graph
        nodes.
    n_bands : int, default=6
        Number of frequency-band features per EEG channel.
    gcn_units : tuple[int, ...], default=(32, 64)
        Output dimensionality of each successive ``GraphConv`` layer. The
        length of this tuple determines the number of graph-convolution blocks.
    emb_dim : int, default=128
        Dimensionality of the latent embedding at each downsampled timestep.
    dropout : float, default=0.10
        Dropout rate applied after each graph-convolution block and after the
        first temporal pooling layer.
    name : str, default="encoder_gcn"
        Name of the Keras model.
    **kwargs
        Additional keyword arguments passed to ``tf.keras.Model``.

    Input shape
    -----------
    ``(batch, timesteps, n_channels * n_bands)``

    Output shape
    ------------
    ``(batch, ceil(timesteps / 4), emb_dim)``

    Notes
    -----
    The learned adjacency matrix of the i-th graph layer can be accessed with:

    ``encoder.gcn_layers[i].layer.A_theta``

    because each ``GraphConv`` layer is wrapped inside ``TimeDistributed``.
    """

    def __init__(
        self,
        timesteps: int,
        n_channels: int = 14,
        n_bands: int = 6,
        gcn_units: tuple[int, ...] = (32, 64),
        emb_dim: int = 128,
        dropout: float = 0.10,
        name: str = "encoder_gcn",
        **kwargs,
    ):
        super().__init__(
            timesteps=timesteps,
            emb_dim=emb_dim,
            name=name,
            **kwargs,
        )

        self.n_channels = n_channels
        self.n_bands = n_bands
        self.gcn_units = tuple(gcn_units)
        self.dropout_rate = dropout

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
            )

            self.dropout_layers.append(
                layers.TimeDistributed(
                    layers.Dropout(dropout),
                    name=f"gcn_do_{i}",
                )
            )

        self.tpool1 = layers.MaxPool1D(
            2,
            padding="same",
            name="enc_tpool1",
        )
        self.tdo1 = layers.Dropout(
            dropout,
            name="enc_tdo1",
        )
        self.tpool2 = layers.MaxPool1D(
            2,
            padding="same",
            name="enc_tpool2",
        )

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
        """Run the graph convolutional encoder forward pass."""
        x = self.to_nodes(inputs)

        for gcn, bn, dropout in zip(
            self.gcn_layers,
            self.bn_layers,
            self.dropout_layers,
        ):
            x = gcn(x, training=training)
            x = bn(x, training=training)
            x = dropout(x, training=training)

        x = tf.reduce_mean(x, axis=2)

        x = self.tpool1(x)
        x = self.tdo1(x, training=training)
        x = self.tpool2(x)

        return self.seq_emb(x)

    def get_config(self) -> dict:
        """Return serializable configuration for the encoder."""
        config = super().get_config()
        config.update(
            {
                "n_channels": self.n_channels,
                "n_bands": self.n_bands,
                "gcn_units": self.gcn_units,
                "dropout": self.dropout_rate,
            }
        )
        return config