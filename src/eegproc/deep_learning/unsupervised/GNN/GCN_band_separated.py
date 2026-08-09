import tensorflow as tf
from tensorflow.keras import layers

from ..BaseEncoder import BaseEncoder
from .GraphConv import GraphConv
from ..utils import _product


@tf.keras.utils.register_keras_serializable(package="eegproc")
class BandSeparatedGCNEncoder(BaseEncoder):
    """Band-separated residual GCN encoder for EEG sequence data.

    Input shape:
        (batch, timesteps, n_channels * n_bands)

    The flattened feature axis is interpreted in channel-major order:
        [ch0_band0, ch0_band1, ..., ch1_band0, ch1_band1, ...]

    At each timestep, the input is reshaped to:
        (batch, timesteps, n_channels, n_bands)

    Each frequency band is then processed by its own independent GCN stack:

        band_0 -> GCN stack 0
        band_1 -> GCN stack 1
        ...
        band_k -> GCN stack k

    Therefore, every band has its own:
      * GraphConv weights
      * learnable electrode adjacency matrices
      * normalization layers
      * residual projections

    Bands never mix inside the graph convolutions. Their node-preserving graph
    outputs are concatenated only after all per-band GCN stacks have completed.

    The output remains:
        (batch, ceil(timesteps / t_down), emb_dim)

    so it remains compatible with downstream BiLSTM sequence models.
    """

    def __init__(
        self,
        timesteps: int,
        t_down: int,
        n_channels: int = 14,
        n_bands: int = 3,
        gcn_units: tuple[int, ...] = (64, 32),
        temporal_pool_sizes: tuple[int, ...] | None = None,
        emb_dim: int = 128,
        dropout: float = 0.10,
        activation: str = "relu",
        use_batch_norm: bool = False,
        graph_self_loop_bias: float = 2.0,
        graph_identity_mix: float = 0.0,
        graph_adjacency_reg_weight: float = 1e-4,
        name: str = "encoder_band_separated_gcn",
        **kwargs,
    ):
        super().__init__(
            timesteps=timesteps,
            emb_dim=emb_dim,
            t_down=t_down,
            name=name,
            **kwargs,
        )

        if timesteps <= 0:
            raise ValueError(f"timesteps must be positive, got {timesteps}.")
        if n_channels <= 0:
            raise ValueError(f"n_channels must be positive, got {n_channels}.")
        if n_bands <= 0:
            raise ValueError(f"n_bands must be positive, got {n_bands}.")
        if len(gcn_units) == 0:
            raise ValueError("gcn_units must contain at least one layer.")
        if any(units <= 0 for units in gcn_units):
            raise ValueError(f"All gcn_units must be positive, got {gcn_units}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")
        if graph_self_loop_bias < 0.0:
            raise ValueError(
                "graph_self_loop_bias must be non-negative, "
                f"got {graph_self_loop_bias}."
            )
        if not 0.0 <= graph_identity_mix <= 1.0:
            raise ValueError(
                "graph_identity_mix must be in [0, 1], "
                f"got {graph_identity_mix}."
            )
        if graph_adjacency_reg_weight < 0.0:
            raise ValueError(
                "graph_adjacency_reg_weight must be non-negative, "
                f"got {graph_adjacency_reg_weight}."
            )

        self.n_channels = int(n_channels)
        self.n_bands = int(n_bands)
        self.gcn_units = tuple(int(units) for units in gcn_units)
        self.temporal_pool_sizes = self._normalize_temporal_pool_sizes(
            temporal_pool_sizes,
            self.t_down,
        )
        self.dropout_rate = float(dropout)
        self.activation = activation
        self.use_batch_norm = bool(use_batch_norm)
        self.graph_self_loop_bias = float(graph_self_loop_bias)
        self.graph_identity_mix = float(graph_identity_mix)
        self.graph_adjacency_reg_weight = float(graph_adjacency_reg_weight)

        self.to_nodes = layers.Reshape(
            (timesteps, n_channels, n_bands),
            name="to_channel_band_grid",
        )

        # One completely independent graph stack per frequency band.
        self.band_gcn_layers: list[list[GraphConv]] = []
        self.band_bn_layers: list[list[layers.Layer | None]] = []
        self.band_residual_projections: list[list[layers.Layer | None]] = []
        self.band_activation_layers: list[list[layers.Layer]] = []
        self.band_dropout_layers: list[list[layers.Layer]] = []

        for band_index in range(self.n_bands):
            gcn_layers = []
            bn_layers = []
            residual_projections = []
            activation_layers = []
            dropout_layers = []

            # Each band begins with exactly one scalar feature per electrode.
            input_units = 1

            for layer_index, units in enumerate(self.gcn_units):
                prefix = f"band_{band_index}_gcn_{layer_index}"

                gcn_layers.append(
                    GraphConv(
                        units=units,
                        n_nodes=self.n_channels,
                        activation=None,
                        self_loop_bias=self.graph_self_loop_bias,
                        identity_mix=self.graph_identity_mix,
                        adjacency_reg_weight=self.graph_adjacency_reg_weight,
                        name=prefix,
                    )
                )

                bn_layers.append(
                    layers.BatchNormalization(
                        name=f"band_{band_index}_gcn_bn_{layer_index}"
                    )
                    if self.use_batch_norm
                    else None
                )

                residual_projections.append(
                    layers.Dense(
                        units,
                        use_bias=False,
                        name=(
                            f"band_{band_index}_gcn_residual_projection_"
                            f"{layer_index}"
                        ),
                    )
                    if input_units != units
                    else None
                )

                activation_layers.append(
                    layers.Activation(
                        self.activation,
                        name=f"band_{band_index}_gcn_activation_{layer_index}",
                    )
                )

                dropout_layers.append(
                    layers.SpatialDropout2D(
                        self.dropout_rate,
                        name=f"band_{band_index}_gcn_spatial_do_{layer_index}",
                    )
                )

                input_units = units

            self.band_gcn_layers.append(gcn_layers)
            self.band_bn_layers.append(bn_layers)
            self.band_residual_projections.append(residual_projections)
            self.band_activation_layers.append(activation_layers)
            self.band_dropout_layers.append(dropout_layers)

        # Preserve electrode identity within every band, then concatenate bands.
        self.band_node_readouts = [
            layers.Reshape(
                (
                    timesteps,
                    self.n_channels * self.gcn_units[-1],
                ),
                name=f"band_{band_index}_node_preserving_readout",
            )
            for band_index in range(self.n_bands)
        ]

        self.band_fusion = layers.Concatenate(
            axis=-1,
            name="band_gcn_feature_concatenation",
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
                self.dropout_rate,
                name=f"enc_tdo_{i}",
            )
            for i, _ in enumerate(self.temporal_pool_sizes)
        ]

        # Only after band-specific graph processing do we mix band features.
        self.seq_emb = layers.Conv1D(
            self.emb_dim,
            kernel_size=1,
            padding="same",
            activation=None,
            name="band_fused_seq_emb",
        )

    @staticmethod
    def _normalize_temporal_pool_sizes(
        pool_sizes: tuple[int, ...] | None,
        t_down: int,
    ) -> tuple[int, ...]:
        """Normalize temporal pooling and ensure it matches ``t_down``."""
        if pool_sizes is None:
            normalized = () if t_down == 1 else (int(t_down),)
        else:
            normalized = tuple(int(value) for value in pool_sizes)

        if any(value < 1 for value in normalized):
            raise ValueError(
                "All temporal pool sizes must be >= 1, "
                f"got {normalized}."
            )

        effective_t_down = _product(normalized) if normalized else 1
        if effective_t_down != t_down:
            raise ValueError(
                f"t_down={t_down}, but temporal_pool_sizes produces "
                f"a downsampling factor of {effective_t_down}. "
                "Set t_down equal to product(temporal_pool_sizes)."
            )

        return normalized

    @property
    def n_features(self) -> int:
        """Flattened number of channel-band features per timestep."""
        return self.n_channels * self.n_bands

    def _encode_one_band(
        self,
        x: tf.Tensor,
        band_index: int,
        training: bool,
    ) -> tf.Tensor:
        """Run one frequency band through only its own graph stack."""

        # x is (batch, time, channels, bands).
        # Preserve a feature axis so GraphConv receives:
        # (batch, time, channels, 1).
        band_x = x[..., band_index : band_index + 1]

        for gcn, bn, residual_projection, activation, dropout in zip(
            self.band_gcn_layers[band_index],
            self.band_bn_layers[band_index],
            self.band_residual_projections[band_index],
            self.band_activation_layers[band_index],
            self.band_dropout_layers[band_index],
        ):
            residual = band_x
            band_x = gcn(band_x, training=training)

            if bn is not None:
                band_x = bn(band_x, training=training)

            if residual_projection is not None:
                residual = residual_projection(residual)

            band_x = activation(band_x + residual)
            band_x = dropout(band_x, training=training)

        return self.band_node_readouts[band_index](band_x)

    def call(self, inputs, training: bool = False):
        """Encode each EEG frequency band with an independent electrode GCN."""

        inputs = tf.convert_to_tensor(inputs)

        if inputs.shape.rank != 3:
            raise ValueError(
                "BandSeparatedGCNEncoder expects "
                "(batch, timesteps, n_channels * n_bands); "
                f"got {inputs.shape}."
            )

        expected_features = self.n_channels * self.n_bands
        static_features = inputs.shape[-1]
        if static_features is not None and int(static_features) != expected_features:
            raise ValueError(
                "Input feature dimension does not match "
                f"n_channels * n_bands: {static_features} != "
                f"{self.n_channels} * {self.n_bands}."
            )

        x = self.to_nodes(inputs)

        band_sequences = [
            self._encode_one_band(
                x=x,
                band_index=band_index,
                training=training,
            )
            for band_index in range(self.n_bands)
        ]

        x = (
            band_sequences[0]
            if self.n_bands == 1
            else self.band_fusion(band_sequences)
        )

        for pool, dropout in zip(
            self.temporal_pool_layers,
            self.temporal_dropout_layers,
        ):
            x = pool(x)
            x = dropout(x, training=training)

        return self.seq_emb(x)

    def get_adjacency_matrices(self) -> dict[str, dict[str, tf.Tensor]]:
        """Return one set of learned adjacency matrices per frequency band."""
        output: dict[str, dict[str, tf.Tensor]] = {}

        for band_index, graph_stack in enumerate(self.band_gcn_layers):
            output[f"band_{band_index}"] = {
                graph_layer.name: graph_layer.normalized_adjacency()
                for graph_layer in graph_stack
            }

        return output

    def get_band_features(
        self,
        inputs,
        training: bool = False,
    ) -> dict[str, tf.Tensor]:
        """Expose per-band post-GCN node-preserving sequences for diagnostics."""
        x = self.to_nodes(tf.convert_to_tensor(inputs))

        return {
            f"band_{band_index}": self._encode_one_band(
                x=x,
                band_index=band_index,
                training=training,
            )
            for band_index in range(self.n_bands)
        }

    def get_config(self) -> dict:
        """Return serializable configuration for the band-separated encoder."""
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
                "graph_self_loop_bias": self.graph_self_loop_bias,
                "graph_identity_mix": self.graph_identity_mix,
                "graph_adjacency_reg_weight": (
                    self.graph_adjacency_reg_weight
                ),
            }
        )
        return config


# Backward-compatible alias for callers that import GCNEncoder.
# New code should import BandSeparatedGCNEncoder explicitly.
GCNEncoder = BandSeparatedGCNEncoder


@tf.keras.utils.register_keras_serializable(package="eegproc")
class GCNDecoder(tf.keras.Model):
    """Graph-aware decoder for :class:`GCNEncoder` latent sequences.

    The decoder first restores temporal resolution. It then projects each latent
    timestep into a distinct feature vector for every electrode, reshapes that
    vector back to a graph, and applies residual graph-convolution blocks before
    reconstructing the frequency-band values at each node.

    This is intentionally different from a purely temporal Conv1D decoder: the
    output is reconstructed as ``n_channels`` graph nodes rather than as one
    undifferentiated flat feature vector.
    """

    def __init__(
        self,
        timesteps: int,
        n_channels: int,
        n_bands: int,
        t_down: int,
        gcn_units: tuple[int, ...],
        temporal_pool_sizes: tuple[int, ...] | None,
        emb_dim: int = 128,
        dropout: float = 0.10,
        activation: str = "relu",
        use_batch_norm: bool = False,
        graph_self_loop_bias: float = 2.0,
        graph_identity_mix: float = 0.0,
        graph_adjacency_reg_weight: float = 1e-4,
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
        if any(units <= 0 for units in gcn_units):
            raise ValueError(f"All gcn_units must be positive, got {gcn_units}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")
        if graph_self_loop_bias < 0.0:
            raise ValueError(
                "graph_self_loop_bias must be non-negative, "
                f"got {graph_self_loop_bias}."
            )
        if not 0.0 <= graph_identity_mix <= 1.0:
            raise ValueError(
                "graph_identity_mix must be in [0, 1], "
                f"got {graph_identity_mix}."
            )
        if graph_adjacency_reg_weight < 0.0:
            raise ValueError(
                "graph_adjacency_reg_weight must be non-negative, "
                f"got {graph_adjacency_reg_weight}."
            )

        self.timesteps = timesteps
        self.n_channels = n_channels
        self.n_bands = n_bands
        self.t_down = t_down
        self.gcn_units = tuple(gcn_units)
        self.temporal_pool_sizes = GCNEncoder._normalize_temporal_pool_sizes(
            temporal_pool_sizes,
            self.t_down,
        )
        self.emb_dim = emb_dim
        self.dropout_rate = dropout
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.graph_self_loop_bias = float(graph_self_loop_bias)
        self.graph_identity_mix = float(graph_identity_mix)
        self.graph_adjacency_reg_weight = float(graph_adjacency_reg_weight)

        graph_seed_units = self.gcn_units[-1]

        self.input_projection = layers.Conv1D(
            graph_seed_units,
            kernel_size=1,
            padding="same",
            activation=None,
            name="dec_input_projection",
        )
        self.input_bn = (
            layers.BatchNormalization(name="dec_input_bn")
            if use_batch_norm
            else None
        )
        self.input_activation = layers.Activation(
            activation,
            name="dec_input_activation",
        )

        self.upsample_layers = []
        self.temporal_conv_layers = []
        self.temporal_bn_layers = []
        self.temporal_activation_layers = []
        self.temporal_dropout_layers = []

        for i, pool_size in enumerate(reversed(self.temporal_pool_sizes)):
            self.upsample_layers.append(
                layers.UpSampling1D(
                    size=pool_size,
                    name=f"dec_upsample_{i}",
                )
            )
            self.temporal_conv_layers.append(
                layers.Conv1D(
                    graph_seed_units,
                    kernel_size=3,
                    padding="same",
                    activation=None,
                    name=f"dec_temporal_conv_{i}",
                )
            )
            self.temporal_bn_layers.append(
                layers.BatchNormalization(name=f"dec_temporal_bn_{i}")
                if use_batch_norm
                else None
            )
            self.temporal_activation_layers.append(
                layers.Activation(
                    activation,
                    name=f"dec_temporal_activation_{i}",
                )
            )
            self.temporal_dropout_layers.append(
                layers.Dropout(dropout, name=f"dec_temporal_do_{i}")
            )

        # Every electrode receives its own seed vector. This is the inverse of
        # the encoder's node-preserving concatenation.
        self.node_seed_projection = layers.Dense(
            n_channels * graph_seed_units,
            activation=None,
            name="dec_node_seed_projection",
        )

        self.graph_layers = []
        self.graph_bn_layers = []
        self.graph_residual_projections = []
        self.graph_activation_layers = []
        self.graph_dropout_layers = []

        input_units = graph_seed_units
        decoder_units = tuple(reversed(self.gcn_units[:-1]))
        for i, units in enumerate(decoder_units):
            # Process every reconstructed timestep in one vectorized
            # graph-convolution call. This keeps one shared electrode graph and
            # adds its regularizer only once per decoder layer/model call.
            self.graph_layers.append(
                GraphConv(
                    units=units,
                    n_nodes=n_channels,
                    activation=None,
                    self_loop_bias=self.graph_self_loop_bias,
                    identity_mix=self.graph_identity_mix,
                    adjacency_reg_weight=self.graph_adjacency_reg_weight,
                    name=f"dec_gcn_{i}",
                )
            )
            self.graph_bn_layers.append(
                layers.BatchNormalization(name=f"dec_gcn_bn_{i}")
                if use_batch_norm
                else None
            )
            self.graph_residual_projections.append(
                layers.Dense(
                    units,
                    use_bias=False,
                    name=f"dec_gcn_residual_projection_{i}",
                )
                if input_units != units
                else None
            )
            self.graph_activation_layers.append(
                layers.Activation(activation, name=f"dec_gcn_activation_{i}")
            )
            self.graph_dropout_layers.append(
                layers.SpatialDropout2D(
                    dropout,
                    name=f"dec_gcn_spatial_do_{i}",
                )
            )
            input_units = units

        # Combine a graph-mixed reconstruction with a node-local bypass. The
        # bypass prevents the final graph layer from erasing electrode identity.
        self.output_graph = GraphConv(
            units=n_bands,
            n_nodes=n_channels,
            activation=None,
            self_loop_bias=self.graph_self_loop_bias,
            identity_mix=self.graph_identity_mix,
            adjacency_reg_weight=self.graph_adjacency_reg_weight,
            name="dec_output_graph",
        )
        self.output_local = layers.Dense(
            n_bands,
            activation=None,
            name="dec_output_local",
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
        """Create a graph-aware decoder from a configured encoder."""
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
            graph_self_loop_bias=encoder.graph_self_loop_bias,
            graph_identity_mix=encoder.graph_identity_mix,
            graph_adjacency_reg_weight=encoder.graph_adjacency_reg_weight,
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
        """Decode latent sequences into channel-specific EEG graph signals."""
        x = self.input_projection(inputs)

        if self.input_bn is not None:
            x = self.input_bn(x, training=training)

        x = self.input_activation(x)

        for upsample, conv, bn, activation, dropout in zip(
            self.upsample_layers,
            self.temporal_conv_layers,
            self.temporal_bn_layers,
            self.temporal_activation_layers,
            self.temporal_dropout_layers,
        ):
            x = upsample(x)
            residual = x
            x = conv(x)

            if bn is not None:
                x = bn(x, training=training)

            x = activation(x + residual)
            x = dropout(x, training=training)

        x = self.node_seed_projection(x)
        batch_size = tf.shape(x)[0]
        time_steps = tf.shape(x)[1]
        x = tf.reshape(
            x,
            (batch_size, time_steps, self.n_channels, self.gcn_units[-1]),
        )

        for gcn, bn, residual_projection, activation, dropout in zip(
            self.graph_layers,
            self.graph_bn_layers,
            self.graph_residual_projections,
            self.graph_activation_layers,
            self.graph_dropout_layers,
        ):
            residual = x
            x = gcn(x, training=training)

            if bn is not None:
                x = bn(x, training=training)

            if residual_projection is not None:
                residual = residual_projection(residual)

            x = activation(x + residual)
            x = dropout(x, training=training)

        x = self.output_graph(x, training=training) + self.output_local(x)
        x = tf.reshape(
            x,
            (batch_size, time_steps, self.n_features),
        )

        return self.fix_length(x)

    def get_adjacency_matrices(self) -> dict[str, tf.Tensor]:
        """Return normalized decoder electrode graphs for diagnostics."""
        matrices = {
            graph_layer.name: graph_layer.normalized_adjacency()
            for graph_layer in self.graph_layers
        }
        matrices[self.output_graph.name] = self.output_graph.normalized_adjacency()
        return matrices

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
                "graph_self_loop_bias": self.graph_self_loop_bias,
                "graph_identity_mix": self.graph_identity_mix,
                "graph_adjacency_reg_weight": self.graph_adjacency_reg_weight,
                "name": self.name,
            }
        )
        return config
