"""MTLFuseNet-style spatio-spectral GCN/GRU encoder for EEGProc.

This module is intentionally separate from ``GCN_band_separated.py`` so the
learned-adjacency EEGProc GCN and the MTLFuseNet-style fixed-MI GCN can be
compared without changing either implementation.

Paper-aligned pieces
--------------------
1. Channel adjacency A is fixed and based on mutual information.
2. Graph convolution uses
       A_tilde = A + I
       A_hat   = D_tilde^-1/2 A_tilde D_tilde^-1/2
       H_B     = ReLU(A_hat V_B W + b)
3. Frequency bands are treated as an ordered spatio-spectral graph sequence.
4. The SAME graph-convolution layer objects (therefore the same W and b) are
   reused for every frequency band.
5. A GRU can process the ordered band sequence at each EEG timestep.

EEGProc compatibility adaptation
---------------------------------
MTLFuseNet uses differential-entropy node features. EEGProc's current joint
pipeline supplies a temporal sequence of preprocessed channel-band values.
This encoder therefore applies the MTL graph/weight-sharing mechanism to those
per-timestep band features so it still returns
    (batch, ceil(timesteps / t_down), emb_dim)
and can be compared against BandSeparatedGCNEncoder in the same pipeline.

For a strict MTLFuseNet reproduction, feed DE features instead of raw/filtered
band amplitudes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from ..BaseEncoder import BaseEncoder
from .GraphConvMTL import GraphConvMTL
from ..utils import _product


def _as_channel_band_array(
    inputs,
    n_channels: int,
    n_bands: int,
) -> np.ndarray:
    """Normalize input to shape (observations, channels, bands)."""
    x = np.asarray(inputs, dtype=np.float64)

    if x.ndim >= 2 and x.shape[-2:] == (n_channels, n_bands):
        return x.reshape(-1, n_channels, n_bands)

    expected_features = n_channels * n_bands
    if x.ndim >= 1 and x.shape[-1] == expected_features:
        return x.reshape(-1, n_channels, n_bands)

    raise ValueError(
        "Expected input ending in (n_channels, n_bands) or in a flattened "
        f"channel-major feature axis of length {expected_features}; got {x.shape}."
    )


def compute_mi_adjacency_from_channels(
    channel_data,
    *,
    n_neighbors: int = 3,
    random_state: int = 42,
    zero_diagonal: bool = False,
) -> np.ndarray:
    """Compute a symmetric mutual-information channel adjacency.

    Parameters
    ----------
    channel_data : array-like, shape (..., n_channels)
        Samples/observations for each EEG channel. All leading dimensions are
        flattened into the observation dimension.
    n_neighbors : int, default=3
        k used by sklearn's continuous mutual-information estimator.
    random_state : int, default=42
        Reproducibility seed for sklearn's MI estimator.
    zero_diagonal : bool, default=False
        The MTLFuseNet definition permits i=j, so the default preserves the
        estimated MI diagonal. Set True only for an ablation where A contains
        cross-channel MI and self-connectivity comes solely from A_tilde=A+I.

    Notes
    -----
    Use TRAINING DATA ONLY. Computing A from validation/test subjects leaks
    information into the graph used by the model.
    """
    try:
        from sklearn.feature_selection import mutual_info_regression
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "scikit-learn is required to compute the MTLFuseNet MI adjacency."
        ) from exc

    x = np.asarray(channel_data, dtype=np.float64)
    if x.ndim < 2:
        raise ValueError(
            "channel_data must have at least 2 dimensions (..., n_channels)."
        )

    x = x.reshape(-1, x.shape[-1])
    if x.shape[0] < max(4, n_neighbors + 1):
        raise ValueError(
            f"Need more observations to estimate MI; got {x.shape[0]}."
        )
    if not np.all(np.isfinite(x)):
        raise ValueError("channel_data contains NaN or infinite values.")

    n_channels = x.shape[1]
    A = np.zeros((n_channels, n_channels), dtype=np.float64)

    for i in range(n_channels):
        start_j = i + 1 if zero_diagonal else i
        for j in range(start_j, n_channels):
            if i == j:
                # sklearn MI(x, x) is estimator-dependent. Keep it only when
                # explicitly requested; the paper's renormalization will still
                # add another identity self-loop afterward.
                mi = float(
                    mutual_info_regression(
                        x[:, i : i + 1],
                        x[:, j],
                        discrete_features=False,
                        n_neighbors=n_neighbors,
                        random_state=random_state,
                    )[0]
                )
            else:
                # The theoretical MI is symmetric. kNN estimates need not be
                # numerically identical in the two regression directions, so
                # average them to enforce A_ij = A_ji.
                mi_ij = float(
                    mutual_info_regression(
                        x[:, i : i + 1],
                        x[:, j],
                        discrete_features=False,
                        n_neighbors=n_neighbors,
                        random_state=random_state,
                    )[0]
                )
                mi_ji = float(
                    mutual_info_regression(
                        x[:, j : j + 1],
                        x[:, i],
                        discrete_features=False,
                        n_neighbors=n_neighbors,
                        random_state=random_state,
                    )[0]
                )
                mi = 0.5 * (mi_ij + mi_ji)

            A[i, j] = mi
            A[j, i] = mi

    if zero_diagonal:
        np.fill_diagonal(A, 0.0)

    return A.astype(np.float32)


def compute_mtl_shared_mi_adjacency(
    inputs,
    *,
    n_channels: int = 14,
    n_bands: int = 3,
    n_neighbors: int = 3,
    random_state: int = 42,
    zero_diagonal: bool = False,
    band_reduction: str = "mean",
) -> np.ndarray:
    """Estimate one shared MTL adjacency from EEGProc channel-band data.

    The MTLFuseNet text uses a single A in every graph G_B while the node
    features V'_B vary by band. If only already-band-separated EEGProc inputs
    are available, this helper estimates MI independently in each band and
    reduces the band-wise MI matrices to ONE shared A.

    ``band_reduction='mean'`` is the default compatibility approximation.
    If you have the pre-band-split channel signals x_i used to construct the
    paper's A, prefer ``compute_mi_adjacency_from_channels`` directly.
    """
    x = _as_channel_band_array(inputs, n_channels, n_bands)

    band_adjacencies = []
    for band_index in range(n_bands):
        band_adjacencies.append(
            compute_mi_adjacency_from_channels(
                x[:, :, band_index],
                n_neighbors=n_neighbors,
                random_state=random_state,
                zero_diagonal=zero_diagonal,
            )
        )

    stacked = np.stack(band_adjacencies, axis=0)
    if band_reduction == "mean":
        A = np.mean(stacked, axis=0)
    elif band_reduction == "max":
        A = np.max(stacked, axis=0)
    elif band_reduction == "median":
        A = np.median(stacked, axis=0)
    else:
        raise ValueError(
            "band_reduction must be one of {'mean', 'max', 'median'}, "
            f"got {band_reduction!r}."
        )

    A = 0.5 * (A + A.T)
    if zero_diagonal:
        np.fill_diagonal(A, 0.0)
    return A.astype(np.float32)


@tf.keras.utils.register_keras_serializable(package="eegproc")
class GCNMTLEncoder(BaseEncoder):
    """MTLFuseNet-style shared-parameter spatio-spectral GCN/GRU encoder.

    Input
    -----
    (batch, timesteps, n_channels * n_bands), channel-major flattened order:
        [ch0_band0, ch0_band1, ..., ch1_band0, ...]

    Output
    ------
    (batch, ceil(timesteps / t_down), emb_dim)

    The adjacency is fixed. It MUST be computed from training data only.
    """

    def __init__(
        self,
        timesteps: int,
        t_down: int,
        adjacency,
        n_channels: int = 14,
        n_bands: int = 3,
        gcn_units: tuple[int, ...] = (64, 32),
        temporal_pool_sizes: tuple[int, ...] | None = None,
        emb_dim: int = 128,
        dropout: float = 0.10,
        activation: str = "relu",
        use_batch_norm: bool = False,
        use_spectral_gru: bool = True,
        spectral_gru_units: int = 128,
        spectral_gru_dropout: float = 0.0,
        graph_add_self_loops: bool = True,
        graph_symmetrize: bool = True,
        graph_epsilon: float = 1e-8,
        # Compatibility-only parameters accepted so JSON/constructor argument
        # sets used by BandSeparatedGCNEncoder do not crash when comparing.
        graph_self_loop_bias: float | None = None,
        graph_identity_mix: float | None = None,
        graph_adjacency_reg_weight: float | None = None,
        name: str = "encoder_gcn_mtl",
        **kwargs,
    ):
        del graph_self_loop_bias, graph_identity_mix, graph_adjacency_reg_weight

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
        if len(gcn_units) == 0 or any(u <= 0 for u in gcn_units):
            raise ValueError(f"gcn_units must be positive, got {gcn_units}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")
        if spectral_gru_units <= 0:
            raise ValueError(
                f"spectral_gru_units must be positive, got {spectral_gru_units}."
            )
        if not 0.0 <= spectral_gru_dropout < 1.0:
            raise ValueError(
                "spectral_gru_dropout must be in [0, 1), got "
                f"{spectral_gru_dropout}."
            )

        adjacency_array = np.asarray(adjacency, dtype=np.float32)
        if adjacency_array.shape != (n_channels, n_channels):
            raise ValueError(
                f"adjacency must be ({n_channels}, {n_channels}), "
                f"got {adjacency_array.shape}."
            )

        self.n_channels = int(n_channels)
        self.n_bands = int(n_bands)
        self.gcn_units = tuple(int(u) for u in gcn_units)
        self.temporal_pool_sizes = self._normalize_temporal_pool_sizes(
            temporal_pool_sizes, self.t_down
        )
        self.dropout_rate = float(dropout)
        self.activation = activation
        self.use_batch_norm = bool(use_batch_norm)
        self.use_spectral_gru = bool(use_spectral_gru)
        self.spectral_gru_units = int(spectral_gru_units)
        self.spectral_gru_dropout = float(spectral_gru_dropout)
        self.graph_add_self_loops = bool(graph_add_self_loops)
        self.graph_symmetrize = bool(graph_symmetrize)
        self.graph_epsilon = float(graph_epsilon)
        self._adjacency_init = adjacency_array.tolist()

        self.to_nodes = layers.Reshape(
            (timesteps, n_channels, n_bands),
            name="mtl_to_channel_band_grid",
        )

        # IMPORTANT: one layer object per depth, reused for every band.
        # Therefore W and b are shared across theta/alpha/beta(/gamma), as
        # described by MTLFuseNet.
        self.shared_gcn_layers: list[GraphConvMTL] = []
        self.shared_bn_layers: list[layers.Layer | None] = []
        self.shared_dropout_layers: list[layers.Layer] = []

        for layer_index, units in enumerate(self.gcn_units):
            self.shared_gcn_layers.append(
                GraphConvMTL(
                    units=units,
                    n_nodes=self.n_channels,
                    adjacency=self._adjacency_init,
                    activation=self.activation,
                    add_self_loops=self.graph_add_self_loops,
                    symmetrize=self.graph_symmetrize,
                    epsilon=self.graph_epsilon,
                    name=f"mtl_shared_gcn_{layer_index}",
                )
            )
            self.shared_bn_layers.append(
                layers.BatchNormalization(name=f"mtl_shared_gcn_bn_{layer_index}")
                if self.use_batch_norm
                else None
            )
            self.shared_dropout_layers.append(
                layers.SpatialDropout2D(
                    self.dropout_rate,
                    name=f"mtl_shared_gcn_spatial_do_{layer_index}",
                )
            )

        self.spectral_gru = (
            layers.GRU(
                self.spectral_gru_units,
                return_sequences=False,
                dropout=self.spectral_gru_dropout,
                name="mtl_spectral_gru",
            )
            if self.use_spectral_gru
            else None
        )

        self.temporal_pool_layers = [
            layers.MaxPool1D(pool_size, padding="same", name=f"mtl_tpool_{i}")
            for i, pool_size in enumerate(self.temporal_pool_sizes)
        ]
        self.temporal_dropout_layers = [
            layers.Dropout(self.dropout_rate, name=f"mtl_tdo_{i}")
            for i, _ in enumerate(self.temporal_pool_sizes)
        ]

        self.seq_emb = layers.Conv1D(
            self.emb_dim,
            kernel_size=1,
            padding="same",
            activation=None,
            name="mtl_seq_emb",
        )

    @staticmethod
    def _normalize_temporal_pool_sizes(
        pool_sizes: tuple[int, ...] | None,
        t_down: int,
    ) -> tuple[int, ...]:
        if pool_sizes is None:
            normalized = () if t_down == 1 else (int(t_down),)
        else:
            normalized = tuple(int(value) for value in pool_sizes)

        if any(value < 1 for value in normalized):
            raise ValueError(
                f"All temporal pool sizes must be >= 1, got {normalized}."
            )

        effective_t_down = _product(normalized) if normalized else 1
        if effective_t_down != t_down:
            raise ValueError(
                f"t_down={t_down}, but temporal_pool_sizes gives "
                f"{effective_t_down}."
            )
        return normalized

    @property
    def n_features(self) -> int:
        return self.n_channels * self.n_bands

    def _encode_one_band(
        self,
        x: tf.Tensor,
        band_index: int,
        training: bool,
    ) -> tf.Tensor:
        # (batch, time, channels, 1)
        band_x = x[..., band_index : band_index + 1]

        for gcn, bn, dropout in zip(
            self.shared_gcn_layers,
            self.shared_bn_layers,
            self.shared_dropout_layers,
        ):
            # No residual block here: MTLFuseNet Eq. (15) is directly
            # ReLU(A_hat V W + b).
            band_x = gcn(band_x, training=training)
            if bn is not None:
                band_x = bn(band_x, training=training)
            band_x = dropout(band_x, training=training)

        return band_x

    def _spectral_sequence_to_temporal_features(
        self,
        band_node_features: list[tf.Tensor],
        training: bool,
    ) -> tf.Tensor:
        # Each band: (B, T, C, U) -> (B, T, C*U)
        band_feature_dim = self.n_channels * self.gcn_units[-1]
        flattened = [
            tf.ensure_shape(
                tf.reshape(
                    band_x,
                    (
                        tf.shape(band_x)[0],
                        tf.shape(band_x)[1],
                        band_feature_dim,
                    ),
                ),
                (None, None, band_feature_dim),
            )
            for band_x in band_node_features
        ]

        if not self.use_spectral_gru:
            return tf.concat(flattened, axis=-1)

        # MTLFuseNet graph sequence GS=(G_theta,G_alpha,G_beta,G_gamma).
        # Here we create that spectral sequence independently at each EEG
        # timestep so EEGProc keeps a temporal output sequence.
        spectral_sequence = tf.stack(flattened, axis=2)
        # (B, T, Bands, C*U) -> (B*T, Bands, C*U)
        batch_size = tf.shape(spectral_sequence)[0]
        time_steps = tf.shape(spectral_sequence)[1]
        spectral_sequence = tf.reshape(
            spectral_sequence,
            (
                batch_size * time_steps,
                self.n_bands,
                self.n_channels * self.gcn_units[-1],
            ),
        )

        x = self.spectral_gru(spectral_sequence, training=training)
        return tf.reshape(
            x,
            (batch_size, time_steps, self.spectral_gru_units),
        )

    def call(self, inputs, training: bool = False):
        inputs = tf.convert_to_tensor(inputs)

        if inputs.shape.rank != 3:
            raise ValueError(
                "GCNMTLEncoder expects (batch, timesteps, channels*bands); "
                f"got {inputs.shape}."
            )

        expected_features = self.n_channels * self.n_bands
        static_features = inputs.shape[-1]
        if static_features is not None and int(static_features) != expected_features:
            raise ValueError(
                f"Expected {expected_features} features, got {static_features}."
            )

        x = self.to_nodes(inputs)
        band_node_features = [
            self._encode_one_band(x, band_index, training)
            for band_index in range(self.n_bands)
        ]

        x = self._spectral_sequence_to_temporal_features(
            band_node_features, training
        )

        for pool, dropout in zip(
            self.temporal_pool_layers,
            self.temporal_dropout_layers,
        ):
            x = pool(x)
            x = dropout(x, training=training)

        return self.seq_emb(x)

    def get_adjacency_matrix(self) -> tf.Tensor:
        """Return the MTL-normalized shared adjacency used by the GCN."""
        if not self.shared_gcn_layers:
            raise RuntimeError("No GCN layers are configured.")
        return self.shared_gcn_layers[0].normalized_adjacency()

    def get_raw_adjacency_matrix(self) -> tf.Tensor:
        """Return the fixed pre-self-loop MI adjacency A."""
        if not self.shared_gcn_layers:
            raise RuntimeError("No GCN layers are configured.")
        return self.shared_gcn_layers[0].raw_adjacency()

    def get_band_features(
        self,
        inputs,
        training: bool = False,
    ) -> dict[str, tf.Tensor]:
        x = self.to_nodes(tf.convert_to_tensor(inputs))
        return {
            f"band_{band_index}": self._encode_one_band(
                x, band_index, training
            )
            for band_index in range(self.n_bands)
        }

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "adjacency": self._adjacency_init,
                "n_channels": self.n_channels,
                "n_bands": self.n_bands,
                "gcn_units": self.gcn_units,
                "temporal_pool_sizes": self.temporal_pool_sizes,
                "dropout": self.dropout_rate,
                "activation": self.activation,
                "use_batch_norm": self.use_batch_norm,
                "use_spectral_gru": self.use_spectral_gru,
                "spectral_gru_units": self.spectral_gru_units,
                "spectral_gru_dropout": self.spectral_gru_dropout,
                "graph_add_self_loops": self.graph_add_self_loops,
                "graph_symmetrize": self.graph_symmetrize,
                "graph_epsilon": self.graph_epsilon,
            }
        )
        return config


# Convenient import alias matching EEGProc's existing GCN modules.
GCNEncoder = GCNMTLEncoder


@tf.keras.utils.register_keras_serializable(package="eegproc")
class GCNMTLDecoder(tf.keras.Model):
    """Graph-aware decoder companion for GCNMTLEncoder.

    MTLFuseNet does not define this decoder. It exists only so EEGProc's joint
    VAE/reconstruction experiments can swap the encoder without losing a graph
    decoder. It uses the SAME fixed MI adjacency and Eq. (15) normalization,
    but should not be described as part of the original MTLFuseNet method.
    """

    def __init__(
        self,
        timesteps: int,
        n_channels: int,
        n_bands: int,
        t_down: int,
        gcn_units: tuple[int, ...],
        temporal_pool_sizes: tuple[int, ...] | None,
        adjacency,
        emb_dim: int = 128,
        dropout: float = 0.10,
        activation: str = "relu",
        use_batch_norm: bool = False,
        graph_add_self_loops: bool = True,
        graph_symmetrize: bool = True,
        graph_epsilon: float = 1e-8,
        name: str = "decoder_gcn_mtl",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.timesteps = int(timesteps)
        self.n_channels = int(n_channels)
        self.n_bands = int(n_bands)
        self.t_down = int(t_down)
        self.gcn_units = tuple(int(u) for u in gcn_units)
        self.temporal_pool_sizes = GCNMTLEncoder._normalize_temporal_pool_sizes(
            temporal_pool_sizes, self.t_down
        )
        self.emb_dim = int(emb_dim)
        self.dropout_rate = float(dropout)
        self.activation = activation
        self.use_batch_norm = bool(use_batch_norm)
        self.graph_add_self_loops = bool(graph_add_self_loops)
        self.graph_symmetrize = bool(graph_symmetrize)
        self.graph_epsilon = float(graph_epsilon)

        adjacency_array = np.asarray(adjacency, dtype=np.float32)
        if adjacency_array.shape != (self.n_channels, self.n_channels):
            raise ValueError(
                f"adjacency must be ({self.n_channels}, {self.n_channels}), "
                f"got {adjacency_array.shape}."
            )
        self._adjacency_init = adjacency_array.tolist()

        graph_seed_units = self.gcn_units[-1]

        self.input_projection = layers.Conv1D(
            graph_seed_units,
            kernel_size=1,
            padding="same",
            activation=self.activation,
            name="mtl_dec_input_projection",
        )

        self.upsample_layers = []
        self.temporal_conv_layers = []
        self.temporal_dropout_layers = []
        for i, pool_size in enumerate(reversed(self.temporal_pool_sizes)):
            self.upsample_layers.append(
                layers.UpSampling1D(size=pool_size, name=f"mtl_dec_upsample_{i}")
            )
            self.temporal_conv_layers.append(
                layers.Conv1D(
                    graph_seed_units,
                    kernel_size=3,
                    padding="same",
                    activation=self.activation,
                    name=f"mtl_dec_temporal_conv_{i}",
                )
            )
            self.temporal_dropout_layers.append(
                layers.Dropout(dropout, name=f"mtl_dec_temporal_do_{i}")
            )

        self.node_seed_projection = layers.Dense(
            self.n_channels * graph_seed_units,
            activation=None,
            name="mtl_dec_node_seed_projection",
        )

        decoder_units = tuple(reversed(self.gcn_units[:-1]))
        self.graph_layers = [
            GraphConvMTL(
                units=units,
                n_nodes=self.n_channels,
                adjacency=self._adjacency_init,
                activation=self.activation,
                add_self_loops=self.graph_add_self_loops,
                symmetrize=self.graph_symmetrize,
                epsilon=self.graph_epsilon,
                name=f"mtl_dec_gcn_{i}",
            )
            for i, units in enumerate(decoder_units)
        ]
        self.graph_dropout_layers = [
            layers.SpatialDropout2D(dropout, name=f"mtl_dec_graph_do_{i}")
            for i, _ in enumerate(decoder_units)
        ]

        self.output_graph = GraphConvMTL(
            units=self.n_bands,
            n_nodes=self.n_channels,
            adjacency=self._adjacency_init,
            activation=None,
            add_self_loops=self.graph_add_self_loops,
            symmetrize=self.graph_symmetrize,
            epsilon=self.graph_epsilon,
            name="mtl_dec_output_graph",
        )
        self.output_local = layers.Dense(
            self.n_bands,
            activation=None,
            name="mtl_dec_output_local",
        )

    @property
    def n_features(self) -> int:
        return self.n_channels * self.n_bands

    @classmethod
    def from_encoder(
        cls,
        encoder: GCNMTLEncoder,
        name: str = "decoder_gcn_mtl",
    ) -> "GCNMTLDecoder":
        if not isinstance(encoder, GCNMTLEncoder):
            raise TypeError(
                "GCNMTLDecoder.from_encoder requires GCNMTLEncoder, got "
                f"{type(encoder).__name__}."
            )

        return cls(
            timesteps=encoder.timesteps,
            n_channels=encoder.n_channels,
            n_bands=encoder.n_bands,
            t_down=encoder.t_down,
            gcn_units=encoder.gcn_units,
            temporal_pool_sizes=encoder.temporal_pool_sizes,
            adjacency=encoder._adjacency_init,
            emb_dim=encoder.emb_dim,
            dropout=encoder.dropout_rate,
            activation=encoder.activation,
            use_batch_norm=encoder.use_batch_norm,
            graph_add_self_loops=encoder.graph_add_self_loops,
            graph_symmetrize=encoder.graph_symmetrize,
            graph_epsilon=encoder.graph_epsilon,
            name=name,
        )

    def fix_length(self, x: tf.Tensor) -> tf.Tensor:
        x = x[:, : self.timesteps, :]
        current_timesteps = tf.shape(x)[1]
        pad_amount = tf.maximum(0, self.timesteps - current_timesteps)
        return tf.pad(x, [[0, 0], [0, pad_amount], [0, 0]])

    def call(self, inputs, training: bool = False):
        x = self.input_projection(inputs)

        for upsample, conv, dropout in zip(
            self.upsample_layers,
            self.temporal_conv_layers,
            self.temporal_dropout_layers,
        ):
            x = upsample(x)
            x = conv(x)
            x = dropout(x, training=training)

        x = self.node_seed_projection(x)
        batch_size = tf.shape(x)[0]
        time_steps = tf.shape(x)[1]
        x = tf.reshape(
            x,
            (batch_size, time_steps, self.n_channels, self.gcn_units[-1]),
        )

        for gcn, dropout in zip(self.graph_layers, self.graph_dropout_layers):
            x = gcn(x, training=training)
            x = dropout(x, training=training)

        x = self.output_graph(x, training=training) + self.output_local(x)
        x = tf.reshape(x, (batch_size, time_steps, self.n_features))
        return self.fix_length(x)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "timesteps": self.timesteps,
                "n_channels": self.n_channels,
                "n_bands": self.n_bands,
                "t_down": self.t_down,
                "gcn_units": self.gcn_units,
                "temporal_pool_sizes": self.temporal_pool_sizes,
                "adjacency": self._adjacency_init,
                "emb_dim": self.emb_dim,
                "dropout": self.dropout_rate,
                "activation": self.activation,
                "use_batch_norm": self.use_batch_norm,
                "graph_add_self_loops": self.graph_add_self_loops,
                "graph_symmetrize": self.graph_symmetrize,
                "graph_epsilon": self.graph_epsilon,
                "name": self.name,
            }
        )
        return config


GCNDecoder = GCNMTLDecoder
