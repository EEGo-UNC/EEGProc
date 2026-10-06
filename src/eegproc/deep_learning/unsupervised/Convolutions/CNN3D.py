"""MTLFuseNet-style spatiotemporal 3D CNN components for DREAMER EEG.

The DREAMER converter stores each timestep in channel-major, band-minor order::

    AF3_theta, AF3_alpha, AF3_beta, F7_theta, ...

The encoder sums the bandpass components back into a raw-like 4--30 Hz signal,
places the 14 Emotiv EPOC electrodes on MTLFuseNet's 9 x 9 scalp grid, and
convolves over time and both scalp axes. Spatial pooling never changes the time
axis; optional 1D temporal pooling provides the standard :class:`BaseEncoder`
``t_down`` contract.

Reference: R. Li et al., "MTLFuseNet: A novel emotion recognition model based
on deep latent feature fusion of EEG signals and multi-task learning",
Knowledge-Based Systems 276 (2023), 110756.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from ..BaseEncoder import BaseEncoder
from ..utils import _product


# DREAMER and AMIGOS use this 14-channel Emotiv EPOC order throughout EEGProc.
DREAMER_ELECTRODE_GRID = (
    (1, 2),  # AF3
    (2, 0),  # F7
    (2, 2),  # F3
    (3, 1),  # FC5
    (4, 0),  # T7
    (6, 0),  # P7
    (8, 2),  # O1
    (8, 4),  # O2
    (6, 8),  # P8
    (4, 8),  # T8
    (3, 7),  # FC6
    (2, 6),  # F4
    (2, 8),  # F8
    (1, 4),  # AF4
)


def _positive_int_tuple(name: str, values: Sequence[int]) -> tuple[int, ...]:
    resolved = tuple(int(value) for value in values)
    if not resolved or any(value < 1 for value in resolved):
        raise ValueError(f"{name} must contain positive integers.")
    return resolved


@tf.keras.utils.register_keras_serializable(package="EEGProc")
class MTLFuseNet3DCNNEncoder(BaseEncoder):
    """Encode channel-band DREAMER waveforms with spatiotemporal Conv3D blocks.

    Parameters
    ----------
    timesteps : int
        Number of samples in each input window.
    t_down : int
        Temporal downsampling factor. This must equal the product of
        ``temporal_pool_sizes``.
    n_channels : int, default=14
        Number of electrodes. The default matches DREAMER.
    n_bands : int, default=3
        Number of contiguous bandpass components per channel. The default
        matches the theta/alpha/beta output of ``prepare_dreamer``.
    electrode_grid : sequence of pairs
        One ``(row, column)`` position per channel.

    Notes
    -----
    Input shape is ``(batch, timesteps, n_channels * n_bands)``. Features must
    be channel-major and band-minor, matching EEGProc's DREAMER converter.
    """

    def __init__(
        self,
        timesteps: int,
        t_down: int = 1,
        n_channels: int = 14,
        n_bands: int = 3,
        grid_size: int = 9,
        electrode_grid: Sequence[Sequence[int]] = DREAMER_ELECTRODE_GRID,
        conv_filters: Sequence[int] = (32, 64, 128),
        temporal_kernel_size: int = 7,
        spatial_kernel_size: int = 3,
        spatial_pool_sizes: Sequence[int] = (2, 2, 1),
        temporal_pool_sizes: Sequence[int] | None = None,
        emb_dim: int = 128,
        dropout: float = 0.20,
        activation: str = "relu",
        use_layer_norm: bool = True,
        name: str = "encoder_mtlfusenet_3dcnn",
        **kwargs,
    ):
        super().__init__(
            timesteps=int(timesteps),
            emb_dim=int(emb_dim),
            t_down=int(t_down),
            name=name,
            **kwargs,
        )
        self.n_channels = int(n_channels)
        self.n_bands = int(n_bands)
        self.grid_size = int(grid_size)
        self.electrode_grid = tuple(
            (int(position[0]), int(position[1])) for position in electrode_grid
        )
        self.conv_filters = _positive_int_tuple("conv_filters", conv_filters)
        self.temporal_kernel_size = int(temporal_kernel_size)
        self.spatial_kernel_size = int(spatial_kernel_size)
        self.spatial_pool_sizes = tuple(int(value) for value in spatial_pool_sizes)
        self.temporal_pool_sizes = self._normalize_temporal_pool_sizes(
            temporal_pool_sizes, self.t_down
        )
        self.dropout_rate = float(dropout)
        self.activation_name = str(activation)
        self.use_layer_norm = bool(use_layer_norm)

        self._validate_configuration()
        projection = np.zeros(
            (self.n_channels, self.grid_size * self.grid_size), dtype=np.float32
        )
        for channel, (row, column) in enumerate(self.electrode_grid):
            projection[channel, row * self.grid_size + column] = 1.0
        self._grid_projection = tf.constant(projection, dtype=tf.float32)

        kernel_size = (
            self.temporal_kernel_size,
            self.spatial_kernel_size,
            self.spatial_kernel_size,
        )
        self.convolutions = [
            layers.Conv3D(
                filters=n_filters,
                kernel_size=kernel_size,
                padding="same",
                use_bias=not self.use_layer_norm,
                name=f"spatiotemporal_conv3d_{index}",
            )
            for index, n_filters in enumerate(self.conv_filters)
        ]
        self.normalizations = [
            layers.LayerNormalization(
                axis=-1, name=f"spatiotemporal_layer_norm_{index}"
            )
            if self.use_layer_norm
            else None
            for index in range(len(self.conv_filters))
        ]
        self.activations = [
            layers.Activation(
                self.activation_name, name=f"spatiotemporal_activation_{index}"
            )
            for index in range(len(self.conv_filters))
        ]
        self.spatial_pool_layers = [
            None
            if pool_size == 1
            else layers.MaxPool3D(
                pool_size=(1, pool_size, pool_size),
                padding="same",
                name=f"spatial_pool3d_{index}",
            )
            for index, pool_size in enumerate(self.spatial_pool_sizes)
        ]
        self.spatial_dropouts = [
            layers.SpatialDropout3D(
                self.dropout_rate, name=f"spatiotemporal_dropout_{index}"
            )
            for index in range(len(self.conv_filters))
        ]
        self.temporal_pool_layers = [
            layers.MaxPool1D(pool_size, padding="same", name=f"temporal_pool_{index}")
            for index, pool_size in enumerate(self.temporal_pool_sizes)
        ]
        self.temporal_dropouts = [
            layers.Dropout(self.dropout_rate, name=f"temporal_dropout_{index}")
            for index in range(len(self.temporal_pool_sizes))
        ]
        self.embedding_projection = layers.Conv1D(
            self.emb_dim,
            kernel_size=1,
            padding="same",
            name="embedding_projection",
        )

    @staticmethod
    def _normalize_temporal_pool_sizes(
        pool_sizes: Sequence[int] | None, t_down: int
    ) -> tuple[int, ...]:
        if pool_sizes is None:
            normalized = () if t_down == 1 else (int(t_down),)
        else:
            normalized = tuple(int(value) for value in pool_sizes)
        if any(value < 1 for value in normalized):
            raise ValueError("temporal_pool_sizes must contain positive integers.")
        effective_t_down = _product(normalized) if normalized else 1
        if effective_t_down != t_down:
            raise ValueError(
                f"t_down={t_down}, but temporal_pool_sizes gives {effective_t_down}."
            )
        return normalized

    def _validate_configuration(self) -> None:
        if self.n_channels < 1 or self.n_bands < 1 or self.grid_size < 1:
            raise ValueError("n_channels, n_bands, and grid_size must be positive.")
        if len(self.electrode_grid) != self.n_channels:
            raise ValueError(
                "electrode_grid must provide exactly one coordinate per channel; "
                f"got {len(self.electrode_grid)} for {self.n_channels} channels."
            )
        if len(set(self.electrode_grid)) != len(self.electrode_grid):
            raise ValueError("electrode_grid coordinates must be unique.")
        if any(
            row < 0 or row >= self.grid_size or column < 0 or column >= self.grid_size
            for row, column in self.electrode_grid
        ):
            raise ValueError("electrode_grid coordinates must lie inside the grid.")
        if self.temporal_kernel_size < 1 or self.spatial_kernel_size < 1:
            raise ValueError("3D-CNN kernel sizes must be positive.")
        if len(self.spatial_pool_sizes) != len(self.conv_filters):
            raise ValueError("spatial_pool_sizes must have one value per Conv3D block.")
        if any(value < 1 for value in self.spatial_pool_sizes):
            raise ValueError("spatial_pool_sizes must contain positive integers.")
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("dropout must be in [0, 1).")

    @property
    def n_features(self) -> int:
        return self.n_channels * self.n_bands

    def to_spatial_grid(self, inputs) -> tf.Tensor:
        """Return ``(batch, time, grid, grid, 1)`` for inspection or reuse."""
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        if inputs.shape.rank != 3:
            raise ValueError(
                "MTLFuseNet3DCNNEncoder expects (batch, timesteps, features); "
                f"got {inputs.shape}."
            )
        if inputs.shape[-1] is not None and int(inputs.shape[-1]) != self.n_features:
            raise ValueError(
                f"Input features={inputs.shape[-1]}, expected {self.n_features}."
            )
        tf.debugging.assert_equal(
            tf.shape(inputs)[-1],
            self.n_features,
            message="3D-CNN input feature width is inconsistent.",
        )
        shape = tf.shape(inputs)
        channel_band = tf.reshape(
            inputs, (shape[0], shape[1], self.n_channels, self.n_bands)
        )
        raw_like_channels = tf.reduce_sum(channel_band, axis=-1, keepdims=True)
        spatial = tf.einsum(
            "ntck,cg->ntgk", raw_like_channels, self._grid_projection
        )
        return tf.reshape(
            spatial, (shape[0], shape[1], self.grid_size, self.grid_size, 1)
        )

    # The old research branch exposed this private spelling. Keep it as a
    # lightweight compatibility hook for downstream notebooks.
    _to_spatial_grid = to_spatial_grid

    def call(self, inputs, training: bool = False):
        x = self.to_spatial_grid(inputs)
        for convolution, normalization, activation, pool, dropout in zip(
            self.convolutions,
            self.normalizations,
            self.activations,
            self.spatial_pool_layers,
            self.spatial_dropouts,
        ):
            x = convolution(x)
            if normalization is not None:
                x = normalization(x)
            x = activation(x)
            if pool is not None:
                x = pool(x)
            x = dropout(x, training=training)

        # Preserve time and aggregate only the two scalp axes.
        x = tf.reduce_mean(x, axis=(2, 3))
        for pool, dropout in zip(self.temporal_pool_layers, self.temporal_dropouts):
            x = pool(x)
            x = dropout(x, training=training)
        return self.embedding_projection(x)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "n_channels": self.n_channels,
                "n_bands": self.n_bands,
                "grid_size": self.grid_size,
                "electrode_grid": self.electrode_grid,
                "conv_filters": self.conv_filters,
                "temporal_kernel_size": self.temporal_kernel_size,
                "spatial_kernel_size": self.spatial_kernel_size,
                "spatial_pool_sizes": self.spatial_pool_sizes,
                "temporal_pool_sizes": self.temporal_pool_sizes,
                "dropout": self.dropout_rate,
                "activation": self.activation_name,
                "use_layer_norm": self.use_layer_norm,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="EEGProc")
class MTLFuseNet3DCNNDecoder(tf.keras.Model):
    """Reconstruct channel-band sequences from 3D-CNN latent sequences."""

    def __init__(
        self,
        timesteps: int,
        n_channels: int,
        n_bands: int,
        t_down: int,
        temporal_pool_sizes: Sequence[int] | None,
        emb_dim: int = 128,
        dropout: float = 0.20,
        activation: str = "relu",
        name: str = "decoder_mtlfusenet_3dcnn",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.timesteps = int(timesteps)
        self.n_channels = int(n_channels)
        self.n_bands = int(n_bands)
        self.t_down = int(t_down)
        self.temporal_pool_sizes = (
            MTLFuseNet3DCNNEncoder._normalize_temporal_pool_sizes(
                temporal_pool_sizes, self.t_down
            )
        )
        self.emb_dim = int(emb_dim)
        self.dropout_rate = float(dropout)
        self.activation_name = str(activation)
        if self.timesteps < 1 or self.n_channels < 1 or self.n_bands < 1:
            raise ValueError("timesteps, n_channels, and n_bands must be positive.")
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.upsampling_layers = [
            layers.UpSampling1D(pool_size, name=f"temporal_upsample_{index}")
            for index, pool_size in enumerate(reversed(self.temporal_pool_sizes))
        ]
        self.temporal_convolutions = [
            layers.Conv1D(
                self.emb_dim,
                kernel_size=3,
                padding="same",
                activation=self.activation_name,
                name=f"temporal_reconstruction_{index}",
            )
            for index in range(len(self.temporal_pool_sizes))
        ]
        self.dropouts = [
            layers.Dropout(self.dropout_rate, name=f"temporal_dropout_{index}")
            for index in range(len(self.temporal_pool_sizes))
        ]
        self.output_projection = layers.Conv1D(
            self.n_features,
            kernel_size=1,
            padding="same",
            name="channel_band_reconstruction",
        )

    @property
    def n_features(self) -> int:
        return self.n_channels * self.n_bands

    @classmethod
    def from_encoder(
        cls, encoder: MTLFuseNet3DCNNEncoder, name: str = "decoder_mtlfusenet_3dcnn"
    ) -> "MTLFuseNet3DCNNDecoder":
        if not isinstance(encoder, MTLFuseNet3DCNNEncoder):
            raise TypeError(
                "MTLFuseNet3DCNNDecoder.from_encoder requires "
                f"MTLFuseNet3DCNNEncoder, got {type(encoder).__name__}."
            )
        return cls(
            timesteps=encoder.timesteps,
            n_channels=encoder.n_channels,
            n_bands=encoder.n_bands,
            t_down=encoder.t_down,
            temporal_pool_sizes=encoder.temporal_pool_sizes,
            emb_dim=encoder.emb_dim,
            dropout=encoder.dropout_rate,
            activation=encoder.activation_name,
            name=name,
        )

    def call(self, inputs, training: bool = False):
        x = inputs
        for upsampling, convolution, dropout in zip(
            self.upsampling_layers, self.temporal_convolutions, self.dropouts
        ):
            x = upsampling(x)
            x = convolution(x)
            x = dropout(x, training=training)
        x = self.output_projection(x)
        x = x[:, : self.timesteps, :]
        pad = tf.maximum(0, self.timesteps - tf.shape(x)[1])
        return tf.pad(x, [[0, 0], [0, pad], [0, 0]])

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "timesteps": self.timesteps,
                "n_channels": self.n_channels,
                "n_bands": self.n_bands,
                "t_down": self.t_down,
                "temporal_pool_sizes": self.temporal_pool_sizes,
                "emb_dim": self.emb_dim,
                "dropout": self.dropout_rate,
                "activation": self.activation_name,
            }
        )
        return config


# Short names consistent with CNN1D.py and CNN2D.py.
CNN3DEncoder = MTLFuseNet3DCNNEncoder
CNN3DDecoder = MTLFuseNet3DCNNDecoder
