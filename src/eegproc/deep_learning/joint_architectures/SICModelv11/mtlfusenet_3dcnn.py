"""MTLFuseNet-style spatio-temporal 3D-CNN encoder for DREAMER EEG.

SIC stores each timestep in channel-major, band-minor order::

    AF3_theta, AF3_alpha, AF3_beta, F7_theta, ...

This layer restores the 14 electrodes to the 9 x 9 scalp grid used by the
repository's MTLFuseNet preprocessing, keeps the three frequency bands as
input channels, and applies convolutions over time and the two scalp axes.
Only the spatial axes are pooled, so the output remains a feature sequence
with one vector per input timestep and can be concatenated directly with the
GCN-GRU branch.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import tensorflow as tf


# DREAMER/AMIGOS use the same 14-channel Emotiv EPOC ordering throughout
# EEGProc. Coordinates match supervised/mtlfusenet/preprocessing.py.
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
class MTLFuseNet3DCNNEncoder(tf.keras.layers.Layer):
    """Encode channel-band waveforms with spatio-temporal 3D convolutions.

    Parameters are intentionally compact relative to the paper's 2D VAE
    stack. SIC passes full 128-sample sequences to every encoder and later
    joins all windows into a trial sequence, so preserving time here is more
    useful (and substantially less memory-intensive) than flattening the
    complete 9 x 9 x 128 volume.
    """

    def __init__(
        self,
        *,
        n_channels: int = 14,
        n_bands: int = 3,
        grid_size: int = 9,
        electrode_grid: Sequence[Sequence[int]] = DREAMER_ELECTRODE_GRID,
        filters: Sequence[int] = (32, 64, 128),
        temporal_kernel_size: int = 7,
        spatial_kernel_size: int = 3,
        spatial_pool_sizes: Sequence[int] = (2, 2, 1),
        dropout: float = 0.20,
        activation: str = "relu",
        name: str = "sic_mtlfusenet_3dcnn_encoder",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.n_channels = int(n_channels)
        self.n_bands = int(n_bands)
        self.grid_size = int(grid_size)
        self.electrode_grid = tuple(
            (int(position[0]), int(position[1])) for position in electrode_grid
        )
        self.filters = _positive_int_tuple("filters", filters)
        self.temporal_kernel_size = int(temporal_kernel_size)
        self.spatial_kernel_size = int(spatial_kernel_size)
        self.spatial_pool_sizes = tuple(int(value) for value in spatial_pool_sizes)
        self.dropout_rate = float(dropout)
        self.activation_name = str(activation)

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
            row < 0 or row >= self.grid_size or col < 0 or col >= self.grid_size
            for row, col in self.electrode_grid
        ):
            raise ValueError("electrode_grid coordinates must lie inside the grid.")
        if self.temporal_kernel_size < 1 or self.spatial_kernel_size < 1:
            raise ValueError("3D-CNN kernel sizes must be positive.")
        if len(self.spatial_pool_sizes) != len(self.filters):
            raise ValueError("spatial_pool_sizes must have one value per filter block.")
        if any(value < 1 for value in self.spatial_pool_sizes):
            raise ValueError("spatial_pool_sizes must contain positive integers.")
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        projection = np.zeros(
            (self.n_channels, self.grid_size * self.grid_size), dtype=np.float32
        )
        for channel, (row, col) in enumerate(self.electrode_grid):
            projection[channel, row * self.grid_size + col] = 1.0
        self._grid_projection = tf.constant(projection, dtype=tf.float32)

        self.convolutions: list[tf.keras.layers.Layer] = []
        self.normalizations: list[tf.keras.layers.Layer] = []
        self.pooling_layers: list[tf.keras.layers.Layer | None] = []
        self.dropouts: list[tf.keras.layers.Layer] = []
        kernel_size = (
            self.temporal_kernel_size,
            self.spatial_kernel_size,
            self.spatial_kernel_size,
        )
        for index, (n_filters, spatial_pool) in enumerate(
            zip(self.filters, self.spatial_pool_sizes)
        ):
            self.convolutions.append(
                tf.keras.layers.Conv3D(
                    filters=n_filters,
                    kernel_size=kernel_size,
                    padding="same",
                    use_bias=False,
                    name=f"spatiotemporal_conv3d_{index}",
                )
            )
            # LayerNorm is stable for SIC's small subject-disjoint batches.
            self.normalizations.append(
                tf.keras.layers.LayerNormalization(
                    axis=-1, name=f"spatiotemporal_layer_norm_{index}"
                )
            )
            self.pooling_layers.append(
                None
                if spatial_pool == 1
                else tf.keras.layers.MaxPool3D(
                    pool_size=(1, spatial_pool, spatial_pool),
                    padding="same",
                    name=f"spatial_pool3d_{index}",
                )
            )
            self.dropouts.append(
                tf.keras.layers.SpatialDropout3D(
                    self.dropout_rate, name=f"spatiotemporal_dropout_{index}"
                )
            )
        self.activation = tf.keras.layers.Activation(
            self.activation_name, name="spatiotemporal_activation"
        )

    @property
    def output_dim(self) -> int:
        return self.filters[-1]

    def build(self, input_shape):
        expected_features = self.n_channels * self.n_bands
        if input_shape[-1] is not None and int(input_shape[-1]) != expected_features:
            raise ValueError(
                f"Input features={input_shape[-1]}, expected {expected_features}."
            )
        shape = tf.TensorShape(
            (input_shape[0], input_shape[1], self.grid_size, self.grid_size, self.n_bands)
        )
        for convolution, normalization, pooling, dropout in zip(
            self.convolutions,
            self.normalizations,
            self.pooling_layers,
            self.dropouts,
        ):
            convolution.build(shape)
            shape = convolution.compute_output_shape(shape)
            normalization.build(shape)
            if pooling is not None:
                pooling.build(shape)
                shape = pooling.compute_output_shape(shape)
            dropout.build(shape)
        super().build(input_shape)

    def _to_spatial_grid(self, inputs):
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        if inputs.shape.rank != 3:
            raise ValueError(
                "MTLFuseNet3DCNNEncoder expects (batch, timesteps, features); "
                f"got {inputs.shape}."
            )
        expected_features = self.n_channels * self.n_bands
        if inputs.shape[-1] is not None and inputs.shape[-1] != expected_features:
            raise ValueError(
                f"Input features={inputs.shape[-1]}, expected {self.n_channels}*"
                f"{self.n_bands}={expected_features}."
            )
        tf.debugging.assert_equal(
            tf.shape(inputs)[-1],
            expected_features,
            message="3D-CNN input feature width is inconsistent.",
        )
        shape = tf.shape(inputs)
        channel_band = tf.reshape(
            inputs, (shape[0], shape[1], self.n_channels, self.n_bands)
        )
        spatial = tf.einsum("ntck,cg->ntgk", channel_band, self._grid_projection)
        return tf.reshape(
            spatial,
            (shape[0], shape[1], self.grid_size, self.grid_size, self.n_bands),
        )

    def call(self, inputs, training: bool = False):
        x = self._to_spatial_grid(inputs)
        for convolution, normalization, pooling, dropout in zip(
            self.convolutions,
            self.normalizations,
            self.pooling_layers,
            self.dropouts,
        ):
            x = convolution(x)
            x = normalization(x)
            x = self.activation(x)
            if pooling is not None:
                x = pooling(x)
            x = dropout(x, training=training)
        # Retain time; aggregate only the two spatial dimensions.
        return tf.reduce_mean(x, axis=(2, 3))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "n_channels": self.n_channels,
                "n_bands": self.n_bands,
                "grid_size": self.grid_size,
                "electrode_grid": self.electrode_grid,
                "filters": self.filters,
                "temporal_kernel_size": self.temporal_kernel_size,
                "spatial_kernel_size": self.spatial_kernel_size,
                "spatial_pool_sizes": self.spatial_pool_sizes,
                "dropout": self.dropout_rate,
                "activation": self.activation_name,
            }
        )
        return config
