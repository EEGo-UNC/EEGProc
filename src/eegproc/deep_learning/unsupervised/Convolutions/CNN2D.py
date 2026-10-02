import math

import tensorflow as tf
from tensorflow.keras import layers

from ..BaseEncoder import BaseEncoder
from ..utils import _product


@tf.keras.utils.register_keras_serializable(package="EEGProc")
class CNN2DEncoder(BaseEncoder):
    """2D CNN encoder that preserves channel-band position information.

    Inputs are reshaped from ``(batch, time, channels * bands)`` to
    ``(batch, time, channels, bands, 1)``. Spatial Conv2D blocks operate on
    each timestep independently. The remaining spatial grid is flattened in a
    fixed order and projected to ``emb_dim`` instead of being globally
    averaged, so electrode and frequency-band location remain available to the
    latent sequence.
    """

    def __init__(
        self,
        timesteps: int,
        t_down: int,
        n_channels: int = 14,
        n_bands: int = 4,
        conv_filters: tuple[int, ...] = (32, 64, 128),
        kernel_sizes: tuple[int, int] | tuple[tuple[int, int], ...] = (
            (3, 3),
            (3, 3),
            (2, 2),
        ),
        spatial_pool_sizes: (
            tuple[int, int]
            | tuple[tuple[int, int] | None, ...]
            | None
        ) = None,
        temporal_pool_sizes: tuple[int, ...] | None = None,
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

        if timesteps <= 0:
            raise ValueError(f"timesteps must be positive, got {timesteps}.")
        if n_channels <= 0:
            raise ValueError(f"n_channels must be positive, got {n_channels}.")
        if n_bands <= 0:
            raise ValueError(f"n_bands must be positive, got {n_bands}.")
        if not conv_filters or any(filters <= 0 for filters in conv_filters):
            raise ValueError(
                f"conv_filters must contain positive values, got {conv_filters}."
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        self.n_channels = int(n_channels)
        self.n_bands = int(n_bands)
        self.conv_filters = tuple(int(value) for value in conv_filters)
        self.kernel_sizes = self._normalize_2d_kernel_sizes(
            kernel_sizes,
            len(self.conv_filters),
        )
        self.spatial_pool_sizes = self._normalize_2d_pool_sizes(
            spatial_pool_sizes,
            len(self.conv_filters),
        )
        self.temporal_pool_sizes = self._normalize_temporal_pool_sizes(
            temporal_pool_sizes,
            self.t_down,
        )
        self.dropout_rate = float(dropout)
        self.activation = activation
        self.use_batch_norm = bool(use_batch_norm)

        self.spatial_shapes = self._compute_spatial_shapes(
            self.n_channels,
            self.n_bands,
            self.spatial_pool_sizes,
        )
        self.encoded_spatial_shape = self.spatial_shapes[-1]

        self.to_grid = layers.Reshape(
            (timesteps, n_channels, n_bands, 1),
            name="to_grid",
        )

        self.conv_layers = []
        self.bn_layers = []
        self.activation_layers = []
        self.dropout_layers = []
        self.spatial_pool_layers = []

        for i, (filters, kernel_size, spatial_pool_size) in enumerate(
            zip(
                self.conv_filters,
                self.kernel_sizes,
                self.spatial_pool_sizes,
            )
        ):
            self.conv_layers.append(
                layers.TimeDistributed(
                    layers.Conv2D(
                        filters,
                        kernel_size,
                        padding="same",
                        activation=None,
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
            self.activation_layers.append(
                layers.Activation(activation, name=f"enc_activation2d_{i}")
            )
            self.dropout_layers.append(
                layers.TimeDistributed(
                    layers.Dropout(dropout),
                    name=f"enc_do2d_{i}",
                )
            )
            self.spatial_pool_layers.append(
                layers.TimeDistributed(
                    layers.MaxPool2D(
                        pool_size=spatial_pool_size,
                        padding="same",
                    ),
                    name=f"enc_spool2d_{i}",
                )
                if spatial_pool_size is not None
                else None
            )

        self.spatial_flatten = layers.TimeDistributed(
            layers.Flatten(),
            name="enc_spatial_flatten",
        )
        self.spatial_projection = layers.TimeDistributed(
            layers.Dense(emb_dim, activation=None),
            name="enc_spatial_projection",
        )
        self.spatial_projection_bn = (
            layers.TimeDistributed(
                layers.BatchNormalization(),
                name="enc_spatial_projection_bn",
            )
            if use_batch_norm
            else None
        )
        self.spatial_projection_activation = layers.Activation(
            activation,
            name="enc_spatial_projection_activation",
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
            layers.Dropout(dropout, name=f"enc_tdo_{i}")
            for i, _ in enumerate(self.temporal_pool_sizes)
        ]
        self.seq_emb = layers.Conv1D(
            emb_dim,
            kernel_size=1,
            padding="same",
            activation=None,
            name="seq_emb",
        )

    @staticmethod
    def _normalize_temporal_pool_sizes(pool_sizes, t_down: int):
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
                f"t_down={t_down}, but temporal_pool_sizes produces "
                f"a downsampling factor of {effective_t_down}. "
                "Set t_down equal to product(temporal_pool_sizes)."
            )
        return normalized

    @staticmethod
    def _normalize_2d_kernel_sizes(kernel_sizes, n_layers: int):
        """Return one validated ``(height, width)`` pair per layer."""
        if (
            isinstance(kernel_sizes, (list, tuple))
            and len(kernel_sizes) == 2
            and all(isinstance(value, int) for value in kernel_sizes)
        ):
            pair = tuple(int(value) for value in kernel_sizes)
            if any(value < 1 for value in pair):
                raise ValueError(f"Kernel dimensions must be >= 1, got {pair}.")
            return tuple(pair for _ in range(n_layers))

        values = tuple(kernel_sizes)
        if len(values) != n_layers:
            raise ValueError(
                f"kernel_sizes must have length {n_layers}, got {len(values)}."
            )

        normalized = []
        for value in values:
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(
                    "Each kernel size must be a pair of integers; "
                    f"got {value!r}."
                )
            pair = tuple(int(dimension) for dimension in value)
            if any(dimension < 1 for dimension in pair):
                raise ValueError(
                    f"Kernel dimensions must be >= 1, got {pair!r}."
                )
            normalized.append(pair)
        return tuple(normalized)

    @staticmethod
    def _normalize_2d_pool_sizes(pool_sizes, n_layers: int):
        """Return one optional spatial-pool pair per convolutional layer."""
        if pool_sizes is None:
            return tuple(None for _ in range(n_layers))

        values = tuple(pool_sizes)
        if not values:
            return tuple(None for _ in range(n_layers))

        if len(values) == 2 and all(isinstance(value, int) for value in values):
            pair = tuple(int(value) for value in values)
            if any(value < 1 for value in pair):
                raise ValueError(
                    f"Spatial pool dimensions must be >= 1, got {pair!r}."
                )
            return tuple(pair for _ in range(n_layers))

        if len(values) != n_layers:
            raise ValueError(
                "spatial_pool_sizes must be one pair, empty/None, or contain "
                f"one pair per Conv2D layer ({n_layers}); got {values!r}."
            )

        normalized = []
        for value in values:
            if value is None:
                normalized.append(None)
                continue
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(
                    "Each spatial pool size must be None or a pair of integers; "
                    f"got {value!r}."
                )
            pair = tuple(int(dimension) for dimension in value)
            if any(dimension < 1 for dimension in pair):
                raise ValueError(
                    f"Spatial pool dimensions must be >= 1, got {pair!r}."
                )
            normalized.append(pair)
        return tuple(normalized)

    @staticmethod
    def _compute_spatial_shapes(n_channels, n_bands, pool_sizes):
        shapes = [(int(n_channels), int(n_bands))]
        height, width = shapes[0]
        for pool_size in pool_sizes:
            if pool_size is not None:
                height = math.ceil(height / pool_size[0])
                width = math.ceil(width / pool_size[1])
            shapes.append((height, width))
        return tuple(shapes)

    @property
    def n_features(self) -> int:
        return self.n_channels * self.n_bands

    def call(self, inputs, training: bool = False):
        x = self.to_grid(inputs)

        for conv, bn, activation, dropout, spatial_pool in zip(
            self.conv_layers,
            self.bn_layers,
            self.activation_layers,
            self.dropout_layers,
            self.spatial_pool_layers,
        ):
            x = conv(x)
            if bn is not None:
                x = bn(x, training=training)
            x = activation(x)
            x = dropout(x, training=training)
            if spatial_pool is not None:
                x = spatial_pool(x)

        x = self.spatial_flatten(x)
        x = self.spatial_projection(x)
        if self.spatial_projection_bn is not None:
            x = self.spatial_projection_bn(x, training=training)
        x = self.spatial_projection_activation(x)

        for pool, dropout in zip(
            self.temporal_pool_layers,
            self.temporal_dropout_layers,
        ):
            x = pool(x)
            x = dropout(x, training=training)

        return self.seq_emb(x)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "n_channels": self.n_channels,
                "n_bands": self.n_bands,
                "conv_filters": self.conv_filters,
                "kernel_sizes": self.kernel_sizes,
                "spatial_pool_sizes": self.spatial_pool_sizes,
                "temporal_pool_sizes": self.temporal_pool_sizes,
                "dropout": self.dropout_rate,
                "activation": self.activation,
                "use_batch_norm": self.use_batch_norm,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="EEGProc")
class CNN2DDecoder(tf.keras.Model):
    """Spatially structured mirror decoder for :class:`CNN2DEncoder`."""

    def __init__(
        self,
        timesteps: int,
        n_channels: int,
        n_bands: int,
        t_down: int,
        conv_filters: tuple[int, ...],
        temporal_pool_sizes: tuple[int, ...] | None,
        emb_dim: int = 128,
        dropout: float = 0.10,
        activation: str = "relu",
        use_batch_norm: bool = True,
        kernel_sizes: tuple[int, int] | tuple[tuple[int, int], ...] = (3, 3),
        spatial_pool_sizes: (
            tuple[int, int]
            | tuple[tuple[int, int] | None, ...]
            | None
        ) = None,
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
        if not conv_filters or any(filters <= 0 for filters in conv_filters):
            raise ValueError(
                f"conv_filters must contain positive values, got {conv_filters}."
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        self.timesteps = int(timesteps)
        self.n_channels = int(n_channels)
        self.n_bands = int(n_bands)
        self.t_down = int(t_down)
        self.conv_filters = tuple(int(value) for value in conv_filters)
        self.kernel_sizes = CNN2DEncoder._normalize_2d_kernel_sizes(
            kernel_sizes,
            len(self.conv_filters),
        )
        self.spatial_pool_sizes = CNN2DEncoder._normalize_2d_pool_sizes(
            spatial_pool_sizes,
            len(self.conv_filters),
        )
        self.temporal_pool_sizes = CNN2DEncoder._normalize_temporal_pool_sizes(
            temporal_pool_sizes,
            self.t_down,
        )
        self.emb_dim = int(emb_dim)
        self.dropout_rate = float(dropout)
        self.activation = activation
        self.use_batch_norm = bool(use_batch_norm)

        self.spatial_shapes = CNN2DEncoder._compute_spatial_shapes(
            self.n_channels,
            self.n_bands,
            self.spatial_pool_sizes,
        )
        self.encoded_spatial_shape = self.spatial_shapes[-1]
        grid_height, grid_width = self.encoded_spatial_shape
        grid_filters = self.conv_filters[-1]

        self.input_projection = layers.Conv1D(
            emb_dim,
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
                layers.UpSampling1D(size=pool_size, name=f"dec_upsample_{i}")
            )
            self.temporal_conv_layers.append(
                layers.Conv1D(
                    emb_dim,
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

        self.grid_seed_projection = layers.Dense(
            grid_height * grid_width * grid_filters,
            activation=None,
            name="dec_grid_seed_projection",
        )
        self.grid_seed_bn = (
            layers.BatchNormalization(name="dec_grid_seed_bn")
            if use_batch_norm
            else None
        )
        self.grid_seed_activation = layers.Activation(
            activation,
            name="dec_grid_seed_activation",
        )

        self.spatial_upsample_layers = []
        self.spatial_conv_layers = []
        self.spatial_bn_layers = []
        self.spatial_activation_layers = []
        self.spatial_dropout_layers = []

        for decoder_index, encoder_index in enumerate(
            reversed(range(len(self.conv_filters)))
        ):
            pool_size = self.spatial_pool_sizes[encoder_index]
            self.spatial_upsample_layers.append(
                layers.TimeDistributed(
                    layers.UpSampling2D(size=pool_size),
                    name=f"dec_spatial_upsample_{decoder_index}",
                )
                if pool_size is not None
                else None
            )

            output_filters = (
                self.conv_filters[encoder_index - 1]
                if encoder_index > 0
                else self.conv_filters[0]
            )
            self.spatial_conv_layers.append(
                layers.TimeDistributed(
                    layers.Conv2D(
                        output_filters,
                        kernel_size=self.kernel_sizes[encoder_index],
                        padding="same",
                        activation=None,
                    ),
                    name=f"dec_conv2d_{decoder_index}",
                )
            )
            self.spatial_bn_layers.append(
                layers.TimeDistributed(
                    layers.BatchNormalization(),
                    name=f"dec_bn2d_{decoder_index}",
                )
                if use_batch_norm
                else None
            )
            self.spatial_activation_layers.append(
                layers.Activation(
                    activation,
                    name=f"dec_activation2d_{decoder_index}",
                )
            )
            self.spatial_dropout_layers.append(
                layers.TimeDistributed(
                    layers.Dropout(dropout),
                    name=f"dec_do2d_{decoder_index}",
                )
            )

        self.x_hat_grid = layers.TimeDistributed(
            layers.Conv2D(
                1,
                kernel_size=1,
                padding="same",
                activation=None,
            ),
            name="x_hat_grid",
        )

    @property
    def n_features(self) -> int:
        return self.n_channels * self.n_bands

    @classmethod
    def from_encoder(
        cls,
        encoder: CNN2DEncoder,
        name: str = "decoder_2dcnn",
    ) -> "CNN2DDecoder":
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
            kernel_sizes=encoder.kernel_sizes,
            spatial_pool_sizes=encoder.spatial_pool_sizes,
            temporal_pool_sizes=encoder.temporal_pool_sizes,
            emb_dim=encoder.emb_dim,
            dropout=encoder.dropout_rate,
            activation=encoder.activation,
            use_batch_norm=encoder.use_batch_norm,
            name=name,
        )

    def fix_length(self, x: tf.Tensor) -> tf.Tensor:
        x = x[:, : self.timesteps, :]
        current_timesteps = tf.shape(x)[1]
        pad_amount = tf.maximum(0, self.timesteps - current_timesteps)
        return tf.pad(x, paddings=[[0, 0], [0, pad_amount], [0, 0]])

    def call(self, inputs, training: bool = False):
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

        x = self.grid_seed_projection(x)
        if self.grid_seed_bn is not None:
            x = self.grid_seed_bn(x, training=training)
        x = self.grid_seed_activation(x)

        batch_size = tf.shape(x)[0]
        time_steps = tf.shape(x)[1]
        grid_height, grid_width = self.encoded_spatial_shape
        x = tf.reshape(
            x,
            [
                batch_size,
                time_steps,
                grid_height,
                grid_width,
                self.conv_filters[-1],
            ],
        )

        for upsample, conv, bn, activation, dropout in zip(
            self.spatial_upsample_layers,
            self.spatial_conv_layers,
            self.spatial_bn_layers,
            self.spatial_activation_layers,
            self.spatial_dropout_layers,
        ):
            if upsample is not None:
                x = upsample(x)
            x = conv(x)
            if bn is not None:
                x = bn(x, training=training)
            x = activation(x)
            x = dropout(x, training=training)

        x = x[:, :, : self.n_channels, : self.n_bands, :]
        x = self.x_hat_grid(x)
        x = tf.reshape(x, [batch_size, time_steps, self.n_features])
        return self.fix_length(x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.timesteps, self.n_features)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "timesteps": self.timesteps,
                "n_channels": self.n_channels,
                "n_bands": self.n_bands,
                "t_down": self.t_down,
                "conv_filters": self.conv_filters,
                "kernel_sizes": self.kernel_sizes,
                "spatial_pool_sizes": self.spatial_pool_sizes,
                "temporal_pool_sizes": self.temporal_pool_sizes,
                "emb_dim": self.emb_dim,
                "dropout": self.dropout_rate,
                "activation": self.activation,
                "use_batch_norm": self.use_batch_norm,
                "name": self.name,
            }
        )
        return config
