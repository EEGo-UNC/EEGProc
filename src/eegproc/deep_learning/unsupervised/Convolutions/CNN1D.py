from collections.abc import Sequence

import tensorflow as tf
from tensorflow.keras import layers

from ..BaseEncoder import BaseEncoder
from ..utils import _ensure_tuple, _product



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


class CNN1DDecoder(tf.keras.Model):
    """Mirror decoder for ``CNN1DEncoder``.

    This decoder expects a latent sequence of shape
    ``(batch, ceil(timesteps / t_down), emb_dim)`` and reconstructs a sequence
    of shape ``(batch, timesteps, n_features)``.

    The decoder mirrors the encoder by reversing the convolutional filter
    schedule and replacing temporal pooling with temporal upsampling.
    """
    def __init__(
        self,
        timesteps: int,
        n_features: int,
        t_down: int,
        conv_filters: tuple[int, ...],
        kernel_sizes: int | tuple[int, ...],
        pool_after_layers: tuple[int, ...],
        pool_sizes: int | tuple[int, ...],
        emb_dim: int,
        dropout: float = 0.10,
        activation: str = "relu",
        use_batch_norm: bool = True,
        name: str = "decoder_mirror",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        if timesteps <= 0:
            raise ValueError(f"timesteps must be positive, got {timesteps}.")
        if n_features <= 0:
            raise ValueError(f"n_features must be positive, got {n_features}.")
        if t_down <= 0:
            raise ValueError(f"t_down must be positive, got {t_down}.")
        if len(conv_filters) == 0:
            raise ValueError("conv_filters must contain at least one layer.")

        self.timesteps = timesteps
        self._n_features = n_features
        self.t_down = t_down
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
        self.emb_dim = emb_dim
        self.dropout_rate = dropout
        self.activation = activation
        self.use_batch_norm = use_batch_norm

        effective_t_down = _product(self.pool_sizes)
        if effective_t_down != self.t_down:
            raise ValueError(
                f"t_down={self.t_down}, but the configured pooling produces "
                f"a downsampling factor of {effective_t_down}. "
                "Set t_down equal to product(pool_sizes)."
            )

        self.input_projection = layers.Conv1D(
            self.conv_filters[-1],
            1,
            padding="same",
            activation=activation,
            name="dec_input_projection",
        )

        self.upsample_layers = {
            i: layers.UpSampling1D(size=pool_size, name=f"dec_upsample1d_{i}")
            for i, pool_size in zip(
                reversed(self.pool_after_layers),
                reversed(self.pool_sizes),
            )
        }

        self.conv_layers = [
            layers.Conv1D(
                filters,
                kernel_size,
                padding="same",
                activation=activation,
                name=f"dec_conv1d_{i}",
            )
            for i, (filters, kernel_size) in enumerate(
                zip(self.conv_filters, self.kernel_sizes)
            )
        ]

        self.bn_layers = [
            layers.BatchNormalization(name=f"dec_bn1d_{i}") if use_batch_norm else None
            for i, _ in enumerate(self.conv_filters)
        ]

        self.dropout_layers = [
            layers.Dropout(dropout, name=f"dec_do1d_{i}")
            for i, _ in enumerate(self.conv_filters)
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
        return self._n_features

    @classmethod
    def from_encoder(
        cls,
        encoder,
        name: str = "decoder_mirror",
    ):
        """Create a mirror decoder from a configured ``CNN1DEncoder``."""
        pool_sizes = getattr(encoder, "pool_sizes")

        return cls(
            timesteps=encoder.timesteps,
            n_features=encoder.n_features,
            t_down=encoder.t_down,
            conv_filters=encoder.conv_filters,
            kernel_sizes=encoder.kernel_sizes,
            pool_after_layers=encoder.pool_after_layers,
            pool_sizes=pool_sizes,
            emb_dim=encoder.emb_dim,
            dropout=encoder.dropout_rate,
            activation=encoder.activation,
            use_batch_norm=encoder.use_batch_norm,
            name=name,
        )

    def fix_length(self, x: tf.Tensor) -> tf.Tensor:
        """Trim or pad sequence to exactly ``timesteps``."""
        x = x[:, : self.timesteps, :]
        current_timesteps = tf.shape(x)[1]
        pad_amount = tf.maximum(0, self.timesteps - current_timesteps)

        return tf.pad(
            x,
            paddings=[[0, 0], [0, pad_amount], [0, 0]],
        )

    def call(self, inputs, training: bool = False):
        x = self.input_projection(inputs)

        for i in reversed(range(len(self.conv_filters))):
            if i in self.upsample_layers:
                x = self.upsample_layers[i](x)

            x = self.conv_layers[i](x)

            if self.bn_layers[i] is not None:
                x = self.bn_layers[i](x, training=training)

            x = self.dropout_layers[i](x, training=training)

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
                "n_features": self.n_features,
                "t_down": self.t_down,
                "conv_filters": self.conv_filters,
                "kernel_sizes": self.kernel_sizes,
                "pool_after_layers": self.pool_after_layers,
                "pool_sizes": self.pool_sizes,
                "emb_dim": self.emb_dim,
                "dropout": self.dropout_rate,
                "activation": self.activation,
                "use_batch_norm": self.use_batch_norm,
            }
        )
        return config
    

