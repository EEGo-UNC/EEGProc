import math
import tensorflow as tf


class BaseEncoder(tf.keras.Model):
    """Base class for EEG encoders.

    Encoders map raw EEG sequences to latent sequence representations for
    downstream tasks such as reconstruction, classification, contrastive
    learning, or sequence modeling.

    All subclasses should map an input sequence of shape
    ``(batch, timesteps, n_features)`` to a latent sequence of shape
    ``(batch, ceil(timesteps / t_down), emb_dim)``.

    Parameters
    ----------
    timesteps : int
        Number of timesteps in the input sequence.
    emb_dim : int
        Dimensionality of the latent embedding at each output timestep.
    t_down : int
        Temporal downsampling factor (divides timesteps by this value).
    name : str, default="base_encoder"
        Name of the Keras model.
    **kwargs
        Additional keyword arguments passed to ``tf.keras.Model``.
    """

    def __init__(
        self,
        timesteps: int,
        emb_dim: int,
        t_down: int,
        name: str = "base_encoder",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        if timesteps <= 0:
            raise ValueError(f"timesteps must be positive, got {timesteps}.")
        if emb_dim <= 0:
            raise ValueError(f"emb_dim must be positive, got {emb_dim}.")
        if t_down <= 0:
            raise ValueError(f"t_down must be positive, got {t_down}.")

        self.timesteps = timesteps
        self._emb_dim = emb_dim
        self.t_down = t_down

    @property
    def n_features(self) -> int:
        """Number of input features per timestep."""
        raise NotImplementedError("Subclasses must implement n_features.")

    @property
    def emb_dim(self) -> int:
        """Dimensionality of the output embedding."""
        return self._emb_dim

    @property
    def output_timesteps(self) -> int:
        """Number of output timesteps after downsampling."""
        return math.ceil(self.timesteps / self.t_down)

    def call(self, inputs, training: bool = False):
        """Run the encoder forward pass."""
        raise NotImplementedError("Subclasses must implement call().")

    def compute_output_shape(self, input_shape):
        """Return the encoder output shape."""
        return (input_shape[0], self.output_timesteps, self.emb_dim)

    def get_config(self) -> dict:
        """Return serializable configuration for the base encoder."""
        config = super().get_config()
        config.update(
            {
                "timesteps": self.timesteps,
                "emb_dim": self.emb_dim,
                "t_down": self.t_down,
                "name": self.name,
            }
        )
        return config