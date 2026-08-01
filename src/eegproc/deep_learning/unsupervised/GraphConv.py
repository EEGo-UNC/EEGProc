"""Graph convolution layer used by EEGProc GCN encoders and decoders."""

import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="eegproc")
class GraphConv(layers.Layer):
    """Graph convolution with a shared, learnable electrode adjacency matrix.

    The graph nodes are EEG electrodes. Any frequency bands are node features,
    so the same ``n_nodes x n_nodes`` electrode graph is applied to every band
    before the learned feature projection mixes the band/features dimension.
    Inputs may contain any number of leading dimensions, including the native
    sequence shape ``(batch, time, n_nodes, features)``; graph operations are
    broadcast over all leading dimensions without ``TimeDistributed``.

    The normalized adjacency is

    ``A = (1 - identity_mix) * softmax(A_theta + self_loop_bias * I)``
    ``    + identity_mix * I``.

    ``self_loop_bias`` makes training start from an electrode-preserving graph,
    while ``identity_mix`` can guarantee a permanent local contribution. A
    small entropy penalty discourages the adjacency rows from remaining close
    to uniform global averaging.

    Parameters
    ----------
    units : int
        Output feature dimensionality per electrode.
    n_nodes : int
        Number of graph nodes/electrodes.
    activation : str or callable or None, default="relu"
        Activation applied after graph aggregation and linear projection.
    use_bias : bool, default=True
        Whether to add a feature bias.
    self_loop_bias : float, default=2.0
        Positive value added to the diagonal adjacency logits before softmax.
        With 14 electrodes, 2.0 starts with a diagonal weight around 0.36
        before any optional identity mixture.
    identity_mix : float, default=0.0
        Fixed proportion of the identity matrix mixed into the learned graph.
        Must be in ``[0, 1]``. The GCN already has residual blocks, so zero is
        the recommended initial setting.
    adjacency_reg_weight : float, default=1e-4
        Weight of the mean row-entropy penalty added through ``add_loss``.
        Set to zero to disable adjacency regularization.
    """

    def __init__(
        self,
        units: int,
        n_nodes: int,
        activation: str | None = "relu",
        use_bias: bool = True,
        self_loop_bias: float = 2.0,
        identity_mix: float = 0.0,
        adjacency_reg_weight: float = 1e-4,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if units <= 0:
            raise ValueError(f"units must be positive, got {units}.")
        if n_nodes <= 0:
            raise ValueError(f"n_nodes must be positive, got {n_nodes}.")
        if self_loop_bias < 0.0:
            raise ValueError(
                f"self_loop_bias must be non-negative, got {self_loop_bias}."
            )
        if not 0.0 <= identity_mix <= 1.0:
            raise ValueError(
                f"identity_mix must be in [0, 1], got {identity_mix}."
            )
        if adjacency_reg_weight < 0.0:
            raise ValueError(
                "adjacency_reg_weight must be non-negative, "
                f"got {adjacency_reg_weight}."
            )

        self.units = int(units)
        self.n_nodes = int(n_nodes)
        self.use_bias = bool(use_bias)
        self.self_loop_bias = float(self_loop_bias)
        self.identity_mix = float(identity_mix)
        self.adjacency_reg_weight = float(adjacency_reg_weight)
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        if len(input_shape) < 3:
            raise ValueError(
                "GraphConv expects inputs shaped (..., n_nodes, features), "
                f"got {input_shape}."
            )

        input_nodes = input_shape[-2]
        if input_nodes is not None and int(input_nodes) != self.n_nodes:
            raise ValueError(
                f"GraphConv was configured for {self.n_nodes} nodes, "
                f"but input shape contains {input_nodes}."
            )

        f_in = input_shape[-1]
        if f_in is None:
            raise ValueError("The input feature dimension must be known at build time.")

        # Start with zero learnable residual logits. The explicit diagonal bias
        # in normalized_adjacency() supplies the electrode-preserving prior.
        self.A_theta = self.add_weight(
            name="A_theta",
            shape=(self.n_nodes, self.n_nodes),
            initializer="zeros",
            trainable=True,
        )
        self.W = self.add_weight(
            name="W",
            shape=(int(f_in), self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        if self.use_bias:
            self.b = self.add_weight(
                name="b",
                shape=(self.units,),
                initializer="zeros",
                trainable=True,
            )
        else:
            self.b = None

        super().build(input_shape)

    def normalized_adjacency(self, dtype=None) -> tf.Tensor:
        """Return the normalized electrode adjacency used by this layer."""
        if not self.built:
            raise RuntimeError("GraphConv must be built before reading adjacency.")

        dtype = tf.as_dtype(dtype or self.compute_dtype)
        identity = tf.eye(self.n_nodes, dtype=dtype)
        logits = tf.cast(self.A_theta, dtype)
        logits = logits + tf.cast(self.self_loop_bias, dtype) * identity
        learned_adjacency = tf.nn.softmax(logits, axis=-1)

        mix = tf.cast(self.identity_mix, dtype)
        return (1.0 - mix) * learned_adjacency + mix * identity

    def call(self, X, training=False):
        """Apply graph aggregation and a learned feature projection."""
        del training  # The layer has no train/inference-specific operation.

        X = tf.convert_to_tensor(X)

        # build() already validates a statically known node dimension. Retain
        # a runtime assertion only for genuinely dynamic shapes, avoiding an
        # unnecessary assertion op in the normal fixed-channel EEG path.
        static_nodes = X.shape[-2]
        if static_nodes is None:
            tf.debugging.assert_equal(
                tf.shape(X)[-2],
                self.n_nodes,
                message="GraphConv input node dimension does not match n_nodes.",
            )
        elif int(static_nodes) != self.n_nodes:
            raise ValueError(
                f"GraphConv expected {self.n_nodes} nodes, got {static_nodes}."
            )

        A_norm = self.normalized_adjacency(dtype=X.dtype)

        if self.adjacency_reg_weight > 0.0:
            epsilon = tf.cast(tf.keras.backend.epsilon(), A_norm.dtype)
            row_entropy = -tf.reduce_sum(
                A_norm * tf.math.log(A_norm + epsilon),
                axis=-1,
            )
            mean_entropy = tf.reduce_mean(row_entropy)
            self.add_loss(
                tf.cast(self.adjacency_reg_weight, mean_entropy.dtype)
                * mean_entropy
            )

        # tf.matmul broadcasts A_norm over every leading dimension of X,
        # e.g. batch and time for rank-4 EEG sequences.
        AX = tf.linalg.matmul(A_norm, X)
        H = tf.linalg.matmul(AX, tf.cast(self.W, X.dtype))

        if self.b is not None:
            H = H + tf.cast(self.b, H.dtype)

        return self.activation(H)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "n_nodes": self.n_nodes,
                "activation": tf.keras.activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "self_loop_bias": self.self_loop_bias,
                "identity_mix": self.identity_mix,
                "adjacency_reg_weight": self.adjacency_reg_weight,
            }
        )
        return config
