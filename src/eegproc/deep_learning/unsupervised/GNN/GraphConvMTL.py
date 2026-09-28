"""MTLFuseNet-style graph convolution for EEGProc.

This layer implements the renormalized first-order spectral GCN used in
MTLFuseNet Eq. (15):

    A_tilde = A + I
    A_hat   = D_tilde^{-1/2} A_tilde D_tilde^{-1/2}
    H       = activation(A_hat X W + b)

Unlike EEGProc's standard GraphConv, the adjacency is fixed and supplied from
outside the neural network (e.g. a mutual-information channel graph). It is
not learned by backpropagation and it is not row-softmax normalized.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="eegproc")
class GraphConvMTL(layers.Layer):
    """MTLFuseNet-style fixed-adjacency graph convolution.

    Parameters
    ----------
    units : int
        Output feature dimensionality per EEG channel/node.
    n_nodes : int
        Number of EEG channels/nodes.
    adjacency : array-like, shape (n_nodes, n_nodes)
        Fixed channel adjacency. For the MTLFuseNet comparison this should be
        the mutual-information adjacency computed from TRAINING DATA ONLY.
    activation : str or callable or None, default="relu"
        Activation after graph aggregation and feature projection.
    use_bias : bool, default=True
        Whether to add the learned feature bias b.
    add_self_loops : bool, default=True
        Implements A_tilde = A + I from MTLFuseNet Eq. (15).
    symmetrize : bool, default=True
        Mutual information is symmetric. This option averages A and A^T to
        remove small estimator asymmetries before normalization.
    epsilon : float, default=1e-8
        Numerical floor used when computing D^{-1/2}.
    """

    def __init__(
        self,
        units: int,
        n_nodes: int,
        adjacency,
        activation: str | None = "relu",
        use_bias: bool = True,
        add_self_loops: bool = True,
        symmetrize: bool = True,
        epsilon: float = 1e-8,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if units <= 0:
            raise ValueError(f"units must be positive, got {units}.")
        if n_nodes <= 0:
            raise ValueError(f"n_nodes must be positive, got {n_nodes}.")
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {epsilon}.")

        adjacency_array = np.asarray(adjacency, dtype=np.float32)
        expected_shape = (int(n_nodes), int(n_nodes))
        if adjacency_array.shape != expected_shape:
            raise ValueError(
                f"adjacency must have shape {expected_shape}, "
                f"got {adjacency_array.shape}."
            )
        if not np.all(np.isfinite(adjacency_array)):
            raise ValueError("adjacency contains NaN or infinite values.")
        if np.any(adjacency_array < 0.0):
            raise ValueError(
                "MTLFuseNet mutual-information adjacency must be non-negative."
            )

        if symmetrize:
            adjacency_array = 0.5 * (adjacency_array + adjacency_array.T)

        self.units = int(units)
        self.n_nodes = int(n_nodes)
        self.use_bias = bool(use_bias)
        self.add_self_loops = bool(add_self_loops)
        self.symmetrize = bool(symmetrize)
        self.epsilon = float(epsilon)
        self.activation = tf.keras.activations.get(activation)

        # Store a JSON/Keras-serializable copy. The actual tensor is created as
        # a non-trainable weight in build().
        self._adjacency_init = adjacency_array.tolist()

    def build(self, input_shape):
        if len(input_shape) < 3:
            raise ValueError(
                "GraphConvMTL expects (..., n_nodes, features), "
                f"got {input_shape}."
            )

        input_nodes = input_shape[-2]
        if input_nodes is not None and int(input_nodes) != self.n_nodes:
            raise ValueError(
                f"GraphConvMTL was configured for {self.n_nodes} nodes, "
                f"but input shape contains {input_nodes}."
            )

        f_in = input_shape[-1]
        if f_in is None:
            raise ValueError("The input feature dimension must be known.")

        self.A = self.add_weight(
            name="A_fixed",
            shape=(self.n_nodes, self.n_nodes),
            initializer=tf.keras.initializers.Constant(self._adjacency_init),
            trainable=False,
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

    def raw_adjacency(self, dtype=None) -> tf.Tensor:
        """Return the fixed MI adjacency A."""
        if not self.built:
            raise RuntimeError("GraphConvMTL must be built before reading A.")
        dtype = tf.as_dtype(dtype or self.compute_dtype)
        return tf.cast(self.A, dtype)

    def normalized_adjacency(self, dtype=None) -> tf.Tensor:
        """Return D_tilde^-1/2 A_tilde D_tilde^-1/2."""
        if not self.built:
            raise RuntimeError(
                "GraphConvMTL must be built before reading normalized adjacency."
            )

        dtype = tf.as_dtype(dtype or self.compute_dtype)
        A = tf.cast(self.A, dtype)

        if self.add_self_loops:
            A = A + tf.eye(self.n_nodes, dtype=dtype)

        degree = tf.reduce_sum(A, axis=-1)
        degree = tf.maximum(degree, tf.cast(self.epsilon, dtype))
        degree_inv_sqrt = tf.math.rsqrt(degree)

        # Equivalent to D^-1/2 @ A @ D^-1/2 without materializing two dense
        # diagonal matrices.
        return (
            degree_inv_sqrt[:, tf.newaxis]
            * A
            * degree_inv_sqrt[tf.newaxis, :]
        )

    def call(self, X, training=False):
        del training  # no train/inference-specific behavior in this layer

        X = tf.convert_to_tensor(X)
        static_nodes = X.shape[-2]
        if static_nodes is None:
            tf.debugging.assert_equal(
                tf.shape(X)[-2],
                self.n_nodes,
                message="GraphConvMTL node dimension does not match n_nodes.",
            )
        elif int(static_nodes) != self.n_nodes:
            raise ValueError(
                f"GraphConvMTL expected {self.n_nodes} nodes, got {static_nodes}."
            )

        A_hat = self.normalized_adjacency(dtype=X.dtype)
        AX = tf.linalg.matmul(A_hat, X)
        H = tf.linalg.matmul(AX, tf.cast(self.W, X.dtype))

        if self.b is not None:
            H = H + tf.cast(self.b, H.dtype)

        return self.activation(H)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "n_nodes": self.n_nodes,
                "adjacency": self._adjacency_init,
                "activation": tf.keras.activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "add_self_loops": self.add_self_loops,
                "symmetrize": self.symmetrize,
                "epsilon": self.epsilon,
            }
        )
        return config
