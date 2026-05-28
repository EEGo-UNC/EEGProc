"""
manifold_net.py
===============
TensorFlow implementation of ManifoldNet (Chakraborty et al., 2020),
adapted for the STSNet framework.

ManifoldNet operates on grids of SPD (Symmetric Positive Definite) matrices.
Its core operation is the weighted Fréchet Mean (wFM) on the SPD manifold,
which replaces the standard Euclidean convolution.

Architecture (per the STSNet paper):
  - 2-layer cascaded wFM (convolutional) layers
  - G-transport / G-expansion nonlinearity (replaces ReLU)
  - Invariant layer  →  distance-to-mean feature vector (MO)

References
----------
Chakraborty et al., "ManifoldNet: A Deep Neural Network for
Manifold-Valued Data with Applications", IEEE TPAMI 2020.

Li et al., "STSNet ...", HISS 2023  (Section: ManifoldNet).
"""

import tensorflow as tf


# ---------------------------------------------------------------------------
# SPD manifold geometry helpers  (all differentiable via tf)
# ---------------------------------------------------------------------------

_EPS = 1e-6


def matrix_sqrt(A: tf.Tensor) -> tf.Tensor:
    """Compute the symmetric matrix square-root via eigen-decomposition.

    Parameters
    ----------
    A : Tensor, shape (..., n, n)  — batch of SPD matrices

    Returns
    -------
    Tensor, same shape  — A^{1/2}
    """
    vals, vecs = tf.linalg.eigh(A)
    vals = tf.maximum(vals, _EPS)
    sqrt_vals = tf.sqrt(vals)                               # (..., n)
    # reconstruct:  V diag(sqrt(λ)) Vᵀ
    return tf.matmul(vecs * sqrt_vals[..., tf.newaxis, :], vecs, transpose_b=True)


def matrix_sqrt_inv(A: tf.Tensor) -> tf.Tensor:
    """Compute A^{-1/2}."""
    vals, vecs = tf.linalg.eigh(A)
    vals = tf.maximum(vals, _EPS)
    inv_sqrt_vals = 1.0 / tf.sqrt(vals)
    return tf.matmul(vecs * inv_sqrt_vals[..., tf.newaxis, :], vecs, transpose_b=True)


def matrix_log_spd(A: tf.Tensor) -> tf.Tensor:
    """Riemannian (matrix) logarithm of an SPD matrix.

    log(A) = V diag(log(λ)) Vᵀ
    """
    vals, vecs = tf.linalg.eigh(A)
    vals = tf.maximum(vals, _EPS)
    log_vals = tf.math.log(vals)
    return tf.matmul(vecs * log_vals[..., tf.newaxis, :], vecs, transpose_b=True)


def matrix_exp_spd(S: tf.Tensor) -> tf.Tensor:
    """Matrix exponential of a symmetric matrix → SPD.

    exp(S) = V diag(exp(λ)) Vᵀ
    """
    vals, vecs = tf.linalg.eigh(S)
    exp_vals = tf.exp(vals)
    return tf.matmul(vecs * exp_vals[..., tf.newaxis, :], vecs, transpose_b=True)


def riemannian_distance_sq(A: tf.Tensor, B: tf.Tensor) -> tf.Tensor:
    """Squared affine-invariant Riemannian distance between SPD matrices.

    d²(A, B) = || log(A^{-1/2} B A^{-1/2}) ||_F²

    Parameters
    ----------
    A, B : Tensor, shape (..., n, n)

    Returns
    -------
    Tensor, shape (...)
    """
    A_inv_sqrt = matrix_sqrt_inv(A)
    M = A_inv_sqrt @ B @ A_inv_sqrt
    log_M = matrix_log_spd(M)
    return tf.reduce_sum(tf.square(log_M), axis=[-2, -1])


# ---------------------------------------------------------------------------
# Weighted Fréchet Mean (wFM) on the SPD manifold
# ---------------------------------------------------------------------------

def weighted_frechet_mean(
    matrices: tf.Tensor,
    weights: tf.Tensor,
    n_iters: int = 10,
) -> tf.Tensor:
    """Compute the weighted Fréchet Mean (wFM) on the SPD manifold.

    Uses the fixed-point / gradient-descent iteration on the manifold:
        μ_{t+1} = μ_t exp( Σ_i w_i log_{μ_t}(M_i) )

    Parameters
    ----------
    matrices : Tensor, shape (batch, N, n, n)  — N SPD matrices
    weights  : Tensor, shape (N,)              — convex weights (sum = 1)
    n_iters  : int — number of fixed-point iterations

    Returns
    -------
    Tensor, shape (batch, n, n)  — the wFM
    """
    # Initialise with the Euclidean mean (a reasonable warm start)
    mean = tf.reduce_sum(
        matrices * weights[tf.newaxis, :, tf.newaxis, tf.newaxis], axis=1
    )  # (batch, n, n)

    for _ in range(n_iters):
        mean_inv_sqrt = matrix_sqrt_inv(mean)  # (batch, n, n)

        # log_{mean}(M_i) = mean^{1/2} log(mean^{-1/2} M_i mean^{-1/2}) mean^{1/2}
        mean_sqrt = matrix_sqrt(mean)          # (batch, n, n)

        # Expand for N matrices: (batch, 1, n, n) ○ (batch, N, n, n)
        mis = mean_inv_sqrt[:, tf.newaxis, :, :]  # (batch, 1, n, n)
        ms  = mean_sqrt[:, tf.newaxis, :, :]       # (batch, 1, n, n)

        inner = mis @ matrices @ mis              # (batch, N, n, n)
        logs  = matrix_log_spd(inner)             # (batch, N, n, n)
        # weight and sum the log-mapped tangent vectors
        tangent = tf.reduce_sum(
            logs * weights[tf.newaxis, :, tf.newaxis, tf.newaxis], axis=1
        )  # (batch, n, n)

        # Retract back to the manifold
        mean = ms[:, 0, :, :] @ matrix_exp_spd(tangent) @ ms[:, 0, :, :]

    return mean  # (batch, n, n)


# ---------------------------------------------------------------------------
# wFM Convolutional Layer
# ---------------------------------------------------------------------------

class WFMLayer(tf.keras.layers.Layer):
    """Single wFM convolutional layer (sliding window over the time axis).

    The input is a sequence of SPD matrices (one per time window per band).
    The kernel slides along the time axis with stride 1.

    Parameters
    ----------
    kernel_size : int   — number of adjacent time steps in each wFM window
    n_fm_iters  : int   — fixed-point iterations for the Fréchet mean
    """

    def __init__(
        self,
        kernel_size: int = 2,
        n_fm_iters: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.n_fm_iters  = n_fm_iters

    def build(self, input_shape):
        # Learnable convex weights for the wFM (one per kernel position)
        # Initialise uniformly; softmax ensures they remain convex.
        self.raw_weights = self.add_weight(
            name="raw_weights",
            shape=(self.kernel_size,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (batch, n_time, n_bands, n, n)

        Returns
        -------
        Tensor, shape (batch, n_time - kernel_size + 1, n_bands, n, n)
        """
        weights = tf.nn.softmax(self.raw_weights)  # convex weights

        batch      = tf.shape(x)[0]
        n_time     = x.shape[1]
        n_bands    = x.shape[2]
        n          = x.shape[3]

        outputs = []
        for t in range(n_time - self.kernel_size + 1):
            # Gather kernel_size adjacent time steps: (batch, n_bands, k, n, n)
            window = tf.stack(
                [x[:, t + k, :, :, :] for k in range(self.kernel_size)],
                axis=2,
            )  # (batch, n_bands, kernel_size, n, n)

            # Flatten batch & band dims so weighted_frechet_mean operates on (batch', N, n, n)
            flat = tf.reshape(window, [-1, self.kernel_size, n, n])
            mean = weighted_frechet_mean(flat, weights, self.n_fm_iters)
            mean = tf.reshape(mean, [batch, n_bands, n, n])
            outputs.append(mean[:, tf.newaxis, :, :, :])

        return tf.concat(outputs, axis=1)  # (batch, T', n_bands, n, n)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"kernel_size": self.kernel_size, "n_fm_iters": self.n_fm_iters})
        return cfg


# ---------------------------------------------------------------------------
# G-transport nonlinearity
# ---------------------------------------------------------------------------

class GTransport(tf.keras.layers.Layer):
    """G-transport operator: Gtr(O; g) = g · O  (learnable geodesic transport).

    A learnable SPD matrix *g* is initialised to the identity and acts as
    a multiplicative transform on the manifold-valued features via
    symmetric conjugation:  g O gᵀ  (= g O g since g is symmetric).

    Parameters
    ----------
    n : int — matrix size (n_channels)
    """

    def __init__(self, matrix_size: int, **kwargs):
        super().__init__(**kwargs)
        self.matrix_size = matrix_size

    def build(self, input_shape):
        # Learnable lower-triangular Cholesky factor L; g = L Lᵀ  ensures SPD
        self.L = self.add_weight(
            name="chol_L",
            shape=(self.matrix_size, self.matrix_size),
            initializer="identity",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (batch, n_time, n_bands, n, n)

        Returns
        -------
        Tensor, same shape
        """
        L   = tf.linalg.band_part(self.L, -1, 0)  # lower triangular
        g   = L @ tf.transpose(L)                  # (n, n) SPD
        # broadcast conjugation: g x gᵀ  over batch / time / band dims
        return g @ x @ tf.transpose(g)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"matrix_size": self.matrix_size})
        return cfg


# ---------------------------------------------------------------------------
# Invariant layer  →  MO feature vector
# ---------------------------------------------------------------------------

class InvariantLayer(tf.keras.layers.Layer):
    """Compute the MO feature vector as Riemannian distances to the global wFM.

    Given d channel outputs {CO_i}, the unweighted FM M_w is computed,
    then moi = d(M_w, CO_i)  for each i, yielding the MO vector (Eq. 8).

    Parameters
    ----------
    n_fm_iters : int — Fréchet mean iterations
    """

    def __init__(self, n_fm_iters: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.n_fm_iters = n_fm_iters

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (batch, n_time, n_bands, n, n)
            Output of the final wFM layer.

        Returns
        -------
        Tensor, shape (batch, n_time * n_bands)  — the MO feature vector
        """
        batch   = tf.shape(x)[0]
        n_time  = x.shape[1]
        n_bands = x.shape[2]
        n       = x.shape[3]
        d       = n_time * n_bands

        # Flatten time & band into a single "channel" axis
        flat = tf.reshape(x, [batch, d, n, n])           # (batch, d, n, n)

        # Unweighted FM = uniform weights 1/d
        uniform_w = tf.ones([d], dtype=x.dtype) / tf.cast(d, x.dtype)
        mw = weighted_frechet_mean(flat, uniform_w, self.n_fm_iters)  # (batch, n, n)

        # Distances d(Mw, CO_i) for each i → MO
        mw_exp = mw[:, tf.newaxis, :, :]   # (batch, 1, n, n)
        mo = tf.sqrt(
            tf.maximum(riemannian_distance_sq(
                tf.broadcast_to(mw_exp, tf.shape(flat)), flat
            ), 0.0)
        )  # (batch, d)

        return mo

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"n_fm_iters": self.n_fm_iters})
        return cfg


# ---------------------------------------------------------------------------
# Full ManifoldNet sub-model
# ---------------------------------------------------------------------------

class ManifoldNet(tf.keras.Model):
    """ManifoldNet sub-model for spatio-spectral feature extraction (MO).

    Processes the 4-D spatio-temporal-spectral EEG representation
    (n_windows, n_bands, n_channels, n_channels).

    Architecture
    ------------
    Input → WFMLayer(k=2) → GTransport → WFMLayer(k=2) → GTransport
          → InvariantLayer → MO (1-D feature vector)

    Parameters
    ----------
    n_channels  : int — number of EEG channels (matrix dimension)
    kernel_size : int — wFM kernel size (default 2, per the paper)
    n_fm_iters  : int — Fréchet mean iterations (trade-off: accuracy vs speed)
    """

    def __init__(
        self,
        n_channels: int,
        kernel_size: int = 2,
        n_fm_iters: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_channels  = n_channels
        self.kernel_size = kernel_size
        self.n_fm_iters  = n_fm_iters

        self.wfm1   = WFMLayer(kernel_size, n_fm_iters, name="wfm_1")
        self.gtrans1= GTransport(n_channels, name="g_transport_1")
        self.wfm2   = WFMLayer(kernel_size, n_fm_iters, name="wfm_2")
        self.gtrans2= GTransport(n_channels, name="g_transport_2")
        self.inv    = InvariantLayer(n_fm_iters, name="invariant")

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (batch, n_windows, n_bands, n_channels, n_channels)

        Returns
        -------
        mo : Tensor, shape (batch, feature_dim)   — MO spatio-spectral features
        """
        h = self.wfm1(x)       # (batch, T-1, n_bands, C, C)
        h = self.gtrans1(h)
        h = self.wfm2(h)       # (batch, T-2, n_bands, C, C)
        h = self.gtrans2(h)
        mo = self.inv(h)       # (batch, (T-2)*n_bands)
        return mo

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "n_channels" : self.n_channels,
            "kernel_size": self.kernel_size,
            "n_fm_iters" : self.n_fm_iters,
        })
        return cfg
