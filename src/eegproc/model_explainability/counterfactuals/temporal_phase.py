"""Temporal smoothing of the latent edit and phase-invariant signal distance.

Two additions to the counterfactual objective, both opt-in.

Temporal smoothing (fused lasso; Tibshirani et al. 2005, JRSS-B; total
variation, Rudin, Osher & Fatemi 1992, Physica D). The latent edit
delta = (z' - z) / sigma, standardized per latent feature, is flattened over
the trial's windows x timesteps in time order. Two penalties act on it:

  temporal_sparsity = mean_{t,f} |delta_{t,f}|
  temporal_tv       = mean_{t,f} |delta_{t+1,f} - delta_{t,f}|

with |.| smoothed as sqrt(x^2 + eps^2) - eps so the gradient is defined at
zero. Sparsity alone is a lasso; TV alone favours piecewise-constant edits;
both together (fused lasso) favour edits that are zero most of the trial and
constant within a few episodes. Flattening across windows assumes contiguous
windows (overlap 0, as in the DREAMER runs); with overlapping or gapped
windows the TV term also penalizes jumps across non-adjacent samples.

Phase-invariant distance (Hilbert amplitude envelopes: Brookes et al. 2011,
PNAS; Hipp et al. 2012, Nature Neuroscience; Fourier/Hilbert equivalence:
Bruns 2004, J Neurosci Methods). Sample-wise MSE between band-filtered
oscillations is dominated by timing: a small phase shift with unchanged power
gives a large error. envelope_distance instead compares log band-power
envelopes:

  A = |hilbert(hann * x)| per window, feature;   A_bar = mean of A over
  pools of up to 32 samples (250 ms at 128 Hz; see resolve_pool);   D_env(a, b) = mean (log(A_bar(a) + eps) - log(A_bar(b) + eps))^2

The Hann taper (as recommended for STFT-domain M/EEG work, Gramfort et al.
2013, NeuroImage) suppresses the per-window Hilbert edge effects; it is
applied identically to both signals, so tapered edges contribute little.
The TensorFlow and NumPy implementations are the same computation.
"""

import numpy as np
import tensorflow as tf

SMOOTH_ABS_EPS = 1e-3
ENVELOPE_EPS = 1e-3
ENVELOPE_POOL_SAMPLES = 32


# --------------------------------------------------------------- smoothing

def _smooth_abs(x, eps=SMOOTH_ABS_EPS):
    return tf.sqrt(tf.square(x) + eps * eps) - eps


def standardized_edit(z_prime, z, scale=None):
    """(N, F) edit in time order: (z' - z) / scale, z treated as fixed."""
    delta = tf.cast(z_prime, tf.float32) - tf.stop_gradient(tf.cast(z, tf.float32))
    if scale is not None:
        delta = delta / tf.constant(np.asarray(scale, dtype=np.float32))
    features = int(delta.shape[-1])
    return tf.reshape(delta, (-1, features))


def temporal_penalties(z_prime, z, scale=None):
    """Return (temporal_sparsity, temporal_tv) for one trial's latent edit."""
    delta = standardized_edit(z_prime, z, scale)
    sparsity = tf.reduce_mean(_smooth_abs(delta))
    if int(delta.shape[0]) < 2:
        return sparsity, tf.zeros_like(sparsity)
    tv = tf.reduce_mean(_smooth_abs(delta[1:] - delta[:-1]))
    return sparsity, tv


def edit_structure(z, z_prime, scale=None, *, active_fraction_of_max=0.1):
    """Descriptive statistics of the standardized latent edit over time.

    l1_mean / tv_mean: unsmoothed versions of the two penalties.
    active_fraction: fraction of time steps whose mean |delta| exceeds
      active_fraction_of_max x its maximum over the trial.
    n_episodes: number of contiguous runs of active time steps.
    The threshold is relative, so an all-zero edit has 0 episodes and any
    nonzero edit at least one; compare arms, not absolute counts.
    """
    z = np.asarray(z, dtype=np.float64)
    z_prime = np.asarray(z_prime, dtype=np.float64)
    delta = (z_prime - z).reshape(-1, z.shape[-1])
    if scale is not None:
        delta = delta / np.asarray(scale, dtype=np.float64)
    magnitude = np.abs(delta).mean(axis=1)
    peak = float(magnitude.max()) if magnitude.size else 0.0
    if peak == 0.0:
        active = np.zeros_like(magnitude, dtype=bool)
    else:
        active = magnitude > active_fraction_of_max * peak
    starts = int(active[0]) + int(np.sum(active[1:] & ~active[:-1])) if active.size else 0
    return {
        "l1_mean": float(np.abs(delta).mean()),
        "tv_mean": float(np.abs(np.diff(delta, axis=0)).mean()) if len(delta) > 1 else 0.0,
        "active_fraction": float(active.mean()) if active.size else 0.0,
        "n_episodes": starts,
        "active_threshold": f"{active_fraction_of_max} x max_t mean_f |delta|",
        "standardized": scale is not None,
    }


def latent_scale(features_iter):
    """Per-feature standard deviation of latent trials, pooled over windows,
    time steps and trials. features_iter yields arrays (..., F). Uses a
    two-pass-equivalent streaming (Chan) update in float64."""
    count, mean, m2 = 0, None, None
    for block in features_iter:
        values = np.asarray(block, dtype=np.float64).reshape(-1, np.asarray(block).shape[-1])
        n = values.shape[0]
        if n == 0:
            continue
        block_mean = values.mean(axis=0)
        block_m2 = ((values - block_mean) ** 2).sum(axis=0)
        if count == 0:
            count, mean, m2 = n, block_mean, block_m2
            continue
        delta = block_mean - mean
        total = count + n
        mean = mean + delta * n / total
        m2 = m2 + block_m2 + delta ** 2 * count * n / total
        count = total
    if count < 2:
        raise ValueError("Need at least two latent samples to estimate a scale")
    std = np.sqrt(m2 / (count - 1))
    # A dead feature would make its standardized edit explode; floor it at
    # 1e-3 of the median feature scale.
    floor = 1e-3 * float(np.median(std[std > 0])) if np.any(std > 0) else 1.0
    return np.maximum(std, floor)


# ------------------------------------------------------- envelope distance

def _analytic_multiplier(samples):
    h = np.zeros(samples, dtype=np.float64)
    if samples % 2 == 0:
        h[0] = h[samples // 2] = 1.0
        h[1:samples // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(samples + 1) // 2] = 2.0
    return h


def _hann(samples):
    """np.hanning (symmetric), the taper signal_diagnostics also uses."""
    return np.hanning(samples)


def resolve_pool(samples, pool=None):
    """pool if given (must divide samples), else the largest divisor of the
    window length that is <= ENVELOPE_POOL_SAMPLES (32 = 250 ms at 128 Hz)."""
    if pool is None:
        return next(p for p in range(min(ENVELOPE_POOL_SAMPLES, samples), 0, -1) if samples % p == 0)
    if pool < 1 or samples % pool:
        raise ValueError(f"envelope pool of {pool} samples must divide the window length {samples}")
    return pool


def envelope_tf(x, *, pool=None):
    """Pooled Hilbert envelope of (1,W,T,F) or (W,T,F): returns (W,T/pool,F)."""
    x = tf.cast(x, tf.float32)
    if x.shape.rank == 4:
        x = x[0]
    windows, samples, features = (int(d) for d in x.shape)
    pool = resolve_pool(samples, pool)
    tapered = tf.transpose(x, (0, 2, 1)) * tf.constant(_hann(samples).astype(np.float32))
    spectrum = tf.signal.fft(tf.complex(tapered, tf.zeros_like(tapered)))
    analytic = tf.signal.ifft(spectrum * tf.constant(_analytic_multiplier(samples).astype(np.complex64)))
    # sqrt(re^2 + im^2 + tiny): |.| has no gradient at exactly zero.
    amplitude = tf.sqrt(tf.square(tf.math.real(analytic)) + tf.square(tf.math.imag(analytic)) + 1e-12)
    pooled = tf.reduce_mean(tf.reshape(amplitude, (windows, features, samples // pool, pool)), axis=-1)
    return tf.transpose(pooled, (0, 2, 1))


def envelope_distance_tf(a, b, *, pool=None, eps=ENVELOPE_EPS):
    """Mean squared difference of log pooled envelopes; b is a fixed reference."""
    log_a = tf.math.log(envelope_tf(a, pool=pool) + eps)
    log_b = tf.stop_gradient(tf.math.log(envelope_tf(b, pool=pool) + eps))
    return tf.reduce_mean(tf.square(log_a - log_b))


def envelope_np(x, *, pool=None):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 4:
        x = x[0]
    windows, samples, features = x.shape
    pool = resolve_pool(samples, pool)
    tapered = np.transpose(x, (0, 2, 1)) * _hann(samples)
    analytic = np.fft.ifft(np.fft.fft(tapered, axis=-1) * _analytic_multiplier(samples), axis=-1)
    pooled = np.abs(analytic).reshape(windows, features, samples // pool, pool).mean(axis=-1)
    return np.transpose(pooled, (0, 2, 1))


def envelope_distance_np(a, b, *, pool=None, eps=ENVELOPE_EPS):
    return float(np.mean((np.log(envelope_np(a, pool=pool) + eps)
                          - np.log(envelope_np(b, pool=pool) + eps)) ** 2))
