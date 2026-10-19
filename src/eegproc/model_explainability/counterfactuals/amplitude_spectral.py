"""Differentiable amplitude and band-power checks for decoded counterfactuals.

VCSC scores coherence and debiased wPLI^2. Both are ratios, so they are
invariant to rescaling a channel and cannot see amplitude or band power. The
independent physiology checks (typicality.physiology.signal_diagnostics) do
score those two families, and decoded counterfactuals fail them. This module
puts the same two families into the objective.

The descriptors here reproduce signal_diagnostics exactly, in TensorFlow:

  * amplitude      99th percentile of |x| over every window and sample,
                   per channel x band, with NumPy's default linear
                   interpolation between order statistics;
  * spectral_power scipy.signal.periodogram(window="hann",
                   detrend="constant", scaling="density", one-sided) per
                   window, averaged over windows, summed over the band's
                   in-passband bins and multiplied by fs / T.

Both are computed in physical units: x * scale + offset, with the trial's own
normalization transform, because the source reference bounds are physical.

The penalty uses the graded deviation that is reported for these families:
z_k = |v_k - m_k| / h_k with m_k, h_k the midpoint and half-width of
component k's source interval, so z_k <= 1 is exactly "inside the interval".
penalty = mean_k relu(z_k - (1 - margin))^2. It is zero when every component
sits inside its interval (shrunk by margin) and grows smoothly outside it.
Feasibility is decided separately, with the reported rule: each family must
have at least required_fraction of its components inside the interval.

The amplitude percentile's gradient reaches only the two order statistics
that bracket it. That is sparse but exact; Adam handles it, and it keeps the
optimized quantity identical to the reported one.
"""

import numpy as np
import tensorflow as tf

FAMILIES = ("amplitude", "spectral_power")


def _hann_periodic(length):
    """scipy.signal.get_window('hann', length), i.e. the periodic window."""
    n = np.arange(length, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * n / length)).astype(np.float32)


def _band_masks(n_samples, fs, band_edges):
    frequencies = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    masks = np.stack([(frequencies >= low) & (frequencies < high) for low, high in band_edges])
    empty = [i for i, row in enumerate(masks) if not row.any()]
    if empty:
        raise ValueError(f"No frequency bins in band(s) {empty}; use longer windows")
    return masks.astype(np.float32)


def amplitude_and_power(x, *, fs, n_channels, band_edges, scale=None, offset=None,
                        quantile=0.99):
    """Return (amplitude, spectral_power), each (n_channels, n_bands).

    x is one decoded trial, (W,T,C*B) or (1,W,T,C*B), channel-major then
    band-minor, in model-input units. scale and offset are the trial's
    normalization arrays (broadcastable to (W,T,C*B)); omit both for
    signals already in physical units.
    """
    x = tf.cast(x, tf.float32)
    if x.shape.rank == 4:
        x = x[0]
    if x.shape.rank != 3:
        raise ValueError("x must be one trial shaped (W,T,F) or (1,W,T,F)")
    n_bands = len(band_edges)
    windows, samples, features = (int(d) for d in x.shape)
    if features != n_channels * n_bands:
        raise ValueError(f"Expected {n_channels * n_bands} features, got {features}")
    if samples < 4:
        raise ValueError("Need at least four samples per window")
    if scale is not None:
        x = x * tf.cast(scale, tf.float32)
    if offset is not None:
        x = x + tf.cast(offset, tf.float32)
    x = tf.reshape(x, (windows, samples, n_channels, n_bands))

    # Amplitude: NumPy quantile with linear interpolation, over W*T samples.
    values = tf.reshape(tf.abs(x), (windows * samples, n_channels, n_bands))
    ordered = tf.sort(tf.transpose(values, (1, 2, 0)), axis=-1)  # (C,B,N)
    position = quantile * (windows * samples - 1)
    lower = int(np.floor(position))
    upper = min(lower + 1, windows * samples - 1)
    fraction = float(position - lower)
    amplitude = ordered[..., lower] + (ordered[..., upper] - ordered[..., lower]) * fraction

    # Spectral power: Hann periodogram per window, density scaling, one-sided.
    window = tf.constant(_hann_periodic(samples))
    centered = x - tf.reduce_mean(x, axis=1, keepdims=True)
    tapered = tf.transpose(centered, (0, 2, 3, 1)) * window  # (W,C,B,T)
    spectrum = tf.signal.rfft(tapered)
    density = tf.math.real(spectrum * tf.math.conj(spectrum))
    density = density / float(fs * np.sum(_hann_periodic(samples).astype(np.float64) ** 2))
    n_bins = samples // 2 + 1
    doubling = np.full(n_bins, 2.0, dtype=np.float32)
    doubling[0] = 1.0
    if samples % 2 == 0:
        doubling[-1] = 1.0
    density = density * tf.constant(doubling)
    mean_density = tf.reduce_mean(density, axis=0)  # (C,B,F)
    masks = tf.constant(_band_masks(samples, fs, band_edges))  # (B,F)
    power = tf.reduce_sum(mean_density * masks[None, :, :], axis=-1) * float(fs / samples)
    return amplitude, power


class AmplitudeSpectralReference:
    """Source-interval bounds for the amplitude and spectral-power families."""

    def __init__(self, lower, upper, *, required_fraction=0.95, margin=0.0):
        if not 0 < required_fraction <= 1:
            raise ValueError("required_fraction must be in (0, 1]")
        if not 0 <= margin < 1:
            raise ValueError("margin must be in [0, 1)")
        self.required_fraction = float(required_fraction)
        self.margin = float(margin)
        self.lower, self.upper, self.defined, self.scorable = {}, {}, {}, {}
        for family in FAMILIES:
            lo = np.asarray(lower[family], dtype=np.float64)
            hi = np.asarray(upper[family], dtype=np.float64)
            if lo.shape != hi.shape:
                raise ValueError(f"{family} bounds have mismatched shapes")
            defined = np.isfinite(lo) & np.isfinite(hi)
            # Zero-width intervals have no half-width to grade against; they
            # still count in the in-range fraction, exactly as assess() does.
            scorable = defined & (hi > lo)
            self.lower[family], self.upper[family] = lo, hi
            self.defined[family], self.scorable[family] = defined, scorable
        if not any(self.scorable[f].any() for f in FAMILIES):
            raise ValueError("No scorable amplitude or spectral-power components")

    @classmethod
    def from_physiological_reference(cls, reference, *, margin=0.0):
        return cls(reference.lower, reference.upper,
                   required_fraction=reference.required_fraction, margin=margin)

    def penalty(self, values):
        """values: {family: (C,B) tensor}. Returns (penalty, diagnostics)."""
        exceedances, diagnostics = [], {}
        for family in FAMILIES:
            value = tf.cast(values[family], tf.float32)
            lo, hi = self.lower[family], self.upper[family]
            scorable = self.scorable[family]
            mid = np.where(scorable, (lo + hi) / 2, 0.0).astype(np.float32)
            half = np.where(scorable, (hi - lo) / 2, 1.0).astype(np.float32)
            z = tf.abs(value - mid) / half
            index = np.flatnonzero(scorable.ravel())
            z_scored = tf.gather(tf.reshape(z, [-1]), index)
            exceedances.append(tf.nn.relu(z_scored - (1.0 - self.margin)))
            defined = self.defined[family]
            inside = tf.logical_and(value >= np.where(defined, lo, np.inf).astype(np.float32),
                                    value <= np.where(defined, hi, -np.inf).astype(np.float32))
            n_defined = int(defined.sum())
            fraction = (tf.reduce_sum(tf.cast(inside, tf.float32)) / n_defined
                        if n_defined else tf.constant(float("nan")))
            diagnostics[f"{family}_in_range"] = tf.stop_gradient(fraction)
            diagnostics[f"{family}_max_z"] = tf.stop_gradient(
                tf.reduce_max(z_scored) if len(index) else tf.constant(0.0))
        stacked = tf.concat(exceedances, axis=0)
        return tf.reduce_mean(tf.square(stacked)), diagnostics

    def feasible(self, diagnostics):
        """The reported rule: every family meets required_fraction."""
        for family in FAMILIES:
            if not self.defined[family].any():
                continue
            fraction = float(diagnostics[f"{family}_in_range"])
            if not np.isfinite(fraction) or fraction < self.required_fraction:
                return False
        return True

    def metadata(self):
        return {"families": list(FAMILIES), "required_fraction": self.required_fraction,
                "margin": self.margin,
                "n_scorable": {f: int(self.scorable[f].sum()) for f in FAMILIES},
                "n_defined": {f: int(self.defined[f].sum()) for f in FAMILIES},
                "penalty": "mean relu(|v-m|/h - (1-margin))^2 over scorable components",
                "feasibility": "per family, fraction of defined components inside the source interval >= required_fraction"}
