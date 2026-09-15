"""Source-calibrated EEG diagnostics and reusable spectral/topography arrays.

The SIC decoder produces band-filtered signals. An aperiodic exponent cannot
be recovered reliably from those bands; it is explicitly unavailable here.
Stored signals and PSDs allow later offline assessment with another estimator.
No amplitude threshold is described as microvolts without unit metadata.
"""

from dataclasses import dataclass
import itertools

import numpy as np
from scipy.signal import periodogram

FAMILIES = ("amplitude", "spectral_power", "aperiodic_exponent", "coherence", "debiased_wpli_squared")
DEFAULT_BANDS = ((4.0, 8.0), (8.0, 13.0), (13.0, 30.0))


def signal_diagnostics(signal, *, fs, n_channels, band_edges=DEFAULT_BANDS,
                       feature_order="channel-major"):
    """Window-replicate spectra; signed debiased wPLI squared (Vinck 2011).

    No concatenation across discontinuous windows. Connectivity is calculated
    across windows independently at each frequency, then averaged in band.
    Degenerate estimates stay NaN; they are never silently treated as zero.
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 3 or x.shape[-1] != n_channels * len(band_edges) or not np.isfinite(x).all():
        raise ValueError("Signal must be a finite (windows,time,channels*bands) trial")
    if not np.isfinite(fs) or fs <= 0 or x.shape[1] < 4:
        raise ValueError("Need positive fs and at least four samples per window")
    if any(not 0 < lo < hi <= fs / 2 for lo, hi in band_edges):
        raise ValueError("Band edges must lie in (0, Nyquist]")
    if feature_order == "channel-major":
        x = x.reshape(*x.shape[:2], n_channels, len(band_edges))
    elif feature_order == "band-major":
        x = x.reshape(*x.shape[:2], len(band_edges), n_channels).transpose(0, 1, 3, 2)
    else:
        raise ValueError("feature_order must be channel-major or band-major")
    frequencies, psd = periodogram(x, fs=fs, window="hann", detrend="constant", axis=1)
    # Shape: electrodes, bands, frequencies, suitable for topographies/heatmaps.
    mean_psd = psd.mean(axis=0).transpose(1, 2, 0)
    centered = x - x.mean(axis=1, keepdims=True)
    fft = np.fft.rfft(centered * np.hanning(x.shape[1])[None, :, None, None], axis=1)
    pairs = np.asarray(list(itertools.combinations(range(n_channels), 2)), dtype=int).reshape(-1, 2)
    coherence = np.full((len(pairs), len(band_edges)), np.nan)
    dwpli = np.full_like(coherence, np.nan)
    power = np.zeros((n_channels, len(band_edges)))
    for band, (lo, hi) in enumerate(band_edges):
        mask = (frequencies >= lo) & (frequencies < hi)
        if not mask.any():
            raise ValueError(f"No frequency bins in band {lo}-{hi}; use longer windows")
        power[:, band] = mean_psd[:, band, mask].sum(axis=-1) * (fs / x.shape[1])
        if len(x) < 2:
            continue
        for pair_index, (i, j) in enumerate(pairs):
            a, b = fft[:, mask, i, band], fft[:, mask, j, band]
            cross = a * b.conj()
            denominator = np.mean(abs(a) ** 2, axis=0) * np.mean(abs(b) ** 2, axis=0)
            c = np.divide(abs(cross.mean(axis=0)) ** 2, denominator,
                          out=np.full_like(denominator, np.nan), where=denominator > 1e-24)
            imaginary = cross.imag
            sum_sq = np.sum(imaginary ** 2, axis=0)
            numerator = imaginary.sum(axis=0) ** 2 - sum_sq
            denominator = np.abs(imaginary).sum(axis=0) ** 2 - sum_sq
            w = np.divide(numerator, denominator, out=np.full_like(denominator, np.nan), where=denominator > 1e-24)
            if np.isfinite(c).any():
                coherence[pair_index, band] = np.nanmean(c)
            if np.isfinite(w).any():
                dwpli[pair_index, band] = np.nanmean(w)
    return {
        "amplitude": np.quantile(abs(x), 0.99, axis=(0, 1)),
        "spectral_power": power,
        "aperiodic_exponent": np.full(n_channels, np.nan),
        "coherence": coherence, "debiased_wpli_squared": dwpli,
        "frequencies_hz": frequencies, "psd": mean_psd,
        "rms": np.sqrt(np.mean(x ** 2, axis=(0, 1))), "pair_indices": pairs,
    }


@dataclass
class PhysiologicalReference:
    lower: dict
    upper: dict
    quantile: float = 0.95
    required_fraction: float = 0.95

    @classmethod
    def fit(cls, diagnostics, *, quantile=0.95, required_fraction=0.95):
        if not 0 < quantile < 1 or not 0 < required_fraction <= 1:
            raise ValueError("Invalid physiological quantile or pass fraction")
        lower, upper = {}, {}
        for family in FAMILIES:
            values = np.stack([d[family] for d in diagnostics])
            # Use only components defined on EVERY source reference trial.
            defined = np.isfinite(values).all(axis=0)
            lower[family] = np.full(values.shape[1:], np.nan)
            upper[family] = np.full(values.shape[1:], np.nan)
            if defined.any():
                tail = (1 - quantile) / 2
                lower[family][defined] = np.quantile(values[:, defined], tail, axis=0)
                upper[family][defined] = np.quantile(values[:, defined], 1 - tail, axis=0)
        return cls(lower, upper, quantile, required_fraction)

    def assess(self, diagnostics):
        checks = {}
        for family in FAMILIES:
            values = diagnostics[family]
            defined = np.isfinite(self.lower[family]) & np.isfinite(self.upper[family])
            count = int(defined.sum())
            passed = (np.isfinite(values) & (values >= self.lower[family]) & (values <= self.upper[family]))
            fraction = float(passed[defined].mean()) if count else None
            checks[family] = {
                "passed": bool(fraction >= self.required_fraction) if fraction is not None else None,
                "fraction_in_range": fraction, "n_checked": count,
                "n_components": int(values.size), "n_undefined_candidate": int((~np.isfinite(values) & defined).sum()),
                "reason": ("not_estimable_from_band_filtered_decoder" if family == "aperiodic_exponent"
                           else "source_reference_undefined") if not count else None,
            }
        available = [v["passed"] for v in checks.values() if v["passed"] is not None]
        return {"checks": checks, "passed_count": sum(available), "available_count": len(available),
                "required_count": len(FAMILIES),
                "all_required_passed": all(available) if len(available) == len(FAMILIES) else None,
                "available_checks_passed": all(available) if available else None}

    def arrays(self):
        return {**{f"lower_{k}": v for k, v in self.lower.items()},
                **{f"upper_{k}": v for k, v in self.upper.items()}}


def source_vcsc_calibration(source_features):
    """Match existing differentiable VCSC estimator, using source trials only."""
    from .counterfactual_loss import _vcsc_band_coherence_wpli
    import tensorflow as tf
    coherence, wpli = [], []
    for x in source_features:
        c, w = _vcsc_band_coherence_wpli(tf.convert_to_tensor(x[None], dtype=tf.float32))
        coherence.append(c.numpy())
        wpli.append(w.numpy())
    c, w = np.stack(coherence), np.stack(wpli)
    return {"c_hat": c.mean(axis=0), "w_hat": w.mean(axis=0),
            "sigma_raw": np.maximum(c.std(axis=0), 1e-4),
            "sigma_spec": np.maximum(w.std(axis=0), 1e-4),
            "source_coherence": c, "source_dwpli_squared": w}


def make_source_loss(calibration, **weights):
    """Use the existing CFO objective with source-only VCSC constants."""
    import tensorflow as tf
    from .counterfactual_loss import CounterfactualLoss, _vcsc_band_coherence_wpli, _VCSC_DISTANCES_CM

    class SourceVCSCLoss(CounterfactualLoss):
        def physiological_validity(self, x_prime):
            c, w = _vcsc_band_coherence_wpli(x_prime)
            zc = (c - tf.constant(calibration["c_hat"], tf.float32)) / tf.constant(calibration["sigma_raw"], tf.float32)
            zw = (w - tf.constant(calibration["w_hat"], tf.float32)) / tf.constant(calibration["sigma_spec"], tf.float32)
            z = tf.minimum(tf.sqrt(tf.reduce_sum(zc ** 2 + zw ** 2, axis=-1) + 1e-12), self.vcsc_z_max)
            weight = tf.exp(tf.nn.relu(self.vcsc_distance_cm - tf.constant(_VCSC_DISTANCES_CM, tf.float32)) / self.vcsc_tau_cm)
            return tf.cast(tf.reduce_mean(weight * tf.math.expm1(tf.nn.relu(z - self.vcsc_z0))), x_prime.dtype)

    return SourceVCSCLoss(**weights)
