"""The mathematical objective for one complete SIC counterfactual trial.

Only z_prime is optimized. The original trial x, original encoder features z,
and all model parameters are constants. Distances are means, not sums, so
their scale does not automatically grow with trial length or feature width.
There is no VAE, KL divergence, classifier training. physiological_validity is
a VCSC (volume-conduction spatial coherence) check, not a complete
physiological model. Electrode positions come from the real standard
10-10/10-05 system (see the module-level constants below). All four
calibration curves are MEASURED from 414 real DREAMER trials and shipped in
vcsc_calibration_dreamer.npz; earlier literature-derived versions (Nunez et
al. 1997 for c_hat, Bendat & Piersol for sigma_raw, w_hat = 0 from Vinck et
al. 2011's debiased estimator) were checked against real EEG and rejected it
outright, so they were replaced. See _vcsc_calibration_curves for why, and
for the caveats that come with an empirical calibration.

Measured discrimination on the calibration set: real trials score below white
noise 88% of the time (median 0.66 vs 12.1), so this separates plausible from
implausible input but is not a clean decision rule -- about 12% of real trials
score worse than noise. Treat it as one graded axis among several, per the
report, not as a standalone accept/reject test.
"""

import functools
import itertools
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

"""
Standard DREAMER/AMIGOS/eegemotions_27 14-channel Emotiv EPOC montage, in the
channel-major order used by eegproc.deep_learning.prepare_datasets.DREAMER_EEG_COLS
(confirmed directly from that file's DREAMER_EEG_COLS list and
DREAMER_FREQUENCY_BANDS dict, whose insertion order is preserved by
bandpass_filter)
"""

_VCSC_CHANNELS = (
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
)
_VCSC_BANDS = ("theta", "alpha", "beta")
#Must match the Butterworth passbands used in preprocessing. If those change
#and these do not, the mask below silently selects the wrong bins.
_VCSC_BAND_EDGES_HZ = ((4.0, 8.0), (8.0, 13.0), (13.0, 30.0))
_VCSC_DEFAULT_FS = 128.0

_VCSC_POSITIONS_CM = {
    "AF3": (-2.8952, 7.3870, 4.7564), "AF4": (2.8952, 7.3870, 4.7564),
    "F7": (-7.1179, 5.1708, 2.8582), "F8": (7.1179, 5.1708, 2.8582),
    "F3": (-4.2467, 5.3650, 6.2253), "F4": (4.2467, 5.3650, 6.2253),
    "FC5": (-7.0837, 2.7750, 5.2614), "FC6": (7.0837, 2.7750, 5.2614),
    "T7": (-8.7977, 0.0000, 2.8582), "T8": (8.7977, 0.0000, 2.8582),
    "P7": (-7.1179, -5.1708, 2.8582), "P8": (7.1179, -5.1708, 2.8582),
    "O1": (-2.7186, -8.3666, 2.8582), "O2": (2.7186, -8.3666, 2.8582),
}
_VCSC_PAIRS = list(itertools.combinations(range(len(_VCSC_CHANNELS)), 2))

@functools.lru_cache(maxsize=8)
def _vcsc_band_bin_mask(n_bins, fs):
    """Boolean (n_bands, n_bins) mask selecting in-passband rfft bins per band.

    x_prime arrives already bandpass filtered per band, so the rfft of a theta
    channel carries real power in roughly 5 of 65 bins and filter stopband in
    the other 60. Averaging coherence over every bin therefore mixes those 5
    with 60 bins whose numerator and denominator are both near zero, where the
    ratio is numerical noise rather than coherence.

    This does not bias the z-scores, because the calibration reference is
    diluted identically -- it compresses the dynamic range, which costs
    discrimination. Restricting the average to in-band bins removes it.

    For an rfft of T real samples, n_bins = T//2 + 1 and bin k sits at
    k*fs/T Hz, so the spacing is fs / (2*(n_bins-1)).
    """
    if n_bins < 2:
        raise ValueError(f"Need at least two rfft bins to build a band mask, got {n_bins}")
    frequencies = np.arange(n_bins, dtype=np.float64) * (fs / (2.0 * (n_bins - 1)))
    mask = np.stack([
        (frequencies >= low) & (frequencies < high)
        for low, high in _VCSC_BAND_EDGES_HZ
    ])
    #A window too short to resolve a band leaves its row empty, which would
    #divide by zero. Short synthetic windows in the tests hit this: T=4 gives
    #3 bins at 32 Hz spacing, so no bin lands in 4-30 Hz. Fall back to every
    #bin for that band, reproducing the pre-mask behaviour exactly rather
    #than failing, and warn since the result is not band-resolved.
    empty = [name for name, row in zip(_VCSC_BANDS, mask) if not row.any()]
    if empty:
        warnings.warn(
            f"No rfft bin falls inside band(s) {empty} at fs={fs} with {n_bins} "
            f"bins ({fs / (2.0 * (n_bins - 1)):.3g} Hz spacing); averaging those "
            "bands over all bins instead. Expected only for windows too short "
            "to resolve the band -- at fs=128 with 1 s windows every band "
            "resolves. If this fires on real data, fs does not match the "
            "preprocessing rate.",
            RuntimeWarning,
            stacklevel=2,
        )
        mask[[not row.any() for row in mask]] = True
    return tf.constant(mask, dtype=tf.float32)


def _vcsc_band_mean(values, mask):
    """Mean of (n_bands, n_bins) values over that band's in-passband bins."""
    return tf.reduce_sum(values * mask, axis=-1) / tf.reduce_sum(mask, axis=-1)


def _vcsc_pair_penalty(z_pair, z0, z_max):
    """phi(Z): flat below z0, exponential to z_max, then linear beyond it.

    A hard clip at z_max bounds exp() against float32 overflow (z_pair=198 on
    a phase-lag violation produced inf, which poisons the total loss and every
    gradient through it) but its derivative above the cap is exactly zero. A
    pair at 20 sigma and a pair at 200 sigma then score identically and
    neither receives gradient, so the constraint stops resisting precisely in
    the implausible regime it exists to penalize.

    Past z_max we therefore continue along the tangent instead of flattening.
    Writing s = exp((z_max - z0)^+), the penalty is s - 1 + s*(Z - z_max) for
    Z > z_max, which matches the exponential branch in both value and first
    derivative at z_max, so phi stays C^1 and the gradient above the cap is
    the constant s rather than 0. exp() is still only ever evaluated on a
    bounded argument, so the overflow guard is preserved.
    """
    capped = tf.minimum(z_pair, z_max)
    excess = tf.nn.relu(z_pair - z_max)
    saturated = tf.exp(tf.nn.relu(capped - z0))
    return saturated - 1.0 + saturated * excess


def _vcsc_pairwise_distances():
    """3D chord distances between all 14 electrodes, in centimeters."""
    positions = [_VCSC_POSITIONS_CM[name] for name in _VCSC_CHANNELS]
    distances = []
    for i, j in _VCSC_PAIRS:
        dx = positions[i][0] - positions[j][0]
        dy = positions[i][1] - positions[j][1]
        dz = positions[i][2] - positions[j][2]
        distances.append(math.sqrt(dx * dx + dy * dy + dz * dz))
    return distances

_VCSC_DISTANCES_CM = _vcsc_pairwise_distances()

_VCSC_CALIBRATION_PATH = Path(__file__).with_name("vcsc_calibration_dreamer.npz")


@functools.lru_cache(maxsize=1)
def _load_vcsc_calibration():
    """Load the measured DREAMER calibration; cached so it reads once."""
    if not _VCSC_CALIBRATION_PATH.is_file():
        raise FileNotFoundError(
            f"VCSC calibration file missing: {_VCSC_CALIBRATION_PATH}. "
            "Regenerate it by computing coherence and debiased wPLI^2 over real "
            "trials (see the module docstring)."
        )
    with np.load(_VCSC_CALIBRATION_PATH, allow_pickle=False) as data:
        curves = {
            key: tf.constant(data[key], dtype=tf.float32)
            for key in ("c_hat", "w_hat", "sigma_raw", "sigma_spec")
        }
        meta = {"n_trials": int(data["n_trials"]), "n_windows": int(data["n_windows"])}
    n_pairs, n_bands = len(_VCSC_PAIRS), len(_VCSC_BANDS)
    for key, value in curves.items():
        if tuple(value.shape) != (n_pairs, n_bands):
            raise ValueError(
                f"Calibration {key} has shape {tuple(value.shape)}, expected "
                f"{(n_pairs, n_bands)}."
            )
    return curves, meta


def _vcsc_calibration_curves(n_windows):
    """Return (c_hat, w_hat, sigma_raw, sigma_spec), each (n_pairs, n_bands).

    These are MEASURED from real data, not derived from formulas. They are the
    per-pair, per-band mean and standard deviation of magnitude-squared
    coherence and debiased wPLI^2 across all 414 real DREAMER trials, computed
    through the same preprocessing the model sees (1 s windows, no overlap,
    128 Hz, global_rms normalization, theta/alpha/beta band-pass).

    Why measured rather than derived. Earlier versions used Nunez et al.'s
    volume-conduction formula for c_hat, the Bendat & Piersol sampling-variance
    result for sigma_raw, and w_hat = 0 on the grounds that Vinck et al.'s
    debiased wPLI^2 is unbiased under the null. Checked against real EEG, that
    combination rejects everything: real trials score a median VCSC of 8.6e7,
    worse than white noise. Two reasons, both instructive:

      * w_hat = 0 is right for UNCORRELATED sources, but real brains have
        genuine phase-lagged coupling. Real debiased wPLI^2 is 0.061, not 0,
        so a normal trial sat ~11 sigma out on the wPLI axis alone.
      * The Bendat & Piersol sigma is the SAMPLING std -- uncertainty in
        estimating one trial's coherence. What the z-score needs is the
        ACROSS-TRIAL std, which also contains real biological variability
        between trials and is roughly 6x larger (0.216 vs 0.035).

    The tradeoff this buys. Calibrating to real data drops separation between
    plausible and implausible input from the ~1e5 the old formulas suggested to
    roughly 16x. That is the honest number: the large separation existed only
    because the calibration was tight enough to reject real EEG too.

    Caveats. Measured at n_windows=60 on DREAMER valence trials; n_windows is
    accepted for signature compatibility and to warn on mismatch, but the
    curves are not rescaled by it, since across-trial biological variance --
    which does not shrink with window count -- dominates the sampling term.
    Calibration is pooled across valence, which the data supports (low 0.262
    vs high 0.278 mean coherence). Applying these to another dataset, montage,
    or preprocessing chain requires regenerating the file.
    """
    curves, meta = _load_vcsc_calibration()
    if int(n_windows) != meta["n_windows"]:
        warnings.warn(
            f"VCSC calibration was measured at n_windows={meta['n_windows']} but "
            f"this trial has {int(n_windows)}. The curves are not rescaled; "
            "z-scores will be biased. Regenerate the calibration to match.",
            RuntimeWarning,
            stacklevel=2,
        )
    return (
        curves["c_hat"],
        curves["w_hat"],
        curves["sigma_raw"],
        curves["sigma_spec"],
    )

def _vcsc_band_coherence_wpli(x_prime, fs=_VCSC_DEFAULT_FS):
    """
    x_prime is (1, W, T, 42): 14 channels x 3 bands (theta, alpha, beta),
    channel-major then band-minor. Each of the W windows is treated as one
    repeated observation/epoch, following the standard convention of
    averaging cross-spectra over trials before forming coherence and wPLI

    The frequency average runs over each band's passband bins only, not every
    rfft bin -- see _vcsc_band_bin_mask. fs must match the preprocessing rate
    or the mask selects the wrong bins.
    """
    n_channels, n_bands = len(_VCSC_CHANNELS), len(_VCSC_BANDS)
    shape = x_prime.shape
    if shape[-1] != n_channels * n_bands:
        raise ValueError(
            f"x_prime's last dimension must be {n_channels * n_bands}, "
            f"not {shape[-1]}."
        )
    reshaped = tf.reshape(x_prime, (shape[1], shape[2], n_channels, n_bands))
    signal = tf.transpose(reshaped, [2, 3, 0, 1])
    spectrum = tf.signal.rfft(signal) #(C,B,W,F_bins)
    mask = _vcsc_band_bin_mask(int(spectrum.shape[-1]), float(fs))  # (B, F_bins)

    coherences, wplis = [], []
    for i, j in _VCSC_PAIRS:
        xi, xj = spectrum[i], spectrum[j]  #(B,W,F_bins)
        cross = xi * tf.math.conj(xj)
        power_i = tf.math.real(xi * tf.math.conj(xi))
        power_j = tf.math.real(xj * tf.math.conj(xj))
        mean_cross = tf.reduce_mean(cross, axis=1)
        mean_power_i = tf.reduce_mean(power_i, axis=1)
        mean_power_j = tf.reduce_mean(power_j, axis=1)
        #real(z*conj(z)), not abs(z)**2: identical value, but the gradient of
        #complex abs is z/|z|, which is NaN at exactly z=0.
        magnitude_squared = tf.math.real(mean_cross * tf.math.conj(mean_cross))
        coherence = magnitude_squared / (mean_power_i * mean_power_j + 1e-12)
        coherences.append(_vcsc_band_mean(coherence, mask))

        #Debiased wPLI^2 (Vinck et al. 2011). The naive |E{imag}|/E{|imag|}
        #form is biased upward at finite sample size -- it read ~0.14 at
        #W=60 on uncorrelated data, where the true value is 0. This estimator
        #is unbiased under the null, so w_hat = 0 holds at any window count.
        imag_cross = tf.math.imag(cross)  # (B, W, F_bins)
        sum_imag = tf.reduce_sum(imag_cross, axis=1)
        sum_sq = tf.reduce_sum(tf.square(imag_cross), axis=1)
        sum_abs = tf.reduce_sum(tf.abs(imag_cross), axis=1)
        wpli = (tf.square(sum_imag) - sum_sq) / (
            tf.square(sum_abs) - sum_sq + 1e-12
        )
        wplis.append(_vcsc_band_mean(wpli, mask))

    return tf.stack(coherences, axis=0), tf.stack(wplis, axis=0)  # each (n_pairs, n_bands)


@dataclass(frozen=True)
class CounterfactualLoss:
    """Weighted target, latent, decoded-signal, and physiological penalties.

    The objective is target_weight * target_loss + latent_weight * MSE(z', z)
    + decoded_weight * mean_r MSE(R_r(z'), R_r(z)) + physiological_weight *
    mean_r VCSC(R_r(z')). The optimizer supplies the selected reconstruction
    paths: independent branch decoders or the saved joint reconstruction.
    The physiological term only participates in optimization when
    physiological_weight is nonzero, and that weight defaults to 0.

    target_probability is a desired softmax confidence, not a replacement
    for the predicted-class decision rule. The optimizer requires both the
    requested class and this confidence for success. Weights trade off the
    numerical penalties; they do not standardize EEG or encoder dimensions.
    Signal distances are in the model's input units, not necessarily microvolts.
    """

    target_weight: float = 1.0
    latent_weight: float = 0.1
    decoded_weight: float = 0.1
    physiological_weight: float = 0.0
    target_probability: float = 0.8
    vcsc_distance_cm: float = 12.0
    vcsc_tau_cm: float = 4.0
    vcsc_z0: float = 2.0
    vcsc_z_max: float = 20.0

    def __post_init__(self):
        for name in (
            "target_weight",
            "latent_weight",
            "decoded_weight",
            "physiological_weight",
            "vcsc_distance_cm",
            "vcsc_tau_cm",
            "vcsc_z0",
            "vcsc_z_max",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if self.target_weight == 0:
            raise ValueError("target_weight must be positive.")
        if self.vcsc_tau_cm <= 0:
            raise ValueError("vcsc_tau_cm must be positive.")
        if self.vcsc_z_max <= self.vcsc_z0:
            raise ValueError("vcsc_z_max must exceed vcsc_z0.")
        if self.vcsc_z_max - self.vcsc_z0 > 80:
            raise ValueError(
                "vcsc_z_max - vcsc_z0 must stay under 80; exp() overflows float32 above ~88."
            )
        if not 0 < self.target_probability < 1:
            raise ValueError("target_probability must be between 0 and 1.")

    def target_loss(self, logits, target_class):
        """Return max(0, log(p_min) - log p(target_class | z_prime)).

        logits has shape (1, n_classes), before softmax; target_class is one
        integer class index. log_softmax avoids taking log of rounded or
        underflowed probabilities. The penalty is zero at/above p_min, so
        confidence beyond the requested threshold earns no further reward.
        Gradients flow from logits through the frozen BiGRU/VC head to z'.
        """

        log_probability = tf.nn.log_softmax(logits, axis=-1)[0, target_class]
        return tf.nn.relu(
            tf.math.log(tf.cast(self.target_probability, logits.dtype))
            - log_probability
        )

    @staticmethod
    def latent_distance(z_prime, z):
        """Return elementwise MSE between equally shaped tensors.

        For latent proximity, both tensors are (1, windows, timesteps,
        combined_features). All entries participate; there is no temporal
        pooling of the representation and no padded-window inference.
        The same exact MSE is also used for decoded EEG tensors. The second
        argument is a fixed reference: stop_gradient prevents accidental
        differentiation through its encoder or its original input.
        """
        tf.debugging.assert_equal(
            tf.shape(z_prime), tf.shape(z), message="MSE shapes must match."
        )
        return tf.reduce_mean(
            tf.square(z_prime - tf.stop_gradient(tf.cast(z, z_prime.dtype)))
        )

    def decoded_distance(self, z_prime, z, decoder, *, reference_reconstructions=None):
        """Decode z' and return (mean MSE, reconstructions, branch MSEs).

        decoder is a differentiable callable accepting the complete latent
        trial and returning {path_name: reconstructed_trial}. Each output
        has shape (1, windows, timesteps, EEG_features). The optimizer
        supplies this adapter for either independent or joint reconstruction.

        Each R_r(z') is compared to the fixed original reconstruction R_r(z).
        The optimizer may supply that cached reference, computed once per
        trial. Otherwise it is decoded here from the fixed original latent.
        Reference gradients are stopped; candidate decoder gradients remain
        active. Branch losses are averaged to preserve the global loss scale.
        """
        if reference_reconstructions is None:
            reference_reconstructions = decoder(tf.stop_gradient(z))
        reconstructions = decoder(z_prime)
        if not reconstructions or set(reconstructions) != set(reference_reconstructions):
            raise ValueError("Candidate and original reconstruction paths must match and be nonempty.")
        distances = {
            name: self.latent_distance(value, reference_reconstructions[name])
            for name, value in reconstructions.items()
        }
        return (
            tf.add_n(list(distances.values())) / len(distances),
            reconstructions,
            distances,
        )
    def physiological_validity(self, x_prime):
        """Return the VCSC penalty for one decoded trial, in x_prime's dtype.

        Computes coherence and wPLI for all 91 electrode pairs per band,
        z-scores both against the calibration curves, and combines them into
        one deviation per pair. Pairs closer than vcsc_distance_cm are weighted
        exponentially harder, because volume conduction guarantees a coherence
        floor at short range that a real head cannot fall below -- so a
        low-coherence reading between nearby electrodes is closer to physically
        impossible than merely unusual. Deviations are penalized only past
        vcsc_z0, then exponentially, and are capped at vcsc_z_max so a single
        extreme pair cannot overflow the exponential or dominate the gradient.
        Returns the mean over pairs, so the scale does not grow with pair count.

        A low score is evidence of nothing worse than volume-conduction
        consistency; it is not a certificate of physiological validity, and
        failing can also be caused by common reference, poor SNR, or common
        input rather than by volume conduction specifically.
        """
        coherence, wpli = _vcsc_band_coherence_wpli(x_prime)
        c_hat, w_hat, sigma_raw, sigma_spec = _vcsc_calibration_curves(
            x_prime.shape[1]
        )
        #Curves are (n_pairs, n_bands), matching coherence/wpli directly, so
        #the calibration is band-resolved rather than shared across bands.
        z_raw = (coherence - c_hat) / sigma_raw
        z_spec = (wpli - w_hat) / sigma_spec
        z_pair = tf.sqrt(
            tf.reduce_sum(tf.square(z_raw) + tf.square(z_spec), axis=-1) + 1e-12
        )
        distance = tf.constant(_VCSC_DISTANCES_CM, dtype=tf.float32)
        weight = tf.exp(
            tf.nn.relu(self.vcsc_distance_cm - distance) / self.vcsc_tau_cm
        )
        penalty = _vcsc_pair_penalty(z_pair, self.vcsc_z0, self.vcsc_z_max)
        #Mean, not sum, over the 91 pairs: keeps the penalty's scale comparable
        #to the target/latent/decoded terms instead of growing with pair count.
        return tf.cast(tf.reduce_mean(weight * penalty), x_prime.dtype)

    def central_loss(
        self,
        *,
        logits,
        target_class,
        z_prime,
        z,
        x,
        decoder,
        reference_reconstructions=None,
        target_loss_override=None,
        target_components=None,
    ):
        """Return (loss_terms, reconstructions) for one optimization step.

        loss_terms contains total, the four unweighted terms, their four
        weighted contributions, and decoded_<path> distances. All are scalar
        tensors, allowing a single GradientTape to differentiate total. A
        model-aware optimizer may replace the default confidence target with
        a pre-gradient focal/VC component and supply separately logged target
        components. The caller handles optimization, finite checks, logging,
        and saving.
        """
        decoded, reconstructions, branch_distances = self.decoded_distance(
            z_prime, z, decoder, reference_reconstructions=reference_reconstructions
        )
        for name, reconstruction in reconstructions.items():
            tf.debugging.assert_equal(
                tf.shape(reconstruction), tf.shape(x),
                message=f"{name} reconstruction must match the original input shape.",
            )
        target = (
            self.target_loss(logits, target_class)
            if target_loss_override is None
            else tf.cast(target_loss_override, logits.dtype)
        )
        terms = {
            "target": target,
            "latent": self.latent_distance(z_prime, z),
            "decoded": decoded,
            "physiological": tf.add_n(
                [self.physiological_validity(v) for v in reconstructions.values()]
            )
            / len(reconstructions),
        }
        weighted = {
            f"weighted_{name}": getattr(self, f"{name}_weight") * value
            for name, value in terms.items()
        }
        result = {
            "total": tf.add_n(list(weighted.values())),
            **terms,
            **weighted,
            **{f"decoded_{name}": value for name, value in branch_distances.items()},
        }
        if target_components:
            overlap = set(result).intersection(target_components)
            if overlap:
                raise ValueError(
                    f"target_components collide with loss terms: {sorted(overlap)}"
                )
            result.update(target_components)
        return result, reconstructions
