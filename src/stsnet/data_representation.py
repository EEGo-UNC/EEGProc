"""
data_representation.py
======================
EEG data representation utilities for STSNet.

Two representations are constructed from raw EEG signals:
  1. 4-D spatio-temporal-spectral representation  →  ManifoldNet input
     Shape: (batch, n_windows, n_bands, n_channels, n_channels)

  2. Spatio-temporal (flattened covariance) representation  →  BiLSTM input
     Shape: (batch, n_windows, n_channels*(n_channels+1)//2)

References
----------
Li et al., "STSNet: a novel spatio-temporal-spectral network for
subject-independent EEG-based emotion recognition", HISS 2023.
"""

import numpy as np
from scipy.signal import butter, filtfilt


# ---------------------------------------------------------------------------
# Band-pass filter helpers
# ---------------------------------------------------------------------------

BAND_RANGES = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def bandpass_filter(
    signal: np.ndarray,
    low: float,
    high: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth band-pass filter.

    Parameters
    ----------
    signal : ndarray, shape (n_channels, n_samples)
    low, high : float  — passband edges in Hz
    fs : float         — sampling frequency
    order : int        — filter order

    Returns
    -------
    ndarray, same shape as *signal*
    """
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal, axis=-1)


def decompose_bands(
    signal: np.ndarray,
    fs: float,
    bands: list[str],
) -> dict[str, np.ndarray]:
    """Decompose a multi-channel EEG signal into frequency bands.

    Parameters
    ----------
    signal : ndarray, shape (n_channels, n_samples)
    fs     : float — sampling frequency
    bands  : list of band names from BAND_RANGES

    Returns
    -------
    dict  band_name → ndarray (n_channels, n_samples)
    """
    return {
        band: bandpass_filter(signal, *BAND_RANGES[band], fs)
        for band in bands
    }


# ---------------------------------------------------------------------------
# Covariance / SPD matrix helpers
# ---------------------------------------------------------------------------

_EPS = 1e-8


def covariance_to_spd(cov: np.ndarray) -> np.ndarray:
    """Convert a covariance matrix to a Symmetric Positive Definite (SPD)
    matrix via eigendecomposition, clamping near-zero eigenvalues.

    Parameters
    ----------
    cov : ndarray, shape (n_channels, n_channels)

    Returns
    -------
    ndarray, shape (n_channels, n_channels)  — guaranteed SPD
    """
    vals, vecs = np.linalg.eigh(cov)          # ascending eigenvalues
    vals = np.where(vals > _EPS, vals, _EPS)  # clamp small/negative values
    return (vecs * vals) @ vecs.T


def compute_spd(segment: np.ndarray) -> np.ndarray:
    """Compute SPD covariance matrix for one EEG segment.

    Parameters
    ----------
    segment : ndarray, shape (n_channels, n_samples)

    Returns
    -------
    ndarray, shape (n_channels, n_channels)
    """
    n = segment.shape[-1]
    cov = (segment @ segment.T) / (n - 1)
    return covariance_to_spd(cov)


def flatten_lower_triangular(spd: np.ndarray) -> np.ndarray:
    """Extract and flatten the lower-triangular elements of an SPD matrix.

    This produces the vector *u* from Eq. (4) in the paper.

    Parameters
    ----------
    spd : ndarray, shape (n_channels, n_channels)

    Returns
    -------
    ndarray, shape (n_channels*(n_channels+1)//2,)
    """
    idx = np.tril_indices(spd.shape[0])
    return spd[idx]


# ---------------------------------------------------------------------------
# 4-D spatio-temporal-spectral representation  (ManifoldNet input)
# ---------------------------------------------------------------------------

def build_4d_representation(
    signal: np.ndarray,
    fs: float,
    bands: list[str],
    window_size: int,
    n_windows: int,
) -> np.ndarray:
    """Build the 4-D spatio-temporal-spectral data representation.

    For each band and each non-overlapping time window the SPD covariance
    matrix is computed.  The result is stacked into a 4-D tensor.

    Parameters
    ----------
    signal     : ndarray, shape (n_channels, n_samples)
    fs         : float — sampling frequency in Hz
    bands      : list of band names (subset of BAND_RANGES)
    window_size: int — window length in samples  (Tlen × fs)
    n_windows  : int — number of time windows (nc)

    Returns
    -------
    ndarray, shape (n_windows, n_bands, n_channels, n_channels)
        Ready to use as ManifoldNet input (one trial).
    """
    n_channels = signal.shape[0]
    n_bands    = len(bands)

    # Pre-filter all bands at once
    band_signals = decompose_bands(signal, fs, bands)

    result = np.zeros((n_windows, n_bands, n_channels, n_channels), dtype=np.float32)

    for b_idx, band in enumerate(bands):
        bsig = band_signals[band]
        for t in range(n_windows):
            start = t * window_size
            end   = start + window_size
            if end > bsig.shape[-1]:
                break
            result[t, b_idx] = compute_spd(bsig[:, start:end])

    return result  # (n_windows, n_bands, C, C)


# ---------------------------------------------------------------------------
# Spatio-temporal representation  (BiLSTM input)
# Using variable-length windows via single-link hierarchical clustering
# ---------------------------------------------------------------------------

def _riemannian_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Approximate affine-invariant Riemannian distance between two SPD matrices.

    Uses the log-Euclidean approximation for numerical stability:
        d(A, B) = || log(A) - log(B) ||_F

    Parameters
    ----------
    A, B : ndarray, shape (n, n)

    Returns
    -------
    float
    """
    def matrix_log(M):
        vals, vecs = np.linalg.eigh(M)
        vals = np.maximum(vals, _EPS)
        return (vecs * np.log(vals)) @ vecs.T

    return np.linalg.norm(matrix_log(A) - matrix_log(B), "fro")


def variable_length_windows_clustering(
    signal: np.ndarray,
    n_windows: int,
    init_window: int = 64,
) -> list[tuple[int, int]]:
    """Segment EEG signal using single-link hierarchical clustering on the
    Riemannian manifold of SPD matrices (Section: Spatio-temporal data
    representation, and reference [41] in the paper).

    A greedy single-linkage strategy is used:
      1. Compute SPD matrices on short initial windows.
      2. Merge the two closest adjacent segments (by Riemannian distance)
         until the target number of windows is reached.

    Parameters
    ----------
    signal     : ndarray, shape (n_channels, n_samples)
    n_windows  : int — desired number of output segments (nc)
    init_window: int — initial granularity in samples

    Returns
    -------
    list of (start, end) sample index pairs, length == n_windows
    """
    n_samples = signal.shape[-1]

    # Build initial fine-grained segments
    starts = list(range(0, n_samples - init_window + 1, init_window))
    segments = [(s, s + init_window) for s in starts]

    # Pre-compute SPD for each segment
    def seg_spd(seg):
        return compute_spd(signal[:, seg[0]:seg[1]])

    spds = [seg_spd(s) for s in segments]

    # Single-link hierarchical merging
    while len(segments) > n_windows:
        # Find adjacent pair with minimum Riemannian distance
        best_idx, best_dist = 0, np.inf
        for i in range(len(segments) - 1):
            d = _riemannian_distance(spds[i], spds[i + 1])
            if d < best_dist:
                best_dist, best_idx = d, i

        # Merge segments[best_idx] and segments[best_idx+1]
        new_seg = (segments[best_idx][0], segments[best_idx + 1][1])
        new_spd = compute_spd(signal[:, new_seg[0]:new_seg[1]])

        segments = segments[:best_idx] + [new_seg] + segments[best_idx + 2:]
        spds     = spds[:best_idx]     + [new_spd]  + spds[best_idx + 2:]

    return segments


def build_spatiotemporal_representation(
    signal: np.ndarray,
    n_windows: int,
    use_variable_windows: bool = True,
    fixed_window_size: int = 512,
) -> np.ndarray:
    """Build the spatio-temporal data representation for the BiLSTM sub-model.

    Each time window is represented by the flattened lower-triangular
    elements of its SPD covariance matrix (the vector *u* in Eq. 4).

    Parameters
    ----------
    signal              : ndarray, shape (n_channels, n_samples)
    n_windows           : int — number of time windows (nc)
    use_variable_windows: bool — if True, use Riemannian clustering (VW);
                          otherwise use fixed-length windows (FW)
    fixed_window_size   : int — window length in samples for FW mode

    Returns
    -------
    ndarray, shape (n_windows, n_channels*(n_channels+1)//2)
        Ready to use as BiLSTM input (one trial).
    """
    n_channels = signal.shape[0]
    feat_dim   = n_channels * (n_channels + 1) // 2

    if use_variable_windows:
        windows = variable_length_windows_clustering(signal, n_windows)
    else:
        windows = [
            (t * fixed_window_size, (t + 1) * fixed_window_size)
            for t in range(n_windows)
        ]

    result = np.zeros((n_windows, feat_dim), dtype=np.float32)
    for t, (start, end) in enumerate(windows):
        if end > signal.shape[-1]:
            end = signal.shape[-1]
        spd       = compute_spd(signal[:, start:end])
        result[t] = flatten_lower_triangular(spd)

    return result  # (n_windows, feat_dim)


# ---------------------------------------------------------------------------
# Dataset-level preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess_dataset(
    eeg_data: np.ndarray,
    labels: np.ndarray,
    fs: float,
    bands: list[str],
    n_windows: int,
    window_size_sec: float,
    use_variable_windows: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess an entire EEG dataset into STSNet-ready representations.

    Parameters
    ----------
    eeg_data            : ndarray, shape (n_trials, n_channels, n_samples)
    labels              : ndarray, shape (n_trials,)
    fs                  : float — sampling frequency
    bands               : list of band names
    n_windows           : int — nc (number of time windows)
    window_size_sec     : float — window size in seconds for the 4-D repr.
    use_variable_windows: bool — VW vs FW for the spatio-temporal repr.

    Returns
    -------
    xd   : ndarray (n_trials, n_windows, n_bands, C, C)  — ManifoldNet input
    bi   : ndarray (n_trials, n_windows, C*(C+1)//2)      — BiLSTM input
    y    : ndarray (n_trials,)                            — labels
    """
    window_size = int(window_size_sec * fs)
    n_trials    = eeg_data.shape[0]

    xd_list, bi_list = [], []

    for trial_idx in range(n_trials):
        signal = eeg_data[trial_idx]  # (n_channels, n_samples)

        xd = build_4d_representation(signal, fs, bands, window_size, n_windows)
        bi = build_spatiotemporal_representation(
            signal, n_windows, use_variable_windows, window_size
        )

        xd_list.append(xd)
        bi_list.append(bi)

    return (
        np.stack(xd_list, axis=0).astype(np.float32),
        np.stack(bi_list, axis=0).astype(np.float32),
        labels.astype(np.int32),
    )
