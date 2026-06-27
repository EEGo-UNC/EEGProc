"""Data loading and representation utilities for JointAutoencoderVariationalClassifierV2.

Adapted from STSNet's dataset loading and label-processing conventions
(``STSNet/prepare_datasets.py`` and ``STSNet/train_eval.py``), but producing
the *raw-signal* windowed representation the V2 joint model expects --
``(n_windows, timesteps, n_channels)`` -- rather than STSNet's covariance/SPD
features.

Why not reuse STSNet's ``data_representation.py`` directly
------------------------------------------------------------
STSNet's ``build_4d_representation`` / ``build_spatiotemporal_representation``
collapse each time window into a covariance or flattened-SPD feature vector,
which is the right representation for STSNet's ManifoldNet/BiLSTM branches.
``JointAutoencoderVariationalClassifierV2`` is different: its decoder
reconstructs the *raw* EEG signal directly (MSE loss against raw amplitudes,
see ``CNN1DDecoder``), so the encoder must also be fed the raw signal. This
module therefore adapts only STSNet's *loading* and *label-binarization*
conventions, and replaces the windowing step with plain raw-signal
segmentation.

Pipeline
--------
1. ``load_raw_eeg_and_labels`` -- load the pre-converted ``*_eeg.npy`` /
   ``*_labels.npy`` arrays (same shapes STSNet's ``prepare_datasets.py``
   produces).
2. ``binarize_dreamer_labels`` -- median-split one label dimension
   (valence/arousal) into binary classes, matching
   ``STSNet/train_eval.py::binarize_labels``.
3. ``zscore_subject_eeg`` -- per-channel, per-subject z-scoring (STSNet
   itself does not normalize the raw signal anywhere; this is a
   stabilization step needed because this model reconstructs raw
   amplitudes). Uses only the subject's own unlabeled signal statistics, so
   it introduces no label leakage and no cross-subject leakage in CV.
4. ``window_trial_signal`` -- segment each trial into fixed-length,
   optionally overlapping windows, channels-last.
5. ``build_joint_v2_dataset`` -- ties the above together into the
   ``(feature_array, label_array, subject_id_array)`` triple expected by
   ``nested_lnso_cv`` / ``train_joint_autoencoder_variational_classifier_v2``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# DREAMER conventions, matching STSNet/train_eval.py::DATASET_CONFIGS["dreamer"]
# ---------------------------------------------------------------------------

DREAMER_FS = 128
"""Sampling rate (Hz). Matches STSNet's DREAMER config."""

DREAMER_MEDIAN_LABEL = 3
"""Median-split threshold for DREAMER's 1-5 Likert scale (STSNet hardcodes
this rather than computing it from data; both valence and arousal use the
same threshold)."""

DREAMER_LABEL_DIMS = {"valence": 0, "arousal": 1}
"""Index of each label dimension within the ``(n_subjects, n_trials, 2)``
labels array (matches STSNet/train_eval.py::LABEL_DIMS, minus 'dominance'
which DREAMER's CSV export does not contain)."""

# Default location of the pre-converted DREAMER arrays already produced via
# STSNet's prepare_datasets.py and checked into this repository.
_DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parents[2] / "supervised" / "stsnet" / "data"
)
DEFAULT_DREAMER_EEG_PATH = _DEFAULT_DATA_DIR / "dreamer_eeg.npy"
DEFAULT_DREAMER_LABELS_PATH = _DEFAULT_DATA_DIR / "dreamer_labels.npy"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_raw_eeg_and_labels(
    eeg_path: str | Path,
    labels_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load pre-converted ``(subject, trial, channel, sample)`` EEG + labels.

    Mirrors the array shapes produced by ``STSNet/prepare_datasets.py``:

        eeg    : float32, shape (n_subjects, n_trials, n_channels, n_samples)
        labels : float32, shape (n_subjects, n_trials, n_label_dims)

    Parameters
    ----------
    eeg_path : str or Path
        Path to a ``*_eeg.npy`` file (e.g. ``dreamer_eeg.npy``).
    labels_path : str or Path
        Path to the matching ``*_labels.npy`` file.

    Returns
    -------
    eeg : np.ndarray, shape (n_subjects, n_trials, n_channels, n_samples)
    labels : np.ndarray, shape (n_subjects, n_trials, n_label_dims)

    Raises
    ------
    ValueError
        If the arrays don't have the expected number of dimensions, or if
        the leading ``(n_subjects, n_trials)`` axes of ``eeg`` and ``labels``
        don't match.
    """
    eeg_path = Path(eeg_path)
    labels_path = Path(labels_path)

    if not eeg_path.is_file():
        raise FileNotFoundError(
            f"EEG array not found at {eeg_path}. Run STSNet's "
            f"prepare_datasets.py first, or pass an explicit --raw-eeg-npy path."
        )
    if not labels_path.is_file():
        raise FileNotFoundError(
            f"Labels array not found at {labels_path}. Run STSNet's "
            f"prepare_datasets.py first, or pass an explicit --raw-labels-npy path."
        )

    eeg = np.load(eeg_path, allow_pickle=False)
    labels = np.load(labels_path, allow_pickle=False)

    if eeg.ndim != 4:
        raise ValueError(
            "Expected eeg array of shape (n_subjects, n_trials, n_channels, "
            f"n_samples); got shape {eeg.shape}."
        )
    if labels.ndim != 3 or labels.shape[:2] != eeg.shape[:2]:
        raise ValueError(
            f"labels shape {labels.shape} does not match eeg shape {eeg.shape} "
            "in the (n_subjects, n_trials) leading dimensions."
        )

    return eeg.astype(np.float32), labels.astype(np.float32)


# ---------------------------------------------------------------------------
# Label processing  (matches STSNet/train_eval.py::binarize_labels)
# ---------------------------------------------------------------------------


def binarize_dreamer_labels(
    raw_labels: np.ndarray,
    dimension: Literal["valence", "arousal"],
    median: float = DREAMER_MEDIAN_LABEL,
) -> np.ndarray:
    """Binarize one label dimension via median split.

    Matches ``STSNet/train_eval.py::binarize_labels``: scores ``>= median``
    map to class 1 ("high"), scores ``< median`` map to class 0 ("low").

    Parameters
    ----------
    raw_labels : np.ndarray, shape (n_subjects, n_trials, n_label_dims)
        Raw (non-binarized) label array, as returned by
        ``load_raw_eeg_and_labels``.
    dimension : {"valence", "arousal"}
        Which label dimension to extract and binarize.
    median : float, default=3
        Threshold used for the median split (3 is the midpoint of DREAMER's
        1-5 Likert scale, matching STSNet's hardcoded DREAMER config).

    Returns
    -------
    np.ndarray of int32, shape (n_subjects, n_trials)
    """
    if dimension not in DREAMER_LABEL_DIMS:
        raise ValueError(
            f"dimension must be one of {list(DREAMER_LABEL_DIMS)}, got {dimension!r}."
        )
    dim_idx = DREAMER_LABEL_DIMS[dimension]
    dim_values = raw_labels[:, :, dim_idx]
    return (dim_values >= median).astype(np.int32)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def zscore_subject_eeg(subject_eeg: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score one subject's EEG per channel, using that subject's own statistics.

    STSNet does not normalize the raw signal anywhere in
    ``data_representation.py`` / ``train_eval.py`` -- it bandpass-filters and
    goes straight to covariance features, which are scale-sensitive but
    never explicitly normalized beforehand. Because this joint model's
    decoder reconstructs the *raw* signal (MSE loss against raw amplitudes),
    per-subject, per-channel z-scoring is applied here as a standard
    stabilization step. It only uses each subject's own unlabeled signal
    statistics (pooled across that subject's trials), so it introduces no
    label leakage and no cross-subject leakage across CV folds.

    Parameters
    ----------
    subject_eeg : np.ndarray, shape (n_trials, n_channels, n_samples)
        All trials for a single subject.
    eps : float, default=1e-8
        Numerical floor added to the standard deviation.

    Returns
    -------
    np.ndarray, same shape as ``subject_eeg``, z-scored per channel.
    """
    n_trials, n_channels, n_samples = subject_eeg.shape

    # Pool all trials/timepoints for this subject to get one mean/std per channel.
    flattened = np.moveaxis(subject_eeg, 1, 0).reshape(n_channels, -1)
    mean = flattened.mean(axis=1).reshape(1, n_channels, 1)
    std = flattened.std(axis=1).reshape(1, n_channels, 1)

    return ((subject_eeg - mean) / (std + eps)).astype(np.float32)


# ---------------------------------------------------------------------------
# Windowing (raw signal, NOT covariance -- contrast with STSNet's
# build_4d_representation / build_spatiotemporal_representation)
# ---------------------------------------------------------------------------


def window_trial_signal(
    trial_signal: np.ndarray,
    window_size: int,
    overlap: float = 0.0,
) -> np.ndarray:
    """Segment one trial's raw multichannel EEG into fixed-length windows.

    Unlike STSNet's covariance-based representations, this keeps the raw
    signal so it can be fed directly to ``CNN1DEncoder`` and reconstructed
    by ``CNN1DDecoder``.

    Parameters
    ----------
    trial_signal : np.ndarray, shape (n_channels, n_samples)
    window_size : int
        Window length in samples (becomes ``timesteps`` for the encoder).
    overlap : float, default=0.0
        Fractional overlap in ``[0, 1)`` between consecutive windows.

    Returns
    -------
    np.ndarray, shape (n_windows, window_size, n_channels)
        Channels-last, ready to use as rows of ``feature_array``.

    Raises
    ------
    ValueError
        If ``window_size`` or ``overlap`` are invalid, or if the trial is
        shorter than one window.
    """
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}.")
    if not (0.0 <= overlap < 1.0):
        raise ValueError(f"overlap must be in [0.0, 1.0), got {overlap}.")

    n_samples = trial_signal.shape[-1]
    hop = max(1, int(round(window_size * (1.0 - overlap))))

    windows = []
    start = 0
    while start + window_size <= n_samples:
        windows.append(trial_signal[:, start : start + window_size].T)
        start += hop

    if not windows:
        raise ValueError(
            f"trial_signal has {n_samples} samples, too short for "
            f"window_size={window_size}."
        )

    return np.stack(windows, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def build_joint_v2_dataset(
    eeg_path: str | Path = DEFAULT_DREAMER_EEG_PATH,
    labels_path: str | Path = DEFAULT_DREAMER_LABELS_PATH,
    label_dimension: Literal["valence", "arousal"] = "valence",
    window_size_sec: float = 4.0,
    fs: float = DREAMER_FS,
    overlap: float = 0.0,
    median_label: float = DREAMER_MEDIAN_LABEL,
    zscore: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(feature_array, label_array, subject_id_array)`` for joint v2 training.

    This is the adapted equivalent of STSNet's ``prepare_datasets.py``
    (loading) + ``train_eval.py`` (label binarization) pipeline, with the
    windowing step replaced by raw-signal segmentation (see
    ``window_trial_signal``) instead of STSNet's covariance/SPD
    representation, since the V2 joint model's decoder needs to reconstruct
    the raw signal.

    Parameters
    ----------
    eeg_path, labels_path : str or Path
        Paths to the pre-converted ``*_eeg.npy`` / ``*_labels.npy`` files
        (see ``load_raw_eeg_and_labels``). Default to the DREAMER arrays
        already checked into ``eegproc/supervised/stsnet/data/``.
    label_dimension : {"valence", "arousal"}, default="valence"
        Which label dimension to classify.
    window_size_sec : float, default=4.0
        Window length in seconds. With ``fs=128`` this gives 512-sample
        windows, matching STSNet's DREAMER config (15 windows per 60 s
        trial: 7680 / 512 = 15).
    fs : float, default=128
        Sampling frequency in Hz (matches STSNet's 128 Hz convention).
    overlap : float, default=0.0
        Fractional overlap between consecutive windows within a trial.
    median_label : float, default=3
        Median-split threshold (STSNet hardcodes 3 for DREAMER's 1-5 scale).
    zscore : bool, default=True
        Whether to z-score each subject's EEG per channel before windowing
        (see ``zscore_subject_eeg``). STSNet itself performs no such
        normalization; this is added here because this model's decoder
        reconstructs raw amplitudes directly.

    Returns
    -------
    feature_array : np.ndarray, shape (n_windows_total, timesteps, n_channels)
    label_array : np.ndarray, shape (n_windows_total,)
    subject_id_array : np.ndarray, shape (n_windows_total,)

    Raises
    ------
    ValueError
        Propagated from ``binarize_dreamer_labels`` / ``window_trial_signal``
        if the requested label dimension or window size are invalid.
    """
    eeg, raw_labels = load_raw_eeg_and_labels(eeg_path, labels_path)
    trial_labels = binarize_dreamer_labels(raw_labels, label_dimension, median_label)

    window_size = int(round(window_size_sec * fs))
    n_subjects, n_trials = eeg.shape[0], eeg.shape[1]

    feature_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    subject_chunks: list[np.ndarray] = []

    for subject_idx in range(n_subjects):
        subject_eeg = eeg[subject_idx]  # (n_trials, n_channels, n_samples)
        if zscore:
            subject_eeg = zscore_subject_eeg(subject_eeg)

        for trial_idx in range(n_trials):
            trial_windows = window_trial_signal(
                subject_eeg[trial_idx],
                window_size=window_size,
                overlap=overlap,
            )
            n_windows_this_trial = trial_windows.shape[0]

            feature_chunks.append(trial_windows)
            label_chunks.append(
                np.full(
                    n_windows_this_trial,
                    trial_labels[subject_idx, trial_idx],
                    dtype=np.int32,
                )
            )
            subject_chunks.append(
                np.full(n_windows_this_trial, subject_idx, dtype=np.int64)
            )

    feature_array = np.concatenate(feature_chunks, axis=0)
    label_array = np.concatenate(label_chunks, axis=0)
    subject_id_array = np.concatenate(subject_chunks, axis=0)

    return feature_array, label_array, subject_id_array
