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

Multi-dataset design
---------------------
Everything that varies between corpora (sampling rate, which label
dimensions exist and at what index, the median-split threshold, default
array paths) is captured in a ``DatasetConfig``. The pipeline functions
themselves (``binarize_labels``, ``zscore_subject_eeg``,
``window_trial_signal``, ``build_dataset``) are dataset-agnostic and take a
``DatasetConfig`` (or a registered dataset name) as a parameter rather than
hardcoding DREAMER's conventions. Adding a new corpus -- e.g. AMIGOS -- is
just a matter of registering a new ``DatasetConfig`` with the right
sampling rate, label layout, and median threshold; no pipeline code needs
to change.

Pipeline
--------
1. ``load_raw_eeg_and_labels`` -- load the pre-converted ``*_eeg.npy`` /
   ``*_labels.npy`` arrays (same shapes STSNet's ``prepare_datasets.py``
   produces, for any dataset).
2. ``binarize_labels`` -- median-split one label dimension (e.g.
   valence/arousal) into binary classes, matching
   ``STSNet/train_eval.py::binarize_labels``, generalized to look up the
   dimension index and threshold from a ``DatasetConfig``.
3. ``zscore_subject_eeg`` -- per-channel, per-subject z-scoring (STSNet
   itself does not normalize the raw signal anywhere; this is a
   stabilization step needed because this model reconstructs raw
   amplitudes). Uses only the subject's own unlabeled signal statistics, so
   it introduces no label leakage and no cross-subject leakage in CV.
4. ``window_trial_signal`` -- segment each trial into fixed-length,
   optionally overlapping windows, channels-last.
5. ``build_dataset`` -- ties the above together into the
   ``(feature_array, label_array, subject_id_array)`` triple expected by
   ``nested_lnso_cv`` / ``train_joint_autoencoder_variational_classifier_v2``,
   for whichever dataset's ``DatasetConfig`` is passed in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

# Root directory where pre-converted *_eeg.npy / *_labels.npy arrays live.
# Matches STSNet's prepare_datasets.py output location, shared across
# datasets that get converted via that script.
_DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parents[2] / "supervised" / "stsnet" / "data"
)


@dataclass(frozen=True)
class DatasetConfig:
    """Everything dataset-specific the joint-v2 pipeline needs to know.

    A ``DatasetConfig`` fully describes one corpus's conventions so that the
    loading / binarization / windowing functions below never need to special
    -case a dataset by name. To support a new dataset, construct a new
    ``DatasetConfig`` (optionally registering it via ``register_dataset``)
    rather than editing the pipeline functions.

    Attributes
    ----------
    name : str
        Short identifier (e.g. ``"dreamer"``, ``"amigos"``). Used only for
        error messages and dataset lookup via ``get_dataset_config``.
    fs : float
        Sampling rate (Hz) of the pre-converted EEG arrays.
    label_dims : dict[str, int]
        Maps a human-readable label name (e.g. ``"valence"``, ``"arousal"``)
        to its index within the ``(n_subjects, n_trials, n_label_dims)``
        labels array.
    median_label : float
        Threshold used for the median split in ``binarize_labels``. Scores
        ``>= median_label`` map to class 1 ("high"); scores ``< median_label``
        map to class 0 ("low").
    eeg_path, labels_path : Path
        Default locations of this dataset's pre-converted ``*_eeg.npy`` /
        ``*_labels.npy`` arrays.
    """

    name: str
    fs: float
    label_dims: dict[str, int]
    median_label: float
    eeg_path: Path
    labels_path: Path


DREAMER_CONFIG = DatasetConfig(
    name="dreamer",
    fs=128,
    # Matches STSNet/train_eval.py::LABEL_DIMS, minus 'dominance' which
    # DREAMER's CSV export does not contain.
    label_dims={"valence": 0, "arousal": 1},
    # DREAMER's CSV uses a 1-5 Likert scale; STSNet hardcodes the midpoint
    # (3) as the split threshold rather than computing it from data.
    median_label=3,
    eeg_path=_DEFAULT_DATA_DIR / "dreamer_eeg.npy",
    labels_path=_DEFAULT_DATA_DIR / "dreamer_labels.npy",
)

AMIGOS_CONFIG = DatasetConfig(
    name="amigos",
    # AMIGOS EEG is recorded with the 14-channel Emotiv EPOC at 128 Hz.
    fs=128,
    # AMIGOS' self-assessment labels (valence, arousal, dominance, liking,
    # familiarity) follow the same column convention STSNet uses for its
    # other "valence first, arousal second" datasets.
    label_dims={"valence": 0, "arousal": 1},
    # AMIGOS' SAM ratings use a 1-9 scale, so 5 is the scale midpoint (vs.
    # DREAMER's 1-5 scale, midpoint 3). Override at call time via
    # ``median_label=`` if you'd rather use each subject's empirical median.
    median_label=5,
    eeg_path=_DEFAULT_DATA_DIR / "amigos_eeg.npy",
    labels_path=_DEFAULT_DATA_DIR / "amigos_labels.npy",
)

_DATASET_REGISTRY: dict[str, DatasetConfig] = {
    DREAMER_CONFIG.name: DREAMER_CONFIG,
    AMIGOS_CONFIG.name: AMIGOS_CONFIG,
}


def register_dataset(config: DatasetConfig, *, overwrite: bool = False) -> None:
    """Register a ``DatasetConfig`` for lookup by name via ``get_dataset_config``.

    Parameters
    ----------
    config : DatasetConfig
        The configuration to register.
    overwrite : bool, default=False
        If ``False`` (default), raises if ``config.name`` is already
        registered. Pass ``True`` to deliberately replace an existing entry
        (e.g. to override DREAMER's or AMIGOS' default paths/thresholds).
    """
    if not overwrite and config.name in _DATASET_REGISTRY:
        raise ValueError(
            f"Dataset {config.name!r} is already registered; pass "
            "overwrite=True to replace it."
        )
    _DATASET_REGISTRY[config.name] = config


def get_dataset_config(dataset: str | DatasetConfig) -> DatasetConfig:
    """Resolve a dataset name (or pass-through ``DatasetConfig``) to a config.

    Parameters
    ----------
    dataset : str or DatasetConfig
        Either a registered dataset name (e.g. ``"dreamer"``, ``"amigos"``)
        or an already-built ``DatasetConfig`` (returned unchanged).

    Returns
    -------
    DatasetConfig

    Raises
    ------
    ValueError
        If ``dataset`` is a string that isn't registered.
    """
    if isinstance(dataset, DatasetConfig):
        return dataset
    try:
        return _DATASET_REGISTRY[dataset]
    except KeyError as exc:
        known = sorted(_DATASET_REGISTRY)
        raise ValueError(
            f"Unknown dataset {dataset!r}; registered datasets are {known}. "
            "Pass a DatasetConfig directly for one-off/custom datasets."
        ) from exc


# Backwards-compatible aliases for the previous DREAMER-only constants.
DREAMER_FS = DREAMER_CONFIG.fs
DREAMER_MEDIAN_LABEL = DREAMER_CONFIG.median_label
DREAMER_LABEL_DIMS = DREAMER_CONFIG.label_dims
DEFAULT_DREAMER_EEG_PATH = DREAMER_CONFIG.eeg_path
DEFAULT_DREAMER_LABELS_PATH = DREAMER_CONFIG.labels_path


# ---------------------------------------------------------------------------
# Loading (dataset-agnostic: shape conventions are the same across corpora)
# ---------------------------------------------------------------------------


def load_raw_eeg_and_labels(
    eeg_path: str | Path,
    labels_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load pre-converted ``(subject, trial, channel, sample)`` EEG + labels.

    Mirrors the array shapes produced by ``STSNet/prepare_datasets.py`` for
    any supported dataset:

        eeg    : float32, shape (n_subjects, n_trials, n_channels, n_samples)
        labels : float32, shape (n_subjects, n_trials, n_label_dims)

    Parameters
    ----------
    eeg_path : str or Path
        Path to a ``*_eeg.npy`` file (e.g. ``dreamer_eeg.npy``,
        ``amigos_eeg.npy``).
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


def binarize_labels(
    raw_labels: np.ndarray,
    dimension: str,
    dataset: str | DatasetConfig,
    median: float | None = None,
) -> np.ndarray:
    """Binarize one label dimension via median split.

    Matches ``STSNet/train_eval.py::binarize_labels``: scores ``>= median``
    map to class 1 ("high"), scores ``< median`` map to class 0 ("low").
    Generalized over ``binarize_dreamer_labels`` to take the label-dimension
    layout and default threshold from a ``DatasetConfig`` instead of
    hardcoding DREAMER's.

    Parameters
    ----------
    raw_labels : np.ndarray, shape (n_subjects, n_trials, n_label_dims)
        Raw (non-binarized) label array, as returned by
        ``load_raw_eeg_and_labels``.
    dimension : str
        Which label dimension to extract and binarize (must be a key in
        ``dataset.label_dims``, e.g. ``"valence"`` or ``"arousal"``).
    dataset : str or DatasetConfig
        Dataset name (e.g. ``"dreamer"``, ``"amigos"``) or an explicit
        ``DatasetConfig``, used to resolve ``dimension`` to a column index
        and to supply the default ``median`` threshold.
    median : float, optional
        Threshold used for the median split. Defaults to
        ``dataset.median_label`` (e.g. 3 for DREAMER's 1-5 scale, 5 for
        AMIGOS' 1-9 scale) if not given explicitly.

    Returns
    -------
    np.ndarray of int32, shape (n_subjects, n_trials)
    """
    config = get_dataset_config(dataset)
    if dimension not in config.label_dims:
        raise ValueError(
            f"dimension must be one of {list(config.label_dims)} for dataset "
            f"{config.name!r}, got {dimension!r}."
        )
    if median is None:
        median = config.median_label

    dim_idx = config.label_dims[dimension]
    dim_values = raw_labels[:, :, dim_idx]
    return (dim_values >= median).astype(np.int32)


def binarize_dreamer_labels(
    raw_labels: np.ndarray,
    dimension: Literal["valence", "arousal"],
    median: float = DREAMER_MEDIAN_LABEL,
) -> np.ndarray:
    """Deprecated DREAMER-only alias for ``binarize_labels``.

    Kept for backwards compatibility with existing call sites; prefer
    ``binarize_labels(raw_labels, dimension, dataset="dreamer")``.
    """
    return binarize_labels(raw_labels, dimension, dataset=DREAMER_CONFIG, median=median)


# ---------------------------------------------------------------------------
# Normalization (dataset-agnostic)
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
# build_4d_representation / build_spatiotemporal_representation).
# Dataset-agnostic: operates on whatever (n_channels, n_samples) it's given.
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


def build_dataset(
    dataset: str | DatasetConfig = "dreamer",
    eeg_path: str | Path | None = None,
    labels_path: str | Path | None = None,
    label_dimension: str = "valence",
    window_size_sec: float = 4.0,
    fs: float | None = None,
    overlap: float = 0.0,
    median_label: float | None = None,
    zscore: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(feature_array, label_array, subject_id_array)`` for joint v2 training.

    This is the adapted equivalent of STSNet's ``prepare_datasets.py``
    (loading) + ``train_eval.py`` (label binarization) pipeline, with the
    windowing step replaced by raw-signal segmentation (see
    ``window_trial_signal``) instead of STSNet's covariance/SPD
    representation, since the V2 joint model's decoder needs to reconstruct
    the raw signal.

    Dataset-specific conventions (sampling rate, label layout, median-split
    threshold, default array paths) are resolved from ``dataset`` via
    ``get_dataset_config``, so this same function works for DREAMER, AMIGOS,
    or any other corpus registered with ``register_dataset``. Any of
    ``eeg_path`` / ``labels_path`` / ``fs`` / ``median_label`` can still be
    overridden explicitly per call.

    Parameters
    ----------
    dataset : str or DatasetConfig, default="dreamer"
        Which dataset's conventions to use -- a registered name (e.g.
        ``"dreamer"``, ``"amigos"``) or an explicit ``DatasetConfig`` for
        one-off/custom datasets.
    eeg_path, labels_path : str or Path, optional
        Paths to the pre-converted ``*_eeg.npy`` / ``*_labels.npy`` files
        (see ``load_raw_eeg_and_labels``). Default to ``dataset``'s
        configured paths if not given.
    label_dimension : str, default="valence"
        Which label dimension to classify (must be a key in the resolved
        dataset's ``label_dims``, e.g. ``"valence"`` or ``"arousal"``).
    window_size_sec : float, default=4.0
        Window length in seconds. With ``fs=128`` this gives 512-sample
        windows, matching STSNet's DREAMER config (15 windows per 60 s
        trial: 7680 / 512 = 15).
    fs : float, optional
        Sampling frequency in Hz. Defaults to ``dataset``'s configured
        sampling rate if not given.
    overlap : float, default=0.0
        Fractional overlap between consecutive windows within a trial.
    median_label : float, optional
        Median-split threshold. Defaults to ``dataset``'s configured
        threshold (e.g. 3 for DREAMER's 1-5 scale, 5 for AMIGOS' 1-9 scale)
        if not given.
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
        Propagated from ``binarize_labels`` / ``window_trial_signal`` if the
        requested label dimension or window size are invalid, or if
        ``dataset`` is an unregistered name.
    """
    config = get_dataset_config(dataset)
    eeg_path = eeg_path if eeg_path is not None else config.eeg_path
    labels_path = labels_path if labels_path is not None else config.labels_path
    fs = fs if fs is not None else config.fs

    eeg, raw_labels = load_raw_eeg_and_labels(eeg_path, labels_path)
    trial_labels = binarize_labels(
        raw_labels, label_dimension, dataset=config, median=median_label
    )

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
    """Deprecated DREAMER-only alias for ``build_dataset``.

    Kept for backwards compatibility with existing call sites (e.g.
    ``nested_lnso_cv`` / ``train_joint_autoencoder_variational_classifier_v2``
    callers written before multi-dataset support); prefer
    ``build_dataset(dataset="dreamer", ...)`` -- or ``dataset="amigos"`` --
    going forward.
    """
    return build_dataset(
        dataset=DREAMER_CONFIG,
        eeg_path=eeg_path,
        labels_path=labels_path,
        label_dimension=label_dimension,
        window_size_sec=window_size_sec,
        fs=fs,
        overlap=overlap,
        median_label=median_label,
        zscore=zscore,
    )
