"""Shared helpers for dataset-to-CSV conversion (no model dependencies)."""
from __future__ import annotations

import gzip
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

EEG_CHANNELS = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8",
    "FC6", "F4", "F8", "AF4",
]
KEYS = ["subject_id", "trial_id", "segment", "sample_idx"]
RATINGS = ["valence", "arousal", "dominance"]


def open_text(path: Path):
    return (gzip.open if path.suffix == ".gz" else open)(
        path, "wt", encoding="utf-8", newline=""
    )


def require_columns(frame: pd.DataFrame, columns: list[str], source) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{source}: missing columns {sorted(missing)}")


def numeric(value) -> np.ndarray:
    array = np.asarray(value)
    while array.dtype == object and array.size == 1:
        array = np.asarray(array.item())
    return np.asarray(array, dtype=float)


def samples_by_channels(value, widths: tuple[int, ...], source) -> np.ndarray:
    array = numeric(value)
    if array.ndim != 2:
        raise ValueError(f"{source}: expected a sample-by-channel matrix, got {array.shape}")
    if array.shape[1] in widths:
        return array
    if array.shape[0] in widths:
        return array.T
    raise ValueError(f"{source}: expected {widths} channels, got {array.shape}")


def mat_cells(value) -> list:
    """Flatten a MATLAB cell array without squeezing its numeric matrices."""
    array = np.asarray(value)
    if array.dtype != object:
        raise ValueError("Expected a MATLAB cell array of trials")
    return list(array.ravel())


def signal_frames(signal, subject, trial, segment, channels, labels, chunksize,
                  *, first_sample=1):
    for start in range(0, len(signal), chunksize):
        block = signal[start:start + chunksize]
        frame = pd.DataFrame(block, columns=channels)
        for name, value in reversed(list(zip(KEYS[:3], [subject, trial, segment]))):
            frame.insert(0, name, value)
        frame.insert(3, "sample_idx", np.arange(start, start + len(block)) + first_sample)
        for name, value in labels.items():
            frame[name] = value
        yield frame


def load_mat(path: Path, **kwargs):
    from scipy.io import loadmat
    try:
        return loadmat(path, **kwargs)
    except NotImplementedError as exc:
        raise ValueError(f"{path}: MATLAB v7.3/HDF5 is unsupported; use the dataset's "
                         "original preprocessed MAT release or exported CSVs") from exc


def skip_empty(source) -> None:
    warnings.warn(f"Skipping empty trial: {source}", stacklevel=2)
