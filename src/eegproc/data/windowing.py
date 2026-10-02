"""Create model arrays without allowing windows to cross trial boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .schema import EEGFrame


@dataclass
class WindowedArrays:
    """Aligned windows, labels, integer identifiers, and original key lookups."""

    features: np.ndarray
    labels: np.ndarray
    subject_ids: np.ndarray
    trial_ids: np.ndarray
    subject_lookup: dict[int, tuple]
    trial_lookup: dict[int, tuple]


def _positive_integer(value, name: str) -> int:
    if isinstance(value, bool) or not np.isfinite(value) or int(value) != value or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def to_supervised_arrays(
    frame: EEGFrame,
    *,
    label_column: str | None = None,
    window_sec: float | None = None,
    window_samples: int | None = None,
    window_rows: int = 1,
    overlap: float = 0.0,
    normalize: str | None = None,
    label_transform: Callable | None = None,
    on_short_trial: str = "error",
) -> WindowedArrays:
    """Window each trial separately and return channels-last float32 arrays.

    ``kind='features'`` treats each row as a precomputed feature window. For
    signals, specify either ``window_sec`` or ``window_samples``. Incomplete
    trailing windows are dropped. Trial identifiers are globally unique across
    subjects, including when subjects reuse the same trial names.

    Optional ``subject_zscore`` uses all unlabeled rows from each subject.
    This includes held-out subjects' own signals; use ``normalize=None`` and
    fold-specific preprocessing when the evaluation requires training-only
    normalization statistics.
    """
    if not isinstance(frame, EEGFrame):
        raise TypeError("frame must be an EEGFrame.")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1).")
    if normalize not in {None, "subject_zscore"}:
        raise ValueError("normalize must be None or 'subject_zscore'.")
    if on_short_trial not in {"error", "skip"}:
        raise ValueError("on_short_trial must be 'error' or 'skip'.")
    if frame.kind == "features":
        if window_sec is not None or window_samples is not None:
            raise ValueError("Feature tables use window_rows, not signal window lengths.")
        size = _positive_integer(window_rows, "window_rows")
    else:
        if (window_sec is None) == (window_samples is None):
            raise ValueError("Signals require exactly one of window_sec or window_samples.")
        if window_sec is not None:
            if not np.isfinite(window_sec) or window_sec <= 0:
                raise ValueError("window_sec must be finite and positive.")
            window_samples = round(frame.fs * window_sec)
        size = _positive_integer(window_samples, "window_samples")
    hop = max(1, round(size * (1 - overlap)))
    label_column = label_column or (frame.label_columns[0] if frame.label_columns else None)
    if label_column not in frame.label_columns:
        raise ValueError("label_column must be declared in frame.label_columns.")

    table = frame.data.reset_index(drop=True)
    features = table[list(frame.feature_columns)].to_numpy(dtype=np.float32, copy=True)
    subject_columns = list(frame.subject_columns)
    trial_columns = list(dict.fromkeys((*frame.subject_columns, *frame.trial_columns)))
    subject_lookup, trial_lookup, subject_codes = {}, {}, {}
    for code, (key, block) in enumerate(table.groupby(subject_columns, sort=False, dropna=False)):
        key = key if isinstance(key, tuple) else (key,)
        subject_lookup[code] = key
        subject_codes[key] = code
        if normalize == "subject_zscore":
            # Contiguous channel-major reduction preserves the existing array
            # assembler's float32 arithmetic.
            values = features[block.index].T.copy()
            mean = values.mean(axis=1, keepdims=True)
            std = values.std(axis=1, keepdims=True)
            features[block.index] = ((values - mean) / (std + 1e-8)).T

    chunks, labels, subjects, trials = [], [], [], []
    for trial_code, (key, block) in enumerate(table.groupby(trial_columns, sort=False, dropna=False)):
        key = key if isinstance(key, tuple) else (key,)
        trial_lookup[trial_code] = key
        if frame.time_column is not None:
            if block[frame.time_column].duplicated().any():
                raise ValueError(f"Trial {key!r} has duplicate sample ordering values.")
            block = block.sort_values(frame.time_column, kind="stable")
        if block[label_column].nunique(dropna=False) != 1:
            raise ValueError(f"Trial {key!r} must have one consistent label.")
        if len(block) < size:
            if on_short_trial == "skip":
                continue
            raise ValueError(f"Trial {key!r} has {len(block)} rows, too short for window {size}.")
        subject_key = tuple(block[name].iloc[0] for name in frame.subject_columns)
        signal = features[block.index]
        windows = np.stack([signal[start:start + size] for start in range(0, len(signal) - size + 1, hop)])
        chunks.append(windows)
        labels.extend([block[label_column].iloc[0]] * len(windows))
        subjects.extend([subject_codes[subject_key]] * len(windows))
        trials.extend([trial_code] * len(windows))
    if not chunks:
        raise ValueError("No complete windows remain.")
    labels = np.asarray(labels)
    if label_transform is not None:
        labels = np.asarray(label_transform(labels))
        if labels.ndim < 1 or labels.shape[0] != len(subjects):
            raise ValueError("label_transform must preserve the number of windows.")
    return WindowedArrays(
        np.concatenate(chunks).astype(np.float32, copy=False), labels,
        np.asarray(subjects, dtype=np.int64), np.asarray(trials, dtype=np.int64),
        subject_lookup, trial_lookup,
    )
