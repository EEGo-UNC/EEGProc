"""Load the archived SIC grading reference without importing model code.

The old SIC adapter was removed from the library branch. Grading needs only
its data preparation, including the exact float32 normalization round trip
used when archived physiological fractions were computed.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eegproc.deep_learning.joint_architectures.joint_models_data import build_joint_v2_dataset


@dataclass
class GradingTrials:
    features: np.ndarray
    subject_ids: np.ndarray
    trial_ids: np.ndarray
    normalization_offset: np.ndarray
    normalization_scale: np.ndarray
    feature_order: str


def load_grading_trials(config):
    """Match the legacy DREAMER loader's windows and reversible scaling."""
    if config.get("dataset", "dreamer") != "dreamer":
        raise ValueError("Archived physiology grading currently supports DREAMER.")
    if config.get("label_threshold_mode", "global") != "global":
        raise ValueError("Archived physiology grading requires global label thresholds.")
    eeg_path = Path(config["raw_eeg_npy"])
    fs = float(config.get("fs", 128.0))
    window_sec = float(config.get("window_sec", 1.0))
    overlap = float(config.get("window_overlap", 0.0))
    features, _, _ = build_joint_v2_dataset(
        eeg_path=eeg_path,
        labels_path=Path(config["raw_labels_npy"]),
        dataset="dreamer",
        label_dimension=config["label_dimension"],
        window_size_sec=window_sec,
        fs=fs,
        overlap=overlap,
        median_label=float(config.get("median_label", 3.0)),
        zscore=False,
    )
    x = np.asarray(features, dtype=np.float32)
    mode = config.get("window_normalization", "global_rms")
    if mode == "none":
        offset = np.zeros((len(x), 1, 1), dtype=np.float32)
        scale = np.ones((len(x), 1, 1), dtype=np.float32)
    elif mode == "global_rms":
        offset = np.zeros((len(x), 1, 1), dtype=np.float32)
        scale = np.sqrt(np.mean(np.square(x, dtype=np.float64),
                                axis=(1, 2), keepdims=True)).astype(np.float32)
        scale = np.maximum(scale, 1e-6)
    elif mode == "feature_zscore":
        offset = np.mean(x, axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
        scale = np.std(x, axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
        scale = np.maximum(scale, 1e-6)
    else:
        raise ValueError("window_normalization must be none, global_rms, or feature_zscore.")
    normalized = ((x - offset) / scale).astype(np.float32)
    raw = np.load(eeg_path, mmap_mode="r", allow_pickle=False)
    n_subjects, n_trials, _, n_samples = raw.shape
    window_size = int(round(window_sec * fs))
    hop = max(1, int(round(window_size * (1.0 - overlap))))
    n_windows = 1 + (n_samples - window_size) // hop
    n_total_trials = n_subjects * n_trials
    if len(x) != n_total_trials * n_windows:
        raise ValueError("Prepared windows do not align with the archived trial layout.")
    return GradingTrials(
        features=normalized.reshape(n_total_trials, n_windows, *x.shape[1:]),
        subject_ids=np.repeat(np.arange(n_subjects, dtype=np.int64), n_trials),
        trial_ids=np.tile(np.arange(n_trials, dtype=np.int64), n_subjects),
        normalization_offset=offset.reshape(n_total_trials, n_windows, *offset.shape[1:]),
        normalization_scale=scale.reshape(n_total_trials, n_windows, *scale.shape[1:]),
        feature_order=str(config.get("feature_order") or "channel-major"),
    )
