"""Smoke tests for joint_v2_data.py (data loading for the v2 joint model).

These exercise the loading/labeling/windowing pipeline against the real
bundled DREAMER arrays in ``eegproc/supervised/stsnet/data/`` so a broken
path, shape mismatch, or label-binarization bug surfaces immediately rather
than during a multi-hour nested-CV run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    from .joint_v2_data import (
        DatasetConfig,
        DEFAULT_DREAMER_EEG_PATH,
        DEFAULT_DREAMER_LABELS_PATH,
        DREAMER_FS,
        EEGEMOTIONS_27_CONFIG,
        binarize_dreamer_labels,
        build_joint_v2_dataset,
        build_dataset,
        load_raw_eeg_and_labels,
        window_trial_signal,
        zscore_subject_eeg,
    )
except ImportError:
    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))

    from joint_v2_data import (
        DatasetConfig,
        DEFAULT_DREAMER_EEG_PATH,
        DEFAULT_DREAMER_LABELS_PATH,
        DREAMER_FS,
        EEGEMOTIONS_27_CONFIG,
        binarize_dreamer_labels,
        build_joint_v2_dataset,
        build_dataset,
        load_raw_eeg_and_labels,
        window_trial_signal,
        zscore_subject_eeg,
    )


def smoke_test_load_raw_eeg_and_labels_matches_dreamer_shapes() -> None:
    """The bundled DREAMER arrays should load with the documented shapes."""
    eeg, labels = load_raw_eeg_and_labels(
        DEFAULT_DREAMER_EEG_PATH, DEFAULT_DREAMER_LABELS_PATH
    )

    assert eeg.ndim == 4
    assert labels.ndim == 3
    assert eeg.shape[:2] == labels.shape[:2]
    assert labels.shape[2] == 2  # [valence, arousal]

    n_subjects, n_trials, n_channels, n_samples = eeg.shape
    assert n_trials == 18
    assert n_channels == 14
    assert n_samples == 7680  # 60s @ 128Hz

    assert np.all(np.isfinite(eeg))
    assert np.all(np.isfinite(labels))


def smoke_test_binarize_dreamer_labels_is_binary_for_both_dimensions() -> None:
    """Both valence and arousal should binarize cleanly via median split."""
    _, raw_labels = load_raw_eeg_and_labels(
        DEFAULT_DREAMER_EEG_PATH, DEFAULT_DREAMER_LABELS_PATH
    )

    for dimension in ("valence", "arousal"):
        binary_labels = binarize_dreamer_labels(raw_labels, dimension, median=3)
        assert binary_labels.dtype == np.int32
        assert binary_labels.shape == raw_labels.shape[:2]
        assert set(np.unique(binary_labels).tolist()) <= {0, 1}


def smoke_test_window_trial_signal_shapes_and_count() -> None:
    """512-sample windows over a 7680-sample trial should yield exactly 15 windows."""
    n_channels, n_samples = 14, 7680
    window_size = 512  # 4s @ 128Hz, matching STSNet's DREAMER config

    rng = np.random.default_rng(0)
    trial_signal = rng.normal(size=(n_channels, n_samples)).astype(np.float32)

    windows = window_trial_signal(trial_signal, window_size=window_size, overlap=0.0)

    assert windows.shape == (15, window_size, n_channels)
    assert np.all(np.isfinite(windows))


def smoke_test_zscore_subject_eeg_normalizes_per_channel() -> None:
    """Per-channel mean/std across a subject's pooled trials should be ~0/~1."""
    rng = np.random.default_rng(1)
    n_trials, n_channels, n_samples = 18, 14, 7680
    # Deliberately large/uneven scale per channel to verify normalization.
    scales = rng.uniform(50.0, 200.0, size=(1, n_channels, 1)).astype(np.float32)
    offsets = rng.uniform(-100.0, 100.0, size=(1, n_channels, 1)).astype(np.float32)
    raw = rng.normal(size=(n_trials, n_channels, n_samples)).astype(np.float32)
    raw = raw * scales + offsets

    normalized = zscore_subject_eeg(raw)

    flattened = np.moveaxis(normalized, 1, 0).reshape(n_channels, -1)
    channel_means = flattened.mean(axis=1)
    channel_stds = flattened.std(axis=1)

    assert np.allclose(channel_means, 0.0, atol=1e-4)
    assert np.allclose(channel_stds, 1.0, atol=1e-4)


def smoke_test_build_joint_v2_dataset_end_to_end_shapes() -> None:
    """Full pipeline should produce CV-ready (feature, label, subject) arrays."""
    feature_array, label_array, subject_id_array = build_joint_v2_dataset(
        eeg_path=DEFAULT_DREAMER_EEG_PATH,
        labels_path=DEFAULT_DREAMER_LABELS_PATH,
        label_dimension="valence",
        window_size_sec=4.0,
        fs=DREAMER_FS,
        overlap=0.0,
        median_label=3,
        zscore=True,
    )

    assert feature_array.ndim == 3  # (n_windows, timesteps, n_channels)
    assert feature_array.shape[1] == 512  # 4s @ 128Hz
    assert feature_array.shape[2] == 14  # DREAMER channel count

    n_windows = feature_array.shape[0]
    assert label_array.shape == (n_windows,)
    assert subject_id_array.shape == (n_windows,)

    # 23 subjects * 18 trials * 15 windows/trial.
    assert n_windows == 23 * 18 * 15

    assert set(np.unique(label_array).tolist()) <= {0, 1}
    assert len(np.unique(subject_id_array)) == 23

    assert np.all(np.isfinite(feature_array))


def smoke_test_build_dataset_preserves_raw_27_way_labels(tmp_path) -> None:
    """Raw-label configs should keep the full 27-way trial vectors intact."""
    assert EEGEMOTIONS_27_CONFIG.name == "eegemotions_27"
    assert EEGEMOTIONS_27_CONFIG.label_mode == "identity"

    eeg_path = tmp_path / "eeg.npy"
    labels_path = tmp_path / "labels.npy"

    eeg = np.arange(16, dtype=np.float32).reshape(1, 1, 2, 8)
    labels = np.eye(27, dtype=np.float32).reshape(1, 1, 27)
    np.save(eeg_path, eeg)
    np.save(labels_path, labels)

    config = DatasetConfig(
        name="eegemotions_27_test",
        fs=2,
        label_dims={},
        median_label=0,
        label_mode="identity",
        eeg_path=eeg_path,
        labels_path=labels_path,
    )

    feature_array, label_array, subject_id_array = build_dataset(
        dataset=config,
        eeg_path=eeg_path,
        labels_path=labels_path,
        window_size_sec=4.0,
        fs=2,
        overlap=0.0,
        zscore=False,
    )

    assert feature_array.shape == (2, 4, 2)
    assert label_array.shape == (2, 27)
    np.testing.assert_array_equal(label_array[0], np.eye(27, dtype=np.float32)[0])
    assert subject_id_array.shape == (2,)


if __name__ == "__main__":
    smoke_test_load_raw_eeg_and_labels_matches_dreamer_shapes()
    smoke_test_binarize_dreamer_labels_is_binary_for_both_dimensions()
    smoke_test_window_trial_signal_shapes_and_count()
    smoke_test_zscore_subject_eeg_normalizes_per_channel()
    smoke_test_build_joint_v2_dataset_end_to_end_shapes()
    print("joint_v2_data smoke tests passed.")
