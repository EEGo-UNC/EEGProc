import csv
from pathlib import Path

import numpy as np
import pytest

from eegproc.data.csv_cowen import EMOTION_ID_ORDER
from eegproc.deep_learning.prepare_datasets import (
    EEGEMOTIONS_N_CHANNELS,
    EEGEMOTIONS_N_TRIALS,
    EEGEMOTIONS_TRIAL_SAMPLES,
    load_cowen_27_mapping,
    load_eegemotions_joined_csv,
    prepare_eegemotions,
)


DREAMER_EEG_COLS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]


def _write_cowen_mapping(path: Path, names=None) -> None:
    """Write a mapping whose rows are NOT in emotion-ID order.

    Emotion ID ``k`` gets valence ``k + 0.1`` and arousal ``k + 0.2``; rows are
    written in reverse, like the quadrant-sorted converter output, so a loader
    that indexes by row position gets every label wrong.
    """
    names = list(EMOTION_ID_ORDER) if names is None else names
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["emotion", "valence", "arousal", "valence_sd", "arousal_sd", "rating_weight", "videos_with_rating", "quadrant"])
        for emotion_id, name in reversed(list(enumerate(names, start=1))):
            writer.writerow([name, emotion_id + 0.1, emotion_id + 0.2, 0.0, 0.0, 1.0, 1, "quadrant"])


def test_emotion_id_order_matches_eegemotions_27():
    """IDs 1..27 as published by the EEGEmotions-27 authors (alphabetical)."""
    assert len(EMOTION_ID_ORDER) == EEGEMOTIONS_N_TRIALS
    assert EMOTION_ID_ORDER[0] == "admiration"
    assert EMOTION_ID_ORDER[4] == "anger"          # README example: 12_5.0.txt -> Anger
    assert EMOTION_ID_ORDER[-1] == "surprise"
    assert list(EMOTION_ID_ORDER) == sorted(EMOTION_ID_ORDER)


def test_cowen_mapping_is_matched_by_name_not_row_position(tmp_path):
    path = tmp_path / "cowen_27_valence_arousal.csv"
    _write_cowen_mapping(path)

    names, va_map = load_cowen_27_mapping(str(path))

    assert names == list(EMOTION_ID_ORDER)
    expected = np.array([[k + 0.1, k + 0.2] for k in range(1, 28)], dtype=np.float32)
    np.testing.assert_allclose(va_map, expected)


def test_cowen_mapping_rejects_unknown_emotions(tmp_path):
    path = tmp_path / "cowen_27_valence_arousal.csv"
    _write_cowen_mapping(path, names=[f"emotion_{k}" for k in range(1, 28)])

    with pytest.raises(ValueError, match="27 emotions"):
        load_cowen_27_mapping(str(path))


def _write_eegemotions_csv(path: Path) -> None:
    fieldnames = ["subject_id", "trial_id", "segment", "sample_idx", *DREAMER_EEG_COLS, "age", "gender", "nation", "source_file", "source_file_label", "emo_label_cowen_27", "emo_label_ekman_6"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for subject_id, trial_limit in [(1, EEGEMOTIONS_N_TRIALS), (2, EEGEMOTIONS_N_TRIALS - 1)]:
            for trial_id in range(1, trial_limit + 1):
                for sample_idx in range(3):
                    row = {
                        "subject_id": subject_id,
                        "trial_id": trial_id,
                        "segment": "eeg_raw",
                        "sample_idx": sample_idx,
                        "age": 3,
                        "gender": 1,
                        "nation": 1,
                        "source_file": f"{subject_id}_{trial_id}.txt",
                        "source_file_label": trial_id,
                        "emo_label_cowen_27": trial_id,
                        "emo_label_ekman_6": 1,
                    }
                    for channel_index, channel_name in enumerate(DREAMER_EEG_COLS):
                        row[channel_name] = float(subject_id * 1000 + trial_id * 10 + sample_idx + channel_index)
                    writer.writerow(row)


def test_load_eegemotions_joined_csv_supports_both_label_modes(tmp_path):
    csv_path = tmp_path / "eegemotions_labeled.csv"
    cowen_path = tmp_path / "cowen_27_valence_arousal.csv"
    _write_eegemotions_csv(csv_path)
    _write_cowen_mapping(cowen_path)

    eeg_onehot, labels_onehot = load_eegemotions_joined_csv(str(csv_path), label_mode="emotion_27")
    assert eeg_onehot.shape == (1, EEGEMOTIONS_N_TRIALS, EEGEMOTIONS_N_CHANNELS, EEGEMOTIONS_TRIAL_SAMPLES)
    assert labels_onehot.shape == (1, EEGEMOTIONS_N_TRIALS, EEGEMOTIONS_N_TRIALS)
    np.testing.assert_allclose(labels_onehot.sum(axis=-1), 1.0)
    np.testing.assert_array_equal(labels_onehot[0, 0], np.eye(EEGEMOTIONS_N_TRIALS, dtype=np.float32)[0])

    eeg_va, labels_va = load_eegemotions_joined_csv(str(csv_path), label_mode="valence_arousal")
    assert eeg_va.shape == (1, EEGEMOTIONS_N_TRIALS, EEGEMOTIONS_N_CHANNELS, EEGEMOTIONS_TRIAL_SAMPLES)
    assert labels_va.shape == (1, EEGEMOTIONS_N_TRIALS, 2)
    # Trial k carries emotion ID k, so it must get that emotion's ratings.
    expected = np.array([[k + 0.1, k + 0.2] for k in range(1, 28)], dtype=np.float32)
    np.testing.assert_allclose(labels_va[0], expected)


def test_prepare_eegemotions_writes_output_files(tmp_path):
    csv_path = tmp_path / "eegemotions_labeled.csv"
    cowen_path = tmp_path / "cowen_27_valence_arousal.csv"
    _write_eegemotions_csv(csv_path)
    _write_cowen_mapping(cowen_path)

    prepare_eegemotions(str(tmp_path), str(tmp_path), label_mode="valence_arousal")

    eeg = np.load(tmp_path / "eegemotions_eeg.npy")
    labels = np.load(tmp_path / "eegemotions_labels.npy")
    assert eeg.shape == (1, EEGEMOTIONS_N_TRIALS, EEGEMOTIONS_N_CHANNELS, EEGEMOTIONS_TRIAL_SAMPLES)
    assert labels.shape == (1, EEGEMOTIONS_N_TRIALS, 2)