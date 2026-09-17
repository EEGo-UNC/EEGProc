"""
prepare_datasets.py
===================
Convert raw DEAP, DREAMER, AMIGOS, and EEGEmotions dataset files into the NumPy format
expected by STSNet's train_eval.py (and by joint_v2_data.py's
``build_dataset``):

    {dataset}_eeg.npy    — float32, shape (n_subjects, n_trials, n_channels, n_samples)
    {dataset}_labels.npy — float32, shape (n_subjects, n_trials, n_label_dims)

Usage
-----
    # DEAP  (point to the folder containing s01.dat ... s32.dat)
    python prepare_datasets.py --dataset deap --input_dir /path/to/deap/data_preprocessed_python

    # DREAMER  (point to the folder containing dreamer_joined.csv)
    python prepare_datasets.py --dataset dreamer --input_dir /path/to/dreamer

    # AMIGOS  (point to the folder containing amigos_joined.csv, or to the
    # legacy unzipped 'data_preprocessed' folder containing
    # Data_Preprocessed_P01.mat ... Data_Preprocessed_P40.mat)
    python prepare_datasets.py --dataset amigos --input_dir /path/to/amigos

    # All three
    python prepare_datasets.py --dataset all \
        --deap_dir /path/to/deap/data_preprocessed_python \
        --dreamer_dir /path/to/dreamer \
        --amigos_dir /path/to/amigos/data_preprocessed

    # EEGEmotions (point to the folder containing eegemotions_labeled.csv and cowen_27_valence_arousal.csv)
    python prepare_datasets.py --dataset eegemotions --input_dir /path/to/eegemotions \
        --eegemotions_label_mode emotion_27

    # EEGEmotions with valence/arousal labels from the Cowen 27 mapping
    python prepare_datasets.py --dataset eegemotions --input_dir /path/to/eegemotions \
        --eegemotions_label_mode valence_arousal

Output files are written to the current working directory (or --output_dir).

Dataset structures
------------------
DEAP (preprocessed Python version):
    s01.dat ... s32.dat — each is a pickle dict with keys:
        'data'   : (40, 40, 8064)  trials × (32 EEG + 8 peripheral) × samples
                   We keep only the first 32 channels (EEG).
                   Signals are already downsampled to 128 Hz and filtered 4–45 Hz.
                   The first 3 seconds of each trial (baseline) are pre-removed
                   in some versions; we trim to the last 60 s (7680 samples).
        'labels' : (40, 4)  valence, arousal, dominance, liking  (1–9 scale)

DREAMER (dreamer_joined.csv):
    Long/tidy format — one row per EEG sample with columns:
        subject_id, trial_id, segment, sample_idx,
        AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4,
        ECG1, ECG2, valence, arousal, dominance
    This script currently uses subject_id/trial_id for grouping,
    the 14 EEG channels listed above for signals, and valence/arousal for labels.
    Extra columns (segment, sample_idx, ECG1, ECG2, dominance) are ignored.
    Signals are already at 128 Hz and filtered 4–30 Hz.
    Trial lengths vary; we take 60 s from the middle (7680 samples).
    NOTE: even though the CSV includes dominance, this converter currently writes
    labels with shape (n_subjects, 18, 2) using [valence, arousal] only.

AMIGOS (joined CSV or legacy preprocessed Matlab version):
    ``amigos_joined.csv`` is a long-form table with one row per EEG sample.
    It contains ``subject_id``, ``trial_id``, ``sample_idx``, the 14 EEG
    channels in the same Emotiv EPOC order as DREAMER, two peripheral ECG
    columns that are ignored here, and ``valence``, ``arousal``,
    ``dominance`` labels. This script groups by subject/trial, keeps the 14
    EEG channels, extracts the center 60 s from each trial, and writes raw
    labels with shape ``(n_subjects, n_trials, 2)`` using [valence,
    arousal] only.

    For backwards compatibility, the legacy ``Data_Preprocessed_P01.mat`` ...
    ``Data_Preprocessed_P40.mat`` layout is still supported if the joined CSV
    is not present.

EEGEmotions (joined CSV + Cowen 27 mapping):
        ``eegemotions_labeled.csv`` stores one EEG sample per row with columns:
                subject_id, trial_id, segment, sample_idx,
                AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4,
                age, gender, nation, source_file, source_file_label,
                emo_label_cowen_27, emo_label_ekman_6
        This converter uses the 14 EEG channels above, keeps the first 27 emotion
        trials for each complete subject, and extracts the center 60 s from each
        trial. The labels can be written in one of two modes:

        - ``emotion_27``: one-hot vectors with shape ``(n_subjects, 27)``
        - ``valence_arousal``: mapped Cowen valence/arousal vectors with shape
            ``(n_subjects, 2)``

        The accompanying ``cowen_27_valence_arousal.csv`` file provides the
        emotion-to-valence/arousal mapping.
"""

import argparse
import csv
import os
import pickle
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEAP_N_SUBJECTS   = 32
DEAP_N_TRIALS     = 40
DEAP_N_CHANNELS   = 32          # first 32 of the 40 recorded channels are EEG
DEAP_FS           = 128
DEAP_TRIAL_SECS   = 60
DEAP_TRIAL_SAMPLES= DEAP_TRIAL_SECS * DEAP_FS   # 7680

DREAMER_N_SUBJECTS  = 23
DREAMER_N_TRIALS    = 18
DREAMER_N_CHANNELS  = 14
DREAMER_FS          = 128
DREAMER_TRIAL_SECS  = 60
DREAMER_TRIAL_SAMPLES = DREAMER_TRIAL_SECS * DREAMER_FS  # 7680

AMIGOS_N_SUBJECTS_TOTAL = 40
# Participant IDs with known-invalid data in the public preprocessed
# release (missing/corrupt signal or label arrays). Matches TorchEEG's
# AMIGOSDataset default `skipped_subjects`.
AMIGOS_SKIPPED_SUBJECTS = [9, 12, 21, 22, 23, 24, 33]
AMIGOS_N_SUBJECTS  = AMIGOS_N_SUBJECTS_TOTAL - len(AMIGOS_SKIPPED_SUBJECTS)  # 33
# Only the 16 short-video trials -- watched by every retained subject, so
# the (n_subjects, n_trials, ...) array stays rectangular. The 4 long-video
# trials (indices 16-19) are skipped entirely by several subjects and are
# not used by this converter.
AMIGOS_N_TRIALS    = 16
AMIGOS_N_CHANNELS  = 14
AMIGOS_FS          = 128
AMIGOS_TRIAL_SECS  = 60
AMIGOS_TRIAL_SAMPLES = AMIGOS_TRIAL_SECS * AMIGOS_FS  # 7680
# Column order within each `labels_selfassessment` trial entry. Note this is
# arousal-then-valence -- the opposite of DREAMER's CSV column order.
AMIGOS_LABEL_NAMES = [
    "arousal", "valence", "dominance", "liking", "familiarity",
    "neutral", "disgust", "happiness", "surprise", "anger", "fear", "sadness",
]

EEGEMOTIONS_N_TRIALS = 27
EEGEMOTIONS_N_CHANNELS = 14
EEGEMOTIONS_FS = 128
EEGEMOTIONS_TRIAL_SECS = 60
EEGEMOTIONS_TRIAL_SAMPLES = EEGEMOTIONS_TRIAL_SECS * EEGEMOTIONS_FS  # 7680
EEGEMOTIONS_LABEL_COL = "emo_label_cowen_27"


# ---------------------------------------------------------------------------
# DEAP
# ---------------------------------------------------------------------------

def load_deap_subject(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one DEAP subject file.

    Parameters
    ----------
    filepath : str — path to s0X.dat

    Returns
    -------
    eeg    : float32 ndarray, shape (40, 32, 7680)
    labels : float32 ndarray, shape (40, 4)  [valence, arousal, dominance, liking]
    """
    with open(filepath, "rb") as f:
        subject = pickle.load(f, encoding="latin1")

    data   = subject["data"].astype(np.float32)    # (40, 40, 8064)
    labels = subject["labels"].astype(np.float32)  # (40, 4)

    # Keep only the 32 EEG channels
    eeg = data[:, :DEAP_N_CHANNELS, :]             # (40, 32, 8064)

    # Trim to the last 60 s — some DEAP versions prepend a 3 s baseline
    # (8064 - 7680 = 384 = 3 s × 128 Hz)
    if eeg.shape[-1] > DEAP_TRIAL_SAMPLES:
        eeg = eeg[:, :, -DEAP_TRIAL_SAMPLES:]      # (40, 32, 7680)
    elif eeg.shape[-1] < DEAP_TRIAL_SAMPLES:
        # Pad with zeros if somehow shorter (shouldn't happen with preprocessed data)
        pad = DEAP_TRIAL_SAMPLES - eeg.shape[-1]
        eeg = np.pad(eeg, ((0, 0), (0, 0), (0, pad)))

    return eeg, labels


def prepare_deap(input_dir: str, output_dir: str) -> None:
    """Convert all DEAP subject files to a single pair of .npy arrays.

    Parameters
    ----------
    input_dir  : str — folder containing s01.dat … s32.dat
    output_dir : str — where to write deap_eeg.npy and deap_labels.npy
    """
    all_eeg, all_labels = [], []

    for subj_idx in range(1, DEAP_N_SUBJECTS + 1):
        filename = os.path.join(input_dir, f"s{subj_idx:02d}.dat")
        if not os.path.isfile(filename):
            raise FileNotFoundError(
                f"Expected DEAP file not found: {filename}\n"
                f"Make sure --deap_dir points to the 'data_preprocessed_python' folder."
            )
        eeg, labels = load_deap_subject(filename)
        all_eeg.append(eeg)
        all_labels.append(labels)
        print(f"  DEAP subject {subj_idx:02d}/{DEAP_N_SUBJECTS}  "
              f"eeg={eeg.shape}  labels={labels.shape}")

    eeg_arr    = np.stack(all_eeg,    axis=0)  # (32, 40, 32, 7680)
    labels_arr = np.stack(all_labels, axis=0)  # (32, 40, 4)

    eeg_path    = os.path.join(output_dir, "deap_eeg.npy")
    labels_path = os.path.join(output_dir, "deap_labels.npy")
    np.save(eeg_path,    eeg_arr)
    np.save(labels_path, labels_arr)

    print(f"\nDEAP saved:")
    print(f"  {eeg_path}    {eeg_arr.shape}  {eeg_arr.dtype}")
    print(f"  {labels_path} {labels_arr.shape}  {labels_arr.dtype}")
    _print_label_stats("DEAP", labels_arr)


# ---------------------------------------------------------------------------
# DREAMER  (CSV version)
# ---------------------------------------------------------------------------

DREAMER_EEG_COLS = ["AF3", "F7", "F3", "FC5", "T7", "P7",
                    "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]

AMIGOS_SHORT_TRIALS = 16


def _extract_centre(arr: np.ndarray, target: int) -> np.ndarray:
    """Return `target` samples from the centre of `arr` (n_samples, n_ch),
    repeat-padding if the trial is shorter than `target`.

    Shared helper: used by both the DREAMER (CSV) and AMIGOS (.mat)
    converters below, since both have variable per-trial trial lengths that
    need to be reduced to a fixed sample count.
    """
    n = arr.shape[0]
    if n >= target:
        mid   = n // 2
        start = mid - target // 2
        return arr[start: start + target, :]
    repeats = (target // n) + 1
    return np.tile(arr, (repeats, 1))[:target, :]


def load_cowen_27_mapping(filepath: str) -> tuple[list[str], np.ndarray]:
    """Load the 27-row Cowen mapping in file order."""
    filepath = os.fspath(filepath)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Cowen 27 mapping not found at {filepath}.")

    rows: list[tuple[str, float, float]] = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Cowen mapping {filepath} does not have a header row.")
        required = {"emotion", "valence", "arousal"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Cowen mapping {filepath} is missing required columns: {sorted(missing)}"
            )
        for row in reader:
            rows.append((row["emotion"].strip(), float(row["valence"]), float(row["arousal"])))

    if len(rows) != 27:
        raise ValueError(f"Expected 27 rows in Cowen mapping {filepath}, found {len(rows)}.")

    emotion_names = [name for name, _, _ in rows]
    va_map = np.asarray([(valence, arousal) for _, valence, arousal in rows], dtype=np.float32)
    return emotion_names, va_map


def _encode_eegemotions_label(label_id: int, label_mode: str, va_map: np.ndarray) -> np.ndarray:
    """Encode one EEGEmotions label as either one-hot or valence/arousal."""
    if not 1 <= label_id <= EEGEMOTIONS_N_TRIALS:
        raise ValueError(
            f"EEGEmotions label id {label_id} is out of range 1..{EEGEMOTIONS_N_TRIALS}."
        )

    mode = label_mode.lower()
    if mode == "emotion_27":
        label = np.zeros(EEGEMOTIONS_N_TRIALS, dtype=np.float32)
        label[label_id - 1] = 1.0
        return label
    if mode == "valence_arousal":
        return np.asarray(va_map[label_id - 1], dtype=np.float32)

    raise ValueError(
        "Unsupported EEGEmotions label mode: "
        f"{label_mode!r}. Expected 'emotion_27' or 'valence_arousal'."
    )


def load_amigos_joined_csv(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """Load the joined AMIGOS CSV and convert it to raw trial arrays.

    The joined CSV stores one EEG sample per row, so this loader streams the
    file in subject/trial order, groups rows into trials, keeps only the 14
    EEG channels, and extracts the center 60 s from each trial.
    """
    filepath = os.fspath(filepath)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"AMIGOS joined CSV not found at {filepath}.")

    required_columns = {
        "subject_id",
        "trial_id",
        "sample_idx",
        *DREAMER_EEG_COLS,
        "valence",
        "arousal",
    }

    all_subject_eeg: list[np.ndarray] = []
    all_subject_labels: list[np.ndarray] = []

    current_subject_id: int | None = None
    current_trial_id: int | None = None
    current_trial_rows: list[list[float]] = []
    current_trial_label: tuple[float, float] | None = None
    current_subject_trials: list[np.ndarray] = []
    current_subject_labels: list[np.ndarray] = []
    expected_trials_per_subject: int = AMIGOS_SHORT_TRIALS

    def finalize_trial() -> None:
        nonlocal current_trial_rows, current_trial_label
        if not current_trial_rows:
            return

        trial_signal = np.asarray(current_trial_rows, dtype=np.float32)
        if trial_signal.shape[1] != len(DREAMER_EEG_COLS):
            raise ValueError(
                f"Expected {len(DREAMER_EEG_COLS)} EEG channels in AMIGOS CSV, "
                f"got shape {trial_signal.shape}."
            )
        if current_trial_label is None:
            raise ValueError("Encountered AMIGOS trial with no labels.")

        trial_windows = _extract_centre(trial_signal, AMIGOS_TRIAL_SAMPLES).T
        current_subject_trials.append(trial_windows)
        current_subject_labels.append(
            np.asarray(current_trial_label, dtype=np.float32)
        )
        current_trial_rows = []
        current_trial_label = None

    def finalize_subject(subject_id: int) -> None:
        nonlocal current_subject_trials, current_subject_labels, expected_trials_per_subject
        if not current_subject_trials:
            return

        trial_count = len(current_subject_trials)
        if trial_count != expected_trials_per_subject:
            raise ValueError(
                f"AMIGOS CSV is not rectangular: subject {subject_id} has "
                f"{trial_count} trials, expected {expected_trials_per_subject}."
            )

        all_subject_eeg.append(np.stack(current_subject_trials, axis=0))
        all_subject_labels.append(np.stack(current_subject_labels, axis=0))
        current_subject_trials = []
        current_subject_labels = []

    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"AMIGOS CSV {filepath} does not have a header row.")
        missing = required_columns - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"AMIGOS CSV {filepath} is missing required columns: {sorted(missing)}"
            )

        for row in reader:
            subject_id = int(row["subject_id"])
            trial_id = int(row["trial_id"])
            sample_idx = int(row["sample_idx"])
            label = (float(row["valence"]), float(row["arousal"]))
            eeg_row = [float(row[col]) for col in DREAMER_EEG_COLS]

            if current_subject_id is None:
                current_subject_id = subject_id
                current_trial_id = trial_id
            elif subject_id < current_subject_id or (
                subject_id == current_subject_id and trial_id < current_trial_id
            ):
                raise ValueError(
                    "AMIGOS CSV must be sorted by subject_id, trial_id, and "
                    "sample_idx."
                )
            elif subject_id != current_subject_id:
                finalize_trial()
                finalize_subject(current_subject_id)
                current_subject_id = subject_id
                current_trial_id = trial_id
            elif trial_id != current_trial_id:
                finalize_trial()
                current_trial_id = trial_id

            if trial_id > AMIGOS_SHORT_TRIALS:
                current_trial_rows = []
                current_trial_label = None
                continue

            if current_trial_label is None:
                current_trial_label = label
            elif current_trial_label != label:
                raise ValueError(
                    f"AMIGOS labels changed within subject {subject_id}, trial {trial_id}."
                )

            current_trial_rows.append(eeg_row)

    if current_subject_id is None:
        raise ValueError(f"AMIGOS CSV {filepath} did not contain any rows.")

    finalize_trial()
    finalize_subject(current_subject_id)

    eeg_arr = np.stack(all_subject_eeg, axis=0)
    labels_arr = np.stack(all_subject_labels, axis=0)
    return eeg_arr.astype(np.float32), labels_arr.astype(np.float32)


def load_eegemotions_joined_csv(filepath: str, label_mode: str = "emotion_27") -> tuple[np.ndarray, np.ndarray]:
    """Load eegemotions_labeled.csv and convert it to raw trial arrays."""
    filepath = os.fspath(filepath)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"EEGEmotions labeled CSV not found at {filepath}.")

    cowen_path = str(Path(filepath).with_name("cowen_27_valence_arousal.csv"))
    _, va_map = load_cowen_27_mapping(cowen_path)

    required_columns = {
        "subject_id",
        "trial_id",
        "sample_idx",
        *DREAMER_EEG_COLS,
        EEGEMOTIONS_LABEL_COL,
    }

    all_subject_eeg: list[np.ndarray] = []
    all_subject_labels: list[np.ndarray] = []
    skipped_subjects: list[int] = []

    current_subject_id: int | None = None
    current_trial_id: int | None = None
    current_trial_rows: list[list[float]] = []
    current_trial_label_id: int | None = None
    current_subject_trials: list[np.ndarray] = []
    current_subject_labels: list[np.ndarray] = []
    skipped_sample_rows = 0

    def finalize_trial() -> None:
        nonlocal current_trial_rows, current_trial_label_id
        if not current_trial_rows:
            return

        trial_signal = np.asarray(current_trial_rows, dtype=np.float32)
        if trial_signal.shape[1] != len(DREAMER_EEG_COLS):
            raise ValueError(
                f"Expected {len(DREAMER_EEG_COLS)} EEG channels in EEGEmotions CSV, got shape {trial_signal.shape}."
            )
        if current_trial_label_id is None:
            raise ValueError("Encountered EEGEmotions trial with no labels.")

        trial_windows = _extract_centre(trial_signal, EEGEMOTIONS_TRIAL_SAMPLES).T
        current_subject_trials.append(trial_windows)
        current_subject_labels.append(_encode_eegemotions_label(current_trial_label_id, label_mode, va_map))
        current_trial_rows = []
        current_trial_label_id = None

    def finalize_subject(subject_id: int) -> None:
        nonlocal current_subject_trials, current_subject_labels
        if not current_subject_trials:
            return

        trial_count = len(current_subject_trials)
        if trial_count != EEGEMOTIONS_N_TRIALS:
            skipped_subjects.append(subject_id)
            current_subject_trials = []
            current_subject_labels = []
            return

        all_subject_eeg.append(np.stack(current_subject_trials, axis=0))
        all_subject_labels.append(np.stack(current_subject_labels, axis=0))
        current_subject_trials = []
        current_subject_labels = []

    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"EEGEmotions CSV {filepath} does not have a header row.")
        missing = required_columns - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"EEGEmotions CSV {filepath} is missing required columns: {sorted(missing)}"
            )

        for row in reader:
            subject_id = int(row["subject_id"])
            trial_id = int(row["trial_id"])
            sample_idx = int(row["sample_idx"])
            label_id = int(row[EEGEMOTIONS_LABEL_COL])
            try:
                eeg_row = [float(row[col]) for col in DREAMER_EEG_COLS]
            except ValueError:
                skipped_sample_rows += 1
                continue

            if current_subject_id is None:
                current_subject_id = subject_id
                current_trial_id = trial_id
            elif subject_id < current_subject_id or (
                subject_id == current_subject_id and trial_id < current_trial_id
            ):
                raise ValueError(
                    "EEGEmotions CSV must be sorted by subject_id, trial_id, and sample_idx."
                )
            elif subject_id != current_subject_id:
                finalize_trial()
                finalize_subject(current_subject_id)
                current_subject_id = subject_id
                current_trial_id = trial_id
            elif trial_id != current_trial_id:
                finalize_trial()
                current_trial_id = trial_id

            if current_trial_label_id is None:
                current_trial_label_id = label_id
            elif current_trial_label_id != label_id:
                raise ValueError(
                    f"EEGEmotions labels changed within subject {subject_id}, trial {trial_id}."
                )

            current_trial_rows.append(eeg_row)

    if current_subject_id is None:
        raise ValueError(f"EEGEmotions CSV {filepath} did not contain any rows.")

    finalize_trial()
    finalize_subject(current_subject_id)

    if not all_subject_eeg:
        raise ValueError(
            "EEGEmotions conversion produced no complete subjects. Check that the input CSV contains all 27 trials for at least one subject."
        )

    if skipped_subjects:
        print(f"  EEGEmotions skipped incomplete subjects: {sorted(set(skipped_subjects))}")
    if skipped_sample_rows:
        print(f"  EEGEmotions skipped malformed sample rows: {skipped_sample_rows}")

    eeg_arr = np.stack(all_subject_eeg, axis=0)
    labels_arr = np.stack(all_subject_labels, axis=0)
    return eeg_arr.astype(np.float32), labels_arr.astype(np.float32)


def prepare_dreamer(input_dir: str, output_dir: str) -> None:
    """Convert dreamer_joined.csv to a single pair of .npy arrays.

    Parameters
    ----------
    input_dir  : str — folder containing dreamer_joined.csv
    output_dir : str — where to write dreamer_eeg.npy and dreamer_labels.npy

    Output shapes
    -------------
    dreamer_eeg.npy    : float32 (n_subjects, 18, 14, 7680)
    dreamer_labels.npy : float32 (n_subjects, 18, 2)  [valence, arousal]
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required to read the CSV: pip install pandas")

    csv_path = os.path.join(input_dir, "dreamer_joined.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"dreamer_joined.csv not found in {input_dir}.\n"
            f"Make sure --dreamer_dir points to the folder containing dreamer_joined.csv."
        )

    print(f"  Loading {csv_path} (this may take a moment)…")
    df = pd.read_csv(csv_path)

    # Some exports store labels like '[3]'; strip brackets and cast.
    # Plain numeric strings are also handled correctly.
    for col in ("valence", "arousal"):
        df[col] = df[col].astype(str).str.strip("[]").astype(np.float32)

    subjects = sorted(df["subject_id"].unique())
    trials   = sorted(df["trial_id"].unique())

    all_eeg, all_labels = [], []

    for subj_idx, subj_id in enumerate(subjects):
        subj_df  = df[df["subject_id"] == subj_id]
        subj_eeg    = []
        subj_labels = []

        for trial_id in trials:
            trial_df = subj_df[subj_df["trial_id"] == trial_id]

            eeg_raw = trial_df[DREAMER_EEG_COLS].values.astype(np.float32)  # (n_samples, 14)
            eeg     = _extract_centre(eeg_raw, DREAMER_TRIAL_SAMPLES).T      # (14, 7680)

            # Label is constant within a trial — take the first row
            valence = trial_df["valence"].iloc[0]
            arousal = trial_df["arousal"].iloc[0]

            subj_eeg.append(eeg)
            subj_labels.append([valence, arousal])

        subj_eeg_arr    = np.stack(subj_eeg,    axis=0)  # (18, 14, 7680)
        subj_labels_arr = np.array(subj_labels, dtype=np.float32)  # (18, 2)

        all_eeg.append(subj_eeg_arr)
        all_labels.append(subj_labels_arr)
        print(f"  DREAMER subject {subj_idx+1:02d}/{len(subjects)}  "
              f"eeg={subj_eeg_arr.shape}  labels={subj_labels_arr.shape}")

    eeg_arr    = np.stack(all_eeg,    axis=0)  # (n_subjects, 18, 14, 7680)
    labels_arr = np.stack(all_labels, axis=0)  # (n_subjects, 18, 2)

    eeg_path    = os.path.join(output_dir, "dreamer_eeg.npy")
    labels_path = os.path.join(output_dir, "dreamer_labels.npy")
    np.save(eeg_path,    eeg_arr)
    np.save(labels_path, labels_arr)

    print(f"\nDREAMER saved:")
    print(f"  {eeg_path}    {eeg_arr.shape}  {eeg_arr.dtype}")
    print(f"  {labels_path} {labels_arr.shape}  {labels_arr.dtype}")
    _print_label_stats("DREAMER", labels_arr)


def prepare_eegemotions(input_dir: str, output_dir: str, label_mode: str = "emotion_27") -> None:
    """Convert eegemotions_labeled.csv to a single pair of .npy arrays."""
    input_path = Path(input_dir)
    csv_path = input_path if input_path.is_file() and input_path.suffix.lower() == ".csv" else input_path / "eegemotions_labeled.csv"
    cowen_path = csv_path.with_name("cowen_27_valence_arousal.csv")

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"eegemotions_labeled.csv not found in {input_dir}.\n"
            f"Make sure --eegemotions_dir (or --input_dir) points to the folder containing eegemotions_labeled.csv."
        )
    if not cowen_path.is_file():
        raise FileNotFoundError(
            f"cowen_27_valence_arousal.csv not found next to {csv_path}.\n"
            "The EEGEmotions converter needs both files in the same folder."
        )

    emotion_names, _ = load_cowen_27_mapping(str(cowen_path))
    print(f"  Loading {csv_path} (this may take a moment)…")
    eeg_arr, labels_arr = load_eegemotions_joined_csv(str(csv_path), label_mode=label_mode)

    eeg_path = os.path.join(output_dir, "eegemotions_eeg.npy")
    labels_path = os.path.join(output_dir, "eegemotions_labels.npy")
    np.save(eeg_path, eeg_arr)
    np.save(labels_path, labels_arr)

    print(f"\nEEGEmotions saved ({label_mode}):")
    print(f"  {eeg_path}    {eeg_arr.shape}  {eeg_arr.dtype}")
    print(f"  {labels_path} {labels_arr.shape}  {labels_arr.dtype}")
    _print_label_stats(
        "EEGEMOTIONS",
        labels_arr,
        dim_names=emotion_names if label_mode.lower() == "emotion_27" else ["valence", "arousal"],
    )


# ---------------------------------------------------------------------------
# AMIGOS  (joined CSV version, with legacy .mat fallback)
# ---------------------------------------------------------------------------

def load_amigos_subject(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one AMIGOS subject .mat file.

    Parameters
    ----------
    filepath : str — path to Data_Preprocessed_P{XX}.mat

    Returns
    -------
    eeg    : float32 ndarray, shape (AMIGOS_N_TRIALS, AMIGOS_N_CHANNELS, AMIGOS_TRIAL_SAMPLES)
    labels : float32 ndarray, shape (AMIGOS_N_TRIALS, 2)  [valence, arousal]

    Raises
    ------
    ValueError
        If one of the first ``AMIGOS_N_TRIALS`` trials is missing its EEG
        signal or its self-assessment label (this shouldn't happen for any
        subject not already in ``AMIGOS_SKIPPED_SUBJECTS``, since the
        missing-data issue in the public release only affects the 4
        long-video trials this converter doesn't use).
    """
    try:
        import scipy.io
    except ImportError:
        raise ImportError("scipy is required to read AMIGOS .mat files: pip install scipy")

    mat = scipy.io.loadmat(filepath, verify_compressed_data_integrity=False)

    # 'joined_data': object array of shape (1, n_trials); each entry is
    # (n_timesteps, 17) -- first 14 columns EEG, remaining 3 peripheral
    # (ECG x2, GSR x1), which we drop.
    trial_signals = mat["joined_data"][0]
    # 'labels_selfassessment': object array of shape (1, n_trials); each
    # entry is (1, 12) = [arousal, valence, dominance, liking, familiarity,
    # neutral, disgust, happiness, surprise, anger, fear, sadness].
    trial_ratings = mat["labels_selfassessment"][0]

    n_trials_available = min(len(trial_signals), AMIGOS_N_TRIALS)
    if n_trials_available < AMIGOS_N_TRIALS:
        raise ValueError(
            f"{filepath}: expected at least {AMIGOS_N_TRIALS} trials in "
            f"'joined_data', found {len(trial_signals)}."
        )

    arousal_idx = AMIGOS_LABEL_NAMES.index("arousal")
    valence_idx = AMIGOS_LABEL_NAMES.index("valence")

    eeg_trials, label_trials = [], []
    for trial_idx in range(AMIGOS_N_TRIALS):
        signal = trial_signals[trial_idx]
        rating = trial_ratings[trial_idx]

        if signal.size == 0 or rating.size == 0:
            raise ValueError(
                f"{filepath}: trial {trial_idx} is missing EEG or label "
                "data. This converter only uses the first AMIGOS_N_TRIALS "
                "short-video trials, which should be present for every "
                "subject not in AMIGOS_SKIPPED_SUBJECTS -- double check "
                "that this subject really belongs in the preprocessed "
                "release you downloaded."
            )

        eeg_only = signal[:, :AMIGOS_N_CHANNELS].astype(np.float32)  # (n_timesteps, 14)
        eeg = _extract_centre(eeg_only, AMIGOS_TRIAL_SAMPLES).T       # (14, 7680)

        rating_values = np.asarray(rating, dtype=np.float32).reshape(-1)  # (12,)
        valence = rating_values[valence_idx]
        arousal = rating_values[arousal_idx]

        eeg_trials.append(eeg)
        label_trials.append([valence, arousal])

    eeg_arr    = np.stack(eeg_trials, axis=0)                    # (16, 14, 7680)
    labels_arr = np.array(label_trials, dtype=np.float32)        # (16, 2)
    return eeg_arr, labels_arr


def prepare_amigos(input_dir: str, output_dir: str) -> None:
    """Convert AMIGOS inputs to a single pair of .npy arrays.

    ``input_dir`` can point to one of three things:
    - a directory containing ``amigos_joined.csv``;
    - the ``amigos_joined.csv`` file itself;
    - a legacy ``data_preprocessed`` folder containing ``Data_Preprocessed_P*.mat``.

    The joined CSV path is preferred because it matches the current AMIGOS
    source file used by the joint-v2 pipeline.
    """
    input_path = Path(input_dir)
    csv_path = input_path if input_path.is_file() and input_path.suffix.lower() == ".csv" else input_path / "amigos_joined.csv"
    legacy_input_dir = input_path.parent if input_path.is_file() and input_path.suffix.lower() == ".mat" else input_path

    used_csv = False
    if csv_path.is_file():
        eeg_arr, labels_arr = load_amigos_joined_csv(csv_path)
        used_csv = True
        print(f"  AMIGOS joined CSV loaded from {csv_path}")
        print(f"  AMIGOS CSV output: eeg={eeg_arr.shape}  labels={labels_arr.shape}")
    else:
        all_eeg, all_labels, kept_subjects = [], [], []

        for subj_idx in range(1, AMIGOS_N_SUBJECTS_TOTAL + 1):
            if subj_idx in AMIGOS_SKIPPED_SUBJECTS:
                print(f"  AMIGOS subject {subj_idx:02d}/{AMIGOS_N_SUBJECTS_TOTAL}  "
                      f"skipped (known invalid data in the preprocessed release)")
                continue

            filename = legacy_input_dir / f"Data_Preprocessed_P{subj_idx:02d}.mat"
            if not filename.is_file():
                raise FileNotFoundError(
                    f"Expected AMIGOS file not found: {filename}\n"
                    f"Make sure --amigos_dir points to either the folder containing "
                    f"amigos_joined.csv or the legacy unzipped 'data_preprocessed' "
                    f"folder containing Data_Preprocessed_P*.mat."
                )

            eeg, labels = load_amigos_subject(str(filename))
            all_eeg.append(eeg)
            all_labels.append(labels)
            kept_subjects.append(subj_idx)
            print(f"  AMIGOS subject {subj_idx:02d}/{AMIGOS_N_SUBJECTS_TOTAL}  "
                  f"eeg={eeg.shape}  labels={labels.shape}")

        eeg_arr    = np.stack(all_eeg,    axis=0)  # (33, 16, 14, 7680)
        labels_arr = np.stack(all_labels, axis=0)  # (33, 16, 2)

    eeg_path    = os.path.join(output_dir, "amigos_eeg.npy")
    labels_path = os.path.join(output_dir, "amigos_labels.npy")
    np.save(eeg_path,    eeg_arr)
    np.save(labels_path, labels_arr)

    if used_csv:
        print("\nAMIGOS saved from joined CSV:")
    else:
        print(f"\nAMIGOS saved ({len(kept_subjects)}/{AMIGOS_N_SUBJECTS_TOTAL} subjects kept; "
              f"skipped {AMIGOS_SKIPPED_SUBJECTS}):")
    print(f"  {eeg_path}    {eeg_arr.shape}  {eeg_arr.dtype}")
    print(f"  {labels_path} {labels_arr.shape}  {labels_arr.dtype}")
    _print_label_stats("AMIGOS", labels_arr)


# ---------------------------------------------------------------------------
# Sanity-check helper
# ---------------------------------------------------------------------------

def _print_label_stats(name: str, labels: np.ndarray, dim_names: list[str] | None = None) -> None:
    """Print basic label statistics to help verify the conversion."""
    default_dim_names = {
        "DEAP":    ["valence", "arousal", "dominance", "liking"],
        "DREAMER": ["valence", "arousal"],  # Converter currently outputs only these two labels
        "AMIGOS":  ["valence", "arousal"],  # Converter currently outputs only these two labels
    }
    names = dim_names if dim_names is not None else default_dim_names.get(name, [f"dim{i}" for i in range(labels.shape[-1])])
    print(f"\n{name} label statistics (across all subjects × trials):")
    flat = labels.reshape(-1, labels.shape[-1])
    if len(names) == labels.shape[-1] and labels.shape[-1] > 2 and np.all((flat == 0.0) | (flat == 1.0)):
        counts = flat.sum(axis=0)
        for dim, count in zip(names, counts):
            print(f"  {dim:12s}  count={int(count)}")
    else:
        for i, dim in enumerate(names):
            col = flat[:, i]
            print(f"  {dim:12s}  min={col.min():.1f}  max={col.max():.1f}  "
                  f"mean={col.mean():.2f}  median={np.median(col):.1f}")


def verify_npy(output_dir: str, dataset: str, label_mode: str | None = None) -> None:
    """Reload and verify the saved .npy files."""
    eeg_path = os.path.join(output_dir, f"{dataset}_eeg.npy")
    labels_path = os.path.join(output_dir, f"{dataset}_labels.npy")

    eeg = np.load(eeg_path)
    labels = np.load(labels_path)

    assert not np.any(np.isnan(eeg)), "NaN values found in EEG data!"
    assert not np.any(np.isinf(eeg)), "Inf values found in EEG data!"
    assert not np.any(np.isnan(labels)), "NaN values found in label data!"
    assert not np.any(np.isinf(labels)), "Inf values found in label data!"
    print(f"  ✓  No NaN/Inf in EEG and label data")

    if dataset == "eegemotions":
        mode = (label_mode or "emotion_27").lower()
        expected_label_dim = EEGEMOTIONS_N_TRIALS if mode == "emotion_27" else 2
        ok = True

        if eeg.shape[1:] == (EEGEMOTIONS_N_TRIALS, len(DREAMER_EEG_COLS), EEGEMOTIONS_TRIAL_SAMPLES):
            print(f"  ✓  {dataset}_eeg.npy  shape={eeg.shape}")
        else:
            print(
                f"  ✗  {dataset}_eeg.npy  shape={eeg.shape}  "
                f"(expected trailing shape {(EEGEMOTIONS_N_TRIALS, len(DREAMER_EEG_COLS), EEGEMOTIONS_TRIAL_SAMPLES)})"
            )
            ok = False

        if labels.shape[1:] == (EEGEMOTIONS_N_TRIALS, expected_label_dim):
            print(f"  ✓  {dataset}_labels.npy  shape={labels.shape}")
        else:
            print(
                f"  ✗  {dataset}_labels.npy  shape={labels.shape}  "
                f"(expected trailing shape {(EEGEMOTIONS_N_TRIALS, expected_label_dim)})"
            )
            ok = False

        if mode == "emotion_27":
            sums = labels.sum(axis=-1)
            if np.allclose(sums, 1.0):
                print("  ✓  One-hot label rows sum to 1")
            else:
                print("  ✗  One-hot label rows do not sum to 1")
                ok = False

        if ok:
            print(f"\n  All checks passed for {dataset.upper()}.")
        else:
            print(f"\n  Shape mismatch detected — check the raw data layout.")
        return

    expected_shapes = {
        "deap":    {"eeg": (32, 40, 32, 7680), "labels": (32, 40, 4)},
        "dreamer": {"eeg": (23, 18, 14, 7680), "labels": (23, 18, 2)},  # labels: valence, arousal
        "amigos":  [
            {"eeg": (15, AMIGOS_SHORT_TRIALS, AMIGOS_N_CHANNELS, AMIGOS_TRIAL_SAMPLES), "labels": (15, AMIGOS_SHORT_TRIALS, 2)},
            {"eeg": (AMIGOS_N_SUBJECTS, AMIGOS_N_TRIALS, AMIGOS_N_CHANNELS, AMIGOS_TRIAL_SAMPLES),
             "labels": (AMIGOS_N_SUBJECTS, AMIGOS_N_TRIALS, 2)},
        ],  # labels: valence, arousal
    }
    exp = expected_shapes[dataset]

    ok = True
    if dataset == "amigos":
        allowed_shapes = exp
        matched = False
        for shape_set in allowed_shapes:
            if eeg.shape == shape_set["eeg"] and labels.shape == shape_set["labels"]:
                matched = True
                break
        if matched:
            print(f"  ✓  {dataset}_eeg.npy  shape={eeg.shape}")
            print(f"  ✓  {dataset}_labels.npy  shape={labels.shape}")
        else:
            expected_text = ", ".join(
                f"eeg={shape_set['eeg']}, labels={shape_set['labels']}" for shape_set in allowed_shapes
            )
            print(f"  ✗  {dataset}_eeg.npy  shape={eeg.shape}  (expected one of {expected_text})")
            print(f"  ✗  {dataset}_labels.npy  shape={labels.shape}  (expected one of {expected_text})")
            ok = False
    else:
        for key, arr, exp_shape in [("eeg", eeg, exp["eeg"]),
                                      ("labels", labels, exp["labels"])]:
            if arr.shape == exp_shape:
                print(f"  ✓  {dataset}_{key}.npy  shape={arr.shape}")
            else:
                print(f"  ✗  {dataset}_{key}.npy  shape={arr.shape}  (expected {exp_shape})")
                ok = False

    if ok:
        print(f"\n  All checks passed for {dataset.upper()}.")
    else:
        print(f"\n  Shape mismatch detected — check the raw data layout.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert DEAP / DREAMER / AMIGOS / EEGEmotions raw files to STSNet-ready .npy arrays"
    )
    parser.add_argument(
        "--dataset", choices=["deap", "dreamer", "amigos", "eegemotions", "both", "all"], default="all",
        help="Which dataset to convert. 'both' is a deprecated alias for "
             "'all' kept for backwards compatibility. (default: all)",
    )
    parser.add_argument(
        "--deap_dir", type=str, default=None,
        help="Folder containing s01.dat … s32.dat (DEAP preprocessed Python version)",
    )
    parser.add_argument(
        "--dreamer_dir", type=str, default=None,
        help="Folder containing dreamer_joined.csv",
    )
    parser.add_argument(
        "--amigos_dir", type=str, default=None,
           help="Folder containing amigos_joined.csv, or the legacy unzipped "
               "'data_preprocessed' folder containing Data_Preprocessed_P01.mat … "
               "Data_Preprocessed_P40.mat",
    )
    parser.add_argument(
        "--eegemotions_dir", type=str, default=None,
        help="Folder containing eegemotions_labeled.csv and cowen_27_valence_arousal.csv",
    )
    parser.add_argument(
        "--eegemotions_label_mode", type=str, default="emotion_27",
        choices=["emotion_27", "valence_arousal"],
        help="How to encode EEGEmotions labels: one-hot 27-way emotion labels or Cowen valence/arousal pairs.",
    )
    # Shorthand when all datasets are in the same place
    parser.add_argument(
        "--input_dir", type=str, default=None,
        help="Shorthand: single folder for all requested datasets",
    )
    parser.add_argument(
        "--output_dir", type=str, default=".",
        help="Where to write the .npy files (default: current directory)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Reload and verify the output files after conversion",
    )
    args = parser.parse_args()

    # Resolve input directories
    deap_dir    = args.deap_dir    or args.input_dir
    dreamer_dir = args.dreamer_dir or args.input_dir
    amigos_dir  = args.amigos_dir  or args.input_dir
    eegemotions_dir = args.eegemotions_dir or args.input_dir

    # 'both' predates AMIGOS support; treat it the same as 'all'.
    requested = "all" if args.dataset == "both" else args.dataset

    os.makedirs(args.output_dir, exist_ok=True)

    if requested in ("deap", "all"):
        if deap_dir is None:
            parser.error("--deap_dir (or --input_dir) is required for DEAP")
        print(f"\n{'='*50}\nConverting DEAP\n{'='*50}")
        prepare_deap(deap_dir, args.output_dir)
        if args.verify:
            print("\nVerifying DEAP output…")
            verify_npy(args.output_dir, "deap")

    if requested in ("dreamer", "all"):
        if dreamer_dir is None:
            parser.error("--dreamer_dir (or --input_dir) is required for DREAMER")
        print(f"\n{'='*50}\nConverting DREAMER\n{'='*50}")
        prepare_dreamer(dreamer_dir, args.output_dir)
        if args.verify:
            print("\nVerifying DREAMER output…")
            verify_npy(args.output_dir, "dreamer")

    if requested in ("amigos", "all"):
        if amigos_dir is None:
            parser.error("--amigos_dir (or --input_dir) is required for AMIGOS")
        print(f"\n{'='*50}\nConverting AMIGOS\n{'='*50}")
        prepare_amigos(amigos_dir, args.output_dir)
        if args.verify:
            print("\nVerifying AMIGOS output…")
            verify_npy(args.output_dir, "amigos")

    if requested == "eegemotions":
        if eegemotions_dir is None:
            parser.error("--eegemotions_dir (or --input_dir) is required for EEGEmotions")
        print(f"\n{'='*50}\nConverting EEGEmotions\n{'='*50}")
        prepare_eegemotions(eegemotions_dir, args.output_dir, label_mode=args.eegemotions_label_mode)
        if args.verify:
            print("\nVerifying EEGEmotions output…")
            verify_npy(args.output_dir, "eegemotions", label_mode=args.eegemotions_label_mode)