"""
prepare_datasets.py
===================
Convert raw DEAP and DREAMER dataset files into the NumPy format
expected by STSNet's train_eval.py:

    {dataset}_eeg.npy    — float32, shape (n_subjects, n_trials, n_channels, n_samples)
    {dataset}_labels.npy — float32, shape (n_subjects, n_trials, n_label_dims)

Usage
-----
    # DEAP  (point to the folder containing s01.dat ... s32.dat)
    python prepare_datasets.py --dataset deap --input_dir /path/to/deap/data_preprocessed_python

    # DREAMER  (point to the folder containing DREAMER_FULL.csv)
    python prepare_datasets.py --dataset dreamer --input_dir /path/to/dreamer

    # Both
    python prepare_datasets.py --dataset both \
        --deap_dir /path/to/deap/data_preprocessed_python \
        --dreamer_dir /path/to/dreamer

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

DREAMER (DREAMER_FULL.csv):
    Long/tidy format — one row per EEG sample with columns:
        patient_index, video_index,
        arousal  (stored as '[value]'),
        valence  (stored as '[value]'),
        AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4  (14 channels)
    Signals are already at 128 Hz and filtered 4–30 Hz.
    Trial lengths vary; we take 60 s from the middle (7680 samples).
    NOTE: the CSV contains no dominance score column, so the labels array
    has shape (n_subjects, 18, 2) with [valence, arousal] only.
"""

import argparse
import os
import pickle
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


def _extract_centre(arr: np.ndarray, target: int) -> np.ndarray:
    """Return `target` samples from the centre of `arr` (n_samples, n_ch),
    repeat-padding if the trial is shorter than `target`."""
    n = arr.shape[0]
    if n >= target:
        mid   = n // 2
        start = mid - target // 2
        return arr[start: start + target, :]
    repeats = (target // n) + 1
    return np.tile(arr, (repeats, 1))[:target, :]


def prepare_dreamer(input_dir: str, output_dir: str) -> None:
    """Convert DREAMER_FULL.csv to a single pair of .npy arrays.

    Parameters
    ----------
    input_dir  : str — folder containing DREAMER_FULL.csv
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

    csv_path = os.path.join(input_dir, "DREAMER_FULL.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"DREAMER_FULL.csv not found in {input_dir}.\n"
            f"Make sure --dreamer_dir points to the folder containing DREAMER_FULL.csv."
        )

    print(f"  Loading {csv_path} (this may take a moment)…")
    df = pd.read_csv(csv_path)

    # Labels are stored as '[3]' strings — strip brackets and cast
    for col in ("valence", "arousal"):
        df[col] = df[col].astype(str).str.strip("[]").astype(np.float32)

    subjects = sorted(df["patient_index"].unique())
    trials   = sorted(df["video_index"].unique())

    all_eeg, all_labels = [], []

    for subj_idx, subj_id in enumerate(subjects):
        subj_df  = df[df["patient_index"] == subj_id]
        subj_eeg    = []
        subj_labels = []

        for trial_id in trials:
            trial_df = subj_df[subj_df["video_index"] == trial_id]

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


# ---------------------------------------------------------------------------
# Sanity-check helper
# ---------------------------------------------------------------------------

def _print_label_stats(name: str, labels: np.ndarray) -> None:
    """Print basic label statistics to help verify the conversion."""
    dim_names = {
        "DEAP":    ["valence", "arousal", "dominance", "liking"],
        "DREAMER": ["valence", "arousal"],  # CSV has no dominance column
    }
    names = dim_names.get(name, [f"dim{i}" for i in range(labels.shape[-1])])
    print(f"\n{name} label statistics (across all subjects × trials):")
    flat = labels.reshape(-1, labels.shape[-1])
    for i, dim in enumerate(names):
        col = flat[:, i]
        print(f"  {dim:12s}  min={col.min():.1f}  max={col.max():.1f}  "
              f"mean={col.mean():.2f}  median={np.median(col):.1f}")


def verify_npy(output_dir: str, dataset: str) -> None:
    """Reload and verify the saved .npy files."""
    eeg_path    = os.path.join(output_dir, f"{dataset}_eeg.npy")
    labels_path = os.path.join(output_dir, f"{dataset}_labels.npy")

    eeg    = np.load(eeg_path)
    labels = np.load(labels_path)

    expected_shapes = {
        "deap":    {"eeg": (32, 40, 32, 7680), "labels": (32, 40, 4)},
        "dreamer": {"eeg": (23, 18, 14, 7680), "labels": (23, 18, 2)},  # labels: valence, arousal
    }
    exp = expected_shapes[dataset]

    ok = True
    for key, arr, exp_shape in [("eeg", eeg, exp["eeg"]),
                                  ("labels", labels, exp["labels"])]:
        if arr.shape == exp_shape:
            print(f"  ✓  {dataset}_{key}.npy  shape={arr.shape}")
        else:
            print(f"  ✗  {dataset}_{key}.npy  shape={arr.shape}  (expected {exp_shape})")
            ok = False

    assert not np.any(np.isnan(eeg)),    "NaN values found in EEG data!"
    assert not np.any(np.isinf(eeg)),    "Inf values found in EEG data!"
    print(f"  ✓  No NaN/Inf in EEG data")

    if ok:
        print(f"\n  All checks passed for {dataset.upper()}.")
    else:
        print(f"\n  Shape mismatch detected — check the raw data layout.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert DEAP / DREAMER raw files to STSNet-ready .npy arrays"
    )
    parser.add_argument(
        "--dataset", choices=["deap", "dreamer", "both"], default="both",
        help="Which dataset to convert (default: both)",
    )
    parser.add_argument(
        "--deap_dir", type=str, default=None,
        help="Folder containing s01.dat … s32.dat (DEAP preprocessed Python version)",
    )
    parser.add_argument(
        "--dreamer_dir", type=str, default=None,
        help="Folder containing DREAMER_FULL.csv",
    )
    # Shorthand when both are in the same place
    parser.add_argument(
        "--input_dir", type=str, default=None,
        help="Shorthand: single folder for both datasets",
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

    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset in ("deap", "both"):
        if deap_dir is None:
            parser.error("--deap_dir (or --input_dir) is required for DEAP")
        print(f"\n{'='*50}\nConverting DEAP\n{'='*50}")
        prepare_deap(deap_dir, args.output_dir)
        if args.verify:
            print("\nVerifying DEAP output…")
            verify_npy(args.output_dir, "deap")

    if args.dataset in ("dreamer", "both"):
        if dreamer_dir is None:
            parser.error("--dreamer_dir (or --input_dir) is required for DREAMER")
        print(f"\n{'='*50}\nConverting DREAMER\n{'='*50}")
        prepare_dreamer(dreamer_dir, args.output_dir)
        if args.verify:
            print("\nVerifying DREAMER output…")
            verify_npy(args.output_dir, "dreamer")
