"""Preprocess DREAMER trials into training-ready, cached samples for MTLFuseNet.

For each (subject, trial) this builds the two branch inputs the model needs and
caches them to disk so the LOSO loop (which trains 23 separate models) never has
to re-run the expensive bandpass / differential-entropy / mutual-information
pipeline more than once.

Per-trial cached sample (all float32 unless noted):
    X_ST : (num_win, 9, 9, 128)  spatio-temporal grid windows, min-max normalized to [0, 1]
    DE   : (num_win, 3, 14)      differential-entropy features, [theta, alpha, beta] x EEG_CHANNELS
    adj  : (3, 14, 14)           symmetric-normalized mutual-information adjacency per band
    valence, arousal : int       binary label (>= median -> 1)
    subject_id, trial_id : int
    num_win : int

The DE window count and the grid window count are made to line up (both use
non-overlapping 1-second windows, per the fix noted in the notebook), then
truncated to the shared minimum as a defensive guard.
"""

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .... import bandpass_filter, psd_bandpowers
    from .preprocessing import (
        EEG_CHANNELS,
        DREAMER_BANDS,
        ELECTRODE_POSITIONS,
        build_adjacency_matrix,
        label_to_binary,
        load_dreamer_csv,
    )
    from .models import normalize_adjacency
except ImportError:
    try:
        from eegproc import bandpass_filter, psd_bandpowers
        from eegproc.deep_learning.supervised.mtlfusenet.preprocessing import (
            EEG_CHANNELS,
            DREAMER_BANDS,
            ELECTRODE_POSITIONS,
            build_adjacency_matrix,
            label_to_binary,
            load_dreamer_csv,
        )
        from eegproc.deep_learning.supervised.mtlfusenet.models import normalize_adjacency
    except ImportError:
        CURRENT_DIR = Path(__file__).resolve().parent
        if str(CURRENT_DIR) not in sys.path:
            sys.path.insert(0, str(CURRENT_DIR))
        from eegproc import bandpass_filter, psd_bandpowers
        from preprocessing import (
            EEG_CHANNELS,
            DREAMER_BANDS,
            ELECTRODE_POSITIONS,
            build_adjacency_matrix,
            label_to_binary,
            load_dreamer_csv,
        )
        from models import normalize_adjacency

BANDS = ("theta", "alpha", "beta")
FS = 128
WINDOW_SAMPLES = 128  # 1 second at 128 Hz


def eeg_trial_to_spatial_tensor(eeg_df, positions=ELECTRODE_POSITIONS, grid_size=9):
    """Map a trial's raw EEG (one row per timestep) into a (L, 9, 9) tensor.

    Vectorized replacement for the notebook's per-row ``iloc`` loop, which was
    unusably slow on full-length trials.
    """
    values = eeg_df[EEG_CHANNELS].to_numpy(dtype=np.float32)  # (L, 14)
    L = values.shape[0]
    tensor = np.zeros((L, grid_size, grid_size), dtype=np.float32)
    for ci, ch in enumerate(EEG_CHANNELS):
        r, c = positions[ch]
        tensor[:, r, c] = values[:, ci]
    return tensor


def window_spatial_tensor(X_ST, window_samples=WINDOW_SAMPLES, step_samples=None):
    """Chunk (L, 9, 9) into non-overlapping windows -> (num_win, 9, 9, window_samples)."""
    if step_samples is None:
        step_samples = window_samples
    L = X_ST.shape[0]
    windows = [
        np.transpose(X_ST[s : s + window_samples], (1, 2, 0))
        for s in range(0, L - window_samples + 1, step_samples)
    ]
    if not windows:
        return np.empty((0, 9, 9, window_samples), dtype=np.float32)
    return np.stack(windows).astype(np.float32)


def de_to_band_channel_array(de_features):
    """Reshape the DE dataframe to (num_win, 3 bands, 14 channels).

    Columns are named ``{channel}_{band}``; we select each band's columns in
    EEG_CHANNELS order so DE rows align with the adjacency matrices.
    """
    num_win = len(de_features)
    out = np.zeros((num_win, len(BANDS), len(EEG_CHANNELS)), dtype=np.float32)
    for bi, band in enumerate(BANDS):
        cols = [f"{ch}_{band}" for ch in EEG_CHANNELS]
        out[:, bi, :] = de_features[cols].to_numpy(dtype=np.float32)
    return out


def build_normalized_adjacencies(filtered, mi_max_samples=5000, seed=0):
    """One symmetric-normalized MI adjacency (14, 14) per band -> (3, 14, 14).

    ``mi_max_samples`` uniformly subsamples timesteps before the (slow) KNN-based
    mutual-information estimate. MI on a few thousand samples is statistically
    stable and keeps preprocessing tractable; set to None to use every sample.
    """
    adj = np.zeros((len(BANDS), len(EEG_CHANNELS), len(EEG_CHANNELS)), dtype=np.float32)
    for bi, band in enumerate(BANDS):
        band_cols = [f"{ch}_{band}" for ch in EEG_CHANNELS]
        band_df = filtered[band_cols]
        if mi_max_samples is not None and len(band_df) > mi_max_samples:
            idx = np.linspace(0, len(band_df) - 1, mi_max_samples).astype(int)
            band_df = band_df.iloc[idx]
        raw = build_adjacency_matrix(band_df, band_cols)
        adj[bi] = normalize_adjacency(raw).astype(np.float32)
    return adj


def build_trial_sample(data, subject_id, trial_id, mi_max_samples=5000, de_eps=1e-12):
    """Build one training-ready cached sample for a (subject, trial)."""
    mask = (data["subject_id"] == subject_id) & (data["trial_id"] == trial_id)
    trial_df = data.loc[mask]
    eeg_df = trial_df[EEG_CHANNELS]

    # --- spatio-spectral branch: filter -> DE features -> adjacency ---
    filtered = bandpass_filter(eeg_df, fs=FS, bands=DREAMER_BANDS)
    band_power = psd_bandpowers(filtered, fs=FS, bands=DREAMER_BANDS, window_sec=1.0, overlap=0.0)
    # guard log(0): psd_bandpowers emits 0.0 where a band has no bins
    de_features = 0.5 * np.log(2 * np.pi * np.e * (band_power + de_eps))
    adj = build_normalized_adjacencies(filtered, mi_max_samples=mi_max_samples)

    # --- spatio-temporal branch: raw EEG -> 9x9 grid -> 1s windows ---
    X_ST = eeg_trial_to_spatial_tensor(eeg_df)
    X_ST_windowed = window_spatial_tensor(X_ST)

    # align the two branches' window counts (both non-overlapping 1s windows)
    n = min(len(de_features), len(X_ST_windowed))
    if n == 0:
        raise ValueError(
            f"trial too short: DE={len(de_features)} X_ST={len(X_ST_windowed)}"
        )
    DE = de_to_band_channel_array(de_features.iloc[:n])
    X_ST_windowed = X_ST_windowed[:n]

    if not np.isfinite(DE).all():
        raise ValueError("non-finite DE features")

    # per-trial min-max normalize grid windows to [0, 1] (decoder uses sigmoid)
    x_min, x_max = float(X_ST_windowed.min()), float(X_ST_windowed.max())
    denom = (x_max - x_min) or 1.0
    X_ST_norm = ((X_ST_windowed - x_min) / denom).astype(np.float32)

    return {
        "X_ST": X_ST_norm,
        "DE": DE,
        "adj": adj,
        "valence": int(label_to_binary(trial_df["valence"].iloc[0])),
        "arousal": int(label_to_binary(trial_df["arousal"].iloc[0])),
        "subject_id": int(subject_id),
        "trial_id": int(trial_id),
        "num_win": int(n),
    }


def preprocess_all(csv_path, out_dir="processed_trials", subjects=None,
                   mi_max_samples=5000, verbose=True):
    """Preprocess every (subject, trial) and cache each to ``out_dir/subjX_trialY.pkl``.

    Returns (manifest, errors). ``manifest`` is a list of dicts with the file
    path, ids, labels and num_win for each cached trial.
    """
    os.makedirs(out_dir, exist_ok=True)
    data = load_dreamer_csv(csv_path)
    pairs = data[["subject_id", "trial_id"]].drop_duplicates().values
    if subjects is not None:
        subjects = set(subjects)
        pairs = [(s, t) for s, t in pairs if s in subjects]

    manifest, errors = [], []
    for i, (subj, trial) in enumerate(pairs):
        try:
            sample = build_trial_sample(data, subj, trial, mi_max_samples=mi_max_samples)
            fname = os.path.join(out_dir, f"subj{subj}_trial{trial}.pkl")
            with open(fname, "wb") as f:
                pickle.dump(sample, f)
            manifest.append({
                "path": fname, "subject_id": int(subj), "trial_id": int(trial),
                "valence": sample["valence"], "arousal": sample["arousal"],
                "num_win": sample["num_win"],
            })
            if verbose:
                print(f"[{i + 1}/{len(pairs)}] subj{subj} trial{trial}: "
                      f"num_win={sample['num_win']}")
        except Exception as e:  # noqa: BLE001 - collect & continue, review after
            errors.append((int(subj), int(trial), str(e)))
            if verbose:
                print(f"[{i + 1}/{len(pairs)}] subj{subj} trial{trial}: ERROR {e}")

    with open(os.path.join(out_dir, "manifest.pkl"), "wb") as f:
        pickle.dump(manifest, f)
    if verbose:
        print(f"\nDone. cached={len(manifest)} errors={len(errors)}")
        for e in errors[:5]:
            print("  ", e)
    return manifest, errors


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Preprocess DREAMER for MTLFuseNet")
    ap.add_argument("--csv", default="datasets/dreamer_joined.csv")
    ap.add_argument("--out", default="processed_trials")
    ap.add_argument("--subjects", type=int, nargs="*", default=None,
                    help="subset of subject ids (default: all)")
    ap.add_argument("--mi-max-samples", type=int, default=5000)
    args = ap.parse_args()
    preprocess_all(args.csv, args.out, subjects=args.subjects,
                   mi_max_samples=args.mi_max_samples)
