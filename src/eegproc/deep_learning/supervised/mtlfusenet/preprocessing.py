import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression

try:
    from .... import bandpass_filter, psd_bandpowers
except ImportError:
    try:
        from eegproc import bandpass_filter, psd_bandpowers
    except ImportError:
        src_root = next(
            (parent for parent in Path(__file__).resolve().parents if parent.name == "src"),
            None,
        )
        if src_root is not None and str(src_root) not in sys.path:
            sys.path.insert(0, str(src_root))
        from eegproc import bandpass_filter, psd_bandpowers

EEG_CHANNELS = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2',
                'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

DREAMER_BANDS = {
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta': (13.0, 30.0)
}

ELECTRODE_POSITIONS = {
    'AF3': (1, 2), 'AF4': (1, 4),
    'F7':  (2, 0), 'F3':  (2, 2), 'F4':  (2, 6), 'F8':  (2, 8),
    'FC5': (3, 1), 'FC6': (3, 7),
    'T7':  (4, 0), 'T8':  (4, 8),
    'P7':  (6, 0), 'P8':  (6, 8),
    'O1':  (8, 2), 'O2':  (8, 4),
}

def load_dreamer_csv(path):
    """Load the DREAMER dataset CSV from a local path."""
    return pd.read_csv(path)


def get_trial_eeg(data, subject_id, trial_id, eeg_channels=EEG_CHANNELS):
    """Filter the full dataframe down to one subject/trial's raw EEG."""
    mask = (data['subject_id'] == subject_id) & (data['trial_id'] == trial_id)
    return data.loc[mask, eeg_channels]


def compute_de_features(eeg_df, fs=128, bands=DREAMER_BANDS, window_sec=1.0, overlap=0.0):
    """Bandpass filter + differential entropy per channel/band/window."""
    filtered = bandpass_filter(eeg_df, fs=fs, bands=bands)
    band_power = psd_bandpowers(filtered, fs=fs, bands=bands, window_sec=window_sec, overlap=overlap)
    de_features = 0.5 * np.log(2 * np.pi * np.e * band_power)
    return filtered, de_features


def build_adjacency_matrix(band_df, channels):
    """Mutual-information adjacency matrix for one frequency band."""
    n = len(channels)
    adj = np.zeros((n, n))
    for i in range(n):
        mi = mutual_info_regression(band_df[channels].values, band_df[channels[i]].values)
        adj[i, :] = mi
    return adj


def build_all_adjacency_matrices(filtered, bands=('theta', 'alpha', 'beta')):
    """Build one adjacency matrix per band, return as a dict."""
    matrices = {}
    for band in bands:
        band_channels = [c for c in filtered.columns if c.endswith(f'_{band}')]
        band_df = filtered[band_channels]
        matrices[band] = build_adjacency_matrix(band_df, band_channels)
    return matrices

def eeg_to_spatial_grid(row_values, positions=None, grid_size=9):
    """Map one timestep's channel readings into a 9x9 electrode grid."""
    if positions is None:
        positions = ELECTRODE_POSITIONS
    grid = np.zeros((grid_size, grid_size))
    for ch, (r, c) in positions.items():
        grid[r, c] = row_values[ch]
    return grid


def eeg_trial_to_spatial_tensor(eeg_df, positions=None, grid_size=9):
    """Map a full trial's raw EEG (one row per timestep) into a (L, 9, 9) tensor.
       Uses RAW EEG, not bandpass filtered, per the paper's spatio-temporal method."""
    if positions is None:
        positions = ELECTRODE_POSITIONS
    L = len(eeg_df)
    tensor = np.zeros((L, grid_size, grid_size))
    for t in range(L):
        row = eeg_df.iloc[t]
        for ch, (r, c) in positions.items():
            tensor[t, r, c] = row[ch]
    return tensor

def window_spatial_tensor(X_ST, window_samples=128, step_samples=None):
    """Chunk a (L, 9, 9) spatial tensor into non-overlapping windows,
       reshaped so time becomes the channel dimension: (num_windows, 9, 9, window_samples)."""
    if step_samples is None:
        step_samples = window_samples
    L = X_ST.shape[0]
    windows = []
    for start in range(0, L - window_samples + 1, step_samples):
        chunk = X_ST[start:start + window_samples]
        chunk = np.transpose(chunk, (1, 2, 0))
        windows.append(chunk)
    return np.stack(windows)

def process_trial(data, subject_id, trial_id, eeg_channels, dreamer_bands, electrode_positions):
    mask = (data['subject_id'] == subject_id) & (data['trial_id'] == trial_id)
    trial_df = data.loc[mask]
    eeg_df = trial_df[eeg_channels]

    filtered = bandpass_filter(eeg_df, fs=128, bands=dreamer_bands)
    band_power = psd_bandpowers(filtered, fs=128, bands=dreamer_bands, window_sec=1.0, overlap=0.0)
    de_features = 0.5 * np.log(2 * np.pi * np.e * band_power)

    X_ST = eeg_trial_to_spatial_tensor(eeg_df, electrode_positions)
    X_ST_windowed = window_spatial_tensor(X_ST, window_samples=128)

    valence_label = label_to_binary(trial_df['valence'].iloc[0])
    arousal_label = label_to_binary(trial_df['arousal'].iloc[0])

    return {
        'de_features': de_features,
        'X_ST_windowed': X_ST_windowed,
        'valence_label': valence_label,
        'arousal_label': arousal_label,
        'subject_id': subject_id,
        'trial_id': trial_id
    }


def label_to_binary(score, median=3):
    return 1 if score >= median else 0