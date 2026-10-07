import numpy as np

FS = 128

def make_sine(freq_hz: float, fs=FS, dur_sec=8.0, amp=1.0, phase=0.0):
    """Make 2pi*freq*t hz sine function"""
    n = int(dur_sec * fs)  # dur_sec gives duration of sine wave
    t = np.arange(n) / fs
    return amp * np.sin(2 * np.pi * freq_hz * t + phase)

def make_white_noise(fs=FS, dur_sec=12.0, amp=1.0, seed=0):
    rng = np.random.RandomState(seed)
    n = int(dur_sec * fs)
    return amp * rng.randn(n)

def expected_mobility(freq_hz: float, fs=FS):
    # mobility = ~2 * sin(pi * f / fs)
    return 2.0 * np.sin(np.pi * freq_hz / fs)

def window_rows(n_samples: int, fs=FS, window_sec=4.0, overlap=0.5):
    win = int(round(window_sec * fs))
    hop = int(round(win * (1.0 - overlap)))
    return 1 + (n_samples - win) // hop

def make_synthetic_trial_arrays(
    n_subjects: int = 3,
    n_trials: int = 4,
    n_channels: int = 4,
    n_samples: int = 1536,
    n_label_dims: int = 2,
    seed: int = 1234,
):
    """Deterministic synthetic EEG/label pair in the on-disk trial-major layout.

    Mirrors what ``prepare_datasets`` writes: channels-first, non-windowed, with
    labels on a 1-5 Likert scale so a median split at 3 produces both classes.

    Returns
    -------
    eeg : np.ndarray, shape (n_subjects, n_trials, n_channels, n_samples), float32
    labels : np.ndarray, shape (n_subjects, n_trials, n_label_dims), float32
    """
    rng = np.random.RandomState(seed)
    eeg = rng.randn(n_subjects, n_trials, n_channels, n_samples).astype(np.float32)
    labels = rng.uniform(1.0, 5.0, size=(n_subjects, n_trials, n_label_dims)).astype(
        np.float32
    )
    return eeg, labels
