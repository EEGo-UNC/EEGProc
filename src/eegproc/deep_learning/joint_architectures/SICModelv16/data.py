"""Prepare raw or prefiltered DREAMER arrays without changing the source files."""
from pathlib import Path
import numpy as np


def prepare_data(eeg_path, labels_path, cache, *, expected_subjects=23):
    # Import lazily: planning and tracker operations need no scientific runtime.
    from .run_experiments import _shared
    eeg_path, cache = Path(eeg_path), Path(cache)
    eeg = np.load(eeg_path, mmap_mode="r", allow_pickle=False)
    if eeg.ndim != 4 or eeg.shape[2] not in (14, 42) or eeg.shape[-1] < 128:
        raise ValueError(f"Expected (subjects,trials,14 or 42,samples>=128); got {eeg.shape}.")
    cache.mkdir(parents=True, exist_ok=True)
    metadata = {"input_shape": list(eeg.shape), "sampling_rate_hz": 128,
                "input_representation": "raw_14_channel" if eeg.shape[2] == 14 else "band_stacked_42_feature",
                "output_order": "channel-major, band-minor: theta, alpha, beta",
                "normalization": "global RMS per 128-sample window", "status": "preparing"}
    _shared.write_json(cache / "preprocessing.json", metadata)
    temporary = cache / "filtered_trials.tmp.npy"
    try:
        if eeg.shape[2] == 14:
            from ...prepare_datasets import _filter_dreamer_trial
            converted = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                shape=(eeg.shape[0], eeg.shape[1], 42, eeg.shape[-1]))
            try:
                for subject in range(eeg.shape[0]):
                    for trial in range(eeg.shape[1]):
                        raw = np.asarray(eeg[subject, trial].T, dtype=np.float32)
                        if not np.isfinite(raw).all():
                            raise ValueError(f"Nonfinite EEG at subject={subject}, trial={trial}.")
                        # Filter complete trials BEFORE windowing, using EEGProc's
                        # established notch/detrend/Butterworth implementation.
                        filtered = _filter_dreamer_trial(raw)
                        if filtered.shape != (eeg.shape[-1], 42) or not np.isfinite(filtered).all():
                            raise ValueError(f"Invalid filtered EEG at subject={subject}, trial={trial}.")
                        converted[subject, trial] = filtered.T
                    print(f"Filtered subject {subject + 1}/{eeg.shape[0]} into separate bands", flush=True)
                converted.flush()
            finally:
                del converted
            metadata["filter"] = "EEGProc _filter_dreamer_trial: theta 4-8, alpha 8-13, beta 13-30 Hz; order 4; notch 50 Hz; detrend; no rereference"
            available = _shared.prepare_data(temporary, labels_path, cache, expected_subjects=expected_subjects)
        else:
            available = _shared.prepare_data(eeg_path, labels_path, cache, expected_subjects=expected_subjects)
        metadata["status"] = "completed"
        _shared.write_json(cache / "preprocessing.json", metadata)
        return available
    except Exception as exc:
        metadata.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        _shared.write_json(cache / "preprocessing.json", metadata)
        raise
    finally:
        if temporary.exists():
            temporary.unlink()
