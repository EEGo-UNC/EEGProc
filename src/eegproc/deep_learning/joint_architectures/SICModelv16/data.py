"""Prepare raw or prefiltered DREAMER arrays without changing the source files."""
from pathlib import Path
import numpy as np


def prepare_data(eeg_path, labels_path, cache, *, expected_subjects=23, data_subjects=None):
    # Import lazily: planning and tracker operations need no scientific runtime.
    from .run_experiments import _shared
    eeg_path, cache = Path(eeg_path), Path(cache)
    eeg = np.load(eeg_path, mmap_mode="r", allow_pickle=False)
    if eeg.ndim != 4 or eeg.shape[2] not in (14, 42) or eeg.shape[-1] < 128:
        raise ValueError(f"Expected (subjects,trials,14 or 42,samples>=128); got {eeg.shape}.")
    if eeg.shape[0] != expected_subjects:
        raise ValueError(f"Expected {expected_subjects} input subjects, got {eeg.shape[0]}.")
    selected = list(range(eeg.shape[0])) if data_subjects is None else list(data_subjects)
    if (not selected or len(set(selected)) != len(selected)
            or any(type(s) is not int or not 0 <= s < eeg.shape[0] for s in selected)):
        raise ValueError("data_subjects must contain distinct valid integer subject IDs.")
    ratings = np.load(labels_path, mmap_mode="r", allow_pickle=False)
    if ratings.ndim != 3 or ratings.shape[:2] != eeg.shape[:2]:
        raise ValueError("Ratings must align with the input EEG subject/trial axes.")
    cache.mkdir(parents=True, exist_ok=True)
    metadata = {"input_shape": list(eeg.shape), "data_subjects": selected, "sampling_rate_hz": 128,
                "input_representation": "raw_14_channel" if eeg.shape[2] == 14 else "band_stacked_42_feature",
                "output_order": "channel-major, band-minor: theta, alpha, beta",
                "normalization": "global RMS per 128-sample window", "status": "preparing"}
    _shared.write_json(cache / "preprocessing.json", metadata)
    temporary = cache / "filtered_trials.tmp.npy"
    temporary_labels = cache / "selected_labels.tmp.npy"
    try:
        if eeg.shape[2] == 14 or data_subjects is not None:
            if eeg.shape[2] == 14:
                from ...prepare_datasets import _filter_dreamer_trial
            converted = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                shape=(len(selected), eeg.shape[1], 42, eeg.shape[-1]))
            try:
                for row_index, subject in enumerate(selected):
                    for trial in range(eeg.shape[1]):
                        raw = np.asarray(eeg[subject, trial].T, dtype=np.float32)
                        if not np.isfinite(raw).all():
                            raise ValueError(f"Nonfinite EEG at subject={subject}, trial={trial}.")
                        # Filter complete trials BEFORE windowing, using EEGProc's
                        # established notch/detrend/Butterworth implementation.
                        filtered = _filter_dreamer_trial(raw) if eeg.shape[2] == 14 else raw
                        if filtered.shape != (eeg.shape[-1], 42) or not np.isfinite(filtered).all():
                            raise ValueError(f"Invalid filtered EEG at subject={subject}, trial={trial}.")
                        converted[row_index, trial] = filtered.T
                    print(f"Prepared subject ID {subject} ({row_index + 1}/{len(selected)})", flush=True)
                converted.flush()
            finally:
                del converted
            if eeg.shape[2] == 14:
                metadata["filter"] = "EEGProc _filter_dreamer_trial: theta 4-8, alpha 8-13, beta 13-30 Hz; order 4; notch 50 Hz; detrend; no rereference"
            np.save(temporary_labels, np.asarray(ratings[selected]))
            _shared.prepare_data(temporary, temporary_labels, cache, expected_subjects=len(selected))
            # Preserve original subject IDs, including non-contiguous subsets.
            np.save(cache / "subjects.npy", np.repeat(np.asarray(selected), eeg.shape[1]))
        else:
            _shared.prepare_data(eeg_path, labels_path, cache, expected_subjects=expected_subjects)
        metadata["status"] = "completed"
        _shared.write_json(cache / "preprocessing.json", metadata)
        return selected
    except Exception as exc:
        metadata.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        _shared.write_json(cache / "preprocessing.json", metadata)
        raise
    finally:
        if temporary.exists():
            temporary.unlink()
        if temporary_labels.exists():
            temporary_labels.unlink()
