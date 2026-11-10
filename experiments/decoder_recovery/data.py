"""Explicit, trial-auditable array inputs for decoder recovery experiments."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


def scalar(value):
    return value.item() if isinstance(value, np.generic) else value


def subject_key(value):
    return json.dumps(scalar(value), sort_keys=True, allow_nan=False)


@dataclass
class TrialData:
    eeg: np.ndarray
    labels: np.ndarray
    subject_ids: np.ndarray
    trial_ids: np.ndarray

    def __post_init__(self):
        self.eeg = np.asarray(self.eeg, dtype=np.float32)
        self.labels = np.asarray(self.labels).reshape(-1)
        self.subject_ids = np.asarray(self.subject_ids).reshape(-1)
        self.trial_ids = np.asarray(self.trial_ids).reshape(-1)
        n = len(self.eeg)
        if not n or self.eeg.ndim not in (3, 4):
            raise ValueError("eeg must be nonempty (N,T,F) or (N,W,T,F).")
        if any(len(a) != n for a in (self.labels, self.subject_ids, self.trial_ids)):
            raise ValueError("EEG, labels, subject_ids and trial_ids must align.")
        if not np.isfinite(self.eeg).all():
            raise ValueError("EEG must contain only finite values.")
        if (not np.issubdtype(self.labels.dtype, np.integer)
                or np.any(self.labels < 0)):
            raise ValueError("labels must be nonnegative integer class IDs.")
        self.labels = self.labels.astype(np.int32)
        self.keys()  # Reject non-JSON / nonfinite IDs before any fitting.

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            return cls(**{key: archive[key] for key in
                          ("eeg", "labels", "subject_ids", "trial_ids")})

    def keys(self):
        return {json.dumps([scalar(s), scalar(t)], allow_nan=False)
                for s, t in zip(self.subject_ids, self.trial_ids)}

    def fingerprint(self):
        digest = hashlib.sha256()
        for array in (self.eeg, self.labels, self.subject_ids, self.trial_ids):
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()


def validate_splits(train, validation):
    if train.eeg.shape[1:] != validation.eeg.shape[1:]:
        raise ValueError("Training and validation EEG shapes must match.")
    if train.keys() & validation.keys():
        raise ValueError("A subject/trial occurs in both training and validation.")


def vocabulary(subject_ids):
    return [scalar(v) for v in np.unique(subject_ids)]


def encode_subjects(subject_ids, classes, *, allow_unknown=False):
    mapping = {subject_key(value): index for index, value in enumerate(classes)}
    result = np.asarray([mapping.get(subject_key(value), -1)
                         for value in subject_ids], dtype=np.int32)
    if not allow_unknown and np.any(result < 0):
        raise ValueError("Subject ID is absent from the training-only vocabulary.")
    return result


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def new_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Use a new output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_fold(data, output, *, test_subjects, validation_subjects,
                 identity_validation_trials=1, seed=42):
    """Reserve whole subjects for source validation/test and trials for ID validation."""
    known = set(map(str, data.subject_ids))
    test, validation = set(map(str, test_subjects)), set(map(str, validation_subjects))
    if not test or not validation or test & validation or (test | validation) - known:
        raise ValueError("Choose nonempty, disjoint, existing test/validation subject IDs.")
    subjects = np.asarray(list(map(str, data.subject_ids)))
    test_mask, val_mask = np.isin(subjects, list(test)), np.isin(subjects, list(validation))
    train_mask = ~(test_mask | val_mask)
    source_subjects = np.unique(subjects[train_mask])
    if len(source_subjects) < 2 or identity_validation_trials < 1:
        raise ValueError("Need at least two training subjects and positive ID-validation trials.")
    rng, identity_mask = np.random.default_rng(seed), np.zeros(len(subjects), dtype=bool)
    for subject in source_subjects:
        mask = subjects == subject
        trial_ids = np.unique(data.trial_ids[mask])
        if len(trial_ids) <= identity_validation_trials:
            raise ValueError(f"Subject {subject} needs more trials than the ID-validation reserve.")
        selected = rng.choice(trial_ids, size=identity_validation_trials, replace=False)
        identity_mask |= mask & np.isin(data.trial_ids, selected)
    train_mask &= ~identity_mask
    output = new_output(output)
    manifest = {"seed": seed, "source_fingerprint": data.fingerprint(), "splits": {}}
    for name, mask in (("train", train_mask), ("identity_validation", identity_mask),
                       ("validation", val_mask), ("test", test_mask)):
        subset = TrialData(data.eeg[mask], data.labels[mask], data.subject_ids[mask], data.trial_ids[mask])
        np.savez_compressed(output / f"{name}.npz", eeg=subset.eeg, labels=subset.labels,
                            subject_ids=subset.subject_ids, trial_ids=subset.trial_ids)
        manifest["splits"][name] = {"subject_ids": vocabulary(subset.subject_ids),
            "n_examples": len(subset.eeg), "n_trials": len(subset.keys()),
            "fingerprint": subset.fingerprint()}
    write_json(output / "fold.json", manifest)
    return manifest
