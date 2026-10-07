"""Equivalence guard for the windowing assembler.

``eegproc.data.to_supervised_arrays`` replaced the research-tree
``build_dataset`` that bridged trial arrays to cross-validation input. That
function's output was recorded as golden data (``data/joint_models_data_golden.npz``)
before it was removed, and these tests hold the replacement to it.

The golden file records, per variant, the exact bytes of the feature array (as a
sha256), the first and last windows and every window mean, plus the labels and
subject ids in full. The byte-level check is the strictest guard; the tolerance
check states the same expectation in a form that does not depend on the CPU's
floating-point summation order.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from .utils import make_synthetic_trial_arrays

GOLDEN_PATH = Path(__file__).parent / "data" / "joint_models_data_golden.npz"

VARIANTS = ["ov0_zs1_valence", "ov50_zs1_valence", "ov0_zs0_valence", "ov0_zs1_arousal"]


def _sha256(array: np.ndarray) -> bytes:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).digest()


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN_PATH.exists():
        pytest.fail(f"golden fixture missing: {GOLDEN_PATH}")
    with np.load(GOLDEN_PATH, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def test_synthetic_input_is_deterministic(golden):
    """The generator must keep producing the bytes the golden data was built from.

    If this fails, the golden file describes a different input and every other
    assertion in this module is meaningless.
    """
    eeg, labels = make_synthetic_trial_arrays(
        n_samples=int(golden["_meta_n_samples"])
    )
    assert _sha256(eeg) == golden["_input_eeg_sha256"].tobytes()
    assert _sha256(labels) == golden["_input_labels_sha256"].tobytes()


@pytest.mark.parametrize("tag", VARIANTS)
def test_golden_window_count_matches_arithmetic(golden, tag):
    """Re-derive the window count independently rather than trusting the recording."""
    n_samples = int(golden["_meta_n_samples"])
    window = int(round(float(golden["_meta_window_sec"]) * float(golden["_meta_fs"])))
    overlap = 0.5 if tag.startswith("ov50") else 0.0
    hop = max(1, int(round(window * (1.0 - overlap))))

    eeg, _ = make_synthetic_trial_arrays(n_samples=n_samples)
    n_subjects, n_trials = eeg.shape[0], eeg.shape[1]
    expected = n_subjects * n_trials * (1 + (n_samples - window) // hop)

    shape = golden[f"{tag}__features_shape"]
    assert shape.tolist() == [expected, window, eeg.shape[2]]


@pytest.mark.parametrize("tag", VARIANTS)
def test_golden_arrays_are_mutually_consistent(golden, tag):
    n_windows = int(golden[f"{tag}__features_shape"][0])
    labels = golden[f"{tag}__labels"]
    subjects = golden[f"{tag}__subject_ids"]

    assert labels.shape == (n_windows,)
    assert subjects.shape == (n_windows,)
    assert set(np.unique(labels).tolist()) == {0, 1}, "median split must yield both classes"
    # every subject contributes the same number of windows (rectangular corpus)
    _, counts = np.unique(subjects, return_counts=True)
    assert len(set(counts.tolist())) == 1


def test_golden_features_depend_on_windowing_not_labels(golden):
    """Changing label_dimension must not perturb the feature array."""
    assert (
        golden["ov0_zs1_valence__features_sha256"].tobytes()
        == golden["ov0_zs1_arousal__features_sha256"].tobytes()
    )
    assert (
        golden["ov0_zs1_valence__features_sha256"].tobytes()
        != golden["ov0_zs0_valence__features_sha256"].tobytes()
    ), "z-scoring must change the features"


def _tidy_from_trial_arrays(eeg, labels):
    """Convert the on-disk trial-major layout into the v2 tidy frame."""
    import pandas as pd

    n_subjects, n_trials, n_channels, n_samples = eeg.shape
    rows = n_subjects * n_trials * n_samples
    data = {
        f"ch{c}": eeg[:, :, c, :].reshape(rows).astype(np.float32)
        for c in range(n_channels)
    }
    data["subject"] = np.repeat(np.arange(n_subjects), n_trials * n_samples)
    data["trial"] = np.tile(np.repeat(np.arange(n_trials), n_samples), n_subjects)
    data["sample_index"] = np.tile(np.arange(n_samples), n_subjects * n_trials)
    data["valence"] = np.repeat(labels[:, :, 0].reshape(-1), n_samples)
    data["arousal"] = np.repeat(labels[:, :, 1].reshape(-1), n_samples)
    return pd.DataFrame(data)


def _assemble_variant(golden, tag):
    from eegproc.data import EEGFrame, median_split, to_supervised_arrays

    n_samples = int(golden["_meta_n_samples"])
    eeg, labels = make_synthetic_trial_arrays(n_samples=n_samples)
    table = _tidy_from_trial_arrays(eeg, labels)

    label = "arousal" if tag.endswith("arousal") else "valence"
    overlap = 0.5 if tag.startswith("ov50") else 0.0
    normalize = "subject_zscore" if "_zs1_" in tag else None

    frame = EEGFrame(
        data=table,
        fs=float(golden["_meta_fs"]),
        kind="signal",
        label_columns=("valence", "arousal"),
        feature_columns=tuple(f"ch{c}" for c in range(eeg.shape[2])),
    )
    return to_supervised_arrays(
        frame,
        window_sec=float(golden["_meta_window_sec"]),
        overlap=overlap,
        label_column=label,
        label_transform=median_split(label, threshold=float(golden["_meta_median_label"])),
        normalize=normalize,
    )


@pytest.mark.parametrize("tag", VARIANTS)
def test_to_supervised_arrays_matches_golden(golden, tag):
    """The assembler must reproduce the reference implementation bit-for-bit."""
    result = _assemble_variant(golden, tag)

    assert list(result.features.shape) == golden[f"{tag}__features_shape"].tolist()
    assert _sha256(result.features) == golden[f"{tag}__features_sha256"].tobytes(), (
        "feature bytes differ from the reference implementation"
    )
    np.testing.assert_array_equal(result.labels, golden[f"{tag}__labels"])
    np.testing.assert_array_equal(result.subject_ids, golden[f"{tag}__subject_ids"])


@pytest.mark.parametrize("tag", VARIANTS)
def test_to_supervised_arrays_matches_golden_values(golden, tag):
    """The same expectation, tolerant of last-bit differences between CPUs."""
    result = _assemble_variant(golden, tag)

    np.testing.assert_allclose(
        result.features[0], golden[f"{tag}__features_first"], rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        result.features[-1], golden[f"{tag}__features_last"], rtol=1e-5, atol=1e-6
    )
    # Window means can be ~1e-5, so the absolute tolerance does the work here.
    np.testing.assert_allclose(
        result.features.mean(axis=(1, 2)),
        golden[f"{tag}__features_window_means"],
        rtol=1e-5,
        atol=1e-6,
    )


def test_assembler_returns_trial_ids_the_reference_dropped(golden):
    """Trial ids are returned, unique per (subject, trial); the reference returned none."""
    from eegproc.data import EEGFrame, to_supervised_arrays

    n_samples = int(golden["_meta_n_samples"])
    eeg, labels = make_synthetic_trial_arrays(n_samples=n_samples)
    frame = EEGFrame(
        data=_tidy_from_trial_arrays(eeg, labels),
        fs=float(golden["_meta_fs"]),
        kind="signal",
        label_columns=("valence",),
        feature_columns=tuple(f"ch{c}" for c in range(eeg.shape[2])),
    )
    result = to_supervised_arrays(frame, window_sec=2.0, label_column="valence")

    n_subjects, n_trials = eeg.shape[0], eeg.shape[1]
    assert len(np.unique(result.trial_ids)) == n_subjects * n_trials
    assert len(np.unique(result.subject_ids)) == n_subjects
    # trial ids must be unique per (subject, trial), not repeated across subjects
    pairs = set(zip(result.subject_ids.tolist(), result.trial_ids.tolist()))
    assert len(pairs) == n_subjects * n_trials
