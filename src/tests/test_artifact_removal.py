"""remove_eye_artifacts on synthetic EEG where the true blink is known."""

import importlib.util
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

from eegproc import bandpass_filter, feature_grouped_by_metadata, remove_eye_artifacts
from eegproc.artifact_removal import (
    _drop_transients,
    _left_right_balance,
    _numerical_rank,
    _passes_power_check,
    _segment_positions,
)

needs_mne = pytest.mark.skipif(
    importlib.util.find_spec("mne") is None, reason="requires eegproc[ica]"
)

FS = 128
CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
BLINK_MAP = dict(zip(CHANNELS, [1.0, 0.5, 0.6, 0.25, 0.05, 0.02, 0.01,
                                0.01, 0.02, 0.05, 0.25, 0.6, 0.5, 1.0]))


def _blink_train(n_samples, rng, amplitude=150.0):
    """Roughly one 0.2-second blink every four seconds."""
    time = np.arange(n_samples) / FS
    train = np.zeros(n_samples)
    for onset in np.arange(2.0, n_samples / FS - 2.0, 4.0) + rng.uniform(-1, 1, 1):
        train += amplitude * np.exp(-0.5 * ((time - onset) / 0.08) ** 2)
    return train


def _recording(seconds=180, blinks=True, seed=0):
    """Return a table of mixed brain-like sources, and those sources without blinks."""
    rng = np.random.default_rng(seed)
    n_samples = FS * seconds
    brain = rng.laplace(scale=8.0, size=(n_samples, len(CHANNELS))) @ (
        np.eye(len(CHANNELS)) + 0.3 * rng.standard_normal((len(CHANNELS), len(CHANNELS)))
    )
    signal = brain.copy()
    if blinks:
        signal += _blink_train(n_samples, rng)[:, None] * np.array([BLINK_MAP[c] for c in CHANNELS])
    return pd.DataFrame(signal, columns=CHANNELS), pd.DataFrame(brain, columns=CHANNELS)


def _band(frame):
    """What the fit sees: the 1-45 Hz content of a table."""
    return bandpass_filter(frame[CHANNELS], FS, bands=None, low=1.0, high=45.0,
                           notch_hz=None, reref=False, detrend=True)


def _rms(values):
    return float(np.sqrt(np.mean(np.square(values))))


@needs_mne
def test_removes_a_known_blink_and_keeps_the_back_of_the_head():
    raw, brain = _recording()
    report = []

    clean = remove_eye_artifacts(raw, FS, eeg_channels=CHANNELS, report=report)

    assert report[0]["status"] == "cleaned"
    assert len(report[0]["components"]) == 1
    before = _rms(_band(raw)["AF3"] - _band(brain)["AF3"])
    after = _rms(_band(clean)["AF3"] - _band(brain)["AF3"])
    assert after < 0.25 * before
    back = ["O1", "O2", "P7", "P8"]
    change = _band(clean)[back] - _band(raw)[back]
    assert _rms(change) < 0.15 * _rms(_band(raw)[back])


@needs_mne
def test_keeps_columns_rows_and_other_values():
    raw, _ = _recording()
    table = raw.assign(trial=1, label="high", note=np.arange(len(raw)))
    table.index = table.index + 500

    clean = remove_eye_artifacts(table, FS, eeg_channels=CHANNELS)

    assert list(clean.columns) == list(table.columns)
    assert clean.index.equals(pd.RangeIndex(len(table)))
    assert clean["note"].tolist() == table["note"].tolist()
    assert (clean["label"] == "high").all()
    assert not np.allclose(clean["AF3"], table["AF3"])


@needs_mne
def test_input_table_is_not_modified():
    raw, _ = _recording()
    original = raw.copy()
    remove_eye_artifacts(raw, FS, eeg_channels=CHANNELS)
    pd.testing.assert_frame_equal(raw, original)


@needs_mne
def test_channel_offsets_survive_removal():
    raw, _ = _recording()
    offset = np.linspace(4000, 4400, len(CHANNELS))
    clean = remove_eye_artifacts(raw + offset, FS, eeg_channels=CHANNELS)
    assert np.abs(clean[CHANNELS].mean().to_numpy() - offset).max() < 40


@needs_mne
def test_data_without_blinks_is_returned_unchanged_with_a_warning():
    raw, _ = _recording(blinks=False)
    report = []

    with pytest.warns(RuntimeWarning, match="no component looks like a blink"):
        clean = remove_eye_artifacts(raw, FS, eeg_channels=CHANNELS, report=report)

    assert report[0]["status"] == "no_blink_component"
    np.testing.assert_array_equal(clean[CHANNELS].to_numpy(), raw[CHANNELS].to_numpy())


@needs_mne
def test_slow_activity_across_the_whole_head_is_not_taken_for_a_blink():
    raw, _ = _recording(blinks=False)
    raw = raw.add(_blink_train(len(raw), np.random.default_rng(1)), axis=0)
    report = []

    with pytest.warns(RuntimeWarning, match="no component looks like a blink"):
        remove_eye_artifacts(raw, FS, eeg_channels=CHANNELS, report=report)

    assert report[0]["status"] == "no_blink_component"


@needs_mne
@pytest.mark.parametrize(
    "weights",
    [
        {"AF4": 1.0, "F4": 0.6, "F8": 0.5},  # one noisy side
        {"AF4": 1.0, "F4": 0.6, "AF3": -1.0, "F3": -0.6},  # sideways eye movement
    ],
)
def test_one_sided_frontal_activity_is_not_taken_for_a_blink(weights):
    raw, _ = _recording(blinks=False)
    train = _blink_train(len(raw), np.random.default_rng(1))
    for channel, weight in weights.items():
        raw[channel] += weight * train
    report = []

    with pytest.warns(RuntimeWarning, match="no component looks like a blink"):
        remove_eye_artifacts(raw, FS, eeg_channels=CHANNELS, report=report)

    assert report[0]["status"] == "no_blink_component"


@needs_mne
def test_fast_frontal_activity_is_not_taken_for_a_blink():
    raw, _ = _recording(blinks=False)
    noise = np.random.default_rng(1).laplace(scale=60.0, size=len(raw))
    raw = raw + noise[:, None] * np.array([BLINK_MAP[c] for c in CHANNELS])
    report = []

    with pytest.warns(RuntimeWarning, match="no component looks like a blink"):
        remove_eye_artifacts(raw, FS, eeg_channels=CHANNELS, report=report)

    assert report[0]["status"] == "no_blink_component"


@needs_mne
def test_too_little_data_is_returned_unchanged_with_a_warning():
    raw, _ = _recording(seconds=10)
    report = []

    with pytest.warns(RuntimeWarning, match="usable samples"):
        clean = remove_eye_artifacts(raw, FS, eeg_channels=CHANNELS, report=report)

    assert report[0]["status"] == "not_enough_data"
    np.testing.assert_array_equal(clean[CHANNELS].to_numpy(), raw[CHANNELS].to_numpy())


@needs_mne
def test_failed_power_check_returns_the_data_unchanged():
    raw, _ = _recording()
    report = []

    with pytest.warns(RuntimeWarning, match="failed the power check"):
        clean = remove_eye_artifacts(
            raw, FS, eeg_channels=CHANNELS, max_power_change=-1.0, report=report,
        )

    assert report[0]["status"] == "failed_power_check"
    np.testing.assert_array_equal(clean[CHANNELS].to_numpy(), raw[CHANNELS].to_numpy())


@needs_mne
def test_common_average_reference_uses_one_fewer_component():
    raw, brain = _recording()
    report = []

    clean = remove_eye_artifacts(raw, FS, eeg_channels=CHANNELS, reref=True, report=report)

    assert report[0]["status"] == "cleaned"
    assert report[0]["n_components"] == len(CHANNELS) - 1
    np.testing.assert_allclose(clean[CHANNELS].mean(axis=1), 0.0, atol=1e-8)
    # Fourteen sources in thirteen dimensions cannot separate fully, so the
    # limit is looser than the unreferenced case.
    truth = _band(brain.sub(brain.mean(axis=1), axis=0))["AF3"]
    reference = _band(raw.sub(raw.mean(axis=1), axis=0))["AF3"]
    assert _rms(_band(clean)["AF3"] - truth) < 0.4 * _rms(reference - truth)


@needs_mne
def test_reference_is_kept_when_nothing_is_removed():
    raw, _ = _recording(seconds=10)
    with pytest.warns(RuntimeWarning):
        clean = remove_eye_artifacts(raw, FS, eeg_channels=CHANNELS, reref=True)
    np.testing.assert_allclose(clean[CHANNELS].mean(axis=1), 0.0, atol=1e-8)


@needs_mne
def test_component_selector_replaces_the_blink_search():
    import mne

    raw, _ = _recording()
    received = {}

    def selector(ica, fitted):
        received.update(ica=ica, raw=fitted)
        return [np.int64(0)]

    report = []
    with pytest.warns(RuntimeWarning, match="power check"):
        remove_eye_artifacts(
            raw, FS, eeg_channels=CHANNELS, component_selector=selector,
            max_power_change=-1.0, report=report,
        )

    assert isinstance(received["ica"], mne.preprocessing.ICA)
    assert isinstance(received["raw"], mne.io.BaseRaw)
    assert report[0]["components"] == [0]


@needs_mne
def test_selector_returning_nothing_is_reported():
    raw, _ = _recording()
    with pytest.warns(RuntimeWarning, match="no component"):
        remove_eye_artifacts(raw, FS, eeg_channels=CHANNELS,
                             component_selector=lambda ica, fitted: [])


@needs_mne
def test_selector_index_out_of_range_raises():
    raw, _ = _recording()
    with pytest.raises(ValueError, match="component_selector returned"):
        remove_eye_artifacts(raw, FS, eeg_channels=CHANNELS,
                             component_selector=lambda ica, fitted: [99])


@needs_mne
def test_runs_per_subject_through_the_grouped_runner():
    tables = []
    for subject in (1, 2):
        recording, _ = _recording(seed=subject)
        recording["subject_id"] = subject
        recording["trial_id"] = np.repeat([1, 2, 3], len(recording) // 3)
        tables.append(recording)
    raw = pd.concat(tables, ignore_index=True)
    report = []

    clean = feature_grouped_by_metadata(
        raw, target_function=remove_eye_artifacts, fs=FS,
        group_by_metadata_columns=["subject_id"],
        eeg_channels=CHANNELS, segment_columns=["trial_id"], report=report,
    )

    assert [entry["status"] for entry in report] == ["cleaned", "cleaned"]
    assert len(clean) == len(raw)
    assert not clean.isna().any().any()
    assert clean["subject_id"].equals(raw["subject_id"])
    assert clean["trial_id"].equals(raw["trial_id"])


@needs_mne
def test_large_transients_do_not_stop_the_blink_being_found():
    raw, brain = _recording()
    rng = np.random.default_rng(3)
    for start in rng.integers(0, len(raw) - FS, 12):
        raw.iloc[start:start + 20, rng.integers(0, len(CHANNELS))] += 6000.0
    report = []

    remove_eye_artifacts(raw, FS, eeg_channels=CHANNELS, report=report)

    assert report[0]["rejected_fraction"] > 0
    assert report[0]["status"] == "cleaned"


@needs_mne
def test_default_montage_is_not_deprecated():
    raw, _ = _recording(seconds=10)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*[Mm]ontage.*", category=FutureWarning)
        with pytest.warns(RuntimeWarning, match="usable samples"):
            remove_eye_artifacts(raw, FS, eeg_channels=CHANNELS)


@needs_mne
def test_missing_samples_are_interpolated():
    raw, _ = _recording()
    raw.iloc[100:110, 0] = np.nan
    clean = remove_eye_artifacts(raw, FS, eeg_channels=CHANNELS)
    assert np.isfinite(clean[CHANNELS].to_numpy()).all()


@needs_mne
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"eeg_channels": ["AF3"]}, "at least two"),
        ({"eeg_channels": ["AF3", "AF3"]}, "at least two"),
        ({"eeg_channels": ["AF3", "NOPE"]}, "not in the table"),
        ({"eeg_channels": ["AF3", "text"]}, "must be numeric"),
        ({"eeg_channels": ["AF3", "empty"]}, "no finite samples"),
        ({"eeg_channels": ["AF3", "X1"]}, "montage"),
        ({"eeg_channels": CHANNELS, "segment_columns": ["trial_id"]}, "segment_columns"),
    ],
)
def test_bad_arguments_raise(kwargs, message):
    raw, _ = _recording(seconds=10)
    table = raw.assign(text="a", empty=np.nan, X1=1.0)
    with pytest.raises(ValueError, match=message):
        remove_eye_artifacts(table, FS, **kwargs)


def test_missing_mne_explains_how_to_install_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "mne", None)
    with pytest.raises(ImportError, match=r"eegproc\[ica\]"):
        remove_eye_artifacts(pd.DataFrame({"AF3": [0.0]}), FS, eeg_channels=["AF3"])


def test_importing_eegproc_does_not_import_mne():
    code = "import sys, eegproc; raise SystemExit('mne' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_transient_limit_follows_the_recording_and_keeps_blink_sized_windows():
    rng = np.random.default_rng(0)
    quiet = rng.standard_normal((2, FS * 100))
    quiet[:, FS * 10:FS * 10 + 20] += 4.0  # blink-sized: within five typical windows
    quiet[:, FS * 50:FS * 50 + 20] += 100.0  # transient
    for scale in (1.0, 1000.0):
        kept, rejected = _drop_transients(quiet * scale, FS, 5.0)
        assert rejected == pytest.approx(0.01)
        assert kept.shape == (2, FS * 99)


def test_transient_check_handles_less_than_one_window():
    kept, rejected = _drop_transients(np.zeros((2, FS - 1)), FS, 5.0)
    assert kept.shape == (2, 0)
    assert rejected == 0.0


def test_left_right_balance():
    maps = np.array([[1.0, 1.0, 1.0, 1.0],
                     [1.0, 0.5, -1.0, 0.0]])
    np.testing.assert_allclose(_left_right_balance(maps, [0], [1]), [1.0, 0.5, 0.0, 0.0])
    np.testing.assert_allclose(_left_right_balance(maps, [0], []), np.ones(4))


def _change(front=-0.5, delta=0.0, alpha=0.0):
    bands = {"delta": delta, "theta": 0.0, "alpha": alpha, "beta": 0.0, "gamma": 0.0}
    return {"front": dict.fromkeys(bands, front), "back": bands}


@pytest.mark.parametrize(
    ("change", "reref", "expected"),
    [
        (_change(), False, True),
        (_change(front=0.2), False, False),  # frontal power rose
        (_change(alpha=0.2), False, False),  # posterior power rose
        (_change(alpha=-0.2), False, False),  # posterior alpha was removed
        (_change(delta=-0.2), False, True),  # slow blink activity reaches the back
        (_change(delta=0.2), False, False),  # but it should never rise
        (_change(alpha=-0.2), True, True),  # CAR spreads blinks to the back
        (_change(alpha=0.2), True, False),
    ],
)
def test_power_check(change, reref, expected):
    assert _passes_power_check(change, 0.1, reref) is expected


def test_rank_drops_by_one_after_common_average_reference():
    data = np.random.default_rng(0).standard_normal((6, 2000))
    assert _numerical_rank(data) == 6
    assert _numerical_rank(data - data.mean(axis=0)) == 5


def test_segments_follow_table_order_even_when_values_repeat():
    table = pd.DataFrame({"trial": [2, 2, 1, 1], "segment": ["a", "b", "a", "a"]})
    positions = _segment_positions(table, ["trial", "segment"])
    assert [p.tolist() for p in positions] == [[0], [1], [2, 3]]
    assert [p.tolist() for p in _segment_positions(table, ())] == [[0, 1, 2, 3]]
