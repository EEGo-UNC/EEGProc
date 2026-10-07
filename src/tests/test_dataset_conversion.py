"""eegproc-to-csv: DREAMER CSV exports, DEAP pickles, and safe output publishing."""

import errno
import pickle
import subprocess
import sys
from importlib.metadata import entry_points
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eegproc.data.to_csv import convert_dataset

EEG_CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
SEGMENTS = (("baseline", 3), ("stimulus", 5))


@pytest.fixture
def dreamer_exports(tmp_path):
    """The three-file DREAMER export: EEG, ECG, and per-trial ratings."""
    rng = np.random.default_rng(0)
    rows = [
        {"subject_id": s, "trial_id": t, "segment": segment, "sample_idx": i,
         **{c: rng.standard_normal() for c in EEG_CHANNELS}}
        for s in (1, 2) for t in (1, 2) for segment, n in SEGMENTS for i in range(1, n + 1)
    ]
    eeg = pd.DataFrame(rows)
    eeg.to_csv(tmp_path / "dreamer_eeg.csv", index=False)
    keys = eeg[["subject_id", "trial_id", "segment", "sample_idx"]]
    keys.assign(ECG1=1.0, ECG2=2.0).to_csv(tmp_path / "dreamer_ecg.csv", index=False)
    pd.DataFrame({
        "subject_id": [1, 1, 2, 2], "trial_id": [1, 2, 1, 2],
        "valence": [1, 4, 3, 5], "arousal": [2, 2, 3, 4], "dominance": [3, 3, 3, 3],
    }).to_csv(tmp_path / "dreamer_labels.csv", index=False)
    return tmp_path


def _no_temporary_files(directory: Path) -> bool:
    return not [p for p in directory.iterdir() if p.name.startswith(".")]


def test_dreamer_exports_join_into_one_table(dreamer_exports):
    output = dreamer_exports / "out" / "dreamer_joined.csv"

    rows = convert_dataset("dreamer", dreamer_exports, output)

    table = pd.read_csv(output)
    assert rows == len(table) == 2 * 2 * sum(n for _, n in SEGMENTS)
    assert table.columns.tolist() == [
        "subject_id", "trial_id", "segment", "sample_idx", *EEG_CHANNELS,
        "ECG1", "ECG2", "valence", "arousal", "dominance",
    ]
    ratings = table.drop_duplicates(["subject_id", "trial_id"]).set_index(["subject_id", "trial_id"])
    assert ratings.loc[(1, 2), "valence"] == 4 and ratings.loc[(2, 2), "arousal"] == 4
    assert (table["ECG1"] == 1.0).all() and (table["ECG2"] == 2.0).all()
    assert _no_temporary_files(output.parent)


def test_gzip_output_round_trips(dreamer_exports):
    plain = dreamer_exports / "plain.csv"
    compressed = dreamer_exports / "compressed.csv.gz"
    convert_dataset("dreamer", dreamer_exports, plain)
    convert_dataset("dreamer", dreamer_exports, compressed)

    pd.testing.assert_frame_equal(pd.read_csv(plain), pd.read_csv(compressed))


def test_existing_output_requires_overwrite(dreamer_exports):
    output = dreamer_exports / "joined.csv"
    output.write_text("keep me")

    with pytest.raises(FileExistsError):
        convert_dataset("dreamer", dreamer_exports, output)
    assert output.read_text() == "keep me"

    convert_dataset("dreamer", dreamer_exports, output, overwrite=True)
    assert pd.read_csv(output).shape[0] == 32


def test_filesystems_without_hard_links_fall_back_to_a_copy(dreamer_exports, monkeypatch):
    """exFAT/FAT drives and some network shares cannot create hard links."""
    def no_hard_links(self, target):
        raise OSError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(Path, "hardlink_to", no_hard_links)
    output = dreamer_exports / "out" / "joined.csv"

    convert_dataset("dreamer", dreamer_exports, output)

    assert pd.read_csv(output).shape[0] == 32
    assert _no_temporary_files(output.parent)


def test_copy_fallback_never_replaces_a_file_created_meanwhile(dreamer_exports, monkeypatch):
    output = dreamer_exports / "out" / "joined.csv"

    def concurrent_writer_then_no_hard_links(self, target):
        Path(self).write_text("written by someone else")
        raise OSError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(Path, "hardlink_to", concurrent_writer_then_no_hard_links)

    with pytest.raises(FileExistsError):
        convert_dataset("dreamer", dreamer_exports, output)
    assert output.read_text() == "written by someone else"
    assert _no_temporary_files(output.parent)


def test_deap_baseline_and_stimulus_segments(tmp_path):
    rng = np.random.default_rng(0)
    contents = {
        "data": rng.standard_normal((1, 40, 8064)).astype(np.float32),
        "labels": np.array([[6.5, 3.0, 5.0, 7.0]]),
    }
    with open(tmp_path / "s01.dat", "wb") as handle:
        pickle.dump(contents, handle)
    output = tmp_path / "deap.csv"

    rows = convert_dataset("deap", tmp_path, output)

    table = pd.read_csv(output)
    assert rows == 8064
    assert table["segment"].value_counts().to_dict() == {"stimulus": 7680, "baseline": 384}
    assert {"valence", "arousal", "dominance", "liking"} <= set(table.columns)
    assert table[["valence", "arousal", "dominance", "liking"]].iloc[0].tolist() == [6.5, 3.0, 5.0, 7.0]


def test_command_line_entry_point():
    scripts = {ep.name: ep.value for ep in entry_points(group="console_scripts")}
    assert scripts.get("eegproc-to-csv") == "eegproc.data.to_csv:main"

    completed = subprocess.run(
        [sys.executable, "-m", "eegproc.data.to_csv", "--help"],
        capture_output=True, text=True, check=True,
    )
    assert "--dataset" in completed.stdout
