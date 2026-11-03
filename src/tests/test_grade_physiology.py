"""Exercise the merged grading CLIs against real synthetic EEG archives."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from eegproc.model_explainability.typicality.grading_data import load_grading_trials
from eegproc.model_explainability.typicality.physiology import (
    FAMILIES, PhysiologicalReference, signal_diagnostics,
)


ROOT = Path(__file__).resolve().parents[2]
GRADERS = ROOT / "src/eegproc/model_explainability/typicality"


@pytest.fixture
def archived_suite(tmp_path):
    eeg = np.random.default_rng(52).normal(size=(3, 2, 42, 384)).astype(np.float32)
    ratings = np.tile(np.array([[1., 5.], [5., 1.]])[None, :, :], (3, 1, 1))
    eeg_path, labels_path = tmp_path / "eeg.npy", tmp_path / "labels.npy"
    np.save(eeg_path, eeg)
    np.save(labels_path, ratings)
    dataset = load_grading_trials({"raw_eeg_npy": str(eeg_path),
                                  "raw_labels_npy": str(labels_path), "label_dimension": "arousal"})
    signals = dataset.features * dataset.normalization_scale + dataset.normalization_offset
    diagnostics = [signal_diagnostics(signal, fs=128, n_channels=14) for signal in signals]
    reference = PhysiologicalReference.fit([d for i, d in enumerate(diagnostics)
                                            if dataset.subject_ids[i] != 0])
    suite = tmp_path / "suite"
    attempts = []
    for trial, objective in ((0, "base"), (1, "base"), (0, "typicality")):
        attempt = suite / "fold_00/subject_0" / f"trial_{trial}" / objective / "attempt_0001"
        attempt.mkdir(parents=True)
        summaries = {}
        for which, factor in (("original", 1.), ("reconstruction", 1.1), ("counterfactual", 1.5)):
            candidate = signal_diagnostics(signals[trial] * factor, fs=128, n_channels=14)
            np.savez_compressed(attempt / f"physiology_{which}.npz",
                                **{key: candidate[key] for key in FAMILIES})
            summaries[which] = reference.assess(candidate)
        (attempt / "physiology.json").write_text(json.dumps(summaries))
        attempts.append(attempt)
    return eeg_path, labels_path, suite, attempts


def run_grader(script, arguments):
    # Fail if grading imports model code, even on a machine with TF installed.
    driver = """
import importlib.abc, runpy, sys
class NoModelRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'tensorflow', 'keras'}:
            raise AssertionError('CPU grading imported ' + fullname)
sys.meta_path.insert(0, NoModelRuntime())
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    return subprocess.run([sys.executable, "-c", driver, str(GRADERS / script),
                           "--repo", str(ROOT), *map(str, arguments)],
                          env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                          capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("which", ["counterfactual", "original", "reconstruction"])
def test_single_archive_grader_and_bounds_gate(archived_suite, which):
    eeg, labels, _, attempts = archived_suite
    args = ["--attempt", attempts[0], "--held-out", "0", "--task", "arousal",
            "--eeg", eeg, "--labels", labels, "--which", which]
    result = run_grader("grade_physiology.py", args)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "gate passed" in result.stdout
    assert "graded families      : 4" in result.stdout
    saved = attempts[0] / "physiology.json"
    payload = json.loads(saved.read_text())
    payload[which]["checks"]["amplitude"]["fraction_in_range"] = .123456
    saved.write_text(json.dumps(payload))
    result = run_grader("grade_physiology.py", args)
    assert result.returncode == 1
    assert "graded numbers withheld" in result.stdout


@pytest.mark.parametrize("statistic", ["worst", "mean"])
def test_table_groups_by_objective_and_rejects_changed_bounds(archived_suite, statistic):
    eeg, labels, suite, attempts = archived_suite
    args = ["--suite", suite, "--task", "arousal", "--eeg", eeg, "--labels", labels,
            "--statistic", statistic]
    result = run_grader("grade_physiology_table.py", args)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [line.split() for line in result.stdout.splitlines()
            if line.startswith(("base ", "typicality "))]
    assert [(row[0], row[1]) for row in rows] == [("base", "2"), ("typicality", "1")]
    assert "LaTeX cell" in result.stdout
    saved = attempts[0] / "physiology.json"
    payload = json.loads(saved.read_text())
    payload["counterfactual"]["checks"]["amplitude"]["fraction_in_range"] = .123456
    saved.write_text(json.dumps(payload))
    result = run_grader("grade_physiology_table.py", args)
    assert result.returncode == 1
    assert "GATE FAILED on 1 trials" in result.stdout


def test_loader_preserves_legacy_float32_normalization_round_trip(archived_suite):
    eeg_path, labels, _, _ = archived_suite
    raw = np.load(eeg_path)
    dataset = load_grading_trials({"raw_eeg_npy": str(eeg_path),
                                  "raw_labels_npy": str(labels), "label_dimension": "valence"})
    windows = raw.reshape(3, 2, 42, 3, 128).transpose(0, 1, 3, 4, 2).reshape(6, 3, 128, 42)
    scales = np.sqrt(np.mean(np.square(windows, dtype=np.float64),
                             axis=(2, 3), keepdims=True)).astype(np.float32)
    expected = (windows / np.maximum(scales, 1e-6)).astype(np.float32)
    np.testing.assert_array_equal(dataset.features, expected)
    np.testing.assert_array_equal(dataset.normalization_scale, scales)
    np.testing.assert_array_equal(dataset.subject_ids, [0, 0, 1, 1, 2, 2])
    np.testing.assert_array_equal(dataset.trial_ids, [0, 1, 0, 1, 0, 1])


def test_zero_width_and_undefined_deviations_are_unscorable():
    lower = {family: np.array([1., 1.]) for family in FAMILIES}
    upper = {family: np.array([1., 3.]) for family in FAMILIES}
    reference = PhysiologicalReference(lower, upper)
    value = reference.deviation("amplitude", np.array([1., 4.]))
    assert value["n_scored"] == value["n_unscorable"] == 1
    assert value["exceedance_mean"] == 1.
    assert reference.deviation("amplitude", np.array([1., np.nan])) is None
