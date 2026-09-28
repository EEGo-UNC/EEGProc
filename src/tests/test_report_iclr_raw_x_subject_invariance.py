"""The raw-X comparison must score both groups against original source X."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from eegproc.model_explainability import report_iclr_raw_x_subject_invariance as report


def test_report_directory_uses_only_its_linked_studies(tmp_path):
    linked = tmp_path / "studies" / "fold_00"
    unlinked = tmp_path / "studies" / "fold_01"
    linked.mkdir(parents=True)
    unlinked.mkdir(parents=True)
    (linked / "study.json").write_text("{}")
    (unlinked / "study.json").write_text("{}")
    report_dir = tmp_path / "final-ICLR-report"
    report_dir.mkdir()
    (report_dir / "report.json").write_text(json.dumps({"studies": [str(linked)]}))
    assert report._roots([report_dir]) == [linked]
    (linked / "study.json").unlink()
    with pytest.raises(ValueError, match="linked study is unavailable"):
        report._roots([report_dir])


def test_diagonal_discrepancy_uses_original_waveform_coordinates():
    source = np.array([[[[0.0, 2.0]]], [[[2.0, 4.0]]]])
    mean, variance = report.fit_region(source)
    np.testing.assert_array_equal(mean, [[[1.0, 3.0]]])
    np.testing.assert_array_equal(variance, [[[1.0, 1.0]]])
    np.testing.assert_allclose(
        report.score_region(np.array([[[3.0, 4.0]]]), mean, variance), [2.5]
    )
    with pytest.raises(ValueError, match="shape"):
        report.score_region(np.ones((1, 1, 3)), mean, variance)


def test_fold_selects_typical_source_x_and_scores_both_groups_without_decoder(tmp_path, monkeypatch):
    root = tmp_path / "study"
    fold_dir = root / "subject_0"
    fold_dir.mkdir(parents=True)
    (fold_dir / "fold.json").write_text(json.dumps({"threshold": 0.5}))
    # The last source trial is true class 1 but fails archived typicality.
    keys = [(1, 0), (2, 0), (3, 0), (3, 1), (0, 0), (0, 1)]
    values = np.array([0, 2, 4, 100, 1, 3], dtype=np.float32).reshape(-1, 1, 1, 1)
    dataset = SimpleNamespace(features=values, labels=np.ones(len(keys), dtype=int))
    index = dict(zip(keys, range(len(keys))))
    source = {
        "subject_ids": [1, 2, 3, 3], "trial_ids": [0, 0, 0, 1],
        "labels": [1, 1, 1, 1],
        "probabilities": [[0.1, 0.9]] * 4,
        "discrepancy": [0.1, 0.2, 0.3, 0.9],
    }
    heldout = {
        "trial_ids": [0, 1], "labels": [1, 1],
        "probabilities": [[0.7, 0.3], [0.1, 0.9]],
        "discrepancy": [4, 0.1],
    }
    monkeypatch.setattr(report, "analyze_fold", lambda *args, **kwargs: (None, None, None, {}))
    monkeypatch.setattr(
        report, "_load_summary",
        lambda stem, required: (stem, source if "source_trials" in str(stem) else heldout),
    )
    row, trial_rows, crossfit_rows, _, _ = report._fold(
        root, {"task": "valence"}, {"subject_id": 0}, dataset=dataset,
        index=index, samples_per_subject=0, seed=42,
        variance_floor=1e-6, quantile=0.95,
    )
    assert row["n_source_typical_class1"] == 3
    assert row["n_heldout_true_class1"] == 2
    assert row["n_cross_subject_comparisons"] == 3
    assert {r["trial_id"] for r in trial_rows if r["role"] == "source_typical_class1_X"} == {0}
    # The held-out trial predicted class 0 is still included because the
    # held-out selection is based on its true class.
    assert {r["trial_id"] for r in trial_rows if r["role"] == "heldout_true_class1_X"} == {0, 1}
    assert row["source_raw_x_median"] == pytest.approx(1.5)
    assert row["heldout_raw_x_median"] == pytest.approx(0.375)
    # When source subject 1 is compared with held-out subject 0, both are
    # scored against the same reference from source subjects 2 and 3.
    comparison = [r for r in crossfit_rows if r["reference_excluded_source_subject"] == 1]
    scores = {(r["role"], r["subject_id"], r["trial_id"]): r["raw_x_discrepancy"]
              for r in comparison}
    assert scores[("source_typical_class1_X", 1, 0)] == pytest.approx(9.0)
    assert scores[("heldout_true_class1_X", 0, 0)] == pytest.approx(4.0)


def test_fold_with_too_few_typical_source_trials_is_explicitly_unscored(tmp_path, monkeypatch):
    root = tmp_path / "study"
    fold_dir = root / "subject_0"
    fold_dir.mkdir(parents=True)
    (fold_dir / "fold.json").write_text(json.dumps({"threshold": 0.5}))
    source = {
        "subject_ids": [1], "trial_ids": [0], "labels": [1],
        "probabilities": [[0.1, 0.9]], "discrepancy": [0.1],
    }
    heldout = {
        "trial_ids": [0], "labels": [1],
        "probabilities": [[0.8, 0.2]], "discrepancy": [2.0],
    }
    dataset = SimpleNamespace(
        features=np.array([0.0, 1.0]).reshape(2, 1, 1, 1),
        labels=np.ones(2, dtype=int),
    )
    monkeypatch.setattr(report, "analyze_fold", lambda *args, **kwargs: (None, None, None, {}))
    monkeypatch.setattr(
        report, "_load_summary",
        lambda stem, required: (stem, source if "source_trials" in str(stem) else heldout),
    )
    row, trials, crossfit, _, _ = report._fold(
        root, {"task": "valence"}, {"subject_id": 0}, dataset=dataset,
        index={(1, 0): 0, (0, 0): 1}, samples_per_subject=0,
        seed=42, variance_floor=1e-6, quantile=0.95,
    )
    assert row["status"] == "insufficient_typical_source_class1_X"
    assert row["n_heldout_true_class1"] == 1
    assert row["n_heldout_scored"] == 0
    assert row["heldout_inside_source_raw_x_percent"] is None
    assert trials == crossfit == []
