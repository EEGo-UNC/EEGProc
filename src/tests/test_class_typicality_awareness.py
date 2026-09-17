"""Offline class-typicality audit tests."""

import csv
import json

import numpy as np
import pytest

from eegproc.model_explainability.typicality.class_awareness import (
    build_class_typicality_audit,
    correct_class_one_mask,
    stratified_sample_indices,
)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _study(root, held_discrepancy=2.5):
    _write_json(root / "study.json", {"task": "valence", "folds": [{"subject_id": 0}]})
    _write_json(
        root / "subject_0/fold.json",
        {"subject_id": 0, "status": "completed", "threshold": 99.0, "eligible_trial_ids": [10]},
    )
    calibration = root / "subject_0/calibration"
    calibration.mkdir(parents=True)
    np.savez_compressed(
        calibration / "source_trials.npz",
        subject_ids=np.array([1, 1, 1, 2, 2, 2]),
        trial_ids=np.array([0, 1, 2, 0, 1, 2]),
        labels=np.array([1, 1, 1, 1, 1, 0]),
        probabilities=np.array(
            [[0.1, 0.9], [0.2, 0.8], [0.8, 0.2], [0.4, 0.6], [0.3, 0.7], [0.9, 0.1]]
        ),
        discrepancy=np.array([1.0, 2.0, 90.0, 3.0, 4.0, 0.1]),
    )
    np.savez_compressed(
        root / "subject_0/observations.npz",
        trial_ids=np.array([0, 1, 2]),
        labels=np.array([1, 1, 0]),
        probabilities=np.array([[0.2, 0.8], [0.7, 0.3], [0.9, 0.1]]),
        discrepancy=np.array([held_discrepancy, 5.0, 8.0]),
    )
    for objective in ("base", "typicality"):
        _write_json(
            root / f"subject_0/trial_10/{objective}/attempt_0001/result.json",
            {
                "status": "completed",
                "report_output": "joint",
                "typicality": {
                    "original_discrepancy": 5.0,
                    "counterfactual_discrepancy": 0.5,
                    "typical": False,
                },
                "latent_counterfactual": {"success": True},
            },
        )
    return root


def _csv_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_correct_class_one_requires_truth_and_prediction():
    labels = [1, 1, 0, 0]
    probabilities = [[0.1, 0.9], [0.9, 0.1], [0.1, 0.9], [0.9, 0.1]]
    np.testing.assert_array_equal(
        correct_class_one_mask(labels, probabilities),
        [True, False, False, False],
    )


def test_sampling_is_deterministic_stratified_and_row_order_invariant():
    subjects = np.array([2, 1, 2, 1, 1, 2])
    trials = np.array([2, 2, 0, 0, 1, 1])
    eligible = np.ones(6, dtype=bool)
    first, _ = stratified_sample_indices(
        subjects, trials, eligible,
        samples_per_subject=2, seed=7, fold_subject=0, role="source",
    )
    order = np.array([3, 0, 5, 2, 1, 4])
    second, _ = stratified_sample_indices(
        subjects[order], trials[order], eligible[order],
        samples_per_subject=2, seed=7, fold_subject=0, role="source",
    )
    first_keys = sorted(zip(subjects[first].tolist(), trials[first].tolist()))
    second_keys = sorted(zip(subjects[order][second].tolist(), trials[order][second].tolist()))
    assert first_keys == second_keys
    assert {subject for subject, _ in first_keys} == {1, 2}
    assert len(first_keys) == 4


def test_audit_recomputes_correct1_threshold_and_counterfactual_membership(tmp_path):
    root = _study(tmp_path / "study")
    payload = build_class_typicality_audit(
        [root], tmp_path / "audit",
        samples_per_source_subject=1,
        samples_per_target_subject=1,
        seed=12,
        quantile=0.95,
    )
    assert payload["definition"] == "true_class == 1 and argmax(probabilities) == 1"
    fold_rows = _csv_rows(tmp_path / "audit/class_typicality_folds.csv")
    assert len(fold_rows) == 2
    assert all(float(row["generation_tau_all_true1"]) == 99.0 for row in fold_rows)
    assert all(float(row["audit_tau_correct1_all_available"]) <= 4.0 for row in fold_rows)
    counterfactuals = _csv_rows(tmp_path / "audit/class_typicality_counterfactuals.csv")
    assert all(row["transition"] == "entered" for row in counterfactuals)
    assert all(row["audit_typicality_success"] == "True" for row in counterfactuals)
    assert all(row["generation_typical"] == "False" for row in counterfactuals)
    audit = json.loads((tmp_path / "audit/class_typicality_audit.json").read_text())
    assert audit["aggregate"][0]["counterfactual_typicality_success_macro_percent"] == 100.0


def test_heldout_scores_cannot_change_source_audit_threshold(tmp_path):
    first = _study(tmp_path / "first", held_discrepancy=0.1)
    second = _study(tmp_path / "second", held_discrepancy=1000.0)
    build_class_typicality_audit(
        [first], tmp_path / "first_out", samples_per_source_subject=1, seed=4
    )
    build_class_typicality_audit(
        [second], tmp_path / "second_out", samples_per_source_subject=1, seed=4
    )
    first_tau = _csv_rows(tmp_path / "first_out/class_typicality_folds.csv")[0][
        "audit_tau_correct1_sampled"
    ]
    second_tau = _csv_rows(tmp_path / "second_out/class_typicality_folds.csv")[0][
        "audit_tau_correct1_sampled"
    ]
    assert first_tau == second_tau


def test_plain_counterfactual_directory_is_rejected(tmp_path):
    legacy = tmp_path / "counterfactual_only"
    legacy.mkdir()
    with pytest.raises(ValueError, match="typicality.runner"):
        build_class_typicality_audit([legacy], tmp_path / "out")
