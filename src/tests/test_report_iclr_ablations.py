"""Archived four-arm reporting works without loading the EEG model."""

import csv
import json

import pytest

from eegproc.model_explainability.report_iclr_ablations import build_report


OBJECTIVES = ("target_latent", "base", "typicality", "typicality_no_physiology")


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _study(root, task, *, protocol="latent_only", absent_check=False):
    shard = root / task / "fold_00"
    _json(shard / "study.json", {
        "task": task, "folds": [{"subject_id": 0}], "objectives": OBJECTIVES,
        "round_trip_evaluation": protocol,
        "arguments": {"target_probability": 0.8},
        "typicality_definition": "test_score", "typicality_representation": "test_embedding",
    })
    _json(shard / "subject_0/fold.json", {
        "subject_id": 0, "status": "completed", "eligible_trial_ids": [2, 3],
    })
    for trial in (2, 3):
        for objective in OBJECTIVES:
            attempt = shard / f"subject_0/trial_{trial}/{objective}/attempt_0001"
            # One failed optimization stays in the denominator and has no distance.
            if trial == 3 and objective == "base":
                _json(attempt / "result.json", {"status": "error", "error": "optimization failed"})
                continue
            typical = objective.startswith("typicality")
            latent = {"predicted_class": 1, "target_probability": 0.54 if trial == 2 else 0.85,
                      "success": trial == 3}
            decoded_cf = {"predicted_class": 0 if trial == 2 else 1,
                          "target_probability": 0.4 if trial == 2 else 0.85,
                          "success": trial == 3, "typical": typical}
            result = {
                "status": "completed", "task": task, "subject_id": 0,
                "trial_id": trial, "objective": objective,
                "report_output": "joint", "target_class": 1,
                "required_target_probability": 0.8, "latent_counterfactual": latent,
                "typicality": {"typical": typical}, "d_z": float(trial),
                "decoded_trials": {"joint": {"decoded_change_mse": 4.0,
                                              "vcsc_counterfactual": 0.0 if trial == 2 else 0.2,
                                              "counterfactual": decoded_cf}},
                "physiological_tolerance": 1e-8,
                "physiology": {"all_required_passed": None if absent_check else trial == 3,
                               "available_checks_passed": trial == 3,
                               "available_count": 4, "required_count": 5},
            }
            _json(attempt / "result.json", result)
            _json(attempt / "complete.json", {"sha256": {}})
    return shard


def test_four_ablations_report_each_user_and_preserve_metric_space(tmp_path):
    root = tmp_path / "final-ICLR"
    _study(root, "valence", absent_check=True)
    _study(root, "arousal", absent_check=True)
    report = build_report(root, tmp_path / "report", expected_subjects=1)
    assert report["complete"]
    assert report["validity_protocol"] == "latent_only"
    assert len(report["population"]) == 8
    valence = [row for row in report["population"] if row["task"] == "valence"]
    assert [row["objective"] for row in valence] == list(OBJECTIVES)
    base = valence[1]
    assert base["n_eligible"] == 2
    assert base["n_error"] == 1
    assert base["flip_percent"] == 50
    assert base["confidence_acquired_percent"] == 0
    assert base["vcsc_passed_percent"] == 50
    assert base["d_z_n"] == 1
    assert base["d_z_median"] == 2
    assert base["physiological_passed_percent"] == 0
    assert valence[2]["physiological_passed_percent"] is None
    assert valence[2]["flip_typical_percent"] == 100
    assert valence[2]["confident_flip_typical_percent"] == 50
    assert valence[2]["available_checks_passed_percent"] == 50
    assert valence[3]["flip_typical_percent"] == 100
    assert valence[3]["confident_flip_typical_percent"] == 50
    latex = (tmp_path / "report/counterfactual_results.tex").read_text()
    assert "Phys. requires all five" in latex
    assert "VCSC" in latex
    assert "Flip (\\%)" not in latex and "Typ. (\\%)" not in latex
    assert r"\lambda_{\mathrm{phys}}=0" in latex
    assert "--" in latex
    assert (tmp_path / "report/users/valence_user_0.md").is_file()
    with (tmp_path / "report/user_optimizations.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 8


def test_round_trip_validity_is_not_taken_from_latent_success(tmp_path):
    root = tmp_path / "runs"
    _study(root, "valence", protocol="full_trial_decoder_encoder_v1")
    report = build_report(root, tmp_path / "report", expected_subjects=1)
    row = report["population"][0]
    assert row["flip_percent"] == 50
    assert row["confidence_acquired_percent"] == 50
    trial_rows = list(csv.DictReader((tmp_path / "report/trial_optimizations.csv").open()))
    first = next(row for row in trial_rows if row["task"] == "valence" and
                 row["trial_id"] == "2" and row["objective"] == "target_latent")
    assert first["flip"] == "False"
    assert report["validity_protocol"] == "full_trial_decoder_encoder_v1"


def test_rejects_missing_arm_and_duplicate_fold(tmp_path):
    root = tmp_path / "runs"
    shard = _study(root, "valence")
    study = json.loads((shard / "study.json").read_text())
    study["objectives"] = list(OBJECTIVES[:-1])
    _json(shard / "study.json", study)
    with pytest.raises(ValueError, match="all four ablations"):
        build_report(root, tmp_path / "report", expected_subjects=1)
    study["objectives"] = list(OBJECTIVES)
    _json(shard / "study.json", study)
    second = root / "valence/fold_01"
    _json(second / "study.json", study)
    with pytest.raises(ValueError, match="Duplicate task/subject"):
        build_report(root, tmp_path / "report", expected_subjects=1)


def test_partial_archive_keeps_population_table_blank(tmp_path):
    root = tmp_path / "runs"
    _study(root, "valence")
    report = build_report(root, tmp_path / "report", expected_subjects=2)
    assert not report["complete"]
    assert report["missing_subjects"] == {"valence": [1], "arousal": [0, 1]}
    assert report["population"][0]["provisional"]
    assert report["population"][0]["flip_percent"] == 100
    assert "Valence & $\\mathcal{L}_{\\mathrm{base}}$ & -- & --" in (
        tmp_path / "report/counterfactual_results.tex").read_text()
    provisional = (tmp_path / "report/counterfactual_results_with_provisional.tex").read_text()
    assert r"Valence$^{\dagger}$" in provisional
    assert "Arousal & $\\mathcal{L}_{\\mathrm{base}}$ & --" in provisional
