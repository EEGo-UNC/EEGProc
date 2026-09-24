"""The decoded EEG comparison must never substitute the optimized latent score."""

import csv
import json

import numpy as np
import pytest

from eegproc.model_explainability.report_iclr_decoded_subject_invariance import (
    build_decoded_subject_invariance_report,
)
from eegproc.model_explainability.report_iclr_subject_invariance import (
    REPRESENTATION, SCORE_DEFINITION,
)
from eegproc.model_explainability.typicality.artifacts import file_sha256


def _archive(tmp_path, *, decoded=True, predicted=1):
    root = tmp_path / "study"
    calibration = root / "subject_0/calibration"
    calibration.mkdir(parents=True)
    (root / "study.json").write_text(json.dumps({
        "task": "arousal", "target_class": 1, "objectives": ["typicality"],
        "folds": [{"subject_id": 0}],
        "typicality_definition": SCORE_DEFINITION,
        "typicality_representation": REPRESENTATION,
    }))
    (calibration / "region.json").write_text(json.dumps({
        "schema_version": 2, "definition": SCORE_DEFINITION,
        "representation": REPRESENTATION, "held_out_subject": 0,
        "source_subject_ids": [1, 2], "quantile": 0.95,
        "quantile_method": "higher",
        "parameters": {"prior_mean": [0], "prior_variance": [1],
                       "tau": 9, "variance_floor": 1e-6},
    }))
    (calibration / "source_trials.json").write_text(json.dumps({
        "subject_ids": [1, 1, 2, 2], "trial_ids": [0, 1, 0, 1],
        "labels": [1, 1, 1, 1], "probabilities": [[0.1, 0.9]] * 4,
        "discrepancy": [0, 1, 4, 9],
    }))
    (root / "subject_0/observations.json").write_text(json.dumps({
        "trial_ids": [0, 1, 2], "labels": [1, 1, 0],
        "probabilities": [[0.9, 0.1], [0.1, 0.9], [0.9, 0.1]],
        "discrepancy": [0.25, 12.25, 25],
    }))
    (root / "subject_0/fold.json").write_text(json.dumps({
        "subject_id": 0, "status": "completed", "threshold": 9,
        "eligible_trial_ids": [2],
        "checkpoint": {"stage": "zero_shot_source_model",
                       "source_subject_ids": [1, 2], "sha256": "checkpoint"},
    }))
    attempt = root / "subject_0/trial_2/typicality/attempt_0001"
    attempt.mkdir(parents=True)
    result = {
        "status": "completed", "task": "arousal", "subject_id": 0,
        "trial_id": 2, "true_class": 0, "objective": "typicality",
        "checkpoint_sha256": "checkpoint", "report_output": "joint",
        "typicality": {"counterfactual_discrepancy": 100.0},
    }
    if decoded:
        result["decoded_counterfactual"] = {
            "input": "decoded_counterfactual_eeg_reencoded_by_frozen_classifier",
            "report_output": "joint", "target_class": 1,
            "classification_embedding": [1.5],
            "probabilities": [0.2, 0.8] if predicted == 1 else [0.8, 0.2],
            "predicted_class": predicted, "discrepancy": 2.25,
            "source_threshold": 9,
        }
    path = attempt / "result.json"
    path.write_text(json.dumps(result))
    (attempt / "complete.json").write_text(json.dumps({
        "schema_version": 1, "sha256": {"result.json": file_sha256(path)},
    }))
    return root, path


def _rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_report_compares_decoded_eeg_and_real_x_to_source_class1(tmp_path):
    root, _ = _archive(tmp_path)
    report = build_decoded_subject_invariance_report([root], tmp_path / "out", samples_per_subject=0)
    fold = _rows(tmp_path / "out/decoded_subject_invariance_folds.csv")[0]
    trials = _rows(tmp_path / "out/decoded_subject_invariance_trials.csv")
    cf = next(row for row in trials if row["role"] == "decoded_counterfactual")
    assert float(cf["discrepancy"]) == 2.25  # latent score in the archive is 100
    assert cf["score_provenance"] == "current_result_json"
    assert float(cf["source_empirical_percentile"]) == 50.0
    assert float(fold["decoded_cf_median_discrepancy"]) == 2.25
    assert float(fold["real_x_median_discrepancy"]) == 6.25
    assert float(fold["source_median_discrepancy"]) == 2.5
    assert fold["n_decoded_class1_flip"] == "1"
    assert report["aggregate"][0]["n_comparable_folds"] == 1
    assert "decoded $R(Z" in (tmp_path / "out/decoded_subject_invariance_paragraph.tex").read_text()


def test_old_latent_only_archive_requires_new_counterfactual_run(tmp_path):
    root, _ = _archive(tmp_path, decoded=False)
    with pytest.raises(ValueError, match="rerun typicality.runner"):
        build_decoded_subject_invariance_report([root], tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_historical_roundtrip_waveform_and_reencoded_embedding_are_usable(tmp_path):
    root, path = _archive(tmp_path, decoded=False)
    result = json.loads(path.read_text())
    result["round_trip_evaluation"] = "full_trial_decoder_encoder_v1"
    result["typicality"]["threshold"] = 9
    result["decoded_trials"] = {"joint": {"counterfactual": {
        "probabilities": [0.2, 0.8], "predicted_class": 1,
        "discrepancy": 2.25,
    }}}
    path.write_text(json.dumps(result))
    signal_path = path.with_name("counterfactual.npz")
    np.savez_compressed(signal_path, x_prime_joint=np.ones((1, 2, 3, 42)),
                        classification_embedding_reencoded_joint=[[1.5]])
    path.with_name("complete.json").write_text(json.dumps({
        "schema_version": 1, "sha256": {
            "result.json": file_sha256(path),
            "counterfactual.npz": file_sha256(signal_path),
        },
    }))
    report = build_decoded_subject_invariance_report([root], tmp_path / "out")
    assert report["aggregate"][0]["n_decoded_class1_flip"] == 1
    cf = next(row for row in _rows(tmp_path / "out/decoded_subject_invariance_trials.csv")
              if row["role"] == "decoded_counterfactual")
    assert float(cf["discrepancy"]) == 2.25
    assert cf["score_provenance"] == "historical_roundtrip_npz"


def test_decoded_nonflip_is_counted_but_not_compared_as_class1(tmp_path):
    root, _ = _archive(tmp_path, predicted=0)
    report = build_decoded_subject_invariance_report([root], tmp_path / "out")
    fold = _rows(tmp_path / "out/decoded_subject_invariance_folds.csv")[0]
    assert fold["n_decoded_nonflip"] == "1"
    assert fold["n_decoded_class1_flip"] == "0"
    assert fold["decoded_cf_median_discrepancy"] == ""
    assert report["aggregate"][0]["n_comparable_folds"] == 0


def test_decoded_score_must_match_reencoded_embedding(tmp_path):
    root, path = _archive(tmp_path)
    result = json.loads(path.read_text())
    result["decoded_counterfactual"]["discrepancy"] = 0.01
    path.write_text(json.dumps(result))
    marker = path.with_name("complete.json")
    marker.write_text(json.dumps({"schema_version": 1, "sha256": {"result.json": file_sha256(path)}}))
    with pytest.raises(ValueError, match="does not match the source class-1 region"):
        build_decoded_subject_invariance_report([root], tmp_path / "out")
