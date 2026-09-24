"""Compare the archived typicality discrepancy, without decoder re-encoding."""

import csv
import json

import numpy as np
import pytest

from eegproc.model_explainability.report_iclr_subject_invariance import (
    REPRESENTATION, SCORE_DEFINITION,
)
from eegproc.model_explainability.report_iclr_typicality_subject_invariance import (
    build_typicality_subject_invariance_report,
)
from eegproc.model_explainability.typicality.artifacts import file_sha256


def _archive(tmp_path):
    root = tmp_path / "incomplete-ICLR/arousal/fold_00"
    fold = root / "subject_0"
    calibration = fold / "calibration"
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
    (fold / "observations.json").write_text(json.dumps({
        "trial_ids": [0, 1, 2], "labels": [1, 1, 0],
        "probabilities": [[0.9, 0.1], [0.1, 0.9], [0.9, 0.1]],
        "discrepancy": [0.25, 12.25, 25],
    }))
    (fold / "fold.json").write_text(json.dumps({
        "subject_id": 0, "status": "running", "threshold": 9,
        "eligible_trial_ids": [2],
        "checkpoint": {"stage": "zero_shot_source_model", "sha256": "checkpoint",
                       "source_subject_ids": [1, 2]},
    }))
    attempt = fold / "trial_2/typicality/attempt_0001"
    attempt.mkdir(parents=True)
    result = {
        "status": "completed", "task": "arousal", "subject_id": 0,
        "trial_id": 2, "true_class": 0, "objective": "typicality",
        "checkpoint_sha256": "checkpoint",
        "typicality_definition": SCORE_DEFINITION,
        "typicality_representation": REPRESENTATION,
        "typicality": {
            "definition": SCORE_DEFINITION, "representation": REPRESENTATION,
            "threshold": 9, "counterfactual_discrepancy": 2.25,
        },
        "latent_counterfactual": {
            "probabilities": [0.2, 0.8], "predicted_class": 1,
        },
    }
    result_path = attempt / "result.json"
    result_path.write_text(json.dumps(result))
    archive = attempt / "counterfactual.npz"
    np.savez_compressed(archive, classification_embedding_prime=[[1.5]],
                        x_prime_joint=np.zeros((1, 2, 3, 42)))
    marker = attempt / "complete.json"
    marker.write_text(json.dumps({"sha256": {"result.json": file_sha256(result_path),
                                             "counterfactual.npz": file_sha256(archive)}}))
    return root, result_path, marker


def _csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_latent_typicality_score_compared_to_real_and_source_without_reencoding(tmp_path):
    root, _, _ = _archive(tmp_path)
    report = build_typicality_subject_invariance_report(
        [tmp_path / "incomplete-ICLR"], tmp_path / "out", samples_per_subject=0,
    )
    fold = _csv(tmp_path / "out/typicality_subject_invariance_folds.csv")[0]
    cf = next(row for row in _csv(tmp_path / "out/typicality_subject_invariance_trials.csv")
              if row["role"] == "typicality_counterfactual")
    assert cf["score_space"] == "optimized_classification_embedding_Zcf"
    assert float(cf["discrepancy"]) == 2.25
    assert cf["score_recomputed_from_archived_Zcf"] == "True"
    assert float(fold["source_typical_median_discrepancy"]) == 2.5
    assert float(fold["real_x_median_discrepancy"]) == 6.25
    assert report["aggregate"][0]["n_comparable_folds"] == 1
    assert report["aggregate"][0]["missing_subject_ids"] == list(range(1, 23))
    assert "do not measure the discrepancy of the decoded waveform" in (
        tmp_path / "out/typicality_subject_invariance_paragraph.tex").read_text()


def test_unfinished_attempt_is_counted_and_not_scored(tmp_path):
    root, result_path, marker = _archive(tmp_path)
    marker.unlink()
    result_path.unlink()
    report = build_typicality_subject_invariance_report([root], tmp_path / "out")
    assert report["aggregate"][0]["n_typicality_pending"] == 1
    assert report["aggregate"][0]["n_comparable_folds"] == 0


def test_archived_optimized_embedding_must_match_saved_score(tmp_path):
    root, result_path, marker = _archive(tmp_path)
    summary = json.loads(result_path.read_text())
    summary["typicality"]["counterfactual_discrepancy"] = 100
    result_path.write_text(json.dumps(summary))
    committed = json.loads(marker.read_text())
    committed["sha256"]["result.json"] = file_sha256(result_path)
    marker.write_text(json.dumps(committed))
    with pytest.raises(ValueError, match=r"saved D\(Zcf\) disagrees"):
        build_typicality_subject_invariance_report([root], tmp_path / "out")
