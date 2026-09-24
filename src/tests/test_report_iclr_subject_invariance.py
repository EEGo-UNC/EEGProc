"""Offline real class-1 subject-invariance analysis."""

import csv
import json

import numpy as np
import pytest

from eegproc.model_explainability.report_iclr_subject_invariance import (
    REPRESENTATION,
    SCORE_DEFINITION,
    TypicalityRegion,
    build_subject_invariance_report,
    main,
)


def _make_fold(root, held_subject, *, prior_mean=0.0):
    fold_dir = root / f"subject_{held_subject}"
    calibration = fold_dir / "calibration"
    calibration.mkdir(parents=True)
    source_subjects = np.array([subject for subject in (0, 1, 2) if subject != held_subject])
    source_ids = np.repeat(source_subjects, 2)
    source_trials = np.tile([0, 1], 2)
    source_embeddings = np.array([0.0, 1.0, 2.0, 3.0])[:, None] + prior_mean
    source_labels = np.ones(4, dtype=int)
    (calibration / "region.json").write_text(json.dumps({
        "schema_version": 2, "definition": SCORE_DEFINITION,
        "representation": REPRESENTATION, "held_out_subject": held_subject,
        "source_subject_ids": source_subjects.tolist(), "quantile": 0.95,
        "quantile_method": "higher",
    }))
    np.savez_compressed(calibration / "region.npz", prior_mean=[prior_mean],
                        prior_variance=[1.0], tau=9.0, variance_floor=1e-6)
    region = TypicalityRegion.load(calibration)
    np.savez_compressed(
        calibration / "source_trials.npz",
        subject_ids=source_ids, trial_ids=source_trials, labels=source_labels,
        probabilities=np.array([[0.9, 0.1], [0.1, 0.9], [0.1, 0.9], [0.1, 0.9]]),
        embeddings=source_embeddings, discrepancy=region.score(source_embeddings),
    )
    held_embeddings = np.array([0.5, 3.5, 5.0])[:, None] + prior_mean
    np.savez_compressed(
        fold_dir / "observations.npz",
        trial_ids=np.array([0, 1, 2]), labels=np.array([1, 1, 0]),
        probabilities=np.array([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1]]),
        embeddings=held_embeddings, discrepancy=region.score(held_embeddings),
    )
    (fold_dir / "fold.json").write_text(json.dumps({
        "subject_id": held_subject, "status": "running", "threshold": region.tau,
        "checkpoint": {"stage": "zero_shot_source_model",
                       "source_subject_ids": source_subjects.tolist()},
    }))


def _read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_real_class_one_includes_misclassified_trials_and_compares_within_fold(tmp_path):
    root = tmp_path / "study"
    root.mkdir()
    (root / "study.json").write_text(json.dumps({
        "task": "valence", "folds": [{"subject_id": 0}, {"subject_id": 1}],
        "typicality_definition": SCORE_DEFINITION,
        "typicality_representation": REPRESENTATION,
    }))
    _make_fold(root, 0)
    _make_fold(root, 1, prior_mean=10.0)
    report = build_subject_invariance_report([tmp_path], tmp_path / "out", samples_per_subject=0)
    rows = _read_csv(tmp_path / "out/subject_invariance_trials.csv")
    folds = _read_csv(tmp_path / "out/subject_invariance_folds.csv")
    held = [row for row in rows if row["role"] == "heldout" and row["fold_subject"] == "0"]
    assert len(held) == 2
    assert {row["predicted_class"] for row in held} == {"0", "1"}
    assert sorted(float(row["source_empirical_percentile"]) for row in held) == [25.0, 100.0]
    assert all(row["true_class"] == "1" for row in rows if row["role"] in ("source", "heldout"))
    controls = [row for row in rows if row["role"] == "heldout_class0_negative_control"]
    assert len(controls) == 2
    assert all(row["true_class"] == "0" for row in controls)
    assert all(row["subject_id"] != row["fold_subject"] for row in rows if row["role"] == "source")
    assert float(folds[0]["source_median_discrepancy"]) == pytest.approx(2.5)
    assert float(folds[0]["heldout_median_discrepancy"]) == pytest.approx(6.25)
    assert float(folds[0]["probability_heldout_discrepancy_higher"]) == pytest.approx(0.625)
    assert float(folds[0]["heldout_minus_source_inside_pp"]) == pytest.approx(-50.0)
    assert report["aggregate"][0]["n_folds"] == 2
    assert report["aggregate"][0]["n_comparable_folds"] == 2
    assert report["aggregate"][0]["heldout_source_percentile_subject_median"] == 62.5
    assert report["aggregate"][0]["heldout_coverage_subject_median_percent"] == 50.0
    assert report["aggregate"][0]["heldout_coverage_subject_bootstrap95_low_percent"] == 50.0
    assert (tmp_path / "out/subject_invariance_paragraph.tex").read_text().startswith(
        r"\paragraph{Subject-invariance.}"
    )
    sampling = json.loads((tmp_path / "out/subject_invariance_sampling.json").read_text())
    assert all("n_available_true_class_1" in row for row in sampling)
    assert all("n_available_correct_class_1" not in row for row in sampling)


def test_saved_discrepancies_must_match_learned_distribution(tmp_path):
    root = tmp_path / "study"
    root.mkdir()
    (root / "study.json").write_text(json.dumps({
        "task": "valence", "folds": [{"subject_id": 0}],
        "typicality_definition": SCORE_DEFINITION,
        "typicality_representation": REPRESENTATION,
    }))
    _make_fold(root, 0)
    path = root / "subject_0/observations.npz"
    with np.load(path, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    values["discrepancy"] = values["discrepancy"] + 1.0
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match="discrepancies do not match"):
        build_subject_invariance_report([root], tmp_path / "out", samples_per_subject=0)


def test_compact_json_and_historical_npz_folds_can_be_analyzed_together(tmp_path):
    root = tmp_path / "final-ICLR/valence/fold_00"
    root.mkdir(parents=True)
    (root / "study.json").write_text(json.dumps({
        "task": "valence", "folds": [{"subject_id": 0}, {"subject_id": 1}],
        "typicality_definition": SCORE_DEFINITION,
        "typicality_representation": REPRESENTATION,
    }))
    _make_fold(root, 0)
    _make_fold(root, 1, prior_mean=10.0)
    compact = root / "subject_1"
    region_path = compact / "calibration/region.json"
    metadata = json.loads(region_path.read_text())
    with np.load(compact / "calibration/region.npz", allow_pickle=False) as archive:
        metadata["parameters"] = {name: archive[name].tolist() for name in archive.files}
    region_path.write_text(json.dumps(metadata))
    (compact / "calibration/region.npz").unlink()
    for npz_path in (compact / "calibration/source_trials.npz", compact / "observations.npz"):
        with np.load(npz_path, allow_pickle=False) as archive:
            summary = {name: archive[name].tolist() for name in archive.files if name != "embeddings"}
        npz_path.with_suffix(".json").write_text(json.dumps(summary))
        npz_path.unlink()

    assert main([str(tmp_path / "final-ICLR"), "--samples-per-subject", "0",
                 "--out-dir", str(tmp_path / "out")]) == 0
    report = json.loads((tmp_path / "out/subject_invariance.json").read_text())
    folds = _read_csv(tmp_path / "out/subject_invariance_folds.csv")
    assert [row["scores_recomputed_from_embeddings"] for row in folds] == ["True", "False"]
    assert report["aggregate"][0]["heldout_source_percentile_subject_median"] == 62.5
    assert report["input_hashes"][1]["source_trials_path"].endswith("source_trials.json")
    assert "region_npz_sha256" not in report["input_hashes"][1]
