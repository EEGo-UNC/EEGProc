"""Typicality vacuity report: counts from archived trial_metrics.csv files."""

import csv
import json
import pytest

from eegproc.model_explainability import report_iclr_typicality_vacuity as vacuity

COLUMNS = ["task", "subject_id", "trial_id", "objective", "status", "threshold", "original_discrepancy",
           "discrepancy"]


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        writer.writerows(rows)


def test_vacuity_counts_per_task(tmp_path):
    suite = tmp_path / "suite"
    tau = 1.0
    _write(suite / "fold_00/report/trial_metrics.csv", [
        # trial 1: original inside; base inside; typicality never ran (identical to base)
        ["arousal", 0, 1, "base", "completed", tau, 0.9, 0.8],
        ["arousal", 0, 1, "typicality", "completed", tau, 0.9, 0.8],
        ["arousal", 0, 1, "target_latent", "completed", tau, 0.9, 0.95],
        # trial 2: original outside; base outside; typicality brings it inside
        ["arousal", 0, 2, "base", "completed", tau, 1.3, 1.2],
        ["arousal", 0, 2, "typicality", "completed", tau, 1.3, 0.9],
        ["arousal", 0, 2, "target_latent", "completed", tau, 1.3, 1.25],
        # failed attempts are ignored
        ["arousal", 0, 3, "base", "failed", tau, 0.5, 0.5],
    ])
    # The same fold copied into the suite-level report is not double counted.
    _write(suite / "report/trial_metrics.csv", [["arousal", 0, 1, "base", "completed", tau, 0.9, 0.8]])
    _write(suite / "fold_01/report/trial_metrics.csv", [
        ["valence", 1, 4, "base", "completed", 2.0, 1.5, 1.0],
        ["valence", 1, 4, "typicality", "completed", 2.0, 1.5, 0.7],
    ])
    out = tmp_path / "out"
    vacuity.main(["--suite", str(suite), "--out", str(out)])
    report = json.loads((out / "typicality_vacuity.json").read_text())["tasks"]

    arousal = report["arousal"]
    assert arousal["n_trials"] == 2 and arousal["original_inside_count"] == 1
    assert arousal["inside_per_arm"]["base"] == {"fraction": 0.5, "count": 1, "n": 2}
    assert arousal["inside_per_arm"]["typicality"]["count"] == 2
    assert arousal["inside_per_arm"]["target_latent"]["count"] == 1
    moved = arousal["moved_by_typicality"]
    assert moved["outside_after_base"] == 1 and moved["brought_inside_by_typicality"] == 1
    assert moved["typicality_identical_to_base"] == 1
    assert moved["fraction_of_typicality_passes_owed_to_penalty"] == 0.5

    valence = report["valence"]
    assert valence["original_inside"] == 1.0 and valence["moved_by_typicality"]["outside_after_base"] == 0
    rows = list(csv.DictReader((out / "typicality_vacuity.csv").open()))
    assert {r["task"] for r in rows} == {"arousal", "valence"}


def test_bidirectional_shards_are_found_without_pooling_or_double_counting(tmp_path):
    suites = tmp_path / 'suite'
    columns = COLUMNS + ['direction']
    for direction, score in [('0_to_1', .5), ('1_to_0', 2.)]:
        path = suites / 'valence/fold_00' / direction / 'report/trial_metrics.csv'
        path.parent.mkdir(parents=True)
        with path.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerow(['valence', 0, 0, 'base', 'completed', 1., score, score, direction])
    rows, files = vacuity.load_rows([suites])
    result = vacuity.summarize(rows)
    assert len(files) == 2
    assert result['valence']['original_inside_count'] == 1
    assert result['valence/1_to_0']['original_inside_count'] == 0
    assert result['valence/1_to_0']['target_class'] == 0
    assert len(vacuity.load_rows([suites, suites])[0]) == 2

    path = tmp_path / 'conflict.csv'
    _write(path, [['valence', 0, 0, 'base', 'completed', 1., .9, .9]])
    with pytest.raises(ValueError, match='Conflicting occupancy'):
        vacuity.load_rows([suites, path])


def test_zero_threshold_still_counts_membership(tmp_path):
    path = tmp_path / 'zero.csv'
    _write(path, [['arousal', 0, 0, 'base', 'completed', 0., 0., 0.]])
    rows, _ = vacuity.load_rows([path])
    result = vacuity.summarize(rows)['arousal']
    assert result['original_inside_count'] == 1
    assert result['original_margin_median'] is None
