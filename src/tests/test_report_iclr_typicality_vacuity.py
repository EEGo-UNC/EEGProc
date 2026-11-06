"""Typicality vacuity report: counts from archived trial_metrics.csv files."""

import csv
import json

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
