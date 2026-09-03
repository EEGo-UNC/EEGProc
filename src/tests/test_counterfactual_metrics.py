"""Tests for atomic aggregation of counterfactual array-task metrics."""

import csv
import json

import pytest

from eegproc.model_explainability.aggregate_counterfactual_metrics import (
    build_metrics_payload,
    write_metrics_json,
)


def _write_trial(root, trial_id, *, success, decoded_mse):
    task_directory = root / f"trial_{trial_id:02d}"
    trial_directory = task_directory / f"subject_0_trial_{trial_id}"
    trial_directory.mkdir(parents=True)
    metrics = {
        "subject_id": 0,
        "trial_id": trial_id,
        "decoder_mode": "joint",
        "joint_reconstruction_alpha": 0.3,
        "latent_counterfactual": {
            "success": success,
            "target_probability": 0.8 if success else 0.4,
        },
        "decoded_trials": {
            "joint": {
                "counterfactual": {
                    "success": success,
                    "target_probability": 0.75 if success else 0.35,
                },
                "counterfactual_to_original_mse": decoded_mse,
            }
        },
        "selected_losses": {"decoded": decoded_mse},
        "steps_completed": 2,
        "elapsed_seconds": 1.5,
    }
    (trial_directory / "result.json").write_text(
        json.dumps(metrics),
        encoding="utf-8",
    )
    (task_directory / "settings.json").write_text(
        json.dumps({"arguments": {"decoder_mode": "joint"}}),
        encoding="utf-8",
    )
    with (trial_directory / "history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("step", "decoded", "success"),
        )
        writer.writeheader()
        writer.writerow(
            {"step": 0, "decoded": decoded_mse, "success": str(success)}
        )
    (trial_directory / "counterfactual.npz").touch()


def test_write_metrics_json_preserves_trials_histories_and_aggregates(tmp_path):
    _write_trial(tmp_path, 0, success=True, decoded_mse=0.2)
    _write_trial(tmp_path, 1, success=False, decoded_mse=0.4)

    payload = write_metrics_json(tmp_path, expected_trial_ids=range(2))
    saved = json.loads((tmp_path / "all_metrics.json").read_text(encoding="utf-8"))

    assert payload["complete"] is True
    assert saved["completed_trial_ids"] == [0, 1]
    assert saved["missing_trial_ids"] == []
    assert saved["aggregate"]["latent_success_rate"] == pytest.approx(0.5)
    assert saved["aggregate"]["decoded_success_rate"]["joint"] == pytest.approx(
        0.5
    )
    decoded_summary = saved["aggregate"]["numeric_metric_summaries"][
        "selected_losses.decoded"
    ]
    assert decoded_summary["count"] == 2
    assert decoded_summary["mean"] == pytest.approx(0.3)
    assert saved["trials"][0]["metrics"]["joint_reconstruction_alpha"] == 0.3
    assert saved["trials"][0]["history"][0]["success"] is True
    assert "trial_00/subject_0_trial_0/counterfactual.npz" in saved["trials"][0][
        "artifacts"
    ]


def test_metrics_payload_reports_missing_trials(tmp_path):
    _write_trial(tmp_path, 0, success=True, decoded_mse=0.2)

    payload = build_metrics_payload(tmp_path, expected_trial_ids=range(3))

    assert payload["complete"] is False
    assert payload["completed_trial_ids"] == [0]
    assert payload["missing_trial_ids"] == [1, 2]
