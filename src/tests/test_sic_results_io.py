"""Tests for eegproc.plotting.sic_results_io and sic_report.

Mirrors test_result_figures.py: forces the headless Agg backend and asserts
figures reach disk. The synthetic run tree here reproduces the artifact schema
that ``subject_calibration_cv`` writes -- ``overall`` from cross_val.py, the wide
per-subject summary from its ``flat_summary`` block, and the source-split
diagnostics from ``PredictionDiagnostics``.
"""

import csv
import json

import pytest

import matplotlib

matplotlib.use("Agg")  # headless backend for CI
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from eegproc.plotting import sic_report
from eegproc.plotting.sic_results_io import (
    load_sic_run,
    load_sic_runs,
    shot_level_frame,
    subject_frame,
    summary_table,
)

METRICS = ("accuracy", "balanced_accuracy", "brier_score", "ece", "roc_auc")
SHOT_LEVELS = ((3, 6), (6, 3), (12, 3))
N_SUBJECTS = 4
SOURCE_EPOCHS = 5


def _scores(offset):
    return {metric: 0.2 + offset for metric in METRICS}


def _write_csv(path, rows):
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_run(root, label, offset=0.0, with_diagnostics=True, with_commit=True):
    """Write one synthetic SIC run tree and return its sweep-root directory."""
    sweep_root = root / label
    config_dir = sweep_root / f"dreamer_valence_sic_sweep_{label}_20260817_120000"
    config_dir = config_dir / "configuration_0001"
    config_dir.mkdir(parents=True, exist_ok=True)

    if with_commit:
        (sweep_root / "run_commit.txt").write_text(
            f"commit=deadbeef\nrun_label={label}\n", encoding="utf-8"
        )

    levels = {}
    for shots, folds in SHOT_LEVELS:
        paired = _scores(offset)
        calibrated = _scores(offset - 0.01 * shots / 3)
        levels[str(shots)] = {
            "calibration_shots": shots,
            "calibration_folds": folds,
            "paired_zero_shot_mean_scores": paired,
            "paired_zero_shot_std_scores": {m: 0.02 for m in METRICS},
            "calibrated_mean_scores": calibrated,
            "calibrated_std_scores": {m: 0.02 for m in METRICS},
            "delta_mean_scores": {m: calibrated[m] - paired[m] for m in METRICS},
            "delta_std_scores": {m: 0.01 for m in METRICS},
            "delta_definition": "post_calibration_minus_paired_zero_shot",
        }

    overall = {
        "aggregation_unit": "subject",
        "n_subjects": N_SUBJECTS,
        "zero_shot_all_trials_mean_scores": _scores(offset),
        "zero_shot_all_trials_std_scores": {m: 0.03 for m in METRICS},
        "calibration_selection_shots": 12,
        "calibration_levels": levels,
    }
    (config_dir / "sic_overall_metrics.json").write_text(json.dumps(overall))
    (config_dir / "model_config.json").write_text(json.dumps({"vc_beta": 1.5}))

    summary_rows = []
    for subject in range(1, N_SUBJECTS + 1):
        row = {"target_subject": subject}
        row.update({f"zero_shot_all_{k}": v for k, v in _scores(offset).items()})
        for shots, _ in SHOT_LEVELS:
            paired = _scores(offset)
            calibrated = _scores(offset - 0.01 * shots / 3)
            row.update({f"{shots}_shot_paired_zero_{k}": v for k, v in paired.items()})
            row.update({f"{shots}_shot_calibrated_{k}": v for k, v in calibrated.items()})
            row.update(
                {f"{shots}_shot_delta_{k}": calibrated[k] - paired[k] for k in METRICS}
            )
        summary_rows.append(row)
    _write_csv(config_dir / "sic_subject_summary.csv", summary_rows)

    if with_diagnostics:
        diagnostic_rows = [
            {
                "fold": subject,
                "epoch": epoch,
                "split": "train",
                "n_samples": 500,
                "accuracy": 0.55,
                "reported_metric": "brier_score",
                "reported_metric_value": 0.24 + offset,
                "confidence_mean": 0.52 + offset,
                "confidence_std": 0.01,
                "brier_score": 0.24 + offset,
                "true_class_0_fraction": 0.5,
                "predicted_class_0_fraction": 0.55,
                "true_class_1_fraction": 0.5,
                "predicted_class_1_fraction": 0.45,
            }
            for subject in range(1, N_SUBJECTS + 1)
            for epoch in range(1, SOURCE_EPOCHS + 1)
        ]
        _write_csv(config_dir / "sic_prediction_diagnostics.csv", diagnostic_rows)

    return sweep_root


@pytest.fixture
def runs(tmp_path):
    root = tmp_path / "sweep"
    make_run(root, "baseline", offset=0.0)
    make_run(root, "arm_hi", offset=-0.02)
    return load_sic_runs([root / "baseline", root / "arm_hi"])


def test_label_comes_from_run_commit(runs):
    assert [run.label for run in runs] == ["baseline", "arm_hi"]
    assert runs[0].commit["commit"] == "deadbeef"


def test_label_falls_back_to_directory_name(tmp_path):
    root = tmp_path / "sweep"
    make_run(root, "no_commit_file", with_commit=False)
    assert load_sic_run(root / "no_commit_file").label == "no_commit_file"


def test_resolves_from_any_level(tmp_path):
    root = tmp_path / "sweep"
    sweep_root = make_run(root, "baseline")
    timestamped = next(path for path in sweep_root.iterdir() if path.is_dir())
    config_dir = timestamped / "configuration_0001"

    for path in (sweep_root, timestamped, config_dir):
        assert load_sic_run(path).config_dir == config_dir


def test_run_properties(runs):
    assert runs[0].shot_levels == [3, 6, 12]
    assert runs[0].selection_shots == 12
    assert runs[0].n_subjects == N_SUBJECTS


def test_shot_level_frame_includes_zero_shot_anchor(runs):
    frame = shot_level_frame(runs)
    anchors = frame[(frame["shots"] == 0) & (frame["metric"] == "brier_score")]
    # One anchor row per run, drawn from zero_shot_all_trials_mean_scores.
    assert len(anchors) == len(runs)
    assert set(frame["phase"]) == {"zero_shot", "calibrated", "delta"}


def test_delta_matches_calibrated_minus_zero_shot(runs):
    frame = shot_level_frame(runs)
    selected = frame[
        (frame["run"] == "baseline")
        & (frame["shots"] == 12)
        & (frame["metric"] == "brier_score")
    ].set_index("phase")["value"]
    expected = selected["calibrated"] - selected["zero_shot"]
    assert selected["delta"] == pytest.approx(expected)


def test_subject_frame_melts_wide_columns(runs):
    frame = subject_frame(runs, shots=12, phase="calibrated")
    assert len(frame) == len(runs) * N_SUBJECTS * len(METRICS)
    assert set(frame["metric"]) == set(METRICS)


def test_subject_frame_rejects_unknown_phase(runs):
    with pytest.raises(ValueError, match="phase must be one of"):
        subject_frame(runs, shots=12, phase="nonsense")


def test_subject_frame_rejects_missing_shot_level(runs):
    with pytest.raises(KeyError, match="no columns starting with"):
        subject_frame(runs, shots=99, phase="calibrated")


def test_diagnostics_loaded(runs):
    assert len(runs[0].diagnostics) == N_SUBJECTS * SOURCE_EPOCHS
    assert set(runs[0].diagnostics["split"]) == {"train"}


def test_diagnostics_empty_when_not_recorded(tmp_path):
    root = tmp_path / "sweep"
    make_run(root, "no_diags", with_diagnostics=False)
    assert load_sic_run(root / "no_diags").diagnostics.empty


def test_no_oracle_artifact_is_read(runs):
    """The loader must not surface per-epoch held-out-subject metrics.

    Selecting against the target subject would make the study an oracle
    estimate rather than a true LOSOCV one, so those values are deliberately
    never loaded even when a directory happens to contain them.
    """
    assert not hasattr(runs[0], "oracle")


def test_duplicate_labels_rejected(tmp_path):
    root = tmp_path / "sweep"
    make_run(root, "baseline")
    with pytest.raises(ValueError, match="must be unique"):
        load_sic_runs([root / "baseline", root / "baseline"])


def test_mismatched_labels_rejected(tmp_path):
    root = tmp_path / "sweep"
    make_run(root, "baseline")
    with pytest.raises(ValueError, match="must align"):
        load_sic_runs([root / "baseline"], ["a", "b"])


def test_ambiguous_grid_directory_rejected(tmp_path):
    root = tmp_path / "sweep"
    sweep_root = make_run(root, "baseline")
    timestamped = next(path for path in sweep_root.iterdir() if path.is_dir())
    source = timestamped / "configuration_0001"
    duplicate = timestamped / "configuration_0002"
    duplicate.mkdir()
    (duplicate / "sic_overall_metrics.json").write_text(
        (source / "sic_overall_metrics.json").read_text()
    )

    with pytest.raises(ValueError, match="configurations and no usable"):
        load_sic_run(timestamped)


def test_best_hyperparameters_disambiguates_grid(tmp_path):
    root = tmp_path / "sweep"
    sweep_root = make_run(root, "baseline")
    timestamped = next(path for path in sweep_root.iterdir() if path.is_dir())
    source = timestamped / "configuration_0001"
    winner = timestamped / "configuration_0002"
    winner.mkdir()
    (winner / "sic_overall_metrics.json").write_text(
        (source / "sic_overall_metrics.json").read_text()
    )
    (timestamped / "best_hyperparameters.json").write_text(
        json.dumps({"configuration_dir": "configuration_0002"})
    )

    assert load_sic_run(timestamped).config_dir == winner


def test_best_hyperparameters_path_is_project_root_relative(tmp_path):
    """The winner path is recorded relative to the training working directory.

    Training runs from the project root, so ``configuration_dir`` holds a full
    ``runs/.../configuration_0002`` path rather than one relative to the run
    directory, and on a cluster it can be absolute. Resolution must fall back to
    the directory name, which is the part that survives relocation.
    """
    root = tmp_path / "sweep"
    sweep_root = make_run(root, "baseline")
    timestamped = next(path for path in sweep_root.iterdir() if path.is_dir())
    source = timestamped / "configuration_0001"
    winner = timestamped / "configuration_0002"
    winner.mkdir()
    (winner / "sic_overall_metrics.json").write_text(
        (source / "sic_overall_metrics.json").read_text()
    )
    (timestamped / "best_hyperparameters.json").write_text(
        json.dumps(
            {
                "configuration_id": 2,
                "configuration_dir": "runs/smoke/elsewhere/configuration_0002",
            }
        )
    )

    assert load_sic_run(timestamped).config_dir == winner


def test_best_hyperparameters_falls_back_to_configuration_id(tmp_path):
    root = tmp_path / "sweep"
    sweep_root = make_run(root, "baseline")
    timestamped = next(path for path in sweep_root.iterdir() if path.is_dir())
    source = timestamped / "configuration_0001"
    winner = timestamped / "configuration_0002"
    winner.mkdir()
    (winner / "sic_overall_metrics.json").write_text(
        (source / "sic_overall_metrics.json").read_text()
    )
    # No configuration_dir at all; only the id identifies the winner.
    (timestamped / "best_hyperparameters.json").write_text(
        json.dumps({"configuration_id": 2})
    )

    assert load_sic_run(timestamped).config_dir == winner


def test_missing_path_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_sic_run(tmp_path / "absent")


def test_summary_table_shape(runs):
    table = summary_table(runs, ("brier_score", "ece"))
    assert list(table.columns) == ["run", "shots", "phase", "brier_score", "ece"]
    # Every run contributes one zero-shot anchor plus a pair per shot level.
    assert len(table) == len(runs) * (1 + 2 * len(SHOT_LEVELS))


def _assert_saves(fig, tmp_path, name):
    assert isinstance(fig, Figure)
    out = tmp_path / f"{name}.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.exists() and out.stat().st_size > 0


def test_metric_vs_shots_figure(runs, tmp_path):
    _assert_saves(sic_report.plot_metric_vs_shots(runs), tmp_path, "brier")


def test_metric_vs_shots_rejects_unknown_metric(runs):
    with pytest.raises(ValueError, match="not found"):
        sic_report.plot_metric_vs_shots(runs, metric="nonsense")


def test_source_diagnostics_figure(runs, tmp_path):
    _assert_saves(sic_report.plot_source_diagnostics(runs), tmp_path, "diagnostics")


def test_source_diagnostics_requires_data(tmp_path):
    root = tmp_path / "sweep"
    make_run(root, "no_diags", with_diagnostics=False)
    with pytest.raises(ValueError, match="No sic_prediction_diagnostics.csv"):
        sic_report.plot_source_diagnostics(load_sic_runs([root / "no_diags"]))


def test_per_subject_figure(runs, tmp_path):
    _assert_saves(sic_report.plot_per_subject(runs, shots=12), tmp_path, "subjects")


def test_calibration_gain_figure(runs, tmp_path):
    _assert_saves(sic_report.plot_calibration_gain(runs), tmp_path, "gain")


def test_build_report_writes_every_panel(runs, tmp_path):
    written = sic_report.build_report(runs, tmp_path / "report")
    names = {path.name for path in written}
    assert names == {
        "brier_vs_shots.png",
        "ece_vs_shots.png",
        "calibration_gain.png",
        "per_subject_brier.png",
        "source_diagnostics.png",
        "summary.csv",
    }
    assert all(path.exists() and path.stat().st_size > 0 for path in written)


def test_build_report_degrades_without_diagnostics(tmp_path):
    root = tmp_path / "sweep"
    make_run(root, "old_a", with_diagnostics=False)
    make_run(root, "old_b", offset=-0.01, with_diagnostics=False)
    runs = load_sic_runs([root / "old_a", root / "old_b"])

    written = sic_report.build_report(runs, tmp_path / "report")
    names = {path.name for path in written}
    assert "source_diagnostics.png" not in names
    assert "brier_vs_shots.png" in names


def test_cli_main(tmp_path):
    root = tmp_path / "sweep"
    make_run(root, "baseline")
    make_run(root, "arm_lo", offset=0.01)
    out_dir = tmp_path / "cli_report"

    exit_code = sic_report.main(
        [str(root / "baseline"), str(root / "arm_lo"), "--out-dir", str(out_dir)]
    )
    assert exit_code == 0
    assert (out_dir / "summary.csv").exists()
