"""Tests for eegproc.plotting.result_figures and eegproc.plotting.report.

Mirrors test_plot.py: forces the headless Agg backend, draws into real figures
and asserts they save to disk. Synthetic result dicts come from test_results_io.
"""

import json

import pytest

import matplotlib

matplotlib.use("Agg")  # headless backend for CI
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from eegproc.plotting import result_figures as rf
from eegproc.plotting.report import build_report
from eegproc.plotting.results_io import load_results

from .test_results_io import make_single_result, make_multi_result


@pytest.fixture
def tables(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(json.dumps(make_single_result()), encoding="utf-8")
    return load_results(path)


def _assert_saves(fig, tmp_path, name):
    assert isinstance(fig, Figure)
    out = tmp_path / f"{name}.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.exists() and out.stat().st_size > 0


def test_confusion_matrix(tables, tmp_path):
    fig = rf.plot_confusion_matrix(tables.predictions, class_names=["low", "high"])
    _assert_saves(fig, tmp_path, "cm")


def test_metric_summary(tables, tmp_path):
    fig = rf.plot_metric_summary(tables.fold_metrics)
    _assert_saves(fig, tmp_path, "summary")


def test_per_subject_accuracy(tables, tmp_path):
    fig = rf.plot_per_subject_accuracy(tables.user_metrics, subject_labels={"0": 101, "1": 202})
    _assert_saves(fig, tmp_path, "subjects")


def test_subject_task_heatmap(tables, tmp_path):
    fig = rf.plot_subject_task_heatmap(tables.predictions)
    _assert_saves(fig, tmp_path, "heatmap")


def test_subject_task_heatmap_requires_task_id(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(json.dumps(make_single_result(with_tasks=False)), encoding="utf-8")
    tables = load_results(path)

    with pytest.raises(ValueError, match="task_id"):
        rf.plot_subject_task_heatmap(tables.predictions)


def test_hyperparameter_sweep_line(tables, tmp_path):
    fig = rf.plot_hyperparameter_sweep(tables.inner_cv, params="learning_rate", metric="accuracy")
    _assert_saves(fig, tmp_path, "sweep_line")


def test_hyperparameter_sweep_heatmap(tables, tmp_path):
    fig = rf.plot_hyperparameter_sweep(
        tables.inner_cv, params=["learning_rate", "lstm_units"], metric="accuracy"
    )
    _assert_saves(fig, tmp_path, "sweep_heatmap")


def test_hyperparameter_sweep_surface(tables, tmp_path):
    fig = rf.plot_hyperparameter_sweep(
        tables.inner_cv, params=["learning_rate", "lstm_units"], metric="accuracy", surface=True
    )
    _assert_saves(fig, tmp_path, "sweep_surface")


def test_hyperparameter_sweep_rejects_unknown_column(tables):
    with pytest.raises(ValueError, match="not found"):
        rf.plot_hyperparameter_sweep(tables.inner_cv, params="does_not_exist")


def test_reliability_and_uncertainty(tables, tmp_path):
    _assert_saves(rf.plot_reliability(tables.predictions), tmp_path, "reliability")
    _assert_saves(rf.plot_uncertainty(tables.intervals), tmp_path, "uncertainty")


def test_build_report_writes_figures_and_tables(tmp_path):
    results_path = tmp_path / "multi.json"
    results_path.write_text(json.dumps(make_multi_result()), encoding="utf-8")
    out_dir = tmp_path / "figs"

    written = build_report(results_path, out_dir, class_names=["low", "high"], formats=("png",))

    assert written, "expected at least one figure to be written"
    for fig_path in written:
        assert fig_path.exists() and fig_path.stat().st_size > 0

    for table in ("metrics_table.csv", "metrics_table.tex", "metrics_table.md"):
        assert (out_dir / table).exists()

    # model_a has tasks (Fig 2 includes the heatmap) and swept hyperparameters (Fig 3).
    names = {p.name for p in out_dir.glob("model_a_*.png")}
    assert "model_a_fig1_headline.png" in names
    assert "model_a_fig3_hyperparameters.png" in names
