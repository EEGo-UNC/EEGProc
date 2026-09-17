"""Tests for eegproc.plotting.results_io.

Uses self-contained synthetic result dicts (so the tests do not depend on any
committed smoke-test output) covering both JSON shapes and the optional
``task_id`` / ``subject_id_mapping`` fields.
"""

import json

import pandas as pd
import pytest

from eegproc.plotting.results_io import (
    ResultsTables,
    class_probability_columns,
    load_results,
)


def make_single_result(with_tasks: bool = True) -> dict:
    """A small but complete single-model result dict (the nested_lnso_cv shape)."""
    task = (lambda i: {"task_id": 10 + (i % 2)}) if with_tasks else (lambda i: {})

    prediction_log = [
        {
            "fold": 1 + i // 2,
            "sample_index": i,
            "subject_id": i // 2,
            **task(i),
            "y_true": i % 2,
            "y_pred": (i % 2) if i != 3 else 0,  # one wrong prediction
            "p_pred": 0.6 + 0.05 * (i % 3),
            "p_class_0": 0.4 - 0.05 * (i % 3),
            "p_class_1": 0.6 + 0.05 * (i % 3),
        }
        for i in range(4)
    ]

    interval_log = [
        {
            "fold": 1,
            "sample_index": i,
            "subject_id": 0,
            **task(i),
            "y_true": i % 2,
            "y_pred": i % 2,
            "p_pred_mean": 0.6,
            "p_pred_ci_low": 0.5,
            "p_pred_ci_high": 0.7,
            "ci_level": 0.95,
            "n_uncertainty_samples": 30,
        }
        for i in range(2)
    ]

    # 2 outer folds x (2 learning rates x 2 lstm widths) inner configs.
    grid = [
        {"learning_rate": lr, "lstm_units": units}
        for lr in (0.01, 0.001)
        for units in (4, 8)
    ]
    inner_cv_results = [
        {
            "outer_fold": fold,
            "inner_mean_scores": [
                {
                    "config_index": idx,
                    "config": {"epochs": 1, "batch_size": 2, **cfg},
                    "loss": 0.6 - 0.01 * idx,
                    "accuracy": 0.5 + 0.05 * idx,
                    "f1": 0.5,
                    "precision": 0.5,
                    "recall": 0.5,
                }
                for idx, cfg in enumerate(grid)
            ],
        }
        for fold in (1, 2)
    ]

    return {
        "fold_metrics": [
            {"fold": 1, "n_samples": 2, "loss": 0.55, "accuracy": 1.0, "f1": 1.0, "precision": 1.0, "recall": 1.0},
            {"fold": 2, "n_samples": 2, "loss": 0.70, "accuracy": 0.5, "f1": 0.0, "precision": 0.0, "recall": 0.0},
        ],
        "user_metrics": [
            {"fold": 1, "subject_id": 0, "n_samples": 2, "accuracy": 1.0, "f1": 1.0, "precision": 1.0, "recall": 1.0},
            {"fold": 2, "subject_id": 1, "n_samples": 2, "accuracy": 0.5, "f1": 0.0, "precision": 0.0, "recall": 0.0},
        ],
        "prediction_log": prediction_log,
        "variational_interval_log": interval_log,
        "best_configs": [
            {"outer_fold": 1, "best_config_index": 3, "best_config": {"epochs": 1, "batch_size": 2, "learning_rate": 0.001, "lstm_units": 8}, "selection_metric": "accuracy", "selection_score": 0.65},
        ],
        "inner_cv_results": inner_cv_results,
        "mean_scores": {"loss": 0.625, "accuracy": 0.75, "f1": 0.5, "precision": 0.5, "recall": 0.5},
        "std_scores": {"loss": 0.075, "accuracy": 0.25, "f1": 0.5, "precision": 0.5, "recall": 0.5},
        "subject_id_mapping": {"0": 101, "1": 202},
    }


def make_multi_result() -> dict:
    return {"model_a": make_single_result(), "model_b": make_single_result(with_tasks=False)}


def _write(tmp_path, payload: dict):
    path = tmp_path / "results.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_loads_single_model_shape(tmp_path):
    tables = load_results(_write(tmp_path, make_single_result()))

    assert isinstance(tables, ResultsTables)
    assert tables.models == ["model"]
    assert tables.has_tasks
    # one row per prediction, with model + subject_label columns added
    assert len(tables.predictions) == 4
    assert {"model", "subject_label", "task_id"}.issubset(tables.predictions.columns)
    # subject_id 0 mapped to original id 101
    assert (tables.predictions.loc[tables.predictions["subject_id"] == 0, "subject_label"] == 101).all()


def test_loads_multi_model_shape(tmp_path):
    tables = load_results(_write(tmp_path, make_multi_result()))

    assert set(tables.models) == {"model_a", "model_b"}
    assert set(tables.predictions["model"].unique()) == {"model_a", "model_b"}
    # inner_cv flattens hyperparameters into columns for both models
    assert {"learning_rate", "lstm_units", "accuracy"}.issubset(tables.inner_cv.columns)
    # summary has one row per (model, metric)
    assert len(tables.summary) == 2 * 5


def test_missing_tables_come_back_empty(tmp_path):
    payload = {"fold_metrics": [{"fold": 1, "accuracy": 1.0}], "mean_scores": {}}
    tables = load_results(_write(tmp_path, payload))

    assert not tables.fold_metrics.empty
    assert tables.predictions.empty
    assert tables.intervals.empty
    assert tables.has_tasks is False


def test_loads_flat_loso_shape(tmp_path):
    payload = {
        "best_config_index": 0,
        "config_results": [
            {
                "config_index": 0,
                "config": {"learning_rate": 0.001},
                "trial_mean_scores": {"accuracy": 0.75, "f1": 0.6},
                "trial_std_scores": {"accuracy": 0.1, "f1": 0.1},
                "fold_metrics": [
                    {"fold": 1, "accuracy": 0.7, "f1": 0.6},
                    {"fold": 2, "accuracy": 0.8, "f1": 0.7},
                ],
            }
        ],
        "user_metrics": [{"fold": 1, "subject_id": 0, "accuracy": 0.7}],
        "window_prediction_log": [
            {"fold": 1, "subject_id": 0, "y_true": 0, "y_pred": 0, "p_class_0": 0.9, "p_class_1": 0.1},
            {"fold": 1, "subject_id": 1, "y_true": 1, "y_pred": 1, "p_class_0": 0.2, "p_class_1": 0.8},
        ],
    }

    tables = load_results(_write(tmp_path, payload))

    assert tables.models == ["model"]
    assert not tables.predictions.empty
    assert not tables.fold_metrics.empty
    assert not tables.user_metrics.empty
    assert not tables.summary.empty
    assert {"model", "y_true", "y_pred"}.issubset(tables.predictions.columns)


def test_class_probability_columns_order():
    df = pd.DataFrame(columns=["p_class_1", "p_class_0", "p_class_10", "p_pred", "other"])
    assert class_probability_columns(df) == ["p_class_0", "p_class_1", "p_class_10"]
