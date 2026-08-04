"""Load ``nested_lnso_cv`` result JSON into tidy DataFrames for reporting.

``nested_lnso_cv`` (see :mod:`eegproc.deep_learning.cross_val`) returns a nested
dict of per-fold metrics, per-subject metrics and per-window prediction logs.
The result is saved to JSON in one of two shapes:

* **single model** — the result dict itself (``fold_metrics``, ``user_metrics``,
  ``prediction_log`` ... as top-level keys), or
* **multi model** — ``{model_name: result_dict, ...}`` (e.g.
  ``all_smoke_tests_results.json``).

:func:`load_results` accepts either shape and flattens everything into a small
set of tidy pandas DataFrames (one ``model`` column identifies the source), so
the plotting functions in :mod:`eegproc.plotting.result_figures` only ever deal
with DataFrames.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Keys that identify a single-model result dict. If a JSON's top level contains
# any of these, it is a single model; otherwise each top-level key is a model.
_RESULT_KEYS = frozenset(
    {
        "fold_metrics",
        "user_metrics",
        "prediction_log",
        "variational_interval_log",
        "best_configs",
        "inner_cv_results",
        "outer_fold_results",
        "mean_scores",
    }
)


@dataclass(frozen=True)
class ResultsTables:
    """Tidy DataFrames extracted from one or more CV result dicts.

    Every DataFrame carries a ``model`` column. Tables that are not present in
    the source JSON come back empty (with no rows), so callers can check
    ``df.empty`` rather than catching ``KeyError``.
    """

    models: list[str]
    predictions: pd.DataFrame
    user_metrics: pd.DataFrame
    fold_metrics: pd.DataFrame
    best_configs: pd.DataFrame
    inner_cv: pd.DataFrame
    intervals: pd.DataFrame
    summary: pd.DataFrame
    subject_id_mapping: dict[str, dict]

    @property
    def has_tasks(self) -> bool:
        """True when prediction rows carry a ``task_id`` (per-task reporting)."""
        return "task_id" in self.predictions.columns and not self.predictions.empty


def class_probability_columns(df: pd.DataFrame) -> list[str]:
    """Return the ``p_class_<i>`` columns of a predictions DataFrame, in order."""
    cols = [c for c in df.columns if c.startswith("p_class_") and c[8:].isdigit()]
    return sorted(cols, key=lambda c: int(c.split("_")[-1]))


def _normalize_models(raw: dict) -> dict[str, dict]:
    """Return a ``{model_name: result_dict}`` mapping for either JSON shape."""
    if not isinstance(raw, dict):
        return {}

    if _RESULT_KEYS & raw.keys():
        return {"model": raw}

    if "loso_cv" in raw and isinstance(raw.get("loso_cv"), dict):
        training_summary = dict(raw)
        loso_cv = dict(raw["loso_cv"])
        loso_cv.setdefault("selected_final_config", raw.get("selected_final_config"))
        loso_cv.setdefault("selected_final_epochs", raw.get("selected_final_epochs"))
        loso_cv.setdefault("selected_final_batch_size", raw.get("selected_final_batch_size"))
        loso_cv.setdefault("final_full_dataset_metrics", raw.get("final_full_dataset_metrics"))
        loso_cv.setdefault("run_dir", raw.get("run_dir"))
        loso_cv.setdefault("encoder_type", raw.get("encoder_type"))
        loso_cv.setdefault("n_channels", raw.get("n_channels"))
        loso_cv.setdefault("n_bands", raw.get("n_bands"))
        loso_cv.setdefault("training_summary", training_summary)
        return {"model": loso_cv}

    return {name: result for name, result in raw.items() if isinstance(result, dict)}


def _normalize_value(value):
    """Convert nested containers to JSON-safe scalars for pandas plotting tables."""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def _rows_to_frame(rows: list[dict], model: str) -> pd.DataFrame:
    """Build a DataFrame from a list of log rows, prepending a ``model`` column."""
    if not rows:
        return pd.DataFrame()

    normalized_rows = [{k: _normalize_value(v) for k, v in row.items()} for row in rows]
    frame = pd.DataFrame(normalized_rows)
    frame.insert(0, "model", model)
    return frame


def _flatten_configs(records: list[dict], model: str, config_key: str) -> pd.DataFrame:
    """Flatten records whose ``config_key`` holds a hyperparameter dict.

    The hyperparameter dict is expanded into one column per hyperparameter; all
    other scalar fields on the record are kept as-is.
    """
    flat_rows: list[dict] = []

    for record in records:
        config = record.get(config_key, {}) or {}
        row = {"model": model}
        row.update({k: _normalize_value(v) for k, v in record.items() if k != config_key})
        row.update({k: _normalize_value(v) for k, v in config.items()})
        flat_rows.append(row)

    return pd.DataFrame(flat_rows)


def _flatten_config_results(result: dict, model: str) -> pd.DataFrame:
    """Flatten ``config_results`` from the flat LOSO JSON into sweepable rows."""
    rows: list[dict] = []

    for item in result.get("config_results", []) or []:
        config = item.get("config", {}) or {}
        row = {"model": model, "config_index": item.get("config_index")}
        row.update({k: _normalize_value(v) for k, v in item.items() if k not in {"config", "config_index"}})

        metrics = item.get("mean_scores") or item.get("trial_mean_scores") or item.get("window_mean_scores") or {}
        row.update({k: _normalize_value(v) for k, v in metrics.items()})
        row.update({k: _normalize_value(v) for k, v in config.items()})

        rows.append(row)

    return pd.DataFrame(rows)


def _inner_cv_frame(result: dict, model: str) -> pd.DataFrame:
    """Flatten per-config inner-CV validation scores.

    One row per (outer_fold, config). Hyperparameters are expanded into columns
    and the validation metrics (``loss``, ``accuracy`` ...) are kept. This is the
    correct source for hyperparameter-vs-accuracy plots because the scores come
    from inner validation folds, never the held-out test set.
    """
    rows: list[dict] = []

    config_results = _flatten_config_results(result, model)
    if not config_results.empty:
        return config_results

    for outer in result.get("inner_cv_results", []):
        outer_fold = outer.get("outer_fold")
        for config_scores in outer.get("inner_mean_scores", []):
            config = config_scores.get("config", {}) or {}
            row = {"model": model, "outer_fold": outer_fold}
            row.update(
                {
                    k: _normalize_value(v)
                    for k, v in config_scores.items()
                    if k != "config"
                }
            )
            row.update({k: _normalize_value(v) for k, v in config.items()})
            rows.append(row)

    return pd.DataFrame(rows)


def _summary_frame(result: dict, model: str) -> pd.DataFrame:
    """Combine ``mean_scores`` / ``std_scores`` into a long mean±std table."""
    if result.get("mean_scores"):
        mean_scores = result.get("mean_scores", {}) or {}
        std_scores = result.get("std_scores", {}) or {}
    elif result.get("final_full_dataset_metrics"):
        mean_scores = result.get("final_full_dataset_metrics", {}) or {}
        std_scores = {}
    else:
        mean_scores = result.get("trial_mean_scores") or result.get("window_mean_scores") or {}
        std_scores = result.get("trial_std_scores") or result.get("window_std_scores") or {}

    rows = [
        {
            "model": model,
            "metric": metric,
            "mean": mean_value,
            "std": std_scores.get(metric, float("nan")),
        }
        for metric, mean_value in mean_scores.items()
    ]

    return pd.DataFrame(rows)


def _attach_subject_labels(
    predictions: pd.DataFrame,
    subject_id_mapping: dict[str, dict],
) -> pd.DataFrame:
    """Add a ``subject_label`` column mapping integer codes to original ids."""
    if predictions.empty or "subject_id" not in predictions.columns:
        return predictions

    # Build a per-(model, subject_id) lookup table and join it onto predictions.
    mapping_rows: list[dict] = []
    for model, mapping in subject_id_mapping.items():
        if not mapping:
            continue
        for code, label in mapping.items():
            mapping_rows.append(
                {"model": model, "subject_id_str": str(code), "subject_label": label}
            )

    predictions = predictions.copy()
    predictions["subject_id_str"] = predictions["subject_id"].astype(str)

    if mapping_rows:
        mapping_df = pd.DataFrame(mapping_rows)
        predictions = predictions.merge(
            mapping_df,
            on=["model", "subject_id_str"],
            how="left",
        )
        predictions["subject_label"] = predictions["subject_label"].fillna(
            predictions["subject_id"]
        )
    else:
        predictions["subject_label"] = predictions["subject_id"]

    return predictions.drop(columns=["subject_id_str"])


def load_results(path: str | Path) -> ResultsTables:
    """Load a CV result JSON into tidy DataFrames.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a JSON file written by the smoke test / cross-validation run.

    Returns
    -------
    ResultsTables
        Tidy tables (predictions, per-user metrics, per-fold metrics, best
        configs, inner-CV scores, variational intervals, mean±std summary) plus
        the per-model subject id mapping.
    """
    path = Path(path)

    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)

    models = _normalize_models(raw)

    predictions_parts: list[pd.DataFrame] = []
    user_metrics_parts: list[pd.DataFrame] = []
    fold_metrics_parts: list[pd.DataFrame] = []
    best_config_parts: list[pd.DataFrame] = []
    inner_cv_parts: list[pd.DataFrame] = []
    interval_parts: list[pd.DataFrame] = []
    summary_parts: list[pd.DataFrame] = []
    subject_id_mapping: dict[str, dict] = {}

    for model, result in models.items():
        subject_id_mapping[model] = result.get("subject_id_mapping", {}) or {}

        if result.get("prediction_log"):
            predictions_parts.append(_rows_to_frame(result["prediction_log"], model))
        if result.get("user_metrics"):
            user_metrics_parts.append(_rows_to_frame(result["user_metrics"], model))
        if result.get("fold_metrics"):
            fold_metrics_parts.append(_rows_to_frame(result["fold_metrics"], model))
        if result.get("variational_interval_log"):
            interval_parts.append(
                _rows_to_frame(result["variational_interval_log"], model)
            )
        if result.get("best_configs"):
            best_config_parts.append(
                _flatten_configs(result["best_configs"], model, "best_config")
            )

        inner_cv_parts.append(_inner_cv_frame(result, model))
        summary_parts.append(_summary_frame(result, model))

    predictions = _concat(predictions_parts)
    predictions = _attach_subject_labels(predictions, subject_id_mapping)

    return ResultsTables(
        models=list(models.keys()),
        predictions=predictions,
        user_metrics=_concat(user_metrics_parts),
        fold_metrics=_concat(fold_metrics_parts),
        best_configs=_concat(best_config_parts),
        inner_cv=_concat(inner_cv_parts),
        intervals=_concat(interval_parts),
        summary=_concat(summary_parts),
        subject_id_mapping=subject_id_mapping,
    )


def _concat(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate frames, returning an empty frame when there is nothing."""
    parts = [part for part in parts if part is not None and not part.empty]

    if not parts:
        return pd.DataFrame()

    return pd.concat(parts, ignore_index=True)
