"""Load legacy CV and SIC calibration JSON into tidy reporting DataFrames.

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

SIC's ``subject_results``, ``subject_summary_rows`` and ``overall`` are adapted
to the same tables. Only one stage/shot selection is loaded. ``fold_metrics``
then contains one outer LOSO subject per row, using within-subject calibration
means when appropriate; saved overall means and standard deviations are kept.
Standalone ``sic_overall_metrics.json`` is supported without predictions.
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
        "window_prediction_log",
        "trial_prediction_log",
        "variational_interval_log",
        "best_configs",
        "inner_cv_results",
        "outer_fold_results",
        "mean_scores",
        "config_results",
        "fold_results",
    }
)

_SIC_KEYS = frozenset({"subject_results", "subject_summary_rows", "overall",
                       "zero_shot_all_trials_mean_scores"})
SIC_STAGES = ("zero_shot_all_trials", "zero_shot_paired", "post_calibration")
_SCORE_PREFIX = dict(zip(SIC_STAGES, ("zero_shot_all_trials", "paired_zero_shot", "calibrated")))


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
    stage: str | None = None
    calibration_shots: int | None = None
    aggregation_unit: str = "outer_fold"

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

    if (_RESULT_KEYS | _SIC_KEYS) & raw.keys():
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


def _sic_evaluations(result: dict, stage: str, shots: int | None):
    """Yield the requested nested evaluation blocks, without top-level duplicates."""
    for subject in result.get("subject_results", []) or []:
        subject_id = subject.get("target_subject")
        outer_fold = subject.get("subject_number")
        if stage == "zero_shot_all_trials":
            block = subject.get("zero_shot_all_trials") or {}
            if block:
                yield subject_id, outer_fold, block
        else:
            for level in subject.get("calibration_levels", []) or []:
                if str(level.get("calibration_shots")) != str(shots):
                    continue
                key = "zero_shot" if stage == "zero_shot_paired" else "calibrated"
                for output in level.get("fold_outputs", []) or []:
                    block = output.get(key) or {}
                    if block:
                        yield subject_id, outer_fold, block


def _select_sic_rows(rows: list[dict], stage: str, shots: int | None) -> list[dict]:
    """Select both stage and shot count; never pool distinct evaluations."""
    return [row for row in rows if row.get("stage") == stage and
            (stage == "zero_shot_all_trials" or
             str(row.get("calibration_shots")) == str(shots))]


def _sic_subject_rows(result: dict, stage: str, shots: int | None,
                      evaluations: list) -> list[dict]:
    """Use saved within-subject means before aggregating across subjects.

    Calibration folds are repeated evaluations of the same subject, not
    independent outer folds. Prefer the saved subject summaries; reconstruct
    a mean across that subject's folds only if those summaries are absent.
    """
    scores_by_subject: dict = {}
    folds: dict = {}
    for subject in result.get("subject_results", []) or []:
        sid = subject.get("target_subject")
        folds[sid] = subject.get("subject_number")
        summary = subject.get("subject_summary") or {}
        if stage == "zero_shot_all_trials":
            scores = summary.get("zero_shot_all_trials_scores") or {}
        else:
            level_summary = (summary.get("calibration_levels") or {}).get(str(shots)) or {}
            if not level_summary:
                level_summary = next((level.get("summary") or {} for level in
                    subject.get("calibration_levels", []) or []
                    if str(level.get("calibration_shots")) == str(shots)), {})
            scores = level_summary.get(_SCORE_PREFIX[stage] + "_mean_scores") or {}
        if scores:
            scores_by_subject[sid] = dict(scores)

    flat_prefix = ("zero_shot_all_" if stage == "zero_shot_all_trials" else
                   f"{shots}_shot_{'paired_zero' if stage == 'zero_shot_paired' else 'calibrated'}_")
    for record in result.get("subject_summary_rows", []) or []:
        sid = record.get("target_subject")
        scores = {key[len(flat_prefix):]: value for key, value in record.items()
                  if key.startswith(flat_prefix)}
        if scores:
            scores_by_subject.setdefault(sid, scores)

    fallback: dict = {}
    for sid, fold, block in evaluations:
        folds.setdefault(sid, fold)
        metrics = block.get("trial_fold_metrics") or block.get("fold_metrics") or {}
        if isinstance(metrics, list):
            metrics = metrics[0] if metrics else {}
        fallback.setdefault(sid, []).append(metrics)
    metadata = {"fold", "subject_id", "target_subject", "calibration_fold", "calibration_shots",
                "n_samples", "n_windows", "n_trials", "windows_per_trial", "prediction_latent_samples"}
    for sid, records in fallback.items():
        if sid in scores_by_subject:
            continue
        frame = pd.DataFrame(records).drop(columns=list(metadata), errors="ignore")
        numeric = frame.select_dtypes(include="number")
        if not numeric.empty:
            scores_by_subject[sid] = numeric.mean().to_dict()

    return [{**scores, "subject_id": sid, "target_subject": sid,
             "fold": folds.get(sid, sid), "stage": stage,
             "calibration_shots": shots, "aggregation_unit": "subject"}
            for sid, scores in scores_by_subject.items()]


def _adapt_sic(result: dict, stage: str, shots: int | None) -> dict:
    """Convert exactly one SIC evaluation to the legacy table input shape."""
    overall = result.get("overall") or result
    if stage == "zero_shot_all_trials":
        score_block = overall
    else:
        score_block = (overall.get("calibration_levels") or {}).get(str(shots)) or {}
        # Older SIC summaries expose only the selected calibration level.
        if not score_block and str(overall.get("calibration_selection_shots")) == str(shots):
            score_block = overall
    prefix = _SCORE_PREFIX[stage]
    mean_scores = score_block.get(prefix + "_mean_scores") or {}
    std_scores = score_block.get(prefix + "_std_scores") or {}
    evaluations = list(_sic_evaluations(result, stage, shots))
    subject_rows = _sic_subject_rows(result, stage, shots, evaluations)

    # SIC reports are trial-level. In particular, don't prefer a window log
    # when both logs exist, and don't concatenate nested and top-level copies.
    predictions = _select_sic_rows(result.get("trial_prediction_log") or [], stage, shots)
    intervals = _select_sic_rows(result.get("trial_variational_interval_log") or [], stage, shots)
    if not predictions:
        predictions = [{**row, "stage": stage, "calibration_shots": shots}
                       for _, _, block in evaluations
                       for row in block.get("trial_prediction_log", []) or []]
    if not intervals:
        intervals = [{**row, "stage": stage, "calibration_shots": shots}
                     for _, _, block in evaluations
                     for row in block.get("trial_variational_interval_log", []) or []]
    if not (mean_scores or subject_rows or predictions):
        description = stage + (f" ({shots} shots)" if shots is not None else "")
        raise ValueError(f"No SIC results for {description}. Check --stage and --calibration-shots.")
    if not mean_scores and subject_rows:
        frame = pd.DataFrame(subject_rows)
        metric_columns = [key for key in subject_rows[0] if key not in
                          {"subject_id", "target_subject", "fold", "stage",
                           "calibration_shots", "aggregation_unit"}]
        numeric = frame[metric_columns].select_dtypes(include="number")
        mean_scores = numeric.mean().to_dict()
        std_scores = numeric.std(ddof=0).to_dict()
    return {"trial_prediction_log": predictions, "variational_interval_log": intervals,
            "user_metrics": subject_rows, "fold_metrics": subject_rows,
            "mean_scores": mean_scores, "std_scores": std_scores,
            "subject_id_mapping": result.get("subject_id_mapping") or {}}


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


def _prediction_rows(result: dict) -> list[dict]:
    """Return prediction rows from either the legacy or flat LOSO JSON format."""
    for key in ("prediction_log", "window_prediction_log", "trial_prediction_log"):
        rows = result.get(key)
        if rows:
            return rows
    return []


def _fold_metric_rows(result: dict) -> list[dict]:
    """Return fold-metric rows from either the legacy or flat LOSO JSON format."""
    if result.get("fold_metrics"):
        return result["fold_metrics"]

    config_results = result.get("config_results") or []
    if not config_results:
        return []

    best_config_index = result.get("best_config_index", 0)
    if not isinstance(best_config_index, int):
        best_config_index = 0
    if best_config_index < 0:
        best_config_index = 0

    selected_config = config_results[best_config_index] if best_config_index < len(config_results) else config_results[0]
    fold_metrics = selected_config.get("fold_metrics") or []
    return fold_metrics


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

        if not mean_scores:
            config_results = result.get("config_results") or []
            if config_results:
                best_config_index = result.get("best_config_index", 0)
                if not isinstance(best_config_index, int):
                    best_config_index = 0
                if best_config_index < 0:
                    best_config_index = 0
                selected_config = (
                    config_results[best_config_index]
                    if best_config_index < len(config_results)
                    else config_results[0]
                )
                mean_scores = selected_config.get("trial_mean_scores") or selected_config.get("window_mean_scores") or {}
                std_scores = selected_config.get("trial_std_scores") or selected_config.get("window_std_scores") or {}

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


def load_results(path: str | Path, *, stage: str = "zero_shot_all_trials",
                 calibration_shots: int | None = None) -> ResultsTables:
    """Load a CV result JSON into tidy DataFrames.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a JSON file written by the smoke test / cross-validation run.
    stage : str
        SIC evaluation to select. Defaults to strict zero-shot on all trials.
        Legacy CV results without stages retain their existing behavior.
    calibration_shots : int, optional
        Required for ``zero_shot_paired`` and ``post_calibration``. Metrics and
        predictions are always selected using the same stage and shot count.

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
    if not models:
        raise ValueError(f"No CV results found in {path}.")
    if stage not in SIC_STAGES:
        raise ValueError(f"Unknown stage {stage!r}; choose from {SIC_STAGES}.")
    if stage == "zero_shot_all_trials" and calibration_shots is not None:
        raise ValueError("--calibration-shots requires --stage post_calibration or zero_shot_paired.")
    if stage != "zero_shot_all_trials" and (calibration_shots is None or calibration_shots <= 0):
        raise ValueError(f"--stage {stage} requires a positive --calibration-shots.")
    sic_models = [name for name, result in models.items() if _SIC_KEYS & result.keys()]
    if sic_models and len(sic_models) != len(models):
        raise ValueError("Report SIC and legacy CV results separately; their aggregation units differ.")
    if stage != "zero_shot_all_trials" and not sic_models:
        raise ValueError("This file contains no SIC calibration stages.")

    predictions_parts: list[pd.DataFrame] = []
    user_metrics_parts: list[pd.DataFrame] = []
    fold_metrics_parts: list[pd.DataFrame] = []
    best_config_parts: list[pd.DataFrame] = []
    inner_cv_parts: list[pd.DataFrame] = []
    interval_parts: list[pd.DataFrame] = []
    summary_parts: list[pd.DataFrame] = []
    subject_id_mapping: dict[str, dict] = {}

    for model, result in models.items():
        if model in sic_models:
            result = _adapt_sic(result, stage, calibration_shots)
        subject_id_mapping[model] = result.get("subject_id_mapping", {}) or {}

        prediction_rows = _prediction_rows(result)
        if prediction_rows:
            predictions_parts.append(_rows_to_frame(prediction_rows, model))
        if result.get("user_metrics"):
            user_metrics_parts.append(_rows_to_frame(result["user_metrics"], model))

        fold_metric_rows = _fold_metric_rows(result)
        if fold_metric_rows:
            fold_metrics_parts.append(_rows_to_frame(fold_metric_rows, model))
        if result.get("variational_interval_log"):
            interval_parts.append(
                _rows_to_frame(result["variational_interval_log"], model)
            )
        if result.get("best_configs"):
            best_config_parts.append(
                _flatten_configs(result["best_configs"], model, "best_config")
            )

        inner_cv_parts.append(_inner_cv_frame(result, model))
        summary_frame = _summary_frame(result, model)
        if model in sic_models and not summary_frame.empty:
            summary_frame = summary_frame.assign(stage=stage, calibration_shots=calibration_shots,
                                                  aggregation_unit="subject")
        summary_parts.append(summary_frame)

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
        stage=stage if sic_models else None,
        calibration_shots=calibration_shots if sic_models else None,
        aggregation_unit="subject" if sic_models else "outer_fold",
    )


def _concat(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate frames, returning an empty frame when there is nothing."""
    parts = [part for part in parts if part is not None and not part.empty]

    if not parts:
        return pd.DataFrame()

    return pd.concat(parts, ignore_index=True)
