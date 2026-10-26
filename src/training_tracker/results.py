"""Framework-free result adapters for canonical metrics and EEGProc LOSO output."""
from __future__ import annotations

import math
from typing import Any


def finite(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("A metric must be numeric, not boolean.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Metrics must be finite.")
    return result


def selected_config(result: dict) -> dict:
    configs = result.get("config_results", [])
    index = result.get("best_config_index", 0)
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(configs):
        raise ValueError("result.json has no valid selected configuration metrics.")
    return configs[index]


def metric_parts(metric: str) -> tuple[str, str]:
    for level in ("trial", "window"):
        if metric.startswith(level + "_"):
            return level, metric[len(level) + 1:]
    return "trial", metric


def result_metric(result: dict, metric: str) -> tuple[float, float]:
    if "metrics" in result:
        item = result["metrics"][metric]
        mean, std = finite(item["mean"]), finite(item["std"])
    else:
        level, name = metric_parts(metric)
        config = selected_config(result)
        mean = finite(config[f"{level}_mean_scores"][name])
        std = finite(config[f"{level}_std_scores"][name])
    if std < 0:
        raise ValueError("Metric standard deviation cannot be negative.")
    return mean, std


def subject_metrics(result: dict, metric: str) -> list[dict]:
    """Read actual subject IDs; never substitute fold positions for subjects."""
    canonical = "metrics" in result
    source = result.get("subject_metrics" if canonical else "user_metrics", [])
    level, name = metric_parts(metric)
    rows = []
    for item in source:
        key = metric
        if key not in item and not canonical:
            # Unprefixed EEGProc scores are valid only at their recorded level.
            if item.get("evaluation_level", result.get("evaluation_level")) == level:
                key = name
        if key in item:
            subject = item["subject_id"]
            if subject is None or isinstance(subject, (dict, list, bool)):
                raise ValueError("subject_id must be a string or number.")
            rows.append({"subject_id": str(subject), metric: finite(item[key])})
    ids = [row["subject_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Repeated subject IDs: aggregate repeated evaluations explicitly before plotting.")
    return rows
