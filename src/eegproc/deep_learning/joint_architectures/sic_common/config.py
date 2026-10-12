"""TensorFlow-free grid and calibration-selection rules shared by SIC versions."""
from __future__ import annotations

import argparse
from itertools import product
import json


SIC_SEQUENCE_HYPERPARAMETER_DEPTHS = {
    "gcn_units": 1,
    "temporal_pool_sizes": 1,
    "classifier_rnn_units": 1,
    "classification_hidden_units": 1,
    "focal_alpha": 1,
}


SIC_DATA_HYPERPARAMETERS = frozenset({"remove_median_label"})


SIC_SELECTION_OVERALL_KEYS = {
    "losocv": "zero_shot_all_trials_mean_scores",
    "calibration": "calibrated_mean_scores",
}


SIC_CLASSIFICATION_METRICS = (
    "accuracy",
    "f1",
    "precision",
    "recall",
    "macro_f1",
    "macro_precision",
    "macro_recall",
    "balanced_accuracy",
    "roc_auc",
    "brier_score",
    "ece",
)


SIC_MINIMIZE_METRICS = frozenset({"loss", "joint_loss", "brier_score", "ece"})


def configuration_training_method(configuration: dict) -> str:
    """Resolve one expanded configuration, including legacy use_vrex."""
    raw_method = configuration.get("training_method")
    legacy_vrex = bool(configuration.get("use_vrex", False))
    if raw_method is None:
        return "vrex" if legacy_vrex else "erm"

    normalized = str(raw_method).strip().lower().replace("-", "_")
    if normalized not in {"erm", "vrex", "mldg"}:
        raise ValueError(
            "training_method must be one of 'erm', 'vrex', or 'mldg'; "
            f"got {raw_method!r}."
        )
    if legacy_vrex and normalized != "vrex":
        raise ValueError(
            "use_vrex=true conflicts with training_method="
            f"{raw_method!r}. Use training_method='vrex' or remove use_vrex."
        )
    return normalized


def metric_mode(metric_name: str) -> str:
    return "min" if str(metric_name).lower() in SIC_MINIMIZE_METRICS else "max"


def calibration_plan(config) -> tuple[tuple[int, int], ...]:
    return config.calibration_levels or (
        (int(config.calibration_trials), int(config.calibration_folds)),
    )


def selected_calibration_shots(config) -> int:
    levels = calibration_plan(config)
    return (
        max(shots for shots, _ in levels)
        if config.calibration_selection_shots is None
        else int(config.calibration_selection_shots)
    )


def selection_score_source(config) -> str:
    if config.hyperparameter_selection_level == "calibration":
        return (
            "overall.calibration_levels."
            f"{selected_calibration_shots(config)}.calibrated_mean_scores"
        )
    return "overall.zero_shot_all_trials_mean_scores"


def _decode_grid_value(name: str, value):
    """Return ``(is_axis, fixed_value_or_candidates)`` for one JSON entry."""
    if isinstance(value, dict):
        wrappers = {key for key in value if key in {"grid", "fixed"}}
        if wrappers:
            if len(value) != 1 or len(wrappers) != 1:
                raise ValueError(
                    f"Hyperparameter {name!r} must use exactly one grid/fixed wrapper."
                )
            wrapper = next(iter(wrappers))
            decoded = value[wrapper]
            if wrapper == "grid":
                if not isinstance(decoded, list) or not decoded:
                    raise ValueError(f"Hyperparameter {name!r} grid must be non-empty.")
                return True, decoded
            return False, decoded

    if not isinstance(value, list):
        return False, value
    if not value:
        if name in SIC_SEQUENCE_HYPERPARAMETER_DEPTHS:
            return False, value
        raise ValueError(f"Hyperparameter {name!r} cannot be an empty list.")
    if name in SIC_SEQUENCE_HYPERPARAMETER_DEPTHS:
        is_nested = any(isinstance(item, (list, tuple)) for item in value)
        return (True, value) if is_nested else (False, value)
    return True, value


def expand_cartesian_grid(model_config: dict) -> tuple[list[dict], dict[str, int]]:
    """Expand a JSON model configuration into a full Cartesian product."""
    if not isinstance(model_config, dict):
        raise ValueError("Hyperparameter configuration must be a JSON object.")
    fixed = {}
    axes = []
    dimensions = {}
    for name, raw_value in model_config.items():
        is_axis, decoded = _decode_grid_value(name, raw_value)
        if is_axis:
            candidates = list(decoded)
            axes.append((name, candidates))
            dimensions[name] = len(candidates)
        else:
            fixed[name] = decoded

    if not axes:
        return [dict(fixed)], dimensions
    configurations = []
    for selected in product(*(candidates for _, candidates in axes)):
        candidate = dict(fixed)
        candidate.update(
            {name: value for (name, _), value in zip(axes, selected)}
        )
        configurations.append(candidate)
    return configurations, dimensions


def split_data_hyperparameters(configuration: dict) -> tuple[dict, dict]:
    model = {
        name: value
        for name, value in configuration.items()
        if name not in SIC_DATA_HYPERPARAMETERS
    }
    data = {
        name: configuration.get(name, False)
        for name in SIC_DATA_HYPERPARAMETERS
    }
    if not isinstance(data["remove_median_label"], bool):
        raise ValueError(
            "remove_median_label must be true or false; got "
            f"{data['remove_median_label']!r}."
        )
    data["remove_median_label"] = bool(data["remove_median_label"])
    return model, data


def _positive_int(value: str) -> int:
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def calibration_levels_from_args(args) -> tuple[tuple[int, int], ...]:
    return tuple(
        tuple(pair)
        for pair in (
            args.calibration_level
            or [(args.calibration_trials, args.calibration_folds)]
        )
    )


def model_config_from_args(args) -> dict:
    try:
        model_config = (
            json.loads(args.hyperparameters_json) if args.hyperparameters_json else {}
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid --hyperparameters-json: {error}") from error
    if not isinstance(model_config, dict):
        raise ValueError("--hyperparameters-json must decode to a JSON object.")

    if args.training_method is not None:
        model_config["training_method"] = args.training_method
        model_config.pop("use_vrex", None)
    if "training_method" not in model_config and "use_vrex" not in model_config:
        model_config["training_method"] = "erm"
    for key in (
        "use_gcn_gru_branch",
        "use_bilstm_branch",
        "use_decoder",
        "remove_median_label",
    ):
        cli_value = getattr(args, key)
        if cli_value is not None:
            model_config[key] = bool(cli_value)
    return model_config
