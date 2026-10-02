"""Hyperparameter grid expansion and configuration selection."""

from __future__ import annotations

import inspect
import itertools
import numpy as np
import warnings

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotations only
    from typing import Callable
    from typing import Literal
    from typing import Mapping
    import tensorflow as tf

from .constants import _DECODER_SCORE_NAMES, _DEFAULT_SEQUENCE_HYPERPARAMETER_DEPTHS, _EMPTY_SEQUENCE_ALLOWED_KEYS, _FIT_RESERVED_KEYS, _JOINT_LOSS_WEIGHT_KEYS
from .reporting import _mean_std_rows

def _sequence_structure_depth(value) -> int:
    """Return the maximum list/tuple nesting depth of one value."""
    if not isinstance(value, (list, tuple)):
        return 0
    if not value:
        return 1
    return 1 + max(_sequence_structure_depth(item) for item in value)


def _copy_sequence_value(value):
    """Copy nested list/tuple values into JSON-friendly lists."""
    if isinstance(value, (list, tuple)):
        return [_copy_sequence_value(item) for item in value]
    return value


def _hyperparameter_candidates(
    key: str,
    value,
    sequence_hyperparameter_depths: Mapping[str, int] | None = None,
) -> list:
    """Return candidate values while preserving architecture sequences.

    ``sequence_hyperparameter_depths`` resolves otherwise ambiguous nested
    values. For CNN2D, for example, one ``kernel_sizes`` architecture has
    depth two (``[[3, 3], [3, 3]]``), while a depth-three value enumerates
    several kernel schedules. For CNN1D the same key has depth one.
    """
    sequence_depths = dict(_DEFAULT_SEQUENCE_HYPERPARAMETER_DEPTHS)
    if sequence_hyperparameter_depths:
        for sequence_key, expected_depth in sequence_hyperparameter_depths.items():
            expected_depth = int(expected_depth)
            if expected_depth < 1:
                raise ValueError(
                    "Sequence hyperparameter depths must be >= 1; got "
                    f"{sequence_key!r}: {expected_depth}."
                )
            sequence_depths[str(sequence_key)] = expected_depth

    if key not in sequence_depths:
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError(f"Hyperparameter {key!r} has an empty candidate list.")
            return list(value)
        return [value]

    if value is None:
        return [None]

    if not isinstance(value, (list, tuple)):
        raise TypeError(
            f"Sequence hyperparameter {key!r} must be a list or tuple, "
            f"got {type(value).__name__}."
        )
    if not value:
        if key in _EMPTY_SEQUENCE_ALLOWED_KEYS:
            return [[]]
        raise ValueError(f"Sequence hyperparameter {key!r} cannot be empty.")

    expected_depth = sequence_depths[key]
    actual_depth = _sequence_structure_depth(value)

    if actual_depth <= expected_depth:
        return [_copy_sequence_value(value)]

    if actual_depth == expected_depth + 1:
        candidates = [_copy_sequence_value(item) for item in value]
        if key not in _EMPTY_SEQUENCE_ALLOWED_KEYS and any(
            isinstance(candidate, list) and not candidate for candidate in candidates
        ):
            raise ValueError(
                f"Sequence hyperparameter {key!r} contains an empty candidate."
            )
        return candidates

    raise ValueError(
        f"Sequence hyperparameter {key!r} has nesting depth {actual_depth}, "
        f"but one architecture expects depth {expected_depth}. Use depth "
        f"{expected_depth} for one architecture or {expected_depth + 1} "
        "to enumerate candidates."
    )


def _warn_if_joint_loss_weights_vary(
    grid_configs: list[dict],
    selection_metric: str,
) -> None:
    """Warn when joint_loss selection compares models with incompatible weight settings."""
    if selection_metric != "joint_loss":
        return

    weight_keys = sorted(_JOINT_LOSS_WEIGHT_KEYS)
    weight_profiles = [
        tuple(config.get(key) for key in weight_keys)
        for config in grid_configs
    ]
    if len({profile for profile in weight_profiles}) > 1:
        warnings.warn(
            "selection_metric='joint_loss' was requested while joint-loss "
            "weight settings vary across configurations. This may make model "
            "comparison inconsistent.",
            UserWarning,
            stacklevel=3,
        )


def _expand_hyperparameter_grid(
    hp: dict | None,
    sequence_hyperparameter_depths: Mapping[str, int] | None = None,
) -> list[dict]:
    """Expand a hyperparameter dictionary into a Cartesian-product grid."""
    if not hp:
        return [{}]

    keys = list(hp)
    candidate_values = [
        _hyperparameter_candidates(
            key,
            hp[key],
            sequence_hyperparameter_depths=sequence_hyperparameter_depths,
        )
        for key in keys
    ]
    return [
        dict(zip(keys, combination))
        for combination in itertools.product(*candidate_values)
    ]


def _split_config(config: dict) -> tuple[dict, dict]:
    """Split a flat config into model-builder kwargs and model.fit kwargs."""
    model_hp = {k: v for k, v in config.items() if k not in _FIT_RESERVED_KEYS}
    fit_hp = {k: v for k, v in config.items() if k in _FIT_RESERVED_KEYS}
    return model_hp, fit_hp


def _build_model_with_fold_training_context(
    model_builder_function: Callable[..., tf.keras.Model],
    model_hp: dict,
    *,
    training_features: np.ndarray,
    training_labels: np.ndarray,
    training_subject_ids: np.ndarray,
    training_trial_ids: np.ndarray,
) -> tf.keras.Model:
    """Build a model with leakage-safe fold-local training context when supported.

    Some architectures need the actual gradient-training partition at
    construction time — for example to estimate a mutual-information adjacency
    matrix, or to determine the fold-local subject-adversarial class count.

    To preserve compatibility with older EEGProc builders, training-context
    arguments are supplied only when the builder explicitly declares them in
    its signature. Validation and test samples are never included.
    """
    builder_kwargs = dict(model_hp)
    training_context = {
        "training_features": training_features,
        "training_labels": training_labels,
        "training_subject_ids": training_subject_ids,
        "training_trial_ids": training_trial_ids,
    }

    try:
        parameters = inspect.signature(model_builder_function).parameters
    except (TypeError, ValueError):
        # Preserve legacy behavior for unusual callables whose signatures
        # cannot be inspected.
        return model_builder_function(**builder_kwargs)

    accepted_context_keys = [
        key for key in training_context if key in parameters
    ]
    for key in accepted_context_keys:
        if key in builder_kwargs:
            raise ValueError(
                f"{key!r} must be supplied fold-locally by loso_cv; do not "
                "put it in the hyperparameter configuration."
            )
        builder_kwargs[key] = training_context[key]

    if accepted_context_keys:
        print(
            "Model builder fold-training context: "
            + ", ".join(accepted_context_keys),
            flush=True,
        )

    return model_builder_function(**builder_kwargs)


def _compact_loso_training_result(fold_output: dict) -> dict:
    """Return the small per-fold training summary for one configuration."""
    fold_record = fold_output["fold_record"]
    return {
        "fold_number": int(fold_record["fold_number"]),
        "epochs_ran": int(fold_record["epochs_ran"]),
        "best_epoch": fold_record["best_epoch"],
        "best_monitored_value": fold_record["best_monitored_value"],
        "stopped_early": bool(fold_record["stopped_early"]),
        "decision_threshold": float(fold_record["decision_threshold"]),
    }


def _aggregate_loso_config_result(
    config_index: int,
    config: dict,
    fold_outputs: list[dict],
    metrics: tuple[str, ...],
    selection_metric: str,
    selection_level: Literal["window", "trial"],
) -> dict:
    """Aggregate a complete LOSO evaluation for one configuration."""
    fold_outputs = sorted(
        fold_outputs,
        key=lambda row: int(row["outer_fold_number"]),
    )

    fold_metrics = [dict(row["fold_metrics"]) for row in fold_outputs]
    window_fold_metrics = [dict(row["window_fold_metrics"]) for row in fold_outputs]
    trial_fold_metrics = [dict(row["trial_fold_metrics"]) for row in fold_outputs]

    mean_scores, std_scores = _mean_std_rows(
        fold_metrics,
        ["loss", "joint_loss", "keras_model_loss", *metrics, *_DECODER_SCORE_NAMES],
    )
    window_mean_scores, window_std_scores = _mean_std_rows(
        window_fold_metrics,
        ["loss", "joint_loss", "keras_model_loss", *metrics, *_DECODER_SCORE_NAMES],
    )
    trial_mean_scores, trial_std_scores = _mean_std_rows(
        trial_fold_metrics,
        ["loss", "joint_loss", "keras_model_loss", *metrics, *_DECODER_SCORE_NAMES],
    )

    selection_means = (
        trial_mean_scores if selection_level == "trial" else window_mean_scores
    )
    selection_stds = (
        trial_std_scores if selection_level == "trial" else window_std_scores
    )

    if selection_metric not in selection_means:
        raise ValueError(
            f"Selection metric {selection_metric!r} was not produced for "
            f"configuration {config_index}. Available metrics: "
            f"{sorted(selection_means)}"
        )

    return {
        "config_index": int(config_index),
        "config": dict(config),
        "selection_score": float(selection_means[selection_metric]),
        "selection_score_std": float(selection_stds[selection_metric]),
        "window_mean_scores": window_mean_scores,
        "window_std_scores": window_std_scores,
        "trial_mean_scores": trial_mean_scores,
        "trial_std_scores": trial_std_scores,
        "fold_metrics": fold_metrics,
        "fold_training": [_compact_loso_training_result(row) for row in fold_outputs],
        "oracle_epoch_log": [
            oracle_row
            for fold_output in fold_outputs
            for oracle_row in fold_output.get("oracle_epoch_log", [])
        ],
    }


def _loso_config_sort_key(
    config_result: dict,
    selection_metric: str,
    selection_level: Literal["window", "trial"],
    maximize_metric: bool,
) -> tuple[float, float, float, int]:
    """Return a deterministic ranking key for flat LOSO grid search.

    The primary criterion is the mean selected metric across held-out subjects.
    Ties are resolved by lower between-subject standard deviation, then lower
    mean log loss, then the earlier configuration index.
    """
    mean_key = f"{selection_level}_mean_scores"
    std_key = f"{selection_level}_std_scores"
    mean_scores = config_result[mean_key]
    std_scores = config_result[std_key]

    primary = float(mean_scores[selection_metric])
    primary_std = float(std_scores[selection_metric])
    mean_loss = float(mean_scores.get("loss", np.inf))

    if not np.isfinite(primary):
        primary_rank = np.inf
    else:
        primary_rank = -primary if maximize_metric else primary

    if not np.isfinite(primary_std):
        primary_std = np.inf
    if not np.isfinite(mean_loss):
        mean_loss = np.inf

    return (
        float(primary_rank),
        float(primary_std),
        float(mean_loss),
        int(config_result["config_index"]),
    )


def _choose_best_loso_config_index(
    config_results: list[dict],
    selection_metric: str,
    selection_level: Literal["window", "trial"],
    maximize_metric: bool,
) -> int:
    """Choose the global configuration after every config completes LOSO."""
    if not config_results:
        raise ValueError("No LOSO configuration results were produced.")

    best_result = min(
        config_results,
        key=lambda row: _loso_config_sort_key(
            config_result=row,
            selection_metric=selection_metric,
            selection_level=selection_level,
            maximize_metric=maximize_metric,
        ),
    )

    best_score = float(best_result["selection_score"])
    if not np.isfinite(best_score):
        raise RuntimeError(
            "All LOSO configurations produced a non-finite selection score."
        )

    return int(best_result["config_index"])


def _choose_best_config_index(
    mean_scores: list[dict],
    selection_metric: str,
    maximize_metric: bool,
) -> int:
    """Choose the best hyperparameter config from inner-CV mean scores."""
    metric_values = [scores[selection_metric] for scores in mean_scores]

    if maximize_metric:
        return int(np.argmax(metric_values))

    return int(np.argmin(metric_values))
