"""Calibration-only Cartesian search across target-specific SIC LOSO models.

This entry point never builds or source-trains a SIC model. ``--models-config``
identifies one serialized Keras checkpoint per target user. A fresh copy of
every checkpoint is loaded for every Cartesian-grid candidate and passed as an
in-memory model to ``cross_val.subject_calibration_cv``.

Only calibration behavior may be searched. The saved encoders, graph,
recurrent architecture, subject adversary, and source-training state remain
fixed. Decoder weights also remain fixed by default, but calibration may
optionally continue training them with a separately weighted reconstruction
loss. Every calibration fold restores the original checkpoint weights and
creates a fresh optimizer through ``prepare_for_subject_calibration``.
Configurations are ranked by a subject-macro average: folds are averaged within
each user first, then all users receive equal weight.
"""

from __future__ import annotations

import argparse
import gc
from datetime import datetime
import json
from pathlib import Path
import re
import sys
import traceback

import numpy as np
import tensorflow as tf
from joblib.externals import cloudpickle

try:
    # Importing sic_model registers SICModel and its custom Keras objects before
    # load_model deserializes the checkpoint.
    from . import sic_model as _sic_model_registration  # noqa: F401
    from .sic_model_args import (
        SIC_CLASSIFICATION_METRICS,
        build_parser as build_sic_parser,
        calibration_levels_from_args,
        expand_cartesian_grid,
        metric_mode,
    )
    from .sic_model_train import (
        _apply_median_label_ablation,
        _configure_logger,
        _configuration_summary_row,
        _ensure_dir,
        _group_consistent_trial_values,
        _group_windows_into_trials,
        _json_default,
        _json_fingerprint,
        _log_trial_distribution,
        _save_calibration_artifacts,
        _selection_score_from_calibration_results,
        _write_csv,
        _write_json,
        load_sic_training_data,
    )
except ImportError:
    HERE = Path(__file__).resolve().parent
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from eegproc.deep_learning.joint_architectures.SICModelv11 import (
        sic_model as _sic_model_registration,  # noqa: F401
    )
    from eegproc.deep_learning.joint_architectures.SICModelv11.sic_model_args import (
        SIC_CLASSIFICATION_METRICS,
        build_parser as build_sic_parser,
        calibration_levels_from_args,
        expand_cartesian_grid,
        metric_mode,
    )
    from eegproc.deep_learning.joint_architectures.SICModelv11.sic_model_train import (
        _apply_median_label_ablation,
        _configure_logger,
        _configuration_summary_row,
        _ensure_dir,
        _group_consistent_trial_values,
        _group_windows_into_trials,
        _json_default,
        _json_fingerprint,
        _log_trial_distribution,
        _save_calibration_artifacts,
        _selection_score_from_calibration_results,
        _write_csv,
        _write_json,
        load_sic_training_data,
    )

try:
    from ... import cross_val as _cross_val
except ImportError:
    from eegproc.deep_learning import cross_val as _cross_val

subject_calibration_cv = _cross_val.subject_calibration_cv

try:
    from ..joint_models_data import get_dataset_config
except ImportError:
    from eegproc.deep_learning.joint_architectures.joint_models_data import (
        get_dataset_config,
    )


_RUNTIME_CALIBRATION_KEYS = frozenset(
    {
        "calibration_epochs",
        "calibration_batch_size",
        "calibration_learning_rate",
        "calibration_optimizer",
        "calibration_weight_decay",
        "calibration_use_class_weight",
        "calibration_shuffle",
    }
)

_MODEL_CALIBRATION_KEYS = frozenset(
    {
        "calibration_unfreeze_layers",
        "calibration_use_vc_target",
        "calibration_vc_alpha",
        "calibration_vc_beta",
        "calibration_vc_gamma",
        "calibration_vc_lambda",
        "calibration_vc_loss_weight",
        "calibration_freeze_decoder",
        "calibration_decoder_loss_weight",
        "calibration_loss",
        "calibration_label_smoothing",
        "calibration_focal_gamma",
        "calibration_focal_alpha",
    }
)

_CALIBRATION_KEYS = _RUNTIME_CALIBRATION_KEYS | _MODEL_CALIBRATION_KEYS

_TARGET_FROM_FILENAME = re.compile(
    r"_target_(-?\d+)_zero_shot\.keras$",
    flags=re.IGNORECASE,
)


def _positive_int(value: str) -> int:
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def build_parser():
    """Extend the SIC data CLI with multi-checkpoint calibration options."""
    parser = build_sic_parser()
    parser.description = (
        "Load one target-specific SIC LOSO checkpoint per user and rank a "
        "Cartesian calibration grid by average user metrics."
    )
    parser.add_argument(
        "--models-config",
        type=Path,
        required=True,
        help=(
            "JSON containing a models list/map or model_directory plus "
            "model_glob. Relative paths are resolved from this file."
        ),
    )
    parser.add_argument(
        "--calibration-hyperparameters-json",
        default=None,
        help=(
            "Calibration-only fixed values or Cartesian axes. Use ordinary "
            "JSON lists as axes, or {\"grid\": [...]} / {\"fixed\": value}. "
            f"Allowed keys: {', '.join(sorted(_CALIBRATION_KEYS))}."
        ),
    )
    parser.add_argument(
        "--calibration-hyperparameters-file",
        type=Path,
        default=None,
        help="File containing the same JSON object accepted by the inline option.",
    )
    parser.add_argument(
        "--calibration-verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print fine-tuning metrics during every user/shot/fold fit.",
    )
    parser.add_argument(
        "--calibration-print-every-n-epochs",
        type=_positive_int,
        default=1,
        help="Verbose print interval; the final epoch is always printed.",
    )
    parser.add_argument(
        "--allow-partial-model-set",
        action="store_true",
        help="Allow the models file to omit dataset users.",
    )
    return parser


def _default_calibration_configuration(args) -> dict:
    return {
        "calibration_epochs": int(args.calibration_epochs),
        "calibration_batch_size": int(args.calibration_batch_size),
        "calibration_learning_rate": float(args.calibration_learning_rate),
        "calibration_optimizer": str(args.calibration_optimizer),
        "calibration_weight_decay": float(args.calibration_weight_decay),
        "calibration_use_class_weight": bool(args.calibration_use_class_weight),
        "calibration_shuffle": True,
    }


def _calibration_grid_from_args(args) -> tuple[list[dict], dict[str, int], dict]:
    """Return the expanded calibration-only grid and its unexpanded JSON spec."""
    if (
        args.calibration_hyperparameters_json is not None
        and args.calibration_hyperparameters_file is not None
    ):
        raise ValueError(
            "Use only one of --calibration-hyperparameters-json and "
            "--calibration-hyperparameters-file."
        )
    try:
        if args.calibration_hyperparameters_file is not None:
            requested = json.loads(
                Path(args.calibration_hyperparameters_file)
                .expanduser()
                .read_text(encoding="utf-8")
            )
        elif args.calibration_hyperparameters_json is not None:
            requested = json.loads(args.calibration_hyperparameters_json)
        else:
            requested = {}
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"Could not read calibration hyperparameters: {error}") from error
    if not isinstance(requested, dict):
        raise ValueError(
            "--calibration-hyperparameters-json must decode to a JSON object."
        )
    unknown = sorted(set(requested) - _CALIBRATION_KEYS)
    if unknown:
        raise ValueError(
            "Calibration-only search cannot change source/model architecture "
            f"keys: {unknown}. Allowed keys are {sorted(_CALIBRATION_KEYS)}."
        )

    grid_spec = _default_calibration_configuration(args)
    grid_spec.update(requested)
    configurations, dimensions = expand_cartesian_grid(grid_spec)
    for index, configuration in enumerate(configurations, start=1):
        _validate_calibration_configuration(configuration, index=index)
    return configurations, dimensions, grid_spec


def _validate_calibration_configuration(configuration: dict, *, index: int) -> None:
    prefix = f"Calibration configuration {index}"
    for name in ("calibration_epochs", "calibration_batch_size"):
        if int(configuration[name]) < 1:
            raise ValueError(f"{prefix}: {name} must be >= 1.")
    if float(configuration["calibration_learning_rate"]) <= 0.0:
        raise ValueError(f"{prefix}: calibration_learning_rate must be positive.")
    if float(configuration["calibration_weight_decay"]) < 0.0:
        raise ValueError(
            f"{prefix}: calibration_weight_decay must be non-negative."
        )
    if configuration["calibration_optimizer"] not in {"adam", "adamw"}:
        raise ValueError(
            f"{prefix}: calibration_optimizer must be 'adam' or 'adamw'."
        )
    if not isinstance(configuration["calibration_use_class_weight"], bool):
        raise ValueError(
            f"{prefix}: calibration_use_class_weight must be true or false."
        )
    if not isinstance(configuration["calibration_shuffle"], bool):
        raise ValueError(f"{prefix}: calibration_shuffle must be true or false.")
    if "calibration_unfreeze_layers" in configuration:
        layers = int(configuration["calibration_unfreeze_layers"])
        if layers not in {1, 2}:
            raise ValueError(
                f"{prefix}: calibration_unfreeze_layers must be 1 or 2."
            )
    if (
        "calibration_use_vc_target" in configuration
        and not isinstance(configuration["calibration_use_vc_target"], bool)
    ):
        raise ValueError(
            f"{prefix}: calibration_use_vc_target must be true or false."
        )
    if (
        "calibration_freeze_decoder" in configuration
        and not isinstance(configuration["calibration_freeze_decoder"], bool)
    ):
        raise ValueError(
            f"{prefix}: calibration_freeze_decoder must be true or false."
        )
    for name in (
        "calibration_vc_loss_weight",
        "calibration_vc_alpha",
        "calibration_vc_beta",
        "calibration_vc_gamma",
        "calibration_vc_lambda",
    ):
        if name in configuration and configuration[name] is not None:
            value = float(configuration[name])
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{prefix}: {name} must be null or a finite non-negative value."
                )
    if "calibration_decoder_loss_weight" in configuration:
        value = configuration["calibration_decoder_loss_weight"]
        if value is None or not np.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(
                f"{prefix}: calibration_decoder_loss_weight must be a finite "
                "non-negative value."
            )
    if "calibration_loss" in configuration:
        loss_name = str(configuration["calibration_loss"]).lower().replace("-", "_")
        if loss_name not in {"focal", "cross_entropy"}:
            raise ValueError(
                f"{prefix}: calibration_loss must be 'focal' or 'cross_entropy'."
            )
    if "calibration_label_smoothing" in configuration:
        value = float(configuration["calibration_label_smoothing"])
        if not 0.0 <= value < 1.0:
            raise ValueError(
                f"{prefix}: calibration_label_smoothing must be in [0, 1)."
            )
    if "calibration_focal_gamma" in configuration:
        value = float(configuration["calibration_focal_gamma"])
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                f"{prefix}: calibration_focal_gamma must be non-negative."
            )
    if (
        "calibration_focal_alpha" in configuration
        and configuration["calibration_focal_alpha"] is not None
    ):
        value = float(configuration["calibration_focal_alpha"])
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{prefix}: calibration_focal_alpha must be null or in [0, 1]."
            )


def _read_models_config(config_path: Path) -> tuple[Path, list[dict], dict]:
    resolved_config = Path(config_path).expanduser().resolve()
    if not resolved_config.is_file():
        raise FileNotFoundError(
            f"Models configuration does not exist: {resolved_config}"
        )
    try:
        payload = json.loads(resolved_config.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid models configuration JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("--models-config must contain a JSON object.")

    config_dir = resolved_config.parent

    def resolve_path(raw_path) -> Path:
        path = Path(str(raw_path)).expanduser()
        if not path.is_absolute():
            path = config_dir / path
        return path.resolve()

    specs: list[dict] = []
    if "models" in payload:
        if "model_directory" in payload or "model_glob" in payload:
            raise ValueError(
                "Use either models or model_directory/model_glob, not both."
            )
        raw_models = payload["models"]
        if isinstance(raw_models, dict):
            raw_models = [
                {"target_subject": subject, "path": path}
                for subject, path in raw_models.items()
            ]
        if not isinstance(raw_models, list) or not raw_models:
            raise ValueError("models must be a non-empty list or subject:path map.")
        for position, entry in enumerate(raw_models):
            if not isinstance(entry, dict):
                raise ValueError(f"models[{position}] must be a JSON object.")
            if "target_subject" not in entry or "path" not in entry:
                raise ValueError(
                    f"models[{position}] requires target_subject and path."
                )
            specs.append(
                {
                    "target_subject": int(entry["target_subject"]),
                    "path": resolve_path(entry["path"]),
                }
            )
    else:
        if "model_directory" not in payload:
            raise ValueError(
                "--models-config requires models or model_directory."
            )
        model_directory = resolve_path(payload["model_directory"])
        if not model_directory.is_dir():
            raise FileNotFoundError(
                f"model_directory does not exist: {model_directory}"
            )
        model_glob = str(
            payload.get(
                "model_glob",
                "loso_fold_*_target_*_zero_shot.keras",
            )
        )
        paths = sorted(path.resolve() for path in model_directory.glob(model_glob))
        if not paths:
            raise FileNotFoundError(
                f"No checkpoints matched {model_glob!r} in {model_directory}."
            )
        for path in paths:
            match = _TARGET_FROM_FILENAME.search(path.name)
            if match is None:
                raise ValueError(
                    "Discovered checkpoints must end with "
                    f"'_target_<ID>_zero_shot.keras': {path.name}"
                )
            specs.append(
                {"target_subject": int(match.group(1)), "path": path}
            )

    seen_targets: dict[int, Path] = {}
    seen_paths: set[Path] = set()
    for spec in specs:
        target_subject = int(spec["target_subject"])
        model_path = Path(spec["path"])
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model for target user {target_subject} does not exist: "
                f"{model_path}"
            )
        if model_path.suffix.lower() != ".keras":
            raise ValueError(f"Checkpoint must use .keras format: {model_path}")
        if target_subject in seen_targets:
            raise ValueError(
                f"Duplicate model for user {target_subject}: "
                f"{seen_targets[target_subject]} and {model_path}"
            )
        if model_path in seen_paths:
            raise ValueError(f"The same checkpoint is listed twice: {model_path}")
        match = _TARGET_FROM_FILENAME.search(model_path.name)
        if match is not None and int(match.group(1)) != target_subject:
            raise ValueError(
                f"Configured user {target_subject} does not match filename "
                f"target {match.group(1)}: {model_path.name}"
            )
        seen_targets[target_subject] = model_path
        seen_paths.add(model_path)

    specs.sort(key=lambda row: int(row["target_subject"]))
    normalized_payload = {
        **payload,
        "resolved_models": [
            {
                "target_subject": int(spec["target_subject"]),
                "path": str(spec["path"]),
            }
            for spec in specs
        ],
    }
    return resolved_config, specs, normalized_payload


def _validate_args(args, calibration_levels) -> None:
    if int(args.n_jobs) < 1:
        raise ValueError("--n-jobs must be >= 1.")
    if args.gpu_ids is not None:
        gpu_ids = tuple(int(gpu_id) for gpu_id in args.gpu_ids)
        if not gpu_ids:
            raise ValueError("--gpu-ids must contain at least one GPU index.")
        if any(gpu_id < 0 for gpu_id in gpu_ids):
            raise ValueError("--gpu-ids values must be non-negative.")
        if len(set(gpu_ids)) != len(gpu_ids):
            raise ValueError("--gpu-ids must not contain duplicates.")
    if args.hyperparameters_json is not None:
        raise ValueError(
            "Use a calibration hyperparameter option. Architecture/source "
            "hyperparameters cannot change after loading checkpoints."
        )
    if args.hyperparameter_selection_level != "calibration":
        raise ValueError(
            "Calibration-only search requires "
            "--hyperparameter-selection-level calibration."
        )
    if args.prediction_diagnostics:
        raise ValueError(
            "--prediction-diagnostics applies to source training, which is skipped."
        )
    shot_levels = [int(shots) for shots, _ in calibration_levels]
    if len(set(shot_levels)) != len(shot_levels):
        raise ValueError("Each calibration SHOTS level must be unique.")
    selection_shots = (
        max(shot_levels)
        if args.calibration_selection_shots is None
        else int(args.calibration_selection_shots)
    )
    if selection_shots not in shot_levels:
        raise ValueError(
            "--calibration-selection-shots must match one configured shot level."
        )
    if not 0.0 < float(args.decision_threshold) < 1.0:
        raise ValueError("--decision-threshold must lie in (0, 1).")
    if int(args.ece_bins) < 2:
        raise ValueError("--ece-bins must be >= 2.")


def _apply_model_calibration_configuration(model, configuration: dict) -> None:
    """Set only serialized SIC fields that govern calibration behavior."""
    for name in sorted(_MODEL_CALIBRATION_KEYS & configuration.keys()):
        if not hasattr(model, name):
            raise AttributeError(
                f"Loaded model has no calibration setting {name!r}; the "
                "checkpoint may predate this calibration-only search API."
            )
        value = configuration[name]
        if value is None and name in {
            "calibration_vc_loss_weight",
            "calibration_vc_alpha",
            "calibration_vc_beta",
            "calibration_vc_gamma",
            "calibration_vc_lambda",
        }:
            # Null means inherit the value already resolved in the checkpoint.
            continue
        if name == "calibration_unfreeze_layers":
            value = int(value)
        elif name in {
            "calibration_use_vc_target",
            "calibration_freeze_decoder",
        }:
            value = bool(value)
        elif name == "calibration_loss":
            value = str(value).lower().replace("-", "_")
        elif value is not None:
            value = float(value)
        setattr(model, name, value)


def _validate_loaded_model(model, *, classification_level: str) -> None:
    if not callable(getattr(model, "prepare_for_subject_calibration", None)):
        raise TypeError(
            "Loaded model must implement prepare_for_subject_calibration(...)."
        )
    checkpoint_level = getattr(model, "classification_level", None)
    if checkpoint_level is not None and str(checkpoint_level) != classification_level:
        raise ValueError(
            "Loaded model classification_level does not match the requested "
            f"data representation: model={checkpoint_level!r}, "
            f"requested={classification_level!r}."
        )


def _load_data(args):
    dataset_config = get_dataset_config(args.dataset)
    eeg_path = args.raw_eeg_npy or dataset_config.eeg_path
    labels_path = args.raw_labels_npy or dataset_config.labels_path
    loaded = load_sic_training_data(
        eeg_path=eeg_path,
        labels_path=labels_path,
        label_dimension=args.label_dimension,
        window_size_sec=args.window_sec,
        fs=args.fs,
        overlap=args.window_overlap,
        median_label=args.median_label,
        window_normalization=args.window_normalization,
        label_threshold_mode=args.label_threshold_mode,
        dataset=dataset_config,
        return_original_ratings=True,
    )
    X, y, subjects, trials, original_ratings = loaded
    if args.classification_level == "trial":
        if original_ratings is not None:
            original_ratings = _group_consistent_trial_values(
                original_ratings,
                subjects,
                trials,
                value_name="original target rating",
            )
        X, y, subjects, trials = _group_windows_into_trials(X, y, subjects, trials)

    X, y, subjects, trials, data_summary = _apply_median_label_ablation(
        X,
        y,
        subjects,
        trials,
        original_ratings,
        remove_median_label=bool(args.remove_median_label),
        median_label=args.median_label,
    )
    return X, y, subjects, trials, data_summary


def _validate_model_coverage(model_specs, subjects, *, allow_partial: bool) -> None:
    available = {int(value) for value in np.unique(subjects).tolist()}
    configured = {int(spec["target_subject"]) for spec in model_specs}
    unknown = sorted(configured - available)
    if unknown:
        raise ValueError(f"Configured users absent from the data: {unknown}.")
    missing = sorted(available - configured)
    if missing and not allow_partial:
        raise ValueError(
            "Every retained dataset user must have one checkpoint. Missing "
            f"users: {missing}. Use --allow-partial-model-set only for an "
            "intentional subset run."
        )


def _select_model_specs(
    model_specs: list[dict],
    *,
    target_subjects,
    max_subjects: int | None,
) -> list[dict]:
    """Select configured target checkpoints using the calibration CLI flags."""
    selected = list(model_specs)
    if target_subjects is not None:
        requested = [int(subject) for subject in target_subjects]
        if len(set(requested)) != len(requested):
            raise ValueError("--target-subjects must not contain duplicates.")
        by_subject = {
            int(spec["target_subject"]): spec
            for spec in model_specs
        }
        missing = [subject for subject in requested if subject not in by_subject]
        if missing:
            raise ValueError(
                "Requested --target-subjects have no checkpoint in "
                f"--models-config: {missing}. Available targets are "
                f"{sorted(by_subject)}."
            )
        # Preserve the explicit command-line order for predictable targeted runs.
        selected = [by_subject[subject] for subject in requested]

    if max_subjects is not None:
        selected = selected[: int(max_subjects)]
    if not selected:
        raise ValueError("Subject selection left no target checkpoints to evaluate.")
    return selected


def _mean_std_metric_rows(rows: list[dict]) -> tuple[dict, dict, dict]:
    keys = sorted({key for row in rows for key in row})
    means, stds, counts = {}, {}, {}
    for key in keys:
        values = np.asarray(
            [float(row[key]) for row in rows if key in row],
            dtype=np.float64,
        )
        values = values[np.isfinite(values)]
        if len(values):
            means[key] = float(np.mean(values))
            stds[key] = float(np.std(values))
            counts[key] = int(len(values))
    return means, stds, counts


def _subject_record(target_subject: int, model_path: Path, results: dict) -> dict:
    overall = dict(results.get("overall", {}))
    return {
        "target_subject": int(target_subject),
        "model_path": str(model_path),
        "zero_shot_all_trials_scores": overall.get(
            "zero_shot_all_trials_mean_scores", {}
        ),
        "calibration_levels": overall.get("calibration_levels", {}),
    }


def _aggregate_subject_records(
    records: list[dict],
    *,
    calibration_levels,
    selection_shots: int,
) -> dict:
    zero_mean, zero_std, zero_counts = _mean_std_metric_rows(
        [row["zero_shot_all_trials_scores"] for row in records]
    )
    levels = {}
    for shots, folds in calibration_levels:
        key = str(int(shots))
        paired_mean, paired_std, paired_counts = _mean_std_metric_rows(
            [
                row["calibration_levels"][key][
                    "paired_zero_shot_mean_scores"
                ]
                for row in records
            ]
        )
        calibrated_mean, calibrated_std, calibrated_counts = (
            _mean_std_metric_rows(
                [
                    row["calibration_levels"][key]["calibrated_mean_scores"]
                    for row in records
                ]
            )
        )
        delta_mean, delta_std, delta_counts = _mean_std_metric_rows(
            [
                row["calibration_levels"][key]["delta_mean_scores"]
                for row in records
            ]
        )
        levels[key] = {
            "calibration_shots": int(shots),
            "calibration_folds_per_subject": int(folds),
            "n_subjects": len(records),
            "aggregation_unit": "subject_fold_mean",
            "paired_zero_shot_mean_scores": paired_mean,
            "paired_zero_shot_std_across_subjects": paired_std,
            "paired_zero_shot_metric_subject_counts": paired_counts,
            "calibrated_mean_scores": calibrated_mean,
            "calibrated_std_across_subjects": calibrated_std,
            "calibrated_metric_subject_counts": calibrated_counts,
            "delta_mean_scores": delta_mean,
            "delta_std_across_subjects": delta_std,
            "delta_metric_subject_counts": delta_counts,
        }
    selected = levels[str(int(selection_shots))]
    return {
        "n_subjects": len(records),
        "aggregation": (
            "Folds are averaged within each user, then user means are "
            "averaged with equal subject weight."
        ),
        "zero_shot_all_trials_mean_scores": zero_mean,
        "zero_shot_all_trials_std_across_subjects": zero_std,
        "zero_shot_all_trials_metric_subject_counts": zero_counts,
        "calibration_selection_shots": int(selection_shots),
        "calibration_levels": levels,
        "paired_zero_shot_mean_scores": selected[
            "paired_zero_shot_mean_scores"
        ],
        "calibrated_mean_scores": selected["calibrated_mean_scores"],
        "delta_mean_scores": selected["delta_mean_scores"],
    }


def _log_user_metrics(logger, record: dict) -> None:
    user = record["target_subject"]
    logger.info(
        "User %s zero-shot all-trial metrics: %s",
        user,
        _json_fingerprint(record["zero_shot_all_trials_scores"]),
    )
    for shots, level in sorted(
        record["calibration_levels"].items(),
        key=lambda item: int(item[0]),
    ):
        logger.info(
            "User %s %s-shot calibrated metrics: %s",
            user,
            shots,
            _json_fingerprint(level.get("calibrated_mean_scores", {})),
        )
        logger.info(
            "User %s %s-shot calibration deltas: %s",
            user,
            shots,
            _json_fingerprint(level.get("delta_mean_scores", {})),
        )


def _run_one_user_calibration(
    *,
    spec: dict,
    configuration: dict,
    configuration_dir: Path,
    feature_array: np.ndarray,
    label_array: np.ndarray,
    subject_id_array: np.ndarray,
    trial_id_array: np.ndarray,
    calibration_levels,
    selection_shots: int,
    args,
) -> dict:
    """Load, fine-tune, evaluate, and save one target user's checkpoint."""
    target_subject = int(spec["target_subject"])
    model_path = Path(spec["path"])
    user_dir = _ensure_dir(configuration_dir / f"user_{target_subject:03d}")

    if args.seed is not None:
        user_seed = int(args.seed) + target_subject
        tf.keras.utils.set_random_seed(user_seed)
        np.random.seed(user_seed)

    model = tf.keras.models.load_model(model_path, compile=False)
    try:
        _validate_loaded_model(
            model,
            classification_level=args.classification_level,
        )
        _apply_model_calibration_configuration(model, configuration)
        results = subject_calibration_cv(
            model_builder_function=None,
            pretrained_model=model,
            feature_array=feature_array,
            label_array=label_array,
            subject_id_array=subject_id_array,
            trial_id_array=trial_id_array,
            fixed_config=None,
            source_epochs=0,
            source_batch_size=int(args.source_batch_size),
            calibration_epochs=int(configuration["calibration_epochs"]),
            calibration_batch_size=int(
                configuration["calibration_batch_size"]
            ),
            calibration_trials=int(args.calibration_trials),
            calibration_folds=int(args.calibration_folds),
            calibration_levels=calibration_levels,
            calibration_selection_shots=selection_shots,
            calibration_learning_rate=float(
                configuration["calibration_learning_rate"]
            ),
            calibration_optimizer=str(configuration["calibration_optimizer"]),
            calibration_weight_decay=float(
                configuration["calibration_weight_decay"]
            ),
            calibration_seed=args.calibration_seed,
            stratify_calibration=not args.no_stratify_calibration,
            validation_subjects_per_fold=0,
            early_stopping_patience=None,
            restore_best_weights=False,
            evaluation_level=args.classification_level,
            metrics=SIC_CLASSIFICATION_METRICS,
            ece_bins=int(args.ece_bins),
            decision_threshold=float(args.decision_threshold),
            prediction_diagnostics=False,
            log_predictions=True,
            source_use_class_weight=False,
            calibration_use_class_weight=bool(
                configuration["calibration_use_class_weight"]
            ),
            source_fit_kwargs=None,
            calibration_fit_kwargs={
                "callbacks": [tf.keras.callbacks.TerminateOnNaN()],
                "shuffle": bool(configuration["calibration_shuffle"]),
            },
            calibration_verbose=bool(args.calibration_verbose),
            calibration_print_every_n_epochs=int(
                args.calibration_print_every_n_epochs
            ),
            verbose=0,
            # A live Keras model intentionally remains single-job inside
            # cross_val. Parallelism is across users in this outer script.
            n_jobs=1,
            gpu_ids=None,
            cpus_per_worker=args.cpus_per_worker,
            max_subjects=None,
            target_subjects=[target_subject],
            source_model_output_dir=None,
        )
        results["loaded_model"] = {
            "path": str(model_path),
            "target_subject": target_subject,
            "load_with_compile": False,
        }
        _save_calibration_artifacts(
            user_dir,
            model_config=configuration,
            results=results,
        )
        return _subject_record(target_subject, model_path, results)
    finally:
        del model
        tf.keras.backend.clear_session()
        gc.collect()


def _calibration_only_process_main(
    worker_state_payload,
    task_queue,
    result_queue,
    gpu_id,
    cpus_per_worker,
    assigned_device_label,
) -> None:
    """Run target-user calibration tasks inside one device-bound process."""
    try:
        _cross_val._configure_tensorflow_worker(
            gpu_id,
            cpus_per_worker,
            assigned_device_label,
        )
        worker_state = cloudpickle.loads(worker_state_payload)
        while True:
            task = task_queue.get()
            if task is None:
                return
            target_subject, user_index, spec = task
            try:
                print(
                    f"Configuration {worker_state['configuration_index']}: "
                    f"calibrating user {target_subject} "
                    f"({user_index}/{worker_state['total_users']}) from "
                    f"{spec['path']}",
                    flush=True,
                )
                record = _run_one_user_calibration(
                    spec=spec,
                    configuration=worker_state["configuration"],
                    configuration_dir=Path(worker_state["configuration_dir"]),
                    feature_array=worker_state["feature_array"],
                    label_array=worker_state["label_array"],
                    subject_id_array=worker_state["subject_id_array"],
                    trial_id_array=worker_state["trial_id_array"],
                    calibration_levels=worker_state["calibration_levels"],
                    selection_shots=worker_state["selection_shots"],
                    args=worker_state["args"],
                )
                result_queue.put(("ok", int(target_subject), record))
            except BaseException:
                result_queue.put(
                    ("error", int(target_subject), traceback.format_exc())
                )
                return
    except BaseException:
        result_queue.put(("error", -1, traceback.format_exc()))


def _resolve_outer_workers(args, n_models: int):
    """Resolve user-worker count and local GPU IDs for the outer search."""
    effective_n_jobs = min(int(args.n_jobs), int(n_models))
    normalized_gpu_ids = None
    if args.gpu_ids is not None:
        if len(args.gpu_ids) < effective_n_jobs:
            raise ValueError(
                f"The selected {n_models} target subject(s) and --n-jobs "
                f"{args.n_jobs} require {effective_n_jobs} GPU ID(s), but "
                f"--gpu-ids supplied {list(args.gpu_ids)}."
            )
        normalized_gpu_ids = tuple(
            int(gpu_id) for gpu_id in args.gpu_ids[:effective_n_jobs]
        )
    elif effective_n_jobs > 1:
        normalized_gpu_ids = _cross_val._auto_assign_gpu_ids(effective_n_jobs)
        if normalized_gpu_ids is not None:
            effective_n_jobs = len(normalized_gpu_ids)
    return effective_n_jobs, normalized_gpu_ids


def run_calibration_only_search(args) -> dict:
    """Evaluate the selected configured models for every calibration candidate."""
    calibration_levels = calibration_levels_from_args(args)
    _validate_args(args, calibration_levels)
    models_config_path, model_specs, models_config = _read_models_config(
        args.models_config
    )
    configurations, dimensions, grid_spec = _calibration_grid_from_args(args)
    selection_shots = (
        max(shots for shots, _ in calibration_levels)
        if args.calibration_selection_shots is None
        else int(args.calibration_selection_shots)
    )

    X, y, subjects, trials, data_summary = _load_data(args)
    _validate_model_coverage(
        model_specs,
        subjects,
        allow_partial=bool(args.allow_partial_model_set),
    )
    available_model_specs = list(model_specs)
    model_specs = _select_model_specs(
        available_model_specs,
        target_subjects=args.target_subjects,
        max_subjects=args.max_subjects,
    )
    effective_n_jobs, normalized_gpu_ids = _resolve_outer_workers(
        args,
        len(model_specs),
    )
    use_spawned_workers = (
        effective_n_jobs > 1 or normalized_gpu_ids is not None
    )
    if use_spawned_workers:
        required_cross_val_helpers = (
            "_auto_assign_gpu_ids",
            "_configure_tensorflow_worker",
            "_run_spawned_fold_pool",
        )
        missing_helpers = [
            name
            for name in required_cross_val_helpers
            if not hasattr(_cross_val, name)
        ]
        if missing_helpers:
            raise ImportError(
                "Parallel user calibration requires the updated cross_val.py; "
                f"missing helpers: {missing_helpers}."
            )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _ensure_dir(
        Path(args.out_dir) / f"{args.run_name}_calibration_only_{timestamp}"
    )
    logger = _configure_logger(run_dir)

    run_config = {
        "protocol": "calibration_only_selected_target_models",
        "models_config_path": str(models_config_path),
        "available_models": models_config["resolved_models"],
        "models": [
            {
                "target_subject": int(spec["target_subject"]),
                "path": str(spec["path"]),
            }
            for spec in model_specs
        ],
        "n_models": len(model_specs),
        "n_available_models": len(available_model_specs),
        "requested_target_subjects": (
            None
            if args.target_subjects is None
            else [int(subject) for subject in args.target_subjects]
        ),
        "max_subjects": args.max_subjects,
        "require_full_subject_coverage": not bool(args.allow_partial_model_set),
        "dataset": args.dataset,
        "label_dimension": args.label_dimension,
        "classification_level": args.classification_level,
        "window_sec": float(args.window_sec),
        "window_overlap": float(args.window_overlap),
        "fs": float(args.fs),
        "window_normalization": args.window_normalization,
        "label_threshold_mode": args.label_threshold_mode,
        "remove_median_label": bool(args.remove_median_label),
        "median_label": float(args.median_label),
        "calibration_plan": [
            {"shots": int(shots), "folds": int(folds)}
            for shots, folds in calibration_levels
        ],
        "calibration_selection_shots": selection_shots,
        "selection_metric": args.selection_metric,
        "decision_threshold": float(args.decision_threshold),
        "calibration_verbose": bool(args.calibration_verbose),
        "calibration_print_every_n_epochs": int(
            args.calibration_print_every_n_epochs
        ),
        "requested_user_workers": int(args.n_jobs),
        "effective_user_workers": int(effective_n_jobs),
        "worker_gpu_ids": (
            None
            if normalized_gpu_ids is None
            else list(normalized_gpu_ids)
        ),
        "cpus_per_worker": args.cpus_per_worker,
        "calibration_grid": grid_spec,
        "seed": args.seed,
    }
    _write_json(run_dir / "calibration_only_config.json", run_config)
    _write_json(run_dir / "resolved_models_config.json", models_config)
    _write_json(run_dir / "dataset_ablation.json", data_summary)
    _write_json(
        run_dir / "calibration_hyperparameter_grid.json",
        {
            "search_type": "full_cartesian_product",
            "dimensions": dimensions,
            "n_configurations": len(configurations),
            "configurations": [
                {"configuration_id": index, "hyperparameters": configuration}
                for index, configuration in enumerate(configurations, start=1)
            ],
        },
    )

    logger.info("Protocol: calibration-only search across selected target models")
    logger.info("Models configuration: %s", models_config_path)
    logger.info(
        "Target users (%d): %s",
        len(model_specs),
        [spec["target_subject"] for spec in model_specs],
    )
    logger.info("Calibration plan: %s", calibration_levels)
    logger.info(
        "User workers: %d; devices: %s; CPUs per worker: %s",
        effective_n_jobs,
        (
            "CPU"
            if normalized_gpu_ids is None
            else f"local GPUs {list(normalized_gpu_ids)}"
        ),
        args.cpus_per_worker,
    )
    logger.info(
        "Cartesian grid: dimensions=%s total_configurations=%d",
        dimensions or {"fixed_configuration": 1},
        len(configurations),
    )
    _log_trial_distribution(logger, data_summary, context="Calibration-only data")

    completed: list[dict] = []
    failed: list[dict] = []
    progress_path = run_dir / "hyperparameter_search_progress.json"

    for index, configuration in enumerate(configurations, start=1):
        configuration_dir = _ensure_dir(run_dir / f"configuration_{index:04d}")
        _write_json(configuration_dir / "calibration_config.json", configuration)
        logger.info(
            "Starting configuration %d/%d across %d users: %s",
            index,
            len(configurations),
            len(model_specs),
            _json_fingerprint(configuration),
        )
        subject_records: list[dict] = []
        active_user: int | None = None
        try:
            if use_spawned_workers:
                worker_state = {
                    "configuration_index": index,
                    "configuration": configuration,
                    "configuration_dir": str(configuration_dir),
                    "feature_array": X,
                    "label_array": y,
                    "subject_id_array": subjects,
                    "trial_id_array": trials,
                    "calibration_levels": calibration_levels,
                    "selection_shots": selection_shots,
                    "args": args,
                    "total_users": len(model_specs),
                }
                tasks = [
                    (
                        int(spec["target_subject"]),
                        user_index,
                        spec,
                    )
                    for user_index, spec in enumerate(model_specs, start=1)
                ]
                subject_records = _cross_val._run_spawned_fold_pool(
                    worker_target=_calibration_only_process_main,
                    worker_state=worker_state,
                    tasks=tasks,
                    n_workers=effective_n_jobs,
                    gpu_ids=normalized_gpu_ids,
                    cpus_per_worker=args.cpus_per_worker,
                    worker_name_prefix="CalibrationUserWorker",
                    worker_description="target-user calibration",
                )
                subject_records.sort(
                    key=lambda row: int(row["target_subject"])
                )
                for record in subject_records:
                    _log_user_metrics(logger, record)
            else:
                for user_index, spec in enumerate(model_specs, start=1):
                    active_user = int(spec["target_subject"])
                    logger.info(
                        "Configuration %d: calibrating user %s (%d/%d) from %s",
                        index,
                        active_user,
                        user_index,
                        len(model_specs),
                        spec["path"],
                    )
                    record = _run_one_user_calibration(
                        spec=spec,
                        configuration=configuration,
                        configuration_dir=configuration_dir,
                        feature_array=X,
                        label_array=y,
                        subject_id_array=subjects,
                        trial_id_array=trials,
                        calibration_levels=calibration_levels,
                        selection_shots=selection_shots,
                        args=args,
                    )
                    subject_records.append(record)
                    _log_user_metrics(logger, record)

            overall = _aggregate_subject_records(
                subject_records,
                calibration_levels=calibration_levels,
                selection_shots=selection_shots,
            )
            aggregate_results = {
                "configuration_id": index,
                "hyperparameters": configuration,
                "overall": overall,
                "subjects": subject_records,
            }
            _write_json(
                configuration_dir / "all_users_calibration_results.json",
                aggregate_results,
            )
            score = _selection_score_from_calibration_results(
                aggregate_results,
                selection_level="calibration",
                selection_metric=args.selection_metric,
                calibration_selection_shots=selection_shots,
            )
            summary = {
                "configuration_id": index,
                "status": "completed",
                "configuration_dir": str(configuration_dir),
                "hyperparameters": configuration,
                "n_subjects": len(subject_records),
                "selection_level": "calibration",
                "selection_metric": args.selection_metric,
                "selection_score": score,
                "zero_shot_all_trials_mean_scores": overall.get(
                    "zero_shot_all_trials_mean_scores", {}
                ),
                "paired_zero_shot_mean_scores": overall.get(
                    "paired_zero_shot_mean_scores", {}
                ),
                "calibrated_mean_scores": overall.get(
                    "calibrated_mean_scores", {}
                ),
                "delta_mean_scores": overall.get("delta_mean_scores", {}),
                "calibration_level_metrics": overall.get(
                    "calibration_levels", {}
                ),
                "subject_metrics": subject_records,
            }
            completed.append(summary)
            logger.info(
                "Completed configuration %d: average user %s=%s",
                index,
                args.selection_metric,
                score,
            )
            for shots, level in sorted(
                overall["calibration_levels"].items(),
                key=lambda item: int(item[0]),
            ):
                logger.info(
                    "Configuration %d average %s-shot metrics: %s",
                    index,
                    shots,
                    _json_fingerprint(level["calibrated_mean_scores"]),
                )
        except Exception as error:
            failed.append(
                {
                    "configuration_id": index,
                    "failed_target_subject": active_user,
                    "n_subjects_completed_before_failure": len(subject_records),
                    "hyperparameters": configuration,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            logger.exception(
                "Configuration %d failed while processing user %s",
                index,
                active_user,
            )
        finally:
            tf.keras.backend.clear_session()

        _write_json(
            progress_path,
            {
                "completed": completed,
                "failed": failed,
                "n_total": len(configurations),
            },
        )

    if not completed:
        raise RuntimeError(
            f"All {len(configurations)} calibration configurations failed. "
            f"See {progress_path}."
        )

    reverse = metric_mode(args.selection_metric) == "max"
    ranked = sorted(
        completed,
        key=lambda row: float(row["selection_score"]),
        reverse=reverse,
    )
    for rank, summary in enumerate(ranked, start=1):
        summary["rank"] = rank
    best = ranked[0]
    search_results = {
        "search_type": "selected_users_calibration_full_cartesian_product",
        "models_config_path": str(models_config_path),
        "n_models": len(model_specs),
        "target_subjects": [spec["target_subject"] for spec in model_specs],
        "selection_metric": args.selection_metric,
        "maximize_metric": reverse,
        "calibration_selection_shots": selection_shots,
        "aggregation_unit": "subject_fold_mean",
        "n_configurations": len(configurations),
        "n_completed": len(ranked),
        "n_failed": len(failed),
        "best_configuration_id": best["configuration_id"],
        "best_hyperparameters": best["hyperparameters"],
        "best_selection_score": best["selection_score"],
        "best_average_metrics": best["calibration_level_metrics"],
        "best_subject_metrics": best["subject_metrics"],
        "configurations": ranked,
        "failed_configurations": failed,
    }
    _write_json(run_dir / "calibration_search_results.json", search_results)
    _write_json(
        run_dir / "best_calibration_hyperparameters.json",
        {
            "configuration_id": best["configuration_id"],
            "selection_metric": args.selection_metric,
            "selection_score": best["selection_score"],
            "aggregation_unit": "subject_fold_mean",
            "hyperparameters": best["hyperparameters"],
            "average_metrics": best["calibration_level_metrics"],
            "subject_metrics": best["subject_metrics"],
            "configuration_dir": best["configuration_dir"],
        },
    )
    _write_csv(
        run_dir / "calibration_search_summary.csv",
        [_configuration_summary_row(row) for row in ranked],
    )
    logger.info(
        "Best configuration: id=%d average user %s=%s hyperparameters=%s",
        best["configuration_id"],
        args.selection_metric,
        best["selection_score"],
        _json_fingerprint(best["hyperparameters"]),
    )
    for shots, level in sorted(
        best["calibration_level_metrics"].items(),
        key=lambda item: int(item[0]),
    ):
        logger.info(
            "Best configuration average %s-shot metrics: %s",
            shots,
            _json_fingerprint(level["calibrated_mean_scores"]),
        )
    logger.info("Best configuration per-user metrics:")
    for record in best["subject_metrics"]:
        _log_user_metrics(logger, record)
    logger.info("Saved calibration-only search artifacts to %s", run_dir)
    return {"run_dir": str(run_dir), "results": search_results}


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    configurations, dimensions, _ = _calibration_grid_from_args(args)
    if args.print_grid_only:
        print(
            json.dumps(
                {
                    "search_type": "calibration_only_full_cartesian_product",
                    "dimensions": dimensions,
                    "n_configurations": len(configurations),
                    "configurations": configurations,
                },
                indent=2,
                default=_json_default,
            )
        )
        return 0
    run_calibration_only_search(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
