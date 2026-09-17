"""Command-line and configuration parsing for SIC training.

This module deliberately contains no TensorFlow or dataset-loading code.  It
owns the CLI surface, JSON Cartesian-grid decoding, configuration validation,
and conversion of parsed arguments into the lightweight training dataclass.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from itertools import product
import json
from pathlib import Path


@dataclass(slots=True)
class SICTrainingConfig:
    output_dir: Path = Path("runs/sic")
    run_name: str = "dreamer_valence_sic"
    dataset: str = "dreamer"
    n_channels: int = 14
    n_bands: int = 3
    classification_level: str = "window"

    training_protocol: str = "subject_calibration"
    source_epochs: int = 100
    source_batch_size: int = 8

    validation_subjects: int = 4
    validation_seed: int | None = 42
    early_stopping_patience: int | None = 10
    early_stopping_min_delta: float = 0.001
    early_stopping_monitor: str = "val_loss"
    early_stopping_mode: str = "auto"
    best_epoch_metric: str | None = None
    selection_metric: str = "balanced_accuracy"
    hyperparameter_selection_level: str = "calibration"
    restore_best_weights: bool = True
    calibration_epochs: int = 30
    calibration_batch_size: int = 6
    calibration_trials: int = 6
    calibration_folds: int = 3
    calibration_levels: tuple[tuple[int, int], ...] = ()
    calibration_selection_shots: int | None = None
    calibration_learning_rate: float = 1e-4
    calibration_optimizer: str = "adamw"
    calibration_weight_decay: float = 0.0
    calibration_seed: int | None = 42
    stratify_calibration: bool = True

    decision_threshold: float = 0.5
    ece_bins: int = 15
    prediction_diagnostics: bool = False
    prediction_diagnostics_metric: str = "accuracy"
    prediction_diagnostics_every_n_epochs: int = 1
    prediction_diagnostics_max_samples: int = 256
    prediction_diagnostics_threshold_tolerance: float = 0.01
    prediction_diagnostics_seed: int | None = 42

    source_use_class_weight: bool = False
    calibration_use_class_weight: bool = False
    n_jobs: int = 1
    gpu_ids: tuple[int, ...] | None = None
    cpus_per_worker: int | None = None
    max_subjects: int | None = None
    verbose: int = 1
    seed: int | None = 42

    label_threshold_mode: str = "global"
    median_label: float = 3.0
    window_normalization: str = "global_rms"
    model_config: dict = field(default_factory=dict)


# A single list is a fixed value for these sequence-valued model arguments;
# nested lists (or an explicit {"grid": ...} wrapper) define candidates.
SIC_SEQUENCE_HYPERPARAMETER_DEPTHS = {
    "gcn_units": 1,
    "temporal_pool_sizes": 1,
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


def calibration_plan(config: SICTrainingConfig) -> tuple[tuple[int, int], ...]:
    return config.calibration_levels or (
        (int(config.calibration_trials), int(config.calibration_folds)),
    )


def selected_calibration_shots(config: SICTrainingConfig) -> int:
    levels = calibration_plan(config)
    return (
        max(shots for shots, _ in levels)
        if config.calibration_selection_shots is None
        else int(config.calibration_selection_shots)
    )


def selection_score_source(config: SICTrainingConfig) -> str:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train SIC with independent deterministic GCN-GRU and BiLSTM "
            "branches, direct feature concatenation, "
            "ERM/V-REx/first-order MLDG, VC regularization, and multi-level "
            "subject calibration."
        )
    )
    parser.add_argument("--out-dir", default="runs/sic")
    parser.add_argument("--run-name", default="dreamer_valence_sic")
    parser.add_argument(
        "--dataset", default="dreamer", choices=("dreamer", "amigos", "eegemotions_27")
    )
    parser.add_argument("--raw-eeg-npy", default=None)
    parser.add_argument("--raw-labels-npy", default=None)
    parser.add_argument(
        "--label-dimension", choices=("valence", "arousal"), default="valence"
    )
    parser.add_argument(
        "--label-threshold-mode", choices=("global", "subject_median"), default="global"
    )
    parser.add_argument("--median-label", type=float, default=3.0)
    parser.add_argument(
        "--remove-median-label",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Remove every complete trial whose original rating equals --median-label.",
    )
    parser.add_argument("--window-sec", type=float, default=4.0)
    parser.add_argument("--window-overlap", type=float, default=0.0)
    parser.add_argument("--fs", type=float, default=30.0)
    parser.add_argument(
        "--window-normalization",
        choices=("none", "global_rms", "feature_zscore"),
        default="global_rms",
    )
    parser.add_argument("--n-channels", type=_positive_int, default=14)
    parser.add_argument("--n-bands", type=_positive_int, default=3)
    parser.add_argument(
        "--classification-level", choices=("trial", "window"), default="window"
    )

    parser.add_argument(
        "--training-protocol",
        choices=("subject_calibration", "loso_validation"),
        default="subject_calibration",
    )
    parser.add_argument("--source-epochs", type=_positive_int, default=100)
    parser.add_argument("--source-batch-size", type=_positive_int, default=8)
    parser.add_argument(
        "--training-method", choices=("erm", "vrex", "mldg"), default=None
    )
    parser.add_argument("--validation-subjects", type=int, default=4)
    parser.add_argument("--validation-seed", type=int, default=42)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--no-early-stopping", action="store_true")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    parser.add_argument("--early-stopping-monitor", default="val_loss")
    parser.add_argument(
        "--best-epoch-metric",
        choices=("accuracy", "balanced_accuracy", "roc_auc", "brier_score", "ece"),
        default=None,
    )
    parser.add_argument(
        "--selection-metric",
        choices=(
            "accuracy", "balanced_accuracy", "f1", "macro_f1", "precision",
            "macro_precision", "recall", "macro_recall", "roc_auc",
            "brier_score", "ece",
        ),
        default="balanced_accuracy",
    )
    parser.add_argument(
        "--hyperparameter-selection-level",
        choices=("losocv", "calibration"),
        default="calibration",
    )
    parser.add_argument("--early-stopping-mode", choices=("auto", "min", "max"), default="auto")
    parser.add_argument("--no-restore-best-weights", action="store_true")

    parser.add_argument("--calibration-epochs", type=_positive_int, default=30)
    parser.add_argument("--calibration-batch-size", type=_positive_int, default=6)
    parser.add_argument("--calibration-trials", type=_positive_int, default=6)
    parser.add_argument("--calibration-folds", type=_positive_int, default=3)
    parser.add_argument(
        "--calibration-level",
        type=_positive_int,
        nargs=2,
        action="append",
        metavar=("SHOTS", "FOLDS"),
        default=None,
    )
    parser.add_argument("--calibration-selection-shots", type=_positive_int, default=None)
    parser.add_argument("--calibration-learning-rate", type=float, default=1e-4)
    parser.add_argument("--calibration-optimizer", choices=("adam", "adamw"), default="adamw")
    parser.add_argument("--calibration-weight-decay", type=float, default=0.0)
    parser.add_argument("--calibration-seed", type=int, default=42)
    parser.add_argument("--no-stratify-calibration", action="store_true")

    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--ece-bins", type=_positive_int, default=15)
    parser.add_argument(
        "--prediction-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Record prediction diagnostics during LOSO source training.",
    )
    parser.add_argument(
        "--prediction-diagnostics-metric",
        choices=SIC_CLASSIFICATION_METRICS,
        default="accuracy",
        help="Classification metric highlighted in prediction diagnostic rows.",
    )
    parser.add_argument(
        "--prediction-diagnostics-every-n-epochs",
        type=_positive_int,
        default=1,
    )
    parser.add_argument(
        "--prediction-diagnostics-max-samples",
        type=_positive_int,
        default=256,
    )
    parser.add_argument(
        "--prediction-diagnostics-threshold-tolerance",
        type=float,
        default=0.01,
    )
    parser.add_argument("--prediction-diagnostics-seed", type=int, default=42)
    parser.add_argument("--source-use-class-weight", action="store_true")
    parser.add_argument("--calibration-use-class-weight", action="store_true")

    parser.add_argument("--n-jobs", type=_positive_int, default=1)
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=None)
    parser.add_argument("--cpus-per-worker", type=_positive_int, default=None)
    parser.add_argument("--max-subjects", type=_positive_int, default=None)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-gcn-gru-branch",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--use-bilstm-branch",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--hyperparameters-json",
        default=None,
        help=(
            "Fixed SIC configuration or Cartesian search grid as JSON. The "
            "complete GCN-GRU and BiLSTM encoder features are concatenated; "
            "there is no z projection or decoder configuration. "
            "Use {\"grid\":[...]} and {\"fixed\":...} for unambiguous values."
        ),
    )
    parser.add_argument("--print-grid-only", action="store_true")
    return parser


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
        "remove_median_label",
    ):
        cli_value = getattr(args, key)
        if cli_value is not None:
            model_config[key] = bool(cli_value)
    return model_config


def validate_args(args, model_config: dict) -> None:
    if args.validation_subjects < 0:
        raise ValueError("--validation-subjects must be >= 0.")
    if args.validation_subjects == 0 and args.best_epoch_metric is not None:
        raise ValueError("--best-epoch-metric requires source validation subjects.")
    if args.early_stopping_patience is not None and args.early_stopping_patience < 1:
        raise ValueError("--early-stopping-patience must be >= 1.")
    if args.early_stopping_min_delta < 0.0:
        raise ValueError("--early-stopping-min-delta must be non-negative.")

    calibration_levels = calibration_levels_from_args(args)
    shot_levels = [shots for shots, _ in calibration_levels]
    if len(set(shot_levels)) != len(shot_levels):
        raise ValueError("Each --calibration-level SHOTS value must be unique.")
    if args.dataset == "dreamer" and any(shots >= 18 for shots in shot_levels):
        raise ValueError("DREAMER calibration shot levels must be < 18.")
    if (
        args.calibration_selection_shots is not None
        and args.calibration_selection_shots not in shot_levels
    ):
        raise ValueError(
            "--calibration-selection-shots must match one configured level; "
            f"available levels are {shot_levels}."
        )
    if args.calibration_learning_rate <= 0.0:
        raise ValueError("calibration-learning-rate must be positive.")
    if args.calibration_weight_decay < 0.0:
        raise ValueError("calibration-weight-decay must be non-negative.")
    if not 0.0 < args.decision_threshold < 1.0:
        raise ValueError("decision-threshold must be in (0, 1).")
    if args.ece_bins < 2:
        raise ValueError("--ece-bins must be >= 2.")
    if args.prediction_diagnostics_threshold_tolerance < 0.0:
        raise ValueError(
            "--prediction-diagnostics-threshold-tolerance must be non-negative."
        )
    if "epochs" in model_config or "batch_size" in model_config:
        raise ValueError(
            "Do not put epochs/batch_size inside --hyperparameters-json; use "
            "the source/calibration CLI arguments."
        )

    configurations, grid_dimensions = expand_cartesian_grid(model_config)
    for index, configuration in enumerate(configurations, start=1):
        builder_config, _ = split_data_hyperparameters(configuration)
        training_method = configuration_training_method(builder_config)
        if (
            builder_config.get("use_gcn_gru_branch") is False
            and builder_config.get("use_bilstm_branch") is False
        ):
            raise ValueError(
                "At least one encoder branch must be active; grid configuration "
                f"{index} disables both."
            )
        if training_method == "mldg":
            meta_train_subjects = builder_config.get("mldg_meta_train_subjects")
            meta_test_subjects = int(builder_config.get("mldg_meta_test_subjects", 4))
            trials_per_subject = int(builder_config.get("mldg_trials_per_subject", 1))
            steps_per_epoch = builder_config.get("mldg_steps_per_epoch")
            inner_lr = float(builder_config.get("mldg_inner_learning_rate", 1e-4))
            meta_weight = float(builder_config.get("mldg_meta_test_weight", 1.0))
            if meta_train_subjects is not None and int(meta_train_subjects) < 1:
                raise ValueError("mldg_meta_train_subjects must be >= 1 or null.")
            if meta_test_subjects < 1 or trials_per_subject < 1:
                raise ValueError("MLDG subject/trial counts must be >= 1.")
            if steps_per_epoch is not None and int(steps_per_epoch) < 1:
                raise ValueError("mldg_steps_per_epoch must be >= 1 or null.")
            if inner_lr <= 0.0 or meta_weight < 0.0:
                raise ValueError("MLDG inner LR must be positive and weight non-negative.")
            if args.dataset == "dreamer" and meta_train_subjects is not None:
                available = 22 - args.validation_subjects
                if int(meta_train_subjects) + meta_test_subjects > available:
                    raise ValueError(
                        f"MLDG configuration {index} requests more subjects than "
                        f"the {available} available source-training subjects."
                    )

    if args.training_protocol == "subject_calibration" and grid_dimensions:
        raise ValueError(
            "Hyperparameter grids require --training-protocol loso_validation; "
            f"grid-valued keys were {sorted(grid_dimensions)}."
        )


def parse_cli(argv=None):
    """Parse and validate CLI input, returning ``(args, model_config)``."""
    args = build_parser().parse_args(argv)
    model_config = model_config_from_args(args)
    validate_args(args, model_config)
    return args, model_config


def training_config_from_args(
    args,
    model_config: dict,
    calibration_levels: tuple[tuple[int, int], ...] | None = None,
) -> SICTrainingConfig:
    """Convert validated CLI values into the runtime training configuration."""
    levels = calibration_levels or calibration_levels_from_args(args)
    selection_shots = (
        max(shots for shots, _ in levels)
        if args.calibration_selection_shots is None
        else int(args.calibration_selection_shots)
    )
    return SICTrainingConfig(
        output_dir=Path(args.out_dir),
        run_name=args.run_name,
        dataset=args.dataset,
        n_channels=args.n_channels,
        n_bands=args.n_bands,
        classification_level=args.classification_level,
        training_protocol=args.training_protocol,
        source_epochs=args.source_epochs,
        source_batch_size=args.source_batch_size,
        validation_subjects=args.validation_subjects,
        validation_seed=args.validation_seed,
        early_stopping_patience=(
            None
            if args.no_early_stopping or args.validation_subjects == 0
            else args.early_stopping_patience
        ),
        early_stopping_min_delta=args.early_stopping_min_delta,
        early_stopping_monitor=args.early_stopping_monitor,
        early_stopping_mode=args.early_stopping_mode,
        best_epoch_metric=args.best_epoch_metric,
        selection_metric=args.selection_metric,
        hyperparameter_selection_level=args.hyperparameter_selection_level,
        restore_best_weights=(
            not args.no_restore_best_weights and args.validation_subjects > 0
        ),
        calibration_epochs=args.calibration_epochs,
        calibration_batch_size=args.calibration_batch_size,
        calibration_trials=args.calibration_trials,
        calibration_folds=args.calibration_folds,
        calibration_levels=levels,
        calibration_selection_shots=selection_shots,
        calibration_learning_rate=args.calibration_learning_rate,
        calibration_optimizer=args.calibration_optimizer,
        calibration_weight_decay=args.calibration_weight_decay,
        calibration_seed=args.calibration_seed,
        stratify_calibration=not args.no_stratify_calibration,
        decision_threshold=args.decision_threshold,
        ece_bins=args.ece_bins,
        prediction_diagnostics=args.prediction_diagnostics,
        prediction_diagnostics_metric=args.prediction_diagnostics_metric,
        prediction_diagnostics_every_n_epochs=(
            args.prediction_diagnostics_every_n_epochs
        ),
        prediction_diagnostics_max_samples=args.prediction_diagnostics_max_samples,
        prediction_diagnostics_threshold_tolerance=(
            args.prediction_diagnostics_threshold_tolerance
        ),
        prediction_diagnostics_seed=args.prediction_diagnostics_seed,
        source_use_class_weight=args.source_use_class_weight,
        calibration_use_class_weight=args.calibration_use_class_weight,
        n_jobs=args.n_jobs,
        gpu_ids=None if args.gpu_ids is None else tuple(args.gpu_ids),
        cpus_per_worker=args.cpus_per_worker,
        max_subjects=args.max_subjects,
        verbose=args.verbose,
        seed=args.seed,
        label_threshold_mode=args.label_threshold_mode,
        median_label=args.median_label,
        window_normalization=args.window_normalization,
        model_config=model_config,
    )
