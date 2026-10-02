"""Subject-calibration cross-validation entry point."""

from __future__ import annotations

import numpy as np
import os

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotations only
    from typing import Callable
    from typing import Literal
    import tensorflow as tf

from .arrays import _python_scalar
from .constants import _CLASSIFICATION_METRICS, _DECODER_SCORE_NAMES, _DEFAULT_ECE_BINS
from .workers import _auto_assign_gpu_ids, _run_spawned_fold_pool
from .probabilities import _normalize_decision_thresholds
from .reporting import _mean_std_rows, _validate_evaluation_level
from .calibration_subject import _run_subject_calibration_subject, _subject_calibration_process_main

def _normalize_calibration_levels(
    calibration_levels,
    *,
    calibration_trials: int,
    calibration_folds: int,
) -> tuple[tuple[int, int], ...]:
    """Return unique ``(shots, folds)`` pairs in requested reporting order."""
    raw_levels = (
        [(calibration_trials, calibration_folds)]
        if calibration_levels is None
        else list(calibration_levels)
    )
    if not raw_levels:
        raise ValueError("calibration_levels must contain at least one pair.")

    normalized: list[tuple[int, int]] = []
    seen_shots: set[int] = set()
    for pair in raw_levels:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(
                "Each calibration level must be a (shots, folds) pair; "
                f"got {pair!r}."
            )
        shots, folds = (int(pair[0]), int(pair[1]))
        if shots < 1:
            raise ValueError("Calibration shots must be >= 1.")
        if folds < 1:
            raise ValueError("Calibration folds must be >= 1.")
        if shots in seen_shots:
            raise ValueError(
                f"Calibration shots must be unique; {shots}-shot was repeated."
            )
        seen_shots.add(shots)
        normalized.append((shots, folds))
    return tuple(normalized)


def subject_calibration_cv(
    model_builder_function: Callable[..., tf.keras.Model] | None,
    feature_array: np.ndarray,
    label_array: np.ndarray,
    subject_id_array: np.ndarray,
    trial_id_array: np.ndarray,
    fixed_config: dict | None = None,
    source_epochs: int = 100,
    source_batch_size: int = 64,
    calibration_epochs: int = 20,
    calibration_batch_size: int = 6,
    *,
    pretrained_model: tf.keras.Model | None = None,
    calibration_trials: int = 6,
    calibration_folds: int = 3,
    calibration_levels: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None = None,
    calibration_selection_shots: int | None = None,
    calibration_learning_rate: float = 1e-4,
    calibration_optimizer: str = "adamw",
    calibration_weight_decay: float = 0.0,
    calibration_seed: int | None = 42,
    stratify_calibration: bool = True,
    allow_overlapping_calibration_folds: bool = False,
    validation_subjects_per_fold: int = 0,
    validation_seed: int | None = 42,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    early_stopping_monitor: str = "val_loss",
    early_stopping_mode: Literal["auto", "min", "max"] = "auto",
    restore_best_weights: bool = True,
    evaluation_level: Literal["window", "trial"] = "trial",
    metrics: list[str] | tuple[str, ...] = (
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
    ),
    ece_bins: int = _DEFAULT_ECE_BINS,
    decision_threshold: float = 0.5,
    prediction_diagnostics_thresholds: (
        list[float] | tuple[float, ...] | None
    ) = None,
    prediction_diagnostics: bool = False,
    prediction_diagnostics_metric: str = "accuracy",
    prediction_diagnostics_every_n_epochs: int = 1,
    prediction_diagnostics_max_samples: int = 256,
    prediction_diagnostics_threshold_tolerance: float = 0.01,
    prediction_diagnostics_seed: int | None = 42,
    n_prediction_latent_samples: int = 0,
    latent_sampling_seed: int | None = None,
    log_predictions: bool = True,
    log_variational_intervals: bool = False,
    n_uncertainty_samples: int = 30,
    ci_level: float = 0.95,
    source_use_class_weight: bool = False,
    calibration_use_class_weight: bool = False,
    source_fit_kwargs: dict | None = None,
    calibration_fit_kwargs: dict | None = None,
    calibration_verbose: bool = False,
    calibration_print_every_n_epochs: int = 1,
    verbose: int = 0,
    n_jobs: int = 1,
    gpu_ids: list[int] | tuple[int, ...] | None = None,
    cpus_per_worker: int | None = None,
    max_subjects: int | None = None,
    target_subjects: list[int] | tuple[int, ...] | None = None,
    source_model_output_dir: str | os.PathLike[str] | None = None,
) -> dict:
    """Strict LOSO pretraining followed by one or more calibration levels.

    Pass exactly one of ``model_builder_function`` and ``pretrained_model``.
    The pretrained path accepts one already-loaded, target-specific LOSO model,
    requires one explicit ``target_subjects`` entry, and skips source fitting.

    For each target subject, a fresh subject-independent source model is first
    trained using the other subjects. When ``validation_subjects_per_fold`` is
    positive, a seeded subset of those source subjects is excluded from gradient
    updates and used only for source checkpoint selection. The target subject is
    never used to construct, optimize, or select that source model. This is
    important for models that learn data-dependent adjacency: the four ``training_*``
    arrays contain only gradient-training source subjects, so its fixed
    mutual-information adjacency remains training-only.

    The source model is evaluated zero-shot on all target trials once. Pass
    ``calibration_levels=((3, 6), (6, 3), ...)`` to request ``(shots, folds)``
    pairs. If omitted, the legacy ``calibration_trials``/``calibration_folds``
    pair is used. The same source checkpoint is restored for every level and
    fold; it is never retrained between calibration levels. Exact partitions
    are used when ``shots * folds`` equals the retained trial count, otherwise
    seeded repeated holdouts evaluate on every non-calibration trial. For each
    fold:

      1. restore the exact source-pretrained weights;
      2. evaluate zero-shot on the trials not used for calibration;
      3. call ``model.prepare_for_subject_calibration(...)``;
      4. fine-tune only the model-defined calibration parameters, printing
         reporting-only oracle predictions on every non-calibration trial
         after each epoch;
      5. evaluate the calibrated model on the same held-out target trials.

    When ``source_model_output_dir`` is provided, the exact source-trained
    0-shot model for every LOSO target is saved there in native ``.keras``
    format before calibration begins. Calibrated fold continuations are not
    saved. Load these source models with ``compile=False`` before calling the
    model's calibration preparation hook, which installs a fresh optimizer.
    Model-specific calibration settings may optionally keep reconstruction
    decoders trainable and add their weighted loss to the calibration objective.

    Thus the default DREAMER protocol remains 6 calibration trials + 12
    evaluation trials repeated three times, while arbitrary valid shot/fold
    pairs can be evaluated in the same LOSO run.

    ``model_builder_function`` contract
    ------------------------------------
    In addition to ``fixed_config`` model kwargs, the builder must accept:
    ``training_features``, ``training_labels``, ``training_subject_ids``, and
    ``training_trial_ids``. The first of these is where v6 should estimate its
    training-only MI graph.

    ``prepare_for_subject_calibration`` contract
    --------------------------------------------
    The returned model must implement::

        model.prepare_for_subject_calibration(
            learning_rate=...,
            optimizer_name=...,
            weight_decay=...,
        )

    The method owns the architecture-specific freezing policy and must compile
    a *fresh optimizer*. A typical policy freezes the shared representation
    layers — graph, spectral and temporal encoders, plus any subject-invariance
    machinery — while the classification head stays trainable; reconstruction
    decoders may also remain trainable when the model's calibration settings
    request it.

    Aggregation
    -----------
    Metrics are first averaged across folds within each subject and shot level,
    then across subjects. Calibration folds are never treated as independent
    subjects. JSON output retains every requested metric at every shot level;
    terminal output reports final mean accuracy, balanced accuracy, and Brier
    score for 0-shot and each calibrated level.
    """
    feature_array = np.asarray(feature_array)
    label_array = np.asarray(label_array)
    subject_id_array = np.asarray(subject_id_array).reshape(-1)
    trial_id_array = np.asarray(trial_id_array).reshape(-1)
    fixed_config = dict(fixed_config or {})
    source_fit_kwargs = dict(source_fit_kwargs or {})
    calibration_fit_kwargs = dict(calibration_fit_kwargs or {})
    calibration_levels = _normalize_calibration_levels(
        calibration_levels,
        calibration_trials=calibration_trials,
        calibration_folds=calibration_folds,
    )
    calibration_level_shots = tuple(shots for shots, _ in calibration_levels)
    if calibration_selection_shots is None:
        calibration_selection_shots = max(calibration_level_shots)
    calibration_selection_shots = int(calibration_selection_shots)
    if calibration_selection_shots not in calibration_level_shots:
        raise ValueError(
            "calibration_selection_shots must match one configured shot level; "
            f"got {calibration_selection_shots}, available={calibration_level_shots}."
        )

    if feature_array.ndim not in {3, 4}:
        raise ValueError(
            "feature_array must be rank 3 (window samples) or rank 4 "
            f"(grouped trial samples); got {feature_array.shape}."
        )
    lengths = (
        len(feature_array),
        len(label_array),
        len(subject_id_array),
        len(trial_id_array),
    )
    if len(set(lengths)) != 1:
        raise ValueError(
            "feature_array, label_array, subject_id_array, and trial_id_array "
            f"must align; got lengths {lengths}."
        )
    if (model_builder_function is None) == (pretrained_model is None):
        raise ValueError(
            "Pass exactly one of model_builder_function or pretrained_model."
        )
    if pretrained_model is not None and fixed_config:
        raise ValueError(
            "fixed_config applies only when building a source model; configure "
            "the loaded model before passing pretrained_model."
        )
    if pretrained_model is None and source_epochs < 1:
        raise ValueError("source_epochs must be >= 1 when fitting a source model.")
    if calibration_epochs < 1:
        raise ValueError("calibration_epochs must be >= 1.")
    if int(calibration_print_every_n_epochs) < 1:
        raise ValueError("calibration_print_every_n_epochs must be >= 1.")
    if source_batch_size < 1 or calibration_batch_size < 1:
        raise ValueError(
            "source_batch_size and calibration_batch_size must be >= 1."
        )
    if validation_subjects_per_fold < 0:
        raise ValueError("validation_subjects_per_fold must be >= 0.")
    if early_stopping_patience is not None and early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be >= 1 when provided.")
    if early_stopping_min_delta < 0.0:
        raise ValueError("early_stopping_min_delta must be non-negative.")
    if early_stopping_mode not in {"auto", "min", "max"}:
        raise ValueError("early_stopping_mode must be 'auto', 'min', or 'max'.")
    if not isinstance(early_stopping_monitor, str) or not early_stopping_monitor:
        raise ValueError("early_stopping_monitor must be a non-empty string.")
    if calibration_learning_rate <= 0.0:
        raise ValueError("calibration_learning_rate must be positive.")
    if calibration_weight_decay < 0.0:
        raise ValueError("calibration_weight_decay must be non-negative.")
    if calibration_optimizer not in {"adam", "adamw"}:
        raise ValueError("calibration_optimizer must be 'adam' or 'adamw'.")
    decision_threshold = float(decision_threshold)
    if not 0.0 < decision_threshold < 1.0:
        raise ValueError("decision_threshold must lie strictly between 0 and 1.")
    raw_diagnostic_thresholds = (
        (decision_threshold,)
        if prediction_diagnostics_thresholds is None
        else (*prediction_diagnostics_thresholds, decision_threshold)
    )
    normalized_diagnostic_thresholds = _normalize_decision_thresholds(
        raw_diagnostic_thresholds
    )
    prediction_diagnostics_thresholds = (
        decision_threshold,
        *(
            threshold
            for threshold in normalized_diagnostic_thresholds
            if threshold != decision_threshold
        ),
    )
    prediction_diagnostics_metric = str(
        prediction_diagnostics_metric
    ).strip().lower()
    if prediction_diagnostics_metric not in _CLASSIFICATION_METRICS:
        raise ValueError(
            "prediction_diagnostics_metric must be one of "
            f"{sorted(_CLASSIFICATION_METRICS)}; got "
            f"{prediction_diagnostics_metric!r}."
        )
    if int(prediction_diagnostics_every_n_epochs) < 1:
        raise ValueError("prediction_diagnostics_every_n_epochs must be >= 1.")
    if int(prediction_diagnostics_max_samples) < 1:
        raise ValueError("prediction_diagnostics_max_samples must be >= 1.")
    if float(prediction_diagnostics_threshold_tolerance) < 0.0:
        raise ValueError(
            "prediction_diagnostics_threshold_tolerance must be non-negative."
        )
    if n_prediction_latent_samples < 0:
        raise ValueError("n_prediction_latent_samples must be >= 0.")
    if int(ece_bins) < 2:
        raise ValueError("ece_bins must be >= 2.")
    if not 0.0 < float(ci_level) < 1.0:
        raise ValueError("ci_level must lie between 0 and 1.")
    if n_jobs < 1:
        raise ValueError("n_jobs must be >= 1.")
    if cpus_per_worker is not None and cpus_per_worker < 1:
        raise ValueError("cpus_per_worker must be >= 1 when provided.")
    if max_subjects is not None and int(max_subjects) < 1:
        raise ValueError("max_subjects must be >= 1 when provided.")
    if source_model_output_dir is not None:
        source_model_output_dir = os.path.abspath(
            os.fspath(source_model_output_dir)
        )
        os.makedirs(source_model_output_dir, exist_ok=True)
    if feature_array.ndim == 4 and evaluation_level != "trial":
        raise ValueError(
            "Rank-4 grouped-trial inputs require evaluation_level='trial'."
        )
    _validate_evaluation_level(evaluation_level, "evaluation_level")

    for forbidden_key in ("epochs", "batch_size"):
        if forbidden_key in fixed_config:
            raise ValueError(
                f"Remove {forbidden_key!r} from fixed_config; "
                "subject_calibration_cv has separate source/calibration values."
            )
    for fit_name, fit_kwargs in (
        ("source_fit_kwargs", source_fit_kwargs),
        ("calibration_fit_kwargs", calibration_fit_kwargs),
    ):
        duplicates = {"epochs", "batch_size", "verbose", "class_weight"}.intersection(
            fit_kwargs
        )
        if duplicates:
            raise ValueError(
                f"{fit_name} must not override managed fit arguments: "
                f"{sorted(duplicates)}."
            )
        if "validation_data" in fit_kwargs:
            raise ValueError(
                f"{fit_name} must not provide validation_data. Source-validation "
                "subjects are managed by subject_calibration_cv, the target "
                "subject cannot select the source model, and the six calibration "
                "trials are intentionally used only for fitting."
            )

    metrics = tuple(metrics)
    for metric in metrics:
        if metric not in _CLASSIFICATION_METRICS:
            raise ValueError(
                f"Unsupported metric {metric!r}. Supported metrics: "
                f"{sorted(_CLASSIFICATION_METRICS)}"
            )

    unique_subjects = np.sort(np.unique(subject_id_array))
    if len(unique_subjects) < 2:
        raise ValueError("subject_calibration_cv requires at least two subjects.")
    if (
        pretrained_model is None
        and validation_subjects_per_fold >= len(unique_subjects) - 1
    ):
        raise ValueError(
            "validation_subjects_per_fold must leave at least one non-target "
            "subject for source fitting. Got "
            f"{validation_subjects_per_fold} validation subjects with "
            f"{len(unique_subjects)} total subjects."
        )
    if target_subjects is None:
        target_subjects = unique_subjects
        if max_subjects is not None:
            target_subjects = target_subjects[: int(max_subjects)]
    else:
        target_subjects = np.asarray(target_subjects)
        missing_subjects = target_subjects[
            ~np.isin(target_subjects, unique_subjects)
        ]
        if len(missing_subjects):
            raise ValueError(
                f"Unknown target subject IDs: {missing_subjects.tolist()}. "
                f"Available subjects: {unique_subjects.tolist()}."
            )
    total_subjects = int(len(target_subjects))
    if pretrained_model is not None:
        if total_subjects != 1:
            raise ValueError(
                "A pretrained_model must be paired with exactly one explicit "
                "target subject. Pass target_subjects=[USER_ID]."
            )
        if validation_subjects_per_fold != 0:
            raise ValueError(
                "validation_subjects_per_fold must be 0 when source fitting is skipped."
            )
        if n_jobs != 1:
            raise ValueError(
                "pretrained_model runs require n_jobs=1 because a live Keras "
                "model is passed directly rather than serialized to workers."
            )

    # Fail early if a usable calibration/evaluation split cannot be formed for
    # any selected subject, rather than discovering it after pretraining.
    for target_subject in target_subjects:
        target_trial_count = len(
            np.unique(trial_id_array[subject_id_array == target_subject])
        )
        invalid_shots = [
            shots for shots, _ in calibration_levels if shots >= target_trial_count
        ]
        if invalid_shots:
            raise ValueError(
                f"Target subject {_python_scalar(target_subject)!r} has "
                f"{target_trial_count} retained trials. Every calibration level "
                "must leave at least one evaluation trial; invalid shot levels="
                f"{invalid_shots}."
            )

    effective_n_jobs = min(int(n_jobs), total_subjects)
    normalized_gpu_ids: tuple[int, ...] | None = None
    if gpu_ids is None and effective_n_jobs > 1:
        normalized_gpu_ids = _auto_assign_gpu_ids(effective_n_jobs)
        if normalized_gpu_ids is not None:
            effective_n_jobs = len(normalized_gpu_ids)
    elif gpu_ids is not None:
        normalized_gpu_ids = tuple(int(gpu_id) for gpu_id in gpu_ids)
        if not normalized_gpu_ids:
            raise ValueError("gpu_ids must contain at least one GPU index.")
        if len(set(normalized_gpu_ids)) != len(normalized_gpu_ids):
            raise ValueError("gpu_ids must not contain duplicate GPU indices.")
        if effective_n_jobs > len(normalized_gpu_ids):
            raise ValueError(
                f"n_jobs={effective_n_jobs} requires at least that many GPU IDs; "
                f"got gpu_ids={normalized_gpu_ids}."
            )
        normalized_gpu_ids = normalized_gpu_ids[:effective_n_jobs]

    tasks = [
        (subject_number, target_subject)
        for subject_number, target_subject in enumerate(target_subjects, start=1)
    ]
    worker_state = {
        "total_subjects": total_subjects,
        "model_builder_function": model_builder_function,
        "pretrained_model": pretrained_model,
        "feature_array": feature_array,
        "label_array": label_array,
        "subject_id_array": subject_id_array,
        "trial_id_array": trial_id_array,
        "fixed_config": fixed_config,
        "source_epochs": int(source_epochs),
        "source_batch_size": int(source_batch_size),
        "validation_subjects_per_fold": int(validation_subjects_per_fold),
        "validation_seed": validation_seed,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": float(early_stopping_min_delta),
        "early_stopping_monitor": str(early_stopping_monitor),
        "early_stopping_mode": str(early_stopping_mode),
        "restore_best_weights": bool(restore_best_weights),
        "calibration_epochs": int(calibration_epochs),
        "calibration_batch_size": int(calibration_batch_size),
        "calibration_trials": int(calibration_trials),
        "calibration_folds": int(calibration_folds),
        "calibration_levels": calibration_levels,
        "calibration_learning_rate": float(calibration_learning_rate),
        "calibration_optimizer": str(calibration_optimizer),
        "calibration_weight_decay": float(calibration_weight_decay),
        "calibration_seed": calibration_seed,
        "stratify_calibration": bool(stratify_calibration),
        "allow_overlapping_calibration_folds": bool(
            allow_overlapping_calibration_folds
        ),
        "evaluation_level": evaluation_level,
        "metrics": metrics,
        "ece_bins": int(ece_bins),
        "decision_threshold": decision_threshold,
        "prediction_diagnostics_thresholds": (
            prediction_diagnostics_thresholds
        ),
        "prediction_diagnostics": bool(prediction_diagnostics),
        "prediction_diagnostics_metric": prediction_diagnostics_metric,
        "prediction_diagnostics_every_n_epochs": int(
            prediction_diagnostics_every_n_epochs
        ),
        "prediction_diagnostics_max_samples": int(
            prediction_diagnostics_max_samples
        ),
        "prediction_diagnostics_threshold_tolerance": float(
            prediction_diagnostics_threshold_tolerance
        ),
        "prediction_diagnostics_seed": prediction_diagnostics_seed,
        "n_prediction_latent_samples": int(n_prediction_latent_samples),
        "latent_sampling_seed": latent_sampling_seed,
        "log_predictions": bool(log_predictions),
        "log_variational_intervals": bool(log_variational_intervals),
        "n_uncertainty_samples": int(n_uncertainty_samples),
        "ci_level": float(ci_level),
        "source_use_class_weight": bool(source_use_class_weight),
        "calibration_use_class_weight": bool(calibration_use_class_weight),
        "source_fit_kwargs": source_fit_kwargs,
        "calibration_fit_kwargs": calibration_fit_kwargs,
        "calibration_verbose": bool(calibration_verbose),
        "calibration_print_every_n_epochs": int(
            calibration_print_every_n_epochs
        ),
        "source_model_output_dir": source_model_output_dir,
        "verbose": int(verbose),
    }

    if effective_n_jobs == 1 and normalized_gpu_ids is None:
        subject_outputs = [
            _run_subject_calibration_subject(
                subject_number=subject_number,
                target_subject=target_subject,
                **worker_state,
            )
            for subject_number, target_subject in tasks
        ]
    else:
        subject_outputs = _run_spawned_fold_pool(
            worker_target=_subject_calibration_process_main,
            worker_state=worker_state,
            tasks=tasks,
            n_workers=effective_n_jobs,
            gpu_ids=normalized_gpu_ids,
            cpus_per_worker=cpus_per_worker,
            worker_name_prefix="SubjectCalibrationWorker",
            worker_description="target-subject calibration",
        )
    subject_outputs.sort(key=lambda row: int(row["subject_number"]))

    metric_names = ("loss", *metrics, "joint_loss", *_DECODER_SCORE_NAMES)
    subject_summary_rows: list[dict] = []
    zero_all_subject_rows: list[dict] = []
    level_subject_rows = {
        str(shots): {
            "paired_zero": [],
            "calibrated": [],
            "delta": [],
        }
        for shots, _ in calibration_levels
    }
    total_calibration_folds = sum(folds for _, folds in calibration_levels)

    results = {
        "cv_strategy": (
            "loaded_pretrained_model_with_subject_calibration"
            if pretrained_model is not None
            else "subject_independent_pretraining_with_subject_calibration"
        ),
        "protocol_name": (
            "target-specific loaded LOSO model with multi-level calibration"
            if pretrained_model is not None
            else "strict LOSO with multi-level calibration continuation"
        ),
        "n_subjects": total_subjects,
        "n_source_model_fits": (
            0 if pretrained_model is not None else total_subjects
        ),
        "source_model_origin": (
            "loaded_pretrained_model"
            if pretrained_model is not None
            else "fit_in_cross_validation"
        ),
        "n_calibration_fits": total_subjects * int(total_calibration_folds),
        "calibration_plan": [
            {"shots": int(shots), "folds": int(folds)}
            for shots, folds in calibration_levels
        ],
        "calibration_selection_shots": int(calibration_selection_shots),
        "calibration_seed": calibration_seed,
        "stratify_calibration": bool(stratify_calibration),
        "allow_overlapping_calibration_folds": bool(
            allow_overlapping_calibration_folds
        ),
        "source_epochs": int(source_epochs),
        "source_batch_size": int(source_batch_size),
        "validation_subjects_per_fold": int(validation_subjects_per_fold),
        "validation_seed": validation_seed,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": float(early_stopping_min_delta),
        "early_stopping_monitor": str(early_stopping_monitor),
        "early_stopping_mode": str(early_stopping_mode),
        "restore_best_weights": bool(restore_best_weights),
        "calibration_epochs": int(calibration_epochs),
        "calibration_batch_size": int(calibration_batch_size),
        "calibration_learning_rate": float(calibration_learning_rate),
        "calibration_optimizer": str(calibration_optimizer),
        "calibration_weight_decay": float(calibration_weight_decay),
        "calibration_verbose": bool(calibration_verbose),
        "calibration_print_every_n_epochs": int(
            calibration_print_every_n_epochs
        ),
        "decision_threshold": decision_threshold,
        "prediction_diagnostics": {
            "enabled": bool(prediction_diagnostics),
            "official_decision_threshold": decision_threshold,
            "decision_thresholds": list(prediction_diagnostics_thresholds),
            "reported_metric": prediction_diagnostics_metric,
            "every_n_epochs": int(prediction_diagnostics_every_n_epochs),
            "max_samples": int(prediction_diagnostics_max_samples),
            "threshold_tolerance": float(
                prediction_diagnostics_threshold_tolerance
            ),
            "seed": prediction_diagnostics_seed,
        },
        "evaluation_level": evaluation_level,
        "metrics": list(metrics),
        "ece_bins": int(ece_bins),
        "fixed_config": fixed_config,
        "zero_shot_source_models": {
            "saved": source_model_output_dir is not None,
            "directory": source_model_output_dir,
            "format": "keras",
            "load_with_compile": False,
            "models": [
                subject_output["zero_shot_model"]
                for subject_output in subject_outputs
                if subject_output.get("zero_shot_model") is not None
            ],
        },
        "subject_results": subject_outputs,
        "subject_summary_rows": subject_summary_rows,
        "oracle_epoch_log": [],
    }
    if prediction_diagnostics:
        results["prediction_diagnostics_log"] = []
    if log_predictions:
        if feature_array.ndim == 3:
            results["window_prediction_log"] = []
        results["trial_prediction_log"] = []
    if log_variational_intervals:
        if feature_array.ndim == 3:
            results["window_variational_interval_log"] = []
        results["trial_variational_interval_log"] = []

    for subject_output in subject_outputs:
        results["oracle_epoch_log"].extend(
            subject_output.get("oracle_epoch_log", [])
        )
        if prediction_diagnostics:
            results["prediction_diagnostics_log"].extend(
                subject_output.get("prediction_diagnostics_log", [])
            )
        summary = subject_output["subject_summary"]
        zero_all = dict(summary["zero_shot_all_trials_scores"])
        zero_all_subject_rows.append(zero_all)

        flat_summary = {"target_subject": subject_output["target_subject"]}
        zero_shot_model = subject_output.get("zero_shot_model")
        if zero_shot_model is not None:
            flat_summary["zero_shot_model_path"] = zero_shot_model["path"]
            flat_summary["zero_shot_model_filename"] = zero_shot_model[
                "filename"
            ]
        flat_summary.update({f"zero_shot_all_{k}": v for k, v in zero_all.items()})
        for level in subject_output["calibration_levels"]:
            shots_key = str(level["calibration_shots"])
            level_summary = dict(level["summary"])
            paired_zero = dict(level_summary["paired_zero_shot_mean_scores"])
            calibrated = dict(level_summary["calibrated_mean_scores"])
            delta = dict(level_summary["delta_mean_scores"])
            level_subject_rows[shots_key]["paired_zero"].append(paired_zero)
            level_subject_rows[shots_key]["calibrated"].append(calibrated)
            level_subject_rows[shots_key]["delta"].append(delta)
            flat_summary.update(
                {f"{shots_key}_shot_paired_zero_{k}": v for k, v in paired_zero.items()}
            )
            flat_summary.update(
                {f"{shots_key}_shot_calibrated_{k}": v for k, v in calibrated.items()}
            )
            flat_summary.update(
                {f"{shots_key}_shot_delta_{k}": v for k, v in delta.items()}
            )
        subject_summary_rows.append(flat_summary)

        if log_predictions:
            evaluations = [subject_output["zero_shot_all_trials"]]
            for level in subject_output["calibration_levels"]:
                for fold_output in level["fold_outputs"]:
                    evaluations.extend(
                        [fold_output["zero_shot"], fold_output["calibrated"]]
                    )
            for evaluation in evaluations:
                if feature_array.ndim == 3:
                    results["window_prediction_log"].extend(
                        evaluation.get("window_prediction_log", [])
                    )
                results["trial_prediction_log"].extend(
                    evaluation.get("trial_prediction_log", [])
                )

        if log_variational_intervals:
            evaluations = [subject_output["zero_shot_all_trials"]]
            for level in subject_output["calibration_levels"]:
                for fold_output in level["fold_outputs"]:
                    evaluations.extend(
                        [fold_output["zero_shot"], fold_output["calibrated"]]
                    )
            for evaluation in evaluations:
                if feature_array.ndim == 3:
                    results["window_variational_interval_log"].extend(
                        evaluation.get("window_variational_interval_log", [])
                    )
                results["trial_variational_interval_log"].extend(
                    evaluation.get("trial_variational_interval_log", [])
                )

    zero_all_mean, zero_all_std = _mean_std_rows(
        zero_all_subject_rows, list(metric_names)
    )
    overall_calibration_levels: dict[str, dict] = {}
    for shots, folds in calibration_levels:
        shots_key = str(shots)
        rows = level_subject_rows[shots_key]
        paired_zero_mean, paired_zero_std = _mean_std_rows(
            rows["paired_zero"], list(metric_names)
        )
        calibrated_mean, calibrated_std = _mean_std_rows(
            rows["calibrated"], list(metric_names)
        )
        delta_mean, delta_std = _mean_std_rows(
            rows["delta"], list(metric_names)
        )
        overall_calibration_levels[shots_key] = {
            "calibration_shots": int(shots),
            "calibration_folds": int(folds),
            "paired_zero_shot_mean_scores": paired_zero_mean,
            "paired_zero_shot_std_scores": paired_zero_std,
            "calibrated_mean_scores": calibrated_mean,
            "calibrated_std_scores": calibrated_std,
            "delta_mean_scores": delta_mean,
            "delta_std_scores": delta_std,
            "delta_definition": "post_calibration_minus_paired_zero_shot",
        }
    selected_level = overall_calibration_levels[str(calibration_selection_shots)]
    results["overall"] = {
        "aggregation_unit": "subject",
        "n_subjects": total_subjects,
        "zero_shot_all_trials_mean_scores": zero_all_mean,
        "zero_shot_all_trials_std_scores": zero_all_std,
        "calibration_selection_shots": int(calibration_selection_shots),
        "calibration_levels": overall_calibration_levels,
        # Compatibility aliases point to the explicitly selected shot level.
        "paired_zero_shot_mean_scores": selected_level["paired_zero_shot_mean_scores"],
        "paired_zero_shot_std_scores": selected_level["paired_zero_shot_std_scores"],
        "calibrated_mean_scores": selected_level["calibrated_mean_scores"],
        "calibrated_std_scores": selected_level["calibrated_std_scores"],
        "delta_mean_scores": selected_level["delta_mean_scores"],
        "delta_std_scores": selected_level["delta_std_scores"],
        "delta_definition": "post_calibration_minus_paired_zero_shot",
    }

    final_metric_names = ("accuracy", "balanced_accuracy", "brier_score")

    def _format_final_means(mean_scores: dict) -> str:
        parts = []
        for metric_name in final_metric_names:
            value = mean_scores.get(metric_name)
            formatted_value = "N/A" if value is None else f"{float(value):.6f}"
            parts.append(f"{metric_name}={formatted_value}")
        return ", ".join(parts)

    print("\nFinal mean metrics by calibration level")
    print("=" * 80)
    print(f"0-shot: {_format_final_means(zero_all_mean)}")
    for shots, _ in calibration_levels:
        calibrated_means = overall_calibration_levels[str(shots)][
            "calibrated_mean_scores"
        ]
        print(f"{shots}-shot: {_format_final_means(calibrated_means)}")
    return results
