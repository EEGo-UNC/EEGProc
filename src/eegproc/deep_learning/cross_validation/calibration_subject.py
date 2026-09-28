"""The body of a single subject-calibration run, per held-out subject."""

from __future__ import annotations

import cloudpickle
import gc
import numpy as np
import os
import tensorflow as tf
import traceback

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotations only
    from typing import Callable
    from typing import Literal

from .arrays import _prepare_fit_inputs_with_subject_ids, _python_scalar
from .constants import _DECODER_SCORE_NAMES
from .workers import _configure_tensorflow_worker
from .splits import _class_weight_from_labels, _make_subject_calibration_splits
from .callbacks import EvaluationLevelValidationMetrics, _PeriodicCalibrationEpochLogger
from .reporting import _calibration_score_dict, _final_history_values, _mean_std_rows, _safe_model_filename_token
from .evaluation import _evaluate_classification_fold
from .search import _split_config
from ..training_outputs import (
    HeldOutUserOracleMetrics as ReportingHeldOutUserOracleMetrics,
    PredictionDiagnostics,
)

def _tag_subject_calibration_evaluation(
    evaluation: dict,
    *,
    target_subject,
    calibration_fold: int | None,
    stage: str,
    calibration_shots: int | None = None,
) -> dict:
    """Annotate evaluation rows so zero-shot/calibrated logs stay separable."""
    tagged = dict(evaluation)
    for key in ("fold_metrics", "window_fold_metrics", "trial_fold_metrics"):
        if key in tagged:
            tagged[key] = {
                **dict(tagged[key]),
                "target_subject": _python_scalar(target_subject),
                "calibration_fold": calibration_fold,
                "calibration_shots": calibration_shots,
                "stage": stage,
            }

    for key in (
        "user_metrics",
        "window_prediction_log",
        "trial_prediction_log",
        "window_variational_interval_log",
        "trial_variational_interval_log",
    ):
        rows = []
        for row in tagged.get(key, []):
            rows.append(
                {
                    **dict(row),
                    "target_subject": _python_scalar(target_subject),
                    "calibration_fold": calibration_fold,
                    "calibration_shots": calibration_shots,
                    "stage": stage,
                }
            )
        tagged[key] = rows
    return tagged


def _save_zero_shot_source_model(
    model: tf.keras.Model,
    *,
    output_dir: str | os.PathLike[str],
    subject_number: int,
    target_subject,
) -> dict:
    """Persist the untouched source-trained model for one LOSO target."""
    resolved_dir = os.path.abspath(os.fspath(output_dir))
    os.makedirs(resolved_dir, exist_ok=True)
    target_token = _safe_model_filename_token(target_subject)
    filename = (
        f"loso_fold_{int(subject_number):04d}_target_{target_token}_zero_shot.keras"
    )
    model_path = os.path.join(resolved_dir, filename)
    model.save(model_path, overwrite=True)
    return {
        "stage": "zero_shot_source_model",
        "format": "keras",
        "path": model_path,
        "filename": filename,
        "target_subject": _python_scalar(target_subject),
        "loso_fold": int(subject_number),
        "load_with_compile": False,
    }


def _run_subject_calibration_subject(
    subject_number: int,
    target_subject,
    total_subjects: int,
    *,
    model_builder_function: Callable[..., tf.keras.Model] | None,
    pretrained_model: tf.keras.Model | None,
    feature_array: np.ndarray,
    label_array: np.ndarray,
    subject_id_array: np.ndarray,
    trial_id_array: np.ndarray,
    fixed_config: dict,
    source_epochs: int,
    source_batch_size: int,
    validation_subjects_per_fold: int,
    validation_seed: int | None,
    early_stopping_patience: int | None,
    early_stopping_min_delta: float,
    early_stopping_monitor: str,
    early_stopping_mode: Literal["auto", "min", "max"],
    restore_best_weights: bool,
    calibration_epochs: int,
    calibration_batch_size: int,
    calibration_trials: int,
    calibration_folds: int,
    calibration_levels: tuple[tuple[int, int], ...],
    calibration_learning_rate: float,
    calibration_optimizer: str,
    calibration_weight_decay: float,
    calibration_seed: int | None,
    stratify_calibration: bool,
    allow_overlapping_calibration_folds: bool,
    evaluation_level: Literal["window", "trial"],
    metrics: tuple[str, ...],
    ece_bins: int,
    decision_threshold: float,
    prediction_diagnostics_thresholds: tuple[float, ...],
    prediction_diagnostics: bool,
    prediction_diagnostics_metric: str,
    prediction_diagnostics_every_n_epochs: int,
    prediction_diagnostics_max_samples: int,
    prediction_diagnostics_threshold_tolerance: float,
    prediction_diagnostics_seed: int | None,
    n_prediction_latent_samples: int,
    latent_sampling_seed: int | None,
    log_predictions: bool,
    log_variational_intervals: bool,
    n_uncertainty_samples: int,
    ci_level: float,
    source_use_class_weight: bool,
    calibration_use_class_weight: bool,
    source_fit_kwargs: dict,
    calibration_fit_kwargs: dict,
    calibration_verbose: bool,
    calibration_print_every_n_epochs: int,
    source_model_output_dir: str | os.PathLike[str] | None,
    verbose: int,
) -> dict:
    """Prepare one source model and run all target-subject calibration folds.

    ``pretrained_model`` is an already-loaded, target-specific LOSO checkpoint.
    When it is provided, source construction and source fitting are skipped.
    """
    target_mask = subject_id_array == target_subject
    target_indices = np.flatnonzero(target_mask)
    outer_source_indices = np.flatnonzero(~target_mask)
    if not len(target_indices) or not len(outer_source_indices):
        raise ValueError(
            f"Invalid source/target split for subject {target_subject!r}: "
            f"source={len(outer_source_indices)}, target={len(target_indices)}."
        )

    outer_source_subjects = np.sort(
        np.unique(subject_id_array[outer_source_indices])
    )
    if validation_subjects_per_fold < 0:
        raise ValueError("validation_subjects_per_fold must be >= 0.")
    if validation_subjects_per_fold >= len(outer_source_subjects):
        raise ValueError(
            "validation_subjects_per_fold must leave at least one non-target "
            f"subject for source fitting; got {validation_subjects_per_fold} "
            f"from {len(outer_source_subjects)} source subjects."
        )
    if validation_subjects_per_fold:
        base_seed = 0 if validation_seed is None else int(validation_seed)
        validation_rng = np.random.default_rng(
            np.random.SeedSequence([base_seed, int(subject_number)])
        )
        validation_subjects = np.sort(
            validation_rng.choice(
                outer_source_subjects,
                size=int(validation_subjects_per_fold),
                replace=False,
            )
        )
        validation_mask = np.isin(
            subject_id_array[outer_source_indices],
            validation_subjects,
        )
        source_validation_indices = outer_source_indices[validation_mask]
        source_indices = outer_source_indices[~validation_mask]
    else:
        validation_subjects = np.asarray([], dtype=outer_source_subjects.dtype)
        source_validation_indices = np.asarray([], dtype=np.int64)
        source_indices = outer_source_indices

    X_source = feature_array[source_indices]
    y_source = label_array[source_indices]
    source_subject_ids = subject_id_array[source_indices]
    source_trial_ids = trial_id_array[source_indices]
    X_source_validation = feature_array[source_validation_indices]
    y_source_validation = label_array[source_validation_indices]
    source_validation_subject_ids = subject_id_array[source_validation_indices]
    source_validation_trial_ids = trial_id_array[source_validation_indices]

    X_target = feature_array[target_indices]
    y_target = label_array[target_indices]
    target_subject_ids = subject_id_array[target_indices]
    target_trial_ids = trial_id_array[target_indices]

    fold_seed = (
        None
        if calibration_seed is None
        else int(calibration_seed) + int(subject_number) - 1
    )
    calibration_splits_by_level: list[tuple[int, int, list[dict]]] = []
    target_trial_count = int(len(np.unique(target_trial_ids)))
    for level_index, (shots, folds) in enumerate(calibration_levels):
        level_seed = (
            None
            if fold_seed is None
            else int(fold_seed) + level_index * 1_000_003
        )
        calibration_splits_by_level.append(
            (
                int(shots),
                int(folds),
                _make_subject_calibration_splits(
                    labels=y_target,
                    trial_ids=target_trial_ids,
                    calibration_trials=int(shots),
                    calibration_folds=int(folds),
                    seed=level_seed,
                    stratify=stratify_calibration,
                    allow_overlapping_folds=(
                        bool(allow_overlapping_calibration_folds)
                        or int(shots) * int(folds) != target_trial_count
                    ),
                ),
            )
        )

    model_hp, ignored_fit_hp = _split_config(fixed_config)
    if ignored_fit_hp:
        raise ValueError(
            "subject_calibration_cv uses explicit source/calibration epoch and "
            "batch-size arguments. Remove epochs/batch_size from fixed_config."
        )

    if pretrained_model is None:
        tf.keras.backend.clear_session()
    try:
        if pretrained_model is not None:
            model = pretrained_model
        else:
            try:
                model = model_builder_function(
                    training_features=X_source,
                    training_labels=y_source,
                    training_subject_ids=source_subject_ids,
                    training_trial_ids=source_trial_ids,
                    **model_hp,
                )
            except TypeError as exc:
                raise TypeError(
                    "For subject_calibration_cv, model_builder_function must accept "
                    "training_features, training_labels, training_subject_ids, and "
                    "training_trial_ids (directly or through **kwargs). v6 uses "
                    "training_features to construct a source-only adjacency "
                    "without target-subject leakage."
                ) from exc

        X_source_for_fit = _prepare_fit_inputs_with_subject_ids(
            model,
            X_source,
            source_subject_ids,
        )

        source_kwargs = dict(source_fit_kwargs)
        source_callbacks = list(source_kwargs.pop("callbacks", []))
        prediction_diagnostics_callback = None
        if prediction_diagnostics:

            prediction_diagnostics_callback = PredictionDiagnostics(
                X_train=X_source,
                y_train=y_source,
                X_val=(X_source_validation if len(source_validation_indices) else None),
                y_val=(y_source_validation if len(source_validation_indices) else None),
                fold_number=int(subject_number),
                batch_size=int(source_batch_size),
                every_n_epochs=int(prediction_diagnostics_every_n_epochs),
                max_samples=int(prediction_diagnostics_max_samples),
                threshold_tolerance=float(
                    prediction_diagnostics_threshold_tolerance
                ),
                reported_metric=str(prediction_diagnostics_metric),
                decision_threshold=float(decision_threshold),
                decision_thresholds=prediction_diagnostics_thresholds,
                ece_bins=int(ece_bins),
                seed=(
                    None
                    if prediction_diagnostics_seed is None
                    else int(prediction_diagnostics_seed) + int(subject_number)
                ),
            )
            source_callbacks.append(prediction_diagnostics_callback)

        oracle_metrics_callback = ReportingHeldOutUserOracleMetrics(
            X_target=X_target,
            y_target=y_target,
            subject_ids_target=target_subject_ids,
            trial_ids_target=target_trial_ids,
            target_subject=target_subject,
            evaluation_level=evaluation_level,
            batch_size=int(1),
            decision_threshold=float(decision_threshold),
            diagnostic_thresholds=prediction_diagnostics_thresholds,
            ece_bins=int(ece_bins),
        )
        source_callbacks.append(oracle_metrics_callback)

        source_class_weight = (
            _class_weight_from_labels(y_source) if source_use_class_weight else None
        )
        if source_class_weight is not None:
            source_kwargs["class_weight"] = source_class_weight

        source_validation_data = None
        if len(source_validation_indices):
            source_validation_data = (X_source_validation, y_source_validation)
            source_callbacks.append(
                EvaluationLevelValidationMetrics(
                    X_val=X_source_validation,
                    y_val=y_source_validation,
                    subject_ids_val=source_validation_subject_ids,
                    trial_ids_val=source_validation_trial_ids,
                    evaluation_level=evaluation_level,
                    batch_size=int(source_batch_size),
                    ece_bins=ece_bins,
                )
            )
            if early_stopping_patience is not None:
                source_callbacks.append(
                    tf.keras.callbacks.EarlyStopping(
                        monitor=early_stopping_monitor,
                        patience=int(early_stopping_patience),
                        min_delta=float(early_stopping_min_delta),
                        mode=early_stopping_mode,
                        restore_best_weights=bool(restore_best_weights),
                        verbose=1 if verbose else 0,
                    )
                )
        if source_callbacks:
            source_kwargs["callbacks"] = source_callbacks

        source_history = None
        if pretrained_model is None:
            source_history = model.fit(
                X_source_for_fit,
                y_source,
                validation_data=source_validation_data,
                epochs=int(source_epochs),
                batch_size=int(source_batch_size),
                verbose=int(verbose),
                **source_kwargs,
            )

        source_history_values = (
            {} if source_history is None else source_history.history
        )
        source_epochs_ran = int(len(source_history_values.get("loss", [])))
        source_best_epoch: int | None = None
        source_best_monitored_value: float | None = None
        monitored_history = source_history_values.get(early_stopping_monitor)
        if monitored_history:
            monitored_values = np.asarray(monitored_history, dtype=np.float64)
            finite_indices = np.flatnonzero(np.isfinite(monitored_values))
            if len(finite_indices):
                finite_values = monitored_values[finite_indices]
                if early_stopping_mode == "max":
                    best_local_index = int(np.argmax(finite_values))
                elif early_stopping_mode == "min":
                    best_local_index = int(np.argmin(finite_values))
                else:
                    maximize_tokens = (
                        "acc",
                        "auc",
                        "f1",
                        "precision",
                        "recall",
                    )
                    maximize = any(
                        token in early_stopping_monitor.lower()
                        for token in maximize_tokens
                    )
                    best_local_index = int(
                        np.argmax(finite_values)
                        if maximize
                        else np.argmin(finite_values)
                    )
                best_history_index = int(finite_indices[best_local_index])
                source_best_epoch = best_history_index + 1
                source_best_monitored_value = float(
                    monitored_values[best_history_index]
                )
        if source_best_epoch is None and source_epochs_ran:
            source_best_epoch = source_epochs_ran

        # Persist the exact population/source model used for zero-shot LOSO
        # evaluation. This happens before calibration changes trainability or
        # compiles a subject-specific optimizer.
        prepare_zero_shot = getattr(
            model, "prepare_for_zero_shot_evaluation", None
        )
        if prepare_zero_shot is not None:
            prepare_zero_shot()
        zero_shot_model_artifact = (
            None
            if source_model_output_dir is None
            else _save_zero_shot_source_model(
                model,
                output_dir=source_model_output_dir,
                subject_number=subject_number,
                target_subject=target_subject,
            )
        )

        # get_weights() captures only model state, not optimizer state. That is
        # intentional: every calibration fold gets a newly compiled calibration
        # optimizer from prepare_for_subject_calibration().
        source_weights = [np.array(value, copy=True) for value in model.get_weights()]

        all_target_evaluation = _evaluate_classification_fold(
            model=model,
            X_test=X_target,
            y_test=y_target,
            subject_ids_test=target_subject_ids,
            trial_ids_test=target_trial_ids,
            fold_index=subject_number,
            metrics=metrics,
            evaluation_level=evaluation_level,
            batch_size=source_batch_size,
            n_prediction_latent_samples=n_prediction_latent_samples,
            latent_sampling_seed=latent_sampling_seed,
            log_predictions=log_predictions,
            log_variational_intervals=log_variational_intervals,
            n_uncertainty_samples=n_uncertainty_samples,
            ci_level=ci_level,
            decision_threshold=decision_threshold,
            ece_bins=ece_bins,
            print_results=False,
        )
        all_target_evaluation = _tag_subject_calibration_evaluation(
            all_target_evaluation,
            target_subject=target_subject,
            calibration_fold=None,
            stage="zero_shot_all_trials",
        )

        prepare_calibration = getattr(
            model,
            "prepare_for_subject_calibration",
            None,
        )
        if prepare_calibration is None:
            raise AttributeError(
                "The model must implement prepare_for_subject_calibration(" 
                "learning_rate=..., optimizer_name=..., weight_decay=...). "
                "That method should freeze the subject-independent/generative "
                "representation, leave only the configured calibration "
                "submodules trainable, and compile a fresh calibration optimizer."
            )

        metric_names = ("loss", *metrics, "joint_loss", *_DECODER_SCORE_NAMES)
        calibration_level_outputs: list[dict] = []
        calibration_oracle_epoch_log: list[dict] = []
        evaluation_offset = (int(subject_number) - 1) * sum(
            level_folds for _, level_folds in calibration_levels
        )

        for calibration_shots, level_folds, calibration_splits in calibration_splits_by_level:
            calibration_rows: list[dict] = []
            fold_outputs: list[dict] = []

            for split in calibration_splits:
                calibration_fold = int(split["calibration_fold"])
                calibration_mask = np.isin(
                    target_trial_ids,
                    np.asarray(split["calibration_trial_ids"]),
                )
                evaluation_mask = np.isin(
                    target_trial_ids,
                    np.asarray(split["evaluation_trial_ids"]),
                )
                calibration_local_indices = np.flatnonzero(calibration_mask)
                evaluation_local_indices = np.flatnonzero(evaluation_mask)
                if not len(calibration_local_indices) or not len(evaluation_local_indices):
                    raise RuntimeError(
                        f"{calibration_shots}-shot calibration fold "
                        f"{calibration_fold} produced an empty calibration or "
                        "evaluation partition."
                    )

                X_calibration = X_target[calibration_local_indices]
                y_calibration = y_target[calibration_local_indices]
                X_evaluation = X_target[evaluation_local_indices]
                y_evaluation = y_target[evaluation_local_indices]
                evaluation_subject_ids = target_subject_ids[evaluation_local_indices]
                evaluation_trial_ids = target_trial_ids[evaluation_local_indices]

                # Every shot/fold continuation starts from the identical strict
                # LOSO source checkpoint and receives a fresh optimizer.
                model.set_weights(source_weights)
                prepare_zero_shot = getattr(
                    model, "prepare_for_zero_shot_evaluation", None
                )
                if prepare_zero_shot is not None:
                    prepare_zero_shot()
                evaluation_index = evaluation_offset + calibration_fold
                zero_shot = _evaluate_classification_fold(
                    model=model,
                    X_test=X_evaluation,
                    y_test=y_evaluation,
                    subject_ids_test=evaluation_subject_ids,
                    trial_ids_test=evaluation_trial_ids,
                    fold_index=evaluation_index,
                    metrics=metrics,
                    evaluation_level=evaluation_level,
                    batch_size=source_batch_size,
                    n_prediction_latent_samples=n_prediction_latent_samples,
                    latent_sampling_seed=latent_sampling_seed,
                    log_predictions=log_predictions,
                    log_variational_intervals=log_variational_intervals,
                    n_uncertainty_samples=n_uncertainty_samples,
                    ci_level=ci_level,
                    decision_threshold=decision_threshold,
                    ece_bins=ece_bins,
                    print_results=False,
                )
                zero_shot = _tag_subject_calibration_evaluation(
                    zero_shot,
                    target_subject=target_subject,
                    calibration_fold=calibration_fold,
                    calibration_shots=calibration_shots,
                    stage="zero_shot_paired",
                )

                model.set_weights(source_weights)
                prepare_calibration(
                    learning_rate=float(calibration_learning_rate),
                    optimizer_name=str(calibration_optimizer),
                    weight_decay=float(calibration_weight_decay),
                )
                trainable_names = [
                    variable.name for variable in model.trainable_variables
                ]
                if not trainable_names:
                    raise RuntimeError(
                        "prepare_for_subject_calibration() left no trainable variables."
                    )

                prepare_inputs = getattr(model, "prepare_calibration_inputs", None)
                X_calibration_for_fit = (
                    prepare_inputs(X_calibration)
                    if prepare_inputs is not None
                    else X_calibration
                )
                calibration_kwargs = dict(calibration_fit_kwargs)
                calibration_class_weight = (
                    _class_weight_from_labels(y_calibration)
                    if calibration_use_class_weight
                    else None
                )
                if calibration_class_weight is not None:
                    calibration_kwargs["class_weight"] = calibration_class_weight

                callbacks = list(calibration_kwargs.get("callbacks") or [])
                if calibration_verbose:
                    callbacks.append(
                        _PeriodicCalibrationEpochLogger(
                            target_subject=target_subject,
                            calibration_shots=calibration_shots,
                            calibration_fold=calibration_fold,
                            total_epochs=int(calibration_epochs),
                            every_n_epochs=int(
                                calibration_print_every_n_epochs
                            ),
                        )
                    )

                # Reporting-only oracle inference runs after every calibration
                # epoch on the complete evaluation partition for this fold.
                # These held-out trials never enter model.fit or its logs, so
                # they cannot affect gradients or model selection.
                calibration_oracle_callback = ReportingHeldOutUserOracleMetrics(
                    X_target=X_evaluation,
                    y_target=y_evaluation,
                    subject_ids_target=evaluation_subject_ids,
                    trial_ids_target=evaluation_trial_ids,
                    target_subject=target_subject,
                    evaluation_level=evaluation_level,
                    batch_size=1,
                    decision_threshold=float(decision_threshold),
                    diagnostic_thresholds=prediction_diagnostics_thresholds,
                    ece_bins=int(ece_bins),
                    calibration_shots=int(calibration_shots),
                    calibration_fold=calibration_fold,
                )
                callbacks.append(calibration_oracle_callback)
                calibration_kwargs["callbacks"] = callbacks

                calibration_history = model.fit(
                    X_calibration_for_fit,
                    y_calibration,
                    epochs=int(calibration_epochs),
                    batch_size=int(calibration_batch_size),
                    # Calibration continuations are intentionally quiet; the
                    # optional periodic callback controls concise epoch output.
                    verbose=0,
                    **calibration_kwargs,
                )
                fold_oracle_epoch_log = [
                    {
                        **dict(row),
                        "stage": "calibration_epoch_oracle",
                        "evaluation_trial_ids": list(
                            split["evaluation_trial_ids"]
                        ),
                    }
                    for row in calibration_oracle_callback.history
                ]
                calibration_oracle_epoch_log.extend(fold_oracle_epoch_log)

                calibrated = _evaluate_classification_fold(
                    model=model,
                    X_test=X_evaluation,
                    y_test=y_evaluation,
                    subject_ids_test=evaluation_subject_ids,
                    trial_ids_test=evaluation_trial_ids,
                    fold_index=evaluation_index,
                    metrics=metrics,
                    evaluation_level=evaluation_level,
                    batch_size=calibration_batch_size,
                    n_prediction_latent_samples=n_prediction_latent_samples,
                    latent_sampling_seed=latent_sampling_seed,
                    log_predictions=log_predictions,
                    log_variational_intervals=log_variational_intervals,
                    n_uncertainty_samples=n_uncertainty_samples,
                    ci_level=ci_level,
                    decision_threshold=decision_threshold,
                    ece_bins=ece_bins,
                    print_results=False,
                )
                calibrated = _tag_subject_calibration_evaluation(
                    calibrated,
                    target_subject=target_subject,
                    calibration_fold=calibration_fold,
                    calibration_shots=calibration_shots,
                    stage="post_calibration",
                )

                zero_scores = _calibration_score_dict(zero_shot, metric_names)
                calibrated_scores = _calibration_score_dict(
                    calibrated, metric_names
                )
                delta_scores = {
                    metric_name: float(
                        calibrated_scores[metric_name] - zero_scores[metric_name]
                    )
                    for metric_name in zero_scores.keys() & calibrated_scores.keys()
                }
                calibration_row = {
                    "target_subject": _python_scalar(target_subject),
                    "calibration_shots": int(calibration_shots),
                    "calibration_fold": calibration_fold,
                    "partition_mode": split["partition_mode"],
                    "calibration_trial_ids": list(split["calibration_trial_ids"]),
                    "evaluation_trial_ids": list(split["evaluation_trial_ids"]),
                    "calibration_class_counts": dict(split["calibration_class_counts"]),
                    "evaluation_class_counts": dict(split["evaluation_class_counts"]),
                    "n_calibration_samples": int(len(calibration_local_indices)),
                    "n_evaluation_samples": int(len(evaluation_local_indices)),
                    "calibration_epochs_ran": int(
                        len(calibration_history.history.get("loss", []))
                    ),
                    "calibration_final_history": _final_history_values(
                        calibration_history
                    ),
                    "calibration_trainable_variables": trainable_names,
                    "oracle_epoch_log": fold_oracle_epoch_log,
                    "zero_shot_scores": zero_scores,
                    "calibrated_scores": calibrated_scores,
                    "delta_scores": delta_scores,
                }
                calibration_rows.append(calibration_row)
                fold_outputs.append(
                    {
                        "calibration_shots": int(calibration_shots),
                        "split": dict(split),
                        "zero_shot": zero_shot,
                        "calibrated": calibrated,
                    }
                )

            zero_rows = [row["zero_shot_scores"] for row in calibration_rows]
            calibrated_rows = [
                row["calibrated_scores"] for row in calibration_rows
            ]
            delta_rows = [row["delta_scores"] for row in calibration_rows]
            paired_zero_mean, paired_zero_std = _mean_std_rows(
                zero_rows, list(metric_names)
            )
            calibrated_mean, calibrated_std = _mean_std_rows(
                calibrated_rows, list(metric_names)
            )
            delta_mean, delta_std = _mean_std_rows(
                delta_rows, list(metric_names)
            )
            calibration_level_outputs.append(
                {
                    "calibration_shots": int(calibration_shots),
                    "calibration_folds": int(level_folds),
                    "calibration_runs": calibration_rows,
                    "fold_outputs": fold_outputs,
                    "summary": {
                        "paired_zero_shot_mean_scores": paired_zero_mean,
                        "paired_zero_shot_std_scores": paired_zero_std,
                        "calibrated_mean_scores": calibrated_mean,
                        "calibrated_std_scores": calibrated_std,
                        "delta_mean_scores": delta_mean,
                        "delta_std_scores": delta_std,
                    },
                }
            )
            evaluation_offset += int(level_folds)

        zero_all_scores = _calibration_score_dict(
            all_target_evaluation,
            metric_names,
        )
        calibration_summaries = {
            str(level["calibration_shots"]): dict(level["summary"])
            for level in calibration_level_outputs
        }
        subject_summary = {
            "target_subject": _python_scalar(target_subject),
            "zero_shot_all_trials_scores": zero_all_scores,
            "calibration_levels": calibration_summaries,
        }
        return {
            "subject_number": int(subject_number),
            "target_subject": _python_scalar(target_subject),
            "source_subjects": [
                _python_scalar(value)
                for value in np.sort(np.unique(source_subject_ids)).tolist()
            ],
            "source_validation_subjects": [
                _python_scalar(value) for value in validation_subjects.tolist()
            ],
            "n_source_samples": int(len(source_indices)),
            "n_source_validation_samples": int(len(source_validation_indices)),
            "n_outer_source_samples": int(len(outer_source_indices)),
            "n_target_samples": int(len(target_indices)),
            "n_target_trials": int(len(np.unique(target_trial_ids))),
            "source_training": {
                "mode": (
                    "loaded_pretrained_model"
                    if pretrained_model is not None
                    else "fit_source_model"
                ),
                "skipped": bool(pretrained_model is not None),
                "epochs_ran": source_epochs_ran,
                "best_epoch": source_best_epoch,
                "best_monitored_value": source_best_monitored_value,
                "early_stopping_monitor": early_stopping_monitor,
                "stopped_early": bool(
                    pretrained_model is None
                    and source_epochs_ran < int(source_epochs)
                ),
                "restore_best_weights": bool(restore_best_weights),
                "final_history": (
                    {}
                    if source_history is None
                    else _final_history_values(source_history)
                ),
                "class_weight": source_class_weight,
                "prediction_diagnostics_metric": (
                    str(prediction_diagnostics_metric)
                    if prediction_diagnostics
                    else None
                ),
            },
            "prediction_diagnostics_log": (
                []
                if prediction_diagnostics_callback is None
                else list(prediction_diagnostics_callback.history)
            ),
            "oracle_epoch_log": [
                *list(oracle_metrics_callback.history),
                *calibration_oracle_epoch_log,
            ],
            "zero_shot_model": zero_shot_model_artifact,
            "zero_shot_all_trials": all_target_evaluation,
            "calibration_levels": calibration_level_outputs,
            "subject_summary": subject_summary,
        }
    finally:
        if "model" in locals():
            del model
        gc.collect()
        tf.keras.backend.clear_session()


def _subject_calibration_process_main(
    worker_state_payload: bytes,
    task_queue,
    result_queue,
    gpu_id: int | None,
    cpus_per_worker: int | None,
    assigned_device_label: str | None,
) -> None:
    """Run target-subject calibration evaluations in a persistent worker."""
    try:
        _configure_tensorflow_worker(
            gpu_id=gpu_id,
            cpus_per_worker=cpus_per_worker,
            assigned_device_label=assigned_device_label,
        )
        worker_state = cloudpickle.loads(worker_state_payload)
        while True:
            task = task_queue.get()
            if task is None:
                return
            subject_number, target_subject = task
            try:
                output = _run_subject_calibration_subject(
                    subject_number=subject_number,
                    target_subject=target_subject,
                    **worker_state,
                )
                result_queue.put(("ok", int(subject_number), output))
            except BaseException:
                result_queue.put(
                    ("error", int(subject_number), traceback.format_exc())
                )
                return
    except BaseException:
        result_queue.put(("error", -1, traceback.format_exc()))
    finally:
        tf.keras.backend.clear_session()
        gc.collect()
