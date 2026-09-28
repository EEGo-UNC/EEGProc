"""The body of a single LOSO fold, run in the main process or a spawned worker."""

from __future__ import annotations

import cloudpickle
import gc
import numpy as np
import tensorflow as tf
import traceback

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotations only
    from typing import Callable
    from typing import Literal

from .arrays import _as_numpy_1d, _count_windows_for_indices, _is_trial_tensor, _prepare_fit_inputs_with_subject_ids, _python_scalar
from .workers import _configure_tensorflow_worker
from .probabilities import _predict_probabilities, _select_binary_decision_threshold
from .splits import _balanced_two_subject_sets
from .aggregation import _aggregate_window_probabilities_by_trial, _direct_trial_aggregation
from ..training_outputs import HeldOutUserOracleMetrics
from .callbacks import TrialValidationMetrics
from .reporting import _print_fold_header
from .evaluation import _apply_preprocessing_strategy, _evaluate_classification_fold, _validate_processed_alignment
from .search import _build_model_with_fold_training_context, _split_config
from ..training_outputs import CompactEpochLogger, PredictionDiagnostics
from ..domain_generalization.alternating_group_learning import (
    AlternatingSubjectSetSequence,
)
from ..domain_generalization.meta_learning import MetaLearningSubjectSequence

def _run_loso_fold(
    fold_number: int,
    test_subject,
    total_folds: int,
    model_builder_function: Callable[..., tf.keras.Model],
    feature_array: np.ndarray,
    label_array: np.ndarray,
    subject_id_array: np.ndarray,
    trial_id_array: np.ndarray,
    fixed_config: dict,
    batch_size: int,
    preprocessing_strategy: Callable | None,
    evaluation_level: Literal["window", "trial"],
    metrics: tuple[str, ...],
    log_predictions: bool,
    log_variational_intervals: bool,
    n_prediction_latent_samples: int,
    latent_sampling_seed: int | None,
    n_uncertainty_samples: int,
    ci_level: float,
    validation_subjects_per_fold: int,
    validation_seed: int | None,
    early_stopping_patience: int | None,
    early_stopping_min_delta: float,
    early_stopping_monitor: str,
    early_stopping_mode: Literal["auto", "min", "max"],
    restore_best_weights: bool,
    prediction_diagnostics: bool,
    prediction_diagnostics_every_n_epochs: int,
    prediction_diagnostics_max_samples: int,
    prediction_diagnostics_threshold_tolerance: float,
    prediction_diagnostics_seed: int | None,
    decision_thresholds: tuple[float, ...],
    threshold_selection_metric: str,
    threshold_selection_level: Literal["window", "trial"],
    verbose: int,
    extra_fit_kwargs: dict,
    fold_description: str | None = None,
    alternate_subject_sets: bool = False,
    alternating_subject_seed: int | None = 42,
    use_mldg: bool = False,
    mldg_meta_train_subjects: int = 6,
    mldg_meta_test_subjects: int = 2,
    mldg_samples_per_subject: int = 4,
    mldg_seed: int | None = 42,
) -> dict:
    """Train and evaluate one LOSO fold with optional seeded validation.

    The LOSO test subject is never used by ``model.fit``. When
    ``validation_subjects_per_fold`` is positive, that many subjects are drawn
    deterministically from the outer-training pool and excluded from gradient
    updates. They provide ``validation_data`` for early stopping without adding
    another model fit.
    """
    test_mask = subject_id_array == test_subject
    test_indices = np.where(test_mask)[0]
    outer_train_indices = np.where(~test_mask)[0]

    if len(outer_train_indices) == 0 or len(test_indices) == 0:
        raise ValueError(
            f"Invalid LOSO split: train={len(outer_train_indices)}, "
            f"test={len(test_indices)} samples."
        )

    outer_train_subjects = np.sort(np.unique(subject_id_array[outer_train_indices]))
    validation_candidate_subjects = outer_train_subjects
    if validation_subjects_per_fold < 0:
        raise ValueError("validation_subjects_per_fold must be >= 0.")
    if alternate_subject_sets and use_mldg:
        raise ValueError("alternate_subject_sets and use_mldg are mutually exclusive.")
    if mldg_meta_train_subjects < 1 or mldg_meta_test_subjects < 1:
        raise ValueError("MLDG A/B subject counts must both be at least 1.")
    if mldg_samples_per_subject < 1:
        raise ValueError("mldg_samples_per_subject must be at least 1.")
    if mldg_seed is not None and mldg_seed < 0:
        raise ValueError("mldg_seed must be >= 0 or None.")
    if alternate_subject_sets and validation_subjects_per_fold != 0:
        raise ValueError(
            "alternate_subject_sets uses all non-test subjects and therefore "
            "requires validation_subjects_per_fold=0."
        )
    if validation_subjects_per_fold > 0 and validation_subjects_per_fold >= len(
        validation_candidate_subjects
    ):
        raise ValueError(
            "validation_subjects_per_fold must leave at least one eligible "
            "subject outside validation. Got "
            f"{validation_subjects_per_fold} validation subjects from "
            f"{len(validation_candidate_subjects)} eligible subjects."
        )

    if validation_subjects_per_fold > 0:
        base_seed = 0 if validation_seed is None else int(validation_seed)
        fold_seed = np.random.SeedSequence([base_seed, int(fold_number)])
        rng = np.random.default_rng(fold_seed)
        validation_subjects = np.sort(
            rng.choice(
                validation_candidate_subjects,
                size=validation_subjects_per_fold,
                replace=False,
            )
        )
        validation_mask_relative = np.isin(
            subject_id_array[outer_train_indices],
            validation_subjects,
        )
        validation_indices = outer_train_indices[validation_mask_relative]
        fit_train_indices = outer_train_indices[~validation_mask_relative]
    else:
        validation_subjects = np.asarray([], dtype=outer_train_subjects.dtype)
        validation_indices = np.asarray([], dtype=np.int64)
        fit_train_indices = outer_train_indices

    sample_level = "trials" if feature_array.ndim == 4 else "windows"
    if fold_description is None:
        fold_description = (
            f"LOSO test subject={_python_scalar(test_subject)!r} "
            f"(fit_train={len(fit_train_indices)}, "
            f"validation={len(validation_indices)}, "
            f"test={len(test_indices)} {sample_level})"
        )
    else:
        fold_description = (
            f"{fold_description} (fit_train={len(fit_train_indices)}, "
            f"validation={len(validation_indices)}, "
            f"test={len(test_indices)} {sample_level})"
        )
    _print_fold_header(
        fold_number,
        total_folds,
        fold_description,
    )
    if len(validation_subjects):
        print(
            "Seeded validation subjects: "
            f"{[_python_scalar(value) for value in validation_subjects]}",
            flush=True,
        )

    # The current preprocessing callback API supports only one train/eval pair.
    # Refuse an ambiguous three-way fit rather than leaking validation subjects
    # into a fitted transform or fitting inconsistent transforms for val/test.
    if validation_subjects_per_fold > 0 and preprocessing_strategy is not None:
        raise ValueError(
            "Seeded subject-level validation currently requires "
            "preprocessing_strategy=None. Preprocess before loso_cv or extend "
            "the strategy API to transform train/validation/test from one "
            "fold-local fitted state."
        )

    X_fit_train = feature_array[fit_train_indices]
    y_fit_train = label_array[fit_train_indices]
    X_validation = feature_array[validation_indices]
    y_validation = label_array[validation_indices]
    X_test = feature_array[test_indices]
    y_test = label_array[test_indices]

    subject_ids_fit_train = subject_id_array[fit_train_indices]
    subject_ids_validation = subject_id_array[validation_indices]
    subject_ids_test = subject_id_array[test_indices]
    trial_ids_fit_train = trial_id_array[fit_train_indices]
    trial_ids_validation = trial_id_array[validation_indices]
    trial_ids_test = trial_id_array[test_indices]

    if validation_subjects_per_fold == 0:
        X_fit_train, y_fit_train, X_test, y_test = _apply_preprocessing_strategy(
            preprocessing_strategy=preprocessing_strategy,
            X_train=X_fit_train,
            y_train=y_fit_train,
            X_eval=X_test,
            y_eval=y_test,
            train_indices=fit_train_indices,
            eval_indices=test_indices,
        )

    _validate_processed_alignment(
        X_fit_train,
        y_fit_train,
        subject_ids_fit_train,
        trial_ids_fit_train,
        "LOSO-fit-training",
    )
    if validation_subjects_per_fold > 0:
        _validate_processed_alignment(
            X_validation,
            y_validation,
            subject_ids_validation,
            trial_ids_validation,
            "LOSO-validation",
        )
    _validate_processed_alignment(
        X_test,
        y_test,
        subject_ids_test,
        trial_ids_test,
        "LOSO-test",
    )

    model_hp, fit_hp = _split_config(fixed_config)
    current_batch_size = int(fit_hp.get("batch_size", batch_size))

    duplicate_fit_keys = set(fit_hp).intersection(extra_fit_kwargs)
    if duplicate_fit_keys:
        raise ValueError(
            "The following model.fit arguments were supplied in both the fixed "
            f"configuration and extra_fit_kwargs: {sorted(duplicate_fit_keys)}"
        )


    fit_call_kwargs = dict(extra_fit_kwargs)
    callbacks = list(fit_call_kwargs.pop("callbacks", []))
    prediction_diagnostics_callback: PredictionDiagnostics | None = None

    if prediction_diagnostics:
        prediction_diagnostics_callback = PredictionDiagnostics(
            X_train=X_fit_train,
            y_train=y_fit_train,
            X_val=(X_validation if validation_subjects_per_fold > 0 else None),
            y_val=(y_validation if validation_subjects_per_fold > 0 else None),
            fold_number=fold_number,
            batch_size=current_batch_size,
            every_n_epochs=prediction_diagnostics_every_n_epochs,
            max_samples=prediction_diagnostics_max_samples,
            threshold_tolerance=prediction_diagnostics_threshold_tolerance,
            seed=(
                None
                if prediction_diagnostics_seed is None
                else int(prediction_diagnostics_seed) + int(fold_number)
            ),
        )
        callbacks.append(prediction_diagnostics_callback)

    oracle_reporting_threshold = (
        float(decision_thresholds[0])
        if len(decision_thresholds) == 1
        else 0.5
    )
    oracle_metrics_callback = HeldOutUserOracleMetrics(
        X_target=X_test,
        y_target=y_test,
        subject_ids_target=subject_ids_test,
        trial_ids_target=trial_ids_test,
        target_subject=test_subject,
        evaluation_level=evaluation_level,
        batch_size=current_batch_size,
        decision_threshold=oracle_reporting_threshold,
    )
    callbacks.append(oracle_metrics_callback)

    if validation_subjects_per_fold > 0:
        if early_stopping_monitor in {
            "val_trial_f1",
            "val_trial_balanced_accuracy",
            "val_trial_loss",
        }:
            # This callback must run before CompactEpochLogger and EarlyStopping
            # so the custom metric is available to both callbacks.
            callbacks.append(
                TrialValidationMetrics(
                    X_val=X_validation,
                    y_val=y_validation,
                    subject_ids_val=subject_ids_validation,
                    trial_ids_val=trial_ids_validation,
                    batch_size=current_batch_size,
                )
            )

    if verbose:
        callbacks.append(CompactEpochLogger(fold_number=fold_number))

    if validation_subjects_per_fold > 0 and early_stopping_patience is not None:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=early_stopping_monitor,
                patience=int(early_stopping_patience),
                min_delta=float(early_stopping_min_delta),
                mode=early_stopping_mode,
                restore_best_weights=bool(restore_best_weights),
                verbose=1 if verbose else 0,
            )
        )

    if callbacks:
        fit_call_kwargs["callbacks"] = callbacks

    tf.keras.backend.clear_session()
    model = _build_model_with_fold_training_context(
        model_builder_function,
        model_hp,
        training_features=X_fit_train,
        training_labels=y_fit_train,
        training_subject_ids=subject_ids_fit_train,
        training_trial_ids=trial_ids_fit_train,
    )
    X_fit_train_for_fit = _prepare_fit_inputs_with_subject_ids(
        model,
        X_fit_train,
        subject_ids_fit_train,
    )

    epochs_ran = 0
    best_epoch: int | None = None
    best_monitored_value: float | None = None
    stopped_early = False

    try:
        y_fit_train_ids = _as_numpy_1d(y_fit_train)
        classes, counts = np.unique(y_fit_train_ids, return_counts=True)

        class_weight = {
            int(class_id): len(y_fit_train_ids) / (len(classes) * count)
            for class_id, count in zip(classes, counts)
        }

        validation_data = (
            (X_validation, y_validation) if validation_subjects_per_fold > 0 else None
        )
        if use_mldg:

            fold_mldg_seed = (
                None if mldg_seed is None else int(mldg_seed) + int(fold_number)
            )
            effective_class_weight = (
                class_weight if bool(getattr(model, "use_class_weight", True)) else None
            )
            mldg_sequence = MetaLearningSubjectSequence(
                X=X_fit_train,
                y=y_fit_train,
                subject_ids=subject_ids_fit_train,
                model=model,
                meta_train_subjects=mldg_meta_train_subjects,
                meta_test_subjects=mldg_meta_test_subjects,
                samples_per_subject=mldg_samples_per_subject,
                class_weight=effective_class_weight,
                seed=fold_mldg_seed,
            )
            print(
                "First-order MLDG episodes (natural within-subject labels): "
                f"A_subjects={mldg_meta_train_subjects}, "
                f"B_subjects={mldg_meta_test_subjects}, "
                f"samples_per_subject={mldg_samples_per_subject}, "
                f"steps_per_epoch={len(mldg_sequence)}",
                flush=True,
            )
            subject_set_a = np.asarray([], dtype=subject_ids_fit_train.dtype)
            subject_set_b = np.asarray([], dtype=subject_ids_fit_train.dtype)
            history = model.fit(
                mldg_sequence,
                validation_data=validation_data,
                verbose=0,
                **fit_hp,
                **fit_call_kwargs,
            )
        elif alternate_subject_sets:

            if validation_subjects_per_fold > 0:
                raise ValueError(
                    "alternate_subject_sets requires validation_subjects_per_fold=0."
                )
            fold_alt_seed = (
                None
                if alternating_subject_seed is None
                else int(alternating_subject_seed) + int(fold_number)
            )
            subject_set_a, subject_set_b = _balanced_two_subject_sets(
                subject_ids_fit_train,
                y_fit_train,
                seed=fold_alt_seed,
            )
            print(
                "Alternating subject sets: "
                f"A={[_python_scalar(v) for v in subject_set_a]} | "
                f"B={[_python_scalar(v) for v in subject_set_b]}",
                flush=True,
            )
            alternating_sequence = AlternatingSubjectSetSequence(
                X=X_fit_train,
                y=y_fit_train,
                subject_ids=subject_ids_fit_train,
                subject_set_a=subject_set_a,
                subject_set_b=subject_set_b,
                batch_size=current_batch_size,
                model=model,
                class_weight=class_weight,
                seed=fold_alt_seed,
            )
            history = model.fit(
                alternating_sequence,
                validation_data=None,
                verbose=0,
                **fit_hp,
                **fit_call_kwargs,
            )
        else:
            subject_set_a = np.asarray([], dtype=subject_ids_fit_train.dtype)
            subject_set_b = np.asarray([], dtype=subject_ids_fit_train.dtype)
            history = model.fit(
                X_fit_train_for_fit,
                y_fit_train,
                validation_data=validation_data,
                class_weight=class_weight,
                verbose=0,
                **fit_hp,
                **fit_call_kwargs,
            )

        epochs_ran = int(len(history.history.get("loss", [])))
        requested_epochs = int(fit_hp.get("epochs", epochs_ran))
        stopped_early = bool(epochs_ran < requested_epochs)

        monitored_history = history.history.get(early_stopping_monitor)
        if monitored_history:
            monitored_values = np.asarray(monitored_history, dtype=np.float64)
            finite_mask = np.isfinite(monitored_values)
            if np.any(finite_mask):
                candidate_indices = np.where(finite_mask)[0]
                candidate_values = monitored_values[finite_mask]
                if early_stopping_mode == "max":
                    local_best = int(np.argmax(candidate_values))
                elif early_stopping_mode == "min":
                    local_best = int(np.argmin(candidate_values))
                else:
                    maximize_tokens = ("acc", "auc", "f1", "precision", "recall")
                    maximize = any(
                        token in early_stopping_monitor.lower()
                        for token in maximize_tokens
                    )
                    local_best = int(
                        np.argmax(candidate_values)
                        if maximize
                        else np.argmin(candidate_values)
                    )
                best_index = int(candidate_indices[local_best])
                best_epoch = best_index + 1
                best_monitored_value = float(monitored_values[best_index])
        if best_epoch is None and epochs_ran > 0:
            best_epoch = epochs_ran

        selected_decision_threshold = float(decision_thresholds[0])
        threshold_validation_score: float | None = None
        threshold_search_results: list[dict] = []
        if validation_subjects_per_fold > 0:
            validation_probabilities = _predict_probabilities(
                model=model,
                X=X_validation,
                batch_size=current_batch_size,
                n_prediction_latent_samples=n_prediction_latent_samples,
                latent_sampling_seed=latent_sampling_seed,
            )
            if threshold_selection_level == "trial":
                if _is_trial_tensor(X_validation):
                    threshold_validation = _direct_trial_aggregation(
                        probabilities=validation_probabilities,
                        y_true=y_validation,
                        subject_ids=subject_ids_validation,
                        trial_ids=trial_ids_validation,
                        n_windows_per_trial=X_validation.shape[1],
                    )
                else:
                    threshold_validation = _aggregate_window_probabilities_by_trial(
                        probabilities=validation_probabilities,
                        y_true=y_validation,
                        subject_ids=subject_ids_validation,
                        trial_ids=trial_ids_validation,
                    )
                threshold_probabilities = threshold_validation["probabilities"]
                threshold_y_true = threshold_validation["y_true"]
            else:
                threshold_probabilities = validation_probabilities
                threshold_y_true = _as_numpy_1d(y_validation).astype(np.int64)

            (
                selected_decision_threshold,
                threshold_validation_score,
                threshold_search_results,
            ) = _select_binary_decision_threshold(
                probabilities=threshold_probabilities,
                y_true=threshold_y_true,
                thresholds=decision_thresholds,
                metric=threshold_selection_metric,
            )
            print(
                f"Fold {fold_number} selected decision threshold "
                f"{selected_decision_threshold:.4f} from validation "
                f"{threshold_selection_level}_{threshold_selection_metric}="
                f"{threshold_validation_score:.6f}",
                flush=True,
            )

        evaluation = _evaluate_classification_fold(
            model=model,
            X_test=X_test,
            y_test=y_test,
            subject_ids_test=subject_ids_test,
            trial_ids_test=trial_ids_test,
            fold_index=fold_number,
            metrics=metrics,
            evaluation_level=evaluation_level,
            batch_size=current_batch_size,
            n_prediction_latent_samples=n_prediction_latent_samples,
            latent_sampling_seed=latent_sampling_seed,
            log_predictions=log_predictions,
            log_variational_intervals=log_variational_intervals,
            n_uncertainty_samples=n_uncertainty_samples,
            ci_level=ci_level,
            decision_threshold=selected_decision_threshold,
        )
    finally:
        del model
        gc.collect()
        tf.keras.backend.clear_session()

    def count_trials(subject_ids: np.ndarray, trial_ids: np.ndarray) -> int:
        return int(len(set(zip(subject_ids.tolist(), trial_ids.tolist()))))

    subject_ids_outer_train = subject_id_array[outer_train_indices]
    trial_ids_outer_train = trial_id_array[outer_train_indices]

    fold_record = {
        "fold_number": int(fold_number),
        "left_out_subjects": [_python_scalar(test_subject)],
        "validation_subjects": [
            _python_scalar(value) for value in validation_subjects.tolist()
        ],
        "n_train_windows": _count_windows_for_indices(
            feature_array, outer_train_indices
        ),
        "n_fit_train_windows": _count_windows_for_indices(
            feature_array, fit_train_indices
        ),
        "n_validation_windows": _count_windows_for_indices(
            feature_array, validation_indices
        ),
        "n_test_windows": _count_windows_for_indices(feature_array, test_indices),
        "n_train_trials": count_trials(subject_ids_outer_train, trial_ids_outer_train),
        "n_fit_train_trials": count_trials(subject_ids_fit_train, trial_ids_fit_train),
        "n_validation_trials": count_trials(
            subject_ids_validation, trial_ids_validation
        ),
        "n_test_trials": count_trials(subject_ids_test, trial_ids_test),
        "epochs_ran": int(epochs_ran),
        "best_epoch": None if best_epoch is None else int(best_epoch),
        "best_monitored_value": best_monitored_value,
        "stopped_early": bool(stopped_early),
        "decision_threshold": float(selected_decision_threshold),
        "alternate_subject_sets": bool(alternate_subject_sets),
        "use_mldg": bool(use_mldg),
        "mldg_meta_train_subjects": int(mldg_meta_train_subjects) if use_mldg else 0,
        "mldg_meta_test_subjects": int(mldg_meta_test_subjects) if use_mldg else 0,
        "mldg_samples_per_subject": int(mldg_samples_per_subject) if use_mldg else 0,
        "subject_set_a": (
            [_python_scalar(value) for value in subject_set_a.tolist()]
            if alternate_subject_sets
            else []
        ),
        "subject_set_b": (
            [_python_scalar(value) for value in subject_set_b.tolist()]
            if alternate_subject_sets
            else []
        ),
    }

    prediction_diagnostics_log = (
        []
        if prediction_diagnostics_callback is None
        else list(prediction_diagnostics_callback.history)
    )

    return {
        "outer_fold_number": int(fold_number),
        "fold_record": fold_record,
        "prediction_diagnostics_log": prediction_diagnostics_log,
        "oracle_epoch_log": list(oracle_metrics_callback.history),
        **evaluation,
    }


def _loso_fold_process_main(
    worker_state_payload: bytes,
    task_queue,
    result_queue,
    gpu_id: int | None,
    cpus_per_worker: int | None,
    assigned_device_label: str | None,
) -> None:
    """Run ordinary LOSO folds in one persistent spawned process."""
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

            fold_number, test_subject = task

            try:
                fold_output = _run_loso_fold(
                    fold_number=fold_number,
                    test_subject=test_subject,
                    **worker_state,
                )
                result_queue.put(("ok", int(fold_number), fold_output))
            except BaseException:
                result_queue.put(
                    (
                        "error",
                        int(fold_number),
                        traceback.format_exc(),
                    )
                )
                return

    except BaseException:
        result_queue.put(("error", -1, traceback.format_exc()))
    finally:
        tf.keras.backend.clear_session()
        gc.collect()
