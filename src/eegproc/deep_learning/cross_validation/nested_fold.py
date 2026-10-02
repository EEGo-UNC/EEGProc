"""The body of a single outer fold in nested leave-N-subjects-out CV."""

from __future__ import annotations

import cloudpickle
from itertools import combinations
import gc
import numpy as np
import tensorflow as tf
import traceback

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotations only
    from typing import Callable
    from typing import Literal

from .arrays import _as_numpy_1d
from .workers import _configure_tensorflow_worker
from .reporting import _mean_std_rows, _print_config, _print_fold_header
from .evaluation import _apply_preprocessing_strategy, _evaluate_classification_fold, _evaluate_inner_config, _validate_processed_alignment
from .search import _choose_best_config_index, _split_config

def _outer_fold_process_main(
    worker_state_payload: bytes,
    task_queue,
    result_queue,
    gpu_id: int | None,
    cpus_per_worker: int | None,
    assigned_device_label: str | None,
) -> None:
    """Run outer-fold tasks in one persistent spawned process."""
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

            outer_fold_number, outer_test_subjects = task

            try:
                fold_output = _run_outer_fold(
                    outer_fold_number=outer_fold_number,
                    outer_test_subjects=np.asarray(outer_test_subjects),
                    **worker_state,
                )
                result_queue.put(("ok", int(outer_fold_number), fold_output))
            except BaseException:
                result_queue.put(
                    (
                        "error",
                        int(outer_fold_number),
                        traceback.format_exc(),
                    )
                )
                return

    except BaseException:
        result_queue.put(("error", -1, traceback.format_exc()))
    finally:
        tf.keras.backend.clear_session()
        gc.collect()


def _run_outer_fold(
    outer_fold_number: int,
    outer_test_subjects: np.ndarray,
    total_outer_folds: int,
    model_builder_function: Callable[..., tf.keras.Model],
    feature_array: np.ndarray,
    label_array: np.ndarray,
    subject_id_array: np.ndarray,
    trial_id_array: np.ndarray,
    n_inner_subjects_to_leave_out: int,
    grid_configs: list[dict],
    batch_size: int,
    preprocessing_strategy: Callable | None,
    selection_metric: str,
    selection_level: Literal["window", "trial"],
    evaluation_level: Literal["window", "trial"],
    maximize_metric: bool,
    metrics: tuple[str, ...],
    log_predictions: bool,
    log_variational_intervals: bool,
    n_prediction_latent_samples: int,
    latent_sampling_seed: int | None,
    n_uncertainty_samples: int,
    ci_level: float,
    verbose: int,
    extra_fit_kwargs: dict,
) -> dict:
    """Run one complete outer fold, including its inner grid search."""
    outer_test_mask = np.isin(subject_id_array, outer_test_subjects)
    outer_train_mask = ~outer_test_mask

    outer_train_indices = np.where(outer_train_mask)[0]
    outer_test_indices = np.where(outer_test_mask)[0]

    outer_train_subject_ids = subject_id_array[outer_train_indices]
    unique_outer_train_subjects = np.sort(np.unique(outer_train_subject_ids))

    inner_subject_splits = list(
        combinations(unique_outer_train_subjects, n_inner_subjects_to_leave_out)
    )

    _print_fold_header(
        outer_fold_number,
        total_outer_folds,
        f"outer test subjects={outer_test_subjects.tolist()} "
        f"(outer_train={len(outer_train_indices)}, "
        f"outer_test={len(outer_test_indices)} windows)",
    )

    inner_scores_by_config: list[list[dict]] = [[] for _ in grid_configs]
    inner_fold_results: list[dict] = []

    # -----------------------------------------------------------------
    # Inner CV: choose hyperparameters.
    # -----------------------------------------------------------------
    for inner_fold_number, inner_val_subjects in enumerate(
        inner_subject_splits,
        start=1,
    ):
        inner_val_subjects = np.asarray(inner_val_subjects)

        inner_val_mask_relative = np.isin(
            outer_train_subject_ids,
            inner_val_subjects,
        )
        inner_train_mask_relative = ~inner_val_mask_relative

        inner_train_indices = outer_train_indices[inner_train_mask_relative]
        inner_val_indices = outer_train_indices[inner_val_mask_relative]

        X_inner_train = feature_array[inner_train_indices]
        y_inner_train = label_array[inner_train_indices]
        X_inner_val = feature_array[inner_val_indices]
        y_inner_val = label_array[inner_val_indices]
        subject_ids_inner_train = subject_id_array[inner_train_indices]
        subject_ids_inner_val = subject_id_array[inner_val_indices]
        trial_ids_inner_train = trial_id_array[inner_train_indices]
        trial_ids_inner_val = trial_id_array[inner_val_indices]

        (
            X_inner_train,
            y_inner_train,
            X_inner_val,
            y_inner_val,
        ) = _apply_preprocessing_strategy(
            preprocessing_strategy=preprocessing_strategy,
            X_train=X_inner_train,
            y_train=y_inner_train,
            X_eval=X_inner_val,
            y_eval=y_inner_val,
            train_indices=inner_train_indices,
            eval_indices=inner_val_indices,
        )

        _validate_processed_alignment(
            X_inner_train, y_inner_train, subject_ids_inner_train,
            trial_ids_inner_train, "inner-training"
        )
        _validate_processed_alignment(
            X_inner_val, y_inner_val, subject_ids_inner_val,
            trial_ids_inner_val, "inner-validation"
        )

        config_results_this_inner_fold: list[dict] = []

        for config_index, config in enumerate(grid_configs):
            model_hp, fit_hp = _split_config(config)
            current_batch_size = fit_hp.get("batch_size", batch_size)

            tf.keras.backend.clear_session()
            model = model_builder_function(**model_hp)

            try:
                fit_kwargs = dict(fit_hp)
                fit_kwargs["validation_data"] = (X_inner_val, y_inner_val)

                y_inner_train_ids = _as_numpy_1d(y_inner_train)
                classes, counts = np.unique(y_inner_train_ids, return_counts=True)

                class_weight = {
                    int(class_id): len(y_inner_train_ids) / (len(classes) * count)
                    for class_id, count in zip(classes, counts)
                }

                model.fit(
                    X_inner_train,
                    y_inner_train,
                    class_weight=class_weight,
                    verbose=verbose,
                    **fit_kwargs,
                    **extra_fit_kwargs,
                )

                val_scores = _evaluate_inner_config(
                    model=model,
                    X_val=X_inner_val,
                    y_val=y_inner_val,
                    subject_ids_val=subject_ids_inner_val,
                    trial_ids_val=trial_ids_inner_val,
                    metrics=metrics,
                    selection_level=selection_level,
                    batch_size=current_batch_size,
                    n_prediction_latent_samples=n_prediction_latent_samples,
                    latent_sampling_seed=latent_sampling_seed,
                )

                config_result = {
                    "outer_fold": int(outer_fold_number),
                    "inner_fold": int(inner_fold_number),
                    "left_out_subjects": inner_val_subjects.tolist(),
                    "config_index": int(config_index),
                    "config": dict(config),
                    **val_scores,
                }

                config_results_this_inner_fold.append(config_result)
                inner_scores_by_config[config_index].append(val_scores)

            finally:
                del model
                gc.collect()
                tf.keras.backend.clear_session()

        inner_fold_results.append(
            {
                "outer_fold": int(outer_fold_number),
                "inner_fold": int(inner_fold_number),
                "left_out_subjects": inner_val_subjects.tolist(),
                "n_train_windows": int(len(inner_train_indices)),
                "n_val_windows": int(len(inner_val_indices)),
                "n_train_trials": int(len(set(zip(
                    subject_ids_inner_train.tolist(), trial_ids_inner_train.tolist()
                )))),
                "n_val_trials": int(len(set(zip(
                    subject_ids_inner_val.tolist(), trial_ids_inner_val.tolist()
                )))),
                "configs": config_results_this_inner_fold,
            }
        )

    # -----------------------------------------------------------------
    # Aggregate inner-CV scores and choose the best configuration.
    # -----------------------------------------------------------------
    inner_mean_scores: list[dict] = []
    inner_std_scores: list[dict] = []
    score_metric_names = [
        "loss", "joint_loss", *metrics, "decoder_accuracy",
        "window_loss", "window_joint_loss", "window_keras_model_loss", "window_decoder_accuracy",
        *[f"window_{metric}" for metric in metrics],
        "trial_loss", *[f"trial_{metric}" for metric in metrics],
    ]

    for config_index, config in enumerate(grid_configs):
        mean_scores_for_config, std_scores_for_config = _mean_std_rows(
            inner_scores_by_config[config_index],
            score_metric_names,
        )

        inner_mean_scores.append(
            {
                "config_index": int(config_index),
                "config": dict(config),
                **mean_scores_for_config,
            }
        )
        inner_std_scores.append(
            {
                "config_index": int(config_index),
                "config": dict(config),
                **std_scores_for_config,
            }
        )

    best_config_index = _choose_best_config_index(
        mean_scores=inner_mean_scores,
        selection_metric=selection_metric,
        maximize_metric=maximize_metric,
    )
    best_config = grid_configs[best_config_index]

    print(
        f"\nBest config from inner CV for outer fold {outer_fold_number}: "
        f"{selection_metric}="
        f"{inner_mean_scores[best_config_index][selection_metric]:.6f}",
        flush=True,
    )
    _print_config("Best config:", best_config)

    best_config_result = {
        "outer_fold": int(outer_fold_number),
        "best_config_index": int(best_config_index),
        "best_config": dict(best_config),
        "selection_metric": selection_metric,
        "selection_level": selection_level,
        "selection_score": float(
            inner_mean_scores[best_config_index][selection_metric]
        ),
    }

    inner_cv_result = {
        "outer_fold": int(outer_fold_number),
        "inner_fold_results": inner_fold_results,
        "inner_mean_scores": inner_mean_scores,
        "inner_std_scores": inner_std_scores,
    }

    # -----------------------------------------------------------------
    # Final outer training and testing.
    # -----------------------------------------------------------------
    X_outer_train = feature_array[outer_train_indices]
    y_outer_train = label_array[outer_train_indices]
    X_outer_test = feature_array[outer_test_indices]
    y_outer_test = label_array[outer_test_indices]
    subject_ids_outer_train = subject_id_array[outer_train_indices]
    subject_ids_outer_test = subject_id_array[outer_test_indices]
    trial_ids_outer_train = trial_id_array[outer_train_indices]
    trial_ids_outer_test = trial_id_array[outer_test_indices]

    (
        X_outer_train,
        y_outer_train,
        X_outer_test,
        y_outer_test,
    ) = _apply_preprocessing_strategy(
        preprocessing_strategy=preprocessing_strategy,
        X_train=X_outer_train,
        y_train=y_outer_train,
        X_eval=X_outer_test,
        y_eval=y_outer_test,
        train_indices=outer_train_indices,
        eval_indices=outer_test_indices,
    )

    _validate_processed_alignment(
        X_outer_train, y_outer_train, subject_ids_outer_train,
        trial_ids_outer_train, "outer-training"
    )
    _validate_processed_alignment(
        X_outer_test, y_outer_test, subject_ids_outer_test,
        trial_ids_outer_test, "outer-test"
    )

    model_hp, fit_hp = _split_config(best_config)
    current_batch_size = fit_hp.get("batch_size", batch_size)

    tf.keras.backend.clear_session()
    final_model = model_builder_function(**model_hp)

    try:
        y_outer_train_ids = _as_numpy_1d(y_outer_train)
        classes, counts = np.unique(y_outer_train_ids, return_counts=True)

        class_weight = {
            int(class_id): len(y_outer_train_ids) / (len(classes) * count)
            for class_id, count in zip(classes, counts)
        }

        final_model.fit(
            X_outer_train,
            y_outer_train,
            class_weight=class_weight,
            verbose=verbose,
            **fit_hp,
            **extra_fit_kwargs,
        )

        fold_result = _evaluate_classification_fold(
            model=final_model,
            X_test=X_outer_test,
            y_test=y_outer_test,
            subject_ids_test=subject_ids_outer_test,
            trial_ids_test=trial_ids_outer_test,
            fold_index=outer_fold_number,
            metrics=metrics,
            evaluation_level=evaluation_level,
            batch_size=current_batch_size,
            n_prediction_latent_samples=n_prediction_latent_samples,
            latent_sampling_seed=latent_sampling_seed,
            log_predictions=log_predictions,
            log_variational_intervals=log_variational_intervals,
            n_uncertainty_samples=n_uncertainty_samples,
            ci_level=ci_level,
        )

    finally:
        del final_model
        gc.collect()
        tf.keras.backend.clear_session()

    outer_fold_result = {
        "outer_fold_number": int(outer_fold_number),
        "left_out_subjects": outer_test_subjects.tolist(),
        "n_outer_train_windows": int(len(outer_train_indices)),
        "n_outer_test_windows": int(len(outer_test_indices)),
        "n_outer_train_trials": int(len(set(zip(
            subject_ids_outer_train.tolist(), trial_ids_outer_train.tolist()
        )))),
        "n_outer_test_trials": int(len(set(zip(
            subject_ids_outer_test.tolist(), trial_ids_outer_test.tolist()
        )))),
        "selection_level": selection_level,
        "evaluation_level": evaluation_level,
        "best_config": dict(best_config),
        "inner_fold_results": inner_fold_results,
        "inner_mean_scores": inner_mean_scores,
        "inner_std_scores": inner_std_scores,
        "fold_metrics": fold_result["fold_metrics"],
        "window_fold_metrics": fold_result["window_fold_metrics"],
        "trial_fold_metrics": fold_result["trial_fold_metrics"],
        "user_metrics": fold_result["user_metrics"],
        "prediction_log": fold_result["prediction_log"],
        "window_prediction_log": fold_result["window_prediction_log"],
        "trial_prediction_log": fold_result["trial_prediction_log"],
        "variational_interval_log": fold_result["variational_interval_log"],
        "window_variational_interval_log": fold_result["window_variational_interval_log"],
        "trial_variational_interval_log": fold_result["trial_variational_interval_log"],
    }

    return {
        "outer_fold_number": int(outer_fold_number),
        "best_config_result": best_config_result,
        "inner_cv_result": inner_cv_result,
        "outer_fold_result": outer_fold_result,
        **fold_result,
    }
