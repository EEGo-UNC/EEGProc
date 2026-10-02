"""Nested leave-N-subjects-out cross-validation entry point."""

from __future__ import annotations

from itertools import combinations
import numpy as np
from pprint import pformat

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotations only
    from typing import Callable
    from typing import Literal
    import tensorflow as tf

from .constants import _CLASSIFICATION_METRICS
from .workers import _auto_assign_gpu_ids, _run_spawned_fold_pool
from .reporting import _mean_std_rows, _validate_evaluation_level
from .search import _expand_hyperparameter_grid, _warn_if_joint_loss_weights_vary
from .nested_fold import _outer_fold_process_main, _run_outer_fold

def nested_lnso_cv(
    model_builder_function: Callable[..., tf.keras.Model],
    feature_array: np.ndarray,
    label_array: np.ndarray,
    subject_id_array: np.ndarray,
    trial_id_array: np.ndarray | None = None,
    n_outer_subjects_to_leave_out: int = 1,
    n_inner_subjects_to_leave_out: int = 1,
    n_epochs: int = 50,
    batch_size: int = 32,
    hyperparameters: dict | None = None,
    preprocessing_strategy: Callable | None = None,
    selection_metric: str = "joint_loss",
    selection_level: Literal["window", "trial"] = "window",
    evaluation_level: Literal["window", "trial"] = "trial",
    maximize_metric: bool | None = None,
    metrics: list[str] | tuple[str, ...] = ("accuracy", "f1", "precision", "recall"),
    log_predictions: bool = True,
    log_variational_intervals: bool = False,
    n_prediction_latent_samples: int = 0,
    latent_sampling_seed: int | None = None,
    n_uncertainty_samples: int = 30,
    ci_level: float = 0.95,
    verbose: int = 0,
    extra_fit_kwargs: dict | None = None,
    n_jobs: int = 1,
    gpu_ids: list[int] | tuple[int, ...] | None = None,
    cpus_per_worker: int | None = None,
) -> dict:
    """Run nested Leave-N-Subjects-Out CV.

    Outer folds can be executed concurrently using multiprocessing with the
    ``spawn`` start method. Inner folds and hyperparameter configurations remain
    sequential inside each worker, preventing nested process oversubscription.

    Trial-level evaluation
    ----------------------
    ``trial_id_array`` must contain one trial ID per input window. Predictions
    are made per window, then probabilities are averaged within each
    ``(subject_id, trial_id)`` group. ``selection_level`` controls inner-CV
    hyperparameter selection and ``evaluation_level`` controls the unprefixed
    primary metrics. Both window- and trial-level metrics/logs are always
    returned.

    Parameters added for concurrency
    --------------------------------
    n_jobs:
        Number of persistent outer-fold worker processes. ``1`` preserves the
        original sequential behavior unless ``gpu_ids`` is supplied.
    gpu_ids:
        Local GPU indices assigned one-per-worker. For example,
        ``gpu_ids=(0, 1, 2, 3)`` with ``n_jobs=4``. When this is ``None`` and
        multiple workers are requested, visible Slurm/TensorFlow GPUs are
        assigned automatically, one per worker.
    cpus_per_worker:
        TensorFlow intra-op CPU threads available to each worker. Keep
        ``n_jobs * cpus_per_worker`` within the CPUs allocated by Slurm.

    Notes
    -----
    Worker state is serialized with cloudpickle, so locally defined model
    builders and preprocessing callables are supported. The training entry
    point must still be protected by ``if __name__ == "__main__":``.
    """
    extra_fit_kwargs = extra_fit_kwargs or {}

    if "validation_data" in extra_fit_kwargs:
        raise ValueError(
            "Do not pass validation_data in extra_fit_kwargs. "
            "nested_lnso_cv creates validation_data from the inner folds."
        )

    if subject_id_array is None:
        raise ValueError("subject_id_array is required for nested LNSO CV.")
    if trial_id_array is None:
        raise ValueError(
            "trial_id_array is required for trial-level prediction and metrics. "
            "Pass one trial ID per window, aligned with feature_array."
        )

    _validate_evaluation_level(selection_level, "selection_level")
    _validate_evaluation_level(evaluation_level, "evaluation_level")

    feature_array = np.asarray(feature_array)
    label_array = np.asarray(label_array)
    subject_id_array = np.asarray(subject_id_array)
    trial_id_array = np.asarray(trial_id_array)

    input_lengths = (
        len(feature_array), len(label_array), len(subject_id_array), len(trial_id_array)
    )
    if len(set(input_lengths)) != 1:
        raise ValueError(
            "feature_array, label_array, subject_id_array, and trial_id_array "
            "must have the same first dimension. Got lengths "
            f"{input_lengths}."
        )

    metrics = tuple(metrics)

    for metric in metrics:
        if metric not in _CLASSIFICATION_METRICS:
            raise ValueError(
                f"Unsupported metric: {metric}. Supported metrics: "
                f"{sorted(_CLASSIFICATION_METRICS)}"
            )

    allowed_selection_metrics = {"loss", "joint_loss", *metrics}

    if selection_metric not in allowed_selection_metrics:
        raise ValueError(
            f"selection_metric='{selection_metric}' is not available. "
            f"Use 'loss', 'joint_loss', or one of metrics={list(metrics)}."
        )

    if selection_metric == "joint_loss" and selection_level != "window":
        raise ValueError(
            "selection_metric='joint_loss' requires selection_level='window' "
            "because the complete Keras VAE+VC objective is evaluated over "
            "validation/test windows, before trial aggregation."
        )

    if maximize_metric is None:
        maximize_metric = selection_metric not in {"loss", "joint_loss"}

    if n_prediction_latent_samples < 0:
        raise ValueError("n_prediction_latent_samples must be >= 0.")

    if not (0.0 < ci_level < 1.0):
        raise ValueError("ci_level must be between 0 and 1.")

    if n_jobs < 1:
        raise ValueError("n_jobs must be >= 1.")

    if cpus_per_worker is not None and cpus_per_worker < 1:
        raise ValueError("cpus_per_worker must be >= 1 when provided.")

    unique_subjects = np.sort(np.unique(subject_id_array))

    if n_outer_subjects_to_leave_out < 1:
        raise ValueError("n_outer_subjects_to_leave_out must be >= 1.")

    if n_inner_subjects_to_leave_out < 1:
        raise ValueError("n_inner_subjects_to_leave_out must be >= 1.")

    if n_outer_subjects_to_leave_out >= len(unique_subjects):
        raise ValueError(
            "n_outer_subjects_to_leave_out must be smaller than the number "
            f"of unique subjects. Got {n_outer_subjects_to_leave_out} for "
            f"{len(unique_subjects)} subjects."
        )

    n_outer_train_subjects = len(unique_subjects) - n_outer_subjects_to_leave_out

    if n_inner_subjects_to_leave_out >= n_outer_train_subjects:
        raise ValueError(
            "n_inner_subjects_to_leave_out must be smaller than the number "
            "of subjects available in each outer-training pool. Got "
            f"{n_inner_subjects_to_leave_out} for {n_outer_train_subjects} "
            "outer-training subjects."
        )

    if hyperparameters is None:
        hyperparameters = {}

    effective_hyperparameters = {
        "epochs": n_epochs,
        "batch_size": batch_size,
        **hyperparameters,
    }

    grid_configs = _expand_hyperparameter_grid(effective_hyperparameters)
    _warn_if_joint_loss_weights_vary(grid_configs, selection_metric)
    outer_subject_splits = list(
        combinations(unique_subjects, n_outer_subjects_to_leave_out)
    )
    total_outer_folds = len(outer_subject_splits)
    effective_n_jobs = min(n_jobs, total_outer_folds)

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
                f"n_jobs={effective_n_jobs} requires at least that many GPU IDs, "
                f"but gpu_ids={normalized_gpu_ids}. Use one GPU per worker."
            )

        normalized_gpu_ids = normalized_gpu_ids[:effective_n_jobs]

    results = {
        "fold_metrics": [],
        "window_fold_metrics": [],
        "trial_fold_metrics": [],
        "user_metrics": [],
        "prediction_log": [],
        "window_prediction_log": [],
        "trial_prediction_log": [],
        "variational_interval_log": [],
        "window_variational_interval_log": [],
        "trial_variational_interval_log": [],
        "best_configs": [],
        "inner_cv_results": [],
        "outer_fold_results": [],
        "mean_scores": {},
        "std_scores": {},
        "window_mean_scores": {},
        "window_std_scores": {},
        "trial_mean_scores": {},
        "trial_std_scores": {},
    }

    print(
        f"\nNested LNSO CV — {total_outer_folds} outer folds, "
        f"{len(grid_configs)} hyperparameter config"
        f"{'s' if len(grid_configs) != 1 else ''}"
    )
    print(f"Requested metrics: {list(metrics)}")
    print(
        f"Selection metric: {selection_level}-level {selection_metric} "
        f"({'maximize' if maximize_metric else 'minimize'})"
    )
    print(f"Primary reported metrics: {evaluation_level}-level")
    print(f"Prediction logging: {log_predictions}")
    print(f"Variational interval logging: {log_variational_intervals}")
    prediction_mode = (
        "posterior mean"
        if n_prediction_latent_samples == 0
        else f"MC average over {n_prediction_latent_samples} latent sample(s)"
    )
    print(f"Prediction latent mode: {prediction_mode}")
    print(f"Outer-fold workers: {effective_n_jobs}")

    if effective_n_jobs > 1 and normalized_gpu_ids is None:
        print("Worker devices: CPU-only")
    elif normalized_gpu_ids is not None:
        print(f"Worker devices: GPUs {list(normalized_gpu_ids)}")
    else:
        print("Worker device: current TensorFlow default")

    tasks = [
        (fold_number, tuple(outer_test_subjects))
        for fold_number, outer_test_subjects in enumerate(
            outer_subject_splits,
            start=1,
        )
    ]

    worker_state = {
        "total_outer_folds": total_outer_folds,
        "model_builder_function": model_builder_function,
        "feature_array": feature_array,
        "label_array": label_array,
        "subject_id_array": subject_id_array,
        "trial_id_array": trial_id_array,
        "n_inner_subjects_to_leave_out": n_inner_subjects_to_leave_out,
        "grid_configs": grid_configs,
        "batch_size": batch_size,
        "preprocessing_strategy": preprocessing_strategy,
        "selection_metric": selection_metric,
        "selection_level": selection_level,
        "evaluation_level": evaluation_level,
        "maximize_metric": bool(maximize_metric),
        "metrics": metrics,
        "log_predictions": log_predictions,
        "log_variational_intervals": log_variational_intervals,
        "n_prediction_latent_samples": n_prediction_latent_samples,
        "latent_sampling_seed": latent_sampling_seed,
        "n_uncertainty_samples": n_uncertainty_samples,
        "ci_level": ci_level,
        "verbose": verbose,
        "extra_fit_kwargs": extra_fit_kwargs,
    }

    # Preserve the old in-process behavior for n_jobs=1 with no explicit GPU
    # assignment. This avoids spawn/pickling overhead for ordinary runs.
    if effective_n_jobs == 1 and normalized_gpu_ids is None:
        fold_outputs = [
            _run_outer_fold(
                outer_fold_number=fold_number,
                outer_test_subjects=np.asarray(outer_test_subjects),
                **worker_state,
            )
            for fold_number, outer_test_subjects in tasks
        ]
    else:
        fold_outputs = _run_spawned_fold_pool(
            worker_target=_outer_fold_process_main,
            worker_state=worker_state,
            tasks=tasks,
            n_workers=effective_n_jobs,
            gpu_ids=normalized_gpu_ids,
            cpus_per_worker=cpus_per_worker,
            worker_name_prefix="OuterFoldWorker",
            worker_description="outer-fold",
        )

    # Results arrive in completion order, so restore deterministic fold order.
    fold_outputs.sort(key=lambda row: row["outer_fold_number"])

    for fold_output in fold_outputs:
        results["fold_metrics"].append(fold_output["fold_metrics"])
        results["window_fold_metrics"].append(fold_output["window_fold_metrics"])
        results["trial_fold_metrics"].append(fold_output["trial_fold_metrics"])
        results["user_metrics"].extend(fold_output["user_metrics"])
        results["prediction_log"].extend(fold_output["prediction_log"])
        results["window_prediction_log"].extend(fold_output["window_prediction_log"])
        results["trial_prediction_log"].extend(fold_output["trial_prediction_log"])
        results["variational_interval_log"].extend(
            fold_output["variational_interval_log"]
        )
        results["window_variational_interval_log"].extend(
            fold_output["window_variational_interval_log"]
        )
        results["trial_variational_interval_log"].extend(
            fold_output["trial_variational_interval_log"]
        )
        results["best_configs"].append(fold_output["best_config_result"])
        results["inner_cv_results"].append(fold_output["inner_cv_result"])
        results["outer_fold_results"].append(fold_output["outer_fold_result"])

    mean_scores, std_scores = _mean_std_rows(
        results["fold_metrics"],
        ["loss", *metrics, "decoder_accuracy"],
    )

    results["mean_scores"] = mean_scores
    results["std_scores"] = std_scores

    window_mean_scores, window_std_scores = _mean_std_rows(
        results["window_fold_metrics"],
        ["loss", *metrics, "decoder_accuracy"],
    )
    trial_mean_scores, trial_std_scores = _mean_std_rows(
        results["trial_fold_metrics"], ["loss", *metrics]
    )
    results["window_mean_scores"] = window_mean_scores
    results["window_std_scores"] = window_std_scores
    results["trial_mean_scores"] = trial_mean_scores
    results["trial_std_scores"] = trial_std_scores

    print("\nNested LNSO CV complete")
    print("=" * 80)
    print("Mean outer scores:")
    print(pformat(mean_scores, indent=4, width=120, sort_dicts=False))
    print("Std outer scores:")
    print(pformat(std_scores, indent=4, width=120, sort_dicts=False))
    print("Window-level mean scores:")
    print(pformat(window_mean_scores, indent=4, width=120, sort_dicts=False))
    print("Trial-level mean scores:")
    print(pformat(trial_mean_scores, indent=4, width=120, sort_dicts=False))

    return results
