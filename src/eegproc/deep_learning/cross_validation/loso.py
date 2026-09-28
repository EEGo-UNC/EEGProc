"""Leave-one-subject-out cross-validation entry points."""

from __future__ import annotations

import numpy as np
from pprint import pformat

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotations only
    from typing import Callable
    from typing import Literal
    import tensorflow as tf

from .arrays import _python_scalar
from .constants import _CLASSIFICATION_METRICS
from .workers import _auto_assign_gpu_ids, _run_spawned_fold_pool
from .probabilities import _normalize_decision_thresholds
from .reporting import _print_config, _validate_evaluation_level
from .search import _aggregate_loso_config_result, _choose_best_loso_config_index, _expand_hyperparameter_grid, _loso_config_sort_key, _warn_if_joint_loss_weights_vary
from .loso_fold import _loso_fold_process_main, _run_loso_fold

def _validate_loso_cv_arguments(
    subject_id_array: np.ndarray,
    trial_id_array: np.ndarray,
    selection_metric: str,
    selection_level: Literal["window", "trial"],
    maximize_metric: bool | None,
    metrics: tuple[str, ...],
    n_prediction_latent_samples: int,
    ci_level: float,
    n_jobs: int,
    cpus_per_worker: int | None,
    validation_subjects_per_fold: int,
    alternate_subject_sets: bool,
    use_mldg: bool,
    mldg_meta_train_subjects: int,
    mldg_meta_test_subjects: int,
    mldg_samples_per_subject: int,
    mldg_seed: int | None,
    validation_seed: int | None,
    early_stopping_patience: int | None,
    early_stopping_min_delta: float,
    early_stopping_monitor: str,
    early_stopping_mode: Literal["auto", "min", "max"],
    prediction_diagnostics_every_n_epochs: int,
    prediction_diagnostics_max_samples: int,
    prediction_diagnostics_threshold_tolerance: float,
    decision_thresholds: list[float] | tuple[float, ...],
    threshold_selection_level: Literal["window", "trial"],
    max_folds: int | None,
) -> tuple[bool, tuple[float, ...], np.ndarray]:
    """Validate user-facing LOSO CV options and return normalized values."""
    if n_jobs < 1:
        raise ValueError("n_jobs must be >= 1.")
    if cpus_per_worker is not None and cpus_per_worker < 1:
        raise ValueError("cpus_per_worker must be >= 1 when provided.")
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
    if validation_seed is not None and validation_seed < 0:
        raise ValueError("validation_seed must be >= 0 or None.")
    if early_stopping_patience is not None and early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be >= 0 or None.")
    if early_stopping_min_delta < 0.0:
        raise ValueError("early_stopping_min_delta must be >= 0.")
    if early_stopping_mode not in {"auto", "min", "max"}:
        raise ValueError("early_stopping_mode must be 'auto', 'min', or 'max'.")
    if not early_stopping_monitor:
        raise ValueError("early_stopping_monitor must be a non-empty string.")
    if prediction_diagnostics_every_n_epochs < 1:
        raise ValueError("prediction_diagnostics_every_n_epochs must be at least 1.")
    if prediction_diagnostics_max_samples < 1:
        raise ValueError("prediction_diagnostics_max_samples must be at least 1.")
    if prediction_diagnostics_threshold_tolerance < 0.0:
        raise ValueError(
            "prediction_diagnostics_threshold_tolerance must be non-negative."
        )
    if n_prediction_latent_samples < 0:
        raise ValueError("n_prediction_latent_samples must be >= 0.")
    if not (0.0 < ci_level < 1.0):
        raise ValueError("ci_level must be between 0 and 1.")

    decision_thresholds = _normalize_decision_thresholds(decision_thresholds)
    _validate_evaluation_level(threshold_selection_level, "threshold_selection_level")

    if len(decision_thresholds) > 1 and validation_subjects_per_fold == 0:
        raise ValueError(
            "Testing multiple decision thresholds requires fold-local validation "
            "subjects. Set validation_subjects_per_fold >= 1."
        )
    if (
        early_stopping_monitor
        in {"val_trial_f1", "val_trial_balanced_accuracy", "val_trial_loss"}
        and validation_subjects_per_fold == 0
        and early_stopping_patience is not None
    ):
        raise ValueError(
            f"{early_stopping_monitor} requires at least one fold-local "
            "validation subject. Set validation_subjects_per_fold >= 1."
        )

    if selection_metric not in {"loss", "joint_loss", *metrics}:
        raise ValueError(
            f"selection_metric={selection_metric!r} is unavailable. "
            f"Use 'loss', 'joint_loss', or one of metrics={list(metrics)}."
        )

    if maximize_metric is None:
        maximize_metric = selection_metric not in {"loss", "joint_loss"}

    unique_subjects = np.sort(np.unique(subject_id_array))
    if len(unique_subjects) < 2:
        raise ValueError(
            "LOSO CV requires at least two unique subjects. "
            f"Got {len(unique_subjects)}."
        )
    if validation_subjects_per_fold >= len(unique_subjects) - 1:
        raise ValueError(
            "validation_subjects_per_fold must leave at least one gradient-"
            "training subject after the LOSO test subject is removed. Got "
            f"{validation_subjects_per_fold} validation subjects for "
            f"{len(unique_subjects)} total subjects."
        )
    if use_mldg:
        available_mldg_subjects = len(unique_subjects) - 1 - validation_subjects_per_fold
        required_mldg_subjects = int(mldg_meta_train_subjects) + int(
            mldg_meta_test_subjects
        )
        if required_mldg_subjects > available_mldg_subjects:
            raise ValueError(
                "MLDG requires "
                f"{required_mldg_subjects} episodic subjects, but each LOSO "
                f"fold leaves only {available_mldg_subjects} gradient-training "
                "subjects after removing test and validation subjects."
            )

    if max_folds is not None and max_folds < 1:
        raise ValueError("max_folds must be >= 1 when provided.")

    test_subjects = (
        unique_subjects[: min(max_folds, len(unique_subjects))]
        if max_folds is not None
        else unique_subjects
    )

    return maximize_metric, decision_thresholds, test_subjects


def _print_loso_cv_summary(
    num_configs: int,
    total_folds: int,
    num_subjects: int,
    max_folds: int | None,
    metrics: tuple[str, ...],
    selection_level: str,
    selection_metric: str,
    maximize_metric: bool,
    evaluation_level: str,
    log_predictions: bool,
    prediction_diagnostics: bool,
    log_variational_intervals: bool,
    decision_thresholds: tuple[float, ...],
    threshold_selection_level: str,
    threshold_selection_metric: str,
    n_prediction_latent_samples: int,
    validation_subjects_per_fold: int,
    validation_seed: int | None,
    early_stopping_monitor: str,
    early_stopping_patience: int | None,
    restore_best_weights: bool,
    use_mldg: bool,
    mldg_meta_train_subjects: int,
    mldg_meta_test_subjects: int,
    mldg_samples_per_subject: int,
    mldg_seed: int | None,
    alternate_subject_sets: bool,
    effective_n_jobs: int,
    normalized_gpu_ids: tuple[int, ...] | None,
) -> None:
    print(
        f"\nFlat LOSO hyperparameter search — {num_configs} "
        f"configuration{'s' if num_configs != 1 else ''}, "
        f"{total_folds} fold{'s' if total_folds != 1 else ''} each"
    )
    print(f"Total available subjects: {num_subjects}")
    print(f"Total LOSO model fits: {num_configs * total_folds}")
    if max_folds is not None:
        print(
            f"Smoke-test fold limit: {total_folds} of "
            f"{num_subjects} subjects per configuration"
        )
    print(f"Requested metrics: {list(metrics)}")
    print(
        f"Configuration selection: {selection_level}-level "
        f"{selection_metric} "
        f"({'maximize' if maximize_metric else 'minimize'})"
    )
    print(f"Primary reported metrics: {evaluation_level}-level")
    print(f"Prediction logging: {log_predictions}")
    print(f"Prediction diagnostics: {prediction_diagnostics}")
    print(f"Variational interval logging: {log_variational_intervals}")
    print(
        "Decision thresholds: "
        f"{list(decision_thresholds)}; selection="
        f"{threshold_selection_level}_{threshold_selection_metric}"
    )
    print(
        "Prediction latent mode: "
        + (
            "posterior mean"
            if n_prediction_latent_samples == 0
            else f"MC average over {n_prediction_latent_samples} latent sample(s)"
        )
    )
    if validation_subjects_per_fold > 0:
        print(
            "Per-fold validation: "
            f"{validation_subjects_per_fold} seeded subject(s), "
            f"seed={validation_seed}, monitor={early_stopping_monitor}, "
            f"patience={early_stopping_patience}, "
            f"restore_best_weights={restore_best_weights}"
        )
    else:
        print("Per-fold validation: disabled")
    if use_mldg:
        print(
            "Optimization: first-order MLDG with natural within-subject labels "
            f"(A={mldg_meta_train_subjects} subjects, "
            f"B={mldg_meta_test_subjects} subjects, "
            f"samples/subject={mldg_samples_per_subject}, seed={mldg_seed})"
        )
    elif alternate_subject_sets:
        print("Optimization: alternating fixed subject sets")
    else:
        print("Optimization: ordinary shuffled minibatches")
    print(f"Fold workers: {effective_n_jobs}")
    if effective_n_jobs > 1 and normalized_gpu_ids is None:
        print("Worker devices: CPU-only")
    elif normalized_gpu_ids is not None:
        print(f"Worker devices: GPUs {list(normalized_gpu_ids)}")
    else:
        print("Worker device: current TensorFlow default")


def loso_cv(
    model_builder_function: Callable[..., tf.keras.Model],
    feature_array: np.ndarray,
    label_array: np.ndarray,
    subject_id_array: np.ndarray,
    trial_id_array: np.ndarray | None = None,
    n_epochs: int = 50,
    batch_size: int = 2,
    hyperparameters: dict | None = None,
    preprocessing_strategy: Callable | None = None,
    evaluation_level: Literal["window", "trial"] = "trial",
    selection_metric: str = "f1",
    selection_level: Literal["window", "trial"] = "trial",
    maximize_metric: bool | None = None,
    metrics: list[str] | tuple[str, ...] = (
        "accuracy",
        "f1",
        "precision",
        "recall",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "balanced_accuracy",
        "binary_f1",
        "binary_precision",
        "binary_recall",
        "roc_auc",
    ),
    log_predictions: bool = True,
    log_variational_intervals: bool = False,
    n_prediction_latent_samples: int = 0,
    latent_sampling_seed: int | None = None,
    n_uncertainty_samples: int = 30,
    ci_level: float = 0.95,
    validation_subjects_per_fold: int = 0,
    validation_seed: int | None = 42,
    early_stopping_patience: int | None = 5,
    early_stopping_min_delta: float = 0.0,
    early_stopping_monitor: str = "val_loss",
    early_stopping_mode: Literal["auto", "min", "max"] = "min",
    restore_best_weights: bool = True,
    prediction_diagnostics: bool = False,
    prediction_diagnostics_every_n_epochs: int = 1,
    prediction_diagnostics_max_samples: int = 256,
    prediction_diagnostics_threshold_tolerance: float = 0.01,
    prediction_diagnostics_seed: int | None = 42,
    decision_thresholds: list[float] | tuple[float, ...] = (0.5,),
    threshold_selection_metric: Literal[
        "accuracy", "f1", "balanced_accuracy", "binary_f1"
    ] = "f1",
    threshold_selection_level: Literal["window", "trial"] = "trial",
    verbose: int = 0,
    extra_fit_kwargs: dict | None = None,
    n_jobs: int = 1,
    gpu_ids: list[int] | tuple[int, ...] | None = None,
    cpus_per_worker: int | None = None,
    max_folds: int | None = None,
    alternate_subject_sets: bool = False,
    alternating_subject_seed: int | None = 42,
    use_mldg: bool = False,
    mldg_meta_train_subjects: int = 6,
    mldg_meta_test_subjects: int = 2,
    mldg_samples_per_subject: int = 4,
    mldg_seed: int | None = 42,
) -> dict:
    """Run a flat hyperparameter search using complete LOSO evaluations.

    For every Cartesian-product hyperparameter configuration, each unique
    subject is held out exactly once. The configuration is therefore evaluated
    on the same complete set of subject-wise folds. After all configurations
    finish, one global configuration is selected from its mean LOSO metric.

    This is *not* nested cross-validation: the held-out LOSO results are used
    both to compare configurations and to report the selected configuration's
    cross-validation performance. This behavior is intentional for a practical
    flat LOSO hyperparameter search.

    Hyperparameter grid
    -------------------
    Scalar values may be supplied directly or as candidate lists/tuples. The
    Cartesian product is evaluated with a complete LOSO run per configuration.

    Sequence-valued encoder settings preserve one complete architecture before
    the Cartesian product is expanded. ``sequence_hyperparameter_depths``
    specifies the nesting depth of one value, resolving CNN1D/CNN2D ambiguity
    for keys such as ``kernel_sizes``. GCN ``gcn_units`` and temporal/spatial
    pooling schedules are preserved in the same way. One additional outer list
    level enumerates multiple architecture candidates.

    ``n_epochs`` and ``batch_size`` provide defaults and are overridden when
    ``hyperparameters`` contains ``epochs`` or ``batch_size``.

    Seeded validation and early stopping
    ------------------------------------
    When ``validation_subjects_per_fold`` is positive, that many subjects are
    sampled deterministically from each outer-training pool. They are excluded
    from gradient updates and passed to ``model.fit`` as ``validation_data``.
    The same fold-local validation subjects are reused for every hyperparameter
    configuration, while the LOSO test subject remains untouched. This adds no
    extra fits; it only changes each fit from train/test to train/validation/test.

    Selection
    ---------
    ``selection_level`` determines whether configurations are ranked using
    window- or trial-level scores. Hierarchical rank-4 inputs require trial-level
    selection. For binary tasks, ``selection_metric='f1'`` uses the binary
    convention: class 1 is positive. ``precision`` and ``recall`` follow the same
    convention. Explicit ``macro_*`` metrics and ``balanced_accuracy`` remain
    available for class-balanced diagnostics, while ``roc_auc`` uses the class-1
    probability.
    Classification metrics are maximized; probability loss and joint loss are minimized unless
    ``maximize_metric`` is explicitly supplied. Ties use lower between-subject
    standard deviation, lower mean log loss, then the earlier grid index.

    Returned results
    ----------------
    ``config_results`` contains per-fold and aggregate metrics for every
    configuration. Top-level prediction logs, user metrics, and fold metadata
    correspond only to the globally selected configuration. Selected fold
    metrics remain available through ``config_results[best_config_index]``.

    Concurrency
    -----------
    LOSO folds for one configuration run concurrently. The next configuration
    starts after the current configuration's folds complete. With one worker per
    GPU, this prevents multiple models from competing for the same GPU while
    bounding parent-process memory to approximately one configuration's logs.

    Smoke testing
    -------------
    ``max_folds`` deterministically limits every configuration to the first N
    sorted subjects. Leave it as ``None`` for complete LOSO evaluation.
    """
    extra_fit_kwargs = extra_fit_kwargs or {}

    if "validation_data" in extra_fit_kwargs:
        raise ValueError(
            "Do not pass a fixed validation_data array to loso_cv. It would not "
            "be reconstructed fold-locally and could create leakage."
        )

    if subject_id_array is None:
        raise ValueError("subject_id_array is required for LOSO CV.")
    if trial_id_array is None:
        raise ValueError(
            "trial_id_array is required for trial-level prediction and metrics. "
            "Pass one trial ID per sample, aligned with feature_array."
        )

    _validate_evaluation_level(evaluation_level, "evaluation_level")
    _validate_evaluation_level(selection_level, "selection_level")

    feature_array = np.asarray(feature_array)
    label_array = np.asarray(label_array)
    subject_id_array = np.asarray(subject_id_array)
    trial_id_array = np.asarray(trial_id_array)

    if feature_array.ndim not in {3, 4}:
        raise ValueError(
            "feature_array must be rank 3 for window samples or rank 4 for "
            f"grouped trial samples; got {feature_array.shape}."
        )
    if feature_array.ndim == 4:
        if selection_level != "trial":
            raise ValueError(
                "Grouped rank-4 trial inputs require selection_level='trial'."
            )
        if evaluation_level != "trial":
            raise ValueError(
                "Grouped rank-4 trial inputs require evaluation_level='trial'."
            )

    input_lengths = (
        len(feature_array),
        len(label_array),
        len(subject_id_array),
        len(trial_id_array),
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

    unique_subjects = np.sort(np.unique(subject_id_array))

    maximize_metric, decision_thresholds, test_subjects = _validate_loso_cv_arguments(
        subject_id_array=subject_id_array,
        trial_id_array=trial_id_array,
        selection_metric=selection_metric,
        selection_level=selection_level,
        maximize_metric=maximize_metric,
        metrics=metrics,
        n_prediction_latent_samples=n_prediction_latent_samples,
        ci_level=ci_level,
        n_jobs=n_jobs,
        cpus_per_worker=cpus_per_worker,
        validation_subjects_per_fold=validation_subjects_per_fold,
        alternate_subject_sets=alternate_subject_sets,
        use_mldg=use_mldg,
        mldg_meta_train_subjects=mldg_meta_train_subjects,
        mldg_meta_test_subjects=mldg_meta_test_subjects,
        mldg_samples_per_subject=mldg_samples_per_subject,
        mldg_seed=mldg_seed,
        validation_seed=validation_seed,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        early_stopping_monitor=early_stopping_monitor,
        early_stopping_mode=early_stopping_mode,
        prediction_diagnostics_every_n_epochs=prediction_diagnostics_every_n_epochs,
        prediction_diagnostics_max_samples=prediction_diagnostics_max_samples,
        prediction_diagnostics_threshold_tolerance=prediction_diagnostics_threshold_tolerance,
        decision_thresholds=decision_thresholds,
        threshold_selection_level=threshold_selection_level,
        max_folds=max_folds,
    )

    effective_hyperparameters = {
        "epochs": n_epochs,
        "batch_size": batch_size,
        **(hyperparameters or {}),
    }
    sequence_hyperparameter_depths = getattr(
        model_builder_function,
        "_sequence_hyperparameter_depths",
        None,
    )
    grid_configs = _expand_hyperparameter_grid(
        effective_hyperparameters,
        sequence_hyperparameter_depths=sequence_hyperparameter_depths,
    )
    _warn_if_joint_loss_weights_vary(grid_configs, selection_metric)
    if not grid_configs:
        raise ValueError("The hyperparameter grid produced no configurations.")

    total_folds = len(test_subjects)
    total_model_fits = len(grid_configs) * total_folds
    effective_n_jobs = min(n_jobs, total_folds)

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

    _print_loso_cv_summary(
        num_configs=len(grid_configs),
        total_folds=total_folds,
        num_subjects=len(unique_subjects),
        max_folds=max_folds,
        metrics=metrics,
        selection_level=selection_level,
        selection_metric=selection_metric,
        maximize_metric=maximize_metric,
        evaluation_level=evaluation_level,
        log_predictions=log_predictions,
        prediction_diagnostics=prediction_diagnostics,
        log_variational_intervals=log_variational_intervals,
        decision_thresholds=decision_thresholds,
        threshold_selection_level=threshold_selection_level,
        threshold_selection_metric=threshold_selection_metric,
        n_prediction_latent_samples=n_prediction_latent_samples,
        validation_subjects_per_fold=validation_subjects_per_fold,
        validation_seed=validation_seed,
        early_stopping_monitor=early_stopping_monitor,
        early_stopping_patience=early_stopping_patience,
        restore_best_weights=restore_best_weights,
        use_mldg=use_mldg,
        mldg_meta_train_subjects=mldg_meta_train_subjects,
        mldg_meta_test_subjects=mldg_meta_test_subjects,
        mldg_samples_per_subject=mldg_samples_per_subject,
        mldg_seed=mldg_seed,
        alternate_subject_sets=alternate_subject_sets,
        effective_n_jobs=effective_n_jobs,
        normalized_gpu_ids=normalized_gpu_ids,
    )

    tasks = [
        (fold_number, _python_scalar(test_subject))
        for fold_number, test_subject in enumerate(test_subjects, start=1)
    ]

    common_worker_state = {
        "total_folds": total_folds,
        "model_builder_function": model_builder_function,
        "feature_array": feature_array,
        "label_array": label_array,
        "subject_id_array": subject_id_array,
        "trial_id_array": trial_id_array,
        "batch_size": batch_size,
        "preprocessing_strategy": preprocessing_strategy,
        "evaluation_level": evaluation_level,
        "metrics": metrics,
        "log_predictions": log_predictions,
        "log_variational_intervals": log_variational_intervals,
        "n_prediction_latent_samples": n_prediction_latent_samples,
        "latent_sampling_seed": latent_sampling_seed,
        "n_uncertainty_samples": n_uncertainty_samples,
        "ci_level": ci_level,
        "validation_subjects_per_fold": validation_subjects_per_fold,
        "validation_seed": validation_seed,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "early_stopping_monitor": early_stopping_monitor,
        "early_stopping_mode": early_stopping_mode,
        "restore_best_weights": restore_best_weights,
        "prediction_diagnostics": bool(prediction_diagnostics),
        "prediction_diagnostics_every_n_epochs": int(
            prediction_diagnostics_every_n_epochs
        ),
        "prediction_diagnostics_max_samples": int(prediction_diagnostics_max_samples),
        "prediction_diagnostics_threshold_tolerance": float(
            prediction_diagnostics_threshold_tolerance
        ),
        "prediction_diagnostics_seed": prediction_diagnostics_seed,
        "decision_thresholds": decision_thresholds,
        "threshold_selection_metric": threshold_selection_metric,
        "threshold_selection_level": threshold_selection_level,
        "verbose": verbose,
        "extra_fit_kwargs": extra_fit_kwargs,
        "alternate_subject_sets": bool(alternate_subject_sets),
        "alternating_subject_seed": alternating_subject_seed,
        "use_mldg": bool(use_mldg),
        "mldg_meta_train_subjects": int(mldg_meta_train_subjects),
        "mldg_meta_test_subjects": int(mldg_meta_test_subjects),
        "mldg_samples_per_subject": int(mldg_samples_per_subject),
        "mldg_seed": mldg_seed,
    }

    config_results: list[dict] = []
    best_so_far_result: dict | None = None
    best_fold_outputs: list[dict] | None = None

    for config_index, config in enumerate(grid_configs):
        print("\n" + "#" * 80)
        print(
            f"Configuration {config_index + 1} / {len(grid_configs)} "
            f"({total_folds} LOSO fits)"
        )
        _print_config("Configuration:", config)

        worker_state = {
            **common_worker_state,
            "fixed_config": config,
        }

        if effective_n_jobs == 1 and normalized_gpu_ids is None:
            fold_outputs = [
                _run_loso_fold(
                    fold_number=fold_number,
                    test_subject=test_subject,
                    **worker_state,
                )
                for fold_number, test_subject in tasks
            ]
        else:
            fold_outputs = _run_spawned_fold_pool(
                worker_target=_loso_fold_process_main,
                worker_state=worker_state,
                tasks=tasks,
                n_workers=effective_n_jobs,
                gpu_ids=normalized_gpu_ids,
                cpus_per_worker=cpus_per_worker,
                worker_name_prefix=f"LOSOConfig{config_index + 1}Worker",
                worker_description=(f"LOSO-fold for configuration {config_index + 1}"),
            )

        fold_outputs.sort(key=lambda row: row["outer_fold_number"])
        config_result = _aggregate_loso_config_result(
            config_index=config_index,
            config=config,
            fold_outputs=fold_outputs,
            metrics=metrics,
            selection_metric=selection_metric,
            selection_level=selection_level,
        )
        config_results.append(config_result)

        if best_so_far_result is None or _loso_config_sort_key(
            config_result=config_result,
            selection_metric=selection_metric,
            selection_level=selection_level,
            maximize_metric=bool(maximize_metric),
        ) < _loso_config_sort_key(
            config_result=best_so_far_result,
            selection_metric=selection_metric,
            selection_level=selection_level,
            maximize_metric=bool(maximize_metric),
        ):
            best_so_far_result = config_result
            best_fold_outputs = fold_outputs

        print(
            f"\nConfiguration {config_index + 1} complete: "
            f"mean {selection_level}_{selection_metric}="
            f"{config_result['selection_score']:.6f} ± "
            f"{config_result['selection_score_std']:.6f}",
            flush=True,
        )

    best_config_index = _choose_best_loso_config_index(
        config_results=config_results,
        selection_metric=selection_metric,
        selection_level=selection_level,
        maximize_metric=bool(maximize_metric),
    )
    best_config_result = config_results[best_config_index]
    best_config = dict(best_config_result["config"])

    if (
        best_so_far_result is None
        or best_fold_outputs is None
        or int(best_so_far_result["config_index"]) != best_config_index
    ):
        raise RuntimeError(
            "Internal LOSO grid-search error: selected configuration logs "
            "were not retained correctly."
        )

    # Surface one canonical copy of each selected-configuration artifact.
    results = {
        "cv_strategy": "flat_loso_hyperparameter_search",
        "hyperparameter_search": True,
        "n_configs": int(len(grid_configs)),
        "n_subjects": int(len(unique_subjects)),
        "n_evaluated_folds_per_config": int(total_folds),
        "n_total_loso_fits": int(total_model_fits),
        "max_folds": max_folds,
        "selection_metric": selection_metric,
        "selection_level": selection_level,
        "evaluation_level": evaluation_level,
        "maximize_metric": bool(maximize_metric),
        "selection_score": float(best_config_result["selection_score"]),
        "selection_score_std": float(best_config_result["selection_score_std"]),
        "n_prediction_latent_samples": int(n_prediction_latent_samples),
        "latent_sampling_seed": latent_sampling_seed,
        "validation_subjects_per_fold": int(validation_subjects_per_fold),
        "validation_seed": validation_seed,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": float(early_stopping_min_delta),
        "early_stopping_monitor": early_stopping_monitor,
        "early_stopping_mode": early_stopping_mode,
        "restore_best_weights": bool(restore_best_weights),
        "use_mldg": bool(use_mldg),
        "mldg_meta_train_subjects": int(mldg_meta_train_subjects) if use_mldg else 0,
        "mldg_meta_test_subjects": int(mldg_meta_test_subjects) if use_mldg else 0,
        "mldg_samples_per_subject": int(mldg_samples_per_subject) if use_mldg else 0,
        "mldg_seed": mldg_seed if use_mldg else None,
        "config_results": config_results,
        "best_config_index": int(best_config_index),
        "best_config": best_config,
        "user_metrics": [],
        "fold_results": [],
    }
    if log_predictions:
        if feature_array.ndim == 3:
            results["window_prediction_log"] = []
        results["trial_prediction_log"] = []
    if log_variational_intervals:
        if feature_array.ndim == 3:
            results["window_variational_interval_log"] = []
        results["trial_variational_interval_log"] = []
    if prediction_diagnostics:
        results["prediction_diagnostics_log"] = []

    for fold_output in best_fold_outputs:
        results["user_metrics"].extend(fold_output["user_metrics"])
        if log_predictions:
            if feature_array.ndim == 3:
                results["window_prediction_log"].extend(
                    fold_output["window_prediction_log"]
                )
            results["trial_prediction_log"].extend(fold_output["trial_prediction_log"])
        if log_variational_intervals:
            if feature_array.ndim == 3:
                results["window_variational_interval_log"].extend(
                    fold_output["window_variational_interval_log"]
                )
            results["trial_variational_interval_log"].extend(
                fold_output["trial_variational_interval_log"]
            )
        if prediction_diagnostics:
            results["prediction_diagnostics_log"].extend(
                fold_output.get("prediction_diagnostics_log", [])
            )
        results["fold_results"].append(dict(fold_output["fold_record"]))

    print("\nFlat LOSO hyperparameter search complete")
    print("=" * 80)
    print(
        f"Selected configuration {best_config_index + 1} / "
        f"{len(grid_configs)} using {selection_level}-level "
        f"{selection_metric}."
    )
    _print_config("Best configuration:", best_config)
    print(
        f"Selection score: {best_config_result['selection_score']:.6f} ± "
        f"{best_config_result['selection_score_std']:.6f}"
    )
    print("Selected configuration primary mean scores:")
    print(
        pformat(
            best_config_result[f"{evaluation_level}_mean_scores"],
            indent=4,
            width=120,
            sort_dicts=False,
        )
    )
    print("Selected configuration primary score standard deviations:")
    print(
        pformat(
            best_config_result[f"{evaluation_level}_std_scores"],
            indent=4,
            width=120,
            sort_dicts=False,
        )
    )
    print("Selected configuration window-level mean scores:")
    print(
        pformat(
            best_config_result["window_mean_scores"],
            indent=4,
            width=120,
            sort_dicts=False,
        )
    )
    print("Selected configuration trial-level mean scores:")
    print(
        pformat(
            best_config_result["trial_mean_scores"],
            indent=4,
            width=120,
            sort_dicts=False,
        )
    )

    return results


def fixed_loso_cv(
    model_builder_function: Callable[..., tf.keras.Model],
    feature_array: np.ndarray,
    label_array: np.ndarray,
    subject_id_array: np.ndarray,
    trial_id_array: np.ndarray,
    fixed_config: dict,
    n_epochs: int,
    batch_size: int,
    *,
    preprocessing_strategy: Callable | None = None,
    evaluation_level: Literal["window", "trial"] = "trial",
    selection_metric: str = "balanced_accuracy",
    selection_level: Literal["window", "trial"] = "trial",
    maximize_metric: bool | None = None,
    metrics: list[str] | tuple[str, ...] = (
        "accuracy",
        "f1",
        "precision",
        "recall",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "balanced_accuracy",
    ),
    log_predictions: bool = True,
    log_variational_intervals: bool = False,
    n_prediction_latent_samples: int = 0,
    latent_sampling_seed: int | None = None,
    n_uncertainty_samples: int = 30,
    ci_level: float = 0.95,
    decision_threshold: float = 0.5,
    prediction_diagnostics: bool = False,
    prediction_diagnostics_every_n_epochs: int = 1,
    prediction_diagnostics_max_samples: int = 256,
    prediction_diagnostics_threshold_tolerance: float = 0.01,
    prediction_diagnostics_seed: int | None = 42,
    verbose: int = 0,
    extra_fit_kwargs: dict | None = None,
    n_jobs: int = 1,
    gpu_ids: list[int] | tuple[int, ...] | None = None,
    cpus_per_worker: int | None = None,
    max_folds: int | None = None,
    alternate_subject_sets: bool = False,
    alternating_subject_seed: int | None = 42,
    use_mldg: bool = False,
    mldg_meta_train_subjects: int = 6,
    mldg_meta_test_subjects: int = 2,
    mldg_samples_per_subject: int = 4,
    mldg_seed: int | None = 42,
) -> dict:
    """Evaluate one fixed configuration with strict LOSOCV and no validation.

    Every fold trains for exactly ``n_epochs`` on all non-test subjects. No
    validation subjects are removed, no validation data are passed to Keras,
    no early-stopping callback is installed, and the supplied decision
    threshold is applied unchanged to every held-out subject.

    This is intended as a post-selection diagnostic after another CV run has
    already chosen the hyperparameters, epoch count, and threshold. It does not
    perform another hyperparameter or threshold search.
    """
    if n_epochs < 1:
        raise ValueError("n_epochs must be at least 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    decision_threshold = float(decision_threshold)
    if not 0.0 < decision_threshold < 1.0:
        raise ValueError("decision_threshold must be strictly between 0 and 1.")

    model_config = dict(fixed_config)
    # The explicit post-selection values must override anything retained from
    # the original search result.
    model_config.pop("epochs", None)
    model_config.pop("batch_size", None)

    print(
        "\nFixed-config no-validation LOSOCV — "
        f"epochs={int(n_epochs)}, batch_size={int(batch_size)}, "
        f"decision_threshold={decision_threshold:.4f}",
        flush=True,
    )

    results = loso_cv(
        model_builder_function=model_builder_function,
        feature_array=feature_array,
        label_array=label_array,
        subject_id_array=subject_id_array,
        trial_id_array=trial_id_array,
        n_epochs=int(n_epochs),
        batch_size=int(batch_size),
        hyperparameters=model_config,
        preprocessing_strategy=preprocessing_strategy,
        evaluation_level=evaluation_level,
        selection_metric=selection_metric,
        selection_level=selection_level,
        maximize_metric=maximize_metric,
        metrics=metrics,
        log_predictions=log_predictions,
        log_variational_intervals=log_variational_intervals,
        n_prediction_latent_samples=n_prediction_latent_samples,
        latent_sampling_seed=latent_sampling_seed,
        n_uncertainty_samples=n_uncertainty_samples,
        ci_level=ci_level,
        validation_subjects_per_fold=0,
        validation_seed=None,
        early_stopping_patience=None,
        early_stopping_min_delta=0.0,
        early_stopping_monitor="loss",
        early_stopping_mode="min",
        restore_best_weights=False,
        prediction_diagnostics=prediction_diagnostics,
        prediction_diagnostics_every_n_epochs=(prediction_diagnostics_every_n_epochs),
        prediction_diagnostics_max_samples=prediction_diagnostics_max_samples,
        prediction_diagnostics_threshold_tolerance=(
            prediction_diagnostics_threshold_tolerance
        ),
        prediction_diagnostics_seed=prediction_diagnostics_seed,
        decision_thresholds=(decision_threshold,),
        threshold_selection_metric="balanced_accuracy",
        threshold_selection_level=selection_level,
        verbose=verbose,
        extra_fit_kwargs=extra_fit_kwargs,
        n_jobs=n_jobs,
        gpu_ids=gpu_ids,
        cpus_per_worker=cpus_per_worker,
        max_folds=max_folds,
        alternate_subject_sets=alternate_subject_sets,
        alternating_subject_seed=alternating_subject_seed,
        use_mldg=use_mldg,
        mldg_meta_train_subjects=mldg_meta_train_subjects,
        mldg_meta_test_subjects=mldg_meta_test_subjects,
        mldg_samples_per_subject=mldg_samples_per_subject,
        mldg_seed=mldg_seed,
    )

    if int(results.get("n_configs", 0)) != 1:
        raise RuntimeError(
            "fixed_loso_cv expected exactly one configuration, but loso_cv "
            f"reported {results.get('n_configs')}."
        )

    results.update(
        {
            "cv_strategy": "fixed_loso_no_validation",
            "hyperparameter_search": False,
            "post_selection_diagnostic": True,
            "fixed_epochs": int(n_epochs),
            "fixed_batch_size": int(batch_size),
            "fixed_decision_threshold": decision_threshold,
            "validation_subjects_per_fold": 0,
            "early_stopping_patience": None,
            "restore_best_weights": False,
        }
    )
    return results
