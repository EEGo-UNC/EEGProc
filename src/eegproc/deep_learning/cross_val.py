import itertools
from itertools import combinations
from typing import Callable
from __future__ import annotations

import itertools

import numpy as np
import tensorflow as tf
import numpy as np
from typing import Callable, Literal

_FIT_RESERVED_KEYS = frozenset({"epochs", "batch_size"})


def _expand_hyperparameter_grid(hp: dict | None) -> list[dict]:
    """Expand a (possibly empty) hyperparameter dict to a list of configs.

    Values that are lists or tuples are treated as a grid axis; scalars are
    held constant. The Cartesian product is returned. ``None`` or empty
    dicts produce ``[{}]`` so the caller can always iterate at least once.
    """
    if not hp:
        return [{}]
    keys = list(hp.keys())
    values = [v if isinstance(v, (list, tuple)) else [v] for v in hp.values()]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _split_config(config: dict) -> tuple[dict, dict]:
    """Split a flat config into (model_builder_kwargs, fit_kwargs).

    Keys in ``_FIT_RESERVED_KEYS`` go to ``model.fit``; the rest are
    forwarded to ``model_builder_function`` via ``**``.
    """
    model_hp = {k: v for k, v in config.items() if k not in _FIT_RESERVED_KEYS}
    fit_hp = {k: v for k, v in config.items() if k in _FIT_RESERVED_KEYS}
    return model_hp, fit_hp


def _collect_metric_names(trained_model: tf.keras.Model) -> list[str]:
    """Return the metric names from a model after evaluation."""
    return trained_model.metrics_names


def _average_fold_scores(
    all_fold_results: list[dict],
    metric_names: list[str],
) -> tuple[dict, dict]:
    """Compute mean and std of each metric across all folds.

    Returns two dicts: mean_scores and std_scores.
    """
    mean_scores = {
        metric_name: float(np.mean([fold[metric_name] for fold in all_fold_results]))
        for metric_name in metric_names
    }
    std_scores = {
        metric_name: float(np.std([fold[metric_name] for fold in all_fold_results]))
        for metric_name in metric_names
    }
    return mean_scores, std_scores


def _print_fold_header(fold_number: int, total_folds: int, description: str) -> None:
    """Print a readable progress line for the current fold."""
    print(f"  [Fold {fold_number:>3} / {total_folds}]  {description}")


def nested_lnso_cv(
    model_builder_function: Callable[..., tf.keras.Model],
    feature_array: np.ndarray,
    label_array: np.ndarray,
    subject_id_array: np.ndarray,
    n_outer_subjects_to_leave_out: int = 1,
    n_inner_subjects_to_leave_out: int = 1,
    n_epochs: int = 50,
    batch_size: int = 32,
    hyperparameters: dict | None = None,
    preprocessing_strategy: Callable | None = None,
    selection_metric: str = "loss",
    maximize_metric: bool | None = None,
    verbose: int = 0,
    extra_fit_kwargs: dict | None = None,
) -> dict:
    """
    Run nested Leave-N-Subjects-Out cross-validation for subject-level evaluation
    and hyperparameter selection.

    This function implements a nested cross-validation procedure where the outer
    loop estimates generalization to unseen subjects and the inner loop selects the
    best hyperparameter configuration using only the outer-training subjects.

    The outer loop leaves out `n_outer_subjects_to_leave_out` subjects at a time.
    These left-out subjects form the outer test set and are not used for fitting,
    validation, early stopping, preprocessing fit, or hyperparameter selection.

    For each outer fold, the remaining subjects form the outer-training pool. Inside
    that pool, an inner Leave-N-Subjects-Out CV is run by leaving out
    `n_inner_subjects_to_leave_out` subjects at a time. For every hyperparameter
    configuration, a model is trained on the inner-training subjects and evaluated
    on the inner-validation subjects. The mean inner-validation score is then used
    to choose the best hyperparameter configuration for that outer fold.

    After the best configuration is selected, a fresh model is built and retrained
    on all outer-training subjects. It is then evaluated once on the outer test
    subjects. The outer test score is the unbiased estimate of performance for that
    fold.

    Important behavior:
        - The outer test subjects are never used during inner CV or model selection.
        - The inner validation subjects are passed to `model.fit` as
        `validation_data`, but this function does not automatically create an
        EarlyStopping callback.
        - Early stopping is only used if the caller explicitly passes callbacks
        through `extra_fit_kwargs`.
        - During final outer retraining, no validation_data is passed by this
        function, so callbacks monitoring `val_loss` will not work unless the
        final training logic is modified.
        - Hyperparameters named `"epochs"` and `"batch_size"` are treated as
        `model.fit` arguments. All other hyperparameters are passed to
        `model_builder_function`.
        - If `preprocessing_strategy` is provided, it is applied separately inside
        each fold using only the training partition and the corresponding
        evaluation partition. This prevents preprocessing leakage across
        train/validation/test subject boundaries.

    Parameters
    ----------
    model_builder_function : Callable[..., tf.keras.Model]
        Function that builds and returns a compiled Keras model. It should accept
        model hyperparameters as keyword arguments and return a fresh model instance.
        A new model is built for every inner-fold/configuration pair and for every
        final outer-fold retraining.

    feature_array : np.ndarray
        Input feature array. The first dimension must correspond to samples/windows.

    label_array : np.ndarray
        Target labels. The first dimension must correspond to the same samples/windows
        as `feature_array`.

    subject_id_array : np.ndarray
        Array of subject identifiers, one per sample/window. This is required because
        all train, validation, and test splits are made at the subject level.

    n_outer_subjects_to_leave_out : int, default=1
        Number of subjects to leave out in each outer fold. Setting this to 1 gives
        nested LOSO CV. Larger values give nested LNSO CV.

    n_inner_subjects_to_leave_out : int, default=1
        Number of subjects to leave out in each inner fold for validation during
        hyperparameter selection.

    n_epochs : int, default=50
        Default number of training epochs. This is inserted into the hyperparameter
        grid as `"epochs"` unless overridden by `hyperparameters`.

    batch_size : int, default=32
        Default batch size. This is inserted into the hyperparameter grid as
        `"batch_size"` unless overridden by `hyperparameters`.

    hyperparameters : dict | None, default=None
        Hyperparameter grid. Values that are lists or tuples are treated as grid
        axes and expanded using a Cartesian product. Scalar values are treated as
        fixed values.

        Example:
            {
                "learning_rate": [1e-3, 1e-4],
                "dropout": [0.2, 0.5],
                "epochs": 30,
                "batch_size": [16, 32],
            }

        Keys `"epochs"` and `"batch_size"` are passed to `model.fit`. All other keys
        are passed to `model_builder_function`.

    preprocessing_strategy : Callable | None, default=None
        Optional fold-local preprocessing function. If provided, it must have the
        form:

            preprocessing_strategy(X_train, y_train, X_eval, y_eval)

        It may return either:

            X_train_processed, X_eval_processed

        or:

            X_train_processed, y_train_processed, X_eval_processed, y_eval_processed

        The strategy is applied separately for each inner train/validation split and
        each outer train/test split. Any fitting inside the preprocessing function
        should be done using only `X_train`/`y_train`.

    selection_metric : str, default="loss"
        Metric used to select the best hyperparameter configuration from the inner
        CV results. Must match one of the names returned by `model.metrics_names`
        after evaluation, such as `"loss"` or `"accuracy"`.

    maximize_metric : bool | None, default=None
        Whether larger values of `selection_metric` are better. If False, the
        configuration with the lowest mean inner score is selected. If True, the
        configuration with the highest mean inner score is selected.

        In the current implementation, None behaves like False, so loss-like metrics
        are minimized by default.

    verbose : int, default=0
        Verbosity level passed to `model.fit`.

    extra_fit_kwargs : dict | None, default=None
        Additional keyword arguments passed to `model.fit`.

        Do not include `"validation_data"` here. The function creates validation data
        from the inner folds and raises an error if `"validation_data"` is supplied.

        This can be used to pass callbacks, class weights, sample weights, or other
        Keras fit options. For example, EarlyStopping can be enabled by passing a
        callback here.

    Returns
    -------
    dict
        Dictionary containing:

        outer_fold_results : list[dict]
            One dictionary per outer fold, including the left-out test subjects,
            number of train/test windows, best hyperparameter configuration,
            detailed inner-fold results, inner mean/std scores, and final outer test
            scores.

        mean_scores : dict
            Mean of each test metric across outer folds.

        std_scores : dict
            Standard deviation of each test metric across outer folds.

    Notes
    -----
    This function is appropriate when the goal is to estimate how well a model and
    hyperparameter selection procedure generalize to unseen subjects. The inner loop
    chooses hyperparameters; the outer loop estimates performance.

    The reported final performance should come from the outer test scores, not from
    the inner validation scores.
    """

    extra_fit_kwargs = extra_fit_kwargs or {}

    if "validation_data" in extra_fit_kwargs:
        raise ValueError(
            "Do not pass validation_data in extra_fit_kwargs. "
            "nested_lnso_cv creates validation_data from the inner folds."
        )

    if subject_id_array is None:
        raise ValueError("subject_id_array is required for nested LNSO CV.")

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

    outer_subject_splits = list(
        combinations(unique_subjects, n_outer_subjects_to_leave_out)
    )

    outer_fold_results: list[dict] = []
    outer_test_scores: list[dict] = []
    metric_names: list[str] = []

    print(
        f"\nNested LNSO CV — {len(outer_subject_splits)} outer folds, "
        f"{len(grid_configs)} hyperparameter config"
        f"{'s' if len(grid_configs) != 1 else ''}\n"
    )

    for outer_fold_number, outer_test_subjects in enumerate(
        outer_subject_splits,
        start=1,
    ):
        outer_test_subjects = np.array(outer_test_subjects)

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
            len(outer_subject_splits),
            f"outer test subjects={outer_test_subjects.tolist()} "
            f"(outer_train={len(outer_train_indices)}, "
            f"outer_test={len(outer_test_indices)} windows)",
        )

        inner_scores_by_config: list[list[dict]] = [[] for _ in grid_configs]
        inner_fold_results: list[dict] = []

        for inner_fold_number, inner_val_subjects in enumerate(
            inner_subject_splits,
            start=1,
        ):
            inner_val_subjects = np.array(inner_val_subjects)

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
            config_results_this_inner_fold: list[dict] = []

            for config_index, config in enumerate(grid_configs):
                model_hp, fit_hp = _split_config(config)

                model = model_builder_function(**model_hp)

                fit_kwargs = dict(fit_hp)
                fit_kwargs["validation_data"] = (X_inner_val, y_inner_val)

                model.fit(
                    X_inner_train,
                    y_inner_train,
                    verbose=verbose,
                    **fit_kwargs,
                    **extra_fit_kwargs,
                )

                val_score_values = model.evaluate(
                    X_inner_val,
                    y_inner_val,
                    verbose=0,
                )

                if not isinstance(val_score_values, (list, tuple)):
                    val_score_values = [val_score_values]

                metric_names = _collect_metric_names(model)
                val_scores = dict(zip(metric_names, val_score_values))

                if selection_metric not in val_scores:
                    raise ValueError(
                        f"selection_metric='{selection_metric}' was not found. "
                        f"Available metrics: {list(val_scores.keys())}"
                    )

                config_result = {
                    "config": dict(config),
                    **val_scores,
                }

                config_results_this_inner_fold.append(config_result)
                inner_scores_by_config[config_index].append(val_scores)

            inner_fold_results.append(
                {
                    "inner_fold_number": inner_fold_number,
                    "left_out_subjects": inner_val_subjects.tolist(),
                    "n_train_windows": int(len(inner_train_indices)),
                    "n_val_windows": int(len(inner_val_indices)),
                    "configs": config_results_this_inner_fold,
                }
            )

        inner_mean_scores: list[dict] = []
        inner_std_scores: list[dict] = []

        for config_index, config in enumerate(grid_configs):
            mean_scores_for_config, std_scores_for_config = _average_fold_scores(
                inner_scores_by_config[config_index],
                metric_names,
            )

            inner_mean_scores.append(
                {
                    "config": dict(config),
                    **mean_scores_for_config,
                }
            )

            inner_std_scores.append(
                {
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
            f"    Best config from inner CV: {best_config} "
            f"({selection_metric}="
            f"{inner_mean_scores[best_config_index][selection_metric]:.4f})"
        )

        X_outer_train = feature_array[outer_train_indices]
        y_outer_train = label_array[outer_train_indices]
        X_outer_test = feature_array[outer_test_indices]
        y_outer_test = label_array[outer_test_indices]

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

        model_hp, fit_hp = _split_config(best_config)

        final_model = model_builder_function(**model_hp)

        final_model.fit(
            X_outer_train,
            y_outer_train,
            verbose=verbose,
            **fit_hp,
            **extra_fit_kwargs,
        )

        test_score_values = final_model.evaluate(
            X_outer_test,
            y_outer_test,
            verbose=0,
        )

        if not isinstance(test_score_values, (list, tuple)):
            test_score_values = [test_score_values]

        metric_names = _collect_metric_names(final_model)
        test_scores = dict(zip(metric_names, test_score_values))
        outer_test_scores.append(test_scores)

        score_summary = "  ".join(
            f"{name}={test_scores[name]:.4f}" for name in metric_names
        )
        print(f"    Outer test scores: {score_summary}")

        outer_fold_results.append(
            {
                "outer_fold_number": outer_fold_number,
                "left_out_subjects": outer_test_subjects.tolist(),
                "n_outer_train_windows": int(len(outer_train_indices)),
                "n_outer_test_windows": int(len(outer_test_indices)),
                "best_config": dict(best_config),
                "inner_fold_results": inner_fold_results,
                "inner_mean_scores": inner_mean_scores,
                "inner_std_scores": inner_std_scores,
                **test_scores,
            }
        )

    mean_scores, std_scores = _average_fold_scores(
        outer_test_scores,
        metric_names,
    )

    print(f"\nNested LNSO CV complete — mean outer scores: {mean_scores}\n")

    return {
        "outer_fold_results": outer_fold_results,
        "mean_scores": mean_scores,
        "std_scores": std_scores,
    }


def _apply_preprocessing_strategy(
    preprocessing_strategy: Callable | None,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
):
    """Apply optional preprocessing inside a CV fold."""
    if preprocessing_strategy is None:
        return X_train, y_train, X_eval, y_eval

    result = preprocessing_strategy(
        X_train,
        y_train,
        X_eval,
        y_eval,
        train_indices,
        eval_indices,
    )

    if not isinstance(result, tuple):
        raise ValueError(
            "preprocessing_strategy must return a tuple with either 2 or 4 values."
        )

    if len(result) == 2:
        X_train_processed, X_eval_processed = result
        return X_train_processed, y_train, X_eval_processed, y_eval

    if len(result) == 4:
        return result

    raise ValueError(
        "preprocessing_strategy must return either "
        "(X_train, X_eval) or (X_train, y_train, X_eval, y_eval)."
    )


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
