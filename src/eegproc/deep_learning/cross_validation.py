"""
cross_validation.py — EEG cross-validation training pipelines.

EEG signals are highly person-specific, so a model that works on one person
may not generalize to another. The cross-validation strategies here are the
standard ways researchers test whether a model truly generalizes.

Four strategies are implemented:

    LNSO CV         Leave One Subject Out
                    Hold out one subject at a time as the test set.
                    This is the gold standard for testing cross-subject
                    generalization. Number of folds = number of subjects.

    LOO CV          Leave One Out (one sample/window at a time)
                    Hold out a single sample window at a time. Very
                    fine-grained, produces many folds, and tests
                    within-dataset generalization. Best used on smaller
                    datasets where LOSO would leave too little training data.

    LKOCV           Leave K Subjects Out
                    Generalization of LOSO: hold out K subjects at once
                    instead of just one. Useful when you want larger test
                    sets per fold, or want to simulate deployment on a
                    group of unseen people.

    Nested LNSO CV  Nested Leave N Subjects Out
                    Two-loop cross-validation. The outer loop holds out
                    N_outer subjects for testing (like LOSO). Within each
                    outer fold, an inner loop holds out N_inner subjects
                    from the remaining training subjects to form a
                    validation set for early stopping or model selection.
                    This prevents "peeking" at the test set during
                    hyperparameter/threshold decisions.

All four strategies share the same call signature so they can be swapped
without changing the rest of your code. The master function
``run_cross_validation`` lets you pick any strategy by name and run
everything in one call.

Typical usage
-------------
1. Feature-extract your EEG data into a 3-D array X of shape
   (n_windows, timesteps, n_features).
2. Build a label array y of shape (n_windows,) and a subject-ID array
   subject_ids of shape (n_windows,).
3. Define a model-builder function that returns a freshly compiled Keras
   model (called once per fold so weights never leak between folds).
4. Call run_cross_validation — done.

Example::

    import numpy as np
    from eegproc.supervised import bilstm_classifier
    from eegproc.unsupervised.cross_validation import run_cross_validation

    # --- Dummy data (replace with your real arrays) ---
    X            = np.random.randn(200, 128, 84).astype("float32")
    y            = np.random.randint(0, 3, size=200)
    subject_ids  = np.repeat(np.arange(10), 20)   # 10 subjects, 20 windows each

    def build_my_model():
        # bilstm_classifier returns an already-compiled model.
        return bilstm_classifier(
            timesteps=128, n_features=84,
            lstm_units=64, n_classes=3,
        )

    results = run_cross_validation(
        cv_strategy="loso",
        model_builder_function=build_my_model,
        feature_array=X,
        label_array=y,
        subject_id_array=subject_ids,
        n_epochs=30,
        batch_size=16,
    )

    print(results["mean_scores"])   # average accuracy and loss across folds
"""

from __future__ import annotations

import itertools

import numpy as np
import tensorflow as tf
from sklearn.model_selection import (
    LeaveOneGroupOut,  # used for LOSO  (leave one subject out)
    LeavePGroupsOut,  # used for LKOCV (leave k subjects out)
    LeaveOneOut,  # used for LOO   (leave one sample out)
)
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# LOSO CV — Leave One Subject Out Cross Validation
# ---------------------------------------------------------------------------


def loso_cv(
    model_builder_function: Callable[..., tf.keras.Model],
    feature_array: np.ndarray,
    label_array: np.ndarray,
    subject_id_array: np.ndarray,
    hyperparameters: dict | None = None,
    validation_n_users: int | None = None,
    verbose: int = 0,
    extra_fit_kwargs: dict | None = None,
) -> dict:
    """Leave One Subject Out Cross Validation (LOSO CV).

    In each fold one subject is completely held out as the test set and the
    model is retrained from scratch on the remaining subjects. The held-out
    subject's data is never seen during training, so this strategy tests
    cross-subject generalization. Note that LNSO and Nested LNSO CV also
    test cross-subject generalization and are typically more robust in
    low-subject regimes — pick LOSO when subject count is moderate and you
    want one fold per subject.

    Number of folds equals the number of unique subjects.

    Parameters
    ----------
    model_builder_function : callable (**hparams) -> tf.keras.Model
        Called once at the start of every fold so weights do not leak
        between folds. Any keys in ``hyperparameters`` that are not in
        ``{"epochs", "batch_size"}`` are forwarded as ``**hparams``. The
        function may accept ``**hparams`` to be tolerant of unknown keys.

        Example::

            from eegproc.supervised import lstm_classifier

            def build_model(**hparams):
                return lstm_classifier(
                    timesteps=128, n_features=84, n_classes=3,
                    lstm_units=hparams.get("lstm_units", 64),
                    dropout=hparams.get("dropout", 0.1),
                )

    feature_array : np.ndarray, shape (n_windows, timesteps, n_features)
        Preprocessed EEG feature windows. Each row is one sliding window.
    label_array : np.ndarray, shape (n_windows,)
        Integer class label for each window.
    subject_id_array : np.ndarray, shape (n_windows,)
        Subject identifier for each window. Windows from the same recording
        session/person should share the same identifier.
    hyperparameters : dict, optional
        Either ``{key: scalar, ...}`` (single config) or ``{key: list, ...}``
        (grid search — Cartesian product across all list-valued keys).
        Two reserved keys are passed to ``model.fit`` instead of the model
        builder: ``"epochs"`` and ``"batch_size"``. Everything else is
        forwarded to ``model_builder_function(**model_hparams)``. ``None``
        is equivalent to ``{}``.

        Optuna example::

            def objective(trial):
                hp = {
                    "lstm_units": trial.suggest_int("lstm_units", 16, 128),
                    "epochs":     trial.suggest_int("epochs", 5, 50),
                    "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
                }
                results = loso_cv(build_model, X, y, subject_ids, hyperparameters=hp)
                return results["mean_scores"]["accuracy"]

    validation_n_users : int, optional
        If provided, hold out this many subjects from each fold's training
        pool as a validation set and pass them to ``model.fit`` via
        ``validation_data``. The held-out *test* subject is never used for
        validation, so there is no leakage. If ``None`` (default), no
        validation set is used.
    verbose : int, optional
        Keras verbosity (0 silent, 1 progress bar, 2 one line per epoch).
    extra_fit_kwargs : dict, optional
        Extra keyword arguments forwarded to ``model.fit`` (e.g. callbacks).

    Returns
    -------
    dict
        ``"fold_results"`` — list with one entry per fold (length =
        n_unique_subjects). Each entry has:

            * ``"fold_number"`` : int (1-indexed)
            * ``"left_out_subject"`` : the subject ID held out for testing
            * ``"n_train_windows"`` / ``"n_val_windows"`` /
              ``"n_test_windows"`` : window counts

        **Single-config case** — the fold entry also has ``"config"``
        (the applied hparams) and one key per Keras metric at the top
        level (e.g. ``"loss"``, ``"accuracy"``). ``"mean_scores"`` and
        ``"std_scores"`` are plain ``dict[metric_name, float]``.

        **Grid-search case** (any list-valued entry in ``hyperparameters``)
        — the fold entry has a ``"configs"`` list, one entry per grid
        point: ``{"config": {...}, "loss": ..., "accuracy": ...}``.
        ``"mean_scores"`` and ``"std_scores"`` are ``list[dict]``, one
        entry per grid point.

    Raises
    ------
    ValueError
        If ``validation_n_users`` is negative or would leave no training
        data in some fold.
    """
    extra_fit_kwargs = extra_fit_kwargs or {}

    if validation_n_users is not None and validation_n_users < 0:
        raise ValueError(
            f"validation_n_users must be >= 0 or None, got {validation_n_users}."
        )

    grid_configs = _expand_hyperparameter_grid(hyperparameters)
    is_grid = len(grid_configs) > 1

    # sklearn's LeaveOneGroupOut treats subject IDs as "groups" and yields
    # splits that hold out one group (subject) at a time.
    leave_one_subject_out_splitter = LeaveOneGroupOut()
    total_number_of_folds = leave_one_subject_out_splitter.get_n_splits(
        feature_array, label_array, subject_id_array
    )

    all_fold_results: list[dict] = []
    # Per-config accumulators (only used in grid mode but cheap in both).
    per_config_fold_scores: list[list[dict]] = [[] for _ in grid_configs]
    metric_names: list[str] = []

    print(
        f"\nLOSO CV — {total_number_of_folds} folds "
        f"({len(np.unique(subject_id_array))} unique subjects, "
        f"{len(grid_configs)} hparam config{'s' if is_grid else ''})\n"
    )

    for fold_number, (pool_indices, test_indices) in enumerate(
        leave_one_subject_out_splitter.split(
            feature_array, label_array, subject_id_array
        ),
        start=1,
    ):
        left_out_subject_id = subject_id_array[test_indices[0]]

        # Partition the training pool into train + (optional) validation.
        pool_subject_ids = subject_id_array[pool_indices]
        unique_pool_subjects = np.sort(np.unique(pool_subject_ids))

        if validation_n_users:
            if validation_n_users >= len(unique_pool_subjects):
                raise ValueError(
                    f"validation_n_users ({validation_n_users}) >= "
                    f"unique training subjects in fold "
                    f"({len(unique_pool_subjects)}) — would leave no "
                    f"training data."
                )
            # Deterministic: first N sorted unique subjects → validation.
            val_subject_set = set(unique_pool_subjects[:validation_n_users].tolist())
            val_mask = np.array([s in val_subject_set for s in pool_subject_ids])
            train_indices = pool_indices[~val_mask]
            val_indices = pool_indices[val_mask]
        else:
            train_indices = pool_indices
            val_indices = np.array([], dtype=int)

        X_train = feature_array[train_indices]
        y_train = label_array[train_indices]
        X_test = feature_array[test_indices]
        y_test = label_array[test_indices]
        if len(val_indices) > 0:
            validation_data = (feature_array[val_indices], label_array[val_indices])
        else:
            validation_data = None

        _print_fold_header(
            fold_number,
            total_number_of_folds,
            f"testing on subject '{left_out_subject_id}'  "
            f"(train={len(train_indices)}, "
            f"val={len(val_indices)}, "
            f"test={len(test_indices)} windows)",
        )

        per_config_results_this_fold: list[dict] = []

        for config_index, config in enumerate(grid_configs):
            model_hp, fit_hp = _split_config(config)

            # Build a fresh model per config per fold — no weight leakage.
            model = model_builder_function(**model_hp)

            fit_kwargs = dict(fit_hp)
            if validation_data is not None:
                fit_kwargs.setdefault("validation_data", validation_data)

            model.fit(
                X_train,
                y_train,
                verbose=verbose,
                **fit_kwargs,
                **extra_fit_kwargs,
            )

            test_score_values = model.evaluate(X_test, y_test, verbose=0)
            metric_names = _collect_metric_names(model)
            metric_scores = dict(zip(metric_names, test_score_values))

            config_result = {"config": dict(config), **metric_scores}
            per_config_results_this_fold.append(config_result)
            per_config_fold_scores[config_index].append(metric_scores)

            score_summary = "  ".join(
                f"{name}={metric_scores[name]:.4f}" for name in metric_names
            )
            config_tag = (
                f" [config {config_index + 1}/{len(grid_configs)}]" if is_grid else ""
            )
            print(f"           →{config_tag} {score_summary}")

        fold_entry: dict = {
            "fold_number": fold_number,
            "left_out_subject": left_out_subject_id,
            "n_train_windows": int(len(train_indices)),
            "n_val_windows": int(len(val_indices)),
            "n_test_windows": int(len(test_indices)),
        }
        if is_grid:
            fold_entry["configs"] = per_config_results_this_fold
        else:
            single = per_config_results_this_fold[0]
            fold_entry["config"] = single["config"]
            for name in metric_names:
                fold_entry[name] = single[name]
        all_fold_results.append(fold_entry)

    # Aggregate scores across folds.
    if is_grid:
        mean_scores: list[dict] | dict = []
        std_scores: list[dict] | dict = []
        for config_index, config in enumerate(grid_configs):
            mean_scores_for_config, std_scores_for_config = _average_fold_scores(
                per_config_fold_scores[config_index], metric_names
            )
            mean_scores.append({"config": dict(config), **mean_scores_for_config})
            std_scores.append({"config": dict(config), **std_scores_for_config})
        print(f"\nLOSO CV complete — {len(grid_configs)} configs evaluated.\n")
    else:
        mean_scores, std_scores = _average_fold_scores(
            per_config_fold_scores[0], metric_names
        )
        print(f"\nLOSO CV complete — mean scores: {mean_scores}\n")

    return {
        "fold_results": all_fold_results,
        "mean_scores": mean_scores,
        "std_scores": std_scores,
    }


# ---------------------------------------------------------------------------
# LOO CV — Leave One Out Cross Validation (one window at a time)
# ---------------------------------------------------------------------------


def loo_cv(
    model_builder_function: Callable[[], tf.keras.Model],
    feature_array: np.ndarray,
    label_array: np.ndarray,
    n_epochs: int = 50,
    batch_size: int = 32,
    verbose: int = 0,
    extra_fit_kwargs: dict | None = None,
) -> dict:
    """Leave One Out Cross Validation (LOO CV).

    In each fold a single sample window is held out for testing and the
    model is retrained on all remaining windows. This produces as many
    folds as there are samples, so it is exhaustive but slow on large
    datasets.

    Use LOO CV when:
    * Your dataset is small enough that LOSO leaves too little training data.
    * You want to estimate per-sample prediction uncertainty.
    * You are working with a single-subject dataset where subject IDs are
      not meaningful.

    Parameters
    ----------
    model_builder_function : callable () -> tf.keras.Model
        Same as in ``loso_cv`` — returns a freshly compiled model each call.
    feature_array : np.ndarray, shape (n_windows, timesteps, n_features)
        Preprocessed EEG feature windows.
    label_array : np.ndarray, shape (n_windows,)
        Integer class label per window.
    n_epochs : int, optional
        Training epochs per fold. Default 50.
    batch_size : int, optional
        Mini-batch size. Default 32.
    verbose : int, optional
        Keras verbosity. Default 0.
    extra_fit_kwargs : dict, optional
        Extra keyword arguments forwarded to ``model.fit``.

    Returns
    -------
    dict
        ``"fold_results"``  — list of dicts, one per fold. Each dict has:

            * ``"fold_number"`` : int (1-indexed)
            * ``"left_out_window_index"`` : index into feature_array
            * ``"true_label"`` : the true class label of the held-out window
            * one key per Keras metric

        ``"mean_scores"``  — dict of per-metric means across all folds.

        ``"std_scores"``   — dict of per-metric standard deviations.

    Notes
    -----
    LOO CV with n_windows folds re-trains the model from scratch n_windows
    times. On large datasets this is computationally very expensive. Consider
    ``lkocv`` with a larger k if training time is a concern.
    """
    extra_fit_kwargs = extra_fit_kwargs or {}

    total_number_of_windows = len(feature_array)
    all_fold_results: list[dict] = []
    metric_names: list[str] = []

    # sklearn's LeaveOneOut works on sample indices directly.
    leave_one_out_splitter = LeaveOneOut()

    print(f"\nLOO CV — {total_number_of_windows} folds " f"(one fold per window)\n")

    for fold_number, (train_indices, test_indices) in enumerate(
        leave_one_out_splitter.split(feature_array),
        start=1,
    ):
        # test_indices always has exactly one element in LOO.
        left_out_window_index = int(test_indices[0])
        true_label_of_held_out_window = int(label_array[left_out_window_index])

        # Only print every 50 folds to avoid flooding the console.
        if (
            fold_number == 1
            or fold_number % 50 == 0
            or fold_number == total_number_of_windows
        ):
            _print_fold_header(
                fold_number,
                total_number_of_windows,
                f"holding out window index {left_out_window_index} "
                f"(label={true_label_of_held_out_window})",
            )

        X_train = feature_array[train_indices]
        y_train = label_array[train_indices]
        X_test = feature_array[test_indices]
        y_test = label_array[test_indices]

        model = model_builder_function()

        model.fit(
            X_train,
            y_train,
            epochs=n_epochs,
            batch_size=batch_size,
            verbose=verbose,
            **extra_fit_kwargs,
        )

        test_score_values = model.evaluate(X_test, y_test, verbose=0)
        metric_names = _collect_metric_names(model)

        fold_result = {
            "fold_number": fold_number,
            "left_out_window_index": left_out_window_index,
            "true_label": true_label_of_held_out_window,
            **dict(zip(metric_names, test_score_values)),
        }
        all_fold_results.append(fold_result)

    mean_scores, std_scores = _average_fold_scores(all_fold_results, metric_names)
    print(f"\nLOO CV complete — mean scores: {mean_scores}\n")

    return {
        "fold_results": all_fold_results,
        "mean_scores": mean_scores,
        "std_scores": std_scores,
    }


# ---------------------------------------------------------------------------
# LKOCV — Leave K Subjects Out Cross Validation
# ---------------------------------------------------------------------------


def lkocv(
    model_builder_function: Callable[[], tf.keras.Model],
    feature_array: np.ndarray,
    label_array: np.ndarray,
    subject_id_array: np.ndarray,
    k_subjects_to_leave_out: int = 2,
    n_epochs: int = 50,
    batch_size: int = 32,
    verbose: int = 0,
    extra_fit_kwargs: dict | None = None,
) -> dict:
    """Leave K Subjects Out Cross Validation (LKOCV).

    A generalization of LOSO CV where K subjects are held out together as
    the test set in each fold, rather than just one. This is useful when:

    * You want each test fold to represent a larger group of unseen people.
    * Your dataset has many subjects and LOSO produces too many folds.
    * You are simulating deployment to a group of people the model has never
      seen (K=1 is identical to LOSO CV).

    Number of folds = C(n_subjects, k_subjects_to_leave_out).

    Parameters
    ----------
    model_builder_function : callable () -> tf.keras.Model
        Returns a freshly compiled Keras model each call.
    feature_array : np.ndarray, shape (n_windows, timesteps, n_features)
        Preprocessed EEG feature windows.
    label_array : np.ndarray, shape (n_windows,)
        Integer class label per window.
    subject_id_array : np.ndarray, shape (n_windows,)
        Subject identifier for each window.
    k_subjects_to_leave_out : int, optional
        Number of subjects to hold out per fold. Must be less than the total
        number of unique subjects. Default is 2.
    n_epochs : int, optional
        Training epochs per fold. Default 50.
    batch_size : int, optional
        Mini-batch size. Default 32.
    verbose : int, optional
        Keras verbosity. Default 0.
    extra_fit_kwargs : dict, optional
        Extra keyword arguments forwarded to ``model.fit``.

    Returns
    -------
    dict
        ``"fold_results"``  — list of dicts, one per fold. Each dict has:

            * ``"fold_number"`` : int (1-indexed)
            * ``"left_out_subjects"`` : list of subject IDs held out
            * ``"n_train_windows"`` : windows used for training
            * ``"n_test_windows"``  : windows used for testing
            * one key per Keras metric

        ``"mean_scores"``  — dict of per-metric means.

        ``"std_scores"``   — dict of per-metric standard deviations.
    """
    extra_fit_kwargs = extra_fit_kwargs or {}

    n_unique_subjects = len(np.unique(subject_id_array))
    if k_subjects_to_leave_out >= n_unique_subjects:
        raise ValueError(
            f"k_subjects_to_leave_out ({k_subjects_to_leave_out}) must be "
            f"less than the total number of unique subjects ({n_unique_subjects}). "
            f"There would be no training data if all subjects are left out."
        )

    # LeavePGroupsOut is sklearn's name for "leave P groups out", which is
    # exactly LKOCV with p = k_subjects_to_leave_out.
    leave_k_subjects_out_splitter = LeavePGroupsOut(n_groups=k_subjects_to_leave_out)

    total_number_of_folds = leave_k_subjects_out_splitter.get_n_splits(
        feature_array, label_array, subject_id_array
    )
    all_fold_results: list[dict] = []
    metric_names: list[str] = []

    print(
        f"\nLKOCV — {total_number_of_folds} folds, "
        f"k={k_subjects_to_leave_out} subjects held out per fold\n"
    )

    for fold_number, (train_indices, test_indices) in enumerate(
        leave_k_subjects_out_splitter.split(
            feature_array, label_array, subject_id_array
        ),
        start=1,
    ):
        # Collect the unique subject IDs that appear in the test split.
        left_out_subject_ids = list(np.unique(subject_id_array[test_indices]))

        _print_fold_header(
            fold_number,
            total_number_of_folds,
            f"testing on subjects {left_out_subject_ids}  "
            f"(train={len(train_indices)}, test={len(test_indices)} windows)",
        )

        X_train = feature_array[train_indices]
        y_train = label_array[train_indices]
        X_test = feature_array[test_indices]
        y_test = label_array[test_indices]

        model = model_builder_function()

        model.fit(
            X_train,
            y_train,
            epochs=n_epochs,
            batch_size=batch_size,
            verbose=verbose,
            **extra_fit_kwargs,
        )

        test_score_values = model.evaluate(X_test, y_test, verbose=0)
        metric_names = _collect_metric_names(model)

        fold_result = {
            "fold_number": fold_number,
            "left_out_subjects": left_out_subject_ids,
            "n_train_windows": len(train_indices),
            "n_test_windows": len(test_indices),
            **dict(zip(metric_names, test_score_values)),
        }
        all_fold_results.append(fold_result)

        score_summary = "  ".join(
            f"{name}={fold_result[name]:.4f}" for name in metric_names
        )
        print(f"           → {score_summary}")

    mean_scores, std_scores = _average_fold_scores(all_fold_results, metric_names)
    print(f"\nLKOCV complete — mean scores: {mean_scores}\n")

    return {
        "fold_results": all_fold_results,
        "mean_scores": mean_scores,
        "std_scores": std_scores,
    }


# ---------------------------------------------------------------------------
# Nested LNSO CV — Nested Leave N Subjects Out Cross Validation
# ---------------------------------------------------------------------------


def nested_lnso_cv(
    model_builder_function: Callable[[], tf.keras.Model],
    feature_array: np.ndarray,
    label_array: np.ndarray,
    subject_id_array: np.ndarray,
    n_outer_subjects_to_leave_out: int = 2,
    n_inner_subjects_to_leave_out: int = 1,
    n_epochs: int = 50,
    batch_size: int = 32,
    verbose: int = 0,
    inner_verbose: int = 0,
    extra_fit_kwargs: dict | None = None,
    inner_extra_fit_kwargs: dict | None = None,
) -> dict:
    """Nested Leave N Subjects Out Cross Validation (Nested LNSO CV).

    Uses two nested cross-validation loops to produce an unbiased estimate
    of test performance while still allowing the model to use a proper
    validation set during training (e.g. for early stopping).

    Outer loop (test evaluation):
        Hold out ``n_outer_subjects_to_leave_out`` subjects as the final
        test set for this fold. The model is *never* shown this data during
        training or validation.

    Inner loop (validation / early stopping):
        From the subjects that remain after removing the outer test subjects,
        hold out ``n_inner_subjects_to_leave_out`` subjects as the validation
        set. The model trains on everyone else and uses the inner validation
        subjects to decide when to stop (via callbacks) or to select
        checkpoints.

    After the inner loop, the model is retrained on *all* non-test subjects
    (train + inner-val combined) and evaluated once on the outer test subjects.
    This final retraining step uses the full available training data and the
    number of epochs reported as best by the inner validation.

    Why use nested CV?
        In a simple LOSO loop it is tempting to use the test subject for
        early stopping, which is data leakage. Nested CV gives you a
        legitimate validation signal without touching the test subjects.

    Parameters
    ----------
    model_builder_function : callable () -> tf.keras.Model
        Returns a freshly compiled Keras model. Called at the start of
        every outer fold (and again for the final retrain).
    feature_array : np.ndarray, shape (n_windows, timesteps, n_features)
        Preprocessed EEG feature windows.
    label_array : np.ndarray, shape (n_windows,)
        Integer class label per window.
    subject_id_array : np.ndarray, shape (n_windows,)
        Subject identifier for each window.
    n_outer_subjects_to_leave_out : int, optional
        How many subjects to hold out for the outer test set per fold.
        Default is 2.
    n_inner_subjects_to_leave_out : int, optional
        How many subjects to hold out for the inner validation set per fold.
        Default is 1.
    n_epochs : int, optional
        Maximum training epochs per fold (outer and inner). Default 50.
    batch_size : int, optional
        Mini-batch size. Default 32.
    verbose : int, optional
        Keras verbosity for the final outer retraining step. Default 0.
    inner_verbose : int, optional
        Keras verbosity for the inner loop training runs. Default 0.
    extra_fit_kwargs : dict, optional
        Extra keyword arguments forwarded to ``model.fit`` during the final
        outer retrain (e.g. early-stopping callbacks).
    inner_extra_fit_kwargs : dict, optional
        Extra keyword arguments forwarded to ``model.fit`` during inner
        loop training. Typically you pass early stopping here:

            ``{"callbacks": [tf.keras.callbacks.EarlyStopping(patience=5)]}``

    Returns
    -------
    dict
        ``"outer_fold_results"`` — list of dicts, one per outer fold. Each
        dict has:

            * ``"outer_fold_number"`` : int (1-indexed)
            * ``"outer_test_subjects"`` : list of subject IDs in the outer test set
            * ``"inner_fold_results"`` : list of dicts for each inner fold
            * ``"best_inner_val_loss"`` : lowest validation loss seen in the inner loop
            * ``"best_inner_epochs"`` : epoch count that produced the best inner val loss
            * ``"n_final_train_windows"`` : windows used for the final retrain
            * ``"n_outer_test_windows"`` : windows in the outer test set
            * one key per Keras metric (final outer test scores)

        ``"mean_scores"``  — dict of per-metric means across outer folds.

        ``"std_scores"``   — dict of per-metric standard deviations.
    """
    extra_fit_kwargs = extra_fit_kwargs or {}
    inner_extra_fit_kwargs = inner_extra_fit_kwargs or {}

    n_unique_subjects = len(np.unique(subject_id_array))
    n_subjects_needed = (
        n_outer_subjects_to_leave_out + n_inner_subjects_to_leave_out + 1
    )
    if n_unique_subjects < n_subjects_needed:
        raise ValueError(
            f"Need at least {n_subjects_needed} unique subjects "
            f"(outer={n_outer_subjects_to_leave_out} + inner={n_inner_subjects_to_leave_out} + 1 for training) "
            f"but only found {n_unique_subjects}."
        )

    # Outer splitter: holds out n_outer subjects for final testing.
    outer_splitter = LeavePGroupsOut(n_groups=n_outer_subjects_to_leave_out)
    # Inner splitter: holds out n_inner subjects from the remaining training subjects.
    inner_splitter = LeavePGroupsOut(n_groups=n_inner_subjects_to_leave_out)

    total_outer_folds = outer_splitter.get_n_splits(
        feature_array, label_array, subject_id_array
    )

    all_outer_fold_results: list[dict] = []
    metric_names: list[str] = []

    print(
        f"\nNested LNSO CV — {total_outer_folds} outer folds "
        f"(outer k={n_outer_subjects_to_leave_out}, inner k={n_inner_subjects_to_leave_out})\n"
    )

    for outer_fold_number, (outer_train_indices, outer_test_indices) in enumerate(
        outer_splitter.split(feature_array, label_array, subject_id_array),
        start=1,
    ):
        outer_test_subject_ids = list(np.unique(subject_id_array[outer_test_indices]))

        print(
            f"\n── Outer fold {outer_fold_number} / {total_outer_folds} "
            f"— test subjects: {outer_test_subject_ids} ──"
        )

        # Data available for training + inner validation in this outer fold.
        X_outer_train_pool = feature_array[outer_train_indices]
        y_outer_train_pool = label_array[outer_train_indices]
        subject_ids_outer_train_pool = subject_id_array[outer_train_indices]

        # ---- Inner loop: find the best number of epochs using a held-out
        #      validation set drawn from the outer training pool. -----------

        inner_fold_results: list[dict] = []
        inner_val_losses_per_fold: list[float] = []

        total_inner_folds = inner_splitter.get_n_splits(
            X_outer_train_pool, y_outer_train_pool, subject_ids_outer_train_pool
        )

        for inner_fold_number, (inner_train_indices, inner_val_indices) in enumerate(
            inner_splitter.split(
                X_outer_train_pool, y_outer_train_pool, subject_ids_outer_train_pool
            ),
            start=1,
        ):
            inner_val_subject_ids = list(
                np.unique(subject_ids_outer_train_pool[inner_val_indices])
            )

            _print_fold_header(
                inner_fold_number,
                total_inner_folds,
                f"inner val subjects: {inner_val_subject_ids}",
            )

            X_inner_train = X_outer_train_pool[inner_train_indices]
            y_inner_train = y_outer_train_pool[inner_train_indices]
            X_inner_val = X_outer_train_pool[inner_val_indices]
            y_inner_val = y_outer_train_pool[inner_val_indices]

            inner_model = model_builder_function()

            inner_history = inner_model.fit(
                X_inner_train,
                y_inner_train,
                validation_data=(X_inner_val, y_inner_val),
                epochs=n_epochs,
                batch_size=batch_size,
                verbose=inner_verbose,
                **inner_extra_fit_kwargs,
            )

            # Track the best validation loss and the epoch it occurred at.
            inner_val_loss_history = inner_history.history.get("val_loss", [])
            if inner_val_loss_history:
                best_val_loss_this_inner_fold = float(min(inner_val_loss_history))
                best_epoch_this_inner_fold = int(
                    np.argmin(inner_val_loss_history)
                    + 1  # +1 because epochs are 1-indexed
                )
            else:
                # No validation loss recorded (e.g. no val data was passed).
                best_val_loss_this_inner_fold = float("inf")
                best_epoch_this_inner_fold = n_epochs

            inner_val_losses_per_fold.append(best_val_loss_this_inner_fold)

            inner_fold_results.append(
                {
                    "inner_fold_number": inner_fold_number,
                    "inner_val_subjects": inner_val_subject_ids,
                    "best_val_loss": best_val_loss_this_inner_fold,
                    "best_epoch": best_epoch_this_inner_fold,
                }
            )

            print(
                f"           → best val loss = {best_val_loss_this_inner_fold:.4f} "
                f"at epoch {best_epoch_this_inner_fold}"
            )

        # The best epoch across all inner folds is the epoch that produced
        # the lowest average validation loss.
        best_inner_fold_index = int(np.argmin(inner_val_losses_per_fold))
        best_number_of_epochs = inner_fold_results[best_inner_fold_index]["best_epoch"]
        best_inner_val_loss = inner_val_losses_per_fold[best_inner_fold_index]

        print(
            f"\n  Inner loop done. Best epoch = {best_number_of_epochs} "
            f"(val_loss = {best_inner_val_loss:.4f})"
        )

        # ---- Final retrain on ALL outer training subjects (train + inner-val
        #      combined) for exactly best_number_of_epochs epochs. -----------

        print(
            f"  Retraining on all {len(outer_train_indices)} outer-train windows "
            f"for {best_number_of_epochs} epochs..."
        )

        final_model = model_builder_function()

        final_model.fit(
            X_outer_train_pool,
            y_outer_train_pool,
            epochs=best_number_of_epochs,
            batch_size=batch_size,
            verbose=verbose,
            **extra_fit_kwargs,
        )

        # ---- Evaluate on the outer test subjects (never seen before). -----
        X_outer_test = feature_array[outer_test_indices]
        y_outer_test = label_array[outer_test_indices]

        outer_test_score_values = final_model.evaluate(
            X_outer_test, y_outer_test, verbose=0
        )
        metric_names = _collect_metric_names(final_model)

        score_summary = "  ".join(
            f"{name}={val:.4f}"
            for name, val in zip(metric_names, outer_test_score_values)
        )
        print(f"  Outer test scores: {score_summary}")

        outer_fold_result = {
            "outer_fold_number": outer_fold_number,
            "outer_test_subjects": outer_test_subject_ids,
            "inner_fold_results": inner_fold_results,
            "best_inner_val_loss": best_inner_val_loss,
            "best_inner_epochs": best_number_of_epochs,
            "n_final_train_windows": len(outer_train_indices),
            "n_outer_test_windows": len(outer_test_indices),
            **dict(zip(metric_names, outer_test_score_values)),
        }
        all_outer_fold_results.append(outer_fold_result)

    mean_scores, std_scores = _average_fold_scores(all_outer_fold_results, metric_names)
    print(f"\nNested LNSO CV complete — mean scores: {mean_scores}\n")

    return {
        "outer_fold_results": all_outer_fold_results,
        "mean_scores": mean_scores,
        "std_scores": std_scores,
    }


# ---------------------------------------------------------------------------
# Master function — run any strategy with a single call
# ---------------------------------------------------------------------------


def run_cross_validation(
    cv_strategy: Literal["loso", "loo", "lkocv", "nested_lnso"],
    model_builder_function: Callable[..., tf.keras.Model],
    feature_array: np.ndarray,
    label_array: np.ndarray,
    subject_id_array: np.ndarray | None = None,
    k_subjects_to_leave_out: int = 2,
    n_outer_subjects_to_leave_out: int = 2,
    n_inner_subjects_to_leave_out: int = 1,
    n_epochs: int = 50,
    batch_size: int = 32,
    hyperparameters: dict | None = None,
    validation_n_users: int | None = None,
    verbose: int = 0,
    extra_fit_kwargs: dict | None = None,
) -> dict:
    """Run any cross-validation strategy with a single function call.

    This is the recommended entry point. Pass the name of the strategy you
    want and this function dispatches to the appropriate implementation,
    forwarding all relevant parameters. All four strategies return the same
    top-level keys (``fold_results`` / ``outer_fold_results``,
    ``mean_scores``, ``std_scores``) so downstream analysis code does not
    need to change when you switch strategies.

    Parameters
    ----------
    cv_strategy : {"loso", "loo", "lkocv", "nested_lnso"}
        Which cross-validation strategy to run:

        * ``"loso"``        — Leave One Subject Out (one fold per subject)
        * ``"loo"``         — Leave One Window Out  (one fold per window)
        * ``"lkocv"``       — Leave K Subjects Out  (k set by ``k_subjects_to_leave_out``)
        * ``"nested_lnso"`` — Nested Leave N Subjects Out (two-loop CV with
                              inner validation)

    model_builder_function : callable () -> tf.keras.Model
        A zero-argument function that returns a freshly initialized, compiled
        Keras model. Called once at the start of every fold so that weights
        never leak between folds.

        Example::

            from eegproc.supervised import bilstm_classifier

            def build_model():
                # bilstm_classifier returns an already-compiled model.
                return bilstm_classifier(
                    timesteps=128, n_features=84,
                    lstm_units=64, n_classes=3,
                )

    feature_array : np.ndarray, shape (n_windows, timesteps, n_features)
        Preprocessed EEG feature windows. Each row is one sliding window.
    label_array : np.ndarray, shape (n_windows,)
        Integer class label for each window.
    subject_id_array : np.ndarray, shape (n_windows,), optional
        Subject identifier for each window. Required for all strategies
        except ``"loo"``.
    k_subjects_to_leave_out : int, optional
        Number of subjects held out per fold. Only used by ``"lkocv"``.
        Default is 2.
    n_outer_subjects_to_leave_out : int, optional
        Subjects held out for the outer test set. Only used by
        ``"nested_lnso"``. Default is 2.
    n_inner_subjects_to_leave_out : int, optional
        Subjects held out for the inner validation set. Only used by
        ``"nested_lnso"``. Default is 1.
    n_epochs : int, optional
        Training epochs per fold. Default 50. Ignored for ``"loso"`` when
        ``hyperparameters`` is provided.
    batch_size : int, optional
        Mini-batch size. Default 32. Ignored for ``"loso"`` when
        ``hyperparameters`` is provided.
    hyperparameters : dict, optional
        Currently only honored by ``cv_strategy="loso"``. Forwarded to
        :func:`loso_cv` (see its docstring). For other strategies this
        kwarg is silently ignored — they still use ``n_epochs`` and
        ``batch_size`` until the broader CV redesign lands.
    validation_n_users : int, optional
        Currently only honored by ``cv_strategy="loso"``. Forwarded to
        :func:`loso_cv` to create a subject-level validation split inside
        each fold.
    verbose : int, optional
        Keras verbosity (0 = silent, 1 = progress bar, 2 = one line per
        epoch). Default 0.
    extra_fit_kwargs : dict, optional
        Extra keyword arguments forwarded to ``model.fit`` in every fold,
        such as early-stopping callbacks::

            extra_fit_kwargs={
                "callbacks": [
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss", patience=5, restore_best_weights=True
                    )
                ]
            }

    Returns
    -------
    dict
        Results dictionary from the selected strategy function. Always
        contains ``"mean_scores"`` and ``"std_scores"`` dicts, plus either
        ``"fold_results"`` (LOSO, LOO, LKOCV) or ``"outer_fold_results"``
        (nested LNSO).

    Raises
    ------
    ValueError
        If ``cv_strategy`` is not one of the four supported names, or if
        ``subject_id_array`` is None for a strategy that requires it.

    Examples
    --------
    LOSO CV with a BiLSTM classifier::

        results = run_cross_validation(
            cv_strategy="loso",
            model_builder_function=build_model,
            feature_array=X,
            label_array=y,
            subject_id_array=subject_ids,
            n_epochs=40,
            batch_size=32,
        )
        print(results["mean_scores"])

    LKOCV leaving 3 subjects out at a time::

        results = run_cross_validation(
            cv_strategy="lkocv",
            model_builder_function=build_model,
            feature_array=X,
            label_array=y,
            subject_id_array=subject_ids,
            k_subjects_to_leave_out=3,
        )

    Nested LNSO with early stopping::

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
        results = run_cross_validation(
            cv_strategy="nested_lnso",
            model_builder_function=build_model,
            feature_array=X,
            label_array=y,
            subject_id_array=subject_ids,
            n_outer_subjects_to_leave_out=2,
            n_inner_subjects_to_leave_out=1,
            extra_fit_kwargs={"callbacks": [early_stop]},
        )
    """
    # Strategies that need subject IDs.
    strategies_requiring_subject_ids = {"loso", "lkocv", "nested_lnso"}
    if cv_strategy in strategies_requiring_subject_ids and subject_id_array is None:
        raise ValueError(
            f"cv_strategy='{cv_strategy}' requires subject_id_array, "
            f"but subject_id_array was not provided (got None). "
            f"Pass a numpy array of subject identifiers, one entry per window."
        )

    legacy_shared_kwargs = dict(
        model_builder_function=model_builder_function,
        feature_array=feature_array,
        label_array=label_array,
        n_epochs=n_epochs,
        batch_size=batch_size,
        verbose=verbose,
        extra_fit_kwargs=extra_fit_kwargs,
    )

    if cv_strategy == "loso":
        # loso_cv no longer takes n_epochs / batch_size directly. Build a
        # legacy-compat hyperparameters dict from those kwargs if the caller
        # didn't supply one.
        effective_hyperparameters = (
            hyperparameters
            if hyperparameters is not None
            else {"epochs": n_epochs, "batch_size": batch_size}
        )
        return loso_cv(
            model_builder_function=model_builder_function,
            feature_array=feature_array,
            label_array=label_array,
            subject_id_array=subject_id_array,
            hyperparameters=effective_hyperparameters,
            validation_n_users=validation_n_users,
            verbose=verbose,
            extra_fit_kwargs=extra_fit_kwargs,
        )

    elif cv_strategy == "loo":
        # LOO does not use subject IDs — it iterates over individual windows.
        return loo_cv(**legacy_shared_kwargs)

    elif cv_strategy == "lkocv":
        return lkocv(
            subject_id_array=subject_id_array,
            k_subjects_to_leave_out=k_subjects_to_leave_out,
            **legacy_shared_kwargs,
        )

    elif cv_strategy == "nested_lnso":
        return nested_lnso_cv(
            subject_id_array=subject_id_array,
            n_outer_subjects_to_leave_out=n_outer_subjects_to_leave_out,
            n_inner_subjects_to_leave_out=n_inner_subjects_to_leave_out,
            **legacy_shared_kwargs,
        )

    else:
        supported_strategies = ["loso", "loo", "lkocv", "nested_lnso"]
        raise ValueError(
            f"Unknown cv_strategy '{cv_strategy}'. "
            f"Choose one of: {supported_strategies}"
        )
