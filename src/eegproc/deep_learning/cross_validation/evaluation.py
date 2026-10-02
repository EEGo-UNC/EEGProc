"""Evaluating a fitted model on one fold."""

from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotations only
    from typing import Callable
    from typing import Literal
    import tensorflow as tf

from .arrays import _as_numpy_1d, _is_trial_tensor, _python_scalar
from .constants import _DEFAULT_ECE_BINS
from .probabilities import _predict_labels, _predict_probabilities
from .aggregation import _aggregate_window_probabilities_by_trial, _direct_trial_aggregation
from .metrics import _print_probability_diagnostics
from .reporting import _keras_decoder_scores, _keras_evaluation_results, _level_scores, _make_prediction_log, _make_trial_prediction_log, _make_variational_interval_logs, _prefix_scores, _print_metric_row, _print_user_metrics, _validate_evaluation_level

def _apply_preprocessing_strategy(
    preprocessing_strategy: Callable | None,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
):
    """Apply optional preprocessing inside a CV fold.

    The preprocessing strategy is deliberately fold-local so that anything
    fit inside the strategy is fit only on the training partition.
    """
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


def _validate_processed_alignment(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    trial_ids: np.ndarray,
    partition_name: str,
) -> None:
    """Ensure fold-local preprocessing preserved sample order and count."""
    lengths = (len(X), len(y), len(subject_ids), len(trial_ids))
    if len(set(lengths)) != 1:
        raise ValueError(
            f"Preprocessing changed the number of {partition_name} samples or "
            "misaligned labels/IDs. Sample creation, removal, reordering, and "
            "resampling must occur before loso_cv. Got lengths "
            f"X/y/subject/trial={lengths}."
        )


def _evaluate_trial_tensor_fold(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    subject_ids_test: np.ndarray,
    trial_ids_test: np.ndarray,
    fold_index: int,
    metrics: list[str] | tuple[str, ...],
    batch_size: int | None,
    n_prediction_latent_samples: int,
    latent_sampling_seed: int | None,
    log_predictions: bool,
    log_variational_intervals: bool,
    n_uncertainty_samples: int,
    ci_level: float,
    decision_threshold: float = 0.5,
    ece_bins: int = _DEFAULT_ECE_BINS,
    print_results: bool = True,
) -> dict:
    """Evaluate a model that emits one classifier prediction per trial."""
    y_true_trial = _as_numpy_1d(y_test).astype(np.int64)
    probabilities_trial = _predict_probabilities(
        model=model,
        X=X_test,
        batch_size=batch_size,
        n_prediction_latent_samples=n_prediction_latent_samples,
        latent_sampling_seed=latent_sampling_seed,
    )
    y_pred_trial = _predict_labels(
        probabilities_trial,
        decision_threshold=decision_threshold,
    )
    if print_results:
        _print_probability_diagnostics(
            label=f"fold {fold_index} test trial",
            probabilities=probabilities_trial,
            y_true=y_true_trial,
        )
    keras_evaluation = _keras_evaluation_results(
        model=model,
        X=X_test,
        y=y_test,
        batch_size=batch_size,
    )
    keras_model_loss = float(keras_evaluation["loss"])
    decoder_scores = _keras_decoder_scores(keras_evaluation)

    trial_scores = _level_scores(
        y_true=y_true_trial,
        y_pred=y_pred_trial,
        probabilities=probabilities_trial,
        metrics=metrics,
        ece_bins=ece_bins,
    )
    trial_scores["joint_loss"] = keras_model_loss
    trial_scores.update(decoder_scores)

    n_trials = int(len(y_true_trial))
    n_windows_per_trial = int(X_test.shape[1])
    n_windows = n_trials * n_windows_per_trial
    fold_scores = {
        "fold": int(fold_index),
        "evaluation_level": "trial",
        "classification_level": "trial",
        "n_samples": n_trials,
        "n_windows": n_windows,
        "n_trials": n_trials,
        "windows_per_trial": n_windows_per_trial,
        "keras_model_loss": keras_model_loss,
        "joint_loss": keras_model_loss,
        "prediction_latent_samples": int(n_prediction_latent_samples),
        "decision_threshold": float(decision_threshold),
        **trial_scores,
        **_prefix_scores(trial_scores, "trial"),
    }

    window_fold_metrics = {
        "fold": int(fold_index),
        "n_windows": n_windows,
        "classification_available": False,
        "joint_loss": keras_model_loss,
        **decoder_scores,
    }
    trial_fold_metrics = {
        "fold": int(fold_index),
        "n_trials": n_trials,
        "windows_per_trial": n_windows_per_trial,
        **trial_scores,
    }

    user_rows: list[dict] = []
    for subject_id in np.unique(subject_ids_test):
        trial_mask = subject_ids_test == subject_id
        user_trial_scores = _level_scores(
            y_true=y_true_trial[trial_mask],
            y_pred=y_pred_trial[trial_mask],
            probabilities=probabilities_trial[trial_mask],
            metrics=metrics,
            ece_bins=ece_bins,
        )
        user_rows.append(
            {
                "fold": int(fold_index),
                "subject_id": _python_scalar(subject_id),
                "evaluation_level": "trial",
                "classification_level": "trial",
                "n_samples": int(trial_mask.sum()),
                "n_windows": int(trial_mask.sum()) * n_windows_per_trial,
                "n_trials": int(trial_mask.sum()),
                **user_trial_scores,
                **_prefix_scores(user_trial_scores, "trial"),
            }
        )

    trial_aggregation = _direct_trial_aggregation(
        probabilities=probabilities_trial,
        y_true=y_true_trial,
        subject_ids=subject_ids_test,
        trial_ids=trial_ids_test,
        n_windows_per_trial=n_windows_per_trial,
        decision_threshold=decision_threshold,
    )
    trial_prediction_rows = (
        _make_trial_prediction_log(fold_index, trial_aggregation)
        if log_predictions
        else []
    )

    window_interval_rows: list[dict] = []
    trial_interval_rows: list[dict] = []
    if log_variational_intervals:
        window_interval_rows, trial_interval_rows = _make_variational_interval_logs(
            model=model,
            X=X_test,
            y_true=y_true_trial,
            subject_ids=subject_ids_test,
            trial_ids=trial_ids_test,
            fold_index=fold_index,
            n_uncertainty_samples=n_uncertainty_samples,
            ci_level=ci_level,
            decision_threshold=decision_threshold,
        )

    if print_results:
        _print_metric_row(
            title=f"Fold {fold_index} metrics (trial primary)",
            row=fold_scores,
        )
        _print_user_metrics(user_rows)

    return {
        "fold_metrics": fold_scores,
        "window_fold_metrics": window_fold_metrics,
        "trial_fold_metrics": trial_fold_metrics,
        "user_metrics": user_rows,
        "window_prediction_log": [],
        "trial_prediction_log": trial_prediction_rows,
        "window_variational_interval_log": window_interval_rows,
        "trial_variational_interval_log": trial_interval_rows,
    }


def _evaluate_classification_fold(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    subject_ids_test: np.ndarray,
    trial_ids_test: np.ndarray,
    fold_index: int,
    metrics: list[str] | tuple[str, ...],
    evaluation_level: Literal["window", "trial"] = "trial",
    batch_size: int | None = None,
    n_prediction_latent_samples: int = 0,
    latent_sampling_seed: int | None = None,
    log_predictions: bool = True,
    log_variational_intervals: bool = False,
    n_uncertainty_samples: int = 30,
    ci_level: float = 0.95,
    decision_threshold: float = 0.5,
    ece_bins: int = _DEFAULT_ECE_BINS,
    print_results: bool = True,
) -> dict:
    """Evaluate one outer fold at the model's native classification level."""
    _validate_evaluation_level(evaluation_level, "evaluation_level")
    if _is_trial_tensor(X_test):
        if evaluation_level != "trial":
            raise ValueError(
                "Hierarchical rank-4 inputs produce trial-level classifier "
                "outputs; evaluation_level must be 'trial'."
            )
        return _evaluate_trial_tensor_fold(
            model=model,
            X_test=X_test,
            y_test=y_test,
            subject_ids_test=subject_ids_test,
            trial_ids_test=trial_ids_test,
            fold_index=fold_index,
            metrics=metrics,
            batch_size=batch_size,
            n_prediction_latent_samples=n_prediction_latent_samples,
            latent_sampling_seed=latent_sampling_seed,
            log_predictions=log_predictions,
            log_variational_intervals=log_variational_intervals,
            n_uncertainty_samples=n_uncertainty_samples,
            ci_level=ci_level,
            decision_threshold=decision_threshold,
            ece_bins=ece_bins,
            print_results=print_results,
        )
    y_true_window = _as_numpy_1d(y_test).astype(np.int64)

    probabilities_window = _predict_probabilities(
        model=model,
        X=X_test,
        batch_size=batch_size,
        n_prediction_latent_samples=n_prediction_latent_samples,
        latent_sampling_seed=latent_sampling_seed,
    )
    y_pred_window = _predict_labels(
        probabilities_window,
        decision_threshold=decision_threshold,
    )
    if print_results:
        _print_probability_diagnostics(
            label=f"fold {fold_index} test window",
            probabilities=probabilities_window,
            y_true=y_true_window,
        )

    # model.evaluate() is retained as a diagnostic because joint Keras models
    # may include reconstruction/regularization terms beyond classification.
    # It also exposes aggregate and branch-specific reconstruction quality.
    keras_evaluation = _keras_evaluation_results(
        model=model,
        X=X_test,
        y=y_test,
        batch_size=batch_size,
    )
    keras_model_loss = keras_evaluation["loss"]
    decoder_scores = _keras_decoder_scores(keras_evaluation)

    window_scores = _level_scores(
        y_true=y_true_window,
        y_pred=y_pred_window,
        probabilities=probabilities_window,
        metrics=metrics,
        ece_bins=ece_bins,
    )
    # ``loss`` above is classifier probability log loss. ``joint_loss`` is
    # the model's complete weighted VAE + VC objective returned by Keras.
    window_scores["joint_loss"] = float(keras_model_loss)
    window_scores.update(decoder_scores)

    trial_aggregation = _aggregate_window_probabilities_by_trial(
        probabilities=probabilities_window,
        y_true=y_true_window,
        subject_ids=subject_ids_test,
        trial_ids=trial_ids_test,
        decision_threshold=decision_threshold,
    )
    if print_results:
        _print_probability_diagnostics(
            label=f"fold {fold_index} test trial-aggregated",
            probabilities=trial_aggregation["probabilities"],
            y_true=trial_aggregation["y_true"],
        )
    trial_scores = _level_scores(
        y_true=trial_aggregation["y_true"],
        y_pred=trial_aggregation["y_pred"],
        probabilities=trial_aggregation["probabilities"],
        metrics=metrics,
        ece_bins=ece_bins,
    )
    trial_scores.update(decoder_scores)

    primary_scores = trial_scores if evaluation_level == "trial" else window_scores
    fold_scores = {
        "fold": int(fold_index),
        "n_samples": int(
            len(trial_aggregation["y_true"])
            if evaluation_level == "trial"
            else len(y_true_window)
        ),
        "n_windows": int(len(y_true_window)),
        "n_trials": int(len(trial_aggregation["y_true"])),
        "keras_model_loss": float(keras_model_loss),
        "joint_loss": float(keras_model_loss),
        "prediction_latent_samples": int(n_prediction_latent_samples),
        "decision_threshold": float(decision_threshold),
        **primary_scores,
        **_prefix_scores(window_scores, "window"),
        **_prefix_scores(trial_scores, "trial"),
    }

    window_fold_metrics = {
        "fold": int(fold_index),
        "n_windows": int(len(y_true_window)),
        "keras_model_loss": float(keras_model_loss),
        "joint_loss": float(keras_model_loss),
        "decision_threshold": float(decision_threshold),
        **window_scores,
    }
    trial_fold_metrics = {
        "fold": int(fold_index),
        "n_trials": int(len(trial_aggregation["y_true"])),
        "decision_threshold": float(decision_threshold),
        **trial_scores,
    }

    user_rows: list[dict] = []
    for subject_id in np.unique(subject_ids_test):
        window_mask = subject_ids_test == subject_id
        trial_mask = trial_aggregation["subject_ids"] == subject_id

        user_window_scores = _level_scores(
            y_true=y_true_window[window_mask],
            y_pred=y_pred_window[window_mask],
            probabilities=probabilities_window[window_mask],
            metrics=metrics,
            ece_bins=ece_bins,
        )
        user_trial_scores = _level_scores(
            y_true=trial_aggregation["y_true"][trial_mask],
            y_pred=trial_aggregation["y_pred"][trial_mask],
            probabilities=trial_aggregation["probabilities"][trial_mask],
            metrics=metrics,
            ece_bins=ece_bins,
        )
        user_primary_scores = (
            user_trial_scores if evaluation_level == "trial" else user_window_scores
        )

        user_rows.append(
            {
                "fold": int(fold_index),
                "subject_id": _python_scalar(subject_id),
                "evaluation_level": evaluation_level,
                "n_samples": int(
                    trial_mask.sum()
                    if evaluation_level == "trial"
                    else window_mask.sum()
                ),
                "n_windows": int(window_mask.sum()),
                "n_trials": int(trial_mask.sum()),
                **user_primary_scores,
                **_prefix_scores(user_window_scores, "window"),
                **_prefix_scores(user_trial_scores, "trial"),
            }
        )

    window_prediction_rows: list[dict] = []
    trial_prediction_rows: list[dict] = []
    if log_predictions:
        window_prediction_rows = _make_prediction_log(
            fold_index=fold_index,
            y_true=y_true_window,
            y_pred=y_pred_window,
            probabilities=probabilities_window,
            subject_ids=subject_ids_test,
            trial_ids=trial_ids_test,
        )
        trial_prediction_rows = _make_trial_prediction_log(
            fold_index=fold_index,
            trial_aggregation=trial_aggregation,
        )

    window_interval_rows: list[dict] = []
    trial_interval_rows: list[dict] = []
    if log_variational_intervals:
        window_interval_rows, trial_interval_rows = _make_variational_interval_logs(
            model=model,
            X=X_test,
            y_true=y_true_window,
            subject_ids=subject_ids_test,
            trial_ids=trial_ids_test,
            fold_index=fold_index,
            n_uncertainty_samples=n_uncertainty_samples,
            ci_level=ci_level,
        )

    if print_results:
        _print_metric_row(
            title=f"Fold {fold_index} metrics ({evaluation_level} primary)",
            row=fold_scores,
        )
        _print_user_metrics(user_rows)

    return {
        "fold_metrics": fold_scores,
        "window_fold_metrics": window_fold_metrics,
        "trial_fold_metrics": trial_fold_metrics,
        "user_metrics": user_rows,
        "window_prediction_log": window_prediction_rows,
        "trial_prediction_log": trial_prediction_rows,
        "window_variational_interval_log": window_interval_rows,
        "trial_variational_interval_log": trial_interval_rows,
    }


def _evaluate_inner_config(
    model: tf.keras.Model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    subject_ids_val: np.ndarray,
    trial_ids_val: np.ndarray,
    metrics: list[str] | tuple[str, ...],
    selection_level: Literal["window", "trial"] = "trial",
    batch_size: int | None = None,
    n_prediction_latent_samples: int = 0,
    latent_sampling_seed: int | None = None,
) -> dict:
    """Evaluate an inner fold at both levels and expose selection-level scores."""
    _validate_evaluation_level(selection_level, "selection_level")
    y_true_window = _as_numpy_1d(y_val).astype(np.int64)
    probabilities_window = _predict_probabilities(
        model,
        X_val,
        batch_size=batch_size,
        n_prediction_latent_samples=n_prediction_latent_samples,
        latent_sampling_seed=latent_sampling_seed,
    )
    y_pred_window = _predict_labels(probabilities_window)

    window_scores = _level_scores(
        y_true=y_true_window,
        y_pred=y_pred_window,
        probabilities=probabilities_window,
        metrics=metrics,
    )
    keras_evaluation = _keras_evaluation_results(
        model, X_val, y_val, batch_size=batch_size
    )
    window_scores["keras_model_loss"] = keras_evaluation["loss"]
    window_scores["joint_loss"] = keras_evaluation["loss"]
    decoder_accuracy = keras_evaluation.get("decoder_accuracy")
    if decoder_accuracy is not None:
        window_scores["decoder_accuracy"] = float(decoder_accuracy)

    trial_aggregation = _aggregate_window_probabilities_by_trial(
        probabilities=probabilities_window,
        y_true=y_true_window,
        subject_ids=subject_ids_val,
        trial_ids=trial_ids_val,
    )
    trial_scores = _level_scores(
        y_true=trial_aggregation["y_true"],
        y_pred=trial_aggregation["y_pred"],
        probabilities=trial_aggregation["probabilities"],
        metrics=metrics,
    )

    primary_scores = trial_scores if selection_level == "trial" else window_scores
    primary_metric_keys = ["loss", *metrics]
    if "joint_loss" in primary_scores:
        primary_metric_keys.append("joint_loss")

    return {
        **{key: primary_scores[key] for key in primary_metric_keys},
        **(
            {"decoder_accuracy": float(decoder_accuracy)}
            if decoder_accuracy is not None
            else {}
        ),
        **_prefix_scores(window_scores, "window"),
        **_prefix_scores(trial_scores, "trial"),
        "selection_level": selection_level,
        "n_val_windows": int(len(y_true_window)),
        "n_val_trials": int(len(trial_aggregation["y_true"])),
    }
