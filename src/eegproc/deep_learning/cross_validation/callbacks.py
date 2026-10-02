"""Keras callbacks that report validation and held-out-user metrics during training."""

from __future__ import annotations

from typing import Literal
from sklearn.metrics import f1_score
import numpy as np
from sklearn.metrics import recall_score
import tensorflow as tf

from .arrays import _as_numpy_1d, _is_trial_tensor, _python_scalar
from .constants import _DEFAULT_ECE_BINS
from .probabilities import _normalize_decision_thresholds, _predict_labels, _predict_probabilities
from .aggregation import _aggregate_window_probabilities_by_trial, _direct_trial_aggregation
from .metrics import _classification_metrics, _decoder_reconstruction_scores, _probability_log_loss

class TrialValidationMetrics(tf.keras.callbacks.Callback):
    """Compute deterministic trial-level validation metrics each epoch.

    Hierarchical models are scored directly from one output per trial. Legacy
    window models are still aggregated within each (subject_id, trial_id) pair.
    The resulting values are added to the Keras epoch logs as
    ``val_trial_f1``, ``val_trial_balanced_accuracy``, and ``val_trial_loss``
    so callbacks such as EarlyStopping can monitor them.
    """

    def __init__(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        subject_ids_val: np.ndarray,
        trial_ids_val: np.ndarray,
        batch_size: int | None = None,
    ) -> None:
        super().__init__()
        self.X_val = np.asarray(X_val)
        self.y_val = np.asarray(y_val)
        self.subject_ids_val = np.asarray(subject_ids_val)
        self.trial_ids_val = np.asarray(trial_ids_val)
        self.batch_size = batch_size

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if logs is None:
            return

        # Use posterior-mean inference here. It is deterministic, inexpensive,
        # and avoids Monte Carlo sampling noise in the stopping decision.
        probabilities_model = _predict_probabilities(
            model=self.model,
            X=self.X_val,
            batch_size=self.batch_size,
            n_prediction_latent_samples=0,
            latent_sampling_seed=None,
        )

        if _is_trial_tensor(self.X_val):
            trial_aggregation = _direct_trial_aggregation(
                probabilities=probabilities_model,
                y_true=self.y_val,
                subject_ids=self.subject_ids_val,
                trial_ids=self.trial_ids_val,
                n_windows_per_trial=self.X_val.shape[1],
            )
        else:
            trial_aggregation = _aggregate_window_probabilities_by_trial(
                probabilities=probabilities_model,
                y_true=self.y_val,
                subject_ids=self.subject_ids_val,
                trial_ids=self.trial_ids_val,
            )

        probabilities_trial = trial_aggregation["probabilities"]
        y_true_trial = trial_aggregation["y_true"]
        y_pred_trial = trial_aggregation["y_pred"]
        expected_labels = list(range(probabilities_trial.shape[1]))

        if probabilities_trial.shape[1] == 2:
            val_trial_f1 = f1_score(
                y_true_trial,
                y_pred_trial,
                average="binary",
                pos_label=1,
                zero_division=0,
            )
        else:
            val_trial_f1 = f1_score(
                y_true_trial,
                y_pred_trial,
                average="macro",
                labels=expected_labels,
                zero_division=0,
            )
        logs["val_trial_f1"] = float(val_trial_f1)
        logs["val_trial_macro_f1"] = float(
            f1_score(
                y_true_trial,
                y_pred_trial,
                average="macro",
                labels=expected_labels,
                zero_division=0,
            )
        )
        logs["val_trial_balanced_accuracy"] = float(
            recall_score(
                y_true_trial,
                y_pred_trial,
                average="macro",
                labels=expected_labels,
                zero_division=0,
            )
        )
        logs["val_trial_loss"] = _probability_log_loss(
            y_true=y_true_trial,
            probabilities=probabilities_trial,
        )


class EvaluationLevelValidationMetrics(tf.keras.callbacks.Callback):
    """Add complete window- or trial-level validation metrics to epoch logs.

    This is used by nested subject-calibration LOSO folds so source-model epoch
    selection is based only on source-validation subjects. The outer target
    subject and all of its calibration/evaluation trials remain untouched.
    """

    _METRICS = (
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

    def __init__(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        subject_ids_val: np.ndarray,
        trial_ids_val: np.ndarray,
        evaluation_level: Literal["window", "trial"],
        batch_size: int | None = None,
        ece_bins: int = _DEFAULT_ECE_BINS,
    ) -> None:
        super().__init__()
        self.X_val = np.asarray(X_val)
        self.y_val = np.asarray(y_val)
        self.subject_ids_val = np.asarray(subject_ids_val)
        self.trial_ids_val = np.asarray(trial_ids_val)
        self.evaluation_level = evaluation_level
        self.batch_size = batch_size
        self.ece_bins = int(ece_bins)

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if logs is None:
            return
        probabilities = _predict_probabilities(
            model=self.model,
            X=self.X_val,
            batch_size=self.batch_size,
            n_prediction_latent_samples=0,
            latent_sampling_seed=None,
        )
        if self.evaluation_level == "trial":
            if _is_trial_tensor(self.X_val):
                aggregation = _direct_trial_aggregation(
                    probabilities=probabilities,
                    y_true=self.y_val,
                    subject_ids=self.subject_ids_val,
                    trial_ids=self.trial_ids_val,
                    n_windows_per_trial=self.X_val.shape[1],
                )
            else:
                aggregation = _aggregate_window_probabilities_by_trial(
                    probabilities=probabilities,
                    y_true=self.y_val,
                    subject_ids=self.subject_ids_val,
                    trial_ids=self.trial_ids_val,
                )
            probabilities = aggregation["probabilities"]
            y_true = aggregation["y_true"]
            y_pred = aggregation["y_pred"]
        else:
            y_true = _as_numpy_1d(self.y_val).astype(np.int64)
            y_pred = _predict_labels(probabilities)

        scores = _classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            probabilities=probabilities,
            metrics=self._METRICS,
            n_classes=probabilities.shape[1],
            ece_bins=self.ece_bins,
        )
        prefix = f"val_{self.evaluation_level}_"
        for metric_name, value in scores.items():
            logs[f"{prefix}{metric_name}"] = float(value)
        logs[f"{prefix}loss"] = _probability_log_loss(
            y_true=y_true,
            probabilities=probabilities,
        )


class HeldOutUserOracleMetrics(tf.keras.callbacks.Callback):
    """Print reporting-only metrics on the outer held-out user each epoch.

    This callback deliberately keeps its values out of the Keras ``logs``
    dictionary. Consequently, the oracle metrics cannot be consumed by
    EarlyStopping, checkpoint selection, or any other training callback. The
    target data are used only for inference after an epoch has finished; they
    never enter a gradient calculation.
    """

    _METRICS = (
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "brier_score",
    )

    def __init__(
        self,
        X_target: np.ndarray,
        y_target: np.ndarray,
        subject_ids_target: np.ndarray,
        trial_ids_target: np.ndarray,
        target_subject,
        evaluation_level: Literal["window", "trial"],
        batch_size: int | None = None,
        decision_threshold: float = 0.5,
        diagnostic_thresholds: list[float] | tuple[float, ...] | None = None,
        ece_bins: int = _DEFAULT_ECE_BINS,
        calibration_shots: int | None = None,
        calibration_fold: int | None = None,
    ) -> None:
        super().__init__()
        self.X_target = np.asarray(X_target)
        self.y_target = np.asarray(y_target)
        self.subject_ids_target = np.asarray(subject_ids_target)
        self.trial_ids_target = np.asarray(trial_ids_target)
        self.target_subject = _python_scalar(target_subject)
        self.evaluation_level = evaluation_level
        self.batch_size = batch_size
        self.decision_threshold = float(decision_threshold)
        raw_thresholds = (
            (self.decision_threshold,)
            if diagnostic_thresholds is None
            else (*diagnostic_thresholds, self.decision_threshold)
        )
        normalized_thresholds = _normalize_decision_thresholds(raw_thresholds)
        self.diagnostic_thresholds = (
            self.decision_threshold,
            *(
                threshold
                for threshold in normalized_thresholds
                if threshold != self.decision_threshold
            ),
        )
        self.ece_bins = int(ece_bins)
        self.calibration_shots = (
            None if calibration_shots is None else int(calibration_shots)
        )
        self.calibration_fold = (
            None if calibration_fold is None else int(calibration_fold)
        )
        self.n_target_trials = int(len(np.unique(self.trial_ids_target)))
        self.history: list[dict] = []

    def _calibration_context(self) -> str:
        """Return shot/fold context when reporting a calibration continuation."""
        if self.calibration_shots is None or self.calibration_fold is None:
            return ""
        return (
            f" shots={self.calibration_shots}"
            f" calibration_fold={self.calibration_fold}"
        )

    def on_train_begin(self, logs: dict | None = None) -> None:
        print(
            "[ORACLE] Held-out-user metrics are reporting only and are not "
            "used for gradients, early stopping, threshold selection, or "
            "checkpoint selection. Official evaluation threshold="
            f"{self.decision_threshold:.4f}; reporting-only thresholds="
            f"{list(self.diagnostic_thresholds)}; "
            f"held-out trials={self.n_target_trials}"
            f"{self._calibration_context()}.",
            flush=True,
        )

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        probabilities = _predict_probabilities(
            model=self.model,
            X=self.X_target,
            batch_size=self.batch_size,
            n_prediction_latent_samples=0,
            latent_sampling_seed=None,
        )
        decoder_scores = _decoder_reconstruction_scores(
            model=self.model,
            X=self.X_target,
            batch_size=self.batch_size,
        )
        if decoder_scores:
            decoder_parts = [
                f"{name}={decoder_scores[name]:.4f}"
                for name in (
                    "decoder_r2",
                    "gcn_gru_decoder_r2",
                    "bilstm_decoder_r2",
                    "reconstruction_loss",
                    "gcn_gru_reconstruction_loss",
                    "bilstm_reconstruction_loss",
                )
                if name in decoder_scores
            ]
            print(
                f"[ORACLE DECODER held-out user={self.target_subject!r} "
                f"epoch={int(epoch) + 1}] "
                + "  ".join(decoder_parts),
                flush=True,
            )

        if self.evaluation_level == "trial":
            if _is_trial_tensor(self.X_target):
                aggregation = _direct_trial_aggregation(
                    probabilities=probabilities,
                    y_true=self.y_target,
                    subject_ids=self.subject_ids_target,
                    trial_ids=self.trial_ids_target,
                    n_windows_per_trial=self.X_target.shape[1],
                    decision_threshold=self.decision_threshold,
                )
            else:
                aggregation = _aggregate_window_probabilities_by_trial(
                    probabilities=probabilities,
                    y_true=self.y_target,
                    subject_ids=self.subject_ids_target,
                    trial_ids=self.trial_ids_target,
                    decision_threshold=self.decision_threshold,
                )
            probabilities = aggregation["probabilities"]
            y_true = aggregation["y_true"]
        else:
            y_true = _as_numpy_1d(self.y_target).astype(np.int64)

        for threshold in self.diagnostic_thresholds:
            y_pred = _predict_labels(
                probabilities,
                decision_threshold=threshold,
            )
            scores = _classification_metrics(
                y_true=y_true,
                y_pred=y_pred,
                probabilities=probabilities,
                metrics=self._METRICS,
                n_classes=probabilities.shape[1],
                ece_bins=self.ece_bins,
            )
            is_official = bool(threshold == self.decision_threshold)
            row = {
                "target_subject": self.target_subject,
                "epoch": int(epoch) + 1,
                "evaluation_level": self.evaluation_level,
                "decision_threshold": float(threshold),
                "official_decision_threshold": self.decision_threshold,
                "is_official_decision_threshold": is_official,
                "n_target_trials": self.n_target_trials,
                **{name: float(value) for name, value in scores.items()},
                **decoder_scores,
            }
            if self.calibration_shots is not None:
                row["calibration_shots"] = self.calibration_shots
            if self.calibration_fold is not None:
                row["calibration_fold"] = self.calibration_fold
            if probabilities.shape[1] == 2:
                row["predicted_class_1_fraction"] = float(np.mean(y_pred == 1))
                row["true_class_1_fraction"] = float(np.mean(y_true == 1))
            self.history.append(row)

            parts = [
                f"{self.evaluation_level}_accuracy={row['accuracy']:.4f}",
                (
                    f"{self.evaluation_level}_balanced_accuracy="
                    f"{row['balanced_accuracy']:.4f}"
                ),
                f"{self.evaluation_level}_roc_auc={row['roc_auc']:.4f}",
                f"{self.evaluation_level}_brier_score={row['brier_score']:.4f}",
            ]
            if probabilities.shape[1] == 2:
                parts.extend(
                    [
                        f"pred1={row['predicted_class_1_fraction']:.4f}",
                        f"true1={row['true_class_1_fraction']:.4f}",
                    ]
                )
            print(
                f"[ORACLE held-out user={self.target_subject!r} "
                f"epoch={int(epoch) + 1}{self._calibration_context()} "
                f"trials={self.n_target_trials} threshold={threshold:.4f} "
                f"official={str(is_official).lower()}] "
                + "  ".join(parts),
                flush=True,
            )


class _PeriodicCalibrationEpochLogger(tf.keras.callbacks.Callback):
    """Print calibration progress with subject/shot/fold context."""

    def __init__(
        self,
        *,
        target_subject,
        calibration_shots: int,
        calibration_fold: int,
        total_epochs: int,
        every_n_epochs: int,
    ):
        super().__init__()
        self.target_subject = _python_scalar(target_subject)
        self.calibration_shots = int(calibration_shots)
        self.calibration_fold = int(calibration_fold)
        self.total_epochs = int(total_epochs)
        self.every_n_epochs = int(every_n_epochs)

    def on_epoch_end(self, epoch, logs=None):
        completed_epoch = int(epoch) + 1
        if (
            completed_epoch % self.every_n_epochs != 0
            and completed_epoch != self.total_epochs
        ):
            return
        metric_parts = []
        for name, value in sorted(dict(logs or {}).items()):
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric_value):
                metric_parts.append(f"{name}={numeric_value:.6g}")
        metrics_text = " ".join(metric_parts) or "metrics=unavailable"
        print(
            "[calibration] "
            f"user={self.target_subject} "
            f"shots={self.calibration_shots} "
            f"fold={self.calibration_fold} "
            f"epoch={completed_epoch}/{self.total_epochs} "
            f"{metrics_text}",
            flush=True,
        )
