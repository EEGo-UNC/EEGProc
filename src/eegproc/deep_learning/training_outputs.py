from __future__ import annotations
import sys
from typing import Mapping, Sequence

import numpy as np
import tensorflow as tf

from .cross_val import (
    _CLASSIFICATION_METRICS,
    HeldOutUserOracleMetrics as _CrossValHeldOutUserOracleMetrics,
    _aggregate_window_probabilities_by_trial,
    _as_numpy_1d,
    _classification_metrics,
    _decoder_reconstruction_scores,
    _direct_trial_aggregation,
    _extract_classifier_output,
    _is_trial_tensor,
    _normalize_decision_thresholds,
    _predict_labels,
    _predict_probabilities,
    _probability_log_loss,
    _prediction_diagnostic_summary,
    _to_probabilities,
)


_DIAGNOSTIC_CLASSIFICATION_METRICS = (
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

_DECODER_METRIC_NAMES = (
    "decoder_r2",
    "gcn_gru_decoder_r2",
    "bilstm_decoder_r2",
    "reconstruction_loss",
    "gcn_gru_reconstruction_loss",
    "bilstm_reconstruction_loss",
)


def _finite_scalar(value) -> float | None:
    """Return one finite scalar metric value, or ``None`` otherwise."""
    if value is None:
        return None
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if array.ndim != 0:
        return None
    result = float(array)
    return result if np.isfinite(result) else None


def _format_metric_value(value: float) -> str:
    """Format one scalar consistently across every epoch-output callback."""
    absolute = abs(value)
    if absolute != 0.0 and (absolute < 1e-4 or absolute >= 1e4):
        return f"{value:.3e}"
    return f"{value:.4f}"

def _numpy_value(value):
    """Convert tensors and array-like values to numpy arrays."""
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)

def _stratified_diagnostic_indices(
    y: np.ndarray,
    max_samples: int,
    seed: int | None,
) -> np.ndarray:
    """Choose a deterministic approximately class-balanced diagnostic subset."""
    y_ids = _as_numpy_1d(y).astype(np.int64)
    if max_samples < 1:
        raise ValueError("max_samples must be at least 1.")
    if len(y_ids) <= max_samples:
        return np.arange(len(y_ids), dtype=np.int64)

    rng = np.random.default_rng(seed)
    classes = np.unique(y_ids)
    per_class = max(1, max_samples // max(1, len(classes)))
    selected: list[int] = []

    for class_id in classes:
        class_indices = np.where(y_ids == class_id)[0]
        take = min(per_class, len(class_indices))
        selected.extend(
            rng.choice(class_indices, size=take, replace=False).tolist()
        )

    selected_array = np.asarray(sorted(set(selected)), dtype=np.int64)
    remaining_slots = max_samples - len(selected_array)
    if remaining_slots > 0:
        remaining = np.setdiff1d(
            np.arange(len(y_ids), dtype=np.int64),
            selected_array,
            assume_unique=False,
        )
        if len(remaining):
            extra = rng.choice(
                remaining,
                size=min(remaining_slots, len(remaining)),
                replace=False,
            )
            selected_array = np.sort(
                np.concatenate([selected_array, extra.astype(np.int64)])
            )

    return selected_array[:max_samples]

def _diagnostic_model_outputs(
    model: tf.keras.Model,
    X: np.ndarray,
    batch_size: int | None,
) -> dict[str, np.ndarray]:
    """Return diagnostic tensors using memory-bounded inference batches.

    A grouped trial has shape ``(windows, timesteps, features)``. Consequently,
    a rank-4 diagnostic array can expand from a seemingly small outer batch to
    thousands of EEG windows inside the SIC encoder. Process those arrays one
    complete trial at a time. This still evaluates every selected trial and
    preserves its full ordered window sequence; only the device scheduling is
    changed.
    """
    X_array = np.asarray(X)
    if len(X_array) == 0:
        raise ValueError("Diagnostic inputs must contain at least one sample.")
    if batch_size is not None and int(batch_size) < 1:
        raise ValueError("Diagnostic batch_size must be at least 1 or None.")

    requested_batch_size = (
        len(X_array) if batch_size is None else int(batch_size)
    )
    # Rank-4 SIC inputs are grouped trials, not ordinary independent windows.
    # One trial can already contain enough flattened windows to occupy most of
    # the GPU, so never combine multiple trials in a diagnostic forward pass.
    effective_batch_size = 1 if X_array.ndim == 4 else requested_batch_size

    def normalize(raw_outputs) -> dict[str, np.ndarray]:
        if isinstance(raw_outputs, Mapping):
            outputs = {
                str(key): _numpy_value(value)
                for key, value in raw_outputs.items()
                if value is not None
            }
            if "probabilities" in outputs:
                probabilities = _to_probabilities(outputs["probabilities"])
            elif "logits" in outputs:
                probabilities = _to_probabilities(outputs["logits"])
            else:
                classifier_output = _extract_classifier_output(raw_outputs)
                probabilities = _to_probabilities(
                    _numpy_value(classifier_output)
                )
            outputs["probabilities"] = probabilities
            return outputs

        classifier_output = _extract_classifier_output(raw_outputs)
        return {
            "probabilities": _to_probabilities(
                _numpy_value(classifier_output)
            ),
        }

    output_batches: list[dict[str, np.ndarray]] = []
    for start in range(0, len(X_array), effective_batch_size):
        stop = min(start + effective_batch_size, len(X_array))
        X_batch = X_array[start:stop]
        if hasattr(model, "predict_diagnostics"):
            raw_outputs = model.predict_diagnostics(
                X_batch,
                batch_size=effective_batch_size,
            )
        else:
            inputs = tf.convert_to_tensor(X_batch, dtype=tf.float32)
            try:
                raw_outputs = model(
                    inputs,
                    training=False,
                    sample_latent=False,
                    include_reconstruction=False,
                )
            except TypeError:
                raw_outputs = model(inputs, training=False)
        output_batches.append(normalize(raw_outputs))

    expected_keys = set(output_batches[0])
    for index, outputs in enumerate(output_batches[1:], start=1):
        if set(outputs) != expected_keys:
            raise ValueError(
                "Diagnostic model output keys changed between batches: "
                f"first={sorted(expected_keys)}, batch_{index}="
                f"{sorted(outputs)}."
            )

    combined: dict[str, np.ndarray] = {}
    for key in output_batches[0]:
        values = [outputs[key] for outputs in output_batches]
        combined[key] = (
            np.stack(values, axis=0)
            if values[0].ndim == 0
            else np.concatenate(values, axis=0)
        )

    if len(combined["probabilities"]) != len(X_array):
        raise ValueError(
            "Diagnostic probabilities do not align with the selected inputs: "
            f"probabilities={len(combined['probabilities'])}, "
            f"inputs={len(X_array)}."
        )
    return combined


class HeldOutUserOracleMetrics(_CrossValHeldOutUserOracleMetrics):
    """Emit one complete reporting-only oracle-prediction row per epoch.

    The parent callback previously printed decoder and classification values on
    separate rows. This replacement preserves its initialization contract and
    reporting-only history while computing the complete classification suite,
    probability log loss, and both branch-specific reconstruction suites in one
    memory-bounded pass per output type.
    """

    _METRICS = _DIAGNOSTIC_CLASSIFICATION_METRICS
    _training_outputs_consolidated = True

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        epoch_number = int(epoch) + 1
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

        probability_loss = _probability_log_loss(
            y_true=y_true,
            probabilities=probabilities,
        )
        official_row: dict | None = None

        # Preserve reporting-only diagnostic-threshold history, but print only
        # the official threshold so every epoch has exactly one oracle row.
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
                "epoch": epoch_number,
                "evaluation_level": self.evaluation_level,
                "decision_threshold": float(threshold),
                "official_decision_threshold": self.decision_threshold,
                "is_official_decision_threshold": is_official,
                "loss": float(probability_loss),
                **{name: float(value) for name, value in scores.items()},
                **decoder_scores,
            }
            if probabilities.shape[1] == 2:
                row["predicted_class_1_fraction"] = float(np.mean(y_pred == 1))
                row["true_class_1_fraction"] = float(np.mean(y_true == 1))
            self.history.append(row)
            if is_official:
                official_row = row

        if official_row is None:
            raise RuntimeError("The official oracle decision threshold was not evaluated.")

        level = self.evaluation_level
        parts = [
            f"{level}_loss={_format_metric_value(official_row['loss'])}",
            f"{level}_accuracy={_format_metric_value(official_row['accuracy'])}",
            (
                f"{level}_balanced_accuracy="
                f"{_format_metric_value(official_row['balanced_accuracy'])}"
            ),
            f"{level}_roc_auc={_format_metric_value(official_row['roc_auc'])}",
            f"{level}_macro_f1={_format_metric_value(official_row['macro_f1'])}",
            f"{level}_f1={_format_metric_value(official_row['f1'])}",
            f"{level}_precision={_format_metric_value(official_row['precision'])}",
            f"{level}_recall={_format_metric_value(official_row['recall'])}",
            (
                f"{level}_macro_precision="
                f"{_format_metric_value(official_row['macro_precision'])}"
            ),
            (
                f"{level}_macro_recall="
                f"{_format_metric_value(official_row['macro_recall'])}"
            ),
            (
                f"{level}_brier_score="
                f"{_format_metric_value(official_row['brier_score'])}"
            ),
            f"{level}_ece={_format_metric_value(official_row['ece'])}",
        ]
        if probabilities.shape[1] == 2:
            parts.extend(
                [
                    "pred1="
                    f"{_format_metric_value(official_row['predicted_class_1_fraction'])}",
                    "true1="
                    f"{_format_metric_value(official_row['true_class_1_fraction'])}",
                ]
            )
        parts.extend(
            f"{name}={_format_metric_value(official_row[name])}"
            for name in _DECODER_METRIC_NAMES
            if name in official_row
        )
        print(
            f"[ORACLE PREDICTION held-out user={self.target_subject!r} "
            f"epoch={epoch_number} level={level} "
            f"threshold={self.decision_threshold:.4f}] "
            + " | ".join(parts),
            flush=True,
        )


class PredictionDiagnostics(tf.keras.callbacks.Callback):
    """Print one complete deterministic training-diagnostics row per epoch.

    Only a fixed, approximately class-balanced subset is evaluated, so the
    callback remains inexpensive relative to a full validation pass. The row
    combines the existing Keras epoch metrics/losses with exact probability
    metrics on that fixed subset, including ROC-AUC and macro F1. Decoder values
    come from the epoch-wide Keras logs when available; a memory-bounded
    inference fallback is used only for models that expose a decoder without
    tracking its metrics.
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        fold_number: int | None = None,
        batch_size: int | None = None,
        every_n_epochs: int = 1,
        max_samples: int = 256,
        threshold_tolerance: float = 0.01,
        reported_metric: str = "accuracy",
        decision_threshold: float = 0.5,
        decision_thresholds: Sequence[float] | None = None,
        ece_bins: int = 15,
        seed: int | None = 42,
    ) -> None:
        super().__init__()
        if every_n_epochs < 1:
            raise ValueError("every_n_epochs must be at least 1.")
        if max_samples < 1:
            raise ValueError("max_samples must be at least 1.")
        if threshold_tolerance < 0.0:
            raise ValueError("threshold_tolerance must be non-negative.")
        reported_metric = str(reported_metric).strip().lower()
        if reported_metric not in _CLASSIFICATION_METRICS:
            raise ValueError(
                f"Unsupported prediction diagnostic metric {reported_metric!r}. "
                f"Supported metrics: {sorted(_CLASSIFICATION_METRICS)}"
            )
        official_threshold = float(decision_threshold)
        if not 0.0 < official_threshold < 1.0:
            raise ValueError("decision_threshold must lie strictly between 0 and 1.")
        raw_thresholds = (
            (official_threshold,)
            if decision_thresholds is None
            else (*decision_thresholds, official_threshold)
        )
        normalized_thresholds = _normalize_decision_thresholds(raw_thresholds)
        if int(ece_bins) < 2:
            raise ValueError("ece_bins must be at least 2.")

        train_indices = _stratified_diagnostic_indices(
            y_train,
            max_samples=max_samples,
            seed=seed,
        )
        self.X_train = np.asarray(X_train)[train_indices]
        self.y_train = np.asarray(y_train)[train_indices]

        self.X_val = None
        self.y_val = None
        if X_val is not None and y_val is not None and len(X_val):
            validation_seed = None if seed is None else int(seed) + 1
            val_indices = _stratified_diagnostic_indices(
                y_val,
                max_samples=max_samples,
                seed=validation_seed,
            )
            self.X_val = np.asarray(X_val)[val_indices]
            self.y_val = np.asarray(y_val)[val_indices]

        self.fold_number = fold_number
        self.batch_size = batch_size
        self.every_n_epochs = int(every_n_epochs)
        self.threshold_tolerance = float(threshold_tolerance)
        self.reported_metric = reported_metric
        self.decision_threshold = official_threshold
        self.decision_thresholds = (
            official_threshold,
            *(
                threshold
                for threshold in normalized_thresholds
                if threshold != official_threshold
            ),
        )
        self.ece_bins = int(ece_bins)
        self.history: list[dict] = []

    def on_train_begin(self, logs: dict | None = None) -> None:
        # CompactEpochLogger checks this marker so the two callbacks never emit
        # duplicate training-diagnostics rows for the same fit.
        self._diagnostic_owner_id = id(self)
        setattr(
            self.model,
            "_consolidated_prediction_diagnostics_owner",
            self._diagnostic_owner_id,
        )
        print(
            "[DIAGNOSTICS] Official evaluation threshold="
            f"{self.decision_threshold:.4f}; reporting-only thresholds="
            f"{list(self.decision_thresholds)}.",
            flush=True,
        )

    def on_train_end(self, logs: dict | None = None) -> None:
        if getattr(
            self.model,
            "_consolidated_prediction_diagnostics_owner",
            None,
        ) == getattr(self, "_diagnostic_owner_id", None):
            delattr(self.model, "_consolidated_prediction_diagnostics_owner")

    def _decoder_scores_for_split(
        self,
        split: str,
        X: np.ndarray,
        logs: Mapping[str, object],
    ) -> dict[str, float]:
        log_prefix = "val_" if split == "validation" else ""
        scores: dict[str, float] = {}
        for name in _DECODER_METRIC_NAMES:
            value = _finite_scalar(logs.get(f"{log_prefix}{name}"))
            if value is not None:
                scores[name] = value

        if scores or not bool(getattr(self.model, "use_decoder", False)):
            return scores
        return _decoder_reconstruction_scores(
            model=self.model,
            X=X,
            batch_size=self.batch_size,
        )

    def _report_split(
        self,
        split: str,
        X: np.ndarray,
        y: np.ndarray,
        epoch_number: int,
        logs: dict,
    ) -> dict:
        internal_outputs = _diagnostic_model_outputs(
            model=self.model,
            X=X,
            batch_size=self.batch_size,
        )
        probabilities = internal_outputs["probabilities"]
        y_ids = _as_numpy_1d(y).astype(np.int64)
        probability_loss = _probability_log_loss(
            y_true=y_ids,
            probabilities=probabilities,
        )
        decoder_scores = self._decoder_scores_for_split(
            split=split,
            X=X,
            logs=logs,
        )
        official_row: dict | None = None

        for threshold in self.decision_thresholds:
            y_pred = _predict_labels(
                probabilities,
                decision_threshold=threshold,
            )
            scores = _classification_metrics(
                y_true=y_ids,
                y_pred=y_pred,
                probabilities=probabilities,
                metrics=_DIAGNOSTIC_CLASSIFICATION_METRICS,
                n_classes=probabilities.shape[1],
                ece_bins=self.ece_bins,
            )
            summary = _prediction_diagnostic_summary(
                probabilities=probabilities,
                y_true=y_ids,
                threshold_tolerance=self.threshold_tolerance,
                internal_outputs=internal_outputs,
                reported_metric=self.reported_metric,
                decision_threshold=threshold,
                ece_bins=self.ece_bins,
            )
            row = {
                "fold": (
                    None if self.fold_number is None else int(self.fold_number)
                ),
                "epoch": int(epoch_number),
                "split": split,
                "decision_threshold": float(threshold),
                "official_decision_threshold": self.decision_threshold,
                "is_official_decision_threshold": bool(
                    threshold == self.decision_threshold
                ),
                "loss": float(probability_loss),
                **{name: float(value) for name, value in scores.items()},
                **summary,
                **decoder_scores,
            }
            # Keep diagnostics in the dedicated callback history only. They are
            # intentionally not inserted into Keras logs, so they cannot affect
            # training callbacks or checkpoint selection.
            self.history.append(row)
            if threshold == self.decision_threshold:
                official_row = row

        if official_row is None:
            raise RuntimeError(
                "The official training-diagnostic decision threshold was not "
                "evaluated."
            )
        return official_row

    def _format_prediction_row(self, row: Mapping[str, object]) -> str:
        split = "val" if row["split"] == "validation" else "train"
        level = str(getattr(self.model, "classification_level", "sample"))
        prefix = f"{split}_{level}"
        parts = [
            f"{prefix}_n={int(row['n_samples'])}",
            f"{prefix}_loss={_format_metric_value(float(row['loss']))}",
            f"{prefix}_accuracy={_format_metric_value(float(row['accuracy']))}",
            (
                f"{prefix}_balanced_accuracy="
                f"{_format_metric_value(float(row['balanced_accuracy']))}"
            ),
            f"{prefix}_roc_auc={_format_metric_value(float(row['roc_auc']))}",
            f"{prefix}_macro_f1={_format_metric_value(float(row['macro_f1']))}",
            f"{prefix}_f1={_format_metric_value(float(row['f1']))}",
            f"{prefix}_precision={_format_metric_value(float(row['precision']))}",
            f"{prefix}_recall={_format_metric_value(float(row['recall']))}",
            (
                f"{prefix}_macro_precision="
                f"{_format_metric_value(float(row['macro_precision']))}"
            ),
            (
                f"{prefix}_macro_recall="
                f"{_format_metric_value(float(row['macro_recall']))}"
            ),
            (
                f"{prefix}_brier_score="
                f"{_format_metric_value(float(row['brier_score']))}"
            ),
            f"{prefix}_ece={_format_metric_value(float(row['ece']))}",
            (
                f"{prefix}_confidence_mean="
                f"{_format_metric_value(float(row['confidence_mean']))}"
            ),
            (
                f"{prefix}_confidence_std="
                f"{_format_metric_value(float(row['confidence_std']))}"
            ),
        ]
        if "predicted_class_1_fraction" in row:
            parts.extend(
                [
                    f"{split}_pred1="
                    f"{_format_metric_value(float(row['predicted_class_1_fraction']))}",
                    f"{split}_true1="
                    f"{_format_metric_value(float(row['true_class_1_fraction']))}",
                ]
            )
        parts.extend(
            f"{split}_{name}={_format_metric_value(float(row[name]))}"
            for name in _DECODER_METRIC_NAMES
            if name in row
        )
        return " ".join(parts)

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        epoch_number = int(epoch) + 1
        if logs is None:
            logs = {}

        # The requested output contract is one row every epoch. Keep accepting
        # every_n_epochs for API compatibility, but intentionally do not skip
        # epochs here.
        rows = [self._report_split(
            split="train",
            X=self.X_train,
            y=self.y_train,
            epoch_number=epoch_number,
            logs=logs,
        )]
        if self.X_val is not None and self.y_val is not None:
            rows.append(self._report_split(
                split="validation",
                X=self.X_val,
                y=self.y_val,
                epoch_number=epoch_number,
                logs=logs,
            ))

        formatter = CompactEpochLogger(fold_number=self.fold_number)
        formatter.set_model(self.model)
        formatter.set_params(self.params)
        sections = []
        epoch_summary = formatter._format_epoch_summary(logs)
        if epoch_summary:
            sections.append(epoch_summary)
        sections.append(
            "PREDICTIONS[" + " | ".join(
                self._format_prediction_row(row) for row in rows
            ) + "]"
        )
        print(
            f"{formatter._prefix(epoch_number)} TRAINING DIAGNOSTICS | "
            + " | ".join(sections),
            flush=True,
        )


class CompactEpochLogger(tf.keras.callbacks.Callback):
    """Print one consolidated training-diagnostics row per epoch.

    Keras' built-in ``verbose=2`` logger places every metric on one very long
    line. This callback retains the readable metric/class/loss groupings without
    emitting several separate rows and leaves ``history.history`` unchanged.
    """

    _PREFERRED_METRIC_ORDER = (
        "window_accuracy",
        "val_window_accuracy",
        "window_balanced_accuracy",
        "val_window_balanced_accuracy",
        "decoder_r2",
        "val_decoder_r2",
        "gcn_gru_decoder_r2",
        "val_gcn_gru_decoder_r2",
        "bilstm_decoder_r2",
        "val_bilstm_decoder_r2",
        "accuracy",
        "val_accuracy",
        "trial_f1",
        "val_trial_f1",
        "trial_balanced_accuracy",
        "val_trial_balanced_accuracy",
        "balanced_accuracy",
        "val_balanced_accuracy",
        "subject_accuracy",
        "val_subject_accuracy",
        "precision",
        "val_precision",
        "recall",
        "val_recall",
        "f1",
        "val_f1",
        "roc_auc",
        "val_roc_auc",
        "learning_rate",
        "lr",
    )

    _KNOWN_LOSS_NAMES = frozenset(
        {
            "loss",
            "base_total_loss",
            "regularization_loss",
            "autoencoder_loss",
            "reconstruction_loss",
            "gcn_gru_reconstruction_loss",
            "bilstm_reconstruction_loss",
            "kl_loss",
            "weighted_kl_loss",
            "vc_loss",
            "vc_cross_entropy",
            "weighted_vc_cross_entropy",
            "vc_latent_kl",
            "weighted_vc_latent_kl",
            "vc_class_prior_kl",
            "weighted_vc_class_prior_kl",
            "vc_discriminator_kl",
            "weighted_vc_discriminator_kl",
            "vc_discriminator_loss",
            "subject_loss",
            "weighted_subject_loss",
            "trial_loss",
        }
    )

    def __init__(
        self,
        fold_number: int | None = None,
        context: str | None = None,
    ) -> None:
        super().__init__()
        self.fold_number = fold_number
        self.context = context

    @staticmethod
    def _float_value(value) -> float | None:
        if value is None:
            return None
        if hasattr(value, "numpy"):
            value = value.numpy()
        array = np.asarray(value)
        if array.ndim != 0:
            return None
        result = float(array)
        return result if np.isfinite(result) else None

    @staticmethod
    def _format_value(value: float) -> str:
        return _format_metric_value(value)

    @staticmethod
    def _base_name(name: str) -> str:
        return name[4:] if name.startswith("val_") else name

    @classmethod
    def _is_class_fraction(cls, name: str) -> bool:
        base = cls._base_name(name)
        return (
            base.startswith("predicted_class_")
            or base.startswith("true_class_")
        ) and base.endswith("_fraction")

    @classmethod
    def _is_loss_like(cls, name: str) -> bool:
        base = cls._base_name(name)
        return (
            base in cls._KNOWN_LOSS_NAMES
            or base.endswith("_loss")
            or base.endswith("_kl")
            or "cross_entropy" in base
            or base.startswith("weighted_")
        )

    def _prefix(self, epoch_number: int) -> str:
        total_epochs = int(self.params.get("epochs", epoch_number))
        parts: list[str] = []
        if self.fold_number is not None:
            parts.append(f"Fold {int(self.fold_number)}")
        if self.context:
            parts.append(str(self.context))
        parts.append(f"Epoch {epoch_number}/{total_epochs}")
        return "[" + "][".join(parts) + "]"

    def _log_value(
        self,
        logs: Mapping[str, object],
        name: str,
    ) -> float | None:
        return self._float_value(logs.get(name))

    def _model_weight(self, name: str) -> float | None:
        return self._float_value(getattr(self.model, name, None))

    def _format_performance(self, logs: Mapping[str, object]) -> str | None:
        metric_names = [
            name
            for name in logs
            if not self._is_class_fraction(name)
            and not self._is_loss_like(name)
        ]
        ordered_names: list[str] = []
        for name in self._PREFERRED_METRIC_ORDER:
            if name in metric_names:
                ordered_names.append(name)
        ordered_names.extend(
            sorted(name for name in metric_names if name not in ordered_names)
        )

        parts = []
        for name in ordered_names:
            value = self._log_value(logs, name)
            if value is not None:
                parts.append(f"{name}={self._format_value(value)}")
        return " | ".join(parts) if parts else None

    def _format_distribution(
        self,
        logs: Mapping[str, object],
        validation: bool,
    ) -> str | None:
        """Print exact class-1 prediction and target fractions for the epoch."""
        prefix = "val_" if validation else ""
        pred1 = self._log_value(
            logs,
            f"{prefix}predicted_class_1_fraction",
        )
        true1 = self._log_value(
            logs,
            f"{prefix}true_class_1_fraction",
        )
        if pred1 is None and true1 is None:
            return None

        split = "val" if validation else "train"
        parts = []
        if pred1 is not None:
            parts.append(f"pred1={self._format_value(pred1)}")
        if true1 is not None:
            parts.append(f"true1={self._format_value(true1)}")
        return f"{split} " + " ".join(parts)

    def _weighted_contribution(
        self,
        logs: Mapping[str, object],
        prefix: str,
        raw_name: str,
        weight_name: str,
    ) -> tuple[float | None, float | None, float | None]:
        raw = self._log_value(logs, f"{prefix}{raw_name}")
        weight = self._model_weight(weight_name)
        contribution = None
        if raw is not None and weight is not None:
            contribution = raw * weight
        return contribution, raw, weight

    def _format_loss_split(
        self,
        logs: Mapping[str, object],
        validation: bool,
    ) -> str | None:
        prefix = "val_" if validation else ""
        split = "val" if validation else "train"
        handled: set[str] = set()
        parts: list[str] = []

        def read(name: str) -> float | None:
            handled.add(f"{prefix}{name}")
            return self._log_value(logs, f"{prefix}{name}")

        total = read("loss")
        base_total = read("base_total_loss")
        regularization = read("regularization_loss")
        if total is not None:
            parts.append(f"total={self._format_value(total)}")
        if base_total is not None:
            parts.append(f"base={self._format_value(base_total)}")
        if regularization is not None:
            parts.append(f"reg={self._format_value(regularization)}")

        ae_contribution, ae_raw, ae_weight = self._weighted_contribution(
            logs, prefix, "autoencoder_loss", "ae_loss_weight"
        )
        handled.add(f"{prefix}autoencoder_loss")
        reconstruction = read("reconstruction_loss")
        gcn_gru_reconstruction = read("gcn_gru_reconstruction_loss")
        bilstm_reconstruction = read("bilstm_reconstruction_loss")
        raw_kl = read("kl_loss")
        weighted_kl = read("weighted_kl_loss")
        if ae_raw is not None:
            ae_head = (
                ae_contribution if ae_contribution is not None else ae_raw
            )
            ae_details = [f"raw={self._format_value(ae_raw)}"]
            if ae_weight is not None:
                ae_details.append(f"w={self._format_value(ae_weight)}")
            if reconstruction is not None:
                ae_details.append(f"recon={self._format_value(reconstruction)}")
            if raw_kl is not None:
                ae_details.append(f"KL={self._format_value(raw_kl)}")
            if weighted_kl is not None:
                ae_details.append(f"wKL={self._format_value(weighted_kl)}")
            parts.append(
                f"AE={self._format_value(ae_head)}[" + ",".join(ae_details) + "]"
            )

        decoder_weight = self._model_weight("reconstruction_loss_weight")
        if reconstruction is not None and decoder_weight is not None:
            decoder_contribution = reconstruction * decoder_weight
            decoder_details = [
                f"raw={self._format_value(reconstruction)}",
                f"w={self._format_value(decoder_weight)}",
            ]
            if gcn_gru_reconstruction is not None:
                decoder_details.append(
                    "gcn_gru="
                    f"{self._format_value(gcn_gru_reconstruction)}"
                )
            if bilstm_reconstruction is not None:
                decoder_details.append(
                    "bilstm="
                    f"{self._format_value(bilstm_reconstruction)}"
                )
            parts.append(
                f"DECODER={self._format_value(decoder_contribution)}["
                + ",".join(decoder_details)
                + "]"
            )

        vc_contribution, vc_raw, vc_weight = self._weighted_contribution(
            logs, prefix, "vc_loss", "vc_loss_weight"
        )
        handled.add(f"{prefix}vc_loss")
        vc_terms = (
            ("weighted_vc_cross_entropy", "CE"),
            ("weighted_vc_latent_kl", "latent"),
            ("weighted_vc_class_prior_kl", "prior"),
            ("weighted_vc_discriminator_kl", "disc"),
        )
        vc_details: list[str] = []
        if vc_raw is not None:
            vc_details.append(f"raw={self._format_value(vc_raw)}")
            if vc_weight is not None:
                vc_details.append(f"w={self._format_value(vc_weight)}")
        for key, label in vc_terms:
            value = read(key)
            if value is not None:
                vc_details.append(f"{label}={self._format_value(value)}")
        # Mark raw diagnostic VC terms as handled so they do not appear again.
        for key in (
            "vc_cross_entropy",
            "vc_latent_kl",
            "vc_class_prior_kl",
            "vc_discriminator_kl",
            "vc_discriminator_loss",
        ):
            handled.add(f"{prefix}{key}")
        if vc_raw is not None or vc_details:
            vc_head = vc_contribution if vc_contribution is not None else vc_raw
            if vc_head is None:
                vc_head = sum(
                    value
                    for key, _ in vc_terms
                    if (value := self._log_value(logs, f"{prefix}{key}"))
                    is not None
                )
            parts.append(
                f"VC={self._format_value(vc_head)}[" + ",".join(vc_details) + "]"
            )

        subject_raw = read("subject_loss")
        weighted_subject = read("weighted_subject_loss")
        if weighted_subject is not None:
            subject_text = f"subject={self._format_value(weighted_subject)}"
            if subject_raw is not None and not np.isclose(
                subject_raw, weighted_subject
            ):
                subject_text += f"[raw={self._format_value(subject_raw)}]"
            parts.append(subject_text)
        elif subject_raw is not None:
            parts.append(f"subject={self._format_value(subject_raw)}")

        trial_loss = read("trial_loss")
        if trial_loss is not None:
            parts.append(f"trial={self._format_value(trial_loss)}")

        extras: list[str] = []
        for name in sorted(logs):
            if name in handled or not name.startswith(prefix):
                continue
            # Do not let the train row consume validation-prefixed values.
            if not validation and name.startswith("val_"):
                continue
            if self._is_loss_like(name):
                value = self._log_value(logs, name)
                if value is not None:
                    display_name = name[len(prefix):] if prefix else name
                    extras.append(
                        f"{display_name}={self._format_value(value)}"
                    )
        if extras:
            parts.append("extra[" + ",".join(extras) + "]")

        return f"{split} " + " | ".join(parts) if parts else None

    def _format_epoch_summary(
        self,
        logs: Mapping[str, object],
    ) -> str | None:
        sections: list[str] = []

        performance = self._format_performance(logs)
        if performance:
            sections.append(f"METRICS[{performance}]")

        distributions = [
            value
            for value in (
                self._format_distribution(logs, validation=False),
                self._format_distribution(logs, validation=True),
            )
            if value is not None
        ]
        if distributions:
            sections.append("CLASSES[" + " | ".join(distributions) + "]")

        losses = [
            value
            for value in (
                self._format_loss_split(logs, validation=False),
                self._format_loss_split(logs, validation=True),
            )
            if value is not None
        ]
        if losses:
            sections.append("LOSSES[" + " | ".join(losses) + "]")
        return " | ".join(sections) if sections else None

    def on_epoch_end(
        self,
        epoch: int,
        logs: dict | None = None,
    ) -> None:
        logs = logs or {}
        epoch_number = int(epoch) + 1
        if getattr(
            self.model,
            "_consolidated_prediction_diagnostics_owner",
            None,
        ) is not None:
            return

        summary = self._format_epoch_summary(logs)
        if summary:
            print(
                f"{self._prefix(epoch_number)} TRAINING DIAGNOSTICS | {summary}",
                flush=True,
            )


# cross_val imports this module lazily immediately before it creates its oracle
# callback. Register the consolidated implementation there without requiring a
# source change to cross_val.py, which keeps the existing public construction
# path and callback ordering intact.
_cross_val_module = sys.modules.get(_CrossValHeldOutUserOracleMetrics.__module__)
if _cross_val_module is not None:
    _cross_val_module.HeldOutUserOracleMetrics = HeldOutUserOracleMetrics
