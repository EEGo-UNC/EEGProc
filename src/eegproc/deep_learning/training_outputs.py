from __future__ import annotations
from typing import Mapping

import numpy as np
import tensorflow as tf

from .cross_val import (
    _as_numpy_1d,
    _extract_classifier_output,
    _prediction_diagnostic_summary,
    _to_probabilities,
)

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
    """Return probabilities and any available internal classifier tensors."""
    if hasattr(model, "predict_diagnostics"):
        raw_outputs = model.predict_diagnostics(X, batch_size=batch_size)
    else:
        inputs = tf.convert_to_tensor(X, dtype=tf.float32)
        try:
            raw_outputs = model(
                inputs,
                training=False,
                sample_latent=False,
                include_reconstruction=False,
            )
        except TypeError:
            raw_outputs = model(inputs, training=False)

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
            probabilities = _to_probabilities(_numpy_value(classifier_output))
        outputs["probabilities"] = probabilities
        return outputs

    classifier_output = _extract_classifier_output(raw_outputs)
    return {
        "probabilities": _to_probabilities(_numpy_value(classifier_output)),
    }


class PredictionDiagnostics(tf.keras.callbacks.Callback):
    """Inspect deterministic train/validation predictions during training.

    Only a fixed, approximately class-balanced subset is evaluated, so the
    callback remains inexpensive relative to a full validation pass. It records
    exact probability spread and, when the model exposes ``predict_diagnostics``,
    latent and logit spread as well.
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
        seed: int | None = 42,
    ) -> None:
        super().__init__()
        if every_n_epochs < 1:
            raise ValueError("every_n_epochs must be at least 1.")
        if max_samples < 1:
            raise ValueError("max_samples must be at least 1.")
        if threshold_tolerance < 0.0:
            raise ValueError("threshold_tolerance must be non-negative.")

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
        self.history: list[dict] = []

    def _report_split(
        self,
        split: str,
        X: np.ndarray,
        y: np.ndarray,
        epoch_number: int,
        logs: dict,
    ) -> None:
        internal_outputs = _diagnostic_model_outputs(
            model=self.model,
            X=X,
            batch_size=self.batch_size,
        )
        summary = _prediction_diagnostic_summary(
            probabilities=internal_outputs["probabilities"],
            y_true=y,
            threshold_tolerance=self.threshold_tolerance,
            internal_outputs=internal_outputs,
        )
        row = {
            "fold": None if self.fold_number is None else int(self.fold_number),
            "epoch": int(epoch_number),
            "split": split,
            **summary,
        }
        # Keep diagnostics in the dedicated callback history only. They are
        # intentionally not inserted into Keras logs and not printed separately;
        # CompactEpochLogger owns the human-readable epoch output.
        self.history.append(row)

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        epoch_number = int(epoch) + 1
        if epoch_number % self.every_n_epochs != 0:
            return
        if logs is None:
            logs = {}

        self._report_split(
            split="train",
            X=self.X_train,
            y=self.y_train,
            epoch_number=epoch_number,
            logs=logs,
        )
        if self.X_val is not None and self.y_val is not None:
            self._report_split(
                split="validation",
                X=self.X_val,
                y=self.y_val,
                epoch_number=epoch_number,
                logs=logs,
            )


class CompactEpochLogger(tf.keras.callbacks.Callback):
    """Print each epoch as compact metric, class-balance, and loss rows.

    Keras' built-in ``verbose=2`` logger places every metric on one very long
    line. This callback groups the same epoch logs into readable categories and
    leaves ``history.history`` unchanged.
    """

    _PREFERRED_METRIC_ORDER = (
        "window_accuracy",
        "val_window_accuracy",
        "window_balanced_accuracy",
        "val_window_balanced_accuracy",
        "decoder_r2",
        "val_decoder_r2",
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
        absolute = abs(value)
        if absolute != 0.0 and (absolute < 1e-4 or absolute >= 1e4):
            return f"{value:.3e}"
        return f"{value:.4f}"

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

    def on_epoch_end(
        self,
        epoch: int,
        logs: dict | None = None,
    ) -> None:
        logs = logs or {}
        epoch_number = int(epoch) + 1
        prefix = self._prefix(epoch_number)

        performance = self._format_performance(logs)
        if performance:
            print(f"{prefix} METRICS | {performance}", flush=True)

        distributions = [
            value
            for value in (
                self._format_distribution(logs, validation=False),
                self._format_distribution(logs, validation=True),
            )
            if value is not None
        ]
        if distributions:
            print(
                f"{prefix} CLASSES | " + " | ".join(distributions),
                flush=True,
            )

        train_loss = self._format_loss_split(logs, validation=False)
        if train_loss:
            print(f"{prefix} LOSS | {train_loss}", flush=True)
        validation_loss = self._format_loss_split(logs, validation=True)
        if validation_loss:
            print(f"{prefix} LOSS | {validation_loss}", flush=True)