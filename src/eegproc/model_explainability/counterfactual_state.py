"""State containers used during counterfactual optimization."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf


@dataclass(slots=True)
class ForwardPass:
    reconstruction: tf.Tensor
    embedding: tf.Tensor
    logits: tf.Tensor

    @property
    def probabilities(self) -> tf.Tensor:
        return tf.nn.softmax(self.logits, axis=-1)


@dataclass(slots=True)
class OptimizationHistory:
    steps: list[int] = field(default_factory=list)
    total: list[float] = field(default_factory=list)
    validity: list[float] = field(default_factory=list)
    signal: list[float] = field(default_factory=list)
    target_probability: list[float] = field(default_factory=list)
    predicted_class: list[np.ndarray] = field(default_factory=list)
    gradient_norm: list[float] = field(default_factory=list)
    feature_steps: list[int] = field(default_factory=list)
    counterfactual_features: list[np.ndarray] = field(default_factory=list)

    def add(
        self,
        *,
        step: int,
        losses: dict[str, tf.Tensor],
        target_probability: tf.Tensor,
        predicted_class: tf.Tensor,
        gradient: tf.Tensor,
        reconstruction: tf.Tensor | None = None,
    ) -> None:
        self.steps.append(step)
        self.total.append(float(losses["total"].numpy()))
        self.validity.append(float(losses["validity"].numpy()))
        self.signal.append(
            float(losses["reconstructed_input_proximity"].numpy())
        )
        self.target_probability.append(
            float(tf.reduce_mean(target_probability).numpy())
        )
        self.predicted_class.append(predicted_class.numpy())
        self.gradient_norm.append(
            float(tf.linalg.global_norm([gradient]).numpy())
        )

        if reconstruction is not None:
            self.feature_steps.append(step)
            self.counterfactual_features.append(
                reconstruction.numpy().astype(np.float32, copy=False)
            )

    def to_arrays(self) -> dict[str, np.ndarray]:
        history = {
            "step": np.asarray(self.steps, dtype=np.int32),
            "total_loss": np.asarray(self.total, dtype=np.float32),
            "validity_loss": np.asarray(self.validity, dtype=np.float32),
            "reconstructed_input_proximity_loss": np.asarray(
                self.signal,
                dtype=np.float32,
            ),
            "target_probability": np.asarray(
                self.target_probability,
                dtype=np.float32,
            ),
            "predicted_class": np.asarray(
                self.predicted_class,
                dtype=np.int32,
            ),
            "gradient_norm": np.asarray(
                self.gradient_norm,
                dtype=np.float32,
            ),
            "feature_step": np.asarray(
                self.feature_steps,
                dtype=np.int32,
            ),
        }

        if self.counterfactual_features:
            history["counterfactual_features"] = np.stack(
                self.counterfactual_features,
                axis=0,
            )
        else:
            history["counterfactual_features"] = np.empty(
                (0,),
                dtype=np.float32,
            )

        return history


def scalar_or_array(value: np.ndarray) -> int | float | np.ndarray:
    value = np.asarray(value)
    if value.size == 1:
        return value.reshape(-1)[0].item()
    return value


def predicted_class(logits: tf.Tensor) -> int | np.ndarray:
    prediction = tf.argmax(
        logits,
        axis=-1,
        output_type=tf.int32,
    ).numpy()
    return scalar_or_array(prediction)
