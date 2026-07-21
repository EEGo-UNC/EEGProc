"""Counterfactual optimization in reconstructed EEG space."""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
import tensorflow as tf

from .counterfactual_losses import (
    CounterfactualLossWeights,
    counterfactual_objective,
    decoded_change_distance,
    normalize_target_classes,
    reconstructed_input_proximity_loss,
    target_probabilities,
)
from .counterfactual_state import (
    ForwardPass,
    OptimizationHistory,
    predicted_class,
    scalar_or_array,
)


LOGGER = logging.getLogger(__name__)
DistanceMetric = Literal["mse", "mae", "rmse"]


class CounterfactualOptimizer:
    """Optimize a latent sequence using reconstructed-input proximity."""

    def __init__(
        self,
        *,
        model: tf.keras.Model,
        learning_rate: float = 1e-2,
        max_steps: int = 500,
        validity_weight: float = 1.0,
        signal_proximity_weight: float = 0.10,
        target_probability: float = 0.80,
        seed: int | None = 42,
        verbose: int = 1,
        gradient_clip_norm: float | None = 5.0,
        signal_metric: DistanceMetric = "mse",
        feature_log_interval: int = 1,
        stop_on_success: bool = False,
    ) -> None:
        self._validate_config(
            model,
            learning_rate,
            max_steps,
            target_probability,
            verbose,
            gradient_clip_norm,
            signal_metric,
            feature_log_interval,
        )
        self.model = model
        self.learning_rate = float(learning_rate)
        self.max_steps = int(max_steps)
        self.target_probability = float(target_probability)
        self.seed = seed
        self.verbose = int(verbose)
        self.gradient_clip_norm = gradient_clip_norm
        self.signal_metric = signal_metric
        self.feature_log_interval = int(feature_log_interval)
        self.stop_on_success = bool(stop_on_success)
        self.loss_weights = CounterfactualLossWeights(
            validity=float(validity_weight),
            signal_proximity=float(signal_proximity_weight),
        )

    def optimize(
        self,
        *,
        inputs: tf.Tensor | np.ndarray,
        target_class: int | tf.Tensor,
    ) -> dict[str, Any]:
        self._set_seed()
        inputs = self._prepare_inputs(inputs)

        original_latent = self._encode(inputs)
        original = self._forward(original_latent)
        self._check_reconstruction_shape(inputs, original.reconstruction)
        targets = self._prepare_targets(target_class, original.logits)

        latent = tf.Variable(
            original_latent,
            trainable=True,
            name="counterfactual_latent",
        )
        optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        history = OptimizationHistory()

        best_latent = None
        best_signal = np.inf
        last_finite = tf.identity(latent)
        stop_reason = "maximum_steps_reached"

        for step in range(self.max_steps + 1):
            with tf.GradientTape() as tape:
                current = self._forward(latent)
                losses = self._losses(current, inputs, targets)

            gradient = tape.gradient(losses["total"], latent)
            if gradient is None:
                raise RuntimeError(
                    "No gradient reached the counterfactual latent."
                )

            target_probability = target_probabilities(
                current.logits,
                targets,
            )
            prediction = tf.argmax(
                current.probabilities,
                axis=-1,
                output_type=tf.int32,
            )
            success = self._is_success(target_probability)

            history.add(
                step=step,
                losses=losses,
                target_probability=target_probability,
                predicted_class=prediction,
                gradient=gradient,
                reconstruction=(
                    current.reconstruction
                    if self._should_log_features(step)
                    else None
                ),
            )
            self._log(step, history, success)

            if not self._finite(losses["total"]):
                latent.assign(last_finite)
                stop_reason = "non_finite_loss"
                break
            if not self._finite(gradient):
                latent.assign(last_finite)
                stop_reason = "non_finite_gradient"
                break

            last_finite = tf.identity(latent)
            signal = float(
                losses["reconstructed_input_proximity"].numpy()
            )

            if success and signal < best_signal:
                best_signal = signal
                best_latent = tf.identity(latent)

            if success and self.stop_on_success:
                stop_reason = "target_probability_reached"
                break
            if step == self.max_steps:
                break

            optimizer.apply_gradients(
                [(self._clip(gradient), latent)]
            )

        if best_latent is not None:
            latent.assign(best_latent)
            if stop_reason == "maximum_steps_reached":
                stop_reason = "best_successful_iterate_selected"

        return self._result(
            inputs=inputs,
            targets=targets,
            original=original,
            latent=latent,
            stop_reason=stop_reason,
            steps_completed=history.steps[-1],
            history=history,
        )

    def _encode(self, inputs: tf.Tensor) -> tf.Tensor:
        outputs = self.model(
            inputs,
            training=False,
            sample_latent=False,
        )
        if not isinstance(outputs, dict) or "z_mean" not in outputs:
            raise TypeError("Joint model output must contain z_mean.")
        return tf.stop_gradient(
            tf.cast(outputs["z_mean"], inputs.dtype)
        )

    def _forward(self, latent: tf.Tensor) -> ForwardPass:
        embedding = self.model.classification_model(
            latent,
            training=False,
        )
        return ForwardPass(
            reconstruction=self.model.decoder(
                latent,
                training=False,
            ),
            embedding=embedding,
            logits=self.model.variational_classifier(
                embedding,
                training=False,
            ),
        )

    def _losses(
        self,
        forward: ForwardPass,
        inputs: tf.Tensor,
        targets: tf.Tensor,
    ) -> dict[str, tf.Tensor]:
        return counterfactual_objective(
            logits=forward.logits,
            target_class=targets,
            counterfactual_reconstruction=forward.reconstruction,
            original_features=inputs,
            weights=self.loss_weights,
            target_probability=self.target_probability,
            signal_metric=self.signal_metric,
        )

    def _result(
        self,
        *,
        inputs: tf.Tensor,
        targets: tf.Tensor,
        original: ForwardPass,
        latent: tf.Variable,
        stop_reason: str,
        steps_completed: int,
        history: OptimizationHistory,
    ) -> dict[str, Any]:
        counterfactual = self._forward(latent)
        final_losses = self._losses(
            counterfactual,
            inputs,
            targets,
        )
        original_target = target_probabilities(
            original.logits,
            targets,
        )
        counterfactual_target = target_probabilities(
            counterfactual.logits,
            targets,
        )
        success_mask = (
            counterfactual_target >= self.target_probability
        )

        return {
            "mode": "deterministic_reconstructed_space",
            "success": bool(tf.reduce_all(success_mask).numpy()),
            "success_mask": success_mask.numpy(),
            "stop_reason": stop_reason,
            "steps_completed": steps_completed,
            "target_class": scalar_or_array(targets.numpy()),
            "original_predicted_class": predicted_class(
                original.logits
            ),
            "counterfactual_predicted_class": predicted_class(
                counterfactual.logits
            ),
            "original_target_probability": scalar_or_array(
                original_target.numpy()
            ),
            "counterfactual_target_probability": scalar_or_array(
                counterfactual_target.numpy()
            ),
            "original_probabilities": original.probabilities.numpy(),
            "counterfactual_probabilities": (
                counterfactual.probabilities.numpy()
            ),
            "counterfactual_latent": latent.numpy(),
            "original_reconstruction": original.reconstruction.numpy(),
            "counterfactual_reconstruction": (
                counterfactual.reconstruction.numpy()
            ),
            "counterfactual_minus_input": (
                counterfactual.reconstruction - inputs
            ).numpy(),
            "original_reconstruction_distance_to_input": self._distance(
                original.reconstruction,
                inputs,
            ),
            "counterfactual_reconstruction_distance_to_input": (
                self._distance(counterfactual.reconstruction, inputs)
            ),
            "reconstruction_change": float(
                decoded_change_distance(
                    counterfactual_reconstruction=(
                        counterfactual.reconstruction
                    ),
                    original_reconstruction=original.reconstruction,
                    metric=self.signal_metric,
                ).numpy()
            ),
            "final_losses": {
                name: float(value.numpy())
                for name, value in final_losses.items()
            },
            "history": history.to_arrays(),
        }

    def _distance(
        self,
        reconstruction: tf.Tensor,
        inputs: tf.Tensor,
    ) -> float:
        return float(
            reconstructed_input_proximity_loss(
                counterfactual_reconstruction=reconstruction,
                original_features=inputs,
                metric=self.signal_metric,
            ).numpy()
        )

    def _prepare_targets(
        self,
        target_class: int | tf.Tensor,
        logits: tf.Tensor,
    ) -> tf.Tensor:
        targets = normalize_target_classes(
            target_class,
            tf.shape(logits)[0],
        )
        n_classes = int(tf.shape(logits)[-1].numpy())
        values = targets.numpy()
        if np.any(values < 0) or np.any(values >= n_classes):
            raise ValueError(
                f"target_class must be in [0, {n_classes - 1}]."
            )
        return targets

    def _is_success(
        self,
        target_probability: tf.Tensor,
    ) -> bool:
        return bool(
            tf.reduce_all(
                target_probability >= self.target_probability
            ).numpy()
        )

    def _clip(self, gradient: tf.Tensor) -> tf.Tensor:
        if self.gradient_clip_norm is None:
            return gradient
        [gradient], _ = tf.clip_by_global_norm(
            [gradient],
            self.gradient_clip_norm,
        )
        return gradient

    def _log(
        self,
        step: int,
        history: OptimizationHistory,
        success: bool,
    ) -> None:
        if self.verbose == 0:
            return

        interval = max(1, self.max_steps // 10)
        if (
            self.verbose == 1
            and step not in (0, self.max_steps)
            and step % interval != 0
            and not success
        ):
            return

        LOGGER.info(
            "step=%d loss=%.6f target_p=%.5f signal=%.6f success=%s",
            step,
            history.total[-1],
            history.target_probability[-1],
            history.signal[-1],
            success,
        )

    def _should_log_features(self, step: int) -> bool:
        return (
            self.feature_log_interval > 0
            and step % self.feature_log_interval == 0
        )

    def _set_seed(self) -> None:
        if self.seed is not None:
            tf.keras.utils.set_random_seed(self.seed)
            np.random.seed(self.seed)

    @staticmethod
    def _prepare_inputs(
        inputs: tf.Tensor | np.ndarray,
    ) -> tf.Tensor:
        inputs = tf.convert_to_tensor(inputs)
        if not inputs.dtype.is_floating:
            inputs = tf.cast(inputs, tf.float32)
        if inputs.shape.rank != 3:
            raise ValueError(
                "inputs must have shape (batch, timesteps, features)."
            )
        return inputs

    @staticmethod
    def _check_reconstruction_shape(
        inputs: tf.Tensor,
        reconstruction: tf.Tensor,
    ) -> None:
        tf.debugging.assert_equal(
            tf.shape(inputs),
            tf.shape(reconstruction),
            message="Decoder output and input must have the same shape.",
        )

    @staticmethod
    def _finite(value: tf.Tensor) -> bool:
        return bool(tf.reduce_all(tf.math.is_finite(value)).numpy())

    @staticmethod
    def _validate_config(
        model: tf.keras.Model,
        learning_rate: float,
        max_steps: int,
        target_probability: float,
        verbose: int,
        gradient_clip_norm: float | None,
        signal_metric: str,
        feature_log_interval: int,
    ) -> None:
        required = (
            "decoder",
            "classification_model",
            "variational_classifier",
        )
        missing = [name for name in required if not hasattr(model, name)]
        if missing:
            raise TypeError(
                f"Model is missing required components: {missing}."
            )
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1.")
        if not 0 < target_probability < 1:
            raise ValueError("target_probability must be between 0 and 1.")
        if verbose not in (0, 1, 2):
            raise ValueError("verbose must be 0, 1, or 2.")
        if gradient_clip_norm is not None and gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive or None.")
        if signal_metric not in ("mse", "mae", "rmse"):
            raise ValueError(
                "signal_metric must be 'mse', 'mae', or 'rmse'."
            )
        if feature_log_interval < 0:
            raise ValueError(
                "feature_log_interval must be zero or positive."
            )
