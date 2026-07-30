"""Counterfactual optimization in reconstructed EEG space.

Window-level models optimize one latent sequence per EEG window. Trial-level
models optimize the complete ordered latent tensor for one or more
subject-trials: the decoder still reconstructs each window independently while
the classifier pools each window latent and applies its BiLSTM across windows.
"""

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
ClassificationLevel = Literal["window", "trial"]


class CounterfactualOptimizer:
    """Optimize window or subject-trial latent means with a frozen model."""

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
        stop_on_success: bool = True,
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
        self.classification_level: ClassificationLevel = self._model_level(model)
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
        window_mask: tf.Tensor | np.ndarray | None = None,
        true_class: int | tf.Tensor | None = None,
    ) -> dict[str, Any]:
        """Generate a counterfactual for windows or complete subject-trials.

        Parameters
        ----------
        inputs:
            Window mode: ``(batch, timesteps, features)``.
            Trial mode: ``(batch, windows, timesteps, features)``.
        target_class:
            Scalar target or one target per classification sample. Trial mode
            emits one classification sample per subject-trial.
        window_mask:
            Optional ``(batch, windows)`` Boolean mask. Required only when the
            trial tensor contains padding that cannot be identified by zeros.
        true_class:
            Optional true class used only for reported classification metrics.
        """
        self._set_seed()
        inputs = self._prepare_inputs(inputs)
        window_mask = self._prepare_window_mask(inputs, window_mask)

        original_latent = self._encode(inputs)
        original = self._forward(original_latent, window_mask)
        self._check_reconstruction_shape(inputs, original.reconstruction)
        targets = self._prepare_targets(target_class, original.logits)
        true_targets = (
            None
            if true_class is None
            else self._prepare_targets(true_class, original.logits)
        )

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
                current = self._forward(latent, window_mask)
                losses = self._losses(
                    current,
                    inputs,
                    targets,
                    window_mask,
                )

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
            true_targets=true_targets,
            original=original,
            latent=latent,
            window_mask=window_mask,
            stop_reason=stop_reason,
            steps_completed=history.steps[-1],
            history=history,
        )

    def _encode(self, inputs: tf.Tensor) -> tf.Tensor:
        try:
            outputs = self.model(
                inputs,
                training=False,
                sample_latent=False,
                include_reconstruction=False,
                include_subject_adversarial=False,
            )
        except TypeError:
            try:
                outputs = self.model(
                    inputs,
                    training=False,
                    sample_latent=False,
                )
            except TypeError:
                outputs = self.model(inputs, training=False)

        if not isinstance(outputs, dict) or "z_mean" not in outputs:
            raise TypeError("Joint model output must contain z_mean.")
        latent = tf.cast(outputs["z_mean"], inputs.dtype)
        expected_rank = 3 if self.classification_level == "window" else 4
        if latent.shape.rank != expected_rank:
            raise ValueError(
                f"{self.classification_level}-level z_mean must have rank "
                f"{expected_rank}; received {latent.shape}."
            )
        return tf.stop_gradient(latent)

    def _forward(
        self,
        latent: tf.Tensor,
        window_mask: tf.Tensor | None,
    ) -> ForwardPass:
        if self.classification_level == "window":
            classification_sequence = latent
            embedding = self.model.classification_model(
                classification_sequence,
                training=False,
            )
            reconstruction = self.model.decoder(
                latent,
                training=False,
            )
        else:
            if window_mask is None:
                raise RuntimeError("Trial optimization requires a window mask.")
            # The training model classifies posterior-mean embeddings obtained
            # by averaging each window across its latent-time axis.
            window_embeddings = tf.reduce_mean(latent, axis=2)
            mask_float = tf.cast(
                window_mask[..., tf.newaxis],
                window_embeddings.dtype,
            )
            classification_sequence = window_embeddings * mask_float
            try:
                embedding = self.model.classification_model(
                    classification_sequence,
                    training=False,
                    mask=window_mask,
                )
            except TypeError:
                embedding = self.model.classification_model(
                    classification_sequence,
                    training=False,
                )

            latent_shape = tf.shape(latent)
            flat_latent = tf.reshape(
                latent,
                [
                    latent_shape[0] * latent_shape[1],
                    latent_shape[2],
                    latent_shape[3],
                ],
            )
            flat_reconstruction = self.model.decoder(
                flat_latent,
                training=False,
            )
            reconstruction_shape = tf.shape(flat_reconstruction)
            reconstruction = tf.reshape(
                flat_reconstruction,
                [
                    latent_shape[0],
                    latent_shape[1],
                    reconstruction_shape[1],
                    reconstruction_shape[2],
                ],
            )

        logits = self.model.variational_classifier(
            embedding,
            training=False,
        )
        return ForwardPass(
            reconstruction=reconstruction,
            embedding=embedding,
            logits=logits,
        )

    def _losses(
        self,
        forward: ForwardPass,
        inputs: tf.Tensor,
        targets: tf.Tensor,
        window_mask: tf.Tensor | None,
    ) -> dict[str, tf.Tensor]:
        return counterfactual_objective(
            logits=forward.logits,
            target_class=targets,
            counterfactual_reconstruction=forward.reconstruction,
            original_features=inputs,
            weights=self.loss_weights,
            target_probability=self.target_probability,
            signal_metric=self.signal_metric,
            sample_mask=window_mask,
        )

    def _result(
        self,
        *,
        inputs: tf.Tensor,
        targets: tf.Tensor,
        true_targets: tf.Tensor | None,
        original: ForwardPass,
        latent: tf.Variable,
        window_mask: tf.Tensor | None,
        stop_reason: str,
        steps_completed: int,
        history: OptimizationHistory,
    ) -> dict[str, Any]:
        counterfactual = self._forward(latent, window_mask)
        final_losses = self._losses(
            counterfactual,
            inputs,
            targets,
            window_mask,
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

        result: dict[str, Any] = {
            "mode": (
                "deterministic_subject_trial_reconstructed_space"
                if self.classification_level == "trial"
                else "deterministic_window_reconstructed_space"
            ),
            "classification_level": self.classification_level,
            "stop_on_success": self.stop_on_success,
            "success": bool(tf.reduce_all(success_mask).numpy()),
            "success_mask": success_mask.numpy(),
            "stop_reason": stop_reason,
            "steps_completed": steps_completed,
            "target_probability_threshold": self.target_probability,
            "target_class": scalar_or_array(targets.numpy()),
            "original_predicted_class": predicted_class(original.logits),
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
                window_mask,
            ),
            "counterfactual_reconstruction_distance_to_input": self._distance(
                counterfactual.reconstruction,
                inputs,
                window_mask,
            ),
            "reconstruction_change": float(
                decoded_change_distance(
                    counterfactual_reconstruction=(
                        counterfactual.reconstruction
                    ),
                    original_reconstruction=original.reconstruction,
                    metric=self.signal_metric,
                    sample_mask=window_mask,
                ).numpy()
            ),
            "final_losses": {
                name: float(value.numpy())
                for name, value in final_losses.items()
            },
            "history": history.to_arrays(),
        }

        if self.classification_level == "trial":
            if window_mask is None:
                raise RuntimeError("Trial result requires a window mask.")
            result.update(
                {
                    "n_trials": int(tf.shape(inputs)[0].numpy()),
                    "n_windows_per_trial": int(tf.shape(inputs)[1].numpy()),
                    "valid_windows_per_trial": tf.reduce_sum(
                        tf.cast(window_mask, tf.int32),
                        axis=1,
                    ).numpy(),
                    "window_mask": window_mask.numpy(),
                    "per_window_original_distance_to_input": (
                        self._per_window_distance(
                            original.reconstruction,
                            inputs,
                            window_mask,
                        )
                    ),
                    "per_window_counterfactual_distance_to_input": (
                        self._per_window_distance(
                            counterfactual.reconstruction,
                            inputs,
                            window_mask,
                        )
                    ),
                    "per_window_reconstruction_change": (
                        self._per_window_distance(
                            counterfactual.reconstruction,
                            original.reconstruction,
                            window_mask,
                        )
                    ),
                }
            )

        classification_metrics = self._classification_metrics(
            original=original,
            counterfactual=counterfactual,
            targets=targets,
            true_targets=true_targets,
        )
        result["classification_metrics"] = classification_metrics
        if self.classification_level == "trial":
            result["trial_level_metrics"] = classification_metrics
        return result

    def _classification_metrics(
        self,
        *,
        original: ForwardPass,
        counterfactual: ForwardPass,
        targets: tf.Tensor,
        true_targets: tf.Tensor | None,
    ) -> dict[str, Any]:
        prefix = "trial" if self.classification_level == "trial" else "window"
        original_probabilities = original.probabilities
        counterfactual_probabilities = counterfactual.probabilities
        original_prediction = tf.argmax(
            original_probabilities,
            axis=-1,
            output_type=tf.int32,
        )
        counterfactual_prediction = tf.argmax(
            counterfactual_probabilities,
            axis=-1,
            output_type=tf.int32,
        )
        original_target = target_probabilities(original.logits, targets)
        counterfactual_target = target_probabilities(
            counterfactual.logits,
            targets,
        )

        metrics: dict[str, Any] = {
            f"{prefix}_count": int(tf.shape(original.logits)[0].numpy()),
            f"{prefix}_prediction_flipped": (
                original_prediction != counterfactual_prediction
            ).numpy(),
            f"{prefix}_target_reached": (
                counterfactual_target >= self.target_probability
            ).numpy(),
            f"{prefix}_original_confidence": tf.reduce_max(
                original_probabilities,
                axis=-1,
            ).numpy(),
            f"{prefix}_counterfactual_confidence": tf.reduce_max(
                counterfactual_probabilities,
                axis=-1,
            ).numpy(),
            f"{prefix}_original_target_probability": original_target.numpy(),
            f"{prefix}_counterfactual_target_probability": (
                counterfactual_target.numpy()
            ),
            f"{prefix}_target_probability_gain": (
                counterfactual_target - original_target
            ).numpy(),
        }

        if true_targets is not None:
            indices = tf.stack(
                [
                    tf.range(tf.shape(true_targets)[0], dtype=tf.int32),
                    true_targets,
                ],
                axis=1,
            )
            metrics.update(
                {
                    f"{prefix}_true_class": true_targets.numpy(),
                    f"{prefix}_original_accuracy": tf.cast(
                        original_prediction == true_targets,
                        tf.float32,
                    ).numpy(),
                    f"{prefix}_counterfactual_accuracy": tf.cast(
                        counterfactual_prediction == true_targets,
                        tf.float32,
                    ).numpy(),
                    f"{prefix}_original_true_class_probability": tf.gather_nd(
                        original_probabilities,
                        indices,
                    ).numpy(),
                    f"{prefix}_counterfactual_true_class_probability": (
                        tf.gather_nd(
                            counterfactual_probabilities,
                            indices,
                        ).numpy()
                    ),
                }
            )
        return metrics

    def _distance(
        self,
        reconstruction: tf.Tensor,
        inputs: tf.Tensor,
        window_mask: tf.Tensor | None,
    ) -> float:
        return float(
            reconstructed_input_proximity_loss(
                counterfactual_reconstruction=reconstruction,
                original_features=inputs,
                metric=self.signal_metric,
                sample_mask=window_mask,
            ).numpy()
        )

    def _per_window_distance(
        self,
        candidate: tf.Tensor,
        reference: tf.Tensor,
        window_mask: tf.Tensor,
    ) -> np.ndarray:
        difference = candidate - reference
        if self.signal_metric == "mse":
            distance = tf.reduce_mean(tf.square(difference), axis=(2, 3))
        elif self.signal_metric == "mae":
            distance = tf.reduce_mean(tf.abs(difference), axis=(2, 3))
        else:
            distance = tf.sqrt(
                tf.reduce_mean(tf.square(difference), axis=(2, 3))
                + tf.cast(1e-8, difference.dtype)
            )
        nan = tf.cast(np.nan, distance.dtype)
        return tf.where(window_mask, distance, nan).numpy()

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

    def _prepare_inputs(
        self,
        inputs: tf.Tensor | np.ndarray,
    ) -> tf.Tensor:
        inputs = tf.convert_to_tensor(inputs)
        if not inputs.dtype.is_floating:
            inputs = tf.cast(inputs, tf.float32)
        expected_rank = 3 if self.classification_level == "window" else 4
        if inputs.shape.rank != expected_rank:
            expected = (
                "(batch, timesteps, features)"
                if expected_rank == 3
                else "(batch, windows, timesteps, features)"
            )
            raise ValueError(
                f"{self.classification_level}-level inputs must have shape "
                f"{expected}; received {inputs.shape}."
            )
        return inputs

    def _prepare_window_mask(
        self,
        inputs: tf.Tensor,
        window_mask: tf.Tensor | np.ndarray | None,
    ) -> tf.Tensor | None:
        if self.classification_level == "window":
            if window_mask is not None:
                raise ValueError("window_mask is only valid in trial mode.")
            return None

        if window_mask is None:
            mask = tf.reduce_any(
                tf.not_equal(inputs, tf.zeros((), dtype=inputs.dtype)),
                axis=(2, 3),
            )
        else:
            mask = tf.cast(tf.convert_to_tensor(window_mask), tf.bool)
        tf.debugging.assert_equal(
            tf.shape(mask),
            tf.shape(inputs)[:2],
            message="window_mask must have shape (batch, windows).",
        )
        tf.debugging.assert_positive(
            tf.reduce_sum(tf.cast(mask, tf.int32), axis=1),
            message="Every subject-trial must contain at least one valid window.",
        )
        return mask

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
            "level=%s step=%d loss=%.6f target_p=%.5f signal=%.6f "
            "success=%s",
            self.classification_level,
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
    def _model_level(model: tf.keras.Model) -> ClassificationLevel:
        level = str(getattr(model, "classification_level", "window")).lower()
        if level not in {"window", "trial"}:
            raise ValueError(
                "model.classification_level must be 'window' or 'trial'; "
                f"received {level!r}."
            )
        return level  # type: ignore[return-value]

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
