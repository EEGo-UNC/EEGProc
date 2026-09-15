"""Gradient counterfactuals over the small CounterfactualAdapter contract."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Mapping

import numpy as np
import tensorflow as tf

from .counterfactual_adapter import CounterfactualAdapter


def _mean_mse(left: tf.Tensor, right: tf.Tensor) -> tf.Tensor:
    left = tf.cast(left, tf.float32)
    right = tf.stop_gradient(tf.cast(right, tf.float32))
    tf.debugging.assert_equal(tf.shape(left), tf.shape(right), message="MSE shapes differ.")
    return tf.reduce_mean(tf.square(left - right))


class ModelAgnosticCounterfactualOptimizer:
    """Optimize adapter state while leaving all model parameters unchanged."""

    def __init__(
        self,
        adapter: CounterfactualAdapter,
        *,
        target_probability: float = 0.8,
        learning_rate: float = 0.01,
        max_steps: int = 200,
        gradient_clip_norm: float | None = 5.0,
        target_weight: float = 1.0,
        state_weight: float | None = None,
        signal_weight: float | None = None,
        constraint_weights: Mapping[str, float] | None = None,
        report_constraints: Iterable[str] = (),
        stop_on_success: bool = False,
    ):
        if not 0 < float(target_probability) < 1:
            raise ValueError("target_probability must be strictly between 0 and 1.")
        if float(learning_rate) <= 0 or not math.isfinite(float(learning_rate)):
            raise ValueError("learning_rate must be finite and positive.")
        if isinstance(max_steps, bool) or int(max_steps) != max_steps or max_steps < 0:
            raise ValueError("max_steps must be a nonnegative integer.")
        if gradient_clip_norm is not None and (
            gradient_clip_norm <= 0 or not math.isfinite(float(gradient_clip_norm))
        ):
            raise ValueError("gradient_clip_norm must be positive or None.")
        self.adapter = adapter
        self.target_probability = float(target_probability)
        self.learning_rate = float(learning_rate)
        self.max_steps = int(max_steps)
        self.gradient_clip_norm = gradient_clip_norm
        self.target_weight = self._weight("target_weight", target_weight, positive=True)
        self.state_weight = self._weight(
            "state_weight",
            adapter.default_state_weight if state_weight is None else state_weight,
        )
        self.signal_weight = self._weight(
            "signal_weight",
            adapter.default_signal_weight if signal_weight is None else signal_weight,
        )
        self.constraint_weights = {
            str(name): self._weight(f"constraint {name!r}", value)
            for name, value in dict(constraint_weights or {}).items()
        }
        self.report_constraints = tuple(dict.fromkeys(str(v) for v in report_constraints))
        self.stop_on_success = bool(stop_on_success)

    @staticmethod
    def _weight(name, value, *, positive=False):
        value = float(value)
        if not math.isfinite(value) or value < 0 or (positive and value == 0):
            qualifier = "positive" if positive else "nonnegative"
            raise ValueError(f"{name} must be finite and {qualifier}.")
        return value

    def _prediction(self, logits: tf.Tensor, target_class: int) -> dict:
        probabilities = tf.nn.softmax(logits, axis=-1).numpy()[0]
        if probabilities.ndim != 1 or not np.isfinite(probabilities).all():
            raise FloatingPointError("Model returned invalid class probabilities.")
        predicted = int(np.argmax(probabilities))
        target_probability = float(probabilities[target_class])
        return {
            "probabilities": probabilities.tolist(),
            "predicted_class": predicted,
            "target_probability": target_probability,
            "success": bool(
                predicted == target_class
                and target_probability >= self.target_probability
            ),
        }

    def _constraints(self, reconstructions, names):
        by_output = self._constraints_by_output(reconstructions, names)
        return {
            name: tf.add_n(list(output_values.values())) / len(output_values)
            for name, output_values in by_output.items()
        }

    def _constraints_by_output(self, reconstructions, names):
        values = {}
        for name in names:
            values[name] = {
                output_name: tf.cast(
                    self.adapter.constraint(name, output_value), tf.float32
                )
                for output_name, output_value in reconstructions.items()
            }
        return values

    def _objective(self, variable, original_state, inputs, target_class, *, baseline_reconstructions=None):
        logits = self.adapter.logits_from_state(variable)
        reconstructions = dict(self.adapter.reconstruct(variable, inputs))
        if not reconstructions:
            raise ValueError("Adapter reconstruct() must return at least one signal.")
        for name, value in reconstructions.items():
            tf.debugging.assert_equal(
                tf.shape(value), tf.shape(inputs),
                message=f"Reconstruction {name!r} must match the input shape.",
            )
        log_probability = tf.nn.log_softmax(logits, axis=-1)[0, target_class]
        target = tf.nn.relu(
            tf.math.log(tf.cast(self.target_probability, logits.dtype))
            - log_probability
        )
        state = _mean_mse(variable, original_state)
        if baseline_reconstructions is None:
            baseline_reconstructions = dict(
                self.adapter.reconstruct(tf.stop_gradient(original_state), inputs)
            )
        if set(reconstructions) != set(baseline_reconstructions):
            raise ValueError("Candidate and original reconstruction paths must match.")
        signal_by_output = {
            name: _mean_mse(value, baseline_reconstructions[name])
            for name, value in reconstructions.items()
        }
        signal = tf.add_n(list(signal_by_output.values())) / len(signal_by_output)
        active_constraint_names = [
            name for name, weight in self.constraint_weights.items() if weight > 0
        ]
        constraints = self._constraints(reconstructions, active_constraint_names)
        weighted_constraints = {
            name: self.constraint_weights[name] * value
            for name, value in constraints.items()
        }
        weighted_target = self.target_weight * target
        weighted_state = self.state_weight * state
        weighted_signal = self.signal_weight * signal
        total = tf.add_n(
            [weighted_target, weighted_state, weighted_signal, *weighted_constraints.values()]
        )
        terms = {
            "total": total,
            "target": target,
            "state": state,
            "signal": signal,
            "weighted_target": weighted_target,
            "weighted_state": weighted_state,
            "weighted_signal": weighted_signal,
            **{f"signal_{name}": value for name, value in signal_by_output.items()},
            **{f"constraint_{name}": value for name, value in constraints.items()},
            **{
                f"weighted_constraint_{name}": value
                for name, value in weighted_constraints.items()
            },
        }
        return logits, reconstructions, terms

    def optimize(
        self,
        inputs,
        *,
        target_class: int | None = None,
        progress: Callable[[dict], None] | None = None,
    ) -> dict:
        started = time.perf_counter()
        x = tf.cast(tf.convert_to_tensor(inputs), tf.float32)
        if x.shape.rank is None or x.shape.rank < 2:
            raise ValueError("inputs must include a batch axis and feature axes.")
        if x.shape[0] != 1:
            raise ValueError("Optimize exactly one trial at a time.")
        tf.debugging.assert_all_finite(x, "Input trial must be finite.")

        original_state = tf.stop_gradient(tf.cast(self.adapter.initial_state(x), tf.float32))
        original_logits = tf.cast(
            self.adapter.logits_from_state(original_state), tf.float32
        )
        if original_logits.shape.rank != 2 or original_logits.shape[0] != 1:
            raise ValueError("Adapter logits must be shaped (1, n_classes).")
        n_classes = int(original_logits.shape[-1])
        original_class = int(tf.argmax(original_logits[0]).numpy())
        if target_class is None:
            if n_classes != 2:
                raise ValueError("Multiclass models require an explicit target_class.")
            target_class = 1 - original_class
        if isinstance(target_class, bool) or not 0 <= int(target_class) < n_classes:
            raise ValueError(f"target_class must be an integer in [0, {n_classes}).")
        target_class = int(target_class)
        original_prediction = self._prediction(original_logits, target_class)
        baseline_reconstructions = {
            name: tf.stop_gradient(value)
            for name, value in self.adapter.reconstruct(original_state, x).items()
        }

        variable = tf.Variable(original_state, name="counterfactual_state")
        descent = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        history = []
        best_key = best_state = selected_step = None
        stop_reason = "max_steps"
        steps_completed = 0

        for step in range(self.max_steps + 1):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(variable)
                logits, _, tensor_terms = self._objective(
                    variable, original_state, x, target_class,
                    baseline_reconstructions=baseline_reconstructions,
                )
            gradient = tape.gradient(tensor_terms["total"], variable)
            if gradient is None:
                raise RuntimeError("No objective gradient reached the adapted state.")
            values = {name: float(value.numpy()) for name, value in tensor_terms.items()}
            if not all(math.isfinite(value) for value in values.values()):
                if best_state is None:
                    raise FloatingPointError("The initial objective is non-finite.")
                stop_reason = "non_finite_loss"
                break
            prediction = self._prediction(logits, target_class)
            proximity = values["weighted_state"] + values["weighted_signal"] + sum(
                values[name]
                for name in values
                if name.startswith("weighted_constraint_")
            )
            key = (
                not prediction["success"],
                proximity if prediction["success"] else values["total"],
            )
            if best_key is None or key < best_key:
                best_key = key
                best_state = tf.identity(variable)
                selected_step = step
            gradient_norm = float(tf.linalg.global_norm([gradient]).numpy())
            finite_gradient = math.isfinite(gradient_norm) and bool(
                tf.reduce_all(tf.math.is_finite(gradient))
            )
            row = {
                "step": step,
                "signal_distance_reference": "original_reconstruction",
                **values,
                **{key: value for key, value in prediction.items() if key != "probabilities"},
                **{
                    f"probability_{index}": value
                    for index, value in enumerate(prediction["probabilities"])
                },
                "gradient_norm": gradient_norm if finite_gradient else None,
            }
            history.append(row)
            if progress is not None:
                progress(dict(row))
            if not finite_gradient:
                stop_reason = "non_finite_gradient"
                break
            if prediction["success"] and (step == 0 or self.stop_on_success):
                stop_reason = "already_satisfied" if step == 0 else "target_reached"
                break
            if step == self.max_steps:
                break
            if gradient_norm == 0:
                stop_reason = "zero_gradient"
                break
            update = gradient
            if self.gradient_clip_norm is not None:
                update = tf.clip_by_norm(update, self.gradient_clip_norm)
            descent.apply_gradients([(update, variable)])
            steps_completed += 1

        final_logits, reconstructions, final_terms = self._objective(
            best_state, original_state, x, target_class,
            baseline_reconstructions=baseline_reconstructions,
        )
        final_prediction = self._prediction(final_logits, target_class)
        all_constraint_names = tuple(
            dict.fromkeys((*self.constraint_weights, *self.report_constraints))
        )
        baseline_constraints = self._constraints_by_output(
            baseline_reconstructions, all_constraint_names
        )
        final_constraints = self._constraints_by_output(
            reconstructions, all_constraint_names
        )
        arrays = {
            "x": x.numpy(),
            "state": original_state.numpy(),
            "state_prime": best_state.numpy(),
        }
        decoded_results = {}
        for name, reconstruction in reconstructions.items():
            baseline = baseline_reconstructions[name]
            arrays[f"x_reconstructed_{name}"] = baseline.numpy()
            arrays[f"x_prime_{name}"] = reconstruction.numpy()
            decoded_results[name] = {
                "original_reconstruction": self._prediction(
                    self.adapter.logits_from_input(baseline), target_class
                ),
                "counterfactual": self._prediction(
                    self.adapter.logits_from_input(reconstruction), target_class
                ),
                "original_reconstruction_mse": float(_mean_mse(baseline, x).numpy()),
                "counterfactual_to_original_mse": float(
                    _mean_mse(reconstruction, x).numpy()
                ),
                "decoded_change_mse": float(_mean_mse(reconstruction, baseline).numpy()),
                "constraints": {
                    constraint_name: {
                        "reference": float(
                            baseline_constraints[constraint_name][name].numpy()
                        ),
                        "counterfactual": float(
                            final_constraints[constraint_name][name].numpy()
                        ),
                        "delta": float(
                            (
                                final_constraints[constraint_name][name]
                                - baseline_constraints[constraint_name][name]
                            ).numpy()
                        ),
                        "weight": self.constraint_weights.get(constraint_name, 0.0),
                    }
                    for constraint_name in all_constraint_names
                },
            }
        summary = {
            "adapter": self.adapter.metadata(),
            "signal_distance_reference": "original_reconstruction",
            "target_class": target_class,
            "required_target_probability": self.target_probability,
            "original": original_prediction,
            "counterfactual": final_prediction,
            "reconstructed_outputs": decoded_results,
            "selected_losses": {
                name: float(value.numpy()) for name, value in final_terms.items()
            },
            "selected_step": selected_step,
            "steps_completed": steps_completed,
            "stop_reason": stop_reason,
            "elapsed_seconds": time.perf_counter() - started,
        }
        return {"history": history, "summary": summary, "arrays": arrays}
