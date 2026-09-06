"""Full-trial latent optimization using the saved SIC BiGRU/VC and decoders.

There is one variable, z_prime. No model is constructed, compiled, fitted, or
calibrated here. All network calls use training=False; the gradient tape
watches only z_prime. Existing model trainability flags are left unchanged.
"""

import math
import time

import numpy as np
import tensorflow as tf

if __package__:
    from .counterfactual_loss import CounterfactualLoss
else:
    from counterfactual_loss import CounterfactualLoss


class CounterfactualOptimizer:
    """Optimize one complete SIC trial, keeping the saved model fixed.

    The model must expose the current SIC v11 feature/decoder interface.
    Its recurrent classifier consumes every timestep of every window, in
    chronological order, followed by its existing VC logits head. The two
    branch feature sequences are split only for their independent decoders.

    Adam performs gradient-based updates to z_prime only. Each optimize()
    call creates fresh Adam state, so trials cannot influence one another.
    By default all max_steps updates are considered. Successful candidates
    are ranked by weighted latent + decoded + physiological proximity;
    if none succeeds, the finite candidate with lowest total loss is returned
    with success=False. This is a best observed iterate, not a global optimum.
    """

    def __init__(
        self,
        model,
        *,
        loss=None,
        learning_rate=0.01,
        max_steps=200,
        gradient_clip_norm=5.0,
        stop_on_success=False,
    ):
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        if (
            isinstance(max_steps, bool)
            or not isinstance(max_steps, (int, np.integer))
            or max_steps < 0
        ):
            raise ValueError("max_steps must be a nonnegative integer.")
        if gradient_clip_norm is not None and (
            not math.isfinite(gradient_clip_norm) or gradient_clip_norm <= 0
        ):
            raise ValueError("gradient_clip_norm must be positive or None.")
        if getattr(model, "classification_level", None) != "trial":
            raise ValueError(
                "A full-trial SIC model is required; window/VAE models are not supported."
            )
        if not getattr(model, "use_decoder", False):
            raise ValueError(
                "This checkpoint has no enabled decoder; supply a SIC model with trained branch decoders."
            )
        self.branches = []
        for name in ("gcn_gru", "bilstm"):
            if getattr(model, f"use_{name}_branch", False):
                if getattr(model, f"{name}_decoder", None) is None:
                    raise ValueError(f"Missing {name} decoder.")
                self.branches.append((name, int(getattr(model, f"{name}_feature_dim"))))
        if not self.branches:
            raise ValueError("At least one active encoder/decoder branch is required.")
        self.model, self.loss = (
            model,
            loss if loss is not None else CounterfactualLoss(),
        )
        self.learning_rate, self.max_steps = float(learning_rate), int(max_steps)
        self.gradient_clip_norm, self.stop_on_success = (
            gradient_clip_norm,
            bool(stop_on_success),
        )

    def _classify(self, latent):
        """Reshape (1,W,T,C) to (1,W*T,C); reuse the saved recurrent/VC head."""
        sequence = tf.reshape(latent, [1, -1, tf.shape(latent)[-1]])
        embedding = self.model.trial_recurrent_classifier(sequence, training=False)
        return tf.cast(self.model.vc_target(embedding, training=False), tf.float32)

    def _decode(self, latent, x):
        """Decode each branch per window, then restore the original trial axes.

        Branch ordering is exactly SIC's concat([gcn_gru, bilstm]). A single
        branch ablation uses its entire feature tensor. The decoder receives
        (W,T,C_branch), never the BiGRU's final state or the fused feature width.
        """
        result, offset = {}, 0
        for name, width in self.branches:
            part = latent[..., offset : offset + width]
            flat = tf.reshape(part, [-1, tf.shape(latent)[2], width])
            decoded = self.model.decode_branch_feature_sequence(name, flat)
            tf.debugging.assert_equal(
                tf.shape(decoded),
                tf.shape(x)[1:],
                message=f"{name} decoder must reconstruct each original window.",
            )
            result[name] = tf.reshape(tf.cast(decoded, tf.float32), tf.shape(x))
            offset += width
        return result

    def _prediction(self, logits, target):
        """Return probabilities and an argmax-plus-confidence success decision."""
        probabilities = tf.nn.softmax(logits, axis=-1).numpy()[0]
        if probabilities.ndim != 1 or not np.isfinite(probabilities).all():
            raise FloatingPointError("Non-finite classification probabilities.")
        predicted = int(np.argmax(probabilities))
        return {
            "probabilities": probabilities.tolist(),
            "predicted_class": predicted,
            "target_probability": float(probabilities[target]),
            "success": bool(
                predicted == target
                and probabilities[target] >= self.loss.target_probability
            ),
        }

    def optimize(self, inputs, *, target_class=None, progress=None):
        """Return scalar history, a summary, and original/counterfactual arrays.

        inputs is one preprocessed trial: (W,T,F) or (1,W,T,F), not a batch
        of independent trials. No normalization, filtering, masking, cropping,
        or padding is performed here. Current SIC training requires equal
        real windows per trial and does not use a classifier padding mask.

        target_class defaults to the opposite ORIGINAL predicted class for
        binary models. Multiclass models require an explicit integer target.
        progress, if supplied, receives one scalar dictionary per finite
        evaluated step. Step 0 is before updates; selected_step may precede
        steps_completed because the best candidate is retained independently.

        Final decoded trials are passed through the FULL saved model again.
        Their target success is distinct from success in latent space. Neither
        success measure demonstrates a causal or physiological EEG intervention.
        """
        started = time.perf_counter()
        x = tf.cast(tf.convert_to_tensor(inputs), tf.float32)
        if x.shape.rank == 3:
            x = x[None, ...]
        if (
            x.shape.rank != 4
            or x.shape[0] != 1
            or any(d is None or d < 1 for d in x.shape)
        ):
            raise ValueError("inputs must be one nonempty trial: (W,T,F) or (1,W,T,F).")
        tf.debugging.assert_all_finite(x, "Original EEG must be finite.")
        features = self.model.get_encoder_features(x)
        z = tf.stop_gradient(tf.cast(features["window_features"], tf.float32))
        tf.debugging.assert_equal(
            tf.shape(z)[:3],
            tf.shape(x)[:3],
            message="SIC must preserve every window and timestep.",
        )
        tf.debugging.assert_equal(
            tf.shape(z)[-1],
            sum(width for _, width in self.branches),
            message="SIC branch feature widths do not match z.",
        )
        tf.debugging.assert_all_finite(z, "Original encoder features must be finite.")
        original_logits = self._classify(z)
        if original_logits.shape.rank != 2 or original_logits.shape[0] != 1:
            raise ValueError("SIC must produce logits shaped (1, n_classes).")
        tf.debugging.assert_all_finite(
            original_logits, "Original logits must be finite."
        )
        tf.debugging.assert_near(
            tf.nn.softmax(original_logits),
            tf.cast(features["probabilities"], tf.float32),
            atol=1e-5,
            rtol=1e-4,
            message="Latent classification does not match the saved model's forward pass.",
        )
        n_classes = int(original_logits.shape[-1])
        original_class = int(tf.argmax(original_logits[0]).numpy())
        if target_class is None:
            if n_classes != 2:
                raise ValueError("Specify target_class for a multiclass model.")
            target_class = 1 - original_class
        if (
            isinstance(target_class, bool)
            or not isinstance(target_class, (int, np.integer))
            or not 0 <= target_class < n_classes
        ):
            raise ValueError(f"target_class must be an integer in [0, {n_classes}).")
        target_class = int(target_class)
        original_prediction = self._prediction(original_logits, target_class)
        original_decoded = self._decode(z, x)
        variable = tf.Variable(z, name="counterfactual_trial_features")
        descent = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        decode = lambda candidate: self._decode(candidate, x)
        history, best_key, best_latent, selected_step = [], None, None, None
        stop_reason, steps_completed = "max_steps", 0

        for step in range(self.max_steps + 1):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(variable)
                logits = self._classify(variable)
                terms, _ = self.loss.central_loss(
                    logits=logits,
                    target_class=target_class,
                    z_prime=variable,
                    z=z,
                    x=x,
                    decoder=decode,
                )
            gradient = tape.gradient(terms["total"], variable)
            if gradient is None:
                raise RuntimeError(
                    "No gradient reached z_prime from the counterfactual objective."
                )
            values = {key: float(value.numpy()) for key, value in terms.items()}
            finite_loss = all(math.isfinite(value) for value in values.values())
            finite_logits = bool(tf.reduce_all(tf.math.is_finite(logits)))
            if not finite_loss or not finite_logits:
                if best_latent is None:
                    raise FloatingPointError(
                        "Non-finite objective at the original trial."
                    )
                stop_reason = "non_finite_loss"
                break
            prediction = self._prediction(logits, target_class)
            proximity = sum(
                values[f"weighted_{name}"]
                for name in ("latent", "decoded", "physiological")
            )
            key = (
                not prediction["success"],
                proximity if prediction["success"] else values["total"],
            )
            if best_key is None or key < best_key:
                best_key, best_latent, selected_step = key, tf.identity(variable), step
            norm = float(tf.linalg.global_norm([gradient]).numpy())
            finite_gradient = math.isfinite(norm) and bool(
                tf.reduce_all(tf.math.is_finite(gradient))
            )
            row = {
                "step": step,
                **values,
                **{k: v for k, v in prediction.items() if k != "probabilities"},
                **{
                    f"probability_{i}": p
                    for i, p in enumerate(prediction["probabilities"])
                },
                "gradient_norm": norm if finite_gradient else None,
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
            if norm == 0:
                stop_reason = "zero_gradient"
                break
            if self.gradient_clip_norm is not None:
                gradient = tf.clip_by_norm(gradient, self.gradient_clip_norm)
            descent.apply_gradients([(gradient, variable)])
            steps_completed += 1

        final_logits = self._classify(best_latent)
        final_terms, decoded = self.loss.central_loss(
            logits=final_logits,
            target_class=target_class,
            z_prime=best_latent,
            z=z,
            x=x,
            decoder=decode,
        )
        latent_prediction = self._prediction(final_logits, target_class)
        arrays = {"x": x.numpy(), "z": z.numpy(), "z_prime": best_latent.numpy()}
        decoded_results = {}
        for name, reconstruction in decoded.items():
            baseline = original_decoded[name]
            arrays[f"x_reconstructed_{name}"] = baseline.numpy()
            arrays[f"x_prime_{name}"] = reconstruction.numpy()
            decoded_results[name] = {
                "original_reconstruction": self._prediction(
                    self.model(baseline, training=False), target_class
                ),
                "counterfactual": self._prediction(
                    self.model(reconstruction, training=False), target_class
                ),
                "original_reconstruction_mse": float(
                    self.loss.latent_distance(baseline, x).numpy()
                ),
                "counterfactual_to_original_mse": float(
                    self.loss.latent_distance(reconstruction, x).numpy()
                ),
                "decoded_change_mse": float(
                    self.loss.latent_distance(reconstruction, baseline).numpy()
                ),
                #VCSC on both decoded signals, so plausibility can be attributed.
                #The decoders do not reconstruct perfectly, so the decoded
                #ORIGINAL already carries some implausibility that has nothing
                #to do with the counterfactual search (observed: 15.4 at step 0
                #on subject 0 trial 0, against a real-trial median of 0.66).
                #Reporting the delta isolates what the optimization itself cost.
                "vcsc_original_reconstruction": float(
                    self.loss.physiological_validity(baseline).numpy()
                ),
                "vcsc_counterfactual": float(
                    self.loss.physiological_validity(reconstruction).numpy()
                ),
                "vcsc_delta": float(
                    (
                        self.loss.physiological_validity(reconstruction)
                        - self.loss.physiological_validity(baseline)
                    ).numpy()
                ),
            }
        return {
            "history": history,
            "summary": {
                "target_class": target_class,
                "required_target_probability": self.loss.target_probability,
                "prediction_rule": "argmax",
                "original": original_prediction,
                "latent_counterfactual": latent_prediction,
                "decoded_trials": decoded_results,
                "selected_losses": {
                    k: float(v.numpy()) for k, v in final_terms.items()
                },
                "selected_step": selected_step,
                "steps_completed": steps_completed,
                "stop_reason": stop_reason,
                "elapsed_seconds": time.perf_counter() - started,
                "physiological_validity": float(
                    final_terms["physiological"].numpy()
                ),
                "physiological_constraint_enforced": self.loss.physiological_weight
                > 0,
            },
            "arrays": arrays,
        }
