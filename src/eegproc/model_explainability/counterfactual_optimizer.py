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

    The model must expose the current SIC feature/decoder interface.
    Its recurrent classifier consumes every timestep of every window, in
    chronological order, followed by its existing VC logits head. In branch
    mode, the two feature sequences are decoded independently. In joint mode,
    both branch decoders run and the saved model's learned convex fusion
    produces the sole decoded reconstruction used by the objective.

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
        learning_rate_decay=1.0,
        target_loss_component="confidence",
        max_steps=200,
        gradient_clip_norm=5.0,
        stop_on_success=False,
        decoder_mode="branches",
        typicality=None,
        typicality_weight=0.0,
        require_decoded_success=False,
    ):
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        if (
            not math.isfinite(learning_rate_decay)
            or not 0 < learning_rate_decay <= 1
        ):
            raise ValueError("learning_rate_decay must be finite and in (0, 1].")
        target_loss_component = str(target_loss_component).strip().lower()
        if target_loss_component not in {"confidence", "focal", "vc", "focal_vc"}:
            raise ValueError(
                "target_loss_component must be 'confidence', 'focal', 'vc', "
                "or 'focal_vc'."
            )
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
        decoder_mode = str(decoder_mode).strip().lower()
        if decoder_mode not in {"branches", "joint"}:
            raise ValueError("decoder_mode must be 'branches' or 'joint'.")
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
        if decoder_mode == "joint":
            if {name for name, _ in self.branches} != {"gcn_gru", "bilstm"}:
                raise ValueError(
                    "Joint decoder mode requires active GCN-GRU and BiLSTM branches."
                )
            if not getattr(model, "use_joint_reconstruction", False):
                raise ValueError(
                    "Joint decoder mode requires a model trained with joint reconstruction."
                )
            if getattr(model, "joint_reconstruction_fusion", None) is None:
                raise ValueError("Missing joint reconstruction fusion layer.")
        if not math.isfinite(typicality_weight) or typicality_weight < 0:
            raise ValueError("typicality_weight must be finite and nonnegative.")
        if typicality_weight > 0 and typicality is None:
            raise ValueError("A fitted source-only typicality region is required.")
        self.typicality = typicality
        self.typicality_weight = float(typicality_weight)
        self.require_decoded_success = bool(require_decoded_success)
        self.model, self.loss = (
            model,
            loss if loss is not None else CounterfactualLoss(),
        )
        self.decoder_mode = decoder_mode
        self.decoded_names = (
            ("joint",)
            if self.decoder_mode == "joint"
            else tuple(name for name, _ in self.branches)
        )
        self.learning_rate = float(learning_rate)
        self.learning_rate_decay = float(learning_rate_decay)
        self.target_loss_component = target_loss_component
        self.max_steps = int(max_steps)
        self.gradient_clip_norm, self.stop_on_success = (
            gradient_clip_norm,
            bool(stop_on_success),
        )

    def _classification_state(self, latent):
        """Return the frozen SIC classifier embedding and its sole logits."""
        sequence = tf.reshape(latent, [1, -1, tf.shape(latent)[-1]])
        embedding = self.model.trial_recurrent_classifier(sequence, training=False)
        logits = tf.cast(self.model.vc_target(embedding, training=False), tf.float32)
        return tf.cast(embedding, tf.float32), logits

    def _classify(self, latent):
        """Reshape the latent trial and return its frozen classifier logits."""
        return self._classification_state(latent)[1]

    def _target_components(self, embedding, logits, target_class):
        """Decompose the frozen classifier objective before differentiation."""
        labels = tf.fill([tf.shape(logits)[0]], tf.cast(target_class, tf.int32))
        vc = self.model.vc_target.vc_loss_components(
            mh=embedding,
            y=labels,
            alpha=float(getattr(self.model, "vc_alpha", 1.0)),
            beta=float(getattr(self.model, "vc_beta", 0.0)),
            gamma=float(getattr(self.model, "vc_gamma", 0.0)),
            lambda_=float(getattr(self.model, "vc_lambda", 0.0)),
            logits=logits,
        )
        confidence = self.loss.target_loss(logits, target_class)
        focal = tf.cast(vc["weighted_focal_loss"], logits.dtype)
        vc_latent = tf.cast(vc["weighted_latent_posterior_kl"], logits.dtype)
        vc_discriminator = tf.cast(vc["weighted_discriminator_kl"], logits.dtype)
        vc_prior = tf.cast(vc["weighted_class_prior_kl"], logits.dtype)
        vc_only = vc_latent + vc_discriminator + vc_prior
        focal_vc = focal + vc_only
        selected = {
            "confidence": confidence,
            "focal": focal,
            "vc": vc_only,
            "focal_vc": focal_vc,
        }[self.target_loss_component]
        return selected, {
            "target_confidence_component": confidence,
            "target_focal_component": focal,
            "target_vc_component": vc_only,
            "target_vc_latent_posterior_kl": vc_latent,
            "target_vc_discriminator_kl": vc_discriminator,
            "target_vc_class_prior_kl": vc_prior,
        }

    def _decode_branches(self, latent, x):
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

    def _decode(self, latent, x):
        """Return the reconstruction paths selected for the objective.

        Joint mode still evaluates both independent decoders, then applies the
        checkpoint's frozen learned fusion weight. Because the gradient tape
        watches only the counterfactual latent variable, gradients flow through
        the fusion to both latent branches without changing model parameters.
        """
        branches = self._decode_branches(latent, x)
        if self.decoder_mode == "branches":
            return branches
        joint = self.model.joint_reconstruction_fusion(
            [branches["gcn_gru"], branches["bilstm"]]
        )
        tf.debugging.assert_equal(
            tf.shape(joint),
            tf.shape(x),
            message="Joint decoder must reconstruct the original trial shape.",
        )
        return {"joint": tf.cast(joint, tf.float32)}

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

    def _objective(self, **kwargs):
        terms, decoded = self.loss.central_loss(**kwargs)
        if self.typicality is not None:
            if kwargs["target_class"] != self.typicality.metadata["target_class"]:
                raise ValueError("Typicality target class does not match optimization target.")
            self._current_typicality_sequence = self.typicality.sequence(kwargs["z_prime"])
            discrepancy = self.typicality.discrepancy(kwargs["z_prime"], sequence=self._current_typicality_sequence)
            penalty = tf.square(tf.nn.relu(discrepancy - self.typicality.tau))
            terms.update(typicality=penalty, discrepancy=discrepancy,
                         weighted_typicality=self.typicality_weight * penalty)
            terms["total"] = terms["total"] + terms["weighted_typicality"]
        return terms, decoded

    def optimize(self, inputs, *, target_class=None, progress=None, state_progress=None):
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

        state_progress(row, arrays) optionally receives every finite iterate,
        full decoded arrays, raw gradient and Adam state for streamed archival.
        With require_decoded_success, selection/stopping also requires decoded
        argmax target validity on every selected output. An active typicality
        penalty adds D <= tau to selection/stopping. The probability threshold
        remains a separate, stricter optimization criterion.

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
        original_decoded = {
            name: tf.stop_gradient(value) for name, value in self._decode(z, x).items()
        }
        original_typicality_sequence = self.typicality.sequence(z) if self.typicality is not None else None
        variable = tf.Variable(z, name="counterfactual_trial_features")
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=1,
            decay_rate=self.learning_rate_decay,
            staircase=True,
        )
        descent = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        decode = lambda candidate: self._decode(candidate, x)
        history, best_key, best_latent, selected_step = [], None, None, None
        stop_reason, steps_completed = "max_steps", 0

        for step in range(self.max_steps + 1):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(variable)
                embedding, logits = self._classification_state(variable)
                target_loss, target_components = self._target_components(
                    embedding, logits, target_class
                )
                terms, current_decoded = self._objective(
                    logits=logits,
                    target_class=target_class,
                    z_prime=variable,
                    z=z,
                    x=x,
                    decoder=decode,
                    reference_reconstructions=original_decoded,
                    target_loss_override=target_loss,
                    target_components=target_components,
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
            typical = self.typicality is None or values["discrepancy"] <= self.typicality.tau
            decoded_step = {}
            if self.require_decoded_success or state_progress is not None:
                decoded_step = {
                    name: self._prediction(self.model(value, training=False), target_class)
                    for name, value in current_decoded.items()
                }
            decoded_valid = all(p["predicted_class"] == target_class for p in decoded_step.values())
            optimization_success = (
                prediction["success"]
                and (self.typicality_weight == 0 or typical)
                and (not self.require_decoded_success or decoded_valid)
            )
            proximity = sum(
                values[f"weighted_{name}"]
                for name in ("latent", "decoded", "physiological")
            )
            key = (
                not optimization_success,
                proximity if optimization_success else values["total"],
            )
            if best_key is None or key < best_key:
                best_key, best_latent, selected_step = key, tf.identity(variable), step
            norm = float(tf.linalg.global_norm([gradient]).numpy())
            finite_gradient = math.isfinite(norm) and bool(
                tf.reduce_all(tf.math.is_finite(gradient))
            )
            row = {
                "step": step,
                "target_loss_component": self.target_loss_component,
                "decoded_distance_reference": "original_reconstruction",
                "learning_rate": self.learning_rate
                * self.learning_rate_decay**step,
                **values,
                **{k: v for k, v in prediction.items() if k != "probabilities"},
                **{
                    f"probability_{i}": p
                    for i, p in enumerate(prediction["probabilities"])
                },
                "gradient_norm": norm if finite_gradient else None,
            }
            if self.typicality is not None:
                row.update(typicality_threshold=self.typicality.tau,
                           typical=bool(typical), optimization_success=bool(optimization_success))
            if self.require_decoded_success or state_progress is not None:
                row.update(decoded_valid=bool(decoded_valid),
                           optimization_success=bool(optimization_success))
                row["decoder_latent_rmse"] = float(tf.sqrt(tf.reduce_mean(tf.square(variable - z))).numpy())
                row["d_z"] = (float(tf.sqrt(tf.reduce_mean(tf.square(self._current_typicality_sequence - original_typicality_sequence))).numpy())
                              if self.typicality is not None else row["decoder_latent_rmse"])
                for name, value in current_decoded.items():
                    row[f"decoded_{name}_target_probability"] = decoded_step[name]["target_probability"]
                    row[f"decoded_{name}_predicted_class"] = decoded_step[name]["predicted_class"]
                    row[f"delta_dec_{name}"] = float(tf.sqrt(tf.reduce_mean(tf.square(value - original_decoded[name]))).numpy())
                row["selected_step_so_far"] = selected_step
            history.append(row)
            if progress is not None:
                progress(dict(row))
            if state_progress is not None:
                state_progress(dict(row), {
                    **({"typicality_sequence": self._current_typicality_sequence.numpy()}
                       if self.typicality is not None else {}),
                    "step": np.asarray(step), "z": variable.numpy(),
                    "best_z": best_latent.numpy(), "selected_step": np.asarray(selected_step),
                    "gradient": gradient.numpy(), "classification_embedding": embedding.numpy(),
                    "logits": logits.numpy(),
                    **{f"x_prime_{name}": value.numpy() for name, value in current_decoded.items()},
                    "optimizer_variable_names": np.asarray([v.name for v in descent.variables]),
                    **{f"optimizer_{i}": value.numpy() for i, value in enumerate(descent.variables)},
                })
            if not finite_gradient:
                stop_reason = "non_finite_gradient"
                break
            if optimization_success and (step == 0 or self.stop_on_success):
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

        final_embedding, final_logits = self._classification_state(best_latent)
        final_target_loss, final_target_components = self._target_components(
            final_embedding, final_logits, target_class
        )
        final_terms, decoded = self._objective(
            logits=final_logits,
            target_class=target_class,
            z_prime=best_latent,
            z=z,
            x=x,
            decoder=decode,
            reference_reconstructions=original_decoded,
            target_loss_override=final_target_loss,
            target_components=final_target_components,
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
        extra_summary = {}
        if self.typicality is not None:
            original_d = float(self.typicality.discrepancy(z).numpy())
            final_d = float(self.typicality.discrepancy(best_latent).numpy())
            typical = final_d <= self.typicality.tau
            decoded_valid = all(v["counterfactual"]["predicted_class"] == target_class for v in decoded_results.values())
            extra_summary = {
                "typicality": {"original_discrepancy": original_d,
                               "counterfactual_discrepancy": final_d,
                               "threshold": self.typicality.tau,
                               "typical": bool(typical), "weight": self.typicality_weight,
                               "decoded_valid": bool(decoded_valid),
                               "joint_success": bool(typical and decoded_valid)},
            }
            arrays["typicality_sequence"] = self.typicality.sequence(z).numpy()
            arrays["typicality_sequence_prime"] = self.typicality.sequence(best_latent).numpy()
            arrays["classification_embedding"] = self._classification_state(z)[0].numpy()
            arrays["classification_embedding_prime"] = final_embedding.numpy()
        return {
            "history": history,
            "summary": {
                **extra_summary,
                "target_class": target_class,
                "target_loss_component": self.target_loss_component,
                "decoder_mode": self.decoder_mode,
                "decoded_distance_reference": "original_reconstruction",
                "joint_reconstruction_alpha": (
                    float(self.model.joint_reconstruction_fusion.alpha.numpy())
                    if self.decoder_mode == "joint"
                    else None
                ),
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
                "physiological_constraint_enforced": (
                    self.loss.physiological_weight > 0
                ),
            },
            "arrays": arrays,
        }
