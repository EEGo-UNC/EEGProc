"""Latent coords idea: frozen, per-input pre/post-ISA coordinate residuals.

Delta(X) = E0(X) - E1(X), decoded candidate = R0(Z1_candidate + Delta(X)).
The residual contains all training changes, not an identified subject factor.
It is never added to the emotion classifier's input.
"""
import tensorflow as tf

from eegproc.model_explainability.counterfactuals.fusion import FrozenJointFusion
from eegproc.model_explainability.counterfactuals.optimizer import CounterfactualOptimizer


def branches(model):
    return [(name, int(getattr(model, f"{name}_feature_dim")))
            for name in ("gcn_gru", "bilstm")
            if getattr(model, f"use_{name}_branch", False)]


class LatentCoordinatesDecoder:
    def __init__(self, reference_model, invariant_model):
        self.reference, self.invariant = reference_model, invariant_model
        if not getattr(reference_model, "use_decoder", False):
            raise ValueError("The pre-ISA reference must contain trained decoders.")
        if branches(reference_model) != branches(invariant_model):
            raise ValueError("Pre/post-ISA models must have the same active branch widths.")
        for attr in ("classification_level", "n_channels", "n_bands"):
            if getattr(reference_model, attr, None) != getattr(invariant_model, attr, None):
                raise ValueError(f"Pre/post-ISA models disagree on {attr}.")
        if not branches(reference_model):
            raise ValueError("No compatible encoder branches.")
        self.branches = branches(reference_model)

    def capture(self, original_eeg):
        """Compute a detached residual once from the ORIGINAL trial(s)."""
        before = self.reference.get_encoder_features(original_eeg)
        after = self.invariant.get_encoder_features(original_eeg)
        residuals = {}
        for name, _ in self.branches:
            key = f"{name}_features"
            tf.debugging.assert_equal(tf.shape(before[key]), tf.shape(after[key]),
                                      message="Pre/post-ISA feature shapes must align.")
            residuals[name] = tf.stop_gradient(before[key] - after[key])
        return residuals

    def decode_branch(self, name, candidate_features, residuals):
        if name not in residuals:
            raise ValueError(f"No original-trial residual for branch {name}.")
        tf.debugging.assert_equal(tf.shape(candidate_features), tf.shape(residuals[name]),
                                  message="Candidate must match its original-trial residual.")
        restored = candidate_features + residuals[name]
        return self.reference.decode_branch_feature_sequence(name, restored)

    def reconstruct(self, original_eeg):
        """Zero-edit reconstruction equals R0(E0(X)), up to roundoff."""
        residuals = self.capture(original_eeg)
        features = self.invariant.get_encoder_features(original_eeg)
        decoded = {name: self.decode_branch(name, features[f"{name}_features"], residuals)
                   for name, _ in self.branches}
        if len(decoded) == 2:
            result = FrozenJointFusion(self.reference, list(decoded))(decoded)
        else:
            result = next(iter(decoded.values()))
        return tf.reshape(result, tf.shape(original_eeg))


class LatentCoordsOptimizer(CounterfactualOptimizer):
    """Existing CFO selection/classification with R0(Z1' + fixed Delta(X))."""
    def __init__(self, model, *, reference_model, **kwargs):
        super().__init__(model, **kwargs)
        self.coordinates = LatentCoordinatesDecoder(reference_model, model)
        self._residuals = None
        if self.decoder_mode == "joint":
            # Zero-edit restoration must use the pre-ISA decoder's mixer too.
            self.joint_fusion = FrozenJointFusion(
                reference_model, [name for name, _ in self.branches],
                fixed_alpha=self.fixed_joint_alpha)

    def _decode_branches(self, latent, x):
        if self._residuals is None:
            raise RuntimeError("Capture the original trial via optimize() first.")
        result, offset = {}, 0
        for name, width in self.branches:
            part = latent[..., offset:offset + width]
            flat = tf.reshape(part, [-1, tf.shape(latent)[2], width])
            decoded = self.coordinates.decode_branch(name, flat, self._residuals)
            tf.debugging.assert_equal(tf.shape(decoded), tf.shape(x)[1:])
            result[name] = tf.reshape(decoded, tf.shape(x))
            offset += width
        return result

    def optimize(self, inputs, **kwargs):
        eeg = tf.convert_to_tensor(inputs, dtype=tf.float32)
        if eeg.shape.rank == 3:
            eeg = eeg[None, ...]
        if eeg.shape.rank != 4 or eeg.shape[0] != 1:
            raise ValueError("Supply one original trial, not a batch.")
        self._residuals = self.coordinates.capture(eeg)
        try:
            result = super().optimize(eeg, **kwargs)
            result["summary"]["decoder_recovery"] = {
                "method": "latent_coords", "residual": "E0(X)-E1(X)",
                "decoder": "pre_isa", "classifier": "post_isa",
                "coordinate_alignment": "raw_paired_initialization",
                "residual_is_subject_only": False,
                "residual_fixed_during_optimization": True,
            }
            for name, value in self._residuals.items():
                result["arrays"][f"coordinate_residual_{name}"] = value.numpy()
            return result
        finally:
            # Reusing the optimizer cannot leak a previous trial's residual.
            self._residuals = None
