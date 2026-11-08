"""Frozen reconstruction mixing shared by SIC adapters and optimization."""

import math

import tensorflow as tf


def validate_fixed_joint_alpha(alpha, decoder_mode):
    if alpha is None:
        return None
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("fixed_joint_alpha must be finite and in [0, 1].")
    if decoder_mode != "joint":
        raise ValueError("fixed_joint_alpha requires joint decoder mode.")
    return alpha


class FrozenJointFusion:
    """Use a checkpoint's learned mixer or an explicit constant, without mutation.

    Alpha always weights the GCN-GRU reconstruction; 1-alpha weights BiLSTM.
    A fixed override also supports v11 checkpoints with independent decoders.
    It adds no model variables and does not modify the saved fusion layer.
    """

    def __init__(self, model, branches, *, fixed_alpha=None):
        self.fixed_alpha = validate_fixed_joint_alpha(fixed_alpha, "joint")
        if set(branches) != {"gcn_gru", "bilstm"}:
            raise ValueError("Joint reconstruction requires both SIC branches.")
        self.learned = getattr(model, "joint_reconstruction_fusion", None)
        if self.fixed_alpha is None:
            if not getattr(model, "use_joint_reconstruction", False):
                raise ValueError("This SIC checkpoint has no joint reconstruction; supply fixed_joint_alpha explicitly.")
            if self.learned is None:
                raise ValueError("The SIC checkpoint is missing its fusion layer.")

    @property
    def alpha(self):
        if self.fixed_alpha is not None:
            return tf.constant(self.fixed_alpha, dtype=tf.float32)
        return self.learned.alpha

    def __call__(self, branches):
        gcn, bilstm = branches["gcn_gru"], branches["bilstm"]
        tf.debugging.assert_equal(tf.shape(gcn), tf.shape(bilstm))
        if self.fixed_alpha is None:
            return self.learned([gcn, bilstm])
        alpha = tf.cast(self.alpha, gcn.dtype)
        return alpha * gcn + (1.0 - alpha) * bilstm

    def metadata(self):
        alpha = self.fixed_alpha if self.fixed_alpha is not None else float(self.alpha.numpy())
        return {
            "joint_reconstruction_alpha": alpha,
            "joint_reconstruction_weights": {"gcn_gru": alpha, "bilstm": 1.0 - alpha},
            "joint_reconstruction_weight_source": "fixed_override" if self.fixed_alpha is not None else "checkpoint_learned",
        }
