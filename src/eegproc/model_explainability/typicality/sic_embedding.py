"""The full-trial terminal embedding used by SIC's frozen VC head."""

import numpy as np
import tensorflow as tf

from .core import REPRESENTATION


class SICTrialEmbedding:
    def __init__(self, model):
        if model.classification_level != "trial":
            raise ValueError("Full-trial typicality requires a trial-level SIC classifier")
        self.model = model
        head = model.vc_target
        if not hasattr(head, "prior_mu") or not hasattr(head, "prior_log_sigma"):
            raise ValueError("Typicality requires the checkpoint's learned Gaussian VC parameters")
        self.dimension = int(head.prior_mu.shape[-1])

    def __call__(self, latent):
        """Preserve chronological window/timestep order; summarize exactly once."""
        z = tf.cast(latent, tf.float32)
        tf.debugging.assert_rank(z, 4)
        tf.debugging.assert_equal(tf.shape(z)[0], 1)
        tf.debugging.assert_positive(tf.shape(z))
        sequence = tf.reshape(z, [1, -1, tf.shape(z)[-1]])
        embedding = self.model.trial_recurrent_classifier(sequence, training=False)
        tf.debugging.assert_shapes([(embedding, (1, self.dimension))])
        return tf.cast(embedding, tf.float32)

    def learned_prior(self, target_class=1):
        head = self.model.vc_target
        mean = np.asarray(head.prior_mu[target_class].numpy(), dtype=np.float64)
        log_sigma = np.asarray(head.prior_log_sigma[target_class].numpy(), dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            variance = np.exp(2.0 * log_sigma)
        if not np.isfinite(mean).all() or not np.isfinite(variance).all():
            raise ValueError("Checkpoint's learned Gaussian parameters are non-finite")
        return mean, variance

    def metadata(self):
        return {"representation": REPRESENTATION, "dimension": self.dimension,
                "input_axes": "one trial, all windows, all timesteps, encoder coordinates",
                "sequence_order": "chronological windows and timesteps flattened together",
                "recurrent_reset": "once per complete supplied trial",
                "classification": "same terminal embedding passed to the frozen VC head",
                "prior_training_support": "full-trial terminal embeddings"}
