"""Explicit mappings from decoder latents into the learned VC coordinates.

SIC's 510-D concatenated decoder features and its 128-D VC embedding are
different spaces. A matching vector length alone would not make them the
same space. Both mappings here reuse the checkpoint's frozen recurrent
weights and leave its original full-trial classification path unchanged.
"""

import numpy as np
import tensorflow as tf


class SICVCSequence:
    def __init__(self, model, *, mode):
        self.model = model
        self.mode = mode
        if mode not in ("vc_window_embeddings", "vc_hidden_sequence"):
            raise ValueError("Choose vc_window_embeddings or vc_hidden_sequence explicitly")
        head = model.vc_target
        if not hasattr(head, "prior_mu") or not hasattr(head, "prior_log_sigma"):
            raise ValueError("Typicality requires the checkpoint's learned Gaussian VC priors")
        self.dimension = int(head.prior_mu.shape[-1])
        self.layers = []
        self.final_index = None
        if mode == "vc_hidden_sequence":
            layers = [layer for layer in model.trial_recurrent_classifier.layers
                      if not isinstance(layer, tf.keras.layers.InputLayer)]
            final = [i for i, layer in enumerate(layers)
                     if isinstance(layer, (tf.keras.layers.GRU, tf.keras.layers.Bidirectional))
                     and not layer.return_sequences]
            if len(final) != 1:
                raise ValueError("Expected exactly one final GRU/BiGRU summarizer")
            self.final_index = final[0]
            self.layers = layers
            last = layers[self.final_index]
            if isinstance(last, tf.keras.layers.Bidirectional):
                if last.merge_mode != "concat":
                    raise ValueError("Only concat BiGRU summaries are supported")
                self.forward = self._sequence_runner(last.forward_layer)
                self.backward = self._sequence_runner(last.backward_layer)
            else:
                self.forward, self.backward = self._sequence_runner(last), None
            if any(not isinstance(layer, (tf.keras.layers.LayerNormalization, tf.keras.layers.Dropout))
                   for layer in layers[self.final_index + 1:]):
                raise ValueError("Unknown post-recurrent transform; provide an explicit mapping")

    @staticmethod
    def _sequence_runner(layer):
        if not isinstance(layer, tf.keras.layers.GRU) or layer.stateful:
            raise ValueError("Only stateless trained GRU cells are supported")
        # Wrap the SAME built cell, not a randomly initialized/cloned cell.
        return tf.keras.layers.RNN(layer.cell, return_sequences=True,
                                   go_backwards=layer.go_backwards, dtype=layer.compute_dtype)

    def __call__(self, latent):
        z = tf.cast(latent, tf.float32)
        tf.debugging.assert_rank(z, 4)
        tf.debugging.assert_equal(tf.shape(z)[0], 1)
        if self.mode == "vc_window_embeddings":
            # The W windows are independent recurrent sequences for q_Z only.
            windows = tf.reshape(z, [-1, tf.shape(z)[2], tf.shape(z)[3]])
            sequence = self.model.trial_recurrent_classifier(windows, training=False)[None]
        else:
            sequence = tf.reshape(z, [1, -1, tf.shape(z)[-1]])
            for i, layer in enumerate(self.layers):
                if i == self.final_index:
                    forward = self.forward(sequence, training=False)
                    sequence = (tf.concat((forward, tf.reverse(self.backward(sequence, training=False), axis=[1])), axis=-1)
                                if self.backward is not None else forward)
                else:
                    sequence = layer(sequence, training=False)
        tf.debugging.assert_equal(tf.shape(sequence)[-1], self.dimension)
        return tf.cast(sequence, tf.float32)

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
        return {"mode": self.mode, "dimension": self.dimension,
                "sequence_axis": "windows" if self.mode == "vc_window_embeddings" else "all ordered window-timesteps",
                "recurrent_reset": "each window" if self.mode == "vc_window_embeddings" else "once per complete trial",
                "classification": "original full-sequence summarizer remains unchanged",
                "prior_training_support": "learned on full-trial terminal embeddings; threshold empirically calibrated on chosen sequences"}
