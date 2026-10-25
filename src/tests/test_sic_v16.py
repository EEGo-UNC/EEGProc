"""Cluster preflight for SIC v16. Uses tiny synthetic tensors, never DREAMER."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
import tensorflow as tf

from eegproc.deep_learning.joint_architectures.SICModelv16.sic_model import (
    BANDS, build_sic_model, split_bands,
)
from eegproc.deep_learning.joint_architectures.SICModelv16.run_experiments import (
    fold_subjects, select_oracle,
)


class ThreeBandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)

    def setUp(self):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(123)
        self.x = np.random.default_rng(10).normal(size=(4, 2, 8, 42)).astype("float32")
        self.y = np.array([0, 1, 0, 1], dtype="int32")

    def model(self, **kwargs):
        return build_sic_model((2, 8, 42), conv_filters=(2, 4),
            spatial_pool_sizes=(2, 2), emb_dim=4, attention_heads=2,
            feed_forward_dim=8, classifier_units=4, dropout=0.0,
            classifier_dropout=0.0, **kwargs)

    def assert_no_gradient(self, tape, loss, variables):
        # Concat/split backpropagation can return a dense zero tensor for an
        # unused slice. Isolation means no nonzero gradient, not necessarily
        # a disconnected graph (None). Check every coefficient exactly.
        gradients = tape.gradient(loss, variables)
        for variable, gradient in zip(variables, gradients, strict=True):
            if gradient is not None:
                np.testing.assert_array_equal(tf.convert_to_tensor(gradient).numpy(), 0,
                    err_msg=f"Unexpected gradient to {variable.path}")

    def test_band_order_and_isolation(self):
        model = self.model(use_transformer=True)
        values = np.arange(42, dtype="float32").reshape(1, 1, 42)
        for index in range(3):
            np.testing.assert_array_equal(split_bands(values).numpy()[0, 0, :, index],
                                          np.arange(index, 42, 3))
        encoder = model.graph_encoder
        before, after = encoder.encode(self.x[:, 0], training=False)
        changed = self.x[:, 0].copy()
        changed[..., 0::3] += 7
        before_changed, after_changed = encoder.encode(changed, training=False)
        # Changing theta cannot alter either of the other branches.
        np.testing.assert_array_equal(before.numpy()[..., 4:], before_changed.numpy()[..., 4:])
        np.testing.assert_array_equal(after.numpy()[..., 4:], after_changed.numpy()[..., 4:])
        self.assertGreater(float(tf.reduce_max(tf.abs(before[..., :4] - before_changed[..., :4]))), 0)
        for i in range(3):
            for j in range(i):
                self.assertTrue(set(map(id, encoder.encoders[i].weights)).isdisjoint(
                    set(map(id, encoder.encoders[j].weights))))

    def test_reconstruction_before_attention_and_encoder_gradients(self):
        model = self.model(use_decoder=True, use_transformer=True)
        with tf.GradientTape(persistent=True) as tape:
            outputs = model._encode(self.x, training=False)
            components = model._reconstruction_components(outputs, training=False)
            loss = components["reconstruction_loss"]
            band_losses = [components[f"{band}_reconstruction_loss"] for band in BANDS]
        np.testing.assert_allclose(loss.numpy(), np.mean([v.numpy() for v in band_losses]), rtol=1e-6)
        for band_index, encoder in enumerate(model.graph_encoder.encoders):
            gradients = tape.gradient(loss, encoder.trainable_variables)
            self.assertTrue(any(g is not None and float(tf.reduce_sum(tf.abs(g))) > 0 for g in gradients))
            # A band's objective cannot update either other band's CNN.
            for other_index, other in enumerate(model.graph_encoder.encoders):
                if other_index != band_index:
                    self.assert_no_gradient(tape, band_losses[band_index], other.trainable_variables)
        transformer_variables = [v for block in model.graph_encoder.transformers for v in block.trainable_variables]
        self.assert_no_gradient(tape, loss, transformer_variables)
        reconstructed = model.reconstruct(self.x)
        for band in BANDS:
            self.assertEqual(tuple(reconstructed[band].shape), (4, 2, 8, 14))
        del tape

    def test_classification_reaches_every_band_and_transformer(self):
        model = self.model(use_transformer=True)
        with tf.GradientTape(persistent=True) as tape:
            logits = model(self.x, training=False)
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
                self.y, logits, from_logits=True))
        for block in list(model.graph_encoder.encoders) + list(model.graph_encoder.transformers):
            gradients = tape.gradient(loss, block.trainable_variables)
            self.assertTrue(any(g is not None and float(tf.reduce_sum(tf.abs(g))) > 0 for g in gradients))
        del tape

    def test_all_four_train_and_checkpoint_roundtrip(self):
        for transformer in (False, True):
            for decoder in (False, True):
                with self.subTest(transformer=transformer, decoder=decoder):
                    model = self.model(use_transformer=transformer, use_decoder=decoder)
                    original = [v.numpy().copy() for v in model.trainable_variables]
                    history = model.fit(self.x, self.y, batch_size=2, epochs=1, verbose=0)
                    self.assertTrue(np.isfinite(history.history["loss"]).all())
                    self.assertTrue(any(not np.array_equal(old, v.numpy())
                                        for old, v in zip(original, model.trainable_variables)))
                    self.assertEqual(bool(model.graph_encoder.transformers), transformer)
                    self.assertEqual(len(model.temporal_bilstms), 0)
                    self.assertFalse(model.use_variational_head)
                    prediction = model(self.x, training=False).numpy()
                    with tempfile.TemporaryDirectory() as folder:
                        path = Path(folder) / "model.keras"
                        model.save(path)
                        restored = tf.keras.models.load_model(path, compile=False)
                        np.testing.assert_allclose(restored(self.x, training=False), prediction, atol=1e-6)
                        for left, right in zip(model.weights, restored.weights, strict=True):
                            np.testing.assert_array_equal(left.numpy(), right.numpy())
                        if decoder:
                            for band in BANDS:
                                np.testing.assert_allclose(model.reconstruct(self.x)[band],
                                    restored.reconstruct(self.x)[band], atol=1e-6)
                        restored.prepare_for_subject_calibration(learning_rate=1e-4, unfreeze_layers=2,
                            freeze_decoder=not decoder)
                        result = restored.train_on_batch(self.x, self.y, return_dict=True)
                        self.assertTrue(np.isfinite(list(result.values())).all())
                        self.assertFalse(restored.graph_encoder.trainable)

    def test_subject_adversary_vrex_and_sic_heads(self):
        subjects = np.arange(4)
        for extra in ({"use_subject_adversarial": True}, {"training_method": "vrex"},
                      {"use_recurrent_classifier": True, "use_variational_head": True,
                       "classifier_rnn_units": (4,), "classifier_rnn_dropout": 0.0}):
            with self.subTest(extra=extra):
                model = self.model(training_subject_ids=subjects,
                    training_trial_ids=np.arange(4), **extra)
                result = model.train_on_batch(model.prepare_fit_inputs(self.x, subjects),
                                               self.y, return_dict=True)
                self.assertTrue(np.isfinite(list(result.values())).all())

    def test_mldg(self):
        # The SIC sampler requires both classes within every source subject,
        # even when an episode samples only a subset of that subject's trials.
        subjects = np.repeat(np.arange(4), 2)
        trials = np.tile(np.arange(2), 4)
        labels = np.tile(np.array([0, 1], dtype="int32"), 4)
        eeg = np.random.default_rng(15).normal(size=(8, 2, 8, 42)).astype("float32")
        model = self.model(training_method="mldg", mldg_meta_train_subjects=2,
            mldg_meta_test_subjects=1, mldg_trials_per_subject=2, mldg_steps_per_epoch=1,
            training_subject_ids=subjects, training_trial_ids=trials, use_decoder=True)
        history = model.fit(model.prepare_fit_inputs(eeg, subjects), labels,
                            epochs=1, verbose=0)
        self.assertTrue(np.isfinite(history.history["loss"]).all())

    def test_temporal_pooling_and_window_mode(self):
        model = self.model(temporal_pool_sizes=(2, 2), use_decoder=True, use_transformer=True)
        self.assertEqual(tuple(model.graph_encoder(self.x[:, 0]).shape), (4, 2, 12))
        self.assertEqual(tuple(model.reconstruct(self.x)["theta"].shape), (4, 2, 8, 14))
        window_model = build_sic_model((8, 42), classification_level="window",
            conv_filters=(2, 4), emb_dim=4, classifier_units=4)
        self.assertEqual(tuple(window_model(self.x[:, 0]).shape), (4, 2))

    def test_subject_split_and_oracle_rule(self):
        train, validation = fold_subjects(np.arange(23), 0, 2, 42)
        self.assertEqual(len(train), 20)
        self.assertEqual(len(validation), 2)
        self.assertTrue(set(train).isdisjoint(validation))
        self.assertNotIn(0, train + validation)
        self.assertEqual((train, validation), fold_subjects(np.arange(23), 0, 2, 42))
        rows = [dict(target_subject=s, epoch=e, balanced_accuracy=score,
                     checkpoint=f"{s}_{e}.keras", decision_threshold=0.5)
                for s, e, score in ((0, 1, .9), (1, 1, .5), (0, 2, .5), (1, 2, .9))]
        result = select_oracle(rows, [0, 1], 2)
        self.assertEqual(result["best_epoch"], 1)
        self.assertAlmostEqual(result["selection_score"], .7)
        self.assertEqual(select_oracle(rows[:-1], [0, 1], 2)["status"], "incomplete")


if __name__ == "__main__":
    unittest.main(verbosity=2)
