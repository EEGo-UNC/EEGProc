"""GPU pre-resume gate: checkpoint optimizer state must survive reload exactly."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import tensorflow as tf
from eegproc.deep_learning.joint_architectures.SICModelv16.sic_model import build_sic_model
from eegproc.deep_learning.joint_architectures.SICModelv16.recovery import checkpoint_progress


class ResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for gpu in tf.config.list_physical_devices("GPU"):
            # Data tests ran in another process; configure before GPU execution here.
            tf.config.experimental.set_memory_growth(gpu, True)

    def test_optimizer_slots_and_next_update_survive_checkpoint(self):
        for attention, reconstruction in ((False, False), (True, False), (True, True)):
            with self.subTest(attention=attention, reconstruction=reconstruction):
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(42)
                x = np.random.default_rng(1).normal(size=(2, 1, 8, 42)).astype("float32")
                y = np.array([0, 1])
                model = build_sic_model((1, 8, 42), conv_filters=(2, 4), emb_dim=4,
                    classifier_units=4, attention_heads=2, feed_forward_dim=8,
                    dropout=0.0, classifier_dropout=0.0,
                    use_transformer=attention, use_decoder=reconstruction)
                model.train_on_batch(x, y)
                model.train_on_batch(x, y)
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / "epoch.keras"
                    model.save(path)
                    restored = tf.keras.models.load_model(path, compile=True)
                    self.assertEqual(int(restored.main_optimizer.iterations.numpy()), 2)
                    for before, after in zip(model.main_optimizer.variables,
                                             restored.main_optimizer.variables, strict=True):
                        np.testing.assert_array_equal(before.numpy(), after.numpy())
                    for before, after in zip(model.weights, restored.weights, strict=True):
                        np.testing.assert_array_equal(before.numpy(), after.numpy())
                    # Compare the next deterministic update, not just predictions:
                    # resetting optimizer moments would fail this equivalence.
                    model.train_on_batch(x, y)
                    restored.train_on_batch(x, y)
                    self.assertEqual(int(restored.main_optimizer.iterations.numpy()), 3)
                    for before, after in zip(model.weights, restored.weights, strict=True):
                        np.testing.assert_allclose(before.numpy(), after.numpy(), atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
