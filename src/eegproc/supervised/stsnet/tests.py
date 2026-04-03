"""
tests.py
========
Unit and smoke tests for the STSNet implementation.

Run with:
    python tests.py

All tests use synthetic data so no real EEG files are needed.
"""

import numpy as np
import tensorflow as tf
import unittest


# ---------------------------------------------------------------------------
# Helper: synthetic EEG trial
# ---------------------------------------------------------------------------

def make_trial(n_channels=32, n_samples=7680, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_channels, n_samples)).astype(np.float32)


# ---------------------------------------------------------------------------
# data_representation tests
# ---------------------------------------------------------------------------

class TestDataRepresentation(unittest.TestCase):

    def setUp(self):
        from data_representation import (
            covariance_to_spd,
            flatten_lower_triangular,
            compute_spd,
            build_4d_representation,
            build_spatiotemporal_representation,
        )
        self.covariance_to_spd          = covariance_to_spd
        self.flatten_lower_triangular   = flatten_lower_triangular
        self.compute_spd                = compute_spd
        self.build_4d                   = build_4d_representation
        self.build_st                   = build_spatiotemporal_representation

    def test_spd_is_positive_definite(self):
        """SPD matrix must have all-positive eigenvalues."""
        seg = make_trial(32, 512)
        spd = self.compute_spd(seg)
        vals = np.linalg.eigvalsh(spd)
        self.assertTrue(np.all(vals > 0), f"Non-positive eigenvalue: {vals.min()}")

    def test_spd_is_symmetric(self):
        seg = make_trial(32, 512)
        spd = self.compute_spd(seg)
        self.assertTrue(np.allclose(spd, spd.T, atol=1e-5))

    def test_flatten_length(self):
        C = 14
        spd = np.eye(C, dtype=np.float32)
        flat = self.flatten_lower_triangular(spd)
        expected = C * (C + 1) // 2
        self.assertEqual(flat.shape[0], expected)

    def test_4d_representation_shape(self):
        sig   = make_trial(32, 7680)
        bands = ["theta", "alpha", "beta", "gamma"]
        xd    = self.build_4d(sig, fs=128, bands=bands, window_size=512, n_windows=15)
        self.assertEqual(xd.shape, (15, 4, 32, 32))

    def test_spatiotemporal_representation_shape_fw(self):
        sig = make_trial(32, 7680)
        bi  = self.build_st(sig, n_windows=15, use_variable_windows=False,
                            fixed_window_size=512)
        expected_feat = 32 * 33 // 2
        self.assertEqual(bi.shape, (15, expected_feat))

    def test_spatiotemporal_representation_shape_vw(self):
        sig = make_trial(32, 7680)
        bi  = self.build_st(sig, n_windows=15, use_variable_windows=True)
        expected_feat = 32 * 33 // 2
        self.assertEqual(bi.shape, (15, expected_feat))


# ---------------------------------------------------------------------------
# manifold_net tests
# ---------------------------------------------------------------------------

class TestManifoldNet(unittest.TestCase):

    def _make_spd_batch(self, batch=4, n_windows=13, n_bands=4, n=32):
        """Create a batch of random SPD matrices."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((batch, n_windows, n_bands, n, n)).astype(np.float32)
        # A Aᵀ + εI  ensures SPD
        A = A @ np.transpose(A, (0, 1, 2, 4, 3)) + 1e-3 * np.eye(n)
        return tf.constant(A)

    def test_manifoldnet_output_shape(self):
        from manifold_net import ManifoldNet
        model = ManifoldNet(n_channels=14, kernel_size=2, n_fm_iters=3)
        x = self._make_spd_batch(batch=2, n_windows=13, n_bands=3, n=14)
        mo = model(x, training=False)
        # After 2× wFM with kernel=2: T' = 13-2 = 11; bands=3; d = 11*3 = 33
        self.assertEqual(mo.shape[0], 2)
        self.assertEqual(mo.shape[1], (13 - 2) * 3)

    def test_wfm_layer_output_shape(self):
        from manifold_net import WFMLayer
        layer = WFMLayer(kernel_size=2, n_fm_iters=3)
        x = self._make_spd_batch(batch=2, n_windows=5, n_bands=3, n=14)
        out = layer(x)
        self.assertEqual(out.shape, (2, 4, 3, 14, 14))

    def test_wfm_output_is_spd(self):
        """wFM output matrices should remain SPD (positive eigenvalues)."""
        from manifold_net import WFMLayer
        layer = WFMLayer(kernel_size=2, n_fm_iters=5)
        x = self._make_spd_batch(batch=1, n_windows=3, n_bands=2, n=8)
        out = layer(x).numpy()
        for t in range(out.shape[1]):
            for b in range(out.shape[2]):
                vals = np.linalg.eigvalsh(out[0, t, b])
                self.assertTrue(np.all(vals > 0))

    def test_invariant_layer_shape(self):
        from manifold_net import InvariantLayer
        layer = InvariantLayer(n_fm_iters=3)
        x = self._make_spd_batch(batch=2, n_windows=4, n_bands=3, n=14)
        mo = layer(x)
        self.assertEqual(mo.shape, (2, 12))


# ---------------------------------------------------------------------------
# BiLSTM tests
# ---------------------------------------------------------------------------

class TestBiLSTMNet(unittest.TestCase):

    def test_output_shape(self):
        from stsnet import BiLSTMNet
        model = BiLSTMNet(hidden_units=64, dropout_rate=0.0)
        x = tf.random.normal((4, 15, 528))   # (batch, n_windows, feat_dim)
        ho = model(x, training=False)
        self.assertEqual(ho.shape, (4, 128))  # 2 * hidden_units


# ---------------------------------------------------------------------------
# Full STSNet smoke test
# ---------------------------------------------------------------------------

class TestSTSNet(unittest.TestCase):

    def _make_batch(self, batch=4, n_windows=13, n_bands=4, C=32):
        rng = np.random.default_rng(0)
        # xd: (batch, n_windows, n_bands, C, C) — random SPD matrices
        A   = rng.standard_normal((batch, n_windows, n_bands, C, C)).astype(np.float32)
        xd  = A @ np.transpose(A, (0, 1, 2, 4, 3)) + 1e-3 * np.eye(C)
        # bi: (batch, n_windows, C*(C+1)//2)
        feat = C * (C + 1) // 2
        bi   = rng.standard_normal((batch, n_windows, feat)).astype(np.float32)
        y    = rng.integers(0, 2, size=(batch,)).astype(np.int32)
        return tf.constant(xd), tf.constant(bi), tf.constant(y)

    def test_forward_pass_shape(self):
        from stsnet import STSNet
        model  = STSNet(n_channels=14, n_classes=2,
                        bilstm_units=32, n_fm_iters=3)
        xd, bi, _ = self._make_batch(batch=2, n_windows=13, n_bands=3, C=14)
        logits = model((xd, bi), training=False)
        self.assertEqual(logits.shape, (2, 2))

    def test_logits_finite(self):
        from stsnet import STSNet
        model  = STSNet(n_channels=14, n_classes=2,
                        bilstm_units=32, n_fm_iters=3)
        xd, bi, _ = self._make_batch(batch=2, n_windows=13, n_bands=3, C=14)
        logits = model((xd, bi), training=False).numpy()
        self.assertTrue(np.all(np.isfinite(logits)))

    def test_single_training_step(self):
        """Verify that a single joint training step runs without error."""
        from stsnet import STSNet
        import tensorflow as tf
        model  = STSNet(n_channels=14, n_classes=2,
                        bilstm_units=32, n_fm_iters=3)
        xd, bi, y = self._make_batch(batch=4, n_windows=13, n_bands=3, C=14)
        loss_fn    = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        opt_m = tf.keras.optimizers.Adam(1e-4)
        opt_b = tf.keras.optimizers.Adam(1e-4)
        opt_f = tf.keras.optimizers.Adam(1e-4)

        loss_m = model._train_step_manifold(xd, bi, y, opt_m, opt_f, loss_fn)
        loss_b = model._train_step_bilstm(xd, bi, y, opt_b, opt_f, loss_fn)
        self.assertTrue(np.isfinite(float(loss_m)))
        self.assertTrue(np.isfinite(float(loss_b)))


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
