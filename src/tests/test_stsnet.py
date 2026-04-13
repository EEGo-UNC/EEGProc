"""
Pytest unit and smoke tests for the STSNet implementation.

All tests use synthetic data so no real EEG files are needed.
"""

from pathlib import Path
import sys

import numpy as np
import tensorflow as tf


# Keep compatibility with local imports used inside the STSNet modules.
STSNET_DIR = Path(__file__).resolve().parents[1] / "eegproc" / "supervised" / "stsnet"
if str(STSNET_DIR) not in sys.path:
    sys.path.insert(0, str(STSNET_DIR))

from eegproc.supervised.stsnet.data_representation import (
    build_4d_representation,
    build_spatiotemporal_representation,
    compute_spd,
    flatten_lower_triangular,
)
from eegproc.supervised.stsnet.manifold_net import InvariantLayer, ManifoldNet, WFMLayer
from eegproc.supervised.stsnet import BiLSTMNet, STSNet


def make_trial(n_channels=32, n_samples=7680, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_channels, n_samples)).astype(np.float32)


def test_spd_is_positive_definite():
    """SPD matrix must have all-positive eigenvalues."""
    seg = make_trial(32, 512)
    spd = compute_spd(seg)
    vals = np.linalg.eigvalsh(spd)
    assert np.all(vals > 0), f"Non-positive eigenvalue: {vals.min()}"


def test_spd_is_symmetric():
    seg = make_trial(32, 512)
    spd = compute_spd(seg)
    assert np.allclose(spd, spd.T, atol=1e-5)


def test_flatten_length():
    c = 14
    spd = np.eye(c, dtype=np.float32)
    flat = flatten_lower_triangular(spd)
    expected = c * (c + 1) // 2
    assert flat.shape[0] == expected


def test_4d_representation_shape():
    sig = make_trial(32, 7680)
    bands = ["theta", "alpha", "beta", "gamma"]
    xd = build_4d_representation(sig, fs=128, bands=bands, window_size=512, n_windows=15)
    assert xd.shape == (15, 4, 32, 32)


def test_spatiotemporal_representation_shape_fw():
    sig = make_trial(32, 7680)
    bi = build_spatiotemporal_representation(
        sig,
        n_windows=15,
        use_variable_windows=False,
        fixed_window_size=512,
    )
    expected_feat = 32 * 33 // 2
    assert bi.shape == (15, expected_feat)


def test_spatiotemporal_representation_shape_vw():
    sig = make_trial(32, 7680)
    bi = build_spatiotemporal_representation(sig, n_windows=15, use_variable_windows=True)
    expected_feat = 32 * 33 // 2
    assert bi.shape == (15, expected_feat)


def _make_spd_batch(batch=4, n_windows=13, n_bands=4, n=32):
    """Create a batch of random SPD matrices."""
    rng = np.random.default_rng(42)
    a = rng.standard_normal((batch, n_windows, n_bands, n, n)).astype(np.float32)
    # A A^T + eps I ensures SPD.
    a = a @ np.transpose(a, (0, 1, 2, 4, 3)) + 1e-3 * np.eye(n)
    return tf.constant(a)


def test_manifoldnet_output_shape():
    model = ManifoldNet(n_channels=14, kernel_size=2, n_fm_iters=3)
    x = _make_spd_batch(batch=2, n_windows=13, n_bands=3, n=14)
    mo = model(x, training=False)
    # After 2x wFM with kernel=2: T' = 13-2 = 11; bands=3; d = 11*3 = 33
    assert mo.shape[0] == 2
    assert mo.shape[1] == (13 - 2) * 3


def test_wfm_layer_output_shape():
    layer = WFMLayer(kernel_size=2, n_fm_iters=3)
    x = _make_spd_batch(batch=2, n_windows=5, n_bands=3, n=14)
    out = layer(x)
    assert out.shape == (2, 4, 3, 14, 14)


def test_wfm_output_is_spd():
    """wFM output matrices should remain SPD (positive eigenvalues)."""
    layer = WFMLayer(kernel_size=2, n_fm_iters=5)
    x = _make_spd_batch(batch=1, n_windows=3, n_bands=2, n=8)
    out = layer(x).numpy()
    for t in range(out.shape[1]):
        for b in range(out.shape[2]):
            vals = np.linalg.eigvalsh(out[0, t, b])
            assert np.all(vals > 0)


def test_invariant_layer_shape():
    layer = InvariantLayer(n_fm_iters=3)
    x = _make_spd_batch(batch=2, n_windows=4, n_bands=3, n=14)
    mo = layer(x)
    assert mo.shape == (2, 12)


def test_bilstm_output_shape():
    model = BiLSTMNet(hidden_units=64, dropout_rate=0.0)
    x = tf.random.normal((4, 15, 528))
    ho = model(x, training=False)
    assert ho.shape == (4, 128)


def _make_batch(batch=4, n_windows=13, n_bands=4, c=32):
    rng = np.random.default_rng(0)
    # xd: (batch, n_windows, n_bands, c, c) random SPD matrices
    a = rng.standard_normal((batch, n_windows, n_bands, c, c)).astype(np.float32)
    xd = a @ np.transpose(a, (0, 1, 2, 4, 3)) + 1e-3 * np.eye(c)
    # bi: (batch, n_windows, c*(c+1)//2)
    feat = c * (c + 1) // 2
    bi = rng.standard_normal((batch, n_windows, feat)).astype(np.float32)
    y = rng.integers(0, 2, size=(batch,)).astype(np.int32)
    return tf.constant(xd), tf.constant(bi), tf.constant(y)


def test_stsnet_forward_pass_shape():
    model = STSNet(n_channels=14, n_classes=2, bilstm_units=32, n_fm_iters=3)
    xd, bi, _ = _make_batch(batch=2, n_windows=13, n_bands=3, c=14)
    logits = model((xd, bi), training=False)
    assert logits.shape == (2, 2)


def test_stsnet_logits_finite():
    model = STSNet(n_channels=14, n_classes=2, bilstm_units=32, n_fm_iters=3)
    xd, bi, _ = _make_batch(batch=2, n_windows=13, n_bands=3, c=14)
    logits = model((xd, bi), training=False).numpy()
    assert np.all(np.isfinite(logits))


def test_stsnet_single_training_step():
    """Verify that a single joint training step runs without error."""
    model = STSNet(n_channels=14, n_classes=2, bilstm_units=32, n_fm_iters=3)
    xd, bi, y = _make_batch(batch=4, n_windows=13, n_bands=3, c=14)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    opt_m = tf.keras.optimizers.Adam(1e-4)
    opt_b = tf.keras.optimizers.Adam(1e-4)
    opt_f = tf.keras.optimizers.Adam(1e-4)

    loss_m = model._train_step_manifold(xd, bi, y, opt_m, opt_f, loss_fn)
    loss_b = model._train_step_bilstm(xd, bi, y, opt_b, opt_f, loss_fn)
    assert np.isfinite(float(loss_m))
    assert np.isfinite(float(loss_b))
