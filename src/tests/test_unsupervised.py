import os
import numpy as np
import pytest
import tensorflow as tf

from eegproc.unsupervised.unsupervised import encoder_1dcnn, training_autoencoder


@pytest.fixture(autouse=True)
def _set_seeds():
    tf.keras.utils.set_random_seed(42)
    np.random.seed(42)


def test_training_reduces_loss_on_simple_signal():
    # Make an easy-to-learn signal (sine waves + small noise) so loss should drop quickly
    T, F = 128, 3
    batch = 64

    t = np.linspace(0, 2 * np.pi, T).astype(np.float32)
    base = np.stack([np.sin(t), np.cos(t), np.sin(2 * t)], axis=-1)
    x = np.tile(base[None, :, :], (batch, 1, 1))
    x += 0.05 * np.random.randn(batch, T, F).astype(np.float32)

    enc = encoder_1dcnn(timesteps=T, n_features=F, base_filters=8, kernel_size=7, emb_dim=16, dropout=0.0)
    ae = training_autoencoder(encoder=enc, timesteps=T, n_features=F, base_filters=8, kernel_size=7, dropout=0.0, lr=1e-3)

    loss0, mae0 = ae.evaluate(x, x, verbose=0)
    ae.fit(x, x, epochs=5, batch_size=16, verbose=0)

    loss1, mae1 = ae.evaluate(x, x, verbose=0)

    assert loss1 < loss0, f"Expected loss to drop. Before={loss0:.6f} After={loss1:.6f}"
    assert mae1 < mae0, f"Expected MAE to drop. Before={mae0:.6f} After={mae1:.6f}"