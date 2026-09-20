"""Small runtime smoke test for GraphConvMTL + GCNMTL.

Run from an environment where these modules are installed in EEGProc, or adapt
imports if testing the files directly. The test checks:
  * MI adjacency construction
  * encoder forward pass
  * MTL-normalized adjacency symmetry
  * gradient flow through trainable GCN/GRU parameters
  * companion decoder output shape
"""

import numpy as np
import tensorflow as tf

from src.eegproc.deep_learning.encoders.GCNMTL import (
    GCNMTLEncoder,
    GCNMTLDecoder,
    compute_mtl_shared_mi_adjacency,
)


def main():
    tf.keras.utils.set_random_seed(42)

    batch = 8
    timesteps = 64
    n_channels = 14
    n_bands = 3
    n_features = n_channels * n_bands

    # Channel-major flattened layout expected by EEGProc:
    # [ch0_band0, ch0_band1, ..., ch1_band0, ...]
    x_np = np.random.default_rng(42).normal(
        size=(batch, timesteps, n_features)
    ).astype(np.float32)

    A = compute_mtl_shared_mi_adjacency(
        x_np,
        n_channels=n_channels,
        n_bands=n_bands,
        n_neighbors=3,
        random_state=42,
        zero_diagonal=False,
    )

    encoder = GCNMTLEncoder(
        timesteps=timesteps,
        t_down=2,
        adjacency=A,
        n_channels=n_channels,
        n_bands=n_bands,
        gcn_units=(16, 8),
        emb_dim=32,
        dropout=0.0,
        use_batch_norm=False,
        use_spectral_gru=True,
        spectral_gru_units=16,
        spectral_gru_dropout=0.0,
    )

    x = tf.convert_to_tensor(x_np)
    with tf.GradientTape() as tape:
        z = encoder(x, training=True)
        loss = tf.reduce_mean(tf.square(z))

    grads = tape.gradient(loss, encoder.trainable_variables)
    non_none_grads = sum(g is not None for g in grads)

    expected_shape = (batch, timesteps // 2, 32)
    assert tuple(z.shape) == expected_shape, (z.shape, expected_shape)

    A_raw = encoder.get_raw_adjacency_matrix().numpy()
    A_hat = encoder.get_adjacency_matrix().numpy()
    assert np.allclose(A_raw, A_raw.T, atol=1e-6)
    assert np.allclose(A_hat, A_hat.T, atol=1e-6)
    assert np.all(np.isfinite(A_hat))
    assert non_none_grads == len(encoder.trainable_variables)

    decoder = GCNMTLDecoder.from_encoder(encoder)
    x_hat = decoder(z, training=False)
    assert tuple(x_hat.shape) == (batch, timesteps, n_features)

    print("GCNMTL smoke test PASSED")
    print("input:", x.shape)
    print("adjacency:", A.shape)
    print("latent:", z.shape)
    print("reconstruction:", x_hat.shape)
    print("trainable variables:", len(encoder.trainable_variables))
    print("variables with gradients:", non_none_grads)


if __name__ == "__main__":
    main()
