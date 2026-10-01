"""CNN/GNN encoders and decoders, Keras registration, and the VAE loss."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="models require eegproc[deep-learning]")

from eegproc.deep_learning.unsupervised.Convolutions.CNN1D import CNN1DDecoder, CNN1DEncoder  # noqa: E402
from eegproc.deep_learning.unsupervised.Convolutions.CNN2D import CNN2DDecoder, CNN2DEncoder  # noqa: E402
from eegproc.deep_learning.unsupervised.GNN import GCN, GCN_band_separated, GCNMTL  # noqa: E402
from eegproc.deep_learning.unsupervised.VariationalAutoencoderLoss import (  # noqa: E402
    GradientReversal,
    VariationalAutoencoderLoss,
)

TIMESTEPS, T_DOWN, N_CHANNELS, N_BANDS = 8, 2, 4, 3


def _encoder_cases():
    flat = N_CHANNELS * N_BANDS
    return {
        "CNN1D": (
            lambda: CNN1DEncoder(timesteps=TIMESTEPS, n_features=6, t_down=T_DOWN,
                                 conv_filters=(4, 8), kernel_sizes=3, emb_dim=5),
            CNN1DDecoder, 6,
        ),
        "CNN2D": (
            lambda: CNN2DEncoder(timesteps=TIMESTEPS, t_down=T_DOWN, n_channels=N_CHANNELS,
                                 n_bands=N_BANDS, conv_filters=(4, 8),
                                 kernel_sizes=((2, 2), (2, 2)), emb_dim=5),
            CNN2DDecoder, flat,
        ),
        "GCN": (
            lambda: GCN.GCNEncoder(timesteps=TIMESTEPS, t_down=T_DOWN, n_channels=N_CHANNELS,
                                   n_bands=N_BANDS, gcn_units=(4,), emb_dim=5),
            GCN.GCNDecoder, flat,
        ),
        "BandSeparatedGCN": (
            lambda: GCN_band_separated.BandSeparatedGCNEncoder(
                timesteps=TIMESTEPS, t_down=T_DOWN, n_channels=N_CHANNELS,
                n_bands=N_BANDS, gcn_units=(4,), emb_dim=5),
            GCN_band_separated.GCNDecoder, flat,
        ),
        "GCNMTL": (
            lambda: GCNMTL.GCNMTLEncoder(timesteps=TIMESTEPS, t_down=T_DOWN,
                                         adjacency=np.eye(N_CHANNELS, dtype="float32"),
                                         n_channels=N_CHANNELS, n_bands=N_BANDS,
                                         gcn_units=(4,), spectral_gru_units=6),
            GCNMTL.GCNMTLDecoder, flat,
        ),
    }


@pytest.mark.parametrize("name", list(_encoder_cases()))
def test_encoder_decoder_round_trip_shapes(name):
    make_encoder, decoder_type, n_features = _encoder_cases()[name]
    encoder = make_encoder()
    x = np.random.default_rng(0).standard_normal((2, TIMESTEPS, n_features)).astype("float32")

    latent = encoder(x)
    reconstruction = decoder_type.from_encoder(encoder)(latent)

    assert tuple(latent.shape) == (2, TIMESTEPS // T_DOWN, encoder.emb_dim)
    assert tuple(reconstruction.shape) == x.shape


def test_gcn_decoders_register_under_distinct_names():
    """Regression: both GCN decoders used to claim "eegproc>GCNDecoder"."""
    get = tf.keras.utils.get_registered_object
    assert get("eegproc>GCNDecoder") is GCN.GCNDecoder
    assert get("eegproc>BandSeparatedGCNDecoder") is GCN_band_separated.GCNDecoder
    assert GCN.GCNDecoder is not GCN_band_separated.GCNDecoder


def test_band_separated_autoencoder_survives_save_and_load(tmp_path):
    encoder = GCN_band_separated.BandSeparatedGCNEncoder(
        timesteps=TIMESTEPS, t_down=T_DOWN, n_channels=N_CHANNELS, n_bands=N_BANDS,
        gcn_units=(4,), emb_dim=5,
    )
    decoder = GCN_band_separated.GCNDecoder.from_encoder(encoder)
    inputs = tf.keras.Input((TIMESTEPS, N_CHANNELS * N_BANDS))
    model = tf.keras.Model(inputs, decoder(encoder(inputs)))
    x = np.random.default_rng(1).standard_normal((2, TIMESTEPS, N_CHANNELS * N_BANDS)).astype("float32")

    path = tmp_path / "autoencoder.keras"
    model.save(path)
    restored = tf.keras.models.load_model(path)

    np.testing.assert_allclose(restored(x), model(x), atol=1e-6)
    assert type(restored.layers[-1]) is GCN_band_separated.GCNDecoder


def test_vae_loss_on_sequence_latents_gives_one_value_per_sample():
    """Regression: KL used to keep the time axis and break the sum with reconstruction."""
    loss = VariationalAutoencoderLoss(reconstruction="mse", beta=1.0)
    x = tf.ones((2, 4, 3))

    perfect = loss(x, x, tf.zeros((2, 4, 5)), tf.zeros((2, 4, 5)))
    shifted = loss(x, tf.zeros_like(x), tf.ones((2, 4, 5)), tf.zeros((2, 4, 5)))

    assert float(perfect["total_loss"]) == pytest.approx(0.0)
    # MSE of 1 everywhere, plus KL(N(1, 1) || N(0, 1)) = 0.5 per coordinate.
    assert float(shifted["reconstruction_loss"]) == pytest.approx(1.0)
    assert float(shifted["kl_loss"]) == pytest.approx(0.5)
    assert float(shifted["total_loss"]) == pytest.approx(1.5)
    assert shifted["kl_loss_per_sample"].shape == (2,)


def test_vae_kl_on_flat_latents_reduces_the_latent_axis():
    loss = VariationalAutoencoderLoss(kl_reduction="mean")
    z_mean = tf.constant([[1.0, 0.0], [0.5, -0.5]])
    z_log_var = tf.constant([[0.0, 0.2], [-0.1, 0.0]])

    expected = (0.5 * (z_mean**2 + tf.exp(z_log_var) - 1 - z_log_var)).numpy().mean(axis=-1)
    np.testing.assert_allclose(loss.compute_kl_loss(z_mean, z_log_var), expected, rtol=1e-6)


def test_gradient_reversal_is_identity_forward_and_flips_gradients():
    layer = GradientReversal(adversarial_weight=0.5)
    x = tf.Variable([[1.0, 2.0]])
    with tf.GradientTape() as tape:
        y = tf.reduce_sum(layer(x) * 3.0)

    np.testing.assert_allclose(layer(x), x)
    np.testing.assert_allclose(tape.gradient(y, x), [[-1.5, -1.5]])
