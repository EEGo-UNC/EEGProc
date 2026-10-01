"""RNN classifier builders, classification heads, and the contrastive loss."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="models require eegproc[deep-learning]")

from eegproc.deep_learning.supervised.rnn_architectures import (  # noqa: E402
    BiGRUClassifier,
    BiLSTMClassifier,
    GRUClassifier,
    LSTMClassifier,
)
from eegproc.deep_learning.supervised.supervised_contrastive_loss import (  # noqa: E402
    SupervisedContrastiveLoss,
)
from eegproc.deep_learning.supervised.variational_classifier import (  # noqa: E402
    DenseClassifier,
    HybridClassifier,
    VariationalClassifier,
)

TIMESTEPS, N_FEATURES, N_CLASSES = 8, 3, 2

BUILDERS = [
    (LSTMClassifier, {"lstm_units": 4, "n_lstm_layers": 1}),
    (BiLSTMClassifier, {"lstm_units": 4, "n_bilstm_layers": 1}),
    (GRUClassifier, {"gru_units": 4, "n_gru_layers": 1}),
    (BiGRUClassifier, {"gru_units": 4, "n_bigru_layers": 1}),
]


def _data(n=16, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, N_CLASSES, n)
    x = (rng.standard_normal((n, TIMESTEPS, N_FEATURES)) + y[:, None, None]).astype("float32")
    return x, y


@pytest.mark.parametrize("loss", ["softmax_crossentropy", "variational"])
@pytest.mark.parametrize("builder,units", BUILDERS, ids=[b.__name__ for b, _ in BUILDERS])
def test_classifier_builds_trains_and_predicts_probabilities(builder, units, loss):
    tf.keras.utils.set_random_seed(0)
    model = builder(
        timesteps=TIMESTEPS, n_features=N_FEATURES, n_classes=N_CLASSES,
        dropout=0.0, loss=loss, **units,
    ).build()
    x, y = _data()

    history = model.fit(x, y, epochs=1, batch_size=8, verbose=0)
    probabilities = model.predict(x[:4], verbose=0)

    assert np.isfinite(history.history["loss"][0])
    assert probabilities.shape == (4, N_CLASSES)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5)


@pytest.mark.parametrize("builder,units", BUILDERS, ids=[b.__name__ for b, _ in BUILDERS])
def test_feature_extractor_and_summarizer_return_one_embedding_per_sample(builder, units):
    rnn = builder(timesteps=TIMESTEPS, n_features=N_FEATURES, n_classes=N_CLASSES, **units)
    x, _ = _data(n=5)

    for model in (rnn.build_feature_extractor(), rnn.build_sequence_summarizer()):
        embedding = model(x)
        assert embedding.shape.rank == 2
        assert embedding.shape[0] == 5


def test_variational_loss_uses_the_heads_logits_not_its_probabilities():
    """Regression: the compiled loss once applied softmax to the softmax output."""
    tf.keras.utils.set_random_seed(0)
    model = BiLSTMClassifier(
        timesteps=TIMESTEPS, n_features=N_FEATURES, n_classes=N_CLASSES,
        lstm_units=4, n_bilstm_layers=1, dropout=0.0, loss="variational", name="probe",
    ).build()
    x, y = _data()

    compiled = model.evaluate(x, y, batch_size=len(x), verbose=0)
    compiled = compiled[0] if isinstance(compiled, list) else compiled

    head = model.get_layer("probe_variational_classifier")
    latent = tf.keras.Model(model.inputs, head.input)(x, training=False)
    logits = head(latent)
    expected = float(head.vc_loss(mh=latent, y=y, logits=logits))
    double_softmax = float(head.vc_loss(mh=latent, y=y, logits=tf.nn.softmax(logits)))

    assert compiled == pytest.approx(expected, rel=1e-5)
    assert compiled != pytest.approx(double_softmax, rel=1e-5)


def test_logit_scale_matches_summed_temperature():
    mean_head = VariationalClassifier(n_classes=2, logit_scale=1.0)
    scaled_head = VariationalClassifier(n_classes=2, logit_scale=128.0)
    mean_head.build((None, 4))
    scaled_head.build((None, 4))
    scaled_head.set_weights(mean_head.get_weights())

    features = tf.constant([[0.2, -0.4, 0.7, 1.1]], dtype=tf.float32)
    np.testing.assert_allclose(
        scaled_head(features).numpy(),
        128.0 * mean_head(features).numpy(),
        rtol=1e-6,
        atol=1e-6,
    )
    assert scaled_head.get_config()["logit_scale"] == pytest.approx(128.0)


@pytest.mark.parametrize("head_type", [DenseClassifier, HybridClassifier, VariationalClassifier])
def test_heads_share_the_loss_component_interface(head_type):
    head = head_type(n_classes=2)
    latent = tf.constant(np.random.default_rng(0).standard_normal((6, 4)), tf.float32)
    components = head.vc_loss_components(mh=latent, y=tf.constant([0, 0, 1, 1, 0, 1]))

    weighted = (
        components["weighted_focal_loss"]
        + components["weighted_latent_posterior_kl"]
        + components["weighted_discriminator_kl"]
        + components["weighted_class_prior_kl"]
    )
    assert float(components["total_loss"]) == pytest.approx(float(weighted), rel=1e-6)
    if head_type is DenseClassifier:
        assert float(components["latent_posterior_kl"]) == 0.0


def test_contrastive_loss_counts_only_cross_subject_positives():
    loss = SupervisedContrastiveLoss()
    embeddings = tf.constant(np.random.default_rng(0).standard_normal((6, 4)), tf.float32)
    labels = tf.constant([0, 0, 1, 1, 0, 1])

    mixed = loss(embeddings, labels=labels, subject_ids=tf.constant([0, 1, 0, 1, 2, 2]))
    same_subject = loss(embeddings, labels=labels, subject_ids=tf.zeros(6, tf.int32))

    # Each class has one sample from each of three subjects: 3 * 2 directed pairs.
    assert float(mixed["positive_pairs"]) == 12.0
    assert float(mixed["valid_anchor_fraction"]) == 1.0
    assert np.isfinite(float(mixed["loss"])) and float(mixed["loss"]) > 0.0
    # With no cross-subject positive, the loss is defined as zero.
    assert float(same_subject["loss"]) == 0.0
    assert float(same_subject["positive_pairs"]) == 0.0
