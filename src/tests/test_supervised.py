import numpy as np
import pytest
import tensorflow as tf

from eegproc.supervised import lstm_classifier, bilstm_classifier


@pytest.fixture(autouse=True)
def _set_seeds():
    tf.keras.utils.set_random_seed(42)
    np.random.seed(42)


def test_lstm_classifier_builds_and_predicts_correct_shape():
    T, F, C = 64, 8, 3
    model = lstm_classifier(
        timesteps=T, n_features=F, n_classes=C,
        lstm_units=8, n_lstm_layers=1, dropout=0.0,
    )
    x = np.random.randn(4, T, F).astype("float32")
    probs = model.predict(x, verbose=0)

    assert probs.shape == (4, C)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_bilstm_classifier_compiles_with_default_loss():
    model = bilstm_classifier(
        timesteps=64, n_features=8, n_classes=3,
        lstm_units=8, n_bilstm_layers=1, dropout=0.0,
    )
    assert isinstance(model.loss, tf.keras.losses.SparseCategoricalCrossentropy)
    assert model.optimizer is not None


def test_variational_loss_not_implemented():
    with pytest.raises(NotImplementedError):
        lstm_classifier(
            timesteps=64, n_features=8, n_classes=3,
            loss="variational",
        )
