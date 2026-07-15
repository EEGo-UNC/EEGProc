import numpy as np
import pytest
import tensorflow as tf

from eegproc.deep_learning.supervised import lstm_classifier, bilstm_classifier


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


def test_loso_cv_end_to_end_with_dumb_hyperparams():
    """Smoke test: rewritten loso_cv runs through every fold without raising."""
    from eegproc.deep_learning.archival_cv import loso_cv

    n_subj = 4
    windows_per_subj = 6
    T, F, C = 16, 8, 2
    n_total = n_subj * windows_per_subj

    X = np.random.randn(n_total, T, F).astype("float32")
    y = np.random.randint(0, C, size=n_total)
    subject_ids = np.repeat(np.arange(n_subj), windows_per_subj)

    def build_model(**hparams):
        return lstm_classifier(
            timesteps=T, n_features=F, n_classes=C,
            lstm_units=4, n_lstm_layers=1, dropout=0.0,
        )

    results = loso_cv(
        model_builder_function=build_model,
        feature_array=X,
        label_array=y,
        subject_id_array=subject_ids,
        hyperparameters={"epochs": 1, "batch_size": 2},
    )

    assert {"fold_results", "mean_scores", "std_scores"} <= results.keys()
    assert len(results["fold_results"]) == n_subj
    for fold in results["fold_results"]:
        assert fold["n_test_windows"] > 0
        assert fold["n_train_windows"] > 0
        assert fold["n_val_windows"] == 0
    for metric_name, metric_value in results["mean_scores"].items():
        assert np.isfinite(metric_value), f"{metric_name} not finite: {metric_value}"
