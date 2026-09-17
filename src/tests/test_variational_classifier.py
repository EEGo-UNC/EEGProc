"""Focused tests for variational-classifier logit temperature scaling."""

import numpy as np
import pytest


tf = pytest.importorskip("tensorflow")

from eegproc.deep_learning.supervised.variational_classifier import (  # noqa: E402
    VariationalClassifier,
)


def test_logit_scale_matches_summed_temperature():
    mean_head = VariationalClassifier(n_classes=2, logit_scale=1.0)
    scaled_head = VariationalClassifier(n_classes=2, logit_scale=128.0)
    mean_head.build((None, 4))
    scaled_head.build((None, 4))
    scaled_head.set_weights(mean_head.get_weights())

    features = tf.constant([[0.2, -0.4, 0.7, 1.1]], dtype=tf.float32)
    mean_logits = mean_head(features)
    scaled_logits = scaled_head(features)

    np.testing.assert_allclose(
        scaled_logits.numpy(),
        128.0 * mean_logits.numpy(),
        rtol=1e-6,
        atol=1e-6,
    )
    assert scaled_head.get_config()["logit_scale"] == pytest.approx(128.0)
