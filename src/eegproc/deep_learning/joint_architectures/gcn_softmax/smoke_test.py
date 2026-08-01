"""Build the GCN-only model and run one synthetic training step."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from .model import build_gcn_softmax_classifier


def main() -> int:
    tf.keras.utils.set_random_seed(42)
    x = np.random.default_rng(42).normal(size=(8, 32, 56)).astype(np.float32)
    y = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    model = build_gcn_softmax_classifier(
        input_shape=(32, 56),
        n_classes=2,
        n_channels=14,
        n_bands=4,
        gcn_units=(32, 16),
        temporal_pool_sizes=(2,),
        t_down=2,
        emb_dim=32,
    )
    model.train_on_batch(x, y)
    logits = model(x, training=False)
    probabilities = tf.nn.softmax(logits, axis=-1)
    if probabilities.shape != (8, 2):
        raise RuntimeError(f"Unexpected output shape: {probabilities.shape}")
    tf.debugging.assert_near(
        tf.reduce_sum(probabilities, axis=-1), tf.ones((8,), dtype=probabilities.dtype)
    )
    print("GCN-only model smoke test passed.")
    model.summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
