"""Smoke tests for the joint autoencoder + variational-classifier model.

This file provides two minimal checks:
1. Forward pass shape and finiteness checks.
2. Single training-step check with gradient-flow assertions for encoder,
   decoder, and variational-classifier branches.
"""

from __future__ import annotations

import sys
from pathlib import Path

import tensorflow as tf

try:
    from .joint_autoencoder_vc import JointAutoencoderVariationalClassifier
    from ..supervised.variational_classifier import VariationalClassifier
    from ..unsupervised.Convolutions.CNN1D import CNN1DDecoder, CNN1DEncoder
except ImportError:
    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))

    from joint_autoencoder_vc import JointAutoencoderVariationalClassifier
    from eegproc.deep_learning.supervised.variational_classifier import (
        VariationalClassifier,
    )
    from eegproc.deep_learning.unsupervised.Convolutions.CNN1D import (
        CNN1DDecoder,
        CNN1DEncoder,
    )


def _make_model(
    timesteps: int = 32,
    n_features: int = 8,
    n_classes: int = 2,
) -> JointAutoencoderVariationalClassifier:
    encoder = CNN1DEncoder(
        timesteps=timesteps,
        n_features=n_features,
        t_down=2,
        conv_filters=(16, 32),
        kernel_sizes=(5, 3),
        pool_after_layers=(0,),
        pool_sizes=(2,),
        emb_dim=16,
        dropout=0.0,
        use_batch_norm=False,
        name="smoke_encoder",
    )

    decoder = CNN1DDecoder.from_encoder(encoder, name="smoke_decoder")
    vc_head = VariationalClassifier(n_classes=n_classes, name="smoke_vc_head")

    model = JointAutoencoderVariationalClassifier(
        encoder=encoder,
        decoder=decoder,
        variational_classifier=vc_head,
        ae_loss_weight=0.5,
        vc_loss_weight=0.5,
        vc_gamma=1e-4,
        name="smoke_joint_ae_vc",
    )

    return model


def _assert_any_finite_nonzero_grad(grads) -> None:
    finite_nonzero = False

    for grad in grads:
        if grad is None:
            continue

        if isinstance(grad, tf.IndexedSlices):
            values = grad.values
        else:
            values = grad

        if tf.reduce_any(tf.math.is_finite(values)) and tf.reduce_any(
            tf.not_equal(values, 0.0)
        ):
            finite_nonzero = True
            break

    assert finite_nonzero, "Expected at least one finite non-zero gradient."


def smoke_test_joint_forward_shapes_and_finite_outputs() -> None:
    """Run one forward pass and verify output contract."""
    tf.random.set_seed(123)

    batch_size = 4
    timesteps = 32
    n_features = 8

    model = _make_model(timesteps=timesteps, n_features=n_features, n_classes=2)

    x = tf.random.normal((batch_size, timesteps, n_features), dtype=tf.float32)
    outputs = model(x, training=False)

    assert set(outputs.keys()) == {
        "latent_sequence",
        "pooled_latent",
        "logits",
        "reconstruction",
    }

    latent_sequence = outputs["latent_sequence"]
    pooled_latent = outputs["pooled_latent"]
    logits = outputs["logits"]
    reconstruction = outputs["reconstruction"]

    assert latent_sequence.shape[0] == batch_size
    assert latent_sequence.shape[-1] == 16

    assert pooled_latent.shape == (batch_size, 16)
    assert logits.shape == (batch_size, 2)
    assert reconstruction.shape == (batch_size, timesteps, n_features)

    for tensor in (latent_sequence, pooled_latent, logits, reconstruction):
        assert bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy())


def smoke_test_joint_single_train_step_and_gradient_flow() -> None:
    """Run one optimization step and verify gradient flow to all branches."""
    tf.random.set_seed(456)

    batch_size = 6
    timesteps = 32
    n_features = 8

    model = _make_model(timesteps=timesteps, n_features=n_features, n_classes=2)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

    x = tf.random.normal((batch_size, timesteps, n_features), dtype=tf.float32)
    y = tf.random.uniform((batch_size,), minval=0, maxval=2, dtype=tf.int32)

    with tf.GradientTape(persistent=True) as tape:
        total_loss, reconstruction_loss, vc_loss = model._compute_weighted_losses(
            x=x,
            y=y,
            training=True,
        )

    encoder_vars = model.encoder.trainable_variables
    decoder_vars = model.decoder.trainable_variables
    classifier_vars = model.variational_classifier.trainable_variables

    encoder_grads = tape.gradient(total_loss, encoder_vars)
    decoder_grads = tape.gradient(total_loss, decoder_vars)
    classifier_grads = tape.gradient(total_loss, classifier_vars)

    del tape

    _assert_any_finite_nonzero_grad(encoder_grads)
    _assert_any_finite_nonzero_grad(decoder_grads)
    _assert_any_finite_nonzero_grad(classifier_grads)

    metrics = model.train_step((x, y))

    assert "loss" in metrics
    assert "reconstruction_loss" in metrics
    assert "vc_loss" in metrics

    for key in ("loss", "reconstruction_loss", "vc_loss"):
        value = tf.convert_to_tensor(metrics[key])
        assert bool(tf.reduce_all(tf.math.is_finite(value)).numpy())
        assert bool(tf.greater_equal(value, 0.0).numpy())

    assert bool(tf.math.is_finite(total_loss).numpy())
    assert bool(tf.math.is_finite(reconstruction_loss).numpy())
    assert bool(tf.math.is_finite(vc_loss).numpy())


if __name__ == "__main__":
    smoke_test_joint_forward_shapes_and_finite_outputs()
    smoke_test_joint_single_train_step_and_gradient_flow()
    print("Joint AE+VC smoke tests passed.")
