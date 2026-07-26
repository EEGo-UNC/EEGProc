"""Classification heads used by EEGProc joint and standalone models.

``VariationalClassifier`` maintains learned Gaussian class priors and
classifies via Bayes' rule, with an optional auxiliary discriminator for
latent-space alignment. ``DenseClassifier`` is a standard trainable linear
logit head that exposes the same loss-component interface, allowing the joint
training pipeline to switch heads without changing its custom train/test steps.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="EEGProc")
class DenseClassifier(tf.keras.layers.Layer):
    """Standard dense logit head with the VC-compatible loss interface.

    The joint model historically expects its classification head to expose
    ``n_classes``, ``vc_loss_components()``, and ``discriminator_loss()``.
    This adapter provides those methods while optimizing only weighted sparse
    categorical cross-entropy. All variational regularization components are
    returned as exact zeros, regardless of the supplied beta/gamma/lambda
    values, so selecting this head is an unambiguous dense-classifier ablation.
    """

    supports_variational_regularization = False
    supports_discriminator = False

    def __init__(
        self,
        n_classes: int = 2,
        use_bias: bool = True,
        kernel_initializer: str | dict = "glorot_uniform",
        bias_initializer: str | dict = "zeros",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if n_classes < 2:
            raise ValueError("n_classes must be at least 2.")
        self.n_classes = int(n_classes)
        self.use_bias = bool(use_bias)
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.logits_layer = tf.keras.layers.Dense(
            self.n_classes,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name="dense_class_logits",
        )

    def call(self, features: tf.Tensor, training: bool = False) -> tf.Tensor:
        return self.logits_layer(features, training=training)

    @staticmethod
    def _class_ids(y: tf.Tensor) -> tf.Tensor:
        y_tensor = tf.convert_to_tensor(y)
        if (
            y_tensor.shape.rank == 2
            and y_tensor.shape[-1] is not None
            and y_tensor.shape[-1] > 1
        ):
            return tf.argmax(y_tensor, axis=-1, output_type=tf.int32)
        return tf.cast(tf.reshape(y_tensor, [-1]), tf.int32)

    @staticmethod
    def _weighted_mean(
        values: tf.Tensor,
        sample_weight: tf.Tensor | None = None,
    ) -> tf.Tensor:
        values = tf.reshape(tf.convert_to_tensor(values), [-1])
        if sample_weight is None:
            return tf.reduce_mean(values)

        weights = tf.cast(tf.reshape(sample_weight, [-1]), values.dtype)
        tf.debugging.assert_equal(
            tf.shape(values)[0],
            tf.shape(weights)[0],
            message="sample_weight must align with the batch.",
        )
        return tf.math.divide_no_nan(
            tf.reduce_sum(values * weights),
            tf.reduce_sum(weights),
        )

    def vc_loss_components(
        self,
        mh: tf.Tensor,
        y: tf.Tensor,
        alpha: float = 1.0,
        beta: float = 0.0,
        gamma: float = 0.0,
        lambda_: float = 0.0,
        logits: tf.Tensor | None = None,
        sample_weight: tf.Tensor | None = None,
    ) -> dict[str, tf.Tensor]:
        """Return cross-entropy plus zero-valued VC regularization terms."""
        del beta, gamma, lambda_
        y = self._class_ids(y)
        if logits is None:
            logits = self(mh, training=True)

        per_sample_cross_entropy = (
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=y,
                logits=logits,
            )
        )
        cross_entropy = self._weighted_mean(
            per_sample_cross_entropy,
            sample_weight=sample_weight,
        )
        weighted_cross_entropy = (
            tf.cast(alpha, cross_entropy.dtype) * cross_entropy
        )
        zero = tf.zeros((), dtype=cross_entropy.dtype)

        return {
            "total_loss": weighted_cross_entropy,
            "cross_entropy": cross_entropy,
            "weighted_cross_entropy": weighted_cross_entropy,
            "latent_posterior_kl": zero,
            "weighted_latent_posterior_kl": zero,
            "discriminator_kl": zero,
            "weighted_discriminator_kl": zero,
            "class_prior_kl": zero,
            "weighted_class_prior_kl": zero,
        }

    def vc_loss(
        self,
        mh: tf.Tensor,
        y: tf.Tensor,
        alpha: float = 1.0,
        beta: float = 0.0,
        gamma: float = 0.0,
        lambda_: float = 0.0,
        logits: tf.Tensor | None = None,
        sample_weight: tf.Tensor | None = None,
    ) -> tf.Tensor:
        return self.vc_loss_components(
            mh=mh,
            y=y,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            lambda_=lambda_,
            logits=logits,
            sample_weight=sample_weight,
        )["total_loss"]

    def discriminator_loss(self, mh: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        del y
        return tf.zeros((), dtype=mh.dtype)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "n_classes": self.n_classes,
                "use_bias": self.use_bias,
                "kernel_initializer": tf.keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": tf.keras.initializers.serialize(
                    self.bias_initializer
                ),
            }
        )
        return config


class VariationalClassifier(tf.keras.layers.Layer):
    """Variational classification head with separately reportable loss terms."""

    supports_variational_regularization = True
    supports_discriminator = True

    def __init__(
        self,
        n_classes: int = 2,
        latent_dim: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if n_classes < 2:
            raise ValueError("n_classes must be at least 2.")
        self.n_classes = int(n_classes)
        self.latent_dim = latent_dim
        self._last_mh = None

    def build(self, input_shape) -> None:
        latent_dim = input_shape[-1]
        if latent_dim is None:
            raise ValueError("The classifier input must have a static last dimension.")
        self.latent_dim = int(latent_dim)

        self.prior_mu = self.add_weight(
            name="prior_mu",
            shape=(self.n_classes, self.latent_dim),
            initializer="glorot_normal",
            trainable=True,
        )
        self.prior_log_sigma = self.add_weight(
            name="prior_log_sigma",
            shape=(self.n_classes, self.latent_dim),
            initializer="zeros",
            trainable=True,
        )
        self.log_class_prior = self.add_weight(
            name="log_class_prior",
            shape=(self.n_classes,),
            initializer="zeros",
            trainable=True,
        )
        self.disc_w = self.add_weight(
            name="disc_w",
            shape=(self.n_classes, self.latent_dim),
            initializer="glorot_normal",
            trainable=True,
        )
        self.disc_b = self.add_weight(
            name="disc_b",
            shape=(self.n_classes,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    @staticmethod
    def _class_ids(y: tf.Tensor) -> tf.Tensor:
        y_tensor = tf.convert_to_tensor(y)
        if (
            y_tensor.shape.rank == 2
            and y_tensor.shape[-1] is not None
            and y_tensor.shape[-1] > 1
        ):
            return tf.argmax(y_tensor, axis=-1, output_type=tf.int32)
        return tf.cast(tf.reshape(y_tensor, [-1]), tf.int32)

    @staticmethod
    def _weighted_mean(
        values: tf.Tensor,
        sample_weight: tf.Tensor | None = None,
    ) -> tf.Tensor:
        values = tf.convert_to_tensor(values)
        if sample_weight is None:
            return tf.reduce_mean(values)

        weights = tf.cast(tf.reshape(sample_weight, [-1]), values.dtype)
        values = tf.reshape(values, [-1])
        tf.debugging.assert_equal(
            tf.shape(values)[0],
            tf.shape(weights)[0],
            message="sample_weight must align with the batch.",
        )
        denominator = tf.maximum(
            tf.reduce_sum(weights),
            tf.cast(tf.keras.backend.epsilon(), values.dtype),
        )
        return tf.reduce_sum(values * weights) / denominator

    def _log_gaussian(
        self,
        z: tf.Tensor,
        mu: tf.Tensor,
        log_sigma: tf.Tensor,
    ) -> tf.Tensor:
        sigma2 = tf.exp(2.0 * log_sigma)
        diff = z - mu[tf.newaxis, :]
        # Mean over latent dimensions keeps the classifier-logit scale stable
        # when the BiLSTM feature width changes.
        return -0.5 * tf.reduce_mean(
            tf.math.log(2.0 * np.pi * sigma2) + tf.square(diff) / sigma2,
            axis=-1,
        )

    def call(self, mh: tf.Tensor, training: bool = False) -> tf.Tensor:
        self._last_mh = mh

        log_class_prior = tf.nn.log_softmax(self.log_class_prior)

        log_likelihoods = tf.stack(
            [
                self._log_gaussian(
                    mh,
                    self.prior_mu[class_index],
                    self.prior_log_sigma[class_index],
                )
                for class_index in range(self.n_classes)
            ],
            axis=1,
        )

        latent_dim = tf.cast(tf.shape(mh)[-1], mh.dtype)
        normalized_log_prior = log_class_prior / latent_dim

        return log_likelihoods + normalized_log_prior[tf.newaxis, :]
    
    def discriminator(self, z: tf.Tensor, y: int) -> tf.Tensor:
        """Return the trainable discriminator score T_psi^y(z)."""
        return tf.linalg.matvec(z, self.disc_w[y]) + self.disc_b[y]

    def _discriminator_for_encoder(self, z: tf.Tensor, y: int) -> tf.Tensor:
        """Score z while freezing discriminator parameters, not z itself.

        Stopping gradients on the complete score would also stop the gradient
        into the encoder. Freezing only ``disc_w`` and ``disc_b`` preserves the
        intended representation-learning signal from the discriminator term.
        """
        frozen_w = tf.stop_gradient(self.disc_w[y])
        frozen_b = tf.stop_gradient(self.disc_b[y])
        return tf.linalg.matvec(z, frozen_w) + frozen_b

    def _gaussian_kl_latent_posterior(
        self,
        mh: tf.Tensor,
        y: tf.Tensor,
    ) -> tf.Tensor:
        """Estimate mean class-conditional KL(q_phi(z|y) || p_theta(z|y))."""
        mh = tf.convert_to_tensor(mh)
        y = self._class_ids(y)

        dtype = mh.dtype
        eps = tf.cast(1e-6, dtype)
        batch_size = tf.cast(tf.shape(mh)[0], dtype)
        expected_kl = tf.zeros((), dtype=dtype)
        valid_probability_mass = tf.zeros((), dtype=dtype)

        for class_index in range(self.n_classes):
            mask = tf.equal(y, class_index)
            z_class = tf.boolean_mask(mh, mask)
            n_class = tf.shape(z_class)[0]

            def compute_class_kl():
                mean_q = tf.reduce_mean(z_class, axis=0)
                centered = z_class - mean_q[tf.newaxis, :]
                variance_q = tf.maximum(
                    tf.reduce_mean(tf.square(centered), axis=0),
                    eps,
                )

                mean_p = tf.cast(self.prior_mu[class_index], dtype)
                variance_p = tf.maximum(
                    tf.exp(
                        2.0
                        * tf.cast(self.prior_log_sigma[class_index], dtype)
                    ),
                    eps,
                )

                class_kl = 0.5 * tf.reduce_mean(
                    tf.math.log(variance_p)
                    - tf.math.log(variance_q)
                    + (
                        variance_q + tf.square(mean_q - mean_p)
                    )
                    / variance_p
                    - 1.0
                )
                class_probability = tf.cast(n_class, dtype) / batch_size
                return class_probability * class_kl, class_probability

            weighted_class_kl, class_probability = tf.cond(
                tf.greater_equal(n_class, 2),
                true_fn=compute_class_kl,
                false_fn=lambda: (
                    tf.zeros((), dtype=dtype),
                    tf.zeros((), dtype=dtype),
                ),
            )
            expected_kl += weighted_class_kl
            valid_probability_mass += class_probability

        return tf.cond(
            tf.greater(valid_probability_mass, 0.0),
            true_fn=lambda: expected_kl / valid_probability_mass,
            false_fn=lambda: tf.zeros((), dtype=dtype),
        )

    def vc_loss_components(
        self,
        mh: tf.Tensor,
        y: tf.Tensor,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.0,
        lambda_: float = 0.0,
        logits: tf.Tensor | None = None,
        sample_weight: tf.Tensor | None = None,
    ) -> dict[str, tf.Tensor]:
        """Return every raw and weighted component of the VC objective.

        The returned ``total_loss`` is exactly the sum of the four weighted
        terms. Passing the logits already produced by the joint model avoids a
        duplicate classifier call and guarantees that the logged cross-entropy
        corresponds to the logits used for the accuracy metric.
        """
        y = self._class_ids(y)
        y_onehot = tf.one_hot(y, self.n_classes, dtype=mh.dtype)
        if logits is None:
            logits = self(mh, training=True)

        cross_entropy_per_sample = (
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=y,
                logits=logits,
            )
        )
        cross_entropy = self._weighted_mean(
            cross_entropy_per_sample,
            sample_weight=sample_weight,
        )
        weighted_cross_entropy = (
            tf.cast(alpha, cross_entropy.dtype) * cross_entropy
        )

        latent_posterior_kl = self._gaussian_kl_latent_posterior(mh, y)
        weighted_latent_posterior_kl = (
            tf.cast(beta, latent_posterior_kl.dtype) * latent_posterior_kl
        )

        discriminator_scores = tf.stack(
            [
                self._discriminator_for_encoder(mh, class_index)
                for class_index in range(self.n_classes)
            ],
            axis=1,
        )
        true_class_scores = tf.reduce_sum(
            y_onehot * discriminator_scores,
            axis=1,
        )
        discriminator_kl = self._weighted_mean(
            tf.nn.relu(true_class_scores),
            sample_weight=sample_weight,
        )
        weighted_discriminator_kl = (
            tf.cast(gamma, discriminator_kl.dtype) * discriminator_kl
        )

        empirical_class_prior = tf.reduce_mean(y_onehot, axis=0)
        learned_log_class_prior = tf.nn.log_softmax(self.log_class_prior)
        class_prior_kl = tf.reduce_sum(
            empirical_class_prior
            * (
                tf.math.log(empirical_class_prior + 1e-8)
                - learned_log_class_prior
            )
        )
        weighted_class_prior_kl = (
            tf.cast(lambda_, class_prior_kl.dtype) * class_prior_kl
        )

        total_loss = (
            weighted_cross_entropy
            + weighted_latent_posterior_kl
            + weighted_discriminator_kl
            + weighted_class_prior_kl
        )

        return {
            "total_loss": total_loss,
            "cross_entropy": cross_entropy,
            "weighted_cross_entropy": weighted_cross_entropy,
            "latent_posterior_kl": latent_posterior_kl,
            "weighted_latent_posterior_kl": weighted_latent_posterior_kl,
            "discriminator_kl": discriminator_kl,
            "weighted_discriminator_kl": weighted_discriminator_kl,
            "class_prior_kl": class_prior_kl,
            "weighted_class_prior_kl": weighted_class_prior_kl,
        }

    def vc_loss(
        self,
        mh: tf.Tensor,
        y: tf.Tensor,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.0,
        lambda_: float = 0.0,
        logits: tf.Tensor | None = None,
        sample_weight: tf.Tensor | None = None,
    ) -> tf.Tensor:
        """Return the complete VC objective while preserving the old API."""
        return self.vc_loss_components(
            mh=mh,
            y=y,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            lambda_=lambda_,
            logits=logits,
            sample_weight=sample_weight,
        )["total_loss"]

    def keras_loss(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.0,
        lambda_: float = 0.0,
    ):
        """Return a Keras-compatible wrapper around :meth:`vc_loss`."""

        def loss_fn(y_true, y_pred):
            if self._last_mh is None:
                raise ValueError(
                    "VariationalClassifier has no stored latent features. "
                    "Make sure the model output comes from this classifier head."
                )
            return self.vc_loss(
                mh=self._last_mh,
                y=y_true,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                lambda_=lambda_,
                logits=y_pred,
            )

        return loss_fn

    def discriminator_loss(self, mh: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """Train T_psi to distinguish q_phi(z|y) from prior samples."""
        y = self._class_ids(y)
        dtype = mh.dtype
        total_loss = tf.zeros((), dtype=dtype)
        valid_classes = tf.zeros((), dtype=dtype)

        for class_index in range(self.n_classes):
            mask = tf.equal(y, class_index)
            z_q = tf.boolean_mask(mh, mask)
            n_class = tf.shape(z_q)[0]

            def compute_class_loss():
                sigma_class = tf.exp(self.prior_log_sigma[class_index])
                z_p = (
                    tf.random.normal(tf.shape(z_q), dtype=z_q.dtype)
                    * sigma_class
                    + self.prior_mu[class_index]
                )
                logits_q = self.discriminator(z_q, class_index)
                logits_p = self.discriminator(z_p, class_index)
                loss_q = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.ones_like(logits_q),
                        logits=logits_q,
                    )
                )
                loss_p = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.zeros_like(logits_p),
                        logits=logits_p,
                    )
                )
                return loss_q + loss_p, tf.ones((), dtype=dtype)

            class_loss, is_valid = tf.cond(
                tf.greater(n_class, 0),
                true_fn=compute_class_loss,
                false_fn=lambda: (
                    tf.zeros((), dtype=dtype),
                    tf.zeros((), dtype=dtype),
                ),
            )
            total_loss += class_loss
            valid_classes += is_valid

        return tf.cond(
            tf.greater(valid_classes, 0.0),
            true_fn=lambda: total_loss / valid_classes,
            false_fn=lambda: tf.zeros((), dtype=dtype),
        )

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"n_classes": self.n_classes})
        return config
