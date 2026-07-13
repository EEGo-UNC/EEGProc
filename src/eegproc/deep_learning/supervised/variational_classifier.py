"""
variational_classifier.py
=========================
Variational Classification fusion head for neural networks.

Maintains learned Gaussian class priors and classifies via Bayes' rule,
with an optional auxiliary discriminator for improved latent space alignment.

References
----------
Li et al., "STSNet ...", HISS 2023.
"""

import tensorflow as tf
import numpy as np


class VariationalClassifier(tf.keras.layers.Layer):
    """
    Variational Classification fusion head.

    Maintains learned Gaussian class priors p_theta(z|y) = N(mu_y, Sigma_y)
    and classifies via Bayes' rule (generalised softmax, Eq. 6).

    The training objective (Eq. 7) has three independently controllable terms:
      Term 1 -- cross-entropy:           always active
      Term 2 -- KL (encoder/prior):      scaled by `beta`   (set 0 to disable)
      Term 2b -- NLL (Gaussian, analytic): scaled by `gamma`  (set 0 to disable)
      Term 3 -- KL (class prior):        scaled by `lambda_` (set 0 to disable)

    An auxiliary discriminator (Eq. 9) can be updated separately or skipped
    entirely via the `training_mode` flag on the parent model.

    Parameters
    ----------
    n_classes  : int   -- number of classes (2 for binary valence/arousal)
    latent_dim : int   -- set automatically by build()
    """

    def __init__(self, n_classes: int = 2, latent_dim: int = None, **kwargs):
        super().__init__(**kwargs)
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self._last_mh = None

    def build(self, input_shape):
        d = input_shape[-1]
        self.latent_dim = d

        self.prior_mu = self.add_weight(
            name="prior_mu",
            shape=(self.n_classes, d),
            initializer="glorot_normal",
            trainable=True,
        )
        self.prior_log_sigma = self.add_weight(
            name="prior_log_sigma",
            shape=(self.n_classes, d),
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
            shape=(self.n_classes, d),
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

    def _log_gaussian(self, z, mu, log_sigma):
        sigma2 = tf.exp(2.0 * log_sigma)
        diff = z - mu[tf.newaxis, :]
        # Use reduce_mean (not reduce_sum) over the latent dimension so that
        # logit magnitudes are O(1) regardless of latent_dim.  With reduce_sum
        # the log-likelihood scales with d (hundreds of dims), swamping the
        # class-prior term and producing near-identical extreme logits before
        # the priors have had time to learn anything useful.
        return -0.5 * tf.reduce_mean(
            tf.math.log(2 * np.pi * sigma2) + diff**2 / sigma2,
            axis=-1,
        )

    def call(self, mh: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Classify latent features using learned Gaussian class priors.

        Parameters
        ----------
        mh : tf.Tensor
            Latent feature vectors with shape (batch, latent_dim).

        Returns
        -------
        tf.Tensor
            Class logits with shape (batch, n_classes).
        """
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

        return log_likelihoods + log_class_prior[tf.newaxis, :]

    def discriminator(self, z: tf.Tensor, y: int) -> tf.Tensor:
        """T_psi^y(z) = w_y^T z + b_y"""
        return tf.linalg.matvec(z, self.disc_w[y]) + self.disc_b[y]

    def _gaussian_kl_latent_posterior(
        self,
        mh: tf.Tensor,
        y: tf.Tensor,
    ) -> tf.Tensor:
        """
        Compute the expected class-conditional KL divergence

            E_{p(y)} [
                KL(
                    q_phi(z | y)
                    ||
                    p_theta(z | y)
                )
            ]

        q_phi(z | y=c) is approximated as a diagonal Gaussian fitted
        to the latent embeddings belonging to class c in the current batch.

        p_theta(z | y=c) is the learned diagonal Gaussian represented by:

            self.prior_mu[c]
            self.prior_log_sigma[c]

        Parameters
        ----------
        mh : tf.Tensor
            Latent embeddings with shape (batch_size, latent_dim).

        y : tf.Tensor
            Integer class labels with shape (batch_size,).

        Returns
        -------
        tf.Tensor
            Scalar expected KL divergence over the empirical batch
            class distribution p(y).
        """
        mh = tf.convert_to_tensor(mh)
        y = tf.cast(tf.reshape(y, [-1]), tf.int32)

        dtype = mh.dtype
        eps = tf.cast(1e-6, dtype)

        batch_size = tf.cast(tf.shape(mh)[0], dtype)

        expected_kl = tf.zeros((), dtype=dtype)
        valid_probability_mass = tf.zeros((), dtype=dtype)

        for c in range(self.n_classes):
            mask = tf.equal(y, c)
            z_c = tf.boolean_mask(mh, mask)
            n_c = tf.shape(z_c)[0]

            def compute_class_kl():
                # Empirical q_phi(z | y=c)
                mu_q = tf.reduce_mean(z_c, axis=0)

                centered = z_c - mu_q[tf.newaxis, :]

                # Maximum-likelihood diagonal variance estimate.
                var_q = tf.reduce_mean(
                    tf.square(centered),
                    axis=0,
                )
                var_q = tf.maximum(var_q, eps)

                # Learned p_theta(z | y=c)
                mu_p = tf.cast(self.prior_mu[c], dtype)

                # prior_log_sigma stores log standard deviation,
                # therefore variance = exp(2 * log_sigma).
                var_p = tf.exp(2.0 * tf.cast(self.prior_log_sigma[c], dtype))
                var_p = tf.maximum(var_p, eps)

                # KL(
                #   N(mu_q, diag(var_q))
                #   ||
                #   N(mu_p, diag(var_p))
                # )
                kl_c = 0.5 * tf.reduce_sum(
                    tf.math.log(var_p)
                    - tf.math.log(var_q)
                    + (var_q + tf.square(mu_q - mu_p)) / var_p
                    - 1.0
                )

                # Empirical p(y=c) from the current batch.
                p_c = tf.cast(n_c, dtype) / batch_size

                return p_c * kl_c, p_c

            weighted_kl_c, probability_c = tf.cond(
                tf.greater_equal(n_c, 2),
                true_fn=compute_class_kl,
                false_fn=lambda: (
                    tf.zeros((), dtype=dtype),
                    tf.zeros((), dtype=dtype),
                ),
            )

            expected_kl += weighted_kl_c
            valid_probability_mass += probability_c

        # Renormalize if any classes had fewer than two samples and were skipped.
        return tf.cond(
            tf.greater(valid_probability_mass, 0.0),
            true_fn=lambda: expected_kl / valid_probability_mass,
            false_fn=lambda: tf.zeros((), dtype=dtype),
        )

    def _gaussian_entropy(
        self,
        mh: tf.Tensor,
        y: tf.Tensor,
    ) -> tf.Tensor:
        """
        Estimate E_{p(y)}[H(q_phi(z|y))] using a diagonal Gaussian
        fitted to the class-conditioned latent vectors in the batch.

        For each class c:

            q_phi(z|y=c) = N(mu_c, diag(var_c))

            H[q_phi(z|y=c)]
                = 0.5 * sum_d log(2 * pi * e * var_{c,d})

        Class entropies are weighted by the empirical batch probability p(y=c).

        Parameters
        ----------
        mh : tf.Tensor
            Latent embeddings with shape (batch_size, latent_dim).

        y : tf.Tensor
            Integer class labels with shape (batch_size,).

        Returns
        -------
        tf.Tensor
            Scalar estimate of E_{p(y)}[H(q_phi(z|y))].
        """
        mh = tf.convert_to_tensor(mh)
        y = tf.cast(tf.reshape(y, [-1]), tf.int32)

        dtype = mh.dtype
        eps = tf.cast(1e-6, dtype)

        batch_size = tf.cast(tf.shape(mh)[0], dtype)
        total_entropy = tf.zeros((), dtype=dtype)

        log_two_pi_e = tf.math.log(tf.cast(2.0 * np.pi * np.e, dtype))

        for c in range(self.n_classes):
            mask = tf.equal(y, c)
            z_c = tf.boolean_mask(mh, mask)
            n_c = tf.shape(z_c)[0]

            def compute_class_entropy() -> tf.Tensor:
                mu_c = tf.reduce_mean(z_c, axis=0)

                centered = z_c - mu_c[tf.newaxis, :]

                # Diagonal sample covariance with Bessel correction.
                var_c = tf.reduce_sum(
                    tf.square(centered),
                    axis=0,
                ) / tf.cast(n_c - 1, dtype)

                var_c = tf.maximum(var_c, eps)

                # Differential entropy of a diagonal multivariate Gaussian.
                entropy_c = 0.5 * tf.reduce_sum(log_two_pi_e + tf.math.log(var_c))

                # Weight by empirical p(y=c) in the current batch.
                class_probability = tf.cast(n_c, dtype) / batch_size

                return class_probability * entropy_c

            weighted_entropy_c = tf.cond(
                tf.greater_equal(n_c, 2),
                true_fn=compute_class_entropy,
                false_fn=lambda: tf.zeros((), dtype=dtype),
            )

            total_entropy += weighted_entropy_c

        return total_entropy

    def vc_loss(
        self,
        mh: tf.Tensor,
        y: tf.Tensor,
        alpha: float = 1.0,
        gamma: float = 0.0, # discriminator term
        beta: float = 1.4, # gaussian assumptionterm
        lambda_: float = 0.0,
        delta: float = 0.0,
    ) -> tf.Tensor:
        """
        VC objective (Eq. 7).

        L_VC = alpha * xent  +  beta * KL(encoder||prior)  +  lambda_ * KL(class prior) -

        Setting beta=0 and lambda_=0 reduces this to plain cross-entropy,
        which is the "disc_only" ablation signal seen by the encoder.

        Parameters
        ----------
        mh      : (batch, d)
        y       : (batch,) int32
        alpha   : weight for cross-entropy               (0 = disabled)
        beta    : KL weight for encoder/prior alignment  (0 = disabled)
        gamma   : NLL weight for Gaussian analytic term  (0 = disabled)
        lambda_ : KL weight for class-prior alignment    (0 = disabled)
        """
        y_onehot = tf.one_hot(y, self.n_classes)

        # Term 1: cross-entropy
        logits = self(mh, training=True)
        xent = alpha * tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        )

        # Term 2: beta * E_y[KL(q_phi(z|y) || p_theta(z|y))] via discriminator scores
        #
        # The discriminator T_psi^y is trained to satisfy the variational
        # lower bound  KL(q||p) >= E_q[T(z)] - log E_p[exp(T(z))]  (Eq. 8).
        # Here we use the simpler density-ratio trick: T_psi^y(z) approximates
        # log q(z|y) - log p(z|y), so E_q[T] is a (positive) KL proxy.
        #
        # Two safeguards keep this term well-behaved:
        #   1. stop_gradient: the discriminator weights are updated by
        #      discriminator_loss; here T is a fixed signal to the encoder.
        #   2. tf.nn.relu: clamps the estimate to >= 0 so a poorly-initialised
        #      or collapsing discriminator cannot make the total loss negative.
        T_vals = tf.stack(
            [self.discriminator(mh, c) for c in range(self.n_classes)], axis=1
        )
        T_vals = tf.stop_gradient(T_vals)
        T_true_class = tf.reduce_sum(y_onehot * T_vals, axis=1)
        kl_term = gamma * tf.reduce_mean(tf.nn.relu(T_true_class))

        # Term 2b: gamma * E_{p(y)}[
        #     KL(q_phi(z | y) || p_theta(z | y))
        # ]
        latent_posterior_kl = self._gaussian_kl_latent_posterior(
            mh,
            y,
        )

        kl_term = beta * latent_posterior_kl

        # Term 3: lambda_ * KL(p(y) || p_pi(y))
        p_y = tf.reduce_mean(y_onehot, axis=0)
        log_p_pi = tf.nn.log_softmax(self.log_class_prior)
        kl_class_prior = tf.reduce_sum(p_y * (tf.math.log(p_y + 1e-8) - log_p_pi))
        prior_term = lambda_ * kl_class_prior

        # Term 4: delta * H(q(y|z))
        # Entropy regularisation:
        q_entropy = self._gaussian_entropy(mh, y)
        entropy_term = -delta * q_entropy

        return xent + kl_term + latent_posterior_kl + prior_term + entropy_term

    def keras_loss(
        self,
        alpha: float = 1.0,
        beta: float = 0.0,
        gamma: float = 1e-4,
        lambda_: float = 0.0,
    ):
        """
        Return a Keras-compatible loss function that uses the latest latent features
        seen by this classifier head.

        This lets models compile with:

            loss=vc_head.keras_loss(...)

        while still allowing vc_loss to use mh internally.
        """

        def loss_fn(y_true, y_pred):
            if self._last_mh is None:
                raise ValueError(
                    "VariationalClassifier has no stored latent features. "
                    "Make sure the model output comes from this classifier head."
                )

            y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)

            return self.vc_loss(
                mh=self._last_mh,
                y=y_true,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                lambda_=lambda_,
            )

        return loss_fn

    def discriminator_loss(self, mh: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """
        Auxiliary discriminator objective (Eq. 9).

        Trains T_psi to separate q_phi(z|y) (actual latents) from
        p_theta(z|y) (Gaussian prior samples) using binary cross-entropy.
        """
        total = 0.0
        for c in range(self.n_classes):
            mask = tf.equal(y, c)
            if tf.reduce_sum(tf.cast(mask, tf.int32)) == 0:
                continue
            z_q = tf.boolean_mask(mh, mask)
            sigma_c = tf.exp(self.prior_log_sigma[c])
            z_p = tf.random.normal(tf.shape(z_q)) * sigma_c + self.prior_mu[c]

            T_q = self.discriminator(z_q, c)
            T_p = self.discriminator(z_p, c)

            loss_q = tf.reduce_mean(tf.math.log(tf.sigmoid(T_q) + 1e-8))
            loss_p = tf.reduce_mean(tf.math.log(1.0 - tf.sigmoid(T_p) + 1e-8))
            total += -(loss_q + loss_p)

        return total / self.n_classes

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"n_classes": self.n_classes})
        return cfg
