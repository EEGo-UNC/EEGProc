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

    def _gaussian_kl_point(self, mh: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """
        Per-sample NLL under the empirical class-conditioned Gaussian prior.

        For each sample i, computes:
            -log p(mh_i | y_i) = 0.5 * sum_d [ (mh_d - mu_c_d)^2 / var_c_d
                                                + log(var_c_d) ]
        where mu_c and var_c are the sample mean and diagonal sample covariance
        of all mh belonging to class c in the current batch:
            mu_c  = (1 / N_c)     * sum_{i: y_i=c} z_i
            var_c = (1 / N_c - 1) * sum_{i: y_i=c} (z_i - mu_c)^2

        This penalises each latent for being far from its class centre,
        weighted by the spread of that class.

        Parameters
        ----------
        mh : (batch, d)  -- fused latent embeddings
        y  : (batch,)    -- integer class labels

        Returns
        -------
        Scalar mean NLL over the batch.
        """
        total_nll = tf.constant(0.0, dtype=tf.float32)
        n_valid = tf.constant(0, dtype=tf.int32)

        for c in range(self.n_classes):
            mask = tf.equal(y, c)  # (batch,) bool
            z_c = tf.boolean_mask(mh, mask)  # (N_c, d)
            n_c = tf.shape(z_c)[0]

            def class_nll() -> tf.Tensor:
                mu_c = tf.reduce_mean(z_c, axis=0)  # (d,)

                # Diagonal sample covariance, Bessel-corrected (PI's Sigma_c formula)
                diff = z_c - mu_c[tf.newaxis, :]  # (N_c, d)
                var_c = tf.reduce_sum(tf.square(diff), axis=0) / tf.cast(
                    tf.maximum(n_c - 1, 1), tf.float32
                )  # (d,)
                var_c = tf.maximum(var_c, 1e-6)  # numerical floor

                # -log N(z_c | mu_c, diag(var_c)) for every sample in this class
                nll_c = 0.5 * tf.reduce_sum(
                    tf.square(z_c - mu_c[tf.newaxis, :]) / var_c[tf.newaxis, :]
                    + tf.math.log(var_c)[tf.newaxis, :],
                    axis=-1,
                )  # (N_c,)

                return tf.reduce_mean(nll_c)

            class_loss = tf.cond(
                tf.greater_equal(n_c, 2),
                true_fn=class_nll,
                false_fn=lambda: tf.constant(0.0, dtype=tf.float32),
            )

            total_nll += class_loss
            n_valid += tf.cast(tf.greater_equal(n_c, 2), tf.int32)

        return tf.cond(
            tf.equal(n_valid, 0),
            true_fn=lambda: tf.constant(0.0, dtype=tf.float32),
            false_fn=lambda: total_nll / tf.cast(n_valid, tf.float32),
        )

    def vc_loss(
        self,
        mh: tf.Tensor,
        y: tf.Tensor,
        alpha: float = 1.0,
        beta: float = 0.0,
        gamma: float = 1e-4,
        lambda_: float = 0.0,
    ) -> tf.Tensor:
        """
        VC objective (Eq. 7).

        L_VC = alpha * xent  +  beta * KL(encoder||prior)  +  lambda_ * KL(class prior)

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
        kl_term = beta * tf.reduce_mean(tf.nn.relu(T_true_class))

        # Term 2b: gamma * per-sample NLL under empirical class Gaussian
        # Clamped to >= 0: log-likelihood of a Gaussian can be positive when
        # var_c < 1, which would make this term a reward rather than a penalty.
        # Since the intent is a regularisation cost, we floor it at zero.
        gaussian_kl_term = gamma * tf.nn.relu(self._gaussian_kl_point(mh, y))

        # Term 3: lambda_ * KL(p(y) || p_pi(y))
        p_y = tf.reduce_mean(y_onehot, axis=0)
        log_p_pi = tf.nn.log_softmax(self.log_class_prior)
        kl_class_prior = tf.reduce_sum(p_y * (tf.math.log(p_y + 1e-8) - log_p_pi))
        prior_term = lambda_ * kl_class_prior

        return xent + kl_term + gaussian_kl_term + prior_term
    
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
