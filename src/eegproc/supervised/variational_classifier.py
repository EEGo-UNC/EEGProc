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
      Term 1 -- cross-entropy:      always active
      Term 2 -- KL (encoder/prior): scaled by `beta`  (set 0 to disable)
      Term 3 -- KL (class prior):   scaled by `lambda_` (set 0 to disable)

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

    def build(self, input_shape):
        d = input_shape[-1]
        self.latent_dim = d

        self.prior_mu = self.add_weight(
            "prior_mu", shape=(self.n_classes, d),
            initializer="glorot_normal", trainable=True,
        )
        self.prior_log_sigma = self.add_weight(
            "prior_log_sigma", shape=(self.n_classes, d),
            initializer="zeros", trainable=True,
        )
        self.log_class_prior = self.add_weight(
            "log_class_prior", shape=(self.n_classes,),
            initializer="zeros", trainable=True,
        )
        self.disc_w = self.add_weight(
            "disc_w", shape=(self.n_classes, d),
            initializer="glorot_normal", trainable=True,
        )
        self.disc_b = self.add_weight(
            "disc_b", shape=(self.n_classes,),
            initializer="zeros", trainable=True,
        )
        super().build(input_shape)

    def _log_gaussian(self, z, mu, log_sigma):
        sigma2 = tf.exp(2.0 * log_sigma)
        diff   = z - mu[tf.newaxis, :]
        return -0.5 * tf.reduce_sum(
            tf.math.log(2 * np.pi * sigma2) + diff ** 2 / sigma2, axis=-1,
        )

    def call(self, mh: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Parameters
        ----------
        mh : (batch, latent_dim) -- pre-concatenated feature vectors

        Returns
        -------
        logits : (batch, n_classes)
        """
        log_class_prior = tf.nn.log_softmax(self.log_class_prior)
        log_likelihoods = tf.stack(
            [self._log_gaussian(mh, self.prior_mu[y], self.prior_log_sigma[y])
             for y in range(self.n_classes)],
            axis=1,
        )
        return log_likelihoods + log_class_prior[tf.newaxis, :]

    def discriminator(self, z: tf.Tensor, y: int) -> tf.Tensor:
        """T_psi^y(z) = w_y^T z + b_y"""
        return tf.linalg.matvec(z, self.disc_w[y]) + self.disc_b[y]

    def vc_loss(
        self,
        mh: tf.Tensor,
        y: tf.Tensor,
        alpha: float = 1.0,
        beta: float = 1.0,
        lambda_: float = 1.0,
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
        lambda_ : KL weight for class-prior alignment    (0 = disabled)
        """
        y_onehot = tf.one_hot(y, self.n_classes)

        # Term 1: cross-entropy
        logits = self(mh, training=True)
        xent = alpha * tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        )

        # Term 2: beta * E_y[KL(q_phi(z|y) || p_theta(z|y))] via discriminator scores
        T_vals = tf.stack(
            [self.discriminator(mh, c) for c in range(self.n_classes)], axis=1
        )
        T_true_class = tf.reduce_sum(y_onehot * T_vals, axis=1)
        kl_term = beta * tf.reduce_mean(T_true_class)

        # Term 3: lambda_ * KL(p(y) || p_pi(y))
        p_y         = tf.reduce_mean(y_onehot, axis=0)
        log_p_pi    = tf.nn.log_softmax(self.log_class_prior)
        kl_class_prior = tf.reduce_sum(p_y * (tf.math.log(p_y + 1e-8) - log_p_pi))
        prior_term  = lambda_ * kl_class_prior

        return xent + kl_term + prior_term

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
            z_q     = tf.boolean_mask(mh, mask)
            sigma_c = tf.exp(self.prior_log_sigma[c])
            z_p     = tf.random.normal(tf.shape(z_q)) * sigma_c + self.prior_mu[c]

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
