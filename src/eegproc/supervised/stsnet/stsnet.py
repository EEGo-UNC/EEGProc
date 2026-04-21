"""
stsnet.py
=========
Full STSNet model: BiLSTM sub-model + ManifoldNet sub-model + fusion head.

Architecture summary (Figure 2 of the paper)
--------------------------------------------
(a) ManifoldNet branch (MO):
      4-D SPD tensor  →  2x wFM conv  →  Invariant layer  →  MO vector

(b) BiLSTM branch (HO):
      Flattened covariance sequence  →  BiLSTM  →  HO vector  (Eq. 10)

(c) Fusion & classification:
      MH = concat(MO, HO)  →  FC  →  Softmax  →  class label

Training uses the joint alternating optimisation from Algorithm 1:
every even iteration fixes HO and trains MO; every odd iteration
fixes MO and trains HO. The FC layer is updated on every step.

References
----------
Li et al., "STSNet ...", HISS 2023.
"""

import tensorflow as tf
import numpy as np
try:
    from .manifold_net import ManifoldNet
except ImportError:
    from manifold_net import ManifoldNet


# ---------------------------------------------------------------------------
# BiLSTM sub-model
# ---------------------------------------------------------------------------

class BiLSTMNet(tf.keras.Model):
    """BiLSTM sub-model for spatio-temporal feature extraction (HO).

    Processes the flattened covariance time-series produced by
    `build_spatiotemporal_representation`.

    Architecture (per Table 1 / Table 3 in the paper)
    --------------------------------------------------
    Input (n_windows, feat_dim) → BiLSTM(256 units) → HO (512-d vector)
    HO = concat(forward hidden state at T, backward hidden state at 1)
    following Eq. (10): HO = H_nc (→) ⊕ H_1 (←)

    Parameters
    ----------
    hidden_units : int — LSTM cell size (default 256)
    dropout_rate : float — recurrent dropout for regularisation
    """

    def __init__(
        self,
        hidden_units: int = 256,
        dropout_rate: float = 0.3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate

        # return_sequences=True so we can manually select the final states
        forward_lstm  = tf.keras.layers.LSTM(
            hidden_units,
            return_sequences=True,
            return_state=True,
            dropout=dropout_rate,
            name="forward_lstm",
        )
        backward_lstm = tf.keras.layers.LSTM(
            hidden_units,
            return_sequences=True,
            return_state=True,
            go_backwards=True,
            dropout=dropout_rate,
            name="backward_lstm",
        )
        self.bilstm = tf.keras.layers.Bidirectional(
            forward_lstm,
            backward_layer=backward_lstm,
            merge_mode=None,   # keep forward / backward separate for Eq. 10
            name="bilstm",
        )

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (batch, n_windows, feat_dim)

        Returns
        -------
        ho : Tensor, shape (batch, 2 * hidden_units)
            HO = H_nc (→) ⊕ H_1 (←)  per Eq. (10)
        """
        outputs = self.bilstm(x, training=training)
        # Bidirectional with merge_mode=None returns:
        #   [fwd_seq, bwd_seq, fwd_h, fwd_c, bwd_h, bwd_c]
        _, _, fwd_h, _, bwd_h, _ = outputs

        # Forward: last output at T (= hidden state at n_windows)
        # Backward: last output going backward (= hidden state at t=1)
        # Both fwd_h / bwd_h are the final hidden states of each direction.
        ho = tf.concat([fwd_h, bwd_h], axis=-1)  # (batch, 2*hidden_units)
        return ho

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "hidden_units": self.hidden_units,
            "dropout_rate": self.dropout_rate,
        })
        return cfg


# ---------------------------------------------------------------------------
# Fusion head (for regular classifier)
# ---------------------------------------------------------------------------

class FusionHead(tf.keras.layers.Layer):
    """Concatenate MO and HO, then classify via a fully-connected softmax layer.

    MH = [MO ⊕ HO]  (Eq. 11)
    Ŷ  = softmax(W · MH + b)  (Eq. 12)

    Parameters
    ----------
    n_classes : int — number of emotion classes (2 for binary valence/arousal)
    """

    def __init__(self, n_classes: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.n_classes = n_classes
        self.fc = tf.keras.layers.Dense(n_classes, name="fc")

    def call(
        self,
        mo: tf.Tensor,
        ho: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """
        Parameters
        ----------
        mo : Tensor, shape (batch, mo_dim)
        ho : Tensor, shape (batch, ho_dim)

        Returns
        -------
        logits : Tensor, shape (batch, n_classes)
        """
        mh = tf.concat([mo, ho], axis=-1)  # (batch, mo_dim + ho_dim)
        return self.fc(mh)                  # (batch, n_classes)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"n_classes": self.n_classes})
        return cfg

# ---------------------------------------------------------------------------
# Variational Fusion Head (for Variational Classifier)
# ---------------------------------------------------------------------------

class VariationalFusionHead(tf.keras.layers.Layer):
    """
    Variational Classification fusion head.

    Replaces softmax(Wz + b) with a probabilistic output layer that:
      1. Maintains learned Gaussian class priors p_θ(z|y) = N(μ_y, Σ_y)
      2. Computes p_θ(y|z) via Bayes' rule (generalised softmax, Eq. 6)
      3. Aligns empirical q_φ(z|y) to p_θ(z|y) via the VC objective (Eq. 7)

    Parameters
    ----------
    n_classes : int   — number of classes (2 for binary valence/arousal)
    latent_dim : int  — dimensionality of MH = [MO ⊕ HO]
    beta      : float — KL weight (β in Eq. 7), default 1.0
    """

    def __init__(self, n_classes: int = 2, latent_dim: int = None, **kwargs):
        super().__init__(**kwargs)
        self.n_classes = n_classes
        self.latent_dim = latent_dim  # set via build()

    def build(self, input_shape):
        # input_shape is the shape of MH
        d = input_shape[-1]
        self.latent_dim = d

        # Learnable Gaussian class priors p_θ(z|y): μ_y and log σ_y
        # Shape: (n_classes, latent_dim)
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

        # Learnable class prior mixing weights p_π(y) (log-space)
        self.log_class_prior = self.add_weight(
            name="log_class_prior",
            shape=(self.n_classes,),
            initializer="zeros",
            trainable=True,
        )

        # Per-class linear discriminator T_ψ^y(z) = w_y^T z + b_y
        # Used to approximate log q(z|y) / p(z|y) (density ratio trick, Eq. 9)
        # Shape: (n_classes, latent_dim) for weights, (n_classes,) for biases
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
        """Evaluate log N(z; mu, diag(exp(log_sigma)^2)).
        
        z         : (batch, d)
        mu        : (d,)
        log_sigma : (d,)
        Returns   : (batch,)
        """
        sigma2 = tf.exp(2.0 * log_sigma)           # (d,)
        diff = z - mu[tf.newaxis, :]               # (batch, d)
        return -0.5 * tf.reduce_sum(
            tf.math.log(2 * np.pi * sigma2) + diff**2 / sigma2,
            axis=-1,
        )

    def call(self, mh: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Compute class logits via Bayes' rule using Gaussian priors.

        log p_θ(y|z) ∝ log p_θ(z|y) + log p_π(y)

        Parameters
        ----------
        mh : (batch, latent_dim) — concatenated [MO ⊕ HO]

        Returns
        -------
        logits : (batch, n_classes)
        """
        log_class_prior = tf.nn.log_softmax(self.log_class_prior)  # normalise
        # (batch, n_classes)
        log_likelihoods = tf.stack(
            [self._log_gaussian(mh, self.prior_mu[y], self.prior_log_sigma[y])
             for y in range(self.n_classes)],
            axis=1,
        )
        return log_likelihoods + log_class_prior[tf.newaxis, :]  # log unnorm. posterior

    def discriminator(self, z: tf.Tensor, y: int) -> tf.Tensor:
        """T_ψ^y(z) = w_y^T z + b_y — approximates log q(z|y)/p(z|y)."""
        return tf.linalg.matvec(z, self.disc_w[y]) + self.disc_b[y]  # (batch,)

    def vc_loss(
        self,
        mh: tf.Tensor,
        y: tf.Tensor,
        beta: float = 1.0,
    ) -> tf.Tensor:
        """
        Full VC objective (Eq. 7):

        L_VC = E[log p_θ(y|z)] 
             + β * E_y[ log p_θ(z|y) ]    ← MAP term
             - β * E_y[ log q_φ(z|y) ]    ← entropy term (approx. via discriminator)
             + E[log p_π(y)]

        In practice:
          - The cross-entropy term uses the Bayes-rule logits from call().
          - The MAP term is the log-Gaussian of z under its true class prior.
          - The entropy term is approximated by T_ψ(z) (the discriminator).

        Parameters
        ----------
        mh   : (batch, d) — latent representations
        y    : (batch,) int32 — true class labels
        beta : float — KL weight

        Returns
        -------
        scalar loss (negates the L_VC which should be maximized, to be minimized instead)
        """
        # --- Classification term: -E[log p_θ(y|z)] ---
        logits = self(mh, training=True)
        xent = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        )

        # --- MAP term: -β * E[log p_θ(z|y)] (pull z toward its class Gaussian) ---
        # Gather per-sample class parameters
        mu_y        = tf.gather(self.prior_mu, y)         # (batch, d)
        log_sigma_y = tf.gather(self.prior_log_sigma, y)  # (batch, d)
        log_prior_z = -0.5 * tf.reduce_sum(               # (batch,)
            2.0 * log_sigma_y
            + ((mh - mu_y) / tf.exp(log_sigma_y)) ** 2,
            axis=-1,
        )
        map_term = -beta * tf.reduce_mean(log_prior_z)

        # --- Entropy term (density ratio approximation): +β * E[T_ψ(z)] ---
        # T_ψ^y(z) ≈ log q(z|y)/p(z|y); adding it to the loss increases entropy
        # of q(z|y) relative to p(z|y), preventing distribution collapse.
        T_vals = tf.stack(
            [self.discriminator(mh, c) for c in range(self.n_classes)], axis=1
        )  # (batch, n_classes)
        # Weight by one-hot of true class (only penalise T for true class)
        y_onehot   = tf.one_hot(y, self.n_classes)         # (batch, n_classes)
        entropy_approx = tf.reduce_mean(tf.reduce_sum(y_onehot * T_vals, axis=1))
        entropy_term = beta * entropy_approx

        # --- Class prior term: -E[log p_π(y)] ---
        log_py = tf.nn.log_softmax(self.log_class_prior)   # (n_classes,)
        prior_term = -tf.reduce_mean(tf.gather(log_py, y))

        return xent + map_term + entropy_term + prior_term

    def discriminator_loss(
        self,
        mh: tf.Tensor,
        y: tf.Tensor,
    ) -> tf.Tensor:
        """
        Auxiliary discriminator objective (Eq. 9):

        -L_aux = E_y[ E_{q(z|y)}[log σ(T_ψ(z))] + E_{p(z|y)}[log(1 - σ(T_ψ(z')))] ]

        Trains T_ψ to distinguish samples from q_φ(z|y) (actual latents) vs
        p_θ(z|y) (Gaussian prior samples).

        Parameters
        ----------
        mh : (batch, d) — actual encoder outputs (≈ samples from q_φ(z|y))
        y  : (batch,) int32 — true class labels

        Returns
        -------
        scalar auxiliary loss
        """
        total = 0.0
        log_class_probs = tf.nn.log_softmax(self.log_class_prior)

        for c in range(self.n_classes):
            mask = tf.equal(y, c)
            if tf.reduce_sum(tf.cast(mask, tf.int32)) == 0:
                continue
            z_q = tf.boolean_mask(mh, mask)              # actual samples from q
            n_q = tf.shape(z_q)[0]

            # Sample from p_θ(z|y=c)
            mu_c    = self.prior_mu[c]
            sigma_c = tf.exp(self.prior_log_sigma[c])
            z_p = tf.random.normal(tf.shape(z_q)) * sigma_c + mu_c  # reparameterisation

            T_q = self.discriminator(z_q, c)  # (n_q,)
            T_p = self.discriminator(z_p, c)  # (n_q,)

            # Binary cross-entropy: q-samples are "real" (label=1), p-samples "fake" (label=0)
            loss_q = tf.reduce_mean(tf.math.log(tf.sigmoid(T_q) + 1e-8))
            loss_p = tf.reduce_mean(tf.math.log(1.0 - tf.sigmoid(T_p) + 1e-8))
            total += -(loss_q + loss_p)  # negate because we minimise

        return total / self.n_classes

# ---------------------------------------------------------------------------
# Full STSNet
# ---------------------------------------------------------------------------

class STSNet(tf.keras.Model):
    """STSNet: Spatio-Temporal-Spectral Network for EEG emotion recognition.

    Combines ManifoldNet (spatio-spectral) and BiLSTM (spatio-temporal)
    branches, then classifies via a shared FC layer.

    Parameters
    ----------
    n_channels      : int   — EEG channel count
    n_classes       : int   — emotion classes (default 2: binary)
    bilstm_units    : int   — BiLSTM hidden units (default 256)
    bilstm_dropout  : float — BiLSTM recurrent dropout
    manifold_kernel : int   — wFM kernel size for ManifoldNet (default 2)
    n_fm_iters      : int   — Fréchet mean iterations
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int = 2,
        bilstm_units: int = 256,
        bilstm_dropout: float = 0.3,
        manifold_kernel: int = 2,
        n_fm_iters: int = 10,
        vc_beta: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.manifold_net = ManifoldNet(
            n_channels=n_channels,
            kernel_size=manifold_kernel,
            n_fm_iters=n_fm_iters,
            name="manifold_net",
        )
        self.bilstm_net = BiLSTMNet(
            hidden_units=bilstm_units,
            dropout_rate=bilstm_dropout,
            name="bilstm_net",
        )
        self.fusion = VariationalFusionHead(n_classes=n_classes, name="fusion") # User Variationl Fusion Head for VC, Fusion Head for standard classifier
        self.vc_beta = vc_beta


    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    
    def call(
        self,
        inputs: tuple[tf.Tensor, tf.Tensor],
        training: bool = False,
    ) -> tf.Tensor:
        """
        Parameters
        ----------
        inputs : (xd, bi)
            xd : Tensor, shape (batch, n_windows, n_bands, C, C)
                 4-D ManifoldNet input
            bi : Tensor, shape (batch, n_windows, C*(C+1)//2)
                 BiLSTM flattened-covariance input

        Returns
        -------
        logits : Tensor, shape (batch, n_classes)
        """
        xd, bi = inputs
        mo = self.manifold_net(xd, training=training)
        ho = self.bilstm_net(bi, training=training)
        return self.fusion(mo, ho, training=training)

    # ------------------------------------------------------------------
    # Joint alternating optimisation  (Algorithm 1)
    # ------------------------------------------------------------------

    @tf.function
    def _train_step_bilstm(
        self,
        xd: tf.Tensor,
        bi: tf.Tensor,
        y: tf.Tensor,
        optimizer_b: tf.keras.optimizers.Optimizer,
        optimizer_f: tf.keras.optimizers.Optimizer,
        optimizer_d: tf.keras.optimizers.Optimizer,
    ) -> tf.Tensor:
        """Odd iteration: update BiLSTM and FC; hold ManifoldNet fixed."""

        # MO is computed without gradient tracking
        mo = self.manifold_net(xd, training=False)
        mo = tf.stop_gradient(mo)

        # Update BiLISTM + fusion priors/class-weights using VC loss
        with tf.GradientTape() as tape:
            ho = self.bilstm_net(bi, training=True)
            mh = tf.concat([mo, ho], axis=-1)
            loss = self.fusion.vc_loss(mh, y, beta=self.vc_beta)
        
        bilstm_vars = self.bilstm_net.trainable_variables
        # Fusion vars = prior_mu, prior_log_sigma, log_class_prior (NOT disc_w/disc_b)
        fusion_prior_vars = [self.fusion.prior_mu, self.fusion.prior_log_sigma,
                         self.fusion.log_class_prior]

        grads = tape.gradient(loss, bilstm_vars + fusion_prior_vars)
        optimizer_b.apply_gradients(zip(grads[:len(bilstm_vars)], bilstm_vars))
        optimizer_f.apply_gradients(zip(grads[len(bilstm_vars):], fusion_prior_vars))

        # Update discriminator with auxiliary loss
        with tf.GradientTape() as tape:
            ho = tf.stop_gradient(self.bilstm_net(bi, training=False))
            mh = tf.concat([mo, ho], axis=-1)
            disc_loss = self.fusion.discriminator_loss(mh, y)
        
        disc_vars = [self.fusion.disc_w, self.fusion.disc_b]
        grads_d = tape.gradient(disc_loss, disc_vars)
        optimizer_d.apply_gradients(zip(grads_d, disc_vars))

        return loss

    @tf.function
    def _train_step_manifold(
        self,
        xd: tf.Tensor,
        bi: tf.Tensor,
        y: tf.Tensor,
        optimizer_m: tf.keras.optimizers.Optimizer,
        optimizer_f: tf.keras.optimizers.Optimizer,
        optimizer_d: tf.keras.optimizers.Optimizer,
    ) -> tf.Tensor:
        """Even iteration: update ManifoldNet and FC; hold BiLSTM fixed."""

        # HO is computed without gradient tracking
        ho = self.bilstm_net(bi, training=False)
        ho = tf.stop_gradient(ho)

        with tf.GradientTape() as tape:
            mo = self.manifold_net(xd, training=True)
            mh = tf.concat([mo, ho], axis=-1)
            loss = self.fusion.vc_loss(mh, y, beta=self.vc_beta)

        manifold_vars = self.manifold_net.trainable_variables
        # Fusion vars = prior_mu, prior_log_sigma, log_class_prior (NOT disc_w/disc_b)
        fusion_prior_vars = [self.fusion.prior_mu, self.fusion.prior_log_sigma,
                         self.fusion.log_class_prior]

        grads = tape.gradient(loss, manifold_vars + fusion_prior_vars)
        optimizer_m.apply_gradients(zip(grads[:len(manifold_vars)], manifold_vars))
        optimizer_f.apply_gradients(zip(grads[len(manifold_vars):], fusion_prior_vars))

        # Update discriminator with auxiliary loss
        with tf.GradientTape() as tape:
            mo = tf.stop_gradient(self.manifold_net(xd, training=False))
            mh = tf.concat([mo, ho], axis=-1)
            disc_loss = self.fusion.discriminator_loss(mh, y)
        
        disc_vars = [self.fusion.disc_w, self.fusion.disc_b]
        grads_d = tape.gradient(disc_loss, disc_vars)
        optimizer_d.apply_gradients(zip(grads_d, disc_vars))

        return loss

    def fit_joint(
        self,
        xd_train: tf.Tensor,
        bi_train: tf.Tensor,
        y_train: tf.Tensor,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-4,
        weight_decay: float = 5e-4,
        validation_data: tuple | None = None,
    ) -> dict:
        """Train STSNet using the joint alternating optimisation (Algorithm 1).

        Parameters
        ----------
        xd_train, bi_train, y_train : training tensors
        epochs          : int
        batch_size      : int
        lr              : float — learning rate (η in Algorithm 1)
        weight_decay    : float — L2 regularisation (λ in Table 1)
        validation_data : optional (xd_val, bi_val, y_val) tuple

        Returns
        -------
        history : dict with keys 'loss', 'val_loss', 'val_acc'
        """
        optimizer_m = tf.keras.optimizers.Adam(lr, weight_decay=weight_decay)
        optimizer_b = tf.keras.optimizers.Adam(lr, weight_decay=weight_decay)
        optimizer_f = tf.keras.optimizers.Adam(lr, weight_decay=weight_decay)
        optimizer_d = tf.keras.optimizers.Adam(lr, weight_decay=weight_decay)

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        n_samples = xd_train.shape[0]
        dataset   = (
            tf.data.Dataset.from_tensor_slices((xd_train, bi_train, y_train))
            .shuffle(n_samples, reshuffle_each_iteration=True)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        history = {"loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(epochs):
            epoch_losses = []

            for step, (xd_b, bi_b, y_b) in enumerate(dataset):
                # Alternate: even steps → manifold; odd steps → bilstm
                if step % 2 == 0:
                    loss = self._train_step_manifold(
                        xd_b, bi_b, y_b,
                        optimizer_m, optimizer_f, optimizer_d,
                    )
                else:
                    loss = self._train_step_bilstm(
                        xd_b, bi_b, y_b,
                        optimizer_b, optimizer_f, optimizer_d,
                    )
                epoch_losses.append(float(loss))

            mean_loss = sum(epoch_losses) / len(epoch_losses)
            history["loss"].append(mean_loss)

            if validation_data is not None:
                xd_v, bi_v, y_v = validation_data
                val_logits = self((xd_v, bi_v), training=False) # Comes from the model as log-posterior, fine to use for accuracy

                # Need to recompute loss using latent, not log-posterior logits
                mo_v = self.manifold_net(xd_v, training=False)
                ho_v = self.bilstm_net(bi_v, training=False)
                mh_v = tf.concat([mo_v, ho_v], axis=-1)
                val_loss   = float(self.fusion.vc_loss(mh_v, y_v, self.vc_beta))

                val_preds  = tf.argmax(val_logits, axis=-1)
                val_acc    = float(
                    tf.reduce_mean(tf.cast(val_preds == tf.cast(y_v, tf.int64), tf.float32))
                )
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                print(
                    f"Epoch {epoch+1:03d}/{epochs}  "
                    f"loss={mean_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
                )
            else:
                print(f"Epoch {epoch+1:03d}/{epochs}  loss={mean_loss:.4f}")

        return history

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "n_channels"     : self.manifold_net.n_channels,
            "n_classes"      : self.fusion.n_classes,
            "bilstm_units"   : self.bilstm_net.hidden_units,
            "bilstm_dropout" : self.bilstm_net.dropout_rate,
            "manifold_kernel": self.manifold_net.kernel_size,
            "n_fm_iters"     : self.manifold_net.n_fm_iters,
        })
        return cfg
