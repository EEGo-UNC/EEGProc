"""
stsnet.py
=========
Full STSNet model: BiLSTM sub-model + ManifoldNet sub-model + fusion head.

Architecture summary (Figure 2 of the paper)
--------------------------------------------
(a) ManifoldNet branch (MO):
      4-D SPD tensor  ->  2x wFM conv  ->  Invariant layer  ->  MO vector

(b) BiLSTM branch (HO):
      Flattened covariance sequence  ->  BiLSTM  ->  HO vector  (Eq. 10)

(c) Fusion & classification:
      MH = concat(MO, HO)  ->  FC  ->  Softmax  ->  class label

Training uses the joint alternating optimisation from Algorithm 1:
every even iteration fixes HO and trains MO; every odd iteration
fixes MO and trains HO. The FC layer is updated on every step.

Ablation modes (controlled by the `training_mode` argument to STSNet and fit_joint)
------------------------------------------------------------------------------------
"vc_only"   : vc_loss only (xent + KL terms); discriminator is never updated.
              disc_w / disc_b are frozen and the discriminator update block is skipped.
"disc_only" : discriminator auxiliary loss only; vc_loss KL terms are zeroed (beta=0,
              lambda=0) so the encoder sees only cross-entropy + discriminator signal.
"both"      : full VC objective — vc_loss (all three terms) + discriminator update.
              This is the default and matches Algorithm 1 of the paper.

References
----------
Li et al., "STSNet ...", HISS 2023.
"""

import tensorflow as tf
import numpy as np
try:
    from .manifold_net import ManifoldNet
    from ..variational_classifier import VariationalClassifier
except ImportError:
    from manifold_net import ManifoldNet
    from variational_classifier import VariationalClassifier

# Valid values for the training_mode argument throughout this module.
TRAINING_MODES = ("vc_only", "disc_only", "both")


# ---------------------------------------------------------------------------
# BiLSTM sub-model
# ---------------------------------------------------------------------------

class BiLSTMNet(tf.keras.Model):
    """BiLSTM sub-model for spatio-temporal feature extraction (HO).

    Processes the flattened covariance time-series produced by
    `build_spatiotemporal_representation`.

    Architecture (per Table 1 / Table 3 in the paper)
    --------------------------------------------------
    Input (n_windows, feat_dim) -> BiLSTM(256 units) -> HO (512-d vector)
    HO = concat(forward hidden state at T, backward hidden state at 1)
    following Eq. (10): HO = H_nc (->) oplus H_1 (<-)

    Parameters
    ----------
    hidden_units : int   -- LSTM cell size (default 256)
    dropout_rate : float -- recurrent dropout for regularisation
    """

    def __init__(self, hidden_units: int = 256, dropout_rate: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate

        forward_lstm = tf.keras.layers.LSTM(
            hidden_units, return_sequences=True, return_state=True,
            dropout=dropout_rate, name="forward_lstm",
        )
        backward_lstm = tf.keras.layers.LSTM(
            hidden_units, return_sequences=True, return_state=True,
            go_backwards=True, dropout=dropout_rate, name="backward_lstm",
        )
        self.bilstm = tf.keras.layers.Bidirectional(
            forward_lstm, backward_layer=backward_lstm,
            merge_mode=None, name="bilstm",
        )

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (batch, n_windows, feat_dim)

        Returns
        -------
        ho : Tensor, shape (batch, 2 * hidden_units)
        """
        outputs = self.bilstm(x, training=training)
        # Bidirectional with merge_mode=None returns:
        #   [fwd_seq, bwd_seq, fwd_h, fwd_c, bwd_h, bwd_c]
        _, _, fwd_h, _, bwd_h, _ = outputs
        return tf.concat([fwd_h, bwd_h], axis=-1)  # (batch, 2*hidden_units)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"hidden_units": self.hidden_units, "dropout_rate": self.dropout_rate})
        return cfg


# ---------------------------------------------------------------------------
# Fusion head (standard, non-variational classifier)
# ---------------------------------------------------------------------------

class FusionHead(tf.keras.layers.Layer):
    """Standard FC fusion head.  MH = [MO || HO]  ->  softmax logits."""

    def __init__(self, n_classes: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.n_classes = n_classes
        self.fc = tf.keras.layers.Dense(n_classes, name="fc")

    def call(self, mh: tf.Tensor, training: bool = False) -> tf.Tensor:
        return self.fc(mh)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"n_classes": self.n_classes})
        return cfg


# ---------------------------------------------------------------------------
# Full STSNet
# ---------------------------------------------------------------------------

class STSNet(tf.keras.Model):
    """STSNet: Spatio-Temporal-Spectral Network for EEG emotion recognition.

    Parameters
    ----------
    n_channels      : int   -- EEG channel count
    n_classes       : int   -- emotion classes (default 2: binary)
    bilstm_units    : int   -- BiLSTM hidden units (default 256)
    bilstm_dropout  : float -- BiLSTM recurrent dropout
    manifold_kernel : int   -- wFM kernel size for ManifoldNet (default 2)
    n_fm_iters      : int   -- Frechet mean iterations
    vc_beta         : float -- beta for vc_loss KL term
    vc_lambda       : float -- lambda for vc_loss class-prior KL term
    training_mode   : str   -- ablation switch; one of TRAINING_MODES:
                               "vc_only"   vc_loss only, no discriminator updates
                               "disc_only" discriminator + cross-entropy only
                                           (beta=0, lambda_=0 in vc_loss)
                               "both"      full VC + discriminator (default)
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
        vc_lambda: float = 1.0,
        training_mode: str = "both",
        **kwargs,
    ):
        if training_mode not in TRAINING_MODES:
            raise ValueError(
                f"training_mode must be one of {TRAINING_MODES}, got '{training_mode}'"
            )
        super().__init__(**kwargs)

        self.manifold_net = ManifoldNet(
            n_channels=n_channels, kernel_size=manifold_kernel,
            n_fm_iters=n_fm_iters, name="manifold_net",
        )
        self.bilstm_net = BiLSTMNet(
            hidden_units=bilstm_units, dropout_rate=bilstm_dropout,
            name="bilstm_net",
        )
        self.fusion = VariationalClassifier(n_classes=n_classes, name="fusion")

        self.vc_beta       = vc_beta
        self.vc_lambda     = vc_lambda
        self.training_mode = training_mode

    # ------------------------------------------------------------------
    # Helpers: effective beta/lambda given training_mode
    # ------------------------------------------------------------------

    @property
    def _effective_beta(self) -> float:
        """KL weight sent to vc_loss -- zeroed when mode is disc_only."""
        return 0.0 if self.training_mode == "disc_only" else self.vc_beta

    @property
    def _effective_lambda(self) -> float:
        """Class-prior KL weight sent to vc_loss -- zeroed when mode is disc_only."""
        return 0.0 if self.training_mode == "disc_only" else self.vc_lambda

    @property
    def _use_discriminator(self) -> bool:
        """Whether discriminator updates should run."""
        return self.training_mode in ("disc_only", "both")

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def call(self, inputs: tuple, training: bool = False) -> tf.Tensor:
        """
        Parameters
        ----------
        inputs : (xd, bi)
            xd : (batch, n_windows, n_bands, C, C)
            bi : (batch, n_windows, C*(C+1)//2)

        Returns
        -------
        logits : (batch, n_classes)
        """
        xd, bi = inputs
        mo = self.manifold_net(xd, training=training)
        ho = self.bilstm_net(bi, training=training)
        mh = tf.concat([mo, ho], axis=-1)
        return self.fusion(mh, training=training)

    # ------------------------------------------------------------------
    # Shared discriminator update block (DRY helper used by both train steps)
    # ------------------------------------------------------------------

    def _update_discriminator(
        self,
        mh: tf.Tensor,
        y: tf.Tensor,
        optimizer_d: tf.keras.optimizers.Optimizer,
    ) -> None:
        """Run one discriminator gradient step if training_mode requires it."""
        if not self._use_discriminator:
            return
        with tf.GradientTape() as tape:
            disc_loss = self.fusion.discriminator_loss(tf.stop_gradient(mh), y)
        disc_vars = [self.fusion.disc_w, self.fusion.disc_b]
        grads = tape.gradient(disc_loss, disc_vars)
        optimizer_d.apply_gradients(zip(grads, disc_vars))

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
        """Odd iteration: update BiLSTM + fusion priors; hold ManifoldNet fixed."""
        mo = tf.stop_gradient(self.manifold_net(xd, training=False))

        with tf.GradientTape() as tape:
            ho  = self.bilstm_net(bi, training=True)
            mh  = tf.concat([mo, ho], axis=-1)
            loss = self.fusion.vc_loss(
                mh, y, beta=self._effective_beta, lambda_=self._effective_lambda,
            )

        bilstm_vars      = self.bilstm_net.trainable_variables
        fusion_prior_vars = [
            self.fusion.prior_mu, self.fusion.prior_log_sigma,
            self.fusion.log_class_prior,
        ]
        grads = tape.gradient(loss, bilstm_vars + fusion_prior_vars)
        optimizer_b.apply_gradients(zip(grads[:len(bilstm_vars)], bilstm_vars))
        optimizer_f.apply_gradients(zip(grads[len(bilstm_vars):], fusion_prior_vars))

        # Discriminator update (skipped when training_mode == "vc_only")
        ho_sg = tf.stop_gradient(self.bilstm_net(bi, training=False))
        mh_sg = tf.concat([mo, ho_sg], axis=-1)
        self._update_discriminator(mh_sg, y, optimizer_d)

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
        """Even iteration: update ManifoldNet + fusion priors; hold BiLSTM fixed."""
        ho = tf.stop_gradient(self.bilstm_net(bi, training=False))

        with tf.GradientTape() as tape:
            mo  = self.manifold_net(xd, training=True)
            mh  = tf.concat([mo, ho], axis=-1)
            loss = self.fusion.vc_loss(
                mh, y, beta=self._effective_beta, lambda_=self._effective_lambda,
            )

        manifold_vars     = self.manifold_net.trainable_variables
        fusion_prior_vars = [
            self.fusion.prior_mu, self.fusion.prior_log_sigma,
            self.fusion.log_class_prior,
        ]
        grads = tape.gradient(loss, manifold_vars + fusion_prior_vars)
        optimizer_m.apply_gradients(zip(grads[:len(manifold_vars)], manifold_vars))
        optimizer_f.apply_gradients(zip(grads[len(manifold_vars):], fusion_prior_vars))

        # Discriminator update (skipped when training_mode == "vc_only")
        mo_sg = tf.stop_gradient(self.manifold_net(xd, training=False))
        mh_sg = tf.concat([mo_sg, ho], axis=-1)
        self._update_discriminator(mh_sg, y, optimizer_d)

        return loss

    # ------------------------------------------------------------------
    # fit_joint
    # ------------------------------------------------------------------

    def fit_joint(
        self,
        xd_train: tf.Tensor,
        bi_train: tf.Tensor,
        y_train: tf.Tensor,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-4,
        weight_decay: float = 5e-4,
        validation_data=None,
    ) -> dict:
        """Train STSNet using the joint alternating optimisation (Algorithm 1).

        The behaviour of each step is controlled by self.training_mode:
          "vc_only"   -- KL terms active, discriminator updates skipped
          "disc_only" -- KL terms zeroed, discriminator updates active
          "both"      -- all terms and updates active (default)

        Parameters
        ----------
        xd_train, bi_train, y_train : training tensors
        epochs          : int
        batch_size      : int
        lr              : float
        weight_decay    : float
        validation_data : optional (xd_val, bi_val, y_val) tuple

        Returns
        -------
        history : dict with keys 'loss', 'val_loss', 'val_acc'
        """
        print(f"Training mode: {self.training_mode}")

        optimizer_m = tf.keras.optimizers.Adam(lr, weight_decay=weight_decay)
        optimizer_b = tf.keras.optimizers.Adam(lr, weight_decay=weight_decay)
        optimizer_f = tf.keras.optimizers.Adam(lr, weight_decay=weight_decay)
        optimizer_d = tf.keras.optimizers.Adam(lr, weight_decay=weight_decay)

        n_samples = xd_train.shape[0]
        dataset = (
            tf.data.Dataset.from_tensor_slices((xd_train, bi_train, y_train))
            .shuffle(n_samples, reshuffle_each_iteration=True)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        history = {"loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(epochs):
            epoch_losses = []

            for step, (xd_b, bi_b, y_b) in enumerate(dataset):
                if step % 2 == 0:
                    loss = self._train_step_manifold(
                        xd_b, bi_b, y_b, optimizer_m, optimizer_f, optimizer_d,
                    )
                else:
                    loss = self._train_step_bilstm(
                        xd_b, bi_b, y_b, optimizer_b, optimizer_f, optimizer_d,
                    )
                epoch_losses.append(float(loss))

            mean_loss = sum(epoch_losses) / len(epoch_losses)
            history["loss"].append(mean_loss)

            if validation_data is not None:
                xd_v, bi_v, y_v = validation_data
                val_logits = self((xd_v, bi_v), training=False)

                mo_v   = self.manifold_net(xd_v, training=False)
                ho_v   = self.bilstm_net(bi_v, training=False)
                mh_v   = tf.concat([mo_v, ho_v], axis=-1)
                val_loss = float(self.fusion.vc_loss(
                    mh_v, y_v,
                    beta=self._effective_beta, lambda_=self._effective_lambda,
                ))
                val_preds = tf.argmax(val_logits, axis=-1)
                val_acc   = float(tf.reduce_mean(
                    tf.cast(val_preds == tf.cast(y_v, tf.int64), tf.float32)
                ))
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
            "vc_beta"        : self.vc_beta,
            "vc_lambda"      : self.vc_lambda,
            "training_mode"  : self.training_mode,
        })
        return cfg
