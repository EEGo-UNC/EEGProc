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
from .manifold_net import ManifoldNet


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
# Fusion head
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
        self.fusion = FusionHead(n_classes=n_classes, name="fusion")

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
        loss_fn: tf.keras.losses.Loss,
    ) -> tf.Tensor:
        """Odd iteration: update BiLSTM and FC; hold ManifoldNet fixed."""

        # MO is computed without gradient tracking
        mo = self.manifold_net(xd, training=False)
        mo = tf.stop_gradient(mo)

        with tf.GradientTape() as tape:
            ho     = self.bilstm_net(bi, training=True)
            logits = self.fusion(mo, ho, training=True)
            loss   = loss_fn(y, logits)
        
        bilstm_vars = self.bilstm_net.trainable_variables
        fc_vars     = self.fusion.trainable_variables

        grads = tape.gradient(loss, bilstm_vars + fc_vars)
        optimizer_b.apply_gradients(zip(grads[:len(bilstm_vars)], bilstm_vars))
        optimizer_f.apply_gradients(zip(grads[len(bilstm_vars):], fc_vars))
        return loss

    @tf.function
    def _train_step_manifold(
        self,
        xd: tf.Tensor,
        bi: tf.Tensor,
        y: tf.Tensor,
        optimizer_m: tf.keras.optimizers.Optimizer,
        optimizer_f: tf.keras.optimizers.Optimizer,
        loss_fn: tf.keras.losses.Loss,
    ) -> tf.Tensor:
        """Even iteration: update ManifoldNet and FC; hold BiLSTM fixed."""

        # HO is computed without gradient tracking
        ho = self.bilstm_net(bi, training=False)
        ho = tf.stop_gradient(ho)

        with tf.GradientTape() as tape:
            mo     = self.manifold_net(xd, training=True)
            logits = self.fusion(mo, ho, training=True)
            loss   = loss_fn(y, logits)

        manifold_vars = self.manifold_net.trainable_variables
        fc_vars       = self.fusion.trainable_variables

        grads = tape.gradient(loss, manifold_vars + fc_vars)
        optimizer_m.apply_gradients(zip(grads[:len(manifold_vars)], manifold_vars))
        optimizer_f.apply_gradients(zip(grads[len(manifold_vars):], fc_vars))
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
                        optimizer_m, optimizer_f, loss_fn,
                    )
                else:
                    loss = self._train_step_bilstm(
                        xd_b, bi_b, y_b,
                        optimizer_b, optimizer_f, loss_fn,
                    )
                epoch_losses.append(float(loss))

            mean_loss = sum(epoch_losses) / len(epoch_losses)
            history["loss"].append(mean_loss)

            if validation_data is not None:
                xd_v, bi_v, y_v = validation_data
                val_logits = self((xd_v, bi_v), training=False)
                val_loss   = float(loss_fn(y_v, val_logits))
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
