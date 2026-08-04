"""End-to-end trainable MTLFuseNet model.

The notebook wired the VAE branch, the GCN-GRU branch, the fusion, and the three
losses, but only ever ran a *forward pass* on a single window with an ad-hoc,
per-window Python loop and non-learnable centers. This module assembles all the
pieces into one ``tf.keras.Model`` that:

  * runs both branches on a *batch* of windows (the GCN is vectorized over the
    batch and the 3 bands with ``einsum`` instead of the per-window loop),
  * treats the triplet-center class centers as trainable weights, and
  * optimizes the combined loss (focal + triplet-center + VAE) end to end via a
    custom ``train_step``.

Forward inputs are a tuple ``(X_ST, DE, adj)``:
    X_ST : (B, 9, 9, 128)   normalized spatio-temporal grid windows
    DE   : (B, 3, 14)       differential-entropy features, band x channel
    adj  : (B, 3, 14, 14)   symmetric-normalized adjacency per band (per trial)
"""

import tensorflow as tf
from tensorflow.keras import layers

from eegproc.deep_learning.supervised.mtlfusenet.models import build_vae_encoder, build_vae_decoder, Sampling
from eegproc.deep_learning.supervised.mtlfusenet.losses import focal_loss, triplet_center_loss


class MTLFuseNet(tf.keras.Model):
    def __init__(self, num_classes=2, vae_latent=128, gcn_dim=32, gru_units=384,
                 beta1=0.7, beta2=0.2, beta3=0.1,
                 focal_alpha=0.7, focal_gamma=2.0, tc_margin=1.0, dropout=0.2, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.vae_latent = vae_latent
        self.gcn_dim = gcn_dim
        self.fused_dim = vae_latent + gru_units
        self.beta1, self.beta2, self.beta3 = beta1, beta2, beta3
        self.focal_alpha, self.focal_gamma = focal_alpha, focal_gamma
        self.tc_margin = tc_margin

        # VAE (spatio-temporal) branch
        self.encoder = build_vae_encoder(input_shape=(9, 9, 128), latent_dim=vae_latent)
        self.decoder = build_vae_decoder(latent_dim=vae_latent, output_shape=(9, 9, 128))
        self.sampling = Sampling()

        # GCN (spatio-spectral) branch: node feature dim is 1 (one DE value / channel)
        self.gcn_W = self.add_weight(shape=(1, gcn_dim), initializer="glorot_uniform",
                                     trainable=True, name="gcn_W")
        self.gcn_b = self.add_weight(shape=(gcn_dim,), initializer="zeros",
                                     trainable=True, name="gcn_b")
        self.gru = layers.GRU(gru_units, name="ss_gru")

        # fusion head
        self.dropout = layers.Dropout(dropout)  # Table 2: dropout 0.2
        self.classifier = layers.Dense(num_classes, activation="softmax", name="classifier")
        self.centers = self.add_weight(shape=(num_classes, self.fused_dim),
                                       initializer="random_normal", trainable=True,
                                       name="tc_centers")

        # metric trackers
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.focal_tracker = tf.keras.metrics.Mean(name="focal")
        self.tc_tracker = tf.keras.metrics.Mean(name="tc")
        self.vae_tracker = tf.keras.metrics.Mean(name="vae")
        self.acc_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name="acc")

    # ------------------------------------------------------------------ forward
    def call(self, inputs, training=False):
        X_ST, DE, adj = inputs

        # --- VAE branch ---
        z_mean, z_log_var = self.encoder(X_ST, training=training)
        z = self.sampling([z_mean, z_log_var])
        recon = self.decoder(z, training=training)

        # --- GCN-GRU branch (batched over B and the 3 bands) ---
        node = DE[..., tf.newaxis]                       # (B, 3, 14, 1)
        # aggregate neighbours: adj @ node   -> (B, 3, 14, 1)
        agg = tf.einsum("bkij,bkjf->bkif", adj, node)
        h = tf.tensordot(agg, self.gcn_W, axes=[[3], [0]]) + self.gcn_b  # (B,3,14,gcn_dim)
        h = tf.nn.relu(h)
        B = tf.shape(h)[0]
        seq = tf.reshape(h, (B, 3, 14 * self.gcn_dim))   # (B, 3, 448) — 3 band "timesteps"
        Z_SS = self.gru(seq, training=training)          # (B, gru_units)

        # --- fusion + classifier ---
        Z_SST = tf.concat([z, Z_SS], axis=-1)            # (B, fused_dim)
        y_pred = self.classifier(self.dropout(Z_SST, training=training))  # (B, num_classes)
        return {"y_pred": y_pred, "Z_SST": Z_SST, "recon": recon,
                "z_mean": z_mean, "z_log_var": z_log_var}

    # --------------------------------------------------------------- loss pieces
    def compute_losses(self, inputs, labels, out):
        X_ST = inputs[0]
        focal = tf.reduce_mean(
            focal_loss(labels, out["y_pred"], alpha=self.focal_alpha, gamma=self.focal_gamma)
        )
        tc = triplet_center_loss(out["Z_SST"], labels, self.centers, margin=self.tc_margin)
        recon_mse = tf.reduce_mean(tf.square(X_ST - out["recon"]))
        kl = -0.5 * tf.reduce_mean(
            1 + out["z_log_var"] - tf.square(out["z_mean"]) - tf.exp(out["z_log_var"])
        )
        vae = recon_mse + kl
        total = self.beta1 * focal + self.beta2 * tc + self.beta3 * vae
        return total, focal, tc, vae

    def _update_trackers(self, total, focal, tc, vae, labels, y_pred):
        self.loss_tracker.update_state(total)
        self.focal_tracker.update_state(focal)
        self.tc_tracker.update_state(tc)
        self.vae_tracker.update_state(vae)
        self.acc_tracker.update_state(labels, y_pred)
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [self.loss_tracker, self.focal_tracker, self.tc_tracker,
                self.vae_tracker, self.acc_tracker]

    # --------------------------------------------------------------- train / test
    def train_step(self, data):
        inputs, labels = data
        with tf.GradientTape() as tape:
            out = self(inputs, training=True)
            total, focal, tc, vae = self.compute_losses(inputs, labels, out)
        grads = tape.gradient(total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return self._update_trackers(total, focal, tc, vae, labels, out["y_pred"])

    def test_step(self, data):
        inputs, labels = data
        out = self(inputs, training=False)
        total, focal, tc, vae = self.compute_losses(inputs, labels, out)
        return self._update_trackers(total, focal, tc, vae, labels, out["y_pred"])
